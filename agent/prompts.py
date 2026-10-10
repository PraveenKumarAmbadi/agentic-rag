"""Prompt templates for grounded answer generation and the agent loop.

Inputs are SciFact-style: usually a declarative claim ("X reduces Y in Z"),
sometimes a question. Every prompt names both forms, because a claim framed
as a "question" leaves the model unsure what an answer looks like, and that
uncertainty comes out as refusals.

The one judgement every turn shares is whether a document ADDRESSES the input.
It is defined the same way everywhere:
- addresses: the document studies what the input is about, in the same
  population, species, disease, intervention and outcome. Reporting the
  opposite result still addresses it (the answer is "contradicted");
  covering one part of it addresses that part.
- does not address: same topic, but a different population, species,
  disease, intervention, outcome or time frame, or an association where the
  input asserts cause and effect. This is what near-miss evidence looks like,
  and it must lead to a refusal, not to an answer about something else.

Turns:
- ANSWER (writer, baseline and agent): answer with [n] citations, or refuse
  with the INSUFFICIENT_EVIDENCE marker as the whole reply, never both.
- DECIDE (controller): one JSON tool call. It sees only document briefs, so
  it routes (answer / search again / give up); the writer, who reads the
  full text, makes the fine-grained call.
- RESCUE (second opinion): re-checks a refusal against the full documents.
  It quotes its evidence before naming a document, so an override has to
  rest on a sentence, not on topical similarity.
- CRITIQUE (fact-checker): lists unsupported claims, including claims that
  stretch a finding beyond the scope the document gives it.

The context block is built by `build_context()`; document numbering in every
prompt is 1-based and maps to `doc_ids[i-1]`. The system prompts double as
routing keys for the test FakeLLM ("controller", "second opinion",
"fact-checker"), so each phrase must appear in its own prompt only.
"""

from agent.citations import REFUSAL_MARKER
from agent.tools import render_tool_docs

# Shared wording of the "addresses" test (see module docstring).
_SCOPE = "population, species, disease, intervention, outcome or time frame"

SYSTEM_PROMPT = f"""\
You are a careful scientific research assistant. The user gives you a claim \
to check or a question to answer. Respond using ONLY the numbered documents \
provided, never your own knowledge.

Before writing, decide silently whether at least one document addresses the \
input, meaning it studies what the input is about, in the same {_SCOPE}:
- A document that reports the opposite result still addresses a claim (the \
claim is contradicted).
- A document that covers only part of the input addresses that part.
- A document on the same topic but about a different {_SCOPE}, or that \
reports an association where the input asserts cause and effect, does not \
address it.

A document that reports the opposite of a claim is never a reason to refuse: \
the answer is that the claim is contradicted, citing that document.

If no document addresses the input, your entire reply is this one line:
{REFUSAL_MARKER}: <what specifically is missing>
Write nothing before or after that line. Never explain in prose that the \
documents do not address the input; use the line.

Otherwise, answer:
- For a claim, start by saying whether the documents support it, contradict \
it, or support only part of it, then give the key evidence. For a question, \
start with the answer.
- Put a citation after every factual statement, e.g. "the effect was \
significant [2]". Cite by document number, and only documents that state \
what you cite them for.
- Keep each finding's scope as the document gives it: if it comes from \
mice, a specific patient group or an observational association, say so.
- If only part of the input is covered, say which part is not.
- Write plain prose, at most 150 words, with no headings or numbered lists.
"""

CONTEXT_HEADER = "Documents:"

# Default per-doc truncation: ~800 chars vs a median abstract of ~1330.
# The generation ablation (scripts/run_generation_ablation.py) found this
# truncation is a real driver of false refusals: on 28 refused cases, 2400
# chars cut refusals from 28 to 17 with the same prompt. Raising it is bounded
# by llm.num_ctx (10 docs x 2400 chars does not fit in 4096 tokens). Config
# `agent.context_chars` overrides this for both pipelines.
DEFAULT_CONTEXT_CHARS = 800

# How much of each document the decide turn sees (full text goes to the
# generation, rescue and critique turns, which judge the evidence).
BRIEF_CHARS = 240


def build_context(docs: list, max_chars: int = DEFAULT_CONTEXT_CHARS) -> str:
    """Format retrieved documents as the numbered context block.

    Args:
        docs: Retrieved results (any objects with doc_id/title/text), best first.
        max_chars: Per-doc text truncation — keep large enough that the
            evidence survives into the prompt (see DEFAULT_CONTEXT_CHARS).

    Returns:
        Context string; each doc truncated to keep the prompt bounded.
    """
    lines = [CONTEXT_HEADER]
    for i, doc in enumerate(docs, start=1):
        title = (doc.title or "").strip()
        text = " ".join((doc.text or "").split())[:max_chars]
        lines.append(f"[{i}] {title}\n{text}\n")
    return "\n".join(lines)


def build_doc_briefs(docs: list, chars: int = BRIEF_CHARS) -> str:
    """Compact numbered doc list (title + snippet) for control turns."""
    lines = []
    for i, doc in enumerate(docs, start=1):
        title = (doc.title or "").strip() or doc.doc_id
        text = " ".join((doc.text or "").split())[:chars]
        lines.append(f"[{i}] {title} — {text}")
    return "\n".join(lines) if lines else "(no documents retrieved)"


def _input_line(question: str) -> str:
    return f"Claim or question: {question}"


def build_messages(
    question: str, docs: list, context_chars: int = DEFAULT_CONTEXT_CHARS
) -> list[dict[str, str]]:
    """Assemble the chat messages for a grounded answer (baseline pipeline)."""
    return build_answer_messages(question, docs, context_chars=context_chars)


def build_answer_messages(
    question: str,
    docs: list,
    hint_doc: int | None = None,
    hint_evidence: str | None = None,
    feedback: list[str] | None = None,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> list[dict[str, str]]:
    """Grounded-answer messages with optional rescue hint / critique feedback.

    Args:
        hint_doc: 1-based doc number the rescue step identified as
            addressing the input.
        hint_evidence: the sentence the rescue step quoted from it, already
            verified to be in the document; the writer answers from it.
        feedback: unsupported claims from a failed critique; the model is
            asked to rewrite the answer without them.
        context_chars: Per-doc truncation for the context block.
    """
    parts = [build_context(docs, context_chars), _input_line(question)]
    if hint_doc is not None:
        doc = docs[hint_doc - 1]
        title = (doc.title or "").strip() or doc.doc_id
        # The rescue turn already applied the strict "addresses" test on the
        # full text and its quote was verified in code. Offering the writer
        # a refusal again here undid 3 of 4 correct rescues on the dev set.
        quoted = f' in this sentence: "{hint_evidence}"' if hint_evidence else ""
        parts.append(
            f"A reviewer checked the documents and found that document "
            f"[{hint_doc}] ({title}) addresses the input{quoted}. Answer from "
            f"document [{hint_doc}]: say whether it supports or contradicts "
            f"the claim, or what it says about the question, and cite [{hint_doc}]."
        )
    if feedback:
        listed = "\n".join(f"- {c}" for c in feedback)
        parts.append(
            "A previous draft made these claims, which the documents do not "
            f"support:\n{listed}\n"
            "Write the reply again without them, citing a document for every "
            "statement. If nothing supported is left to say, refuse."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# ---------------------------------------------------------------------------
# Agent-loop prompts
# ---------------------------------------------------------------------------

DECIDE_SYSTEM_PROMPT = f"""\
You are the controller of a scientific research assistant. You choose the \
next step for handling the user's input (a claim to check or a question); \
you do not write the reply. You see only the start of each retrieved \
document.

Tools:
{render_tool_docs()}

How to choose:
- answer: at least one document is about the input's specific subject. List \
those documents. You do not need to be certain; the writer reads them in \
full and refuses if they fall short.
- hybrid_search: no document is about the input's specific subject, and \
different wording might find one. Never repeat an earlier query.
- insufficient_evidence: only after at least one hybrid_search, when the \
documents are still not about the input.

Reply with one JSON object and no other text, for example:
{{"tool": "answer", "args": {{"documents": [1, 3]}}}}
{{"tool": "hybrid_search", "args": {{"query": "statin therapy dementia risk in older adults"}}}}
{{"tool": "insufficient_evidence", "args": {{"reason": "no document studies children"}}}}
"""


def build_decide_messages(
    question: str,
    docs: list,
    history: list[str] | None = None,
    iteration: int = 1,
    max_iterations: int = 4,
) -> list[dict[str, str]]:
    """Messages for one decide turn of the ReAct loop."""
    parts = [
        _input_line(question),
        f"\nRetrieved documents (best first):\n{build_doc_briefs(docs)}",
    ]
    if history:
        parts.append("\nActions so far: " + "; ".join(history))
    if iteration >= max_iterations:
        parts.append(
            f"\nStep {iteration} of {max_iterations}, the last one: a new "
            "search would not be used. Choose answer or insufficient_evidence."
        )
    else:
        parts.append(f"\nStep {iteration} of {max_iterations}. Choose the next step.")
    return [
        {"role": "system", "content": DECIDE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


RESCUE_SYSTEM_PROMPT = f"""\
You are a second opinion on a refusal. A writer read the documents and \
refused to respond to the user's input (a claim to check or a question), \
saying the evidence is insufficient. Check whether that refusal is wrong.

Writers often refuse wrongly when a document does address the input but \
uses different wording, reports the opposite result, or covers only part of \
it. They are right to refuse when the documents are on the same topic but \
concern a different {_SCOPE}, or report an association where the input \
asserts cause and effect.

Find the single document that best addresses the input as stated. Reply \
with one JSON object and no other text:
{{"evidence": "<the sentence from that document that addresses the input, copied exactly>", "relevant_doc": <its number>}}

If no document addresses the input as stated, reply:
{{"evidence": "", "relevant_doc": null}}
"""


def build_rescue_messages(
    question: str, docs: list, reason: str, context_chars: int = DEFAULT_CONTEXT_CHARS
) -> list[dict]:
    # Full text, not briefs: telling "same population" from "different
    # population" needs the methods and results, which a snippet cuts off.
    return [
        {"role": "system", "content": RESCUE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{build_context(docs, context_chars)}\n"
                f"{_input_line(question)}\n\n"
                f"The writer's reason for refusing: {reason}"
            ),
        },
    ]


CRITIQUE_SYSTEM_PROMPT = f"""\
You are a strict fact-checker. You are given numbered documents, the user's \
input (a claim to check or a question) and a draft reply written from those \
documents. Find the statements in the draft that the documents do not \
support.

A statement is supported when a document states it, or states something \
that directly implies it. A statement is NOT supported when:
- no document states it or anything that implies it, or a document \
contradicts it;
- it widens a finding beyond the scope the document gives it: a different \
{_SCOPE} than the document studied, or cause and effect where the \
document reports only an association.

The draft may restate the user's claim in order to give a verdict on it \
("the claim that X is contradicted"). The restated claim is not an assertion: \
never list it. Check the verdict instead: "supported" needs a document that \
states the claim, "contradicted" needs a document that states the opposite.

Judge facts only, not style or completeness. A missing citation alone does \
not make a statement unsupported.

Reply with one JSON object and no other text. List the unsupported \
statements first (quote each briefly), then the verdict:
{{"unsupported_claims": ["<statement>", ...], "supported": <true if the list is empty, otherwise false>}}
"""


def build_critique_messages(
    question: str, docs: list, answer: str, context_chars: int = DEFAULT_CONTEXT_CHARS
) -> list[dict]:
    return [
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{build_context(docs, context_chars)}\n\n{_input_line(question)}\n\n"
                f"Draft reply:\n{answer}"
            ),
        },
    ]


HEDGE_PREFIX = (
    "Note: parts of this answer may not be fully supported by the "
    "retrieved documents.\n\n"
)


def build_refusal_text(reason: str, docs: list, limit: int = 3) -> str:
    """Deterministic honest refusal: state the gap, cite what WAS found.

    Starts with the INSUFFICIENT_EVIDENCE marker so `is_refusal()` parses it,
    and names the closest retrieved documents so the user sees what the
    corpus did contain.
    """
    reason = " ".join((reason or "no supporting documents were found").split())
    reason = reason.rstrip(" .")
    if not docs:
        return f"{REFUSAL_MARKER}: {reason}. No documents were retrieved."
    found = "; ".join(
        f"[{i}] {(d.title or '').strip() or d.doc_id}"
        for i, d in enumerate(docs[:limit], start=1)
    )
    return (
        f"{REFUSAL_MARKER}: {reason}. The closest documents retrieved "
        f"were: {found}. None supports an answer to the question."
    )
