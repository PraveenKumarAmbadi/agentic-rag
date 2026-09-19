"""Prompt templates for grounded answer generation and the agent loop.

Design rules baked into the answer system prompt:
- Answer ONLY from the numbered documents (no parametric knowledge).
- Cite every claim inline as [n] referencing the document number.
- An explicit escape hatch ("INSUFFICIENT_EVIDENCE: ...") so refusal is a
  measurable, parseable event rather than a vibe.

The agent-loop prompts follow the same philosophy:
- DECIDE (controller): one JSON object choosing the next tool; biased toward
  answering because the measured failure mode is over-refusal, not guessing.
- RESCUE (second opinion): challenges a proposed refusal by asking which
  retrieved document is most relevant — the direct counter to false refusals.
- CRITIQUE (fact-checker): lists claims not supported by the documents.

The context block is built by `build_context()`; document numbering in every
prompt is 1-based and maps to `doc_ids[i-1]`.
"""

from agent.tools import render_tool_docs

SYSTEM_PROMPT = """\
You are a careful research assistant. Answer the question using ONLY the \
numbered documents provided below — never from your own knowledge.

Rules:
1. Every factual claim must be followed by a citation to the document it \
came from, in square brackets, e.g. the effect was significant [2].
2. Cite the minimum set of documents needed; cite by document number only.
3. If the documents do not contain enough information to answer, reply \
with exactly one line starting with: INSUFFICIENT_EVIDENCE: followed by a \
short description of what is missing. Do not guess.
4. Be concise: at most 150 words.
"""

CONTEXT_HEADER = "Documents:"

# Default per-doc truncation: ~800 chars vs a median gold abstract of
# ~1200. Truncation does NOT discriminate refusals from answers (gold docs
# exceeded it in 53/56 refused vs 28/31 answered) — the baseline
# over-refusal is model caution under the strict prompt, not missing
# evidence. Config `agent.context_chars` overrides this for both pipelines.
DEFAULT_CONTEXT_CHARS = 800

# How much of each document the decide/rescue turns see (full text is only
# given to the generation and critique turns, which need it).
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
        lines.append(f"[{i}] ({doc.doc_id}) {title}\n{text}\n")
    return "\n".join(lines)


def build_doc_briefs(docs: list, chars: int = BRIEF_CHARS) -> str:
    """Compact numbered doc list (title + snippet) for control turns."""
    lines = []
    for i, doc in enumerate(docs, start=1):
        title = (doc.title or "").strip() or doc.doc_id
        text = " ".join((doc.text or "").split())[:chars]
        lines.append(f"[{i}] {title} — {text}")
    return "\n".join(lines) if lines else "(no documents retrieved)"


def build_messages(
    question: str, docs: list, context_chars: int = DEFAULT_CONTEXT_CHARS
) -> list[dict[str, str]]:
    """Assemble the chat messages for a grounded answer (baseline pipeline)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{build_context(docs, context_chars)}\nQuestion: {question}"},
    ]


def build_answer_messages(
    question: str,
    docs: list,
    hint_doc: int | None = None,
    feedback: list[str] | None = None,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> list[dict[str, str]]:
    """Grounded-answer messages with optional rescue hint / critique feedback.

    Args:
        hint_doc: 1-based doc number the rescue step identified as relevant;
            generation switches to quote-then-answer mode for that doc.
        feedback: unsupported claims from a failed critique; the model is
            asked to rewrite the answer without them.
        context_chars: Per-doc truncation for the context block.
    """
    parts = [build_context(docs, context_chars), f"Question: {question}"]
    if hint_doc is not None:
        doc = docs[hint_doc - 1]
        title = (doc.title or "").strip() or doc.doc_id
        parts.append(
            f"An independent reviewer examined the retrieved documents and "
            f"identified document [{hint_doc}] ({title}) as relevant to this "
            "question. First locate the sentence(s) in document "
            f"[{hint_doc}] that bear on the question and quote them, then "
            "answer the question using what they say. Refuse only if, after "
            "quoting, those sentences truly cannot answer the question."
        )
    if feedback:
        listed = "\n".join(f"- {c}" for c in feedback)
        parts.append(
            "Your previous draft contained claims a fact-checker could not "
            f"verify against the documents:\n{listed}\n"
            "Write the answer again. Keep only claims the documents support, "
            "and cite the supporting document for every claim."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# ---------------------------------------------------------------------------
# Agent-loop prompts
# ---------------------------------------------------------------------------

DECIDE_SYSTEM_PROMPT = f"""\
You are the controller of a retrieval-augmented research assistant. You \
decide the next action for answering the user's question from a scientific \
corpus.

{render_tool_docs()}

Decision policy — refusing when a retrieved document actually addresses the \
question is a WORSE failure than an imperfect answer:
- If any retrieved document plausibly bears on the question (even partially, \
even for only one aspect), choose "answer".
- Choose "hybrid_search" with a reformulated query only if the documents so \
far are clearly off-topic for the question.
- Choose "insufficient_evidence" only after a search whose results still do \
not address the question at all.

Reply with exactly ONE JSON object and nothing else.
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
        f"Question: {question}",
        f"\nRetrieved documents (best first):\n{build_doc_briefs(docs)}",
    ]
    if history:
        parts.append("\nActions so far: " + "; ".join(history))
    parts.append(
        f"\nIteration {iteration} of {max_iterations}. Choose the next action."
    )
    return [
        {"role": "system", "content": DECIDE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


RESCUE_SYSTEM_PROMPT = """\
You are a skeptical second opinion. Another assistant wants to refuse to \
answer the question, claiming insufficient evidence. Your job is to check \
whether that refusal is justified by the retrieved documents.

Reply with exactly ONE JSON object:
{"relevant_doc": <document number or null>, "justification": "<one sentence>"}

Set relevant_doc to the number of the single document most relevant to the \
question if ANY retrieved document bears on the question — even partially, \
even for only one aspect. Set it to null only if every document is truly \
unrelated to the question. Do not set it to null just because no document \
answers the question completely.
"""


def build_rescue_messages(question: str, docs: list, reason: str) -> list[dict]:
    return [
        {"role": "system", "content": RESCUE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Proposed refusal reason: {reason}\n\n"
                f"Retrieved documents (best first):\n{build_doc_briefs(docs, 400)}"
            ),
        },
    ]


CRITIQUE_SYSTEM_PROMPT = """\
You are a strict fact-checker. You are given numbered documents and a draft \
answer supposedly written using only those documents. Identify claims in the \
draft that are NOT supported by the documents.

A claim is supported only if one of the documents states it or something \
that directly implies it. A claim without a citation is not, by itself, \
unsupported — only contradicted or invented facts are.

Reply with exactly ONE JSON object:
{"supported": true|false, "unsupported_claims": ["<claim>", ...]}

"supported" is true when every factual claim in the draft is backed by the \
documents. Otherwise it is false and each unsupported claim is quoted briefly.
"""


def build_critique_messages(
    question: str, docs: list, answer: str, context_chars: int = DEFAULT_CONTEXT_CHARS
) -> list[dict]:
    return [
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{build_context(docs, context_chars)}\n\nQuestion: {question}\n\n"
                f"Draft answer:\n{answer}"
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
    found = "; ".join(
        f"[{i}] {(d.title or '').strip() or d.doc_id}"
        for i, d in enumerate(docs[:limit], start=1)
    )
    reason = " ".join((reason or "no supporting documents were found").split())
    return (
        f"INSUFFICIENT_EVIDENCE: {reason}. The closest documents retrieved "
        f"were: {found}. None supports an answer to the question."
    )
    
    
    

