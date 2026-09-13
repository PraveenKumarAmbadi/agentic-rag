"""Prompt templates for grounded answer generation.

Design rules baked into the system prompt:
- Answer ONLY from the numbered documents (no parametric knowledge).
- Cite every claim inline as [n] referencing the document number.
- An explicit escape hatch ("INSUFFICIENT_EVIDENCE: ...") so refusal is a
  measurable, parseable event rather than a vibe.

The context block is built by `build_context()`; document numbering in the
prompt is 1-based and maps to `doc_ids[i-1]`.
"""

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

CONTEXT_HEADER = "Documents:\n"


def build_context(docs: list) -> str:
    """Format retrieved documents as the numbered context block.

    Args:
        docs: Retrieved results (any objects with doc_id/title/text), best first.

    Returns:
        Context string; each doc truncated to keep the prompt bounded.
    """
    lines = [CONTEXT_HEADER]
    for i, doc in enumerate(docs, start=1):
        title = (doc.title or "").strip()
        text = " ".join((doc.text or "").split())[:800]
        lines.append(f"[{i}] ({doc.doc_id}) {title}\n{text}\n")
    return "\n".join(lines)


def build_messages(question: str, docs: list) -> list[dict[str, str]]:
    """Assemble the chat messages for a grounded answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{build_context(docs)}\nQuestion: {question}"},
    ]
