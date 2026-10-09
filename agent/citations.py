"""Citation and refusal parsing — deterministic, no LLM involved.

The prompt forces the model to cite documents as [1], [2], ... This module
extracts those markers from the generated text, rejects hallucinated marker
numbers (out of range = cited a document we never retrieved), and maps
markers back to real corpus doc_ids for the API response and evals.

It is also the single definition of what counts as a refusal. The streaming
gate (agent.RefusalGate), the agent loop, and the eval runner all classify
text through ``classify_refusal_prefix()`` / ``is_refusal()``, so a refusal
that is intercepted while streaming is exactly a refusal that is scored.
"""

import re
from typing import Literal

# [12] style markers; 1-2 digits covers any realistic context size.
CITATION_RE = re.compile(r"\[(\d{1,2})\]")

# The escape hatch the answer prompt asks for. Models vary the spelling
# ("INSUFFICIENT EVIDENCE", "**Insufficient_Evidence**:", "... - reason"),
# so matching is case-, underscore-, whitespace- and markdown-insensitive.
REFUSAL_MARKER = "INSUFFICIENT_EVIDENCE"
_CANONICAL = "INSUFFICIENT EVIDENCE"
_LEADING_NOISE = " \t\r\n*#>`"
_REFUSAL_RE = re.compile(
    r"^[\s*#>`]*insufficient[\s_]+evidence(?![a-z0-9])[\s*`_]*[:\-–—.]?\s*(.*)\Z",
    re.IGNORECASE | re.DOTALL,
)

RefusalPrefix = Literal["refusal", "undecided", "answer"]


def _normalize_prefix(text: str) -> str:
    stripped = text.lstrip(_LEADING_NOISE).upper().replace("_", " ")
    return " ".join(stripped.split())


def classify_refusal_prefix(text: str, final: bool = False) -> RefusalPrefix:
    """Classify the start of a (possibly partial) answer.

    Args:
        text: Answer text so far.
        final: True when the stream has ended, so "could still become the
            marker" must be resolved one way or the other.

    Returns:
        "refusal"   the text starts with the refusal marker,
        "answer"    it cannot be a refusal,
        "undecided" it is still a prefix of the marker (only when not final).
    """
    probe = _normalize_prefix(text)
    if probe.startswith(_CANONICAL):
        rest = probe[len(_CANONICAL):]
        if not rest:
            return "refusal" if final else "undecided"
        return "answer" if rest[0].isalnum() else "refusal"
    if not final and _CANONICAL.startswith(probe):
        return "undecided"
    return "answer"


def is_refusal(text: str) -> bool:
    """True if the complete text uses the INSUFFICIENT_EVIDENCE escape hatch."""
    return classify_refusal_prefix(text, final=True) == "refusal"


def refusal_reason(text: str) -> str:
    """The model's stated reason after the refusal marker ('' if none)."""
    match = _REFUSAL_RE.match(text)
    return " ".join(match.group(1).split()) if match else ""


def parse_citations(text: str, num_docs: int) -> dict:
    """Extract and validate inline citation markers from generated text.

    Args:
        text: The full generated answer.
        num_docs: How many documents were in the context (valid marker range).

    Returns:
        {
            "markers":   sorted unique valid 1-based markers, e.g. [1, 3],
            "invalid":   markers cited but out of range (hallucinated),
            "grounded":  True if at least one valid citation and none invalid,
        }
    """
    found = [int(m) for m in CITATION_RE.findall(text)]
    valid = sorted({n for n in found if 1 <= n <= num_docs})
    invalid = sorted({n for n in found if not (1 <= n <= num_docs)})
    return {
        "markers": valid,
        "invalid": invalid,
        "grounded": bool(valid) and not invalid,
    }


def build_citation_map(markers: list[int], doc_ids: list[str]) -> dict[str, str]:
    """Map 1-based citation markers to real corpus doc_ids.

    Args:
        markers: Valid marker numbers from parse_citations().
        doc_ids: The doc_ids passed in the context, in prompt order.

    Returns:
        {"1": "17930286", ...} — only for cited markers.
    """
    return {str(n): doc_ids[n - 1] for n in markers if n <= len(doc_ids)}
