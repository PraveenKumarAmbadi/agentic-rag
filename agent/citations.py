"""Citation extraction and validation — deterministic, no LLM involved.

The prompt forces the model to cite documents as [1], [2], ... This module
extracts those markers from the generated text, rejects hallucinated marker
numbers (out of range = cited a document we never retrieved), and maps
markers back to real corpus doc_ids for the API response and evals.
"""

import re

# [12] style markers; 1-2 digits covers any realistic context size.
CITATION_RE = re.compile(r"\[(\d{1,2})\]")


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


def is_refusal(text: str) -> bool:
    """True if the model used the INSUFFICIENT_EVIDENCE escape hatch."""
    return text.lstrip().startswith("INSUFFICIENT_EVIDENCE")
