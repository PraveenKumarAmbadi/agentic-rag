"""Shared data types for the retrieval layer.

All retrieval methods (BM25, SPLADE, dense) return SearchResult.
Fusion returns FusedResult, which is kept separate because its
score field has different semantics (RRF score, not raw similarity).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """One ranked result from any first-stage retrieval method.

    The ``score`` field is the raw retrieval score:
    - BM25: Elasticsearch relevance score (unbounded, higher = better)
    - SPLADE: sparse vector dot product (higher = better)
    - Dense: cosine similarity via inner product (0 to 1, higher = better)
    """
    doc_id: str
    score: float
    title: str
    text: str


@dataclass(frozen=True)
class FusedResult:
    """One result from RRF fusion.

    The ``score`` field is the RRF score:
    sum over methods of 1 / (k + rank_in_that_method).
    """
    doc_id: str
    score: float
    title: str
    text: str


@dataclass(frozen=True)
class RerankedResult:
    """One result after cross-encoder reranking.

    The ``score`` field is the cross-encoder relevance score.
    Range is model-dependent (typically unbounded, often 0 to ~10).
    Higher = more relevant.

    ``original_rank`` preserves the position before reranking
    (1-based) for analysis and debugging.
    """
    doc_id: str
    score: float
    title: str
    text: str
    original_rank: int
