"""Standard information retrieval metrics, implemented from scratch.

These metrics currently assume binary relevance. Graded nDCG is not implemented yet.

Each metric is a pure function: given a ranked list and a set of relevant docs,
it returns a scalar.  No I/O, no side effects, trivially unit-testable.
"""

from typing import List, Set
import math

def _validate_k(k: int):
    
    if k < 0: 
       raise ValueError("K must be postive")

def recall_at_k(ranked_doc_ids: List[str], relevant_docs: Set[str], k: int) -> float:
    """Fraction of all relevant documents found in the top-K results.

    Recall@K = |relevant ∩ top_K| / |relevant|

    Returns 0.0 if there are no relevant documents for this query.
    """
    _validate_k(k=k)

    if not relevant_docs:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    return len(relevant_docs & top_k) / len(relevant_docs)


def precision_at_k(ranked_doc_ids: List[str], relevant_docs: Set[str], k: int) -> float:
    """Fraction of top-K results that are relevant.

    Precision@K = |relevant ∩ top_K| / K

    Returns 0.0 if K == 0.
    """
    _validate_k(k=k)

    if k == 0:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    return len(relevant_docs & top_k) / k


def _dcg_at_k(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain for a relevance vector.

    DCG@K = Σ_{i=1}^{K} rel_i / log2(i + 1)

    ``relevances[i]`` is the relevance score (0 or 1 for binary) of the
    document at rank ``i + 1``.
    """
    _validate_k(k=k)

    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_doc_ids: List[str], relevant_docs: Set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain.

    nDCG@K = DCG@K / IDCG@K

    IDCG is the DCG of the ideal ranking (all relevant docs at the top).
    For binary relevance, IDCG@K = Σ_{i=1}^{min(R,K)} 1 / log2(i + 1)
    where R = number of relevant docs.

    Returns 0.0 if there are no relevant documents.
    """
    
    _validate_k(k=k)

    if not relevant_docs:
        return 0.0

    # Build relevance vector: 1 if doc is relevant, 0 otherwise
    relevances = [1 if doc_id in relevant_docs else 0 for doc_id in ranked_doc_ids]

    dcg = _dcg_at_k(relevances, k)

    # Ideal DCG: all relevant docs ranked first
    r = len(relevant_docs)
    ideal_relevances = [1] * r
    idcg = _dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranked_doc_ids: List[str], relevant_docs: Set[str]) -> float:
    """Reciprocal rank: 1 / rank_of_first_relevant_doc.

    Returns 0.0 if no relevant document appears in the ranked list.
    """
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_docs:
            return 1.0 / i
    return 0.0


def average_precision_at_k(ranked_doc_ids: List[str], relevant_docs: Set[str], k: int) -> float:
    """Average Precision at K.

    AP@K = (1 / min(R, K)) * Σ_{i=1}^{K} [ Precision@i * rel(i) ]

    Where R = number of relevant documents.  For each rank i where a
    relevant document appears, we record the precision at that rank, then
    average those precision values.

    Returns 0.0 if there are no relevant documents.
    """
 
    _validate_k(k=k)

    if len(relevant_docs) == 0 or k == 0:
        return 0.0

    precisions_at_relevant_ranks = []
    hits_so_far = 0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_docs:
            hits_so_far += 1
            precisions_at_relevant_ranks.append(hits_so_far/i)

    min_RK = min(len(relevant_docs), k)
    return sum(precisions_at_relevant_ranks) / min_RK 