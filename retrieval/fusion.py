"""Hybrid retrieval fusion using Reciprocal Rank Fusion (RRF).

Combines ranked lists from multiple retrieval methods (BM25, SPLADE, dense)
into a single fused ranking. The core function ``fuse_rrf()`` is pure —
no I/O, no config, no model loading — making it trivially testable.
"""



from elasticsearch import Elasticsearch
from contextlib import nullcontext

from indexing.dense_encoder import DenseEncoder
from indexing.splade_encoder import SpladeEncoder
from indexing.faiss_indexer import FaissIndex
from ingestion.loaders import Document
from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense
from retrieval.splade import search_splade
from retrieval.types import FusedResult, SearchResult
from tracing import TraceContext, track_step
from uuid import uuid4
import logging
import time
from tracing import TraceContext, track_step, CircuitBreaker


def fuse_rrf(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[FusedResult]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Title and text are taken from the first list where each document appears.

    Args:
        ranked_lists: List of ranked result lists, one per retrieval method.
            Each inner list must be sorted by relevance (best first).
        k: RRF constant. Default 60 (standard from the original RRF paper).
            Higher values flatten the rank distribution, giving more weight
            to lower-ranked items.

    Returns:
        Fused results sorted by RRF score descending.
    """
    # doc_id -> {"score": accumulated_rrf, "title": str, "text": str}
    scores: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            doc_id = result.doc_id
            rrf_score = 1.0 / (k + rank)

            if doc_id not in scores:
                scores[doc_id] = {
                    "score": 0.0,
                    "title": result.title,
                    "text": result.text,
                }
            scores[doc_id]["score"] += rrf_score

    # Sort by RRF score descending
    sorted_docs = sorted(
        scores.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )

    return [
        FusedResult(
            doc_id=doc_id,
            score=info["score"],
            title=info["title"],
            text=info["text"],
        )
        for doc_id, info in sorted_docs
    ]


def search_hybrid(
    query_text: str,
    query_id: str,
    bm25_client: Elasticsearch,
    splade_client: Elasticsearch, 
    faiss_index: FaissIndex,
    corpus: dict[str, Document],
    bm25_breaker: CircuitBreaker | nullcontext = nullcontext(), 
    splade_breaker:CircuitBreaker | nullcontext = nullcontext(), 
    dense_breaker:CircuitBreaker | nullcontext = nullcontext(), 
    dense_encoder: DenseEncoder | None = None,
    splade_encoder: SpladeEncoder | None = None,
    bm25_index_name: str = "scifact_bm25",
    splade_index_name: str = "scifact_splade",
    bm25_fields: list[str] | None = None,
    ctx: TraceContext | None = None,
    logger: logging.Logger | None = None,
    top_k: int = 100,
    rrf_k: int = 60,
    per_method_top_k: int = 100,
) -> list[FusedResult]:

    """Run BM25 + SPLADE + dense retrieval and fuse with RRF.

    Args:
        query_text: The query string.
        bm25_client: Active Elasticsearch client for BM25 search.
        faiss_index: Pre-loaded FaissIndex for dense search.
        corpus: Mapping doc_id -> Document for metadata lookup.
        dense_encoder: Optional pre-initialized DenseEncoder. Passing an
            existing encoder avoids reloading the model on every call.
        bm25_index_name: ES index for BM25. Default: "scifact_bm25".
        splade_index_name: ES index for SPLADE. Default: "scifact_splade".
        top_k: Number of fused results to return.
        rrf_k: RRF constant passed to ``fuse_rrf()``.
        per_method_top_k: How many results to fetch from each individual
            method before fusing. Should be >= top_k.
        bm25_fields: Fields to search in BM25. Defaults to ["full_text"].

    Returns:
        Fused results sorted by RRF score, truncated to ``top_k`` items.
    """
    if ctx is None:
        trace_id = query_id or str(uuid4())

        start_time = time.perf_counter()
        ctx = TraceContext(trace_id=trace_id, start_time=start_time)

    if logger is None:
      raise ValueError("logger is required")
    
    bm25_breaker = bm25_breaker if bm25_breaker is not None else nullcontext()
    splade_breaker = splade_breaker if splade_breaker is not None else nullcontext()
    dense_breaker = dense_breaker if dense_breaker is not None else nullcontext()

    try:
      with track_step(ctx, 'bm25_retrieval'), bm25_breaker:
          bm25_results = search_bm25(
              client=bm25_client,
              index_name=bm25_index_name,
              query_text=query_text,
              top_k=per_method_top_k,
              fields=bm25_fields,
          )
    except Exception as e:
        logger.warning(
            f"BM25 retrieval failed, using empty fallback. Error: {e}",
            extra={"trace_id": ctx.trace_id},  exc_info=True
        )

        bm25_results = []

    try:
      with track_step(ctx, 'spade_retrieval'), splade_breaker:
          splade_results = search_splade(
              client=splade_client,
              query_text=query_text,
              index_name=splade_index_name,
              encoder = splade_encoder,
              top_k=per_method_top_k,
          )
    except Exception as e:
        logger.warning(
            f"splade retrieval failed, using empty fallback. Error: {e}",
            extra={"trace_id": ctx.trace_id},
            exc_info=True
        )

        splade_results = []

    try:
      with  track_step(ctx, 'dense_retrieval'),dense_breaker:
          dense_results = search_dense(
              query_text=query_text,
              faiss_index=faiss_index,
              corpus=corpus,
              encoder=dense_encoder,
              top_k=per_method_top_k,
          )
    except Exception as e:
        logger.warning(
            f"dense retrieval failed, using empty fallback. Error: {e}",
            extra={"trace_id": ctx.trace_id},  exc_info=True
        )

        dense_results = []

    with track_step(ctx, "fusion"):
        fused = fuse_rrf(
            ranked_lists=[bm25_results, splade_results, dense_results],
            k=rrf_k,
        )

    Total_elpased_time = (time.perf_counter() - ctx.start_time) * 1000

    if Total_elpased_time >= 200:
      logger.warning(f"Query ID {query_id} took too long: {query_text}" , extra={"trace_id": ctx.trace_id, "timings": ctx.timings})
    else:
      logger.info("Hybrid search completed", extra={"trace_id": ctx.trace_id, "timings": ctx.timings})

    return fused[:top_k]
