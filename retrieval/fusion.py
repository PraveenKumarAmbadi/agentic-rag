"""Hybrid retrieval fusion using Reciprocal Rank Fusion (RRF).

Combines ranked lists from multiple retrieval methods (BM25, SPLADE, dense)
into a single fused ranking. The core function ``fuse_rrf()`` is pure —
no I/O, no config, no model loading — making it trivially testable.

``search_hybrid()`` degrades gracefully: a failing (or circuit-broken)
method contributes an empty list and is reported in
``HybridSearchResult.failed_methods`` so callers can surface the degradation.
"""

from __future__ import annotations

import logging
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense
from retrieval.splade import search_splade
from retrieval.types import FusedResult, SearchResult
from tracing import CircuitOpenError, TraceContext, track_step

if TYPE_CHECKING:  # annotations only: keep torch/faiss out of import time
    from elasticsearch import Elasticsearch

    from indexing.dense_encoder import DenseEncoder
    from indexing.faiss_indexer import FaissIndex
    from indexing.splade_encoder import SpladeEncoder
    from ingestion.loaders import Document

_default_logger = logging.getLogger(__name__)

METHODS = ("bm25", "splade", "dense")


@dataclass(frozen=True)
class HybridSearchResult:
    results: list[FusedResult]
    failed_methods: tuple[str, ...]


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


def _run_method(
    name: str,
    breaker: AbstractContextManager | None,
    ctx: TraceContext,
    logger: logging.Logger,
    search: Callable[[], list[SearchResult]],
    failed: list[str],
) -> list[SearchResult]:
    """Run one retrieval method; on any failure record it and return []."""
    try:
        with track_step(ctx, f"{name}_retrieval"), (breaker or nullcontext()):
            return search()
    except CircuitOpenError as exc:
        logger.warning(
            "%s retrieval skipped: %s", name, exc,
            extra={"trace_id": ctx.trace_id, "method": name},
        )
    except Exception:
        logger.warning(
            "%s retrieval failed, continuing without it", name,
            extra={"trace_id": ctx.trace_id, "method": name}, exc_info=True,
        )
    failed.append(name)
    return []


def search_hybrid(
    query_text: str,
    query_id: str,
    bm25_client: Elasticsearch,
    splade_client: Elasticsearch,
    faiss_index: FaissIndex,
    corpus: dict[str, Document],
    bm25_breaker: AbstractContextManager | None = None,
    splade_breaker: AbstractContextManager | None = None,
    dense_breaker: AbstractContextManager | None = None,
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
    latency_budget_ms: float = 200.0,
) -> HybridSearchResult:
    """Run BM25 + SPLADE + dense retrieval and fuse with RRF.

    Args:
        query_text: The query string.
        query_id: Identifier used as the trace id when no ``ctx`` is given.
        bm25_client / splade_client: Active Elasticsearch clients.
        faiss_index: Pre-loaded FaissIndex for dense search.
        corpus: Mapping doc_id -> Document for metadata lookup.
        *_breaker: Optional circuit breakers, one per method.
        dense_encoder / splade_encoder: Pre-initialized encoders (avoids
            reloading models per call).
        bm25_fields: Fields to search in BM25. Defaults to ["full_text"].
        ctx: Trace context receiving per-method step timings.
        top_k: Number of fused results to return.
        rrf_k: RRF constant passed to ``fuse_rrf()``.
        per_method_top_k: Results fetched per method before fusing (>= top_k).
        latency_budget_ms: Searches slower than this log a warning.

    Returns:
        HybridSearchResult with fused results (best first, at most ``top_k``)
        and the names of methods that failed or were skipped.
    """
    logger = logger or _default_logger
    if ctx is None:
        ctx = TraceContext(trace_id=query_id or str(uuid4()), start_time=time.perf_counter())
    started = time.perf_counter()
    failed: list[str] = []

    bm25_results = _run_method(
        "bm25", bm25_breaker, ctx, logger,
        lambda: search_bm25(
            client=bm25_client,
            index_name=bm25_index_name,
            query_text=query_text,
            top_k=per_method_top_k,
            fields=bm25_fields,
        ),
        failed,
    )
    splade_results = _run_method(
        "splade", splade_breaker, ctx, logger,
        lambda: search_splade(
            client=splade_client,
            query_text=query_text,
            index_name=splade_index_name,
            encoder=splade_encoder,
            top_k=per_method_top_k,
        ),
        failed,
    )
    dense_results = _run_method(
        "dense", dense_breaker, ctx, logger,
        lambda: search_dense(
            query_text=query_text,
            faiss_index=faiss_index,
            corpus=corpus,
            encoder=dense_encoder,
            top_k=per_method_top_k,
        ),
        failed,
    )

    with track_step(ctx, "fusion"):
        fused = fuse_rrf(
            ranked_lists=[bm25_results, splade_results, dense_results],
            k=rrf_k,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    # Never log the query text itself: it is user input.
    log_extra = {
        "trace_id": ctx.trace_id,
        "elapsed_ms": round(elapsed_ms, 1),
        "query_chars": len(query_text),
        "failed_methods": failed,
        "timings": ctx.timings,
    }
    if elapsed_ms >= latency_budget_ms:
        logger.warning("hybrid search exceeded %.0f ms budget", latency_budget_ms, extra=log_extra)
    else:
        logger.info("hybrid search completed", extra=log_extra)

    return HybridSearchResult(results=fused[:top_k], failed_methods=tuple(failed))
