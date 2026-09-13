"""Cross-encoder reranking for hybrid retrieval results.

Takes the top-K candidates from first-stage retrieval (BM25 + SPLADE + dense
fused via RRF) and re-scores each (query, document) pair using a
cross-encoder.  The cross-encoder sees both texts simultaneously, producing a
much more accurate relevance score than any bi-encoder or sparse method.

Usage::
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query_text, fused_results, top_k=10)
"""

import logging
from typing import List

import torch
from sentence_transformers import CrossEncoder

from configs.loader import load_config
from retrieval.types import FusedResult, RerankedResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Lazy-loading cross-encoder reranker.

    The model is loaded on first use (not at import time) so that importing
    this module is cheap and testable without GPU/MODEL overhead.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_seq_length: int = 512,
        device: str | None = None,
    ) -> None:
        """Initialise with config-driven or explicit overrides.

        Args:
            model_name: HuggingFace model identifier.  Defaults to
                ``config["reranker"]["model_name"]``.
            max_seq_length: Max tokens per (query+doc) pair.  Longer
                sequences are truncated.
            device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
        """
        self.model_name = model_name or load_config()["reranker"]["model_name"]
        self.max_seq_length = max_seq_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder once."""
        if self._model is None:
            logger.info(
                "Loading cross-encoder %s on %s ...",
                self.model_name,
                self.device,
            )
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_seq_length,
                device=self.device,
            )
        return self._model

    def rerank(
        self,
        query_text: str,
        results: List[FusedResult],
        batch_size: int = 16,
        top_k: int | None = None,
    ) -> List[RerankedResult]:
        """Re-score *results* with the cross-encoder and return re-sorted list.

        The cross-encoder concatenates ``query_text`` with each result's
        title+text, runs full self-attention over the pair, and outputs a
        relevance score.  Because it sees both texts together it is far more
        accurate than any first-stage retriever, but it is also ~100x slower
        per document — hence we only run it on the top-K candidates.

        Args:
            query_text: The original query string.
            results: Ranked list from first-stage retrieval (typically
                ``search_hybrid()`` output).  Must contain ``doc_id``,
                ``title``, and ``text``.
            batch_size: Number of (query, doc) pairs per GPU batch.
                Tune down if you hit OOM; tune up for speed.
            top_k: If given, truncate the final re-sorted list to this many
                items.  ``None`` means return all reranked results.

        Returns:
            ``RerankedResult`` list sorted by cross-encoder score descending.
            ``original_rank`` preserves the 1-based position from the input
            ``results`` list for before/after analysis.
        """
        if not results:
            return []

        model = self._load_model()
        top_k = top_k or len(results)

        # Build (query, document) pairs for the cross-encoder.
        # We concatenate title + text so the model sees the full context,
        # matching how BM25 searches against the combined ``full_text`` field.
        pairs: list[tuple[str, str]] = []
        for r in results:
            doc_text = f"{r.title} {r.text}".strip() if r.title else r.text
            pairs.append((query_text, doc_text))

        #logger.info(
        #    "Reranking %d results (batch_size=%d, device=%s) ...",
        #    len(pairs),
        #    batch_size,
        #    self.device,
        #)

        # Run inference.  CrossEncoder.predict returns a 1-D numpy array of
        # scores when ``convert_to_numpy=True`` (the default).
        raw_scores = model.predict(  # type: ignore[reportUnknownMemberType]
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Attach scores and original rank, then re-sort.
        reranked = [
            RerankedResult(
                doc_id=results[i].doc_id,
                score=float(raw_scores[i]),
                title=results[i].title,
                text=results[i].text,
                original_rank=i + 1,
            )
            for i in range(len(results))
        ]

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]