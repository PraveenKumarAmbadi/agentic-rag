"""Shared RAG pipeline — one code path for the API and the eval runner.

`answer_stream()` is the single implementation of:
  hybrid retrieval -> grounded prompt -> streamed generation -> citation
  validation -> final result.

The FastAPI layer maps these events to SSE; the eval runner consumes the
`done` event. Refactoring retrieval or prompting only touches this file, so
eval numbers can never silently drift from what production serves.
"""

import asyncio
import time
from uuid import uuid4

from agent.citations import build_citation_map, is_refusal, parse_citations
from agent.prompts import build_messages
from retrieval.fusion import search_hybrid
from tracing import TraceContext


class RagPipeline:
    def __init__(
        self,
        cfg: dict,
        es,
        corpus: dict,
        faiss_index,
        dense_encoder,
        splade_encoder,
        llm,
        logger,
    ) -> None:
        self.cfg = cfg
        self.es = es
        self.corpus = corpus
        self.faiss_index = faiss_index
        self.dense_encoder = dense_encoder
        self.splade_encoder = splade_encoder
        self.llm = llm
        self.logger = logger

    async def retrieve(self, question: str, trace_id: str, ctx: TraceContext):
        """Hybrid retrieval. Sync GPU/ES work runs in a worker thread."""

        def _run():
            return search_hybrid(
                query_text=question,
                query_id=trace_id,
                bm25_client=self.es,
                splade_client=self.es,
                faiss_index=self.faiss_index,
                corpus=self.corpus,
                dense_encoder=self.dense_encoder,
                splade_encoder=self.splade_encoder,
                bm25_index_name=self.cfg["bm25"]["index_name"],
                splade_index_name=self.cfg["splade"]["index_name"],
                logger=self.logger,
                ctx=ctx,
                top_k=self.cfg["agent"]["top_k"],
                per_method_top_k=self.cfg["agent"]["per_method_top_k"],
            )

        start = time.perf_counter()
        results = await asyncio.to_thread(_run)
        return results, (time.perf_counter() - start) * 1000

    async def answer_stream(self, question: str, trace_id: str | None = None):
        """Yield {'type': 'delta', 'text': ...} events, then one 'done' event.

        The 'done' event carries the full answer text, refusal flag, validated
        citations, retrieved doc_ids, and timing breakdown.
        """
        trace_id = trace_id or str(uuid4())
        start = time.perf_counter()
        ctx = TraceContext(trace_id=trace_id, start_time=start)

        results, retrieval_ms = await self.retrieve(question, trace_id, ctx)
        doc_ids = [r.doc_id for r in results]
        messages = build_messages(question, results)

        first_token_ms = None
        chunks: list[str] = []
        async for delta in self.llm.stream_chat(messages):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - start) * 1000
            chunks.append(delta)
            yield {"type": "delta", "text": delta}

        text = "".join(chunks)
        cites = parse_citations(text, len(results))
        timings = {
            "retrieval_ms": round(retrieval_ms, 1),
            "first_token_ms": round(first_token_ms or 0.0, 1),
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
        }
        yield {
            "type": "done",
            "trace_id": trace_id,
            "text": text,
            "refusal": is_refusal(text),
            "citations": build_citation_map(cites["markers"], doc_ids),
            "invalid_citations": cites["invalid"],
            "retrieved_doc_ids": doc_ids,
            "step_timings": dict(ctx.timings),
            "timings": timings,
        }
