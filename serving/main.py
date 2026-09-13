"""FastAPI app — Phase 1: hybrid retrieval + grounded, cited answer streaming.

Retrieval stack (ES BM25 + SPLADE + FAISS dense, fused via RRF) is vendored
from the rag-hybrid-retrieval project and loaded/warmed once in lifespan.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.citations import build_citation_map, is_refusal, parse_citations
from agent.llm import LLMClient, OllamaError
from agent.prompts import build_messages
from configs.loader import load_config
from es_client import get_es_client
from indexing.dense_encoder import DenseEncoder
from indexing.faiss_indexer import FaissIndex
from indexing.splade_encoder import SpladeEncoder
from ingestion.loaders import load_corpus
from retrieval.fusion import search_hybrid
from tracing import JsonFormatter, TraceContext, setup_logger


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class AnswerRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    logger = setup_logger(
        name="agentic_rag", formatter=JsonFormatter(), level=logging.INFO
    )
    cfg = load_config()
    loop = asyncio.get_running_loop()

    logger.info("Loading retrieval stack (corpus, FAISS, encoders) ...")
    corpus = await loop.run_in_executor(
        None, lambda: load_corpus(Path(cfg["data"]["corpus_path"]))
    )
    faiss_index = await loop.run_in_executor(
        None,
        lambda: FaissIndex.load(cfg["dense"]["index_path"], cfg["dense"]["ids_path"]),
    )
    dense_encoder = DenseEncoder()
    splade_encoder = SpladeEncoder()
    await loop.run_in_executor(None, lambda: dense_encoder.encode_query("warmup"))
    await loop.run_in_executor(None, lambda: splade_encoder.encode_query("warmup"))
    es = get_es_client()

    app.state.cfg = cfg
    app.state.logger = logger
    app.state.llm = LLMClient.from_config(cfg)
    app.state.es = es
    app.state.corpus = corpus
    app.state.faiss_index = faiss_index
    app.state.dense_encoder = dense_encoder
    app.state.splade_encoder = splade_encoder

    logger.info("Agentic RAG API started (llm=%s)", cfg["llm"]["model"])
    yield
    await app.state.llm.close()


app = FastAPI(lifespan=lifespan, title="Agentic RAG")


@app.get("/health")
async def health(request: Request):
    es_ok = bool(request.app.state.es.ping())
    llm_ok = await request.app.state.llm.is_up()
    return {
        "elasticsearch": "up" if es_ok else "down",
        "llm": "up" if llm_ok else "down",
    }


@app.post("/answer")
async def answer(request: Request, payload: AnswerRequest):
    """Retrieve via hybrid search, then stream a grounded, cited answer."""
    import asyncio

    trace_id = str(uuid4())
    start = time.perf_counter()
    ctx = TraceContext(trace_id=trace_id, start_time=start)
    state = request.app.state
    llm: LLMClient = state.llm
    logger: logging.Logger = state.logger
    cfg = state.cfg

    if not await llm.is_up():
        raise OllamaError("LLM backend is down — start Ollama first.")

    # --- Stage 1: hybrid retrieval (sync GPU/ES work in a worker thread) ---
    def _retrieve():
        return search_hybrid(
            query_text=payload.question,
            query_id=trace_id,
            bm25_client=state.es,
            splade_client=state.es,
            faiss_index=state.faiss_index,
            corpus=state.corpus,
            dense_encoder=state.dense_encoder,
            splade_encoder=state.splade_encoder,
            bm25_index_name=cfg["bm25"]["index_name"],
            splade_index_name=cfg["splade"]["index_name"],
            logger=logger,
            ctx=ctx,
            top_k=cfg["agent"]["top_k"],
            per_method_top_k=cfg["agent"]["per_method_top_k"],
        )

    results = await asyncio.to_thread(_retrieve)
    retrieval_ms = (time.perf_counter() - start) * 1000
    doc_ids = [r.doc_id for r in results]

    # --- Stage 2: grounded generation ---
    messages = build_messages(payload.question, results)
    first_token_ms = None
    chunks: list[str] = []

    async def event_stream():
        nonlocal first_token_ms
        async for delta in llm.stream_chat(messages):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - start) * 1000
            chunks.append(delta)
            yield _sse({"delta": delta})

        answer_text = "".join(chunks)
        cites = parse_citations(answer_text, len(results))
        timings = {
            "retrieval_ms": round(retrieval_ms, 1),
            "first_token_ms": round(first_token_ms or 0.0, 1),
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
        }
        yield _sse(
            {
                "done": True,
                "trace_id": trace_id,
                "refusal": is_refusal(answer_text),
                "citations": build_citation_map(cites["markers"], doc_ids),
                "invalid_citations": cites["invalid"],
                "retrieved_doc_ids": doc_ids,
                "timings": timings,
            }
        )
        logger.info(
            "answer streamed",
            extra={
                "trace_id": trace_id,
                "timings": {**ctx.timings, **timings},
            },
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.main:app", host="0.0.0.0", port=8001)
