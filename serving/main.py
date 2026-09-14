"""FastAPI app — SSE adapter over the shared RagPipeline.

All RAG logic lives in agent/pipeline.py; this file only wires infra
(config, ES, encoders, LLM) in lifespan and maps pipeline events to SSE.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.llm import LLMClient, OllamaError
from agent.pipeline import RagPipeline
from configs.loader import load_config
from es_client import get_es_client
from indexing.dense_encoder import DenseEncoder
from indexing.faiss_indexer import FaissIndex
from indexing.splade_encoder import SpladeEncoder
from ingestion.loaders import load_corpus
from tracing import JsonFormatter, setup_logger


def _sse(payload: dict) -> str:
    import json

    return f"data: {json.dumps(payload)}\n\n"


class AnswerRequest(BaseModel):
    question: str


async def _load_stack(cfg: dict, logger: logging.Logger) -> dict:
    """Load heavy resources once (runs during lifespan, off the event loop)."""
    loop = asyncio.get_running_loop()
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
    return {
        "corpus": corpus,
        "faiss_index": faiss_index,
        "dense_encoder": dense_encoder,
        "splade_encoder": splade_encoder,
        "es": get_es_client(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = setup_logger(
        name="agentic_rag", formatter=JsonFormatter(), level=logging.INFO
    )
    cfg = load_config()
    stack = await _load_stack(cfg, logger)
    app.state.cfg = cfg
    app.state.logger = logger
    app.state.llm = LLMClient.from_config(cfg)
    app.state.pipeline = RagPipeline(logger=logger, llm=app.state.llm, **stack)
    logger.info("Agentic RAG API started (llm=%s)", cfg["llm"]["model"])
    yield
    await app.state.llm.close()


app = FastAPI(lifespan=lifespan, title="Agentic RAG")


@app.get("/health")
async def health(request: Request):
    pipeline: RagPipeline = request.app.state.pipeline
    llm_ok = await request.app.state.llm.is_up()
    return {
        "elasticsearch": "up" if bool(pipeline.es.ping()) else "down",
        "llm": "up" if llm_ok else "down",
    }


@app.post("/answer")
async def answer(request: Request, payload: AnswerRequest):
    pipeline: RagPipeline = request.app.state.pipeline
    if not await request.app.state.llm.is_up():
        raise OllamaError("LLM backend is down — start Ollama first.")

    async def event_stream():
        async for ev in pipeline.answer_stream(payload.question):
            if ev["type"] == "delta":
                yield _sse({"delta": ev["text"]})
            else:
                yield _sse({k: v for k, v in ev.items() if k != "text"})
                request.app.state.logger.info(
                    "answer streamed",
                    extra={"trace_id": ev["trace_id"], "timings": ev["timings"]},
                )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.main:app", host="0.0.0.0", port=8001)
