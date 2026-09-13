"""FastAPI app — Phase 0: health + baseline LLM streaming (no retrieval yet)."""

import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.llm import LLMClient, OllamaError
from configs.loader import load_config
from es_client import get_es_client
from tracing import JsonFormatter, TraceContext, setup_logger


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class AnswerRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = setup_logger(
        name="agentic_rag", formatter=JsonFormatter(), level=logging.INFO
    )
    cfg = load_config()
    app.state.cfg = cfg
    app.state.logger = logger
    app.state.llm = LLMClient.from_config(cfg)
    app.state.es = get_es_client()
    logger.info(
        "Agentic RAG API started (llm=%s@%s)",
        cfg["llm"]["model"],
        cfg["llm"]["base_url"],
    )
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
    """Phase 0: stream a raw LLM response. Retrieval + citations come in Phase 1."""
    trace_id = str(uuid4())
    start = time.perf_counter()
    ctx = TraceContext(trace_id=trace_id, start_time=start)
    llm: LLMClient = request.app.state.llm
    logger = request.app.state.logger

    if not await llm.is_up():
        raise OllamaError("LLM backend is down — start Ollama first.")

    # Phase 0 placeholder prompt; replaced by retrieved context in Phase 1.
    messages = [
        {
            "role": "system",
            "content": "You are a careful research assistant. Answer concisely.",
        },
        {"role": "user", "content": payload.question},
    ]

    async def event_stream():
        first_token_ms = None
        async for delta in llm.stream_chat(messages):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - start) * 1000
            yield _sse({"delta": delta})
        timings = {
            "first_token_ms": round(first_token_ms or 0.0, 1),
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
        }
        yield _sse({"done": True, "trace_id": trace_id, "timings": timings})
        logger.info(
            "answer streamed",
            extra={"trace_id": trace_id, "timings": ctx.timings},
        )

    return StreamingResponse(
        event_stream(), media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.main:app", host="0.0.0.0", port=8001)
