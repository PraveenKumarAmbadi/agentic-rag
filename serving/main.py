"""FastAPI app — SSE adapter over the shared pipeline.

All RAG logic lives in agent/; this module owns only HTTP concerns:
authentication, input validation, admission control, timeouts, and turning
pipeline events (or failures) into SSE.

SSE wire format (one JSON object per `data:` line):
  {"delta": "..."}                              answer text
  {"type": "step"|"reset", ...}                 agent progress / discard text so far
  {"type": "done", ...}                         final metadata (text omitted: it is the deltas)
  {"type": "error", "code", "message", "trace_id"}  terminal failure

Run:  python -m serving.main      (or: uvicorn serving.main:app --port 8001)
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from agent.llm import LLMClient, OllamaError
from agent.pipeline import RagPipeline, RetrievalUnavailable
from agent.stack import build_pipeline
from configs.loader import load_config
from serving.auth import API_KEY_HEADER, ApiKeyAuth
from serving.limits import CachedProbe, ConcurrencyLimiter, Saturated, SlotLease
from tracing import configure_logging, trace_id_var

logger = logging.getLogger("agentic_rag.api")

MAX_BODY_BYTES = 64 * 1024
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_QUEUE_SIZE = 64

# Client-facing messages: stable and free of internals (details go to logs).
PUBLIC_ERRORS = {
    "llm_unavailable": "The language model backend is unavailable. Retry shortly.",
    "retrieval_unavailable": "Document retrieval is unavailable. Retry shortly.",
    "timeout": "The request exceeded its time limit.",
    "internal": "Internal error. Quote the trace_id when reporting it.",
}


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Absolute parse-time cap; the configured serving.max_question_chars
    # (<= this) is enforced in the handler.
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)
    ]

    @field_validator("question")
    @classmethod
    def _no_control_characters(cls, value: str) -> str:
        if _CONTROL_CHARS_RE.search(value):
            raise ValueError("question must not contain control characters")
        return value


@dataclass
class Services:
    cfg: dict
    pipeline: RagPipeline
    llm: LLMClient
    auth: ApiKeyAuth | None
    limiter: ConcurrencyLimiter
    llm_probe: CachedProbe

    async def aclose(self) -> None:
        await self.llm.close()
        es = getattr(self.pipeline, "es", None)
        if es is not None:
            with suppress(Exception):
                await asyncio.to_thread(es.close)


async def build_services() -> Services:
    """Production wiring: config, auth, models, indexes, LLM client."""
    configure_logging()
    cfg = load_config()
    serving = cfg["serving"]
    # Validate auth before loading models: misconfiguration fails in ms.
    auth = ApiKeyAuth.from_env(serving["auth_enabled"])
    llm = LLMClient.from_config(cfg)
    try:
        pipeline = await build_pipeline(cfg, llm=llm, logger=logging.getLogger("agentic_rag"))
    except BaseException:
        await llm.close()
        raise
    services = Services(
        cfg=cfg,
        pipeline=pipeline,
        llm=llm,
        auth=auth,
        limiter=ConcurrencyLimiter(
            serving["max_concurrent"], serving["max_queue"], serving["queue_timeout_s"]
        ),
        llm_probe=CachedProbe(llm.is_up, serving["llm_health_ttl_s"]),
    )
    if not await llm.is_up():
        logger.warning("LLM backend unreachable at startup; /answer returns 503 until it is up")
    logger.info(
        "Agentic RAG API ready",
        extra={
            "llm_model": cfg["llm"]["model"],
            "pipeline": cfg["agent"]["mode"],
            "auth_enabled": auth is not None,
            "max_concurrent": serving["max_concurrent"],
        },
    )
    return services


# ---------------------------------------------------------------- plumbing


def _services(request: Request) -> Services:
    return request.app.state.services


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


class _LeasedStreamingResponse(StreamingResponse):
    """Releases its concurrency slot however the response ends.

    Releasing in the body generator alone would leak the slot if the
    generator never starts (e.g. the client disconnects first).
    """

    def __init__(self, lease: SlotLease, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lease = lease

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._lease.release()


async def _request_context(request: Request, call_next):
    """Trace id from a safe X-Request-ID (or a new one) + body size cap."""
    incoming = request.headers.get("x-request-id", "")
    trace_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else str(uuid4())
    request.state.trace_id = trace_id
    token = trace_id_var.set(trace_id)
    try:
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_BODY_BYTES:
            response = JSONResponse({"detail": "request body too large"}, status_code=413)
        else:
            response = await call_next(request)
    finally:
        trace_id_var.reset(token)
    response.headers["X-Request-ID"] = trace_id
    return response


async def _ollama_error_handler(request: Request, exc: OllamaError) -> JSONResponse:
    logger.warning("LLM backend error: %s", exc)
    return JSONResponse(
        {"detail": PUBLIC_ERRORS["llm_unavailable"]},
        status_code=503,
        headers={"Retry-After": "10"},
    )


async def require_api_key(
    request: Request,
    api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
) -> None:
    auth = _services(request).auth
    if auth is None:
        return
    if not auth.verify(api_key):
        logger.warning(
            "rejected request: missing or invalid API key",
            extra={
                "client": request.client.host if request.client else None,
                "path": request.url.path,
            },
        )
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


def _error_code(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, OllamaError):
        return "llm_unavailable"
    if isinstance(exc, RetrievalUnavailable):
        return "retrieval_unavailable"
    return "internal"


async def _event_stream(services: Services, question: str, trace_id: str):
    """Run the pipeline in a producer task and relay its events as SSE.

    One producer task owns the pipeline from start to finish, so a hard
    timeout or a client disconnect cancels it cleanly in the task where its
    HTTP streams and context managers live.
    """
    trace_id_var.set(trace_id)
    start = time.perf_counter()
    deadline = start + services.cfg["agent"]["request_timeout_s"]
    hard_deadline = start + services.cfg["serving"]["hard_timeout_s"]
    queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_SIZE)

    async def produce() -> None:
        try:
            events = services.pipeline.answer_stream(question, trace_id=trace_id, deadline=deadline)
            async with aclosing(events):
                async for ev in events:
                    await queue.put(("event", ev))
            await queue.put(("end", None))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(("error", exc))

    producer = asyncio.create_task(produce())
    try:
        while True:
            remaining = hard_deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("hard request timeout")
            async with asyncio.timeout(remaining):
                kind, item = await queue.get()
            if kind == "end":
                return
            if kind == "error":
                raise item
            if item["type"] == "delta":
                yield _sse({"delta": item["text"]})
                continue
            yield _sse({k: v for k, v in item.items() if k != "text"})
            if item["type"] == "done":
                logger.info(
                    "answer streamed",
                    extra={
                        "timings": item["timings"],
                        "usage": item["usage"],
                        "refusal": item["refusal"],
                        "degraded": item["degraded"],
                    },
                )
    except asyncio.CancelledError:
        logger.info("client disconnected; generation cancelled")
        raise
    except Exception as exc:
        code = _error_code(exc)
        if code == "llm_unavailable":
            services.llm_probe.mark_down()
        if code == "internal":
            logger.error("answer failed", exc_info=exc)
        else:
            logger.warning("answer failed: %s (%s)", code, exc)
        yield _sse({
            "type": "error",
            "trace_id": trace_id,
            "code": code,
            "message": PUBLIC_ERRORS[code],
        })
    finally:
        if not producer.done():
            producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer


# ------------------------------------------------------------------ routes

router = APIRouter()


async def _es_up(es) -> bool:
    try:
        return bool(await asyncio.to_thread(es.options(request_timeout=2).ping))
    except Exception:
        return False


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Readiness: 200 when answering is possible (maybe degraded), else 503."""
    services = _services(request)
    es_ok, llm_ok = await asyncio.gather(_es_up(services.pipeline.es), services.llm.is_up())
    breakers = services.pipeline.breaker_states()
    if not llm_ok or all(state == "open" for state in breakers.values()):
        status = "down"
    elif not es_ok or any(state != "closed" for state in breakers.values()):
        status = "degraded"
    else:
        status = "ok"
    body = {
        "status": status,
        "elasticsearch": "up" if es_ok else "down",
        "llm": "up" if llm_ok else "down",
        "retrievers": breakers,
        "load": services.limiter.stats,
    }
    return JSONResponse(body, status_code=503 if status == "down" else 200)


@router.post("/answer", dependencies=[Depends(require_api_key)])
async def answer(request: Request, payload: AnswerRequest) -> StreamingResponse:
    services = _services(request)
    max_chars = services.cfg["serving"]["max_question_chars"]
    if len(payload.question) > max_chars:
        raise HTTPException(
            status_code=422,
            detail=[{
                "loc": ["body", "question"],
                "msg": f"question must be at most {max_chars} characters",
                "type": "string_too_long",
            }],
        )
    if not await services.llm_probe():
        raise HTTPException(
            status_code=503,
            detail=PUBLIC_ERRORS["llm_unavailable"],
            headers={"Retry-After": "10"},
        )
    try:
        await services.limiter.acquire()
    except Saturated as exc:
        logger.warning("request rejected: %s", exc, extra={"load": services.limiter.stats})
        raise HTTPException(
            status_code=503,
            detail="Server is at capacity. Retry shortly.",
            headers={"Retry-After": "5"},
        ) from None
    lease = SlotLease(services.limiter)
    try:
        return _LeasedStreamingResponse(
            lease,
            _event_stream(services, payload.question, request.state.trace_id),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    except BaseException:
        lease.release()
        raise


# --------------------------------------------------------------------- app


def create_app(
    services_factory: Callable[[], Awaitable[Services]] = build_services,
) -> FastAPI:
    """App factory; tests inject a services_factory with fakes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = await services_factory()
        app.state.services = services
        try:
            yield
        finally:
            await services.aclose()

    app = FastAPI(lifespan=lifespan, title="Agentic RAG")
    app.middleware("http")(_request_context)
    app.add_exception_handler(OllamaError, _ollama_error_handler)
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    configure_logging()
    serving_cfg = load_config()["serving"]
    # One worker: every worker would load its own copy of the models.
    uvicorn.run(
        "serving.main:app",
        host=serving_cfg["host"],
        port=serving_cfg["port"],
        workers=1,
        log_config=None,
    )
