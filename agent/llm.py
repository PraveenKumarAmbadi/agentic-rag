"""LLM client for Ollama.

Streaming chat over Ollama's NDJSON `/api/chat` endpoint. Kept dependency-light
(httpx only) and hand-rolled so every wire detail is visible and testable.

Two entry points with deliberately different failure semantics:

- ``stream()``   user-facing generation. Retries only before the first token;
                 a failure after that raises ``StreamInterrupted`` because
                 replaying would duplicate text the client already has.
- ``complete()`` internal turns (decide / rescue / critique). Any failed
                 attempt is discarded and retried; ``CallSpec.on_length``
                 decides what a max_tokens cut-off means.

Every call is described by a ``CallSpec`` so per-call policy (temperature,
token ceiling, timeout, retries) lives in one place — see
``agent.pipeline.CallSpecs``.
"""

import asyncio
import dataclasses
import json
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

# Warn when a prompt fills this share of num_ctx: Ollama truncates the
# overflow from the front of the prompt (system prompt first) without error.
CONTEXT_WARN_RATIO = 0.95


@dataclass(frozen=True, slots=True)
class Token:
    text: str


@dataclass(frozen=True, slots=True)
class StreamEnd:
    finish_reason: str                 # Ollama done_reason: "stop" | "length" | ...
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    attempts: int


@dataclass(frozen=True, slots=True)
class CallSpec:
    """Everything that differs between call types, in one place."""
    name: str
    temperature: float = 0.1
    max_tokens: int = 1024            # runaway ceiling, not the length control
    json_schema: dict | None = None   # constrains the output shape
    timeout_s: float = 60.0           # max silent gap, includes prefill
    think: bool | None = None         # None = don't send the field
    on_length: Literal["fail", "escalate", "accept"] = "fail"
    max_attempts: int = 3


class OllamaError(RuntimeError):
    def __init__(self, msg: str, retryable: bool = True) -> None:
        super().__init__(msg)
        self.retryable = retryable


class StreamInterrupted(OllamaError):
    """The stream failed after tokens were already delivered."""


class TruncatedOutput(OllamaError):
    """An internal turn hit max_tokens and its spec does not accept that."""


def _backoff(attempt: int) -> float:
    return min(8.0, 0.5 * 2 ** attempt) * (0.5 + random.random() / 2)


class LLMClient:
    """Async streaming chat client. One instance per process."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        num_ctx: int = 4096,
        keep_alive_s: float = 360.0,
        connect_timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.keep_alive_s = keep_alive_s
        self.connect_timeout_s = connect_timeout_s
        self._client = httpx.AsyncClient(transport=transport)

    @classmethod
    def from_config(cls, cfg: dict, **kwargs) -> "LLMClient":
        llm = cfg["llm"]
        return cls(
            base_url=llm["base_url"],
            model=llm["model"],
            num_ctx=llm["num_ctx"],
            keep_alive_s=llm["keep_alive_s"],
            connect_timeout_s=llm["connect_timeout_s"],
            **kwargs,
        )

    async def is_up(self, timeout_s: float = 2.0) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags", timeout=timeout_s)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _payload(self, messages: list[dict], spec: CallSpec) -> dict:
        p = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive_s,
            "options": {
                "temperature": spec.temperature,
                "num_predict": spec.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if spec.think is not None:
            p["think"] = spec.think
        if spec.json_schema is not None:
            p["format"] = spec.json_schema
        return p

    def _check_context(self, spec: CallSpec, end: StreamEnd) -> None:
        if end.prompt_tokens and end.prompt_tokens >= self.num_ctx * CONTEXT_WARN_RATIO:
            logger.warning(
                "%s prompt used %d of num_ctx=%d tokens; Ollama truncates the "
                "front of longer prompts — raise llm.num_ctx or lower "
                "agent.context_chars / agent.top_k",
                spec.name, end.prompt_tokens, self.num_ctx,
                extra={"llm_call": spec.name, "prompt_tokens": end.prompt_tokens},
            )

    async def _attempt(self, messages: list[dict], spec: CallSpec) -> AsyncIterator[Token | StreamEnd]:
        """ONE attempt — the only place that talks HTTP."""
        timeout = httpx.Timeout(
            connect=self.connect_timeout_s, read=spec.timeout_s, write=10.0, pool=5.0
        )
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/api/chat",
                json=self._payload(messages, spec), timeout=timeout,
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:200]
                    raise OllamaError(
                        f"Ollama returned {resp.status_code}: {body}",
                        retryable=resp.status_code >= 500 or resp.status_code == 429,
                    )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise OllamaError(f"Ollama error: {event['error']}")
                    delta = event.get("message", {}).get("content", "")
                    if delta:
                        yield Token(delta)
                    if event.get("done"):
                        end = StreamEnd(
                            event.get("done_reason", "stop"),
                            event.get("prompt_eval_count"),
                            event.get("eval_count"),
                        )
                        self._check_context(spec, end)
                        yield end
                        return
            raise OllamaError("stream closed before 'done'")
        except (httpx.TransportError, json.JSONDecodeError) as exc:
            raise OllamaError(f"{type(exc).__name__}: {exc}") from exc

    async def stream(self, messages: list[dict], spec: CallSpec) -> AsyncIterator[Token | StreamEnd]:
        """User-facing: retry only before the first token."""
        started = False
        for attempt in range(spec.max_attempts):
            try:
                async for ev in self._attempt(messages, spec):
                    started = started or isinstance(ev, Token)
                    yield ev
                return
            except OllamaError as exc:
                if started:
                    raise StreamInterrupted(str(exc)) from exc
                if not exc.retryable or attempt == spec.max_attempts - 1:
                    raise
                logger.warning(
                    "%s attempt %d/%d failed, retrying: %s",
                    spec.name, attempt + 1, spec.max_attempts, exc,
                )
                await asyncio.sleep(_backoff(attempt))

    async def complete(self, messages: list[dict], spec: CallSpec) -> Completion:
        """Internal turns: any failed attempt is discarded and retried."""
        max_tokens, last = spec.max_tokens, None
        for attempt in range(spec.max_attempts):
            try:
                parts, end = [], None
                async for ev in self._attempt(
                    messages, dataclasses.replace(spec, max_tokens=max_tokens)
                ):
                    if isinstance(ev, Token):
                        parts.append(ev.text)
                    else:
                        end = ev
            except OllamaError as exc:
                if not exc.retryable:
                    raise
                last = exc
                logger.warning(
                    "%s attempt %d/%d failed: %s",
                    spec.name, attempt + 1, spec.max_attempts, exc,
                )
                if attempt < spec.max_attempts - 1:
                    await asyncio.sleep(_backoff(attempt))
                continue
            if end.finish_reason != "length" or spec.on_length == "accept":
                return Completion(
                    "".join(parts), end.finish_reason,
                    end.prompt_tokens, end.completion_tokens, attempt + 1,
                )
            if spec.on_length == "escalate" and max_tokens == spec.max_tokens:
                max_tokens *= 2          # one escalation, then fail
                continue
            raise TruncatedOutput(
                f"{spec.name} hit max_tokens={max_tokens}", retryable=False
            )
        if last is None:  # attempts ran out while escalating max_tokens
            raise TruncatedOutput(
                f"{spec.name} hit max_tokens={max_tokens}", retryable=False
            )
        raise OllamaError(
            f"{spec.name} failed after {spec.max_attempts} attempts: {last}"
        ) from last

    async def close(self) -> None:
        await self._client.aclose()
