"""Tracing, structured logging, and circuit breaking.

- ``TraceContext`` / ``track_step``: per-request step timings (ms).
- ``trace_id_var``: the current request's trace id. It is a ContextVar, so it
  follows the request into ``asyncio.to_thread`` workers and every log line
  emitted while serving that request carries it without manual plumbing.
- ``JsonFormatter`` / ``configure_logging``: one JSON line per record for
  the whole process (app, libraries, uvicorn), including tracebacks.
- ``CircuitBreaker``: thread-safe breaker used per retrieval method.
"""

import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


@dataclass
class TraceContext:
    trace_id: str
    start_time: float
    timings: dict[str, float] = field(default_factory=dict)


@contextmanager
def track_step(ctx: TraceContext, step_name: str):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        ctx.timings[step_name] = (time.perf_counter() - start_time) * 1000


# Attributes every LogRecord has; anything else on a record came from `extra=`.
# color_message is uvicorn's ANSI-coloured duplicate of the message.
_RECORD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None))
) | {"message", "asctime", "color_message"}


class JsonFormatter(logging.Formatter):
    """One JSON object per record: core fields, trace id, extras, traceback."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = getattr(record, "trace_id", None) or trace_id_var.get()
        if trace_id:
            entry["trace_id"] = trace_id
        for key, value in record.__dict__.items():
            if key not in _RECORD_ATTRS and key not in entry and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            entry["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


_NOISY_LOGGERS = ("elastic_transport", "elasticsearch", "urllib3", "httpx", "httpcore")
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def configure_logging(level: int = logging.INFO, json_format: bool = True) -> None:
    """Route every logger in the process through one root handler.

    Idempotent. Library chatter is capped at WARNING; uvicorn's own handlers
    are removed so its lines are JSON too.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if json_format
        else logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in _UVICORN_LOGGERS:
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.propagate = True


class CircuitOpenError(RuntimeError):
    """The protected call was skipped because its circuit is open."""


class CircuitBreaker:
    """Closed -> (N consecutive failures) -> open -> (cooldown) -> half-open.

    In half-open exactly one trial call is let through; success closes the
    circuit, failure re-opens it for another cooldown. Thread-safe because
    retrieval runs in worker threads.

    Usage::

        with breaker:
            call_dependency()
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._trial_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN and self._clock() - self._opened_at >= self.cooldown_s:
                return self.HALF_OPEN
            return self._state

    def __enter__(self) -> "CircuitBreaker":
        with self._lock:
            if self._state == self.OPEN:
                remaining = self.cooldown_s - (self._clock() - self._opened_at)
                if remaining > 0:
                    raise CircuitOpenError(
                        f"{self.name} circuit open; next trial in {remaining:.0f}s"
                    )
                self._state = self.HALF_OPEN
            if self._state == self.HALF_OPEN:
                if self._trial_in_flight:
                    raise CircuitOpenError(f"{self.name} circuit half-open; trial in progress")
                self._trial_in_flight = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        with self._lock:
            self._trial_in_flight = False
            if exc_type is None:
                self._state = self.CLOSED
                self._failures = 0
            else:
                self._failures += 1
                if self._state == self.HALF_OPEN or self._failures >= self.failure_threshold:
                    self._state = self.OPEN
                    self._opened_at = self._clock()
        return False
