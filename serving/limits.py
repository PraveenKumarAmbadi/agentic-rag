"""Admission control and cached dependency probes for the API."""

import asyncio
import time
from collections.abc import Awaitable, Callable


class Saturated(Exception):
    """No generation slot is available within the admission policy."""


class ConcurrencyLimiter:
    """At most `max_concurrent` requests generate at once.

    Up to `max_queue` more may wait, each for at most `queue_timeout_s`;
    anything beyond that is rejected immediately. Rejecting at the door keeps
    latency bounded instead of letting requests pile up inside Ollama, where
    they cannot be observed or timed out.
    """

    def __init__(self, max_concurrent: int, max_queue: int, queue_timeout_s: float) -> None:
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self._sem = asyncio.Semaphore(max_concurrent)
        self._in_flight = 0
        self._waiting = 0

    @property
    def stats(self) -> dict:
        return {
            "in_flight": self._in_flight,
            "waiting": self._waiting,
            "max_concurrent": self.max_concurrent,
        }

    async def acquire(self) -> None:
        if self._sem.locked() and self._waiting >= self.max_queue:
            raise Saturated("queue full")
        self._waiting += 1
        try:
            async with asyncio.timeout(self.queue_timeout_s):
                await self._sem.acquire()
        except TimeoutError:
            raise Saturated("timed out waiting for a slot") from None
        finally:
            self._waiting -= 1
        self._in_flight += 1

    def release(self) -> None:
        self._in_flight -= 1
        self._sem.release()


class SlotLease:
    """Idempotent release handle for one acquired slot."""

    def __init__(self, limiter: ConcurrencyLimiter) -> None:
        self._limiter = limiter
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._limiter.release()


class CachedProbe:
    """Caches a boolean health probe for `ttl_s` seconds.

    Lets every request check "is the LLM up?" while costing at most one
    real probe per TTL. ``mark_down()`` invalidates it as soon as a real call
    fails, so the next request re-probes instead of trusting a stale "up".
    """

    def __init__(self, probe: Callable[[], Awaitable[bool]], ttl_s: float) -> None:
        self._probe = probe
        self._ttl_s = ttl_s
        self._value = False
        self._checked_at = float("-inf")
        self._lock = asyncio.Lock()

    async def __call__(self) -> bool:
        if time.monotonic() - self._checked_at < self._ttl_s:
            return self._value
        async with self._lock:
            if time.monotonic() - self._checked_at >= self._ttl_s:
                self._value = await self._probe()
                self._checked_at = time.monotonic()
        return self._value

    def mark_down(self) -> None:
        self._value = False
        self._checked_at = float("-inf")
