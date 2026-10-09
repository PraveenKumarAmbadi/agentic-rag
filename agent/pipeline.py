"""Shared RAG pipeline — one code path for the API and the eval runner.

`answer_stream()` is the single implementation of:
  hybrid retrieval -> grounded prompt -> streamed generation -> citation
  validation -> final result.

The FastAPI layer maps these events to SSE; the eval runner consumes the
`done` event. Refactoring retrieval or prompting only touches this file, so
eval numbers can never silently drift from what production serves.

Event contract (also kept by AgentPipeline):
  {"type": "delta", "text"}   answer text; deltas concatenate to done.text
  {"type": "done", ...}       exactly one, last: text, refusal, citations,
                              invalid_citations, retrieved_doc_ids, timings,
                              step_timings, usage, degraded, truncated
Exceptions (RetrievalUnavailable, OllamaError, ...) propagate to the caller,
which owns turning them into an error response.
"""

import asyncio
import time
from contextlib import aclosing
from dataclasses import dataclass, field
from uuid import uuid4

from agent.citations import build_citation_map, is_refusal, parse_citations
from agent.llm import CallSpec, StreamEnd
from agent.prompts import build_messages, build_refusal_text
from tracing import CircuitBreaker, TraceContext


class RetrievalUnavailable(RuntimeError):
    """Every retrieval method failed: an outage, not an empty result."""


@dataclass(frozen=True)
class CallSpecs:
    """Per-call LLM policy, derived once from config."""

    answer: CallSpec
    decide: CallSpec
    rescue: CallSpec
    critique: CallSpec

    @classmethod
    def from_config(cls, cfg: dict) -> "CallSpecs":
        llm = cfg["llm"]
        common = {"timeout_s": llm["read_timeout_s"], "max_attempts": llm["max_attempts"]}
        return cls(
            # Streamed; truncation is surfaced as done["truncated"].
            answer=CallSpec(
                "answer", temperature=llm["temperature"],
                max_tokens=llm["max_tokens"], on_length="accept", **common,
            ),
            # Control turns emit one small JSON object; a cut-off object fails
            # to parse and takes the documented parse-failure path.
            decide=CallSpec("decide", temperature=0.0, max_tokens=200, on_length="accept", **common),
            rescue=CallSpec("rescue", temperature=0.0, max_tokens=200, on_length="accept", **common),
            # A truncated verdict would be misread, so give it one doubling.
            critique=CallSpec(
                "critique", temperature=llm["judge_temperature"],
                max_tokens=400, on_length="escalate", **common,
            ),
        )


@dataclass
class Usage:
    llm_calls: int = 0
    llm_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens or 0
        self.completion_tokens += completion_tokens or 0

    def failed(self) -> None:
        self.llm_calls += 1
        self.llm_errors += 1

    def as_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "llm_errors": self.llm_errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class RunState:
    """Mutable state of one answer_stream() call."""

    ctx: TraceContext
    t0: float
    deadline: float | None
    usage: Usage = field(default_factory=Usage)
    degraded: set[str] = field(default_factory=set)
    retrieval_ms: float = 0.0
    truncated: bool = False
    # Request-level latency marks: first token from any generation, and the
    # first text actually sent to the client (later attempts don't reset them).
    first_token_ms: float | None = None
    first_visible_ms: float | None = None

    def mark_token(self) -> None:
        if self.first_token_ms is None:
            self.first_token_ms = self.elapsed_ms()

    def mark_visible(self) -> None:
        if self.first_visible_ms is None:
            self.first_visible_ms = self.elapsed_ms()

    def out_of_time(self) -> bool:
        return self.deadline is not None and time.perf_counter() >= self.deadline

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000


@dataclass(frozen=True)
class Retrieval:
    docs: list
    elapsed_ms: float
    failed_methods: tuple[str, ...]


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
        self.specs = CallSpecs.from_config(cfg)
        res = cfg["resilience"]
        self.breakers = {
            name: CircuitBreaker(
                name,
                failure_threshold=res["breaker_failure_threshold"],
                cooldown_s=res["breaker_cooldown_s"],
            )
            for name in ("bm25", "splade", "dense")
        }

    # ------------------------------------------------------------ retrieval

    def breaker_states(self) -> dict[str, str]:
        return {name: b.state for name, b in self.breakers.items()}

    async def retrieve(
        self,
        query: str,
        ctx: TraceContext,
        *,
        top_k: int | None = None,
        label: str = "retrieve",
    ) -> Retrieval:
        """Hybrid retrieval. Sync GPU/ES work runs in a worker thread.

        Per-method timings land in ctx.timings as "<label>.<step>" so repeated
        searches in one request never overwrite each other.

        Raises:
            RetrievalUnavailable: when every retrieval method failed.
        """
        # Imported here so the light modules (agent loop, API, tests) do not
        # pull torch/faiss in at import time.
        from retrieval.fusion import METHODS, search_hybrid

        sub = TraceContext(trace_id=ctx.trace_id, start_time=time.perf_counter())

        def _run():
            return search_hybrid(
                query_text=query,
                query_id=ctx.trace_id,
                bm25_client=self.es,
                splade_client=self.es,
                faiss_index=self.faiss_index,
                corpus=self.corpus,
                bm25_breaker=self.breakers["bm25"],
                splade_breaker=self.breakers["splade"],
                dense_breaker=self.breakers["dense"],
                dense_encoder=self.dense_encoder,
                splade_encoder=self.splade_encoder,
                bm25_index_name=self.cfg["bm25"]["index_name"],
                splade_index_name=self.cfg["splade"]["index_name"],
                logger=self.logger,
                ctx=sub,
                top_k=top_k or self.cfg["agent"]["top_k"],
                rrf_k=self.cfg["fusion"]["rrf_k"],
                per_method_top_k=self.cfg["agent"]["per_method_top_k"],
            )

        start = time.perf_counter()
        result = await asyncio.to_thread(_run)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ctx.timings[label] = elapsed_ms
        for step, ms in sub.timings.items():
            ctx.timings[f"{label}.{step}"] = ms
        if len(result.failed_methods) == len(METHODS):
            raise RetrievalUnavailable("all retrieval methods failed")
        return Retrieval(result.results, elapsed_ms, result.failed_methods)

    # ---------------------------------------------------------------- events

    def _start_run(self, trace_id: str | None, deadline: float | None) -> RunState:
        t0 = time.perf_counter()
        return RunState(
            ctx=TraceContext(trace_id=trace_id or str(uuid4()), start_time=t0),
            t0=t0,
            deadline=deadline,
        )

    def _done_event(
        self,
        run: RunState,
        text: str,
        refusal: bool,
        docs: list,
        **extra,
    ) -> dict:
        doc_ids = [d.doc_id for d in docs]
        cites = parse_citations(text, len(docs))
        return {
            "type": "done",
            "trace_id": run.ctx.trace_id,
            "text": text,
            "refusal": refusal,
            "citations": build_citation_map(cites["markers"], doc_ids),
            "invalid_citations": cites["invalid"],
            "retrieved_doc_ids": doc_ids,
            "step_timings": dict(run.ctx.timings),
            "timings": {
                "retrieval_ms": round(run.retrieval_ms, 1),
                "first_token_ms": round(run.first_token_ms or 0.0, 1),
                "first_visible_ms": round(run.first_visible_ms or 0.0, 1),
                "total_ms": round(run.elapsed_ms(), 1),
            },
            "usage": run.usage.as_dict(),
            "degraded": sorted(run.degraded),
            "truncated": run.truncated,
            **extra,
        }

    def _note_finish(self, run: RunState, end: StreamEnd | None, step: str) -> None:
        if end is None:
            run.usage.add(None, None)
            return
        run.usage.add(end.prompt_tokens, end.completion_tokens)
        if end.finish_reason == "length":
            run.truncated = True
            self.logger.warning(
                "%s hit max_tokens; answer is truncated", step,
                extra={"trace_id": run.ctx.trace_id},
            )

    # ------------------------------------------------------------- pipeline

    async def answer_stream(
        self, question: str, trace_id: str | None = None, deadline: float | None = None
    ):
        """Yield {'type': 'delta', 'text': ...} events, then one 'done' event.

        Args:
            deadline: perf_counter() timestamp; unused by the single-shot
                baseline (the caller's hard timeout bounds it).
        """
        run = self._start_run(trace_id, deadline)
        retrieval = await self.retrieve(question, run.ctx)
        run.retrieval_ms = retrieval.elapsed_ms
        run.degraded.update(retrieval.failed_methods)
        docs = retrieval.docs

        if not docs:
            text = build_refusal_text("no documents matched the question", docs)
            run.mark_visible()
            yield {"type": "delta", "text": text}
            yield self._done_event(run, text, True, docs)
            return

        messages = build_messages(question, docs, self.cfg["agent"]["context_chars"])
        text = ""
        end = None
        async with aclosing(self.llm.stream(messages, self.specs.answer)) as stream:
            async for ev in stream:
                if isinstance(ev, StreamEnd):
                    end = ev
                    continue
                run.mark_token()
                run.mark_visible()
                text += ev.text
                yield {"type": "delta", "text": ev.text}
        self._note_finish(run, end, "answer")

        yield self._done_event(run, text, is_refusal(text), docs)
