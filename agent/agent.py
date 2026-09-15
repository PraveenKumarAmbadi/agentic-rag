"""Agent pipeline — a hand-rolled ReAct loop with self-critique.

Subclass of RagPipeline: same constructor, same `answer_stream()` event
contract (delta/done), so the SSE API and the eval runner work unchanged.
The `done` event gains an `agent` block describing everything the loop did.

Loop per question (bounded by agent.max_iterations):

1. hybrid_search(question)     — system-side pre-fetch, observation 1
2. decide turn (LLM, temp 0)   — ONE JSON object choosing the next tool:
      answer              -> proceed to generation
      hybrid_search(query)-> new observation (merged with previous), loop
      insufficient_evidence(reason) -> RESCUE before accepting:
          a second-opinion call picks the most relevant retrieved doc; if
          one exists the refusal is OVERRIDDEN and generation is hinted at
          that doc, otherwise the refusal stands (honest, cites what was
          found). This is the direct counter to the baseline's measured
          failure mode: 69% false refusals with the gold doc in context.
3. generate (streamed)         — a leading INSUFFICIENT_EVIDENCE is
                                 intercepted, not streamed, and rescued too
4. critique (LLM judge)        — unsupported claims -> regenerate with
                                 feedback (up to agent.max_retries); if still
                                 unsupported, hedge the answer.

Every LLM/tool step lands in the trace timings (decide_1, generate_2, ...)
and in done["agent"], so agent behaviour is as measurable as retrieval was.
"""

import asyncio
import time
from uuid import uuid4

from agent.citations import build_citation_map, parse_citations
from agent.pipeline import RagPipeline
from agent.prompts import (
    HEDGE_PREFIX,
    build_answer_messages,
    build_critique_messages,
    build_decide_messages,
    build_refusal_text,
    build_rescue_messages,
)
from agent.tools import ToolCallError, extract_json, parse_action
from tracing import TraceContext, track_step

REFUSAL_MARKER = "INSUFFICIENT_EVIDENCE"
_MARKER_LEN = len(REFUSAL_MARKER)


class _Draft:
    """Mutable state filled while a generation turn streams."""

    __slots__ = ("text", "refusal", "first_token_ms")

    def __init__(self) -> None:
        self.text = ""
        self.refusal = False
        self.first_token_ms: float | None = None


class AgentPipeline(RagPipeline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reranker = None  # lazy: only loaded when agent.rerank is true

    # ------------------------------------------------------------------ tools

    async def _search_tool(
        self, query: str, ctx: TraceContext, call_no: int
    ) -> list:
        """Execute the hybrid_search tool (optionally reranked)."""
        with track_step(ctx, f"retrieve_{call_no}"):
            results, _ = await self.retrieve(query, ctx.trace_id, ctx)
        if self.cfg["agent"].get("rerank"):
            with track_step(ctx, f"rerank_{call_no}"):
                reranker = self._get_reranker()
                top_k = self.cfg["agent"]["top_k"]
                results = await asyncio.to_thread(
                    reranker.rerank, query, results, top_k
                )
        return results

    def _get_reranker(self):
        if self._reranker is None:
            from reranking.cross_encoder import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(
                model_name=self.cfg["reranker"]["model_name"]
            )
        return self._reranker

    @staticmethod
    def _merge_docs(current: list, new: list, top_k: int) -> list:
        """Union of two result lists, reformulated-search results first."""
        seen: set[str] = set()
        merged = []
        for doc in list(new) + list(current):
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                merged.append(doc)
        return merged[:top_k]

    # ----------------------------------------------------------- LLM helpers

    async def _decide(self, messages: list[dict]) -> str:
        return await self.llm.chat(messages, temperature=0.0, max_tokens=200)

    async def _rescue(
        self, question: str, docs: list, reason: str, ctx: TraceContext
    ) -> dict:
        """Second opinion on a proposed refusal.

        Returns {"overridden", "relevant_doc", "justification"}: when a
        relevant doc is found the caller retries generation with a hint at
        that doc instead of refusing.
        """
        with track_step(ctx, "rescue"):
            raw = await self.llm.chat(
                build_rescue_messages(question, docs, reason),
                temperature=0.0,
                max_tokens=200,
            )
        obj = extract_json(raw) or {}
        doc_no = obj.get("relevant_doc")
        if isinstance(doc_no, float) and doc_no.is_integer():
            doc_no = int(doc_no)
        if isinstance(doc_no, str) and doc_no.strip().isdigit():
            doc_no = int(doc_no.strip())
        valid = isinstance(doc_no, int) and 1 <= doc_no <= len(docs)
        return {
            "overridden": valid,
            "relevant_doc": doc_no if valid else None,
            "justification": str(obj.get("justification", ""))[:300],
        }

    async def _critique(
        self, question: str, docs: list, answer: str, ctx: TraceContext, call_no: int
    ) -> dict:
        """Fact-check the draft against the documents it cited."""
        with track_step(ctx, f"critique_{call_no}"):
            raw = await self.llm.chat(
                build_critique_messages(question, docs, answer),
                temperature=self.cfg["llm"].get("judge_temperature", 0.0),
                max_tokens=400,
            )
        obj = extract_json(raw) or {}
        claims = obj.get("unsupported_claims") or []
        if isinstance(claims, str):
            claims = [claims]
        return {
            # Parse failure defaults to "supported": a judge that cannot be
            # parsed must not block the answer.
            "supported": bool(obj.get("supported", True)),
            "unsupported_claims": [
                str(c) for c in claims if str(c).strip()
            ][:10],
        }

    async def _generate(
        self, messages: list[dict], draft: _Draft, t0: float
    ):
        """Stream one generation turn.

        Yields delta events ONLY when the draft is not a refusal: a leading
        INSUFFICIENT_EVIDENCE is intercepted (buffered, never emitted) so a
        false refusal can be rescued without the user ever seeing it.
        """
        pending = ""
        resolved = False  # is the leading-refusal question decided?
        async for delta in self.llm.stream_chat(messages):
            if draft.first_token_ms is None:
                draft.first_token_ms = (time.perf_counter() - t0) * 1000
            draft.text += delta
            if resolved:
                if not draft.refusal:
                    yield {"type": "delta", "text": delta}
                continue
            pending += delta
            stripped = pending.lstrip()
            if stripped.startswith(REFUSAL_MARKER):
                resolved = True
                draft.refusal = True
                pending = ""
            elif len(stripped) >= _MARKER_LEN or not REFUSAL_MARKER.startswith(
                stripped
            ):
                resolved = True
                yield {"type": "delta", "text": pending}
                pending = ""
        if not resolved and pending:
            # Stream ended while the marker was still ambiguous: it never
            # completed, so this is not a refusal — flush what we buffered.
            yield {"type": "delta", "text": pending}

    def _done_event(
        self,
        text: str,
        refusal: bool,
        docs: list,
        ctx: TraceContext,
        t0: float,
        ag: dict,
        first_token_ms: float | None,
        retrieval_ms: float,
    ) -> dict:
        doc_ids = [d.doc_id for d in docs]
        cites = parse_citations(text, len(docs))
        return {
            "type": "done",
            "trace_id": ctx.trace_id,
            "text": text,
            "refusal": refusal,
            "citations": build_citation_map(cites["markers"], doc_ids),
            "invalid_citations": cites["invalid"],
            "retrieved_doc_ids": doc_ids,
            "step_timings": dict(ctx.timings),
            "timings": {
                "retrieval_ms": round(retrieval_ms, 1),
                "first_token_ms": round(first_token_ms or 0.0, 1),
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
            "agent": ag,
        }

    # -------------------------------------------------------------- main loop

    async def answer_stream(self, question: str, trace_id: str | None = None):
        trace_id = trace_id or str(uuid4())
        t0 = time.perf_counter()
        ctx = TraceContext(trace_id=trace_id, start_time=t0)
        agent_cfg = self.cfg["agent"]
        max_iterations = int(agent_cfg.get("max_iterations", 4))
        max_retries = int(agent_cfg.get("max_retries", 2))
        top_k = int(agent_cfg.get("top_k", 10))
        context_chars = int(agent_cfg.get("context_chars", 800))
        ag: dict = {
            "iterations": 0,
            "actions": [],
            "decide_parse_errors": 0,
            "rescue": None,
            "critique": None,
            "hedged": False,
            "llm_calls": 0,
        }

        # -- observation 1: system-side pre-fetch with the raw question ------
        docs = await self._search_tool(question, ctx, 1)
        retrieval_ms = ctx.timings.get("retrieve_1", 0.0)
        yield {"type": "step", "step": "retrieve", "n_docs": len(docs)}

        # -- ReAct loop: decide -> act -> observe ----------------------------
        hint_doc: int | None = None
        refusal: tuple[str, list] | None = None
        for iteration in range(1, max_iterations + 1):
            ag["iterations"] = iteration
            history = [
                f'{a["tool"]}({", ".join(f"{k}={v!r}" for k, v in a["args"].items())})'
                for a in ag["actions"]
            ]
            with track_step(ctx, f"decide_{iteration}"):
                raw = await self._decide(
                    build_decide_messages(
                        question, docs, history, iteration, max_iterations
                    )
                )
            ag["llm_calls"] += 1
            try:
                action = parse_action(raw)
            except ToolCallError as exc:
                ag["decide_parse_errors"] += 1
                self.logger.warning(
                    "invalid tool call at iteration %d: %s",
                    iteration,
                    exc,
                    extra={"trace_id": trace_id},
                )
                action = None
            if action is None:
                # Unparseable decide turn: fall through to generation — the
                # safest default given the measured over-refusal failure mode.
                break
            ag["actions"].append(
                {"iteration": iteration, "tool": action.tool, "args": action.args}
            )
            yield {"type": "step", "step": "decide", "tool": action.tool}

            if action.tool == "answer":
                break
            if action.tool == "hybrid_search":
                new_docs = await self._search_tool(
                    action.args["query"], ctx, iteration + 1
                )
                docs = self._merge_docs(docs, new_docs, top_k)
                retrieval_ms += ctx.timings.get(f"retrieve_{iteration + 1}", 0.0)
                yield {"type": "step", "step": "retrieve", "n_docs": len(docs)}
                continue
            # action.tool == "insufficient_evidence": challenge it.
            reason = action.args["reason"]
            rescue = await self._rescue(question, docs, reason, ctx)
            ag["llm_calls"] += 1
            ag["rescue"] = {"trigger": "decide", "reason": reason, **rescue}
            if rescue["overridden"]:
                hint_doc = rescue["relevant_doc"]
                yield {
                    "type": "step",
                    "step": "rescue_override",
                    "doc": hint_doc,
                }
                break
            refusal = (reason, docs)
            break

        if refusal is not None:
            reason, ref_docs = refusal
            text = build_refusal_text(reason, ref_docs)
            yield self._done_event(
                text, True, ref_docs, ctx, t0, ag, None, retrieval_ms
            )
            return
        # Loop exhausted with endless reformulation: generate from what we
        # have — retrieval hit something in almost every baseline miss, and
        # the critique step still guards the output.

        # -- generation + critique -------------------------------------------
        attempt = 0
        feedback: list[str] | None = None
        while True:
            attempt += 1
            messages = build_answer_messages(
                question,
                docs,
                hint_doc=hint_doc,
                feedback=feedback,
                context_chars=context_chars,
            )
            draft = _Draft()
            with track_step(ctx, f"generate_{attempt}"):
                async for ev in self._generate(messages, draft, t0):
                    yield ev
            ag["llm_calls"] += 1

            if draft.refusal:
                reason = draft.text.split(":", 1)[-1].strip() or (
                    "the retrieved documents did not answer the question"
                )
                if ag["rescue"] is None:
                    rescue = await self._rescue(question, docs, reason, ctx)
                    ag["llm_calls"] += 1
                    ag["rescue"] = {
                        "trigger": "generate",
                        "reason": reason,
                        **rescue,
                    }
                    if rescue["overridden"]:
                        hint_doc = rescue["relevant_doc"]
                        feedback = None
                        yield {
                            "type": "step",
                            "step": "rescue_override",
                            "doc": hint_doc,
                        }
                        continue
                yield self._done_event(
                    draft.text, True, docs, ctx, t0, ag,
                    draft.first_token_ms, retrieval_ms,
                )
                return

            verdict = await self._critique(
                question, docs, draft.text, ctx, attempt
            )
            ag["llm_calls"] += 1
            ag["critique"] = verdict
            if verdict["supported"]:
                yield self._done_event(
                    draft.text, False, docs, ctx, t0, ag,
                    draft.first_token_ms, retrieval_ms,
                )
                return
            if attempt <= max_retries:
                feedback = verdict["unsupported_claims"]
                yield {
                    "type": "step",
                    "step": "correction",
                    "unsupported_claims": len(feedback),
                }
                continue
            # Retries exhausted: hedge rather than refuse — an answer with a
            # visible warning beats a false refusal.
            ag["hedged"] = True
            yield self._done_event(
                HEDGE_PREFIX + draft.text, False, docs, ctx, t0, ag,
                draft.first_token_ms, retrieval_ms,
            )
            return
