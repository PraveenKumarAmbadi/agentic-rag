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
3. generate (streamed)         — a leading refusal marker is intercepted,
                                 not streamed, and rescued too
4. critique (LLM judge)        — unsupported claims -> regenerate with
                                 feedback (up to agent.max_retries); if still
                                 unsupported, hedge the answer.

Degrade policy — internal turns never take the request down:
  decide fails / unparseable   -> fall through to generation
  rescue fails                 -> the refusal stands
  critique fails / no budget   -> answer delivered, critique marked skipped
  critique unparseable         -> treated as supported (documented policy)
Only generation itself, or a total retrieval outage, ends in an error.

Every LLM/tool step lands in the trace timings (decide_1, generate_2, ...)
and in done["agent"], so agent behaviour is as measurable as retrieval was.
"""

import asyncio
import re
from contextlib import aclosing
from dataclasses import dataclass

from agent.citations import classify_refusal_prefix, is_refusal, refusal_reason
from agent.llm import CallSpec, OllamaError, StreamEnd, StreamInterrupted
from agent.pipeline import RagPipeline, RetrievalUnavailable, RunState
from agent.prompts import (
    HEDGE_PREFIX,
    build_answer_messages,
    build_critique_messages,
    build_decide_messages,
    build_refusal_text,
    build_rescue_messages,
)
from agent.tools import Action, ToolCallError, extract_json, parse_action
from tracing import track_step

DEFAULT_REFUSAL_REASON = "the retrieved documents did not answer the question"


class RefusalGate:
    """Holds back the start of a stream until we know it is not a refusal."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._pending = ""
        self.resolved = False
        self.refusal = False

    def feed(self, delta: str) -> str:
        """Return the text that is safe to show to the user now ('' if none)."""
        if self.resolved:
            return "" if self.refusal else delta
        self._pending += delta
        state = classify_refusal_prefix(self._pending)
        if state == "undecided":
            return ""
        return self._resolve(state == "refusal")

    def flush(self) -> str:
        """Stream ended: resolve whatever is still buffered."""
        if self.resolved:
            return ""
        return self._resolve(is_refusal(self._pending))

    def _resolve(self, refusal: bool) -> str:
        self.resolved = True
        self.refusal = refusal
        out, self._pending = ("" if refusal else self._pending), ""
        return out


@dataclass
class _Draft:
    text: str = ""
    refusal: bool = False
    restarts: int = 0

    def restart(self) -> None:
        self.text = ""
        self.refusal = False
        self.restarts += 1


def _strict_bool(value, default: bool) -> bool:
    """JSON booleans, or the strings "true"/"false"; anything else -> default."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return default


def _parse_doc_number(value) -> int | None:
    if isinstance(value, bool):  # bool is an int subclass: true is not doc 1
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        return int(match.group()) if match else None
    return None


def _new_agent_record() -> dict:
    return {
        "iterations": 0,
        "actions": [],
        "decide_parse_errors": 0,
        "decide_no_json": 0,
        "rescue": None,
        "critique": None,
        "critique_retries": 0,
        "hedged": False,
        "llm_calls": 0,
        "llm_errors": [],
        "deadline_hit": None,
        "restarts": 0,
    }


class AgentPipeline(RagPipeline):
    def __init__(self, *args, reranker=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reranker = reranker  # preloaded by the stack when agent.rerank is on

    # ------------------------------------------------------------------ tools

    def _get_reranker(self):
        if self._reranker is None:
            from reranking.cross_encoder import CrossEncoderReranker

            rcfg = self.cfg["reranker"]
            self._reranker = CrossEncoderReranker(
                model_name=rcfg["model_name"], max_seq_length=rcfg["max_seq_length"]
            )
        return self._reranker

    async def _rerank(self, query: str, docs: list, top_k: int) -> list:
        reranker = self._get_reranker()
        return await asyncio.to_thread(
            reranker.rerank, query, docs,
            batch_size=self.cfg["reranker"]["batch_size"], top_k=top_k,
        )

    async def _search(self, query: str, run: RunState, call_no: int) -> list:
        """Execute the hybrid_search tool.

        With agent.rerank on, the first search is reranked here; later
        searches are reranked after merging (see _merge_docs).
        """
        agent_cfg = self.cfg["agent"]
        top_k = agent_cfg["top_k"]
        rerank = agent_cfg["rerank"]
        retrieval = await self.retrieve(
            query, run.ctx,
            top_k=agent_cfg["rerank_candidates"] if rerank else top_k,
            label=f"retrieve_{call_no}",
        )
        run.retrieval_ms += retrieval.elapsed_ms
        run.degraded.update(retrieval.failed_methods)
        docs = retrieval.docs
        if rerank and call_no == 1:
            with track_step(run.ctx, f"rerank_{call_no}"):
                docs = await self._rerank(query, docs, top_k)
            run.retrieval_ms += run.ctx.timings[f"rerank_{call_no}"]
        return docs

    @staticmethod
    def _dedup_docs(new: list, current: list) -> list:
        seen: set[str] = set()
        merged = []
        for doc in (*new, *current):
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                merged.append(doc)
        return merged

    @staticmethod
    def _rrf_merge(current: list, new: list, k: int = 60) -> list:
        scores, by_id = {}, {}
        for ranking in (current, new):
            for rank, doc in enumerate(ranking, start=1):
                by_id.setdefault(doc.doc_id, doc)
                scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank)
        return [by_id[d] for d in sorted(scores, key=lambda d: scores[d], reverse=True)]

    async def _merge_docs(
        self, question: str, current: list, new: list, run: RunState, call_no: int
    ) -> list:
        """Merge a reformulated search into the evidence set.

        Reformulated queries improve recall, but the final evidence set is
        selected against the ORIGINAL question: merged, deduped, and (with
        rerank on) reranked before applying top_k. This prevents good
        documents from being lost to merge order or to scores that are not
        comparable across queries.
        """
        top_k = self.cfg["agent"]["top_k"]
        if self.cfg["agent"]["rerank"]:
            step = f"merge_select_{call_no}"
            with track_step(run.ctx, step):
                merged = await self._rerank(question, self._dedup_docs(new, current), top_k)
            run.retrieval_ms += run.ctx.timings[step]
            return merged
        return self._rrf_merge(current, new, k=self.cfg["fusion"]["rrf_k"])[:top_k]

    # ----------------------------------------------------------- LLM helpers

    async def _internal_call(
        self, step: str, messages: list[dict], spec: CallSpec, run: RunState, ag: dict
    ) -> str | None:
        """One internal LLM turn. Returns None on failure — callers degrade."""
        with track_step(run.ctx, step):
            try:
                completion = await self.llm.complete(messages, spec)
            except OllamaError as exc:
                run.usage.failed()
                ag["llm_errors"].append({"step": step, "error": type(exc).__name__})
                self.logger.warning(
                    "%s call failed; degrading: %s", step, exc,
                    extra={"trace_id": run.ctx.trace_id},
                )
                return None
        run.usage.add(completion.prompt_tokens, completion.completion_tokens)
        return completion.text

    async def _decide(
        self, question: str, docs: list, iteration: int, run: RunState, ag: dict
    ) -> Action | None:
        history = [
            f'{a["tool"]}({", ".join(f"{k}={v!r}" for k, v in a["args"].items())})'
            for a in ag["actions"]
        ]
        raw = await self._internal_call(
            f"decide_{iteration}",
            build_decide_messages(
                question, docs, history, iteration, self.cfg["agent"]["max_iterations"]
            ),
            self.specs.decide, run, ag,
        )
        if raw is None:
            return None
        try:
            action = parse_action(raw)
        except ToolCallError as exc:
            ag["decide_parse_errors"] += 1
            self.logger.warning(
                "invalid tool call at iteration %d: %s", iteration, exc,
                extra={"trace_id": run.ctx.trace_id},
            )
            return None
        if action is None:
            ag["decide_no_json"] += 1
        return action

    @staticmethod
    def _picked_doc_ids(numbers: list[int], docs: list) -> list[str] | None:
        """Map the controller's 1-based document numbers to doc ids.

        Out-of-range numbers are dropped and duplicates collapsed. Returns
        None when the controller named no document.
        """
        if not numbers:
            return None
        picked: list[str] = []
        for n in numbers:
            if 1 <= n <= len(docs) and docs[n - 1].doc_id not in picked:
                picked.append(docs[n - 1].doc_id)
        return picked

    async def _rescue(
        self, question: str, docs: list, reason: str, trigger: str, run: RunState, ag: dict
    ) -> dict:
        """Second opinion on a proposed refusal.

        Records and returns {"trigger", "reason", "overridden", "relevant_doc",
        "justification"}: when a relevant doc is found the caller retries
        generation with a hint at that doc instead of refusing.
        """
        raw = await self._internal_call(
            "rescue", build_rescue_messages(question, docs, reason),
            self.specs.rescue, run, ag,
        )
        obj = (extract_json(raw) if raw is not None else None) or {}
        doc_no = _parse_doc_number(obj.get("relevant_doc"))
        valid = doc_no is not None and 1 <= doc_no <= len(docs)
        rescue = {
            "trigger": trigger,
            "reason": reason,
            "overridden": valid,
            "relevant_doc": doc_no if valid else None,
            "justification": str(obj.get("justification", ""))[:300],
        }
        if raw is None:
            rescue["skipped"] = "error"
        ag["rescue"] = rescue
        return rescue

    async def _critique(
        self, question: str, docs: list, answer: str, attempt: int, run: RunState, ag: dict
    ) -> dict:
        """Fact-check the draft against the retrieved documents."""
        raw = await self._internal_call(
            f"critique_{attempt}",
            build_critique_messages(
                question, docs, answer, context_chars=self.cfg["agent"]["context_chars"]
            ),
            self.specs.critique, run, ag,
        )
        if raw is None:
            return {"supported": None, "unsupported_claims": [], "skipped": "error"}
        obj = extract_json(raw)
        if obj is None:
            # A judge that cannot be parsed must not block the answer.
            return {"supported": True, "unsupported_claims": [], "parse_error": True}
        claims = obj.get("unsupported_claims") or []
        if isinstance(claims, str):
            claims = [claims]
        if not isinstance(claims, list):
            claims = []
        claims = [str(c) for c in claims if str(c).strip()][:10]
        return {
            # Missing/garbled "supported" falls back to what the claims say.
            "supported": _strict_bool(obj.get("supported"), default=not claims),
            "unsupported_claims": claims,
        }

    async def _generate(self, messages: list[dict], draft: _Draft, run: RunState, ag: dict, max_restarts: int = 1):
        """Stream one generation turn.

        Yields delta events ONLY when the draft is not a refusal: a leading
        refusal marker is intercepted (buffered, never emitted) so a false
        refusal can be rescued without the user ever seeing it. A stream that
        breaks mid-answer is restarted once (with a reset event if text was
        already shown), budget permitting.
        """
        gate = RefusalGate()
        shown = False
        while True:
            end = None
            try:
                async with aclosing(self.llm.stream(messages, self.specs.answer)) as stream:
                    async for ev in stream:
                        if isinstance(ev, StreamEnd):
                            end = ev
                            continue
                        run.mark_token()
                        draft.text += ev.text
                        out = gate.feed(ev.text)
                        if out:
                            run.mark_visible()
                            shown = True
                            yield {"type": "delta", "text": out}
                tail = gate.flush()
                draft.refusal = gate.refusal
                if tail:
                    run.mark_visible()
                    yield {"type": "delta", "text": tail}
                self._note_finish(run, end, "generate")
                return
            except StreamInterrupted as exc:
                run.usage.failed()
                if draft.restarts >= max_restarts or run.out_of_time():
                    raise
                self.logger.warning(
                    "stream broke mid-answer, restarting generation: %s", exc,
                    extra={"trace_id": run.ctx.trace_id},
                )
                if shown:
                    yield {"type": "reset"}
                draft.restart()
                ag["restarts"] += 1
                gate.reset()
                shown = False

    # --------------------------------------------------------------- endings

    def _finish(self, run: RunState, ag: dict, text: str, refusal: bool, docs: list) -> dict:
        ag["llm_calls"] = run.usage.llm_calls
        return self._done_event(run, text, refusal, docs, agent=ag)

    def _refuse(self, run: RunState, ag: dict, reason: str, docs: list):
        """Deterministic honest refusal, delivered as text (delta) then done."""
        text = build_refusal_text(reason, docs)
        run.mark_visible()
        return [
            {"type": "delta", "text": text},
            self._finish(run, ag, text, True, docs),
        ]

    # -------------------------------------------------------------- main loop

    async def answer_stream(
        self, question: str, trace_id: str | None = None, deadline: float | None = None
    ):
        run = self._start_run(trace_id, deadline)
        ag = _new_agent_record()
        agent_cfg = self.cfg["agent"]

        # -- observation 1: system-side pre-fetch with the raw question ------
        docs = await self._search(question, run, 1)
        yield {"type": "step", "step": "retrieve", "n_docs": len(docs)}
        if not docs:
            for ev in self._refuse(run, ag, "no documents matched the question", docs):
                yield ev
            return

        # -- ReAct loop: decide -> act -> observe ----------------------------
        hint_doc: int | None = None
        for iteration in range(1, agent_cfg["max_iterations"] + 1):
            if run.out_of_time():
                ag["deadline_hit"] = "decide"
                break
            ag["iterations"] = iteration
            action = await self._decide(question, docs, iteration, run, ag)
            if action is None:
                # Failed or unparseable decide turn: fall through to
                # generation — the safest default given the measured
                # over-refusal failure mode.
                break
            record = {"iteration": iteration, "tool": action.tool, "args": action.args}
            ag["actions"].append(record)
            yield {"type": "step", "step": "decide", "tool": action.tool}

            if action.tool == "answer":
                # Recorded for analysis only (does the controller pick the
                # gold doc, or the near-miss?). Deliberately not fed to
                # generation as a hint: steering the writer toward a doc is
                # exactly what makes near-miss evidence turn into an answer.
                picked = self._picked_doc_ids(action.args.get("documents", []), docs)
                if picked is not None:
                    record["doc_ids"] = picked
                break

            if action.tool == "hybrid_search":
                try:
                    new_docs = await self._search(action.args["query"], run, iteration + 1)
                except RetrievalUnavailable:
                    # We already hold evidence from earlier searches; an
                    # outage on a reformulation must not sink the request.
                    self.logger.warning(
                        "reformulated search failed; keeping current evidence",
                        extra={"trace_id": run.ctx.trace_id},
                    )
                    break
                docs = await self._merge_docs(question, docs, new_docs, run, iteration + 1)
                yield {"type": "step", "step": "retrieve", "n_docs": len(docs)}
                continue

            if action.tool == "insufficient_evidence":
                reason = action.args["reason"]
                rescue = await self._rescue(question, docs, reason, "decide", run, ag)
                if rescue["overridden"]:
                    hint_doc = rescue["relevant_doc"]
                    yield {"type": "step", "step": "rescue_override", "doc": hint_doc}
                    break
                for ev in self._refuse(run, ag, reason, docs):
                    yield ev
                return

            # Defensive: a tool added to TOOL_SPECS without a handler here.
            self.logger.error(
                "no handler for tool %r; falling through to generation", action.tool,
                extra={"trace_id": run.ctx.trace_id},
            )
            break
        # Loop exhausted with endless reformulation: generate from what we
        # have — the critique step still guards the output.

        # -- generation + critique -------------------------------------------
        attempt = 0
        feedback: list[str] | None = None
        while True:
            attempt += 1
            messages = build_answer_messages(
                question, docs, hint_doc=hint_doc, feedback=feedback,
                context_chars=agent_cfg["context_chars"],
            )
            draft = _Draft()
            with track_step(run.ctx, f"generate_{attempt}"):
                async for ev in self._generate(messages, draft, run, ag):
                    yield ev

            if draft.refusal:
                reason = refusal_reason(draft.text) or DEFAULT_REFUSAL_REASON
                if ag["rescue"] is None and not run.out_of_time():
                    rescue = await self._rescue(question, docs, reason, "generate", run, ag)
                    if rescue["overridden"]:
                        hint_doc = rescue["relevant_doc"]
                        feedback = None
                        yield {"type": "step", "step": "rescue_override", "doc": hint_doc}
                        continue
                for ev in self._refuse(run, ag, reason, docs):
                    yield ev
                return

            if run.out_of_time():
                ag["deadline_hit"] = "critique"
                ag["critique"] = {"supported": None, "unsupported_claims": [], "skipped": "deadline"}
                yield self._finish(run, ag, draft.text, False, docs)
                return

            verdict = await self._critique(question, docs, draft.text, attempt, run, ag)
            ag["critique"] = verdict
            if verdict["supported"] is not False:
                yield self._finish(run, ag, draft.text, False, docs)
                return
            if ag["critique_retries"] < agent_cfg["max_retries"] and not run.out_of_time():
                ag["critique_retries"] += 1
                feedback = verdict["unsupported_claims"]
                yield {"type": "step", "step": "correction", "unsupported_claims": len(feedback)}
                yield {"type": "reset"}
                continue
            # Retries exhausted: hedge rather than refuse — an answer with a
            # visible warning beats a false refusal.
            ag["hedged"] = True
            hedged_text = HEDGE_PREFIX + draft.text
            yield {"type": "reset"}
            yield {"type": "delta", "text": hedged_text}
            yield self._finish(run, ag, hedged_text, False, docs)
            return
