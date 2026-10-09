"""Unit tests for the agent loop — no LLM, ES, or GPU required.

A FakeLLM is scripted per turn category (decide / answer / rescue / critique,
distinguished by their system prompts) and retrieval is patched at the
instance level, so these tests exercise the real AgentPipeline control flow:
ReAct bounding, refusal rescue, refusal interception during streaming,
critique retries, the hedge fallback, and the degrade policy.
"""

import asyncio
import logging
import time
import unittest

from agent.agent import AgentPipeline, RefusalGate
from agent.llm import Completion, OllamaError, StreamEnd, StreamInterrupted, Token
from agent.pipeline import RagPipeline, Retrieval, RetrievalUnavailable
from agent.prompts import HEDGE_PREFIX, build_answer_messages, build_refusal_text
from agent.tools import Action, ToolCallError, extract_json, parse_action, render_tool_docs
from configs.loader import validate_config
from retrieval.types import FusedResult

logging.disable(logging.CRITICAL)


def make_docs(n=3, prefix="doc"):
    return [
        FusedResult(
            doc_id=f"{prefix}{i}",
            score=1.0 - i * 0.1,
            title=f"Title {i} about topic {i}",
            text=f"Body of document {i}. " * 10,
        )
        for i in range(1, n + 1)
    ]


DEFAULTS = {
    "decide": '{"tool": "answer", "args": {}}',
    "answer": "The result holds [1].",
    "rescue": '{"relevant_doc": null, "justification": "none"}',
    "critique": '{"supported": true, "unsupported_claims": []}',
}


class FakeLLM:
    """Scripted LLM implementing the LLMClient interface (complete/stream).

    Args:
        fail: turn categories whose calls raise OllamaError.
        interrupt: number of answer streams that break after one token.
    """

    def __init__(self, decide=None, answer=None, rescue=None, critique=None,
                 fail=(), interrupt=0):
        self.script = {
            "decide": list(decide or []),
            "answer": list(answer or []),
            "rescue": list(rescue or []),
            "critique": list(critique or []),
        }
        self.fail = set(fail)
        self.interrupt = interrupt
        self.calls = []  # (category, messages)

    def _respond(self, messages):
        system = messages[0]["content"]
        if "controller" in system:
            key = "decide"
        elif "second opinion" in system:
            key = "rescue"
        elif "fact-checker" in system:
            key = "critique"
        else:
            key = "answer"
        self.calls.append((key, messages))
        if key in self.fail:
            raise OllamaError(f"{key} backend down", retryable=False)
        queue = self.script[key]
        return queue.pop(0) if queue else DEFAULTS[key]

    async def complete(self, messages, spec):
        text = self._respond(messages)
        return Completion(text, "stop", 100, 10, 1)

    async def stream(self, messages, spec):
        text = self._respond(messages)
        words = text.split(" ")
        if self.interrupt:
            self.interrupt -= 1
            yield Token(words[0] + " ")
            raise StreamInterrupted("connection reset")
        for word in words:
            yield Token(word + " ")
        yield StreamEnd("stop", 100, len(words))


def make_cfg(**agent):
    return validate_config({"agent": {"mode": "agent", "rerank": False, **agent}})


def make_pipe(llm, search_results=None, cls=AgentPipeline, **agent_cfg):
    pipe = cls(
        cfg=make_cfg(**agent_cfg),
        es=None,
        corpus={},
        faiss_index=None,
        dense_encoder=None,
        splade_encoder=None,
        llm=llm,
        logger=logging.getLogger("test_agent"),
    )
    searches: list[str] = []

    async def fake_retrieve(query, ctx, *, top_k=None, label="retrieve"):
        searches.append(query)
        result = search_results(query) if callable(search_results) else search_results
        if isinstance(result, Retrieval):
            return result
        docs = list(make_docs() if result is None else result)
        ctx.timings[label] = 5.0
        return Retrieval(docs, 5.0, ())

    pipe.retrieve = fake_retrieve
    pipe.searches = searches
    return pipe


def run_pipe(pipe, question="What is X?", deadline=None):
    async def go():
        events = []
        async for ev in pipe.answer_stream(question, deadline=deadline):
            events.append(ev)
        return events

    return asyncio.run(go())


def streamed_text(events):
    """What a client ends up displaying: deltas, with resets clearing them."""
    text = ""
    for ev in events:
        if ev["type"] == "reset":
            text = ""
        elif ev["type"] == "delta":
            text += ev["text"]
    return text


class TestTools(unittest.TestCase):
    def test_parse_valid(self):
        action = parse_action('Here you go: {"tool": "hybrid_search", "args": {"query": "q"}}')
        self.assertEqual(action, Action(tool="hybrid_search", args={"query": "q"}))

    def test_parse_no_json_returns_none(self):
        self.assertIsNone(parse_action("I will answer directly."))

    def test_unknown_tool_raises(self):
        with self.assertRaises(ToolCallError):
            parse_action('{"tool": "google", "args": {}}')

    def test_missing_required_param_raises(self):
        with self.assertRaises(ToolCallError):
            parse_action('{"tool": "hybrid_search", "args": {}}')

    def test_wrong_type_raises(self):
        with self.assertRaises(ToolCallError):
            parse_action('{"tool": "hybrid_search", "args": {"query": 5}}')

    def test_empty_string_param_raises(self):
        with self.assertRaises(ToolCallError):
            parse_action('{"tool": "hybrid_search", "args": {"query": "  "}}')

    def test_unknown_param_raises(self):
        with self.assertRaises(ToolCallError):
            parse_action('{"tool": "hybrid_search", "args": {"query": "q", "top": 3}}')

    def test_answer_action_has_empty_args(self):
        action = parse_action('{"tool": "answer"}')
        self.assertEqual(action.tool, "answer")
        self.assertEqual(action.args, {})

    def test_answer_with_documents_as_qwen_emits_it(self):
        # Verbatim decide output from qwen2.5:7b-instruct; it used to be
        # rejected as "unknown params for answer: ['documents']".
        action = parse_action('{"tool": "answer", "args": {"documents": [1, 5, 9]}}')
        self.assertEqual(action, Action(tool="answer", args={"documents": [1, 5, 9]}))

    def test_array_items_are_type_checked(self):
        for args in ('{"documents": 1}', '{"documents": ["1"]}',
                     '{"documents": [true]}', '{"documents": [1.5]}'):
            with self.subTest(args=args):
                with self.assertRaises(ToolCallError):
                    parse_action('{"tool": "answer", "args": %s}' % args)

    def test_tool_docs_render_array_params(self):
        self.assertIn("answer(documents (array of integers, optional))", render_tool_docs())

    def test_extract_json_nested_and_prose(self):
        obj = extract_json('sure! {"a": {"b": [1, 2]}} thanks')
        self.assertEqual(obj, {"a": {"b": [1, 2]}})
        self.assertIsNone(extract_json("no braces at all"))
        self.assertIsNone(extract_json("broken {"))


class TestRefusalGate(unittest.TestCase):
    def feed_all(self, chunks):
        gate = RefusalGate()
        shown = "".join(gate.feed(c) for c in chunks) + gate.flush()
        return shown, gate.refusal

    def test_answer_passes_through(self):
        self.assertEqual(self.feed_all(["The ", "result [1]."]), ("The result [1].", False))

    def test_marker_split_across_chunks_is_swallowed(self):
        shown, refusal = self.feed_all(["INSUFF", "ICIENT_EVI", "DENCE: none", " here"])
        self.assertEqual((shown, refusal), ("", True))

    def test_marker_variants_are_refusals(self):
        for text in ("INSUFFICIENT EVIDENCE - nothing", "**Insufficient_Evidence**: no",
                     "  insufficient evidence.", "INSUFFICIENT_EVIDENCE"):
            with self.subTest(text=text):
                self.assertEqual(self.feed_all([text]), ("", True))

    def test_near_miss_is_streamed(self):
        self.assertEqual(self.feed_all(["Insufficient", " data exists"]),
                         ("Insufficient data exists", False))


class TestAgentLoop(unittest.TestCase):
    def test_happy_path(self):
        llm = FakeLLM()
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertFalse(done["refusal"])
        self.assertEqual(done["citations"], {"1": "doc1"})
        self.assertEqual(done["agent"]["iterations"], 1)
        self.assertEqual(done["agent"]["llm_calls"], 3)  # decide + generate + critique
        self.assertEqual(done["agent"]["actions"][0]["tool"], "answer")
        self.assertTrue(done["agent"]["critique"]["supported"])
        self.assertEqual(done["usage"]["llm_calls"], 3)
        self.assertGreater(done["usage"]["prompt_tokens"], 0)
        self.assertEqual(done["degraded"], [])
        self.assertEqual(streamed_text(events), done["text"])
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "step")
        self.assertIn("delta", types)
        self.assertEqual(types[-1], "done")

    def test_answer_records_picked_docs_without_hinting_generation(self):
        llm = FakeLLM(decide=['{"tool": "answer", "args": {"documents": [2, 2, 7, 1]}}'])
        done = run_pipe(make_pipe(llm))[-1]
        self.assertEqual(done["agent"]["decide_parse_errors"], 0)
        action = done["agent"]["actions"][0]
        self.assertEqual(action["tool"], "answer")
        self.assertEqual(action["doc_ids"], ["doc2", "doc1"])  # 7 is out of range
        answer_prompt = next(m for key, m in llm.calls if key == "answer")
        self.assertEqual(answer_prompt, build_answer_messages("What is X?", make_docs(),
                                                              context_chars=make_cfg()["agent"]["context_chars"]))

    def test_search_reformulation(self):
        llm = FakeLLM(
            decide=[
                '{"tool": "hybrid_search", "args": {"query": "X reformulated"}}',
                '{"tool": "answer", "args": {}}',
            ]
        )
        pipe = make_pipe(llm)
        done = run_pipe(pipe)[-1]
        self.assertEqual(pipe.searches, ["What is X?", "X reformulated"])
        self.assertEqual(done["retrieved_doc_ids"][0], "doc1")
        self.assertEqual(done["agent"]["iterations"], 2)
        self.assertEqual(len(done["agent"]["actions"]), 2)
        # repeated searches keep separate timing keys
        self.assertIn("retrieve_1", done["step_timings"])
        self.assertIn("retrieve_2", done["step_timings"])

    def test_merge_prefers_new_results(self):
        merged = AgentPipeline._dedup_docs(new=make_docs(2, "new"), current=make_docs(2, "old"))
        self.assertEqual([d.doc_id for d in merged], ["new1", "new2", "old1", "old2"])
        merged = AgentPipeline._dedup_docs(new=make_docs(2, "same"), current=make_docs(2, "same"))
        self.assertEqual([d.doc_id for d in merged], ["same1", "same2"])

    def test_rrf_merge_ranks_docs_found_by_both_searches_first(self):
        current = make_docs(3, "a")
        new = [make_docs(3, "a")[2], *make_docs(2, "b")]
        merged = AgentPipeline._rrf_merge(current, new)
        self.assertEqual(merged[0].doc_id, "a3")

    def test_refusal_accepted_when_rescue_agrees(self):
        llm = FakeLLM(
            decide=['{"tool": "insufficient_evidence", "args": {"reason": "off topic"}}'],
            rescue=['{"relevant_doc": null, "justification": "unrelated"}'],
        )
        events = run_pipe(make_pipe(llm))
        done = events[-1]
        self.assertTrue(done["refusal"])
        self.assertTrue(done["text"].startswith("INSUFFICIENT_EVIDENCE:"))
        # honest refusal names the closest retrieved docs and cites them
        self.assertIn("Title 1", done["text"])
        self.assertIn("doc1", done["citations"].values())
        self.assertFalse(done["agent"]["rescue"]["overridden"])
        self.assertEqual(done["agent"]["llm_calls"], 2)  # decide + rescue only
        # the refusal text reaches the client as a delta
        self.assertEqual(streamed_text(events), done["text"])

    def test_rescue_overrides_decide_refusal(self):
        llm = FakeLLM(
            decide=['{"tool": "insufficient_evidence", "args": {"reason": "nope"}}'],
            rescue=['{"relevant_doc": 2, "justification": "doc 2 bears on it"}'],
        )
        events = run_pipe(make_pipe(llm))
        done = events[-1]
        self.assertFalse(done["refusal"])
        self.assertTrue(done["agent"]["rescue"]["overridden"])
        self.assertEqual(done["agent"]["rescue"]["trigger"], "decide")
        answer_calls = [m for cat, m in llm.calls if cat == "answer"]
        self.assertIn("as relevant to this question", answer_calls[0][1]["content"])
        self.assertIn("rescue_override", [e.get("step") for e in events])

    def test_rescue_ignores_boolean_doc_number(self):
        llm = FakeLLM(
            decide=['{"tool": "insufficient_evidence", "args": {"reason": "nope"}}'],
            rescue=['{"relevant_doc": true, "justification": "yes"}'],
        )
        done = run_pipe(make_pipe(llm))[-1]
        self.assertTrue(done["refusal"])
        self.assertFalse(done["agent"]["rescue"]["overridden"])

    def test_generation_refusal_intercepted_and_rescued(self):
        llm = FakeLLM(
            answer=["INSUFFICIENT_EVIDENCE: nothing relevant", "The answer is yes [1]."],
            rescue=['{"relevant_doc": 1, "justification": "doc 1 works"}'],
        )
        events = run_pipe(make_pipe(llm))
        deltas = streamed_text(events)
        done = events[-1]
        self.assertNotIn("INSUFFICIENT", deltas)
        self.assertTrue(deltas.strip().startswith("The answer is yes"))
        self.assertFalse(done["refusal"])
        self.assertEqual(done["agent"]["rescue"]["trigger"], "generate")

    def test_generation_refusal_variant_without_colon_is_intercepted(self):
        llm = FakeLLM(
            answer=["**INSUFFICIENT EVIDENCE** - no study covers this"],
            rescue=['{"relevant_doc": null, "justification": "unrelated"}'],
        )
        events = run_pipe(make_pipe(llm))
        done = events[-1]
        self.assertTrue(done["refusal"])
        self.assertIn("no study covers this", done["text"])
        self.assertEqual(streamed_text(events), done["text"])

    def test_generation_refusal_accepted_after_failed_rescue(self):
        llm = FakeLLM(
            answer=["INSUFFICIENT_EVIDENCE: cannot answer"],
            rescue=['{"relevant_doc": null, "justification": "truly unrelated"}'],
        )
        events = run_pipe(make_pipe(llm))
        done = events[-1]
        self.assertTrue(done["refusal"])
        self.assertTrue(done["text"].startswith("INSUFFICIENT_EVIDENCE: cannot answer"))
        self.assertIn("Title 1", done["text"])
        self.assertIn("doc1", done["citations"].values())
        # only the deterministic refusal is shown, never the raw model output
        self.assertEqual(streamed_text(events), done["text"])
        # latency marks are request-level: the refusal text was visible
        self.assertGreater(done["timings"]["first_token_ms"], 0)
        self.assertGreaterEqual(
            done["timings"]["first_visible_ms"], done["timings"]["first_token_ms"]
        )

    def test_critique_retry_then_pass(self):
        llm = FakeLLM(
            answer=["Wild invented claim [1].", "Careful supported claim [1]."],
            critique=[
                '{"supported": false, "unsupported_claims": ["Wild invented claim"]}',
                '{"supported": true, "unsupported_claims": []}',
            ],
        )
        events = run_pipe(make_pipe(llm))
        done = events[-1]
        self.assertIn("correction", [e.get("step") for e in events])
        self.assertIn("reset", [e["type"] for e in events])
        self.assertIn("Careful supported claim", done["text"])
        self.assertNotIn("Wild invented", done["text"])
        self.assertEqual(done["agent"]["critique"]["supported"], True)
        self.assertEqual(done["agent"]["critique_retries"], 1)
        self.assertEqual(streamed_text(events), done["text"])

    def test_critique_string_false_is_not_supported(self):
        llm = FakeLLM(
            answer=["Claim A [1].", "Claim B [1]."],
            critique=['{"supported": "false", "unsupported_claims": ["Claim A"]}'],
        )
        done = run_pipe(make_pipe(llm))[-1]
        self.assertEqual(done["agent"]["critique_retries"], 1)
        self.assertIn("Claim B", done["text"])

    def test_critique_missing_verdict_uses_claims(self):
        llm = FakeLLM(
            answer=["Claim A [1].", "Claim B [1]."],
            critique=['{"unsupported_claims": ["Claim A"]}'],
        )
        done = run_pipe(make_pipe(llm))[-1]
        self.assertEqual(done["agent"]["critique_retries"], 1)

    def test_critique_exhausted_hedges(self):
        llm = FakeLLM(
            answer=["Bad claim one [1].", "Bad claim two [1].", "Bad claim three [1]."],
            critique=['{"supported": false, "unsupported_claims": ["bad"]}'] * 3,
        )
        events = run_pipe(make_pipe(llm))
        done = events[-1]
        self.assertFalse(done["refusal"])
        self.assertTrue(done["text"].startswith(HEDGE_PREFIX))
        self.assertTrue(done["agent"]["hedged"])
        self.assertEqual(streamed_text(events), done["text"])

    def test_max_iterations_bounded_then_generates(self):
        llm = FakeLLM(decide=['{"tool": "hybrid_search", "args": {"query": "again"}}'] * 4)
        pipe = make_pipe(llm)
        done = run_pipe(pipe)[-1]
        self.assertEqual(len(pipe.searches), 5)  # pre-fetch + one per iteration
        self.assertEqual(done["agent"]["iterations"], 4)
        self.assertFalse(done["refusal"])

    def test_decide_parse_error_falls_through_to_generation(self):
        llm = FakeLLM(decide=['{"tool": "teleport", "args": {}}'])
        done = run_pipe(make_pipe(llm))[-1]
        self.assertFalse(done["refusal"])
        self.assertEqual(done["agent"]["decide_parse_errors"], 1)

    def test_decide_unparseable_falls_through(self):
        llm = FakeLLM(decide=["Let me just answer this directly."])
        done = run_pipe(make_pipe(llm))[-1]
        self.assertFalse(done["refusal"])
        self.assertEqual(done["agent"]["decide_parse_errors"], 0)
        self.assertEqual(done["agent"]["decide_no_json"], 1)
        self.assertEqual(done["agent"]["actions"], [])

    def test_refusal_text_builder(self):
        text = build_refusal_text("no data on X", make_docs(2))
        self.assertTrue(text.startswith("INSUFFICIENT_EVIDENCE: no data on X."))
        self.assertIn("[1] Title 1", text)
        self.assertIn("[2] Title 2", text)
        self.assertIn("No documents were retrieved", build_refusal_text("x", []))


class TestDegradePolicy(unittest.TestCase):
    def test_decide_failure_falls_through_to_generation(self):
        llm = FakeLLM(fail={"decide"})
        done = run_pipe(make_pipe(llm))[-1]
        self.assertFalse(done["refusal"])
        self.assertEqual(done["agent"]["llm_errors"][0]["step"], "decide_1")
        self.assertEqual(done["usage"]["llm_errors"], 1)

    def test_critique_failure_delivers_answer_marked_skipped(self):
        llm = FakeLLM(fail={"critique"})
        done = run_pipe(make_pipe(llm))[-1]
        self.assertFalse(done["refusal"])
        self.assertEqual(done["agent"]["critique"]["skipped"], "error")
        self.assertIsNone(done["agent"]["critique"]["supported"])

    def test_rescue_failure_lets_refusal_stand(self):
        llm = FakeLLM(
            decide=['{"tool": "insufficient_evidence", "args": {"reason": "off topic"}}'],
            fail={"rescue"},
        )
        done = run_pipe(make_pipe(llm))[-1]
        self.assertTrue(done["refusal"])
        self.assertEqual(done["agent"]["rescue"]["skipped"], "error")

    def test_generation_failure_propagates(self):
        llm = FakeLLM(fail={"answer"})
        with self.assertRaises(OllamaError):
            run_pipe(make_pipe(llm))

    def test_stream_interrupted_restarts_once_with_reset(self):
        llm = FakeLLM(interrupt=1)
        events = run_pipe(make_pipe(llm))
        done = events[-1]
        self.assertIn("reset", [e["type"] for e in events])
        self.assertEqual(done["agent"]["restarts"], 1)
        self.assertEqual(streamed_text(events), done["text"])

    def test_stream_interrupted_twice_raises(self):
        llm = FakeLLM(interrupt=2)
        with self.assertRaises(StreamInterrupted):
            run_pipe(make_pipe(llm))

    def test_expired_deadline_skips_optional_steps(self):
        llm = FakeLLM()
        done = run_pipe(make_pipe(llm), deadline=time.perf_counter() - 1)[-1]
        self.assertFalse(done["refusal"])
        self.assertEqual(done["agent"]["iterations"], 0)
        self.assertEqual(done["agent"]["deadline_hit"], "critique")
        self.assertEqual(done["agent"]["critique"]["skipped"], "deadline")
        self.assertEqual([cat for cat, _ in llm.calls], ["answer"])

    def test_partial_retrieval_failure_is_reported(self):
        llm = FakeLLM()
        pipe = make_pipe(llm, search_results=Retrieval(make_docs(), 5.0, ("bm25",)))
        done = run_pipe(pipe)[-1]
        self.assertEqual(done["degraded"], ["bm25"])

    def test_total_retrieval_outage_raises(self):
        def outage(query):
            raise RetrievalUnavailable("all retrieval methods failed")

        with self.assertRaises(RetrievalUnavailable):
            run_pipe(make_pipe(FakeLLM(), search_results=outage))

    def test_outage_on_reformulation_keeps_current_evidence(self):
        def search(query):
            if query != "What is X?":
                raise RetrievalUnavailable("down")
            return make_docs()

        llm = FakeLLM(decide=['{"tool": "hybrid_search", "args": {"query": "other"}}'])
        done = run_pipe(make_pipe(llm, search_results=search))[-1]
        self.assertFalse(done["refusal"])
        self.assertEqual(done["retrieved_doc_ids"], ["doc1", "doc2", "doc3"])

    def test_no_documents_refuses_without_llm(self):
        llm = FakeLLM()
        events = run_pipe(make_pipe(llm, search_results=[]))
        done = events[-1]
        self.assertTrue(done["refusal"])
        self.assertEqual(llm.calls, [])
        self.assertEqual(streamed_text(events), done["text"])


class TestBaselinePipeline(unittest.TestCase):
    def test_streams_answer_and_reports_usage(self):
        llm = FakeLLM(answer=["Yes it does [2]."])
        events = run_pipe(make_pipe(llm, cls=RagPipeline))
        done = events[-1]
        self.assertFalse(done["refusal"])
        self.assertEqual(done["citations"], {"2": "doc2"})
        self.assertEqual(done["usage"]["llm_calls"], 1)
        self.assertEqual(streamed_text(events), done["text"])

    def test_refusal_variant_detected(self):
        llm = FakeLLM(answer=["Insufficient evidence: nothing"])
        done = run_pipe(make_pipe(llm, cls=RagPipeline))[-1]
        self.assertTrue(done["refusal"])


if __name__ == "__main__":
    unittest.main()
