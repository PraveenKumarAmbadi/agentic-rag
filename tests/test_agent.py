"""Unit tests for the Phase 3 agent loop — no LLM, ES, or GPU required.

A FakeLLM is scripted per turn category (decide / answer / rescue / critique,
distinguished by their system prompts) and retrieval is patched at the
instance level, so these tests exercise the real AgentPipeline control flow:
ReAct bounding, refusal rescue, refusal interception during streaming,
critique retries, and the hedge fallback.
"""

import asyncio
import logging
import unittest

from agent.agent import AgentPipeline
from agent.prompts import HEDGE_PREFIX, build_refusal_text
from agent.tools import Action, ToolCallError, extract_json, parse_action
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
    """Scripted LLM: queues of replies per turn category."""

    def __init__(self, decide=None, answer=None, rescue=None, critique=None):
        self.script = {
            "decide": list(decide or []),
            "answer": list(answer or []),
            "rescue": list(rescue or []),
            "critique": list(critique or []),
        }
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
        queue = self.script[key]
        return queue.pop(0) if queue else DEFAULTS[key]

    async def chat(self, messages, temperature=None, max_tokens=None):
        return self._respond(messages)

    async def stream_chat(self, messages):
        for word in self._respond(messages).split(" "):
            yield word + " "


CFG = {
    "agent": {
        "mode": "agent",
        "max_iterations": 4,
        "max_retries": 2,
        "rerank": False,
        "top_k": 10,
    },
    "llm": {"judge_temperature": 0.0},
}


def make_pipe(llm, search_results=None):
    pipe = AgentPipeline(
        cfg=CFG,
        es=None,
        corpus={},
        faiss_index=None,
        dense_encoder=None,
        splade_encoder=None,
        llm=llm,
        logger=logging.getLogger("test_agent"),
    )
    searches: list[str] = []

    async def fake_retrieve(question, trace_id, ctx):
        searches.append(question)
        if callable(search_results):
            return search_results(question), 5.0
        return list(search_results or make_docs()), 5.0

    pipe.retrieve = fake_retrieve
    pipe.searches = searches
    return pipe


def run_pipe(pipe, question="What is X?"):
    async def go():
        events = []
        async for ev in pipe.answer_stream(question):
            events.append(ev)
        return events

    return asyncio.run(go())


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

    def test_extract_json_nested_and_prose(self):
        obj = extract_json('sure! {"a": {"b": [1, 2]}} thanks')
        self.assertEqual(obj, {"a": {"b": [1, 2]}})
        self.assertIsNone(extract_json("no braces at all"))
        self.assertIsNone(extract_json("broken {"))


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
        # events before deltas, deltas before done
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "step")
        self.assertIn("delta", types)
        self.assertEqual(types[-1], "done")

    def test_search_reformulation(self):
        llm = FakeLLM(
            decide=[
                '{"tool": "hybrid_search", "args": {"query": "X reformulated"}}',
                '{"tool": "answer", "args": {}}',
            ]
        )
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        done = events[-1]
        self.assertEqual(pipe.searches, ["What is X?", "X reformulated"])
        # merged context puts the reformulated-search results first
        self.assertEqual(done["retrieved_doc_ids"][0], "doc1")
        self.assertEqual(done["agent"]["iterations"], 2)
        self.assertEqual(len(done["agent"]["actions"]), 2)

    def test_merge_prefers_new_results(self):
        merged = AgentPipeline._merge_docs(make_docs(2, "old"), make_docs(2, "new"), 10)
        self.assertEqual([d.doc_id for d in merged], ["new1", "new2", "old1", "old2"])
        # dedupe keeps the first occurrence when both searches return a doc
        merged = AgentPipeline._merge_docs(make_docs(2, "same"), make_docs(2, "same"), 10)
        self.assertEqual([d.doc_id for d in merged], ["same1", "same2"])
        # top_k truncates after the merge (5 unique docs, cap 4)
        merged = AgentPipeline._merge_docs(make_docs(3, "a"), make_docs(2, "b"), 4)
        self.assertEqual([d.doc_id for d in merged], ["b1", "b2", "a1", "a2"])

    def test_refusal_accepted_when_rescue_agrees(self):
        llm = FakeLLM(
            decide=['{"tool": "insufficient_evidence", "args": {"reason": "off topic"}}'],
            rescue=['{"relevant_doc": null, "justification": "unrelated"}'],
        )
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        done = events[-1]
        self.assertTrue(done["refusal"])
        self.assertTrue(done["text"].startswith("INSUFFICIENT_EVIDENCE:"))
        # honest refusal names the closest retrieved docs and cites them
        self.assertIn("Title 1", done["text"])
        self.assertIn("doc1", done["citations"].values())
        self.assertFalse(done["agent"]["rescue"]["overridden"])
        self.assertEqual(done["agent"]["llm_calls"], 2)  # decide + rescue only
        # no deltas streamed for a refusal
        self.assertNotIn("delta", [e["type"] for e in events])

    def test_rescue_overrides_decide_refusal(self):
        llm = FakeLLM(
            decide=['{"tool": "insufficient_evidence", "args": {"reason": "nope"}}'],
            rescue=['{"relevant_doc": 2, "justification": "doc 2 bears on it"}'],
        )
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        done = events[-1]
        self.assertFalse(done["refusal"])
        self.assertTrue(done["agent"]["rescue"]["overridden"])
        self.assertEqual(done["agent"]["rescue"]["trigger"], "decide")
        # generation was hinted at document 2
        answer_calls = [m for cat, m in llm.calls if cat == "answer"]
        self.assertIn("likely relevant", answer_calls[0][1]["content"])
        self.assertIn("rescue_override", [e.get("step") for e in events])

    def test_generation_refusal_intercepted_and_rescued(self):
        llm = FakeLLM(
            answer=["INSUFFICIENT_EVIDENCE: nothing relevant", "The answer is yes [1]."],
            rescue=['{"relevant_doc": 1, "justification": "doc 1 works"}'],
        )
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        deltas = "".join(e["text"] for e in events if e["type"] == "delta")
        done = events[-1]
        # the refusal was never streamed to the user
        self.assertNotIn("INSUFFICIENT", deltas)
        self.assertTrue(deltas.strip().startswith("The answer is yes"))
        self.assertFalse(done["refusal"])
        self.assertEqual(done["agent"]["rescue"]["trigger"], "generate")

    def test_generation_refusal_accepted_after_failed_rescue(self):
        llm = FakeLLM(
            answer=["INSUFFICIENT_EVIDENCE: cannot answer"],
            rescue=['{"relevant_doc": null, "justification": "truly unrelated"}'],
        )
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        done = events[-1]
        self.assertTrue(done["refusal"])
        self.assertTrue(done["text"].startswith("INSUFFICIENT_EVIDENCE:"))
        self.assertNotIn("delta", [e["type"] for e in events])

    def test_critique_retry_then_pass(self):
        llm = FakeLLM(
            answer=["Wild invented claim [1].", "Careful supported claim [1]."],
            critique=[
                '{"supported": false, "unsupported_claims": ["Wild invented claim"]}',
                '{"supported": true, "unsupported_claims": []}',
            ],
        )
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        done = events[-1]
        self.assertIn("correction", [e.get("step") for e in events])
        self.assertIn("Careful supported claim", done["text"])
        self.assertNotIn("Wild invented", done["text"])
        self.assertEqual(done["agent"]["critique"]["supported"], True)

    def test_critique_exhausted_hedges(self):
        llm = FakeLLM(
            answer=["Bad claim one [1].", "Bad claim two [1].", "Bad claim three [1]."],
            critique=['{"supported": false, "unsupported_claims": ["bad"]}'] * 3,
        )
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        done = events[-1]
        self.assertFalse(done["refusal"])
        self.assertTrue(done["text"].startswith(HEDGE_PREFIX))
        self.assertTrue(done["agent"]["hedged"])

    def test_max_iterations_bounded_then_generates(self):
        llm = FakeLLM(decide=['{"tool": "hybrid_search", "args": {"query": "again"}}'] * 4)
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        done = events[-1]
        # pre-fetch + one search per iteration
        self.assertEqual(len(pipe.searches), 5)
        self.assertEqual(done["agent"]["iterations"], 4)
        self.assertFalse(done["refusal"])  # generates from best docs anyway

    def test_decide_parse_error_falls_through_to_generation(self):
        llm = FakeLLM(decide=['{"tool": "teleport", "args": {}}'])
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        done = events[-1]
        self.assertFalse(done["refusal"])
        self.assertEqual(done["agent"]["decide_parse_errors"], 1)

    def test_decide_unparseable_falls_through(self):
        llm = FakeLLM(decide=["Let me just answer this directly."])
        pipe = make_pipe(llm)
        events = run_pipe(pipe)
        done = events[-1]
        self.assertFalse(done["refusal"])
        self.assertEqual(done["agent"]["decide_parse_errors"], 0)
        self.assertEqual(done["agent"]["actions"], [])

    def test_refusal_text_builder(self):
        text = build_refusal_text("no data on X", make_docs(2))
        self.assertTrue(text.startswith("INSUFFICIENT_EVIDENCE: no data on X"))
        self.assertIn("[1] Title 1", text)
        self.assertIn("[2] Title 2", text)


if __name__ == "__main__":
    unittest.main()
