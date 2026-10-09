"""Eval runner math, scoring, and failure handling (no models needed)."""

import asyncio
import io
import json
import logging
import unittest

from evaluation.evaluate_rag import EvalAborted, eval_set, percentile, score_row, summarize

logging.disable(logging.CRITICAL)


def done_event(refusal=False, citations=None, retrieved=("g1", "x1"), total_ms=100.0):
    return {
        "type": "done",
        "text": "INSUFFICIENT_EVIDENCE: ... [1]" if refusal else "answer [1]",
        "refusal": refusal,
        "citations": citations if citations is not None else {"1": "g1"},
        "invalid_citations": [],
        "retrieved_doc_ids": list(retrieved),
        "timings": {"total_ms": total_ms},
        "usage": {"llm_calls": 1, "llm_errors": 0, "prompt_tokens": 10, "completion_tokens": 5},
        "degraded": [],
        "truncated": False,
    }


class ScriptedPipe:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    async def answer_stream(self, question, trace_id=None, deadline=None):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        yield outcome


REC = {"query_id": "q1", "question": "Q?", "gold_doc_ids": ["g1"]}


class TestPercentile(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(percentile([], 50), 0.0)
        self.assertEqual(percentile([7.0], 95), 7.0)
        self.assertEqual(percentile([1.0, 2.0], 95), 2.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 50), 2.0)
        ten = [float(i) for i in range(1, 11)]
        self.assertEqual(percentile(ten, 95), 10.0)
        self.assertEqual(percentile(ten, 50), 5.0)
        hundred = [float(i) for i in range(1, 101)]
        self.assertEqual(percentile(hundred, 95), 95.0)


class TestScoring(unittest.TestCase):
    def test_answer_citation_precision(self):
        row = score_row(REC, done_event(citations={"1": "g1", "2": "x1"}))
        self.assertEqual(row["citation_precision"], 0.5)
        self.assertEqual(row["gold_doc_ids_retrieved"], ["g1"])

    def test_refusal_listing_is_not_counted_as_citations(self):
        row = score_row(REC, done_event(refusal=True, citations={"1": "g1"}))
        self.assertEqual(row["cited_doc_ids"], [])
        self.assertEqual(row["citation_precision"], 0.0)
        self.assertEqual(row["refusal_listed_doc_ids"], ["g1"])


class TestEvalSet(unittest.TestCase):
    def test_errors_are_recorded_and_excluded(self):
        pipe = ScriptedPipe([done_event(), RuntimeError("ollama hiccup"), done_event(refusal=True)])
        checkpoint = io.StringIO()
        rows = asyncio.run(eval_set(pipe, [REC] * 3, "answerable", checkpoint))
        self.assertIn("error", rows[1])
        lines = [json.loads(line) for line in checkpoint.getvalue().splitlines()]
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0]["set"], "answerable")

        summary = summarize(rows, has_gold=True)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["n_errors"], 1)
        self.assertEqual(summary["false_refusal_rate"], 0.5)
        self.assertEqual(summary["citation_precision"], 0.5)

    def test_consecutive_failures_abort(self):
        pipe = ScriptedPipe([RuntimeError("down")] * 3)
        with self.assertRaises(EvalAborted):
            asyncio.run(eval_set(pipe, [REC] * 3, "answerable", max_consecutive_errors=3))

    def test_agent_metrics(self):
        ev = done_event()
        ev["agent"] = {
            "iterations": 2, "llm_calls": 4, "hedged": False,
            "rescue": {"overridden": True}, "critique": {"supported": None, "skipped": "error"},
        }
        rows = asyncio.run(eval_set(ScriptedPipe([ev]), [REC]))
        s = summarize(rows, has_gold=True)
        self.assertEqual(s["rescue_override_rate"], 1.0)
        self.assertEqual(s["critique_pass_rate"], 0.0)  # a skipped critique is not a pass


if __name__ == "__main__":
    unittest.main()
