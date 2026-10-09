"""JSON logging and the circuit breaker."""

import json
import logging
import unittest

from tracing import CircuitBreaker, CircuitOpenError, JsonFormatter, trace_id_var


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def fail(breaker):
    try:
        with breaker:
            raise ValueError("dependency down")
    except ValueError:
        pass


class TestCircuitBreaker(unittest.TestCase):
    def test_opens_after_threshold_and_half_opens_after_cooldown(self):
        clock = FakeClock()
        b = CircuitBreaker("bm25", failure_threshold=2, cooldown_s=10, clock=clock)
        fail(b)
        self.assertEqual(b.state, "closed")
        fail(b)
        self.assertEqual(b.state, "open")
        with self.assertRaises(CircuitOpenError):
            with b:
                pass
        clock.now = 10
        self.assertEqual(b.state, "half_open")
        with b:  # trial succeeds
            pass
        self.assertEqual(b.state, "closed")

    def test_failed_trial_reopens(self):
        clock = FakeClock()
        b = CircuitBreaker("dense", failure_threshold=1, cooldown_s=5, clock=clock)
        fail(b)
        clock.now = 5
        fail(b)  # the half-open trial fails
        self.assertEqual(b.state, "open")
        clock.now = 9
        with self.assertRaises(CircuitOpenError):
            with b:
                pass

    def test_only_one_trial_in_flight(self):
        clock = FakeClock()
        b = CircuitBreaker("splade", failure_threshold=1, cooldown_s=1, clock=clock)
        fail(b)
        clock.now = 1
        b.__enter__()
        with self.assertRaises(CircuitOpenError):
            b.__enter__()
        b.__exit__(None, None, None)
        self.assertEqual(b.state, "closed")

    def test_success_resets_failure_count(self):
        b = CircuitBreaker("bm25", failure_threshold=2)
        fail(b)
        with b:
            pass
        fail(b)
        self.assertEqual(b.state, "closed")


class TestJsonFormatter(unittest.TestCase):
    def format(self, record):
        return json.loads(JsonFormatter().format(record))

    def make_record(self, msg="hello", exc_info=None, **extra):
        record = logging.LogRecord("agentic_rag", logging.WARNING, __file__, 1, msg, (), exc_info)
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_core_fields_extras_and_context_trace_id(self):
        token = trace_id_var.set("ctx-trace")
        try:
            entry = self.format(self.make_record(timings={"a": 1.0}, query_chars=12))
        finally:
            trace_id_var.reset(token)
        self.assertEqual(entry["level"], "WARNING")
        self.assertEqual(entry["logger"], "agentic_rag")
        self.assertEqual(entry["trace_id"], "ctx-trace")
        self.assertEqual(entry["timings"], {"a": 1.0})
        self.assertEqual(entry["query_chars"], 12)

    def test_explicit_trace_id_wins(self):
        token = trace_id_var.set("ctx-trace")
        try:
            entry = self.format(self.make_record(trace_id="explicit"))
        finally:
            trace_id_var.reset(token)
        self.assertEqual(entry["trace_id"], "explicit")

    def test_traceback_is_kept(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            entry = self.format(self.make_record(exc_info=sys.exc_info()))
        self.assertIn("ValueError: boom", entry["exc_info"])


if __name__ == "__main__":
    unittest.main()
