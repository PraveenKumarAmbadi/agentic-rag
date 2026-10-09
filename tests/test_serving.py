"""HTTP-layer tests: auth, validation, SSE contract, errors, admission, health.

The app is built through create_app() with a fake services factory, so no
models, Elasticsearch, or Ollama are involved.
"""

import asyncio
import json
import logging
import unittest

from fastapi.testclient import TestClient

from agent.llm import OllamaError
from configs.loader import validate_config
from serving.auth import API_KEY_HEADER, ApiKeyAuth, AuthConfigError
from serving.limits import CachedProbe, ConcurrencyLimiter
from serving.main import Services, create_app

logging.disable(logging.CRITICAL)

KEY = "k" * 40
AUTH = {API_KEY_HEADER: KEY}


class FakeES:
    def __init__(self, up=True):
        self.up = up

    def options(self, **kwargs):
        return self

    def ping(self):
        return self.up

    def close(self):
        pass


class FakePipeline:
    def __init__(self, events=None, error=None, delay=0.0, es_up=True, breakers=None):
        self.events = events if events is not None else [
            {"type": "step", "step": "retrieve", "n_docs": 3},
            {"type": "delta", "text": "Hello "},
            {"type": "delta", "text": "world [1]."},
            {
                "type": "done", "trace_id": "t", "text": "Hello world [1].", "refusal": False,
                "citations": {"1": "d1"}, "timings": {"total_ms": 1.0},
                "usage": {"llm_calls": 1}, "degraded": [],
            },
        ]
        self.error = error
        self.delay = delay
        self.es = FakeES(es_up)
        self._breakers = breakers or {"bm25": "closed", "splade": "closed", "dense": "closed"}
        self.calls = []

    def breaker_states(self):
        return self._breakers

    async def answer_stream(self, question, trace_id=None, deadline=None):
        self.calls.append({"question": question, "trace_id": trace_id, "deadline": deadline})
        for ev in self.events:
            if ev["type"] == "done" and self.error:
                raise self.error
            if self.delay:
                await asyncio.sleep(self.delay)
            yield ev


class FakeLLMClient:
    def __init__(self, up=True):
        self.up = up
        self.closed = False

    async def is_up(self, timeout_s=2.0):
        return self.up

    async def close(self):
        self.closed = True


def make_client(pipeline=None, llm_up=True, auth=True, serving=None, agent=None, hold_slots=0):
    cfg = validate_config({"serving": serving or {}, "agent": agent or {}})
    llm = FakeLLMClient(llm_up)
    pipeline = pipeline or FakePipeline()

    async def factory():
        s = cfg["serving"]
        limiter = ConcurrencyLimiter(s["max_concurrent"], s["max_queue"], s["queue_timeout_s"])
        for _ in range(hold_slots):
            await limiter.acquire()
        return Services(
            cfg=cfg,
            pipeline=pipeline,
            llm=llm,
            auth=ApiKeyAuth([KEY]) if auth else None,
            limiter=limiter,
            llm_probe=CachedProbe(llm.is_up, 0.0),
        )

    return TestClient(create_app(factory)), pipeline, llm


def sse_events(response):
    return [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


class TestAuth(unittest.TestCase):
    def test_missing_and_wrong_key_rejected(self):
        client, pipeline, _ = make_client()
        with client:
            for headers in ({}, {API_KEY_HEADER: "x" * 40}):
                r = client.post("/answer", json={"question": "q"}, headers=headers)
                self.assertEqual(r.status_code, 401)
                self.assertEqual(r.headers["WWW-Authenticate"], API_KEY_HEADER)
        self.assertEqual(pipeline.calls, [])

    def test_valid_key_accepted(self):
        client, _, _ = make_client()
        with client:
            r = client.post("/answer", json={"question": "q"}, headers=AUTH)
        self.assertEqual(r.status_code, 200)

    def test_auth_disabled(self):
        client, _, _ = make_client(auth=False)
        with client:
            self.assertEqual(client.post("/answer", json={"question": "q"}).status_code, 200)

    def test_health_is_open(self):
        client, _, _ = make_client()
        with client:
            self.assertEqual(client.get("/health").status_code, 200)

    def test_from_env_fails_closed(self):
        with self.assertRaises(AuthConfigError):
            ApiKeyAuth.from_env(True, environ={})
        with self.assertRaises(AuthConfigError):
            ApiKeyAuth.from_env(True, environ={"AGENTIC_RAG_API_KEYS": "short"})
        self.assertIsNone(ApiKeyAuth.from_env(False, environ={}))

    def test_key_rotation_accepts_any_configured_key(self):
        other = "o" * 40
        auth = ApiKeyAuth.from_env(True, environ={"AGENTIC_RAG_API_KEYS": f"{KEY}, {other}"})
        self.assertTrue(auth.verify(KEY))
        self.assertTrue(auth.verify(other))
        self.assertFalse(auth.verify("z" * 40))
        self.assertFalse(auth.verify(None))


class TestValidation(unittest.TestCase):
    def assert_422(self, body, **kwargs):
        client, pipeline, _ = make_client(**kwargs)
        with client:
            r = client.post("/answer", json=body, headers=AUTH)
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(pipeline.calls, [])

    def test_empty_and_blank_questions(self):
        self.assert_422({"question": ""})
        self.assert_422({"question": "   \n "})

    def test_too_long_question(self):
        self.assert_422({"question": "x" * 11}, serving={"max_question_chars": 10})

    def test_control_characters(self):
        self.assert_422({"question": "what\x00is this"})

    def test_unknown_fields_and_missing_question(self):
        self.assert_422({"question": "q", "extra": 1})
        self.assert_422({})

    def test_question_is_stripped(self):
        client, pipeline, _ = make_client()
        with client:
            client.post("/answer", json={"question": "  what is X?  "}, headers=AUTH)
        self.assertEqual(pipeline.calls[0]["question"], "what is X?")

    def test_oversized_body_rejected(self):
        client, _, _ = make_client()
        with client:
            r = client.post("/answer", content=b"{" + b" " * 70_000 + b"}",
                            headers={**AUTH, "content-type": "application/json"})
        self.assertEqual(r.status_code, 413)


class TestStreaming(unittest.TestCase):
    def test_sse_contract(self):
        client, pipeline, _ = make_client()
        with client:
            r = client.post("/answer", json={"question": "q"}, headers=AUTH)
        self.assertEqual(r.headers["content-type"].split(";")[0], "text/event-stream")
        self.assertEqual(r.headers["cache-control"], "no-cache")
        events = sse_events(r)
        self.assertEqual(events[0]["type"], "step")
        self.assertEqual("".join(e["delta"] for e in events if "delta" in e), "Hello world [1].")
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertNotIn("text", done)  # the deltas are the text
        self.assertIsNotNone(pipeline.calls[0]["deadline"])

    def test_request_id_propagates(self):
        client, pipeline, _ = make_client()
        with client:
            r = client.post("/answer", json={"question": "q"}, headers={**AUTH, "X-Request-ID": "abc-123"})
            self.assertEqual(r.headers["X-Request-ID"], "abc-123")
            self.assertEqual(pipeline.calls[0]["trace_id"], "abc-123")
            r = client.post("/answer", json={"question": "q"},
                            headers={**AUTH, "X-Request-ID": "bad id\n<script>"})
            self.assertNotEqual(r.headers["X-Request-ID"], "bad id\n<script>")

    def test_failure_mid_stream_becomes_error_event(self):
        for exc, code in [
            (OllamaError("down"), "llm_unavailable"),
            (RuntimeError("bug"), "internal"),
        ]:
            with self.subTest(code=code):
                client, _, _ = make_client(pipeline=FakePipeline(error=exc))
                with client:
                    r = client.post("/answer", json={"question": "q"},
                                    headers={**AUTH, "X-Request-ID": "trace-1"})
                events = sse_events(r)
                self.assertEqual(events[-1]["type"], "error")
                self.assertEqual(events[-1]["code"], code)
                self.assertEqual(events[-1]["trace_id"], "trace-1")
                self.assertNotIn("bug", events[-1]["message"])  # no internals leak

    def test_hard_timeout(self):
        client, _, _ = make_client(
            pipeline=FakePipeline(delay=0.3),
            serving={"hard_timeout_s": 0.2},
            agent={"request_timeout_s": 0.1},
        )
        with client:
            r = client.post("/answer", json={"question": "q"}, headers=AUTH)
        self.assertEqual(sse_events(r)[-1]["code"], "timeout")


class TestAdmission(unittest.TestCase):
    def test_llm_down_returns_503_before_streaming(self):
        client, pipeline, _ = make_client(llm_up=False)
        with client:
            r = client.post("/answer", json={"question": "q"}, headers=AUTH)
        self.assertEqual(r.status_code, 503)
        self.assertIn("Retry-After", r.headers)
        self.assertEqual(pipeline.calls, [])

    def test_saturated_returns_503(self):
        client, pipeline, _ = make_client(serving={"max_concurrent": 1, "max_queue": 0}, hold_slots=1)
        with client:
            r = client.post("/answer", json={"question": "q"}, headers=AUTH)
        self.assertEqual(r.status_code, 503)
        self.assertEqual(pipeline.calls, [])

    def test_queue_wait_times_out(self):
        client, _, _ = make_client(
            serving={"max_concurrent": 1, "max_queue": 1, "queue_timeout_s": 0.1}, hold_slots=1
        )
        with client:
            r = client.post("/answer", json={"question": "q"}, headers=AUTH)
        self.assertEqual(r.status_code, 503)

    def test_slot_released_after_success_and_failure(self):
        for pipeline in (FakePipeline(), FakePipeline(error=RuntimeError("x"))):
            client, _, _ = make_client(pipeline=pipeline)
            with client:
                client.post("/answer", json={"question": "q"}, headers=AUTH)
                load = client.get("/health").json()["load"]
            self.assertEqual(load["in_flight"], 0)


class TestHealth(unittest.TestCase):
    def check(self, expected_status, expected_code, **kwargs):
        client, _, _ = make_client(**kwargs)
        with client:
            r = client.get("/health")
        self.assertEqual(r.status_code, expected_code)
        self.assertEqual(r.json()["status"], expected_status)

    def test_ok(self):
        self.check("ok", 200)

    def test_llm_down_is_down(self):
        self.check("down", 503, llm_up=False)

    def test_es_down_is_degraded(self):
        self.check("degraded", 200, pipeline=FakePipeline(es_up=False))

    def test_open_breaker_is_degraded(self):
        breakers = {"bm25": "open", "splade": "closed", "dense": "closed"}
        self.check("degraded", 200, pipeline=FakePipeline(breakers=breakers))

    def test_all_retrievers_open_is_down(self):
        breakers = {"bm25": "open", "splade": "open", "dense": "open"}
        self.check("down", 503, pipeline=FakePipeline(breakers=breakers))


class TestLifespan(unittest.TestCase):
    def test_services_closed_on_shutdown(self):
        client, _, llm = make_client()
        with client:
            pass
        self.assertTrue(llm.closed)


if __name__ == "__main__":
    unittest.main()
