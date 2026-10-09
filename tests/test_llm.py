"""LLMClient wire-protocol and retry semantics, against a mock transport."""

import json
import logging
import unittest
from unittest import mock

import httpx

from agent import llm as llm_module
from agent.llm import (
    CallSpec,
    LLMClient,
    OllamaError,
    StreamEnd,
    StreamInterrupted,
    Token,
    TruncatedOutput,
)


def enable_logging(test: unittest.TestCase) -> None:
    """Other test modules disable logging process-wide; undo that locally."""
    previous = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    test.addCleanup(logging.disable, previous)


def ndjson(*events) -> bytes:
    return b"".join(json.dumps(e).encode() + b"\n" for e in events)


def chunk(text):
    return {"message": {"content": text}, "done": False}


def done(reason="stop", prompt=50, completion=5):
    return {"done": True, "done_reason": reason, "prompt_eval_count": prompt, "eval_count": completion}


class _BrokenStream(httpx.AsyncByteStream):
    """Delivers some bytes, then the connection drops."""

    def __init__(self, first: bytes) -> None:
        self._first = first

    async def __aiter__(self):
        yield self._first
        raise httpx.ReadError("connection reset")


class TestLLMClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.object(llm_module, "_backoff", return_value=0)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.requests: list[dict] = []

    def client(self, responses, num_ctx=4096):
        responses = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content) if request.content else {})
            resp = responses.pop(0)
            return resp() if callable(resp) else resp

        c = LLMClient("http://ollama", "m", num_ctx=num_ctx, transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(c.close)
        return c

    async def collect(self, client, spec=CallSpec("answer")):
        return [ev async for ev in client.stream([{"role": "user", "content": "q"}], spec)]

    async def test_stream_yields_tokens_then_end_and_sends_options(self):
        c = self.client([httpx.Response(200, content=ndjson(chunk("Hel"), chunk("lo"), done()))])
        events = await self.collect(c, CallSpec("answer", temperature=0.3, max_tokens=77))
        self.assertEqual(events, [Token("Hel"), Token("lo"), StreamEnd("stop", 50, 5)])
        sent = self.requests[0]
        self.assertEqual(sent["options"], {"temperature": 0.3, "num_predict": 77, "num_ctx": 4096})
        self.assertTrue(sent["stream"])

    async def test_stream_retries_before_first_token(self):
        c = self.client([
            httpx.Response(503, content=b"loading"),
            httpx.Response(200, content=ndjson(chunk("ok"), done())),
        ])
        events = await self.collect(c)
        self.assertEqual(events[0], Token("ok"))
        self.assertEqual(len(self.requests), 2)

    async def test_stream_does_not_retry_client_errors(self):
        c = self.client([httpx.Response(404, content=b"model not found")])
        with self.assertRaises(OllamaError) as cm:
            await self.collect(c)
        self.assertFalse(cm.exception.retryable)
        self.assertEqual(len(self.requests), 1)

    async def test_stream_failure_after_tokens_raises_interrupted(self):
        c = self.client([
            lambda: httpx.Response(200, stream=_BrokenStream(ndjson(chunk("partial")))),
        ])
        with self.assertRaises(StreamInterrupted):
            await self.collect(c)
        self.assertEqual(len(self.requests), 1)  # never replayed

    async def test_stream_closed_without_done_is_an_error(self):
        c = self.client([httpx.Response(200, content=ndjson(chunk("x")))] * 3)
        with self.assertRaises(StreamInterrupted):
            await self.collect(c)

    async def test_in_stream_error_event(self):
        c = self.client([httpx.Response(200, content=ndjson({"error": "out of memory"}))] * 3)
        with self.assertRaises(OllamaError):
            await self.collect(c)
        self.assertEqual(len(self.requests), 3)

    async def test_complete_retries_then_succeeds(self):
        c = self.client([
            httpx.Response(500, content=b"boom"),
            httpx.Response(200, content=ndjson(chunk('{"a": 1}'), done())),
        ])
        result = await c.complete([], CallSpec("decide"))
        self.assertEqual(result.text, '{"a": 1}')
        self.assertEqual(result.attempts, 2)

    async def test_complete_gives_up_after_max_attempts(self):
        c = self.client([httpx.Response(500, content=b"boom")] * 2)
        with self.assertRaises(OllamaError):
            await c.complete([], CallSpec("decide", max_attempts=2))

    async def test_complete_on_length_policies(self):
        cut = lambda: httpx.Response(200, content=ndjson(chunk("{"), done(reason="length")))
        whole = lambda: httpx.Response(200, content=ndjson(chunk("{}"), done()))

        c = self.client([cut])
        self.assertEqual((await c.complete([], CallSpec("x", on_length="accept"))).finish_reason, "length")

        c = self.client([cut])
        with self.assertRaises(TruncatedOutput):
            await c.complete([], CallSpec("x", on_length="fail"))

        self.requests.clear()
        c = self.client([cut, whole])
        result = await c.complete([], CallSpec("x", max_tokens=100, on_length="escalate"))
        self.assertEqual(result.text, "{}")
        self.assertEqual([r["options"]["num_predict"] for r in self.requests], [100, 200])

    async def test_prompt_near_num_ctx_logs_warning(self):
        enable_logging(self)
        c = self.client(
            [httpx.Response(200, content=ndjson(chunk("x"), done(prompt=1000)))], num_ctx=1024
        )
        with self.assertLogs("agent.llm", level="WARNING") as logs:
            await self.collect(c)
        self.assertIn("num_ctx", logs.output[0])

    async def test_is_up(self):
        c = self.client([httpx.Response(200, json={"models": []})])
        self.assertTrue(await c.is_up())

        def refuse(request):
            raise httpx.ConnectError("refused")

        down = LLMClient("http://ollama", "m", transport=httpx.MockTransport(refuse))
        self.addAsyncCleanup(down.close)
        self.assertFalse(await down.is_up())


if __name__ == "__main__":
    unittest.main()
