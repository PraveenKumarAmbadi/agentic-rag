#!/usr/bin/env python3
"""End-to-end smoke check of a running API (standard library only).

Checks that:
  1. /health reports the service ready (status ok or degraded);
  2. /answer rejects a request without a valid API key (401);
  3. /answer streams a well-formed answer: deltas, then exactly one `done`
     event, no `error` event, and the deltas add up to a non-empty text.

Usage::
    python scripts/smoke_api.py --key "$AGENTIC_RAG_API_KEYS" [--url http://127.0.0.1:8001]

Exit code 0 on success, 1 on the first failed check.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

QUESTION = "Does vitamin D supplementation reduce the risk of fractures?"


def _request(url: str, body: dict | None = None, key: str | None = None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("X-API-Key", key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--key", required=True)
    parser.add_argument("--timeout", type=float, default=900, help="seconds for the answer")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    status, body = _request(f"{base}/health")
    health = json.loads(body) if body.startswith("{") else {}
    check(status == 200 and health.get("status") in ("ok", "degraded"), f"/health -> {status} {body}")

    status, _ = _request(f"{base}/answer", {"question": QUESTION}, key="wrong-" + "x" * 40)
    check(status == 401, f"/answer with a wrong key -> {status} (want 401)")

    status, body = _request(f"{base}/answer", {"question": QUESTION}, key=args.key, timeout=args.timeout)
    check(status == 200, f"/answer -> {status}")
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    errors = [e for e in events if e.get("type") == "error"]
    check(not errors, f"no error events {errors}")
    done = [e for e in events if e.get("type") == "done"]
    check(len(done) == 1 and events[-1] is done[0], "exactly one done event, last")
    text = ""
    for e in events:
        if e.get("type") == "reset":
            text = ""
        elif "delta" in e:
            text += e["delta"]
    check(bool(text.strip()), f"answer text streamed ({len(text)} chars)")
    d = done[0]
    print(
        f"refusal={d['refusal']} citations={d['citations']} degraded={d['degraded']} "
        f"total_ms={d['timings']['total_ms']} llm_calls={d['usage']['llm_calls']}"
    )
    print(f"answer: {text.strip()[:400]}")


if __name__ == "__main__":
    main()
