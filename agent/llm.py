"""LLM client for Ollama (OpenAI-compatible providers can be added later).

Streaming chat over Ollama's NDJSON `/api/chat` endpoint. Kept dependency-light
(httpx only) and hand-rolled so every wire detail is visible and testable.
"""

import json
from collections.abc import AsyncIterator

import httpx

from configs.loader import load_config


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


class LLMClient:
    """Async streaming chat client. One instance per app, created in lifespan."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMClient":
        llm = cfg["llm"]
        return cls(
            base_url=llm["base_url"],
            model=llm["model"],
            temperature=llm.get("temperature", 0.1),
            max_tokens=llm.get("max_tokens", 1024),
        )

    async def is_up(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Yield assistant content chunks for the given chat messages."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise OllamaError(
                        f"Ollama returned {resp.status_code}: {body.decode()[:200]}"
                    )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise OllamaError(event["error"])
                    delta = event.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if event.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Is it running? ({exc})"
            ) from exc

    async def close(self) -> None:
        await self._client.aclose()
