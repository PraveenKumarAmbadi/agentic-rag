"""API-key authentication.

Keys come from the AGENTIC_RAG_API_KEYS environment variable
(comma-separated, so a key can be rotated by deploying old+new, then
removing the old one). They are never stored in config files or logs; only
their SHA-256 digests are kept in memory and compared in constant time.

Fail closed: if auth is enabled and no valid key is configured, the
service refuses to start rather than silently running open.
"""

import hashlib
import hmac
import os
from collections.abc import Iterable, Mapping

API_KEYS_ENV = "AGENTIC_RAG_API_KEYS"
API_KEY_HEADER = "X-API-Key"
MIN_KEY_LENGTH = 32


class AuthConfigError(RuntimeError):
    """Auth is enabled but not correctly configured."""


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8")).digest()


class ApiKeyAuth:
    def __init__(self, keys: Iterable[str]) -> None:
        keys = list(keys)
        if not keys:
            raise AuthConfigError("at least one API key is required")
        short = [k for k in keys if len(k) < MIN_KEY_LENGTH]
        if short:
            raise AuthConfigError(
                f"API keys must be at least {MIN_KEY_LENGTH} characters "
                f"({len(short)} key(s) too short)"
            )
        self._digests = [_digest(k) for k in keys]

    @classmethod
    def from_env(cls, enabled: bool, environ: Mapping[str, str] | None = None) -> "ApiKeyAuth | None":
        """Build from the environment; None when auth is disabled."""
        if not enabled:
            return None
        environ = os.environ if environ is None else environ
        keys = [k.strip() for k in environ.get(API_KEYS_ENV, "").split(",") if k.strip()]
        if not keys:
            raise AuthConfigError(
                f"serving.auth_enabled is true but {API_KEYS_ENV} is empty. "
                "Generate a key with: python -c \"import secrets; "
                "print(secrets.token_urlsafe(32))\" and export it, or set "
                "serving.auth_enabled: false for local experiments."
            )
        return cls(keys)

    def verify(self, presented: str | None) -> bool:
        if not presented:
            return False
        candidate = _digest(presented)
        # Compare against every key so timing does not reveal which matched.
        matched = False
        for ref in self._digests:
            matched |= hmac.compare_digest(candidate, ref)
        return matched
