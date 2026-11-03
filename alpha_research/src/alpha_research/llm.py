"""LLM client abstraction (Phase 3).

This is the only module that may talk to a model provider. Everything else depends on the
small ``LLMClient`` protocol, so the generator can be tested with ``ScriptedLLMClient``
and never needs a network.

* ``ScriptedLLMClient`` replays canned replies. Used by the tests and by the offline
  demo (``cli generate --llm scripted --llm-script replies.json``).
* ``AnthropicLLMClient`` calls the Anthropic Messages API. The SDK is an optional
  dependency (``pip install -e ".[llm]"``) and is imported only when the client is built.
  The key is read from the KEY_ENV_VAR environment variable and is never stored,
  logged or written to disk.
* ``GeminiLLMClient`` calls Google's Gemini ``generateContent`` REST endpoint with the
  standard library (``urllib``), so it needs no extra package. The key is read from
  GEMINI_KEY_ENV_VAR and sent only in the ``x-goog-api-key`` header.

The provider is inferred from the model id (``provider_of``): ``claude-*`` is Anthropic,
``gemini-*`` is Google. Agents compare providers by this name, never by client class.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Protocol, Sequence


KEY_ENV_VAR = "ANTHROPIC_API_KEY"
GEMINI_KEY_ENV_VAR = "GEMINI_API_KEY"
GEMINI_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TIMEOUT_SECONDS = 300.0   # thinking models can take minutes on a long prompt

PROVIDER_PREFIXES = {"claude-": "anthropic", "gemini-": "google"}


def provider_of(model: str) -> str | None:
    """Provider for a model id, from its prefix. None when the prefix is unknown."""
    if model == "scripted":
        return "scripted"
    for prefix, provider in PROVIDER_PREFIXES.items():
        if isinstance(model, str) and model.startswith(prefix):
            return provider
    return None


class LLMUnavailable(RuntimeError):
    """No usable model: missing SDK, missing key, no scripted reply left, or a failed call."""


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str: ...


class ScriptedLLMClient:
    """Returns the next canned reply on every call. Each reply is a string or a function
    ``(system, user) -> str``. Every call is recorded in ``calls``."""

    model = "scripted"
    provider = "scripted"

    def __init__(self, replies: Sequence[str | Callable[[str, str], str]]):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        self.calls.append({"system": system, "user": user, "max_output": max_output})
        if not self._replies:
            raise LLMUnavailable("scripted client has no more replies")
        reply = self._replies.pop(0)
        return reply(system, user) if callable(reply) else reply


class AnthropicLLMClient:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None = None):
        key = api_key or os.environ.get(KEY_ENV_VAR)
        if not key:
            raise LLMUnavailable(f"{KEY_ENV_VAR} is not set; export it or use --llm scripted")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable(
                "the 'anthropic' package is not installed; run: pip install -e \".[llm]\"") from exc
        self.model = model
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_output, system=system,
                messages=[{"role": "user", "content": user}])
        except Exception as exc:  # provider errors (rate limit, auth, network) -> clean exit
            raise LLMUnavailable(f"model call failed: {type(exc).__name__}: {exc}") from exc
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise LLMUnavailable("model returned an empty reply")
        return text


# transport(url, body, headers, timeout) -> (status, response body); tests pass a fake
Transport = Callable[[str, bytes, dict, float], tuple[int, bytes]]


def _urllib_transport(url: str, body: bytes, headers: dict, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class GeminiLLMClient:
    provider = "google"

    def __init__(self, model: str, api_key: str | None = None, *,
                 transport: Transport | None = None, timeout: float = GEMINI_TIMEOUT_SECONDS):
        key = api_key or os.environ.get(GEMINI_KEY_ENV_VAR)
        if not key:
            raise LLMUnavailable(
                f"{GEMINI_KEY_ENV_VAR} is not set; export it or use --llm scripted")
        self.model = model
        self._key = key
        self._transport = transport or _urllib_transport
        self._timeout = timeout

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        url = f"{GEMINI_ROOT}/models/{urllib.parse.quote(self.model)}:generateContent"
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_output},
        }).encode()
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._key}
        try:
            status, raw = self._transport(url, body, headers, self._timeout)
        except (OSError, TimeoutError) as exc:  # network failure -> clean exit
            raise LLMUnavailable(f"model call failed: {type(exc).__name__}: {exc}") from exc
        try:
            data = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {}
        if status != 200:
            msg = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
            raise LLMUnavailable(f"model call failed: HTTP {status}: {msg or 'no message'}")
        candidates = data.get("candidates") if isinstance(data, dict) else None
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMUnavailable(f"model returned no candidates (block reason: {block})")
        first = candidates[0]
        parts = (first.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts
                       if isinstance(p, dict) and not p.get("thought"))
        if not text.strip():
            finish = first.get("finishReason")
            hint = ("; thinking used the whole output cap, raise max_output in "
                    "config/agents.yaml" if finish == "MAX_TOKENS" else "")
            raise LLMUnavailable(f"model returned an empty reply (finishReason {finish}){hint}")
        return text


def make_real_client(model: str) -> LLMClient:
    """The real model client for a model id. Kept here so no other module has to name a
    provider."""
    provider = provider_of(model)
    if provider == "anthropic":
        return AnthropicLLMClient(model)
    if provider == "google":
        return GeminiLLMClient(model)
    raise LLMUnavailable(f"unknown provider for model {model!r}: expected a name starting "
                         f"with one of {sorted(PROVIDER_PREFIXES)}")
