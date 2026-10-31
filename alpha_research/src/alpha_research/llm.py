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
"""

from __future__ import annotations

import os
from typing import Callable, Protocol, Sequence


KEY_ENV_VAR = "ANTHROPIC_API_KEY"


class LLMUnavailable(RuntimeError):
    """No usable model: missing SDK, missing key, no scripted reply left, or a failed call."""


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str: ...


class ScriptedLLMClient:
    """Returns the next canned reply on every call. Each reply is a string or a function
    ``(system, user) -> str``. Every call is recorded in ``calls``."""

    model = "scripted"

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


def make_real_client(model: str) -> LLMClient:
    """The real model client. Kept here so no other module has to name the provider."""
    return AnthropicLLMClient(model)
