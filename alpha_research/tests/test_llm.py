import json
import sys
from types import SimpleNamespace

import pytest

from alpha_research.llm import (AnthropicLLMClient, GeminiLLMClient, LLMUnavailable,
                                ScriptedLLMClient, make_real_client, provider_of)


def test_scripted_client_replays_in_order_and_records_calls():
    llm = ScriptedLLMClient(["one", lambda system, user: f"{system}|{user}"])
    assert llm.model == "scripted"
    assert llm.complete("s1", "u1", max_output=100) == "one"
    assert llm.complete("s2", "u2") == "s2|u2"
    assert [c["user"] for c in llm.calls] == ["u1", "u2"]
    assert llm.calls[0]["max_output"] == 100


def test_scripted_client_raises_when_replies_run_out():
    llm = ScriptedLLMClient(["only"])
    llm.complete("s", "u")
    with pytest.raises(LLMUnavailable):
        llm.complete("s", "u")


def test_anthropic_client_without_a_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable):
        AnthropicLLMClient("some-model")


def test_anthropic_client_without_the_sdk_is_unavailable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` fail
    with pytest.raises(LLMUnavailable, match="not installed"):
        AnthropicLLMClient("some-model")


def _fake_sdk(monkeypatch, create):
    seen = {}

    class FakeAnthropic:
        def __init__(self, api_key):
            seen["key"] = api_key
            self.messages = SimpleNamespace(create=create)

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    return seen


def test_anthropic_client_call_shape_and_text_blocks(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    captured = {}

    def create(**kw):
        captured.update(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="hello "),
                                        SimpleNamespace(type="other"),
                                        SimpleNamespace(type="text", text="world")])

    seen = _fake_sdk(monkeypatch, create)
    llm = AnthropicLLMClient("model-x")
    assert llm.complete("SYS", "USR", max_output=321) == "hello world"
    assert seen["key"] == "dummy"
    assert captured == {"model": "model-x", "max_tokens": 321, "system": "SYS",
                        "messages": [{"role": "user", "content": "USR"}]}


def test_anthropic_client_wraps_provider_errors(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    def create(**kw):
        raise ConnectionError("boom")

    _fake_sdk(monkeypatch, create)
    with pytest.raises(LLMUnavailable, match="boom"):
        AnthropicLLMClient("m").complete("s", "u")


def test_anthropic_client_rejects_empty_reply(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    _fake_sdk(monkeypatch, lambda **kw: SimpleNamespace(content=[]))
    with pytest.raises(LLMUnavailable, match="empty"):
        AnthropicLLMClient("m").complete("s", "u")


# ---------------------------------------------------------------- providers and Gemini


def test_provider_of():
    assert provider_of("claude-opus-5-5") == "anthropic"
    assert provider_of("gemini-3.1-pro-preview") == "google"
    assert provider_of("scripted") == "scripted"
    assert provider_of("gpt-5") is None


def gemini_reply(parts, finish="STOP"):
    return 200, json.dumps({"candidates": [{"content": {"parts": parts},
                                            "finishReason": finish}]}).encode()


def fake_transport(reply):
    seen = {}

    def transport(url, body, headers, timeout):
        seen.update(url=url, body=json.loads(body), headers=headers, timeout=timeout)
        if isinstance(reply, Exception):
            raise reply
        return reply
    return transport, seen


def test_gemini_without_a_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
        GeminiLLMClient("gemini-x")


def test_gemini_call_shape_and_thought_parts_skipped(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    transport, seen = fake_transport(gemini_reply(
        [{"text": "thinking...", "thought": True}, {"text": "hello "}, {"text": "world"}]))
    llm = GeminiLLMClient("gemini-x", transport=transport)
    assert llm.complete("SYS", "USR", max_output=321) == "hello world"
    assert seen["url"].endswith("/models/gemini-x:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "dummy"
    assert seen["body"] == {"systemInstruction": {"parts": [{"text": "SYS"}]},
                            "contents": [{"role": "user", "parts": [{"text": "USR"}]}],
                            "generationConfig": {"maxOutputTokens": 321}}
    assert llm.provider == "google" and llm.model == "gemini-x"


@pytest.mark.parametrize("reply, match", [
    ((429, json.dumps({"error": {"message": "quota exceeded"}}).encode()), "HTTP 429: quota exceeded"),
    ((500, b"not json"), "HTTP 500: no message"),
    ((200, json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode()), "SAFETY"),
    (gemini_reply([{"text": "x", "thought": True}], finish="MAX_TOKENS"), "raise max_output"),
    (gemini_reply([]), "empty reply"),
    (ConnectionError("down"), "down"),
])
def test_gemini_errors_are_unavailable(monkeypatch, reply, match):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    transport, _ = fake_transport(reply)
    with pytest.raises(LLMUnavailable, match=match):
        GeminiLLMClient("gemini-x", transport=transport).complete("s", "u")


def test_make_real_client_picks_the_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    assert isinstance(make_real_client("gemini-x"), GeminiLLMClient)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        make_real_client("claude-x")
    with pytest.raises(LLMUnavailable, match="unknown provider"):
        make_real_client("gpt-5")
