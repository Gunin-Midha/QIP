import sys
from types import SimpleNamespace

import pytest

from alpha_research.llm import AnthropicLLMClient, LLMUnavailable, ScriptedLLMClient


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
