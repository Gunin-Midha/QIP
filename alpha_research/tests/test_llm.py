import json
import sys
from types import SimpleNamespace

import pytest

from alpha_research.llm import (AnthropicLLMClient, GeminiLLMClient, LLMUnavailable,
                                ChatCompletionsLLMClient, ScriptedLLMClient, make_real_client,
                                provider_of, rate_limit_wait)


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
    # hosted ids belong to the model's maker, not to the host
    assert provider_of("qwen/qwen3-8b:free") == "qwen"
    assert provider_of("groq:qwen/qwen3.8-27b") == "qwen"
    assert provider_of("nvidia/nemotron-3.5-lightning:free") == "nvidia"
    assert provider_of("deepseek/deepseek-r1:free") == "deepseek"
    assert provider_of("google/gemini-x") == "google"
    assert provider_of("/x") is None and provider_of("qwen/") is None


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
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    assert make_real_client("qwen/qwen3-8b:free").host == "openrouter"
    assert make_real_client("google/gemini-x").host == "openrouter"
    assert make_real_client("groq:qwen/qwen3.8-27b").host == "groq"


# ---------------------------------------------------------------- hosted models (Groq, OpenRouter)


def hosted_reply(content, finish="stop"):
    return 200, json.dumps({"choices": [{"message": {"role": "assistant", "content": content},
                                         "finish_reason": finish}]}).encode()


@pytest.mark.parametrize("model, env_var", [("qwen/qwen3-8b:free", "OPENROUTER_API_KEY"),
                                            ("groq:qwen/qwen3.8-27b", "GROQ_API_KEY")])
def test_hosted_without_a_key_is_unavailable(monkeypatch, model, env_var):
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(LLMUnavailable, match=env_var):
        ChatCompletionsLLMClient(model)


@pytest.mark.parametrize("model, url, wire_model", [
    ("qwen/qwen3-8b:free", "https://openrouter.ai/api/v1/chat/completions", "qwen/qwen3-8b:free"),
    ("groq:qwen/qwen3.8-27b", "https://api.groq.com/openai/v1/chat/completions", "qwen/qwen3.8-27b"),
])
def test_hosted_call_shape(monkeypatch, model, url, wire_model):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    transport, seen = fake_transport(hosted_reply("hello"))
    llm = ChatCompletionsLLMClient(model, transport=transport)
    assert llm.complete("SYS", "USR", max_output=321) == "hello"
    assert seen["url"] == url
    assert seen["headers"]["Authorization"] == "Bearer dummy"
    assert seen["headers"]["User-Agent"].startswith("qip-")  # Cloudflare blocks urllib's default
    assert seen["body"] == {"model": wire_model,
                            "messages": [{"role": "system", "content": "SYS"},
                                         {"role": "user", "content": "USR"}],
                            "max_tokens": 321}
    assert llm.provider == "qwen" and llm.model == model


def test_hosted_strips_inline_thinking(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    reply = '<think>maybe {"draft": 1}</think>\n{"final": 2}'
    transport, _ = fake_transport(hosted_reply(reply))
    assert ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport) \
        .complete("s", "u").strip() == '{"final": 2}'
    transport, _ = fake_transport(hosted_reply("<think>ran out of room", finish="length"))
    with pytest.raises(LLMUnavailable, match="raise max_output"):
        ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport).complete("s", "u")


def test_hosted_rejects_an_id_without_a_maker():
    with pytest.raises(LLMUnavailable, match="not a hosted model id"):
        ChatCompletionsLLMClient("groq:llama", api_key="dummy")


@pytest.mark.parametrize("reply, match", [
    ((429, json.dumps({"error": {"message": "rate limited"}}).encode()), "HTTP 429: rate limited"),
    ((200, json.dumps({"error": {"message": "upstream down"}}).encode()), "upstream down"),
    ((502, b""), "HTTP 502: no message"),
    ((403, b"error code: 1010\n"), "HTTP 403: error code: 1010"),
    ((200, json.dumps({"choices": []}).encode()), "no choices"),
    (hosted_reply(None, finish="length"), "raise max_output"),
    (hosted_reply("  "), "empty reply"),
    (ConnectionError("down"), "down"),
])
def test_hosted_errors_are_unavailable(monkeypatch, reply, match):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    transport, _ = fake_transport(reply)
    with pytest.raises(LLMUnavailable, match=match):
        ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport).complete("s", "u")


# ---------------------------------------------------------------- rate limits: wait and retry


GROQ_429 = (429, json.dumps({"error": {"message": "Rate limit reached for model `qwen/qwen3.8-27b` "
                                       "on tokens per minute. Please try again in 7.954s."}}).encode())
GEMINI_429 = (429, json.dumps({"error": {"code": 429, "message": "Quota exceeded.", "details": [
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "43s"}]}}).encode())


def sequence_transport(*replies):
    calls = []

    def transport(url, body, headers, timeout):
        calls.append(url)
        return replies[min(len(calls), len(replies)) - 1]
    return transport, calls


@pytest.mark.parametrize("message, wait", [
    ("Please try again in 7.954s.", 7.954), ("try again in 450ms", 0.45),
    ("Please try again in 1m30.5s", 90.5), ("Please retry in 2h", 7200.0),
    ("Rate limit reached", None),
])
def test_rate_limit_wait_parses_the_hosts_hint(message, wait):
    got = rate_limit_wait({"error": {"message": message}}, b"")
    assert got == (pytest.approx(wait) if wait is not None else None)


def test_rate_limit_wait_reads_gemini_retry_delay_and_plain_text():
    assert rate_limit_wait(json.loads(GEMINI_429[1]), GEMINI_429[1]) == 43.0
    assert rate_limit_wait({}, b"slow down, try again in 3s") == 3.0


def test_hosted_waits_and_retries_on_a_short_rate_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    transport, calls = sequence_transport(GROQ_429, GROQ_429, hosted_reply("ok"))
    slept = []
    llm = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport, sleep=slept.append)
    assert llm.complete("s", "u") == "ok"
    assert len(calls) == 3 and slept == [pytest.approx(8.954), pytest.approx(8.954)]


def test_gemini_waits_and_retries_on_a_short_rate_limit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    transport, calls = sequence_transport(GEMINI_429, gemini_reply([{"text": "ok"}]))
    slept = []
    llm = GeminiLLMClient("gemini-x", transport=transport, sleep=slept.append)
    assert llm.complete("s", "u") == "ok" and slept == [44.0] and len(calls) == 2


@pytest.mark.parametrize("reply", [
    (429, json.dumps({"error": {"message": "Daily quota used. Please try again in 5h3m"}}).encode()),
    (429, json.dumps({"error": {"message": "Rate limit reached"}}).encode()),
])
def test_long_or_unknown_waits_fail_without_sleeping(monkeypatch, reply):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    transport, calls = sequence_transport(reply)
    slept = []
    llm = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport, sleep=slept.append)
    with pytest.raises(LLMUnavailable, match="HTTP 429"):
        llm.complete("s", "u")
    assert slept == [] and len(calls) == 1


def test_rate_limit_retries_run_out(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    transport, calls = sequence_transport(GROQ_429)
    slept = []
    llm = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport, sleep=slept.append)
    with pytest.raises(LLMUnavailable, match="try again in 7.954s"):
        llm.complete("s", "u")
    assert len(calls) == 4 and len(slept) == 3   # RATE_LIMIT_RETRIES = 3
