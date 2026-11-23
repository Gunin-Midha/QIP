import json
import sys
from types import SimpleNamespace

import pytest

from alpha_research.llm import (AnthropicLLMClient, GeminiLLMClient, LLMUnavailable,
                                ChatCompletionsLLMClient, ScriptedLLMClient, make_real_client,
                                list_models, provider_of, rate_limit_wait, split_hosted)


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
    (gemini_reply([{"text": "x", "thought": True}], finish="MAX_TOKENS"), "models.max_output"),
    (gemini_reply([]), "empty reply"),
    (ConnectionError("down"), "down"),
])
def test_gemini_errors_are_unavailable(monkeypatch, reply, match):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    transport, _ = fake_transport(reply)
    with pytest.raises(LLMUnavailable, match=match):
        GeminiLLMClient("gemini-x", transport=transport, sleep=lambda s: None).complete("s", "u")


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
    with pytest.raises(LLMUnavailable, match="models.max_output"):
        ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport).complete("s", "u")


def test_hosted_rejects_an_id_without_a_maker():
    with pytest.raises(LLMUnavailable, match="not a hosted model id"):
        ChatCompletionsLLMClient("groq:mystery-model", api_key="dummy")


def test_new_hosts_and_bare_ids():
    assert split_hosted("groq:gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")
    assert split_hosted("cerebras:gpt-oss-120b") == ("cerebras", "gpt-oss-120b")  # as is
    assert split_hosted("cerebras:qwen-3-32b") == ("cerebras", "qwen-3-32b")
    assert provider_of("cerebras:gpt-oss-120b") == "openai"
    assert provider_of("cerebras:llama-3.3-70b") == "meta"
    assert split_hosted("groq:llama-3.3-70b-versatile") == ("groq", "llama-3.3-70b-versatile")
    assert split_hosted("nvidia:meta/llama-3.3-70b-instruct") == \
        ("nvidia", "meta/llama-3.3-70b-instruct")
    assert split_hosted("gpt-oss-120b") is None          # a bare name needs a host
    assert provider_of("groq:gpt-oss-120b") == "openai"
    assert provider_of("nvidia:meta/llama-3.3-70b-instruct") == "meta"
    assert provider_of("groq:llama-3.3-70b-versatile") == "meta"
    assert provider_of("nvidia:deepseek-ai/deepseek-v3.2") == "deepseek"
    assert provider_of("meta-llama/llama-4-scout:free") == "meta"
    assert provider_of("gpt-5") is None


def test_nvidia_client_sends_the_namespaced_id():
    seen = {}

    def transport(url, body, headers, timeout):
        seen.update(url=url, body=json.loads(body), auth=headers["Authorization"])
        return 200, json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    c = ChatCompletionsLLMClient("nvidia:gpt-oss-120b", api_key="k", transport=transport)
    assert c.complete("s", "u") == "ok"
    assert seen["url"].startswith("https://integrate.api.nvidia.com/")
    assert seen["body"]["model"] == "openai/gpt-oss-120b" and seen["auth"] == "Bearer k"


def test_list_models():
    seen = {}

    def transport(url, body, headers, timeout):
        seen.update(url=url, body=body)
        return 200, json.dumps({"data": [{"id": "b/x"}, {"id": "a/y"}, {"nope": 1}]}).encode()

    assert list_models("nvidia", api_key="k", transport=transport) == ["a/y", "b/x"]
    assert seen == {"url": "https://integrate.api.nvidia.com/v1/models", "body": None}
    seen.clear()
    plain = lambda url, body, headers, timeout: (seen.update(url=url) or 200,
                                                 json.dumps([{"id": "gpt-x"}]).encode())
    assert list_models("cerebras", api_key="k", transport=plain) == ["gpt-x"]  # bare list
    assert seen["url"] == "https://api.cerebras.ai/v1/models"
    with pytest.raises(LLMUnavailable, match="HTTP 401"):
        list_models("groq", api_key="k", transport=lambda *a: (401, b"no"))
    with pytest.raises(LLMUnavailable, match="unknown host"):
        list_models("mars", api_key="k")


@pytest.mark.parametrize("reply, match", [
    ((429, json.dumps({"error": {"message": "rate limited"}}).encode()), "HTTP 429: rate limited"),
    ((200, json.dumps({"error": {"message": "upstream down"}}).encode()), "upstream down"),
    ((502, b""), "HTTP 502: no message"),
    ((403, b"error code: 1010\n"), "HTTP 403: error code: 1010"),
    ((200, json.dumps({"choices": []}).encode()), "no choices"),
    (hosted_reply(None, finish="length"), "models.max_output"),
    (hosted_reply("  "), "empty reply"),
    (ConnectionError("down"), "down"),
])
def test_hosted_errors_are_unavailable(monkeypatch, reply, match):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    transport, _ = fake_transport(reply)
    with pytest.raises(LLMUnavailable, match=match):
        ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport,
                                 sleep=lambda s: None).complete("s", "u")


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


def test_an_unstated_wait_gets_one_retry_after_20s(monkeypatch):
    # OpenRouter's "Provider returned error" (the free host is busy) names no wait
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    busy = (429, json.dumps({"error": {"message": "Provider returned error"}}).encode())
    transport, calls = sequence_transport(busy, hosted_reply("ok"))
    slept = []
    llm = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport, sleep=slept.append)
    assert llm.complete("s", "u") == "ok" and slept == [20.0] and len(calls) == 2
    transport, calls = sequence_transport(busy)
    slept = []
    llm = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport, sleep=slept.append)
    with pytest.raises(LLMUnavailable, match="Provider returned error"):
        llm.complete("s", "u")
    assert slept == [20.0] and len(calls) == 2           # one retry only


@pytest.mark.parametrize("reply", [
    (429, json.dumps({"error": {"message": "Daily quota used. Please try again in 5h3m"}}).encode()),
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


GEMINI_503 = (503, json.dumps({"error": {"code": 503, "message": "This model is currently "
                                         "experiencing high demand."}}).encode())


def test_temporary_server_errors_are_retried_with_growing_waits(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    transport, calls = sequence_transport(GEMINI_503, GEMINI_503, gemini_reply([{"text": "ok"}]))
    slept = []
    llm = GeminiLLMClient("gemini-x", transport=transport, sleep=slept.append)
    assert llm.complete("s", "u") == "ok"
    assert slept == [10.0, 30.0] and len(calls) == 3


def test_server_error_retries_run_out(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    transport, calls = sequence_transport((502, b""))
    slept = []
    llm = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport, sleep=slept.append)
    with pytest.raises(LLMUnavailable, match="HTTP 502"):
        llm.complete("s", "u")
    assert slept == [10.0, 30.0, 60.0] and len(calls) == 4


def test_client_errors_are_not_retried(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    transport, calls = sequence_transport((403, b"error code: 1010\n"))
    slept = []
    llm = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport, sleep=slept.append)
    with pytest.raises(LLMUnavailable, match="HTTP 403"):
        llm.complete("s", "u")
    assert slept == [] and len(calls) == 1


def test_output_cap_client_replaces_every_calls_cap(monkeypatch):
    from alpha_research.llm import OutputCapClient
    inner = ScriptedLLMClient(["a", "b"])
    capped = OutputCapClient(inner, 16384)
    assert capped.complete("s", "u", max_output=256) == "a"
    assert capped.complete("s", "u") == "b"
    assert [c["max_output"] for c in inner.calls] == [16384, 16384]
    assert capped.model == inner.model


def test_make_real_client_wraps_only_when_a_cap_is_given(monkeypatch):
    from alpha_research.llm import OutputCapClient
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    plain = make_real_client("nvidia/x:free")
    capped = make_real_client("nvidia/x:free", 16384)
    assert isinstance(plain, ChatCompletionsLLMClient)
    assert isinstance(capped, OutputCapClient) and capped.max_output == 16384
    assert capped.model == "nvidia/x:free" and capped.provider == "nvidia"


def test_research_llm_factory_applies_per_model_caps(monkeypatch):
    from types import SimpleNamespace
    from alpha_research.cli import _llm_factory
    from alpha_research.llm import ChatCompletionsLLMClient, OutputCapClient
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    f = _llm_factory(SimpleNamespace(llm="real"), ["a/b:free", "groq:qwen/q"],
                     {"a/b:free": 16384})
    assert isinstance(f("a/b:free"), OutputCapClient) and f("a/b:free").max_output == 16384
    assert isinstance(f("groq:qwen/q"), ChatCompletionsLLMClient)


def test_list_models_without_ids_says_what_came_back():
    with pytest.raises(LLMUnavailable, match="no model ids.*'OK'"):
        list_models("nvidia", api_key="k", transport=lambda *a: (200, b"OK"))


def test_retired_github_host_fails_clearly(monkeypatch):
    for call in (lambda: make_real_client("github:openai/gpt-4.1-mini"),
                 lambda: ChatCompletionsLLMClient("github:openai/gpt-4.1-mini", api_key="k"),
                 lambda: list_models("github", api_key="k")):
        with pytest.raises(LLMUnavailable, match="retired on 2026-07-30"):
            call()


def test_cerebras_client_sends_the_bare_name(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "k")
    seen = {}

    def transport(url, body, headers, timeout):
        seen.update(url=url, body=json.loads(body))
        return 200, json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    c = ChatCompletionsLLMClient("cerebras:gpt-oss-120b", transport=transport)
    assert c.complete("s", "u") == "ok" and c.host == "cerebras" and c.provider == "openai"
    assert seen["url"] == "https://api.cerebras.ai/v1/chat/completions"
    assert seen["body"]["model"] == "gpt-oss-120b"


def test_a_cut_off_reply_is_retried_like_a_server_error(monkeypatch):
    # run 2026-11-23 crashed on http.client.IncompleteRead from a model host
    import http.client
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    replies = [http.client.IncompleteRead(b"x"), hosted_reply("ok")]

    def transport(url, body, headers, timeout):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    slept = []
    llm = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=transport, sleep=slept.append)
    assert llm.complete("s", "u") == "ok" and slept == [10.0]
    always = lambda *a: (_ for _ in ()).throw(http.client.IncompleteRead(b""))
    llm = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", transport=always, sleep=lambda s: None)
    with pytest.raises(LLMUnavailable, match="IncompleteRead"):
        llm.complete("s", "u")


def test_usage_is_read_from_hosted_replies(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    body = json.loads(hosted_reply("ok")[1])
    body["usage"] = {"total_tokens": 5123}
    llm = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b",
                                   transport=lambda *a: (200, json.dumps(body).encode()))
    llm.complete("s", "u")
    assert llm.last_usage == 5123


class _Timed:
    model, provider = "m", "x"

    def __init__(self, usage=None):
        self.last_usage, self.calls = usage, 0

    def complete(self, system, user, *, max_output=4096):
        self.calls += 1
        return "r" * 400


def test_pacing_waits_for_the_token_budget_before_sending():
    from alpha_research.llm import PacedClient
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(round(s, 1))
        now[0] += s
    inner = _Timed(usage=5000)          # the host says each call cost 5,000 tokens
    paced = PacedClient(inner, tokens_per_minute=8000, clock=lambda: now[0], sleep=sleep)
    paced.complete("s" * 4000, "u")     # empty window: goes at once
    assert slept == []
    now[0] += 10
    paced.complete("s" * 4000, "u")     # 5,000 + ~3,250 would pass 8,000: waits for the window
    assert slept == [50.5] and inner.calls == 2


def test_pacing_by_calls_per_minute():
    from alpha_research.llm import PacedClient
    now, slept = [0.0], []
    paced = PacedClient(_Timed(), calls_per_minute=2, clock=lambda: now[0],
                        sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    paced.complete("s", "u")
    paced.complete("s", "u")
    assert slept == []
    paced.complete("s", "u")            # third call in the same minute waits
    assert len(slept) == 1 and slept[0] == pytest.approx(60.5)


def test_make_real_client_adds_pacing(monkeypatch):
    from alpha_research.llm import OutputCapClient, PacedClient
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    c = make_real_client("nvidia/x:free", 16384, {"tokens_per_minute": 8000})
    assert isinstance(c, PacedClient) and isinstance(c.inner, OutputCapClient)
    assert c.tokens_per_minute == 8000 and c.model == "nvidia/x:free"
