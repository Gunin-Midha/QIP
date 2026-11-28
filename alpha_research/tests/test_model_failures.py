"""Run rs20261127153052 (2026-11-27) ended with every model down: Nemotron Lightning and
Ernie after one empty reply each (reasoning used the 16k cap), Qwen on Groq's output tokens
per minute (a 429 "Request too large ... (OTPM): Limit 1000, Requested 1371"), gpt-oss on a
tokens-per-day 429 that asked for 13 minutes. Now: an empty reply is retried once with a
larger cap and fails the call only (down after EMPTY_REPLIES_DOWN in a row); an OTPM refusal
is retried under the host's limit, which is remembered; a 429 with a stated wait of at most
PAUSE_MAX_SECONDS pauses the model instead of taking it out for the run."""

import json

import pytest

from alpha_research.llm import (CallFailed, ChatCompletionsLLMClient, EmptyReply, ModelDown,
                                OutputCapClient, PacedClient, RateLimited, RequestFitClient,
                                RequestTooLarge, too_large)
from alpha_research.research import (EMPTY_REPLIES_DOWN, BudgetedLLM, CallCounter,
                                     FallbackLLM)

OTPM = ("Request too large for model `qwen/qwen3.8-27b` in organization `o` service tier "
        "`on_demand` on output tokens per minute (OTPM): Limit 1000, Requested 1371. The "
        "request's expected output tokens exceed the enforced limit; reduce max_tokens")


def _hosted(replies):
    """A chat-completions client on a fake transport answering ``replies`` in turn."""
    sent = []

    def transport(url, body, headers, timeout):
        sent.append(json.loads(body))
        status, payload = replies[min(len(sent), len(replies)) - 1]
        return status, json.dumps(payload).encode()

    c = ChatCompletionsLLMClient("groq:qwen/qwen3.8-27b", api_key="k", transport=transport,
                                 sleep=lambda s: None)
    return c, sent


def _reply(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"total_tokens": 900, "prompt_tokens": 700}}


def test_otpm_429_is_too_large_and_not_waited_on():
    e = too_large("m", 429, OTPM)
    assert isinstance(e, RequestTooLarge) and e.output_only
    assert (e.limit, e.requested) == (1000, 1371)
    c, sent = _hosted([(429, {"error": {"message": OTPM}})])
    with pytest.raises(RequestTooLarge):
        c.complete("s", "u", max_output=1371)
    assert len(sent) == 1                     # no 20 s wait-and-retry on a too-large 429


def test_otpm_refusal_is_retried_under_the_limit_and_remembered():
    c, sent = _hosted([(429, {"error": {"message": OTPM}}), (200, _reply("ok")),
                       (200, _reply("ok"))])
    fit = RequestFitClient(c, 7000)
    assert fit.complete("s", "u", max_output=1371) == "ok"
    assert [b["max_tokens"] for b in sent] == [1371, 1000]
    fit.complete("s", "u", max_output=4096)   # the learned limit applies at once
    assert sent[-1]["max_tokens"] == 1000 and fit.stats()["output_limit"] == 1000


def test_configured_output_limit_without_a_token_limit():
    c, sent = _hosted([(200, _reply("ok"))])
    fit = RequestFitClient(c, None, output_tokens_per_minute=800)
    fit.complete("s", "u", max_output=4096)
    assert sent[0]["max_tokens"] == 800


def test_empty_reply_at_the_cap_is_retried_once_with_a_larger_cap():
    c, sent = _hosted([(200, _reply("", "length")), (200, _reply("ok"))])
    capped = OutputCapClient(c, 16384)
    assert capped.complete("s", "u") == "ok"
    assert [b["max_tokens"] for b in sent] == [16384, 32768]
    c, sent = _hosted([(200, _reply("", "length"))])
    with pytest.raises(EmptyReply):
        OutputCapClient(c, 16384).complete("s", "u")
    assert len(sent) == 2


def test_empty_replies_fail_the_call_and_down_the_model_only_in_a_row():
    counter = CallCounter(None)

    class Model:
        model = "nvidia/x:free"
        def __init__(self):
            self.script = []
        def complete(self, s, u, *, max_output=4096):
            r = self.script.pop(0)
            if r is None:
                raise EmptyReply(self.model, "empty", at_cap=True)
            return r

    m = Model()
    b = BudgetedLLM(m, counter=counter)
    m.script = [None, None, "ok"] + [None] * EMPTY_REPLIES_DOWN
    for _ in range(2):
        with pytest.raises(CallFailed):
            b.complete("s", "u")
    assert b.complete("s", "u") == "ok" and counter.down == {}
    for _ in range(EMPTY_REPLIES_DOWN - 1):
        with pytest.raises(EmptyReply):
            b.complete("s", "u")
    with pytest.raises(ModelDown) as exc:
        b.complete("s", "u")
    assert not isinstance(exc.value, CallFailed) and "nvidia/x:free" in counter.down
    assert counter.empty["nvidia/x:free"] == 2 + EMPTY_REPLIES_DOWN


def test_short_rate_limit_pauses_the_model_and_a_daily_quota_downs_it():
    now = [0.0]
    counter = CallCounter(None, clock=lambda: now[0], pause_max_seconds=1800)

    class Limited:
        model = "groq:gpt-oss-120b"
        wait = 800.0
        calls = 0
        def complete(self, s, u, *, max_output=4096):
            self.calls += 1
            if self.wait:
                raise RateLimited(self.model, "tokens per day", self.wait)
            return "ok"

    class Other:
        model = "gemini-x"
        def complete(self, s, u, *, max_output=4096):
            return "other"

    m = Limited()
    a, b = BudgetedLLM(m, counter=counter), BudgetedLLM(Other(), counter=counter)
    assert FallbackLLM([a, b]).complete("s", "u") == "other"   # paused: next model answers
    assert not counter.is_up(m.model) and m.model not in counter.down
    with pytest.raises(RateLimited):
        a.complete("s", "u")
    assert m.calls == 1                         # no call sent while paused
    now[0] = 802.0
    m.wait = 0
    assert counter.is_up(m.model) and a.complete("s", "u") == "ok"
    m.wait = 16 * 3600.0                         # Gemini's daily quota: out for the run
    with pytest.raises(ModelDown) as exc:
        a.complete("s", "u")
    assert not isinstance(exc.value, CallFailed) and m.model in counter.down


def test_429_with_a_long_stated_wait_becomes_rate_limited():
    msg = "Rate limit reached ... tokens per day (TPD): Limit 200000. Please try again in 13m20.06s."
    c, sent = _hosted([(429, {"error": {"message": msg}})])
    with pytest.raises(RateLimited) as exc:
        c.complete("s", "u")
    assert 799 < exc.value.wait < 801


def test_paced_client_keeps_output_tokens_within_the_minute():
    now, waits = [0.0], []

    class M:
        model = "groq:q/q"
        last_usage, last_prompt_tokens = 1700, 700
        def complete(self, s, u, *, max_output=4096):
            return "x"

    def sleep(s):
        waits.append(s)
        now[0] += s

    p = PacedClient(M(), output_tokens_per_minute=1000, clock=lambda: now[0], sleep=sleep)
    p.complete("s", "u", max_output=1000)      # 1,000 output tokens used
    p.complete("s", "u", max_output=1000)      # must wait for the minute to pass
    assert waits and now[0] >= 60
