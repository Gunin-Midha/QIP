"""Run rs20261125130750 (2026-11-25): one ~11,000-token Strategy Evolution request was refused
by gpt-oss (8,000 tokens a minute) and then by Qwen (7,000 input tokens a minute), and each
refusal marked that model down for the rest of the run. Now: a too-large request is its own
error that never marks a model down, the size client learns real token counts from the host
and retries a 413 with a cap that fits, and the Strategy prompt is cut to a character cap."""

import json

import pytest

from alpha_research.llm import (LLMUnavailable, ModelDown, RequestFitClient, RequestTooLarge,
                                too_large)
from alpha_research.research import BudgetedLLM, CallCounter, FallbackLLM
from alpha_research.strategy import StrategyConfig, StrategyEvolution, trimmed_summary

GROQ_TPM = ("Too large for model `gpt-oss-120b` in organization `o` service tier "
            "`on_demand` on tokens per minute (TPM): Limit 8000, Requested 11002, please reduce")
GROQ_ITPM = ("Too large for model `qwen3.8-27b` in organization `o` service tier "
             "`on_demand` on input tokens per minute (ITPM): Limit 7000, Requested 11644, please")


class Host:
    """A fake model: refuses calls whose prompt (chars / ratio) + cap exceed ``limit``,
    else replies and reports prompt_tokens like a chat-completions host."""
    model = "groq:fake/model"
    provider = "fake"

    def __init__(self, limit=8000, ratio=2.5, input_only=False):
        self.limit, self.ratio, self.input_only = limit, ratio, input_only
        self.caps = []
        self.last_prompt_tokens = None

    def complete(self, system, user, *, max_output=4096):
        self.caps.append(max_output)
        prompt = int((len(system) + len(user)) / self.ratio)
        asked = prompt if self.input_only else prompt + max_output
        if asked > self.limit:
            kind = "input tokens per minute (ITPM)" if self.input_only else "tokens per minute (TPM)"
            raise too_large(self.model, 413, f"... on {kind}: Limit {self.limit}, "
                                             f"Requested {asked}, please reduce")
        self.last_prompt_tokens = prompt
        return "ok"


def test_413_becomes_request_too_large_with_the_hosts_numbers():
    e = too_large("m", 413, GROQ_TPM)
    assert isinstance(e, RequestTooLarge) and isinstance(e, ModelDown)
    assert (e.requested, e.limit, e.input_only) == (11002, 8000, False)
    e = too_large("m", 413, GROQ_ITPM)
    assert (e.requested, e.limit, e.input_only) == (11644, 7000, True)
    assert too_large("m", 429, GROQ_TPM) is None
    assert too_large("m", 413, None).requested is None


def test_a_413_is_retried_once_with_a_cap_that_fits_and_the_ratio_is_learned():
    host = Host(limit=8000, ratio=2.5)
    fit = RequestFitClient(host, 8000)                # assumes 3.3 characters a token
    text = "x" * 15000                                # 6,000 real tokens
    assert fit.complete(text, "", max_output=4096) == "ok"
    first, retry = host.caps
    assert first > retry and 6000 + retry <= 8000     # the refused cap, then one that fits
    assert fit.stats()["chars_per_token"] == 2.5 and fit.stats()["refused"] == 1
    assert fit.stats()["max_prompt_tokens"] == 6000


def test_a_prompt_too_large_for_any_cap_is_refused_before_sending_once_measured():
    host = Host(limit=8000, ratio=2.5)
    fit = RequestFitClient(host, 8000)
    fit.complete("x" * 5000, "", max_output=1000)     # measures 2.5 characters a token
    n = len(host.caps)
    with pytest.raises(RequestTooLarge):
        fit.complete("x" * 25000, "", max_output=4096)   # 10,000 tokens: cannot fit
    assert len(host.caps) == n                        # nothing sent


def test_an_input_only_limit_is_not_retried():
    host = Host(limit=7000, ratio=2.5, input_only=True)
    fit = RequestFitClient(host, 7000)
    with pytest.raises(RequestTooLarge) as exc:
        fit.complete("x" * 25000, "", max_output=4096)
    assert exc.value.input_only and len(host.caps) == 1
    # the host's input-only count is the prompt itself: 25,000 / 25,000 / 2.5 = 2.5
    assert fit.stats()["chars_per_token"] == 2.5


def test_too_large_never_marks_a_model_down_and_fallback_tries_the_next():
    counter = CallCounter(None)

    class Big:
        model = "groq:a/big"
        def complete(self, s, u, *, max_output=4096):
            raise RequestTooLarge(self.model, "too big", 11002, 8000)

    class Fine:
        model = "gemini-x"
        def complete(self, s, u, *, max_output=4096):
            return "ok"

    a = BudgetedLLM(Big(), counter=counter)
    b = BudgetedLLM(Fine(), counter=counter)
    assert FallbackLLM([a, b]).complete("s", "u") == "ok"
    assert counter.down == {} and counter.too_large == {"groq:a/big": 1}
    with pytest.raises(RequestTooLarge):
        FallbackLLM([a]).complete("s", "u")
    assert counter.down == {}

    class Broken:
        model = "groq:c/broken"
        def complete(self, s, u, *, max_output=4096):
            raise LLMUnavailable("status 500")

    with pytest.raises(ModelDown):
        BudgetedLLM(Broken(), counter=counter).complete("s", "u")
    assert "groq:c/broken" in counter.down


def _summary(n_arms=80):
    return {"regions": {"USA": {"decided": 9, "accepted": 1, "failures": {"sharpe": 5},
                                "datasets": {f"d{i}": i for i in range(30)},
                                "neutralizations": {}}},
            "decided": 100, "accepted": 2, "failure_classes": {"sharpe": 60},
            "families": {f"fam{i}": {"decided": i, "accepted": 0, "failures": {}}
                         for i in range(40)},
            "arms": {f"arm{i}": {"decided": i, "accepted": 0} for i in range(n_arms)},
            "best_accepted": [{"expression": "x" * 300}] * 5,
            "recent_failures": [{"expression": "y" * 300}] * 5,
            "datasets": {f"d{i}": i for i in range(40)},
            "least_tried": ["a", "b"],
            "what_fixes_worked": [{"failure_class": "sharpe", "lever": "l", "fixed": 1,
                                   "posterior_mean": 0.5}] * 8}


def test_trimmed_summary_cuts_the_least_useful_parts_first():
    s = _summary()
    full = len(json.dumps(s, separators=(",", ":")))
    t = trimmed_summary(s, full // 2)
    assert len(json.dumps(t, separators=(",", ":"))) <= full // 2
    assert t["decided"] == 100 and t["least_tried"] == ["a", "b"]
    assert len(t["recent_failures"]) == 2
    assert "arm79" in t["arms"] and "arm0" not in t["arms"]     # biggest arms kept
    assert len(s["arms"]) == 80                                 # input untouched
    assert trimmed_summary(s, 10 ** 9) == json.loads(json.dumps(s))


def test_strategy_prompt_is_cut_to_its_cap(monkeypatch):
    import alpha_research.strategy as strategy
    seen = {}

    class Model:
        model = "m"
        def complete(self, system, user, *, max_output=4096):
            seen["size"] = len(system) + len(user)
            return json.dumps({"guidance": "g", "hypotheses": []})

    monkeypatch.setattr(strategy, "summarize", lambda outcomes, **kw: _summary(200))
    evo = StrategyEvolution(StrategyConfig(max_prompt_chars=9000), llm=Model(),
                            context="ctx").evolve([], "gen-v17")
    assert seen["size"] <= 9000
    assert any("records trimmed" in n for n in evo.notes)
    uncut = StrategyEvolution(StrategyConfig(), llm=Model(), context="ctx").evolve([], "gen-v17")
    assert seen["size"] > 9000 and not any("trimmed" in n for n in uncut.notes)


def test_strategy_config_reads_max_prompt_chars(tmp_path):
    import yaml
    from pathlib import Path
    from alpha_research.research import load_agents_config
    root = Path(__file__).resolve().parents[1]
    assert load_agents_config(root / "config" / "agents.yaml").strategy.max_prompt_chars == 15000
    raw = yaml.safe_load((root / "config" / "agents.yaml").read_text())
    raw["strategy"]["max_prompt_chars"] = None
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert load_agents_config(p).strategy.max_prompt_chars is None


def test_an_empty_reply_under_a_lowered_cap_is_too_large_not_down():
    # run rs20261126082238: gpt-oss thought through a squeezed cap, replied empty, went down
    class Thinker:
        model, provider = "groq:fake/model", "fake"

        def complete(self, system, user, *, max_output=4096):
            if max_output < 4096:
                raise LLMUnavailable("model returned an empty reply (finish_reason length)")
            return "ok"

    fit = RequestFitClient(Thinker(), 8000, chars_per_token=3.0)
    counter = CallCounter(None)
    llm = BudgetedLLM(fit, counter=counter)
    with pytest.raises(RequestTooLarge, match="output cap lowered"):
        llm.complete("s", "x" * 15000, max_output=4096)
    assert "groq:fake/model" not in counter.down            # stays up for smaller calls
    assert llm.complete("s", "x" * 30, max_output=4096) == "ok"
    # an empty reply at the requested cap is still the model's failure
    with pytest.raises(ModelDown):
        BudgetedLLM(RequestFitClient(Thinker(), 8000), counter=counter).complete(
            "s", "x", max_output=1000)
