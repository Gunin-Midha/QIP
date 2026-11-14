"""The closed research loop, end to end, with a routing fake model and a stub BRAIN client."""

import dataclasses
import json
import re
from collections import deque

import pytest

from alpha_research import cli
from alpha_research.brain_client import FixtureBrainClient
from alpha_research.config import ConfigError
from alpha_research.fastexpr import FastExprSpec
from alpha_research.generator import GenerationSettings, Source, load_generator_config
from alpha_research.llm import LLMUnavailable, ModelDown, ScriptedLLMClient, provider_of
from alpha_research.research import (ArmPlan, Budget, BudgetedLLM, CallCounter, FallbackLLM,
                                     LLMBudgetExhausted, ModelsConfig, ResearchRun, agents_config_from_dict,
                                     load_agents_config, load_plan)
from alpha_research.simulator import VirtualClock

from conftest import BRAIN_DIR, ROOT, TEST_CONFIG

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
GEN_CONFIG = load_generator_config(ROOT / "config" / "generator.yaml")
MAIN_AGENTS = load_agents_config(ROOT / "config" / "agents.yaml")
# one generator model and one critic from the other provider, no shadows: one arm per dataset
ONE_MODEL = ModelsConfig(default="claude-test", generator=("claude-test",), critic=("gemini-test",),
                         shadow_critics=False, prefilter="claude-test", debate="claude-test",
                         repair="claude-test", strategy="claude-test")
# no refill passes: these tests count what one allocation does (refill has its own tests)
AGENTS = dataclasses.replace(MAIN_AGENTS, models=ONE_MODEL, refill_passes=0)
ARM = ("pv1", "USA", "gen-v6", "claude-test", 1)
SETTINGS = {"region": "USA", "universe": "TOP3000", "delay": 1, "decay": 4,
            "neutralization": "SUBINDUSTRY", "truncation": 0.08,
            "start_date": "2019-01-01", "end_date": "2023-12-31"}


class RouterLLM:
    """Answers each agent by its system prompt. Generated alphas cycle through three
    kinds: close-price (BRAIN stub accepts), volume (turnover failure until decay 8) and a
    look-ahead ts_delay(close, -k) that the Critic must block."""

    model = "router"

    def __init__(self):
        self.calls = []
        self.systems = []
        self.all_systems = []
        self.k = 2

    def complete(self, system, user, *, max_output=4096):
        self.all_systems.append(system)
        ids = re.findall(r'<candidate id="([^"]+)"', user)
        if "Critic in a quant" in system:
            self.calls.append("critic")
            return json.dumps({"reviews": [{"candidate_id": i, "verdict": "simulate"} for i in ids]})
        if "judge in a debate" in system:
            self.calls.append("judge")
            return json.dumps({"rulings": [{"candidate_id": i, "ruling": "simulate", "reason": "x"}
                                           for i in ids]})
        if "research strategist" in system:
            self.calls.append("strategy")
            return json.dumps({"guidance": "avoid volume spikes without decay",
                               "hypotheses": [{"hypothesis": "evolved idea", "dataset": "pv1"}]})
        if "repair WorldQuant" in system:
            self.calls.append("repair")
            return json.dumps({"expression": "rank(-ts_delta(close, 7))"})
        self.calls.append("generator")
        self.systems.append(system)
        n = int(re.search(r"Write (\d+) distinct", user).group(1))
        alphas = []
        for j in range(n):
            self.k += 1
            kind = (self.k + j) % 3
            expr = {0: f"rank(-ts_delta(close, {self.k}))",
                    1: f"rank(ts_zscore(volume, {self.k}))",
                    2: f"rank(ts_delay(close, -{self.k}))"}[kind]
            alphas.append({"expression": expr, "family": "pv", "rationale": f"idea {self.k}"})
        return json.dumps({"alphas": alphas})


class StubBrain:
    def __init__(self):
        self.fx = FixtureBrainClient(BRAIN_DIR)
        self.accepts = 0
        self.submitted = []

    def submit(self, c):
        self.submitted.append(c)
        if "volume" in c.expression and c.decay < 8:
            fid = "sim_tvr_01"
        else:
            fid = f"sim_acc_0{self.accepts % 5 + 1}"
            self.accepts += 1
        self.fx.register(c.candidate_id, fid)
        return self.fx.submit(c)

    def poll(self, sim_id):
        return self.fx.poll(sim_id)


def plans():
    return [ArmPlan("pv1", GenerationSettings.from_mapping({**SETTINGS, "dataset": "pv1"}),
                    deque([Source("hypothesis", "short-term reversal"),
                           Source("hypothesis", "volume shocks")]))]


def make_run(memory, config, playbook, *, llm=None, brain=None, budget=None, agents=AGENTS,
             plan=None):
    vc = VirtualClock()
    return ResearchRun(plan or plans(), agents, budget or Budget(max_simulations=8, per_round=4,
                                                               max_llm_calls=100),
                       llm_factory=lambda model, r=llm or RouterLLM(): r, client=brain or StubBrain(), config=config,
                       playbook=playbook, spec=SPEC, generator_config=GEN_CONFIG, memory=memory,
                       seed=0, sleep=vc.sleep, clock=vc.clock, run_id="rs1")


def test_closed_loop_end_to_end(memory, config, playbook):
    llm, brain = RouterLLM(), StubBrain()
    run = make_run(memory, config, playbook, llm=llm, brain=brain)
    report = run.run(3)

    # budget respected and every simulation counted
    assert report["simulations_used"] == len(brain.submitted) <= 8
    assert report["model_calls"] == len(llm.calls)
    # the critic blocked every look-ahead expression before BRAIN saw it
    assert not any("ts_delay(close, -" in c.expression for c in brain.submitted)
    dropped = [d for r in report["rounds"] for a in r["arms"] for d in a["critic_dropped"]]
    assert dropped and all("window" in why for _, why in dropped)
    # a turnover failure was repaired by raising decay, and the child was accepted
    children = [c for c in brain.submitted if c.lever == "increase_decay"]
    assert children, "a repair child should have been simulated"
    child = children[0]
    assert child.decay == 8 and child.parent_id and child.candidate_id.endswith("_r1")
    assert memory.decisions_for(child.candidate_id)[-1].decision == "accept"
    assert memory.lineage_levers(child.candidate_id) == ["increase_decay"]
    # strategy evolution added an arm with a new prompt version and its guidance
    evo = next(r["evolution"] for r in report["rounds"] if "evolution" in r)
    assert evo["prompt_version"] == "gen-v6+evo1" and evo["source"] == "model"
    assert "pv1/USA/D1/gen-v6+evo1/claude-test" in report["arms"]
    new_key = ("pv1", "USA", "gen-v6+evo1", "claude-test", 1)
    assert run.arms[new_key].guidance == "avoid volume spikes without decay"
    assert "evolved idea" in [s.text for s in run.arms[new_key].sources]
    # the director learned from the outcomes
    base = report["arms"]["pv1/USA/D1/gen-v6/claude-test"]
    assert base["accepted"] >= 2 and base["accepted"] + base["not_accepted"] >= 3
    assert report["pool_size"] >= 2
    assert set(llm.calls) >= {"generator", "critic", "judge", "strategy"}


def test_guidance_reaches_the_generator_prompt(memory, config, playbook):
    # 10 slots over 2 arms in round 3: the floor guarantees the evolved arm at least 1
    llm = RouterLLM()
    run = make_run(memory, config, playbook, llm=llm,
                   budget=Budget(max_simulations=40, per_round=10, max_llm_calls=100))
    run.run(3)
    gen = run.generators[("pv1", "USA", "gen-v6+evo1", "claude-test", 1)]
    assert gen.config.prompt_version == "gen-v6+evo1"
    assert sum("avoid volume spikes without decay" in s for s in llm.systems) >= 1
    assert "avoid volume spikes" not in llm.systems[0]
    evolved = [r for r in memory.candidate_outcomes()
               if r["candidate"].prompt_version == "gen-v6+evo1"]
    assert evolved and all(r["candidate"].dataset == "pv1" for r in evolved)


def test_llm_budget_stops_cleanly(memory, config, playbook):
    run = make_run(memory, config, playbook,
                   budget=Budget(max_simulations=8, per_round=4, max_llm_calls=1))
    report = run.run(3)
    assert report["stop_reason"] == "model call budget of 1 used up"
    assert report["model_calls"] == 1 and report["simulations_used"] == 0


def test_simulation_budget_stops_cleanly(memory, config, playbook):
    brain = StubBrain()
    report = make_run(memory, config, playbook, brain=brain,
                      budget=Budget(max_simulations=2, per_round=4, max_llm_calls=100)).run(5)
    assert report["stop_reason"] == "simulation budget used up"
    assert report["simulations_used"] == 2 == len(brain.submitted)


def test_second_run_warm_starts_from_memory(memory, config, playbook):
    first = make_run(memory, config, playbook)
    first.run(1)
    decided = sum(s.successes + s.failures for s in first.director.state.values())
    second = make_run(memory, config, playbook)
    assert second.warm_started == decided > 0
    key = ARM
    assert second.director.state[key].successes == first.director.state[key].successes


def test_rules_only_agents(memory, config, playbook):
    raw = json.loads(json.dumps({
        "prefilter": {"near_duplicate_similarity": 0.9, "use_model": False, "min_model_score": None},
        "critic": {"use_model": False}, "debate": {"use_model": False}, "repair": {"use_model": False},
        "strategy": {"every_rounds": 0, "max_hypotheses": 3, "max_guidance_chars": 100},
        "director": {"floor_share": 0.2}, "max_output": 1024,
        "models": {"default": "claude-test", "shadow_critics": False}}))
    llm = RouterLLM()
    report = make_run(memory, config, playbook, llm=llm, agents=agents_config_from_dict(raw)).run(2)
    assert set(llm.calls) == {"generator"} and report["simulations_used"] > 0


TWO_MODELS = ModelsConfig(default="claude-a", generator=("claude-a", "gemini-b"),
                          critic=("gemini-b", "claude-a"), shadow_critics=True,
                          prefilter="claude-a", debate="claude-a", repair="claude-a",
                          strategy="claude-a")


def test_each_generator_model_is_an_arm_with_an_other_provider_critic(memory, config, playbook):
    agents = dataclasses.replace(AGENTS, models=TWO_MODELS)
    run = make_run(memory, config, playbook, agents=agents,
                   budget=Budget(max_simulations=12, per_round=6, max_llm_calls=200))
    report = run.run(2)
    assert {"pv1/USA/D1/gen-v6/claude-a", "pv1/USA/D1/gen-v6/gemini-b"} <= set(report["arms"])
    # both arms start from the same hypothesis: separate source queues
    assert run.arms[("pv1", "USA", "gen-v6", "claude-a", 1)].sources is not \
        run.arms[("pv1", "USA", "gen-v6", "gemini-b", 1)].sources
    reviews = memory.critic_reviews()
    primary = {(r["generator_model"], r["critic_model"]) for r in reviews if r["role"] == "primary"}
    shadow = {(r["generator_model"], r["critic_model"]) for r in reviews if r["role"] == "shadow"}
    assert primary == {("claude-a", "gemini-b"), ("gemini-b", "claude-a")}
    assert shadow == {("claude-a", "claude-a"), ("gemini-b", "gemini-b")}
    # every reviewed fresh candidate has one primary and one shadow verdict; repair children
    # get the primary's rule check only
    per_cid = {}
    for r in reviews:
        per_cid.setdefault(r["candidate_id"], []).append(r["role"])
    repairs = {c.candidate_id for c in (memory.get_candidate(cid) for cid in per_cid)
               if c is not None and c.lever is not None}
    assert all(sorted(roles) == (["primary"] if cid in repairs else ["primary", "shadow"])
               for cid, roles in per_cid.items())
    # candidates carry their generator's model; calls are counted per model
    models = {r["candidate"].model for r in memory.candidate_outcomes()
              if r["candidate"].parent_id is None}
    assert models == {"claude-a", "gemini-b"}
    assert set(report["model_calls_by_model"]) == {"claude-a", "gemini-b"}
    assert sum(report["model_calls_by_model"].values()) == report["model_calls"]
    assert set(report["comparison"]["generators"]) == {"claude-a", "gemini-b"}
    assert set(report["comparison"]["critics"]) == {"claude-a", "gemini-b"}


def test_shadow_critics_do_not_change_what_is_simulated(memory, config, playbook, tmp_path):
    from alpha_research.memory import MemoryStore
    submitted = []
    for shadows, mem in ((False, memory), (True, MemoryStore(tmp_path / "shadow"))):
        brain = StubBrain()
        agents = dataclasses.replace(
            AGENTS, models=dataclasses.replace(TWO_MODELS, shadow_critics=shadows))
        make_run(mem, config, playbook, agents=agents, brain=brain,
                 budget=Budget(max_simulations=8, per_round=4, max_llm_calls=200)).run(2)
        submitted.append([c.expression for c in brain.submitted])
        if shadows:
            assert any(r["role"] == "shadow" for r in mem.critic_reviews())
            mem.close()
    assert submitted[0] == submitted[1] and submitted[0]


def test_warm_start_credits_each_generator_arm(memory, config, playbook):
    agents = dataclasses.replace(AGENTS, models=TWO_MODELS)
    first = make_run(memory, config, playbook, agents=agents,
                     budget=Budget(max_simulations=12, per_round=6, max_llm_calls=200))
    first.run(1)
    second = make_run(memory, config, playbook, agents=agents)
    assert second.warm_started > 0
    for key, s in first.director.state.items():
        assert (second.director.state[key].successes, second.director.state[key].failures) == \
            (s.successes, s.failures)


def test_model_flag_replaces_the_default_only():
    m = dataclasses.replace(TWO_MODELS, debate="gemini-b").with_default("claude-z")
    assert m.default == "claude-z" and m.generator == ("claude-z", "gemini-b")
    assert m.critic == ("gemini-b", "claude-z") and m.debate == "gemini-b"
    assert m.repair == m.prefilter == m.strategy == "claude-z"


def test_used_models_skip_agents_with_the_model_off():
    agents = dataclasses.replace(AGENTS, models=dataclasses.replace(TWO_MODELS, repair="claude-r"),
                                 repair_model=False)
    assert "claude-r" not in agents.used_models()
    assert set(agents.used_models()) == {"claude-a", "gemini-b"}


def test_budgeted_llm_counts_and_stops():
    b = BudgetedLLM(ScriptedLLMClient(["a", "b", "c"]), max_calls=2)
    assert [b.complete("s", "u"), b.complete("s", "u")] == ["a", "b"]
    with pytest.raises(LLMBudgetExhausted):
        b.complete("s", "u")
    assert b.calls == 2 and b.model == "scripted"


class DownAfter:
    """Wraps a client: after ``n`` calls every call fails, like a free tier's daily quota."""

    def __init__(self, inner, n, model):
        self.inner, self.n, self.model = inner, n, model

    def complete(self, system, user, *, max_output=4096):
        if self.n <= 0:
            raise LLMUnavailable("quota exceeded, retry in 23h")
        self.n -= 1
        return self.inner.complete(system, user, max_output=max_output)


def test_budgeted_llm_marks_a_failed_model_down_and_fallback_moves_on():
    counter = CallCounter(None)
    a = BudgetedLLM(DownAfter(ScriptedLLMClient(["a1"]), 1, "m-a"), counter=counter, model="m-a")
    b = BudgetedLLM(ScriptedLLMClient(["b1", "b2"]), counter=counter, model="m-b")
    fb = FallbackLLM([a, b])
    assert fb.complete("s", "u") == "a1" and fb.model == "m-a"
    assert fb.complete("s", "u") == "b1"          # a failed: marked down, b answered
    assert "m-a" in counter.down and fb.model == "m-b"
    with pytest.raises(ModelDown):
        a.complete("s", "u")                      # down: raises without spending budget
    assert counter.by_model == {"m-a": 2, "m-b": 1}
    assert fb.complete("s", "u") == "b2"
    with pytest.raises(ModelDown):
        fb.complete("s", "u")                     # b's script is used up: every model down


def test_a_down_generator_model_does_not_stop_the_run(memory, config, playbook):
    router = RouterLLM()
    clients = {"claude-a": DownAfter(router, 2, "claude-a"), "gemini-b": router}
    agents = dataclasses.replace(AGENTS, models=TWO_MODELS)
    vc = VirtualClock()
    run = ResearchRun(plans(), agents, Budget(max_simulations=12, per_round=4, max_llm_calls=200),
                      llm_factory=lambda m: clients[m], client=StubBrain(), config=config,
                      playbook=playbook, spec=SPEC, generator_config=GEN_CONFIG, memory=memory,
                      seed=0, sleep=vc.sleep, clock=vc.clock, run_id="rs1")
    report = run.run(4)
    assert "claude-a" in report["models_down"] and "gemini-b" not in report["models_down"]
    assert not report["stop_reason"].startswith("claude-a")
    assert report["simulations_used"] > 2
    # after it went down, claude-a's arms got no slots and the other agents used gemini-b
    later = [r for r in report["rounds"] if r["round"] >= 3]
    assert later and all(n == 0 for r in later for arm, n in r.get("allocation", {}).items()
                         if arm.endswith("claude-a"))


def test_slots_of_a_model_that_goes_down_go_to_live_arms(memory, config, playbook):
    router = RouterLLM()
    clients = {"claude-a": DownAfter(router, 0, "claude-a"), "gemini-b": router}
    agents = dataclasses.replace(AGENTS, models=TWO_MODELS)
    vc = VirtualClock()
    run = ResearchRun(plans(), agents, Budget(max_simulations=12, per_round=4, max_llm_calls=200),
                      llm_factory=lambda m: clients[m], client=StubBrain(), config=config,
                      playbook=playbook, spec=SPEC, generator_config=GEN_CONFIG, memory=memory,
                      seed=0, sleep=vc.sleep, clock=vc.clock, run_id="rs1")
    report = run.run(1)
    r1 = report["rounds"][0]
    assert r1["allocation"]["pv1/USA/D1/gen-v6/claude-a"] > 0          # claude-a was not known down
    assert r1["refills"] and all(arm.endswith("gemini-b") for f in r1["refills"] for arm in f)
    refilled = [a for a in r1["arms"] if a.get("refill")]
    assert refilled and sum(len(a["outcomes"]) for a in refilled) > 0
    ids = [o["candidate_id"] for a in r1["arms"] for o in a["outcomes"]]
    assert len(ids) == len(set(ids))                                  # refill batches get own ids


def test_slots_emptied_by_the_critic_are_refilled_up_to_refill_passes(memory, config, playbook,
                                                                     tmp_path):
    def round_one(passes, mem):
        agents = dataclasses.replace(AGENTS, refill_passes=passes)
        return make_run(mem, config, playbook, agents=agents,
                        budget=Budget(max_simulations=20, per_round=6, max_llm_calls=200)).run(1)
    from alpha_research.memory import MemoryStore
    without = round_one(0, memory)["rounds"][0]
    assert "refills" not in without
    with_ = round_one(1, MemoryStore(tmp_path / "second"))["rounds"][0]
    assert len(with_["refills"]) == 1                                 # one extra pass at most
    sims = lambda r: sum(len(a["outcomes"]) for a in r["arms"])
    assert sims(with_) > sims(without)


class NewDatasetStrategy(RouterLLM):
    def complete(self, system, user, *, max_output=4096):
        if "research strategist" in system:
            self.calls.append("strategy")
            return json.dumps({"guidance": "try news", "hypotheses": [
                {"hypothesis": "news tone drift", "dataset": "news12"},
                {"hypothesis": "made-up data idea", "dataset": "nosuch9"}]})
        return super().complete(system, user, max_output=max_output)


def test_hypothesis_on_a_new_catalog_dataset_gets_its_own_arms(memory, config, playbook):
    vc = VirtualClock()
    run = ResearchRun(plans(), AGENTS, Budget(max_simulations=8, per_round=4, max_llm_calls=100),
                      llm_factory=lambda m, r=NewDatasetStrategy(): r, client=StubBrain(),
                      config=config, playbook=playbook, spec=SPEC, generator_config=GEN_CONFIG,
                      memory=memory, field_meta={"tone": {"dataset": "news12", "type": "MATRIX"}},
                      seed=0, sleep=vc.sleep, clock=vc.clock, run_id="rs1")
    run.run(2)
    new = ("news12", "USA", "gen-v6+evo1", "claude-test", 1)
    assert new in run.arms and run.arms[new].settings.dataset == "news12"
    assert [s.text for s in run.arms[new].sources] == ["news tone drift"]
    # the old dataset's evolved arm keeps the unknown-dataset idea but not the news12 one
    texts = [s.text for s in run.arms[("pv1", "USA", "gen-v6+evo1", "claude-test", 1)].sources]
    assert "made-up data idea" in texts and "news tone drift" not in texts


def test_run_stops_when_every_generator_model_is_down(memory, config, playbook):
    run = make_run(memory, config, playbook, llm=DownAfter(RouterLLM(), 0, "claude-test"))
    report = run.run(3)
    assert report["stop_reason"].startswith("every generator model is down")
    assert report["simulations_used"] == 0


# ---------------------------------------------------------------- config and plan


def test_generator_is_told_brain_bar_for_its_market(memory, config, playbook, candidate):
    from alpha_research.schemas import ResultRecord
    memory.put_candidate(candidate(candidate_id="seen1"))
    memory.put_result(ResultRecord("seen1", "complete", sharpe=0.5, check_values={
        "LOW_SHARPE": {"value": 0.5, "limit": 1.58}}))
    run = make_run(memory, config, playbook)
    gen = run._generator(ARM)
    assert "BRAIN's bar in this market (USA delay 1, from its checks on our alphas): " \
           "Sharpe >= 1.58" in gen.guidance


def test_main_agents_config_loads():
    a = MAIN_AGENTS
    assert a.prefilter.near_duplicate_similarity == 0.9 and a.critic_model
    assert a.strategy.every_rounds == 2 and a.floor_share == 0.2
    assert a.sparse.categories == ("news", "sentiment", "socialmedia") and a.sparse.lookback == 20
    # free models only (owner, 2026-11-03); GPT and Nemotron (OpenRouter, generator only)
    # added 2026-11-04, no Mistral (Llama and DeepSeek wait for a free host that answers);
    # shadow critics off to save the quotas
    assert {provider_of(g) for g in a.models.generator} == \
        {"google", "qwen", provider_of("groq:gpt-oss-120b"), "nvidia"}
    assert not any(m.endswith(":free") for m in a.models.critic)
    assert "mistral" not in {provider_of(m) for m in a.models.all_models()}
    assert not a.models.shadow_critics
    # Gemini (20 calls/day) is no agent's default and only the last-resort critic
    assert "gemini" not in a.models.default and a.models.critic[-1].startswith("gemini")
    for g in a.models.generator:
        assert provider_of(a.models.primary_critic(g)) != provider_of(g)


@pytest.mark.parametrize("patch, match", [
    ({"critic": None}, "missing section 'critic'"),
    ({"prefilter": {"near_duplicate_similarity": 2, "use_model": False}}, r"\[0, 1\]"),
    ({"debate": {"use_model": "yes"}}, "true or false"),
    ({"director": {"floor_share": 1.5}}, "floor_share"),
    ({"max_output": 10}, "max_output"),
    ({"models": None}, "missing section 'models'"),
    ({"models": {"default": "claude-x", "shadow_critics": "yes"}}, "shadow_critics"),
    ({"models": {"default": "claude-x", "shadow_critics": True, "generator": []}}, "models.generator"),
    ({"models": {"default": "claude-x", "shadow_critics": True, "debate": 3}}, "models.debate"),
    ({"models": {"default": "gpt-x", "shadow_critics": True}}, "unknown provider"),
    # critic from the generator's own provider only: refused
    ({"models": {"default": "claude-x", "shadow_critics": True,
                 "generator": ["claude-x", "gemini-y"], "critic": ["claude-z"]}},
     "needs a critic from a different provider"),
])
def test_agents_config_errors(patch, match):
    import yaml
    raw = yaml.safe_load((ROOT / "config" / "agents.yaml").read_text())
    raw.update(patch)
    with pytest.raises(ConfigError, match=match):
        agents_config_from_dict(raw).check_models()


def write_plan(tmp_path, arms):
    import yaml
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({"arms": arms}))
    return p


def test_example_plan_and_papers(tmp_path):
    assert [a.dataset for a in load_plan(ROOT / "config" / "research_plan.example.yaml")] == \
        ["pv1", "fundamental6"]
    (tmp_path / "paper.txt").write_text("A paper about momentum.")
    plan = load_plan(write_plan(tmp_path, [{"dataset": "pv1", "settings": SETTINGS,
                                            "papers": ["paper.txt"]}]))
    src = plan[0].sources[0]
    assert (src.kind, src.title, src.text) == ("paper", "paper", "A paper about momentum.")
    assert plan[0].settings.dataset == "pv1"


@pytest.mark.parametrize("arms, match", [
    ([], "non-empty 'arms'"),
    ([{"settings": SETTINGS, "hypotheses": ["h"]}], "dataset"),
    ([{"dataset": "pv1", "hypotheses": ["h"]}], "settings"),
    ([{"dataset": "pv1", "settings": SETTINGS}], "at least one hypothesis"),
    ([{"dataset": "pv1", "settings": SETTINGS, "papers": ["missing.txt"]}], "cannot read paper"),
    ([{"dataset": "pv1", "settings": SETTINGS, "hypotheses": ["h"]}] * 2, "twice"),
    ([{"dataset": "pv1", "settings": {**SETTINGS, "region": None}, "hypotheses": ["h"]}], "region"),
])
def test_plan_errors(tmp_path, arms, match):
    with pytest.raises(ConfigError, match=match):
        load_plan(write_plan(tmp_path, arms))


# ---------------------------------------------------------------- CLI


def test_cli_research_end_to_end(tmp_path, monkeypatch, capsys):
    import yaml
    brain = StubBrain()
    monkeypatch.setattr(cli, "make_session", lambda cfg, on_biometric=None: None)
    monkeypatch.setattr(cli, "RealBrainClient", lambda session, cfg, capture_dir=None: brain)
    monkeypatch.setattr("alpha_research.pipeline.time.sleep", lambda s: None, raising=False)
    agents = yaml.safe_load((ROOT / "config" / "agents.yaml").read_text())
    for name in ("critic", "debate", "repair"):
        agents[name]["use_model"] = False
    agents["strategy"]["every_rounds"] = 0
    agents["models"]["generator"] = ["claude-sonnet-5-5"]
    (tmp_path / "agents.yaml").write_text(yaml.safe_dump(agents))
    plan = write_plan(tmp_path, [{"dataset": "pv1", "settings": SETTINGS, "hypotheses": ["reversal"]}])
    script = tmp_path / "script.json"
    script.write_text(json.dumps([json.dumps({"alphas": [
        {"expression": "rank(-ts_delta(close, 3))"}, {"expression": "rank(-ts_delta(close, 9))"}]})]))
    report = tmp_path / "report.json"
    rc = cli.main(["research", "--plan", str(plan), "--max-simulations", "2", "--max-llm-calls", "5",
                   "--per-round", "2", "--rounds", "1", "--config", str(TEST_CONFIG),
                   "--playbook", str(ROOT / "config" / "repair_playbook.yaml"),
                   "--agents", str(tmp_path / "agents.yaml"),
                   "--fastexpr", str(ROOT / "config" / "fastexpr.yaml"),
                   "--generator-config", str(ROOT / "config" / "generator.yaml"),
                   "--brain-config", str(ROOT / "config" / "brain.yaml"),
                   "--data-dir", str(tmp_path / "data"), "--report", str(report),
                   "--llm", "scripted", "--llm-script", str(script)])
    out = capsys.readouterr().out
    assert rc == 0, out
    data = json.loads(report.read_text())
    assert data["simulations_used"] == 2 and data["stop_reason"] == "simulation budget used up"
    assert "stopped: simulation budget used up" in out and "pv1/USA/D1/gen-v6/claude-sonnet-5-5: accepted 2" in out


def test_cli_research_needs_a_holdout(tmp_path, no_holdout_config):
    plan = write_plan(tmp_path, [{"dataset": "pv1", "settings": SETTINGS, "hypotheses": ["h"]}])
    rc = cli.main(["research", "--plan", str(plan), "--max-simulations", "1", "--max-llm-calls", "1",
                   "--config", str(no_holdout_config),
                   "--data-dir", str(tmp_path / "d")])
    assert rc == cli.EXIT_CONFIG


def _orphan_repair(memory, candidate):
    """A turnover repair decision from an earlier run whose child never reached BRAIN."""
    from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord
    parent = candidate(candidate_id="old_001", expression="rank(ts_zscore(volume, 9))", decay=4,
                       prompt_version="gen-v6", model="claude-test")
    memory.put_candidate(parent)
    memory.put_result(ResultRecord("old_001", "complete", sharpe=1.7, turnover=0.85))
    memory.put_decision(DecisionRecord("old_001", "repair", "turnover", gate_results=[
        GateResult("turnover_range", False, 0.85, (0.01, 0.7), "")]))


def test_lost_repair_is_resumed_and_simulated_first(memory, config, playbook, candidate):
    _orphan_repair(memory, candidate)
    brain = StubBrain()
    report = make_run(memory, config, playbook, brain=brain).run(1)
    assert [r["child"] for r in report["resumed_repairs"]] == ["old_001_r1"]
    first = brain.submitted[0]
    assert (first.candidate_id, first.lever, first.decay) == ("old_001_r1", "increase_decay", 8)
    assert report["rounds"][0]["arms"][0]["repairs_first"] is True
    assert memory.decisions_for("old_001_r1")[-1].decision == "accept"
    # the accepted child is evidence: increase_decay fixed a turnover failure
    row = next(r for r in report["lever_stats"] if r["lever"] == "increase_decay")
    assert (row["failure_class"], row["events"], row["fixed"]) == ("turnover", 1, 1)
    # a parent with a child is not resumed again (repairs the first run queued but did not
    # simulate are)
    again = make_run(memory, config, playbook).run(1)["resumed_repairs"]
    assert "old_001" not in [r["parent"] for r in again]


def test_open_repair_survives_a_prompt_version_bump(memory, config, playbook, candidate):
    # gen-v3 -> gen-v4 left earlier runs' open repairs without an arm (2026-11-11)
    from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord
    memory.put_candidate(candidate(candidate_id="v3_001", expression="rank(ts_zscore(volume, 9))",
                                   decay=4, prompt_version="gen-v0-old", model="claude-test"))
    memory.put_result(ResultRecord("v3_001", "complete", sharpe=1.7, turnover=0.85))
    memory.put_decision(DecisionRecord("v3_001", "repair", "turnover", gate_results=[
        GateResult("turnover_range", False, 0.85, (0.01, 0.7), "")]))
    run = make_run(memory, config, playbook)
    assert run.warm_started == 1
    report = run.run(1)
    assert [r["child"] for r in report["resumed_repairs"]] == ["v3_001_r1"]
    assert memory.decisions_for("v3_001_r1")


def test_result_brain_finished_after_a_timeout_is_recovered(memory, config, playbook, candidate):
    from alpha_research.schemas import DecisionRecord, ResultRecord
    c = candidate(candidate_id="late_001", expression="rank(ts_mean(close, 5))",
                  prompt_version="gen-v6", model="claude-test")
    memory.put_candidate(c)
    memory.put_result(ResultRecord("late_001", "timeout", brain_sim_id="sim_acc_01"))
    memory.put_decision(DecisionRecord("late_001", "reject", "incomplete"))
    brain = StubBrain()
    report = make_run(memory, config, playbook, brain=brain).run(1)
    assert report["recovered_results"] == [
        {"candidate_id": "late_001", "decision": "accept", "failure_class": "none"}]
    assert memory.decisions_for("late_001")[-1].decision == "accept"
    assert "late_001" not in [x.candidate_id for x in brain.submitted]   # no new simulation
    assert memory.unfinished_submissions() == []


def test_resume_can_be_switched_off(memory, config, playbook, candidate):
    _orphan_repair(memory, candidate)
    agents = dataclasses.replace(AGENTS, repair=dataclasses.replace(AGENTS.repair,
                                                                    resume_from_memory=False))
    brain = StubBrain()
    report = make_run(memory, config, playbook, brain=brain, agents=agents).run(1)
    assert report["resumed_repairs"] == []
    assert "old_001_r1" not in [c.candidate_id for c in brain.submitted]


def test_resume_takes_the_closest_repairs_first_up_to_max_resume(memory, config, playbook, candidate):
    from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord
    for cid, tvr in (("far", 2.0), ("near", 0.75)):
        memory.put_candidate(candidate(candidate_id=cid, expression=f"rank(ts_zscore(volume, {len(cid)}))",
                                       decay=4, prompt_version="gen-v6", model="claude-test"))
        memory.put_result(ResultRecord(cid, "complete", sharpe=1.7, turnover=tvr))
        memory.put_decision(DecisionRecord(cid, "repair", "turnover", gate_results=[
            GateResult("turnover_range", False, tvr, (0.01, 0.7), "")]))
    agents = dataclasses.replace(AGENTS, repair=dataclasses.replace(AGENTS.repair, max_resume=1))
    report = make_run(memory, config, playbook, agents=agents).run(1)
    assert [r["parent"] for r in report["resumed_repairs"]] == ["near"]
