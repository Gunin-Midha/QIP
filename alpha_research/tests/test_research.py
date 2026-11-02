"""The closed research loop, end to end, with a routing fake model and a stub BRAIN client."""

import json
import re
from collections import deque

import pytest

from alpha_research import cli
from alpha_research.brain_client import FixtureBrainClient
from alpha_research.config import ConfigError
from alpha_research.fastexpr import FastExprSpec
from alpha_research.generator import GenerationSettings, Source, load_generator_config
from alpha_research.llm import ScriptedLLMClient
from alpha_research.research import (ArmPlan, Budget, BudgetedLLM, LLMBudgetExhausted,
                                     ResearchRun, agents_config_from_dict, load_agents_config,
                                     load_plan)
from alpha_research.simulator import VirtualClock

from conftest import BRAIN_DIR, ROOT, TEST_CONFIG

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
GEN_CONFIG = load_generator_config(ROOT / "config" / "generator.yaml")
AGENTS = load_agents_config(ROOT / "config" / "agents.yaml")
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
                       llm=llm or RouterLLM(), client=brain or StubBrain(), config=config,
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
    assert evo["prompt_version"] == "gen-v1+evo1" and evo["source"] == "model"
    assert "pv1/USA/gen-v1+evo1" in report["arms"]
    new_key = ("pv1", "USA", "gen-v1+evo1")
    assert run.arms[new_key].guidance == "avoid volume spikes without decay"
    assert "evolved idea" in [s.text for s in run.arms[new_key].sources]
    # the director learned from the outcomes
    base = report["arms"]["pv1/USA/gen-v1"]
    assert base["accepted"] >= 2 and base["accepted"] + base["not_accepted"] >= 3
    assert report["pool_size"] >= 2
    assert set(llm.calls) >= {"generator", "critic", "judge", "strategy"}


def test_guidance_reaches_the_generator_prompt(memory, config, playbook):
    # 10 slots over 2 arms in round 3: the floor guarantees the evolved arm at least 1
    llm = RouterLLM()
    run = make_run(memory, config, playbook, llm=llm,
                   budget=Budget(max_simulations=40, per_round=10, max_llm_calls=100))
    run.run(3)
    gen = run.generators[("pv1", "USA", "gen-v1+evo1")]
    assert gen.config.prompt_version == "gen-v1+evo1"
    assert sum("avoid volume spikes without decay" in s for s in llm.systems) >= 1
    assert "avoid volume spikes" not in llm.systems[0]
    evolved = [r for r in memory.candidate_outcomes()
               if r["candidate"].prompt_version == "gen-v1+evo1"]
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
    key = ("pv1", "USA", "gen-v1")
    assert second.director.state[key].successes == first.director.state[key].successes


def test_rules_only_agents(memory, config, playbook):
    raw = json.loads(json.dumps({
        "prefilter": {"near_duplicate_similarity": 0.9, "use_model": False, "min_model_score": None},
        "critic": {"use_model": False}, "debate": {"use_model": False}, "repair": {"use_model": False},
        "strategy": {"every_rounds": 0, "max_hypotheses": 3, "max_guidance_chars": 100},
        "director": {"floor_share": 0.2}, "max_output": 1024}))
    llm = RouterLLM()
    report = make_run(memory, config, playbook, llm=llm, agents=agents_config_from_dict(raw)).run(2)
    assert set(llm.calls) == {"generator"} and report["simulations_used"] > 0


def test_budgeted_llm_counts_and_stops():
    b = BudgetedLLM(ScriptedLLMClient(["a", "b", "c"]), max_calls=2)
    assert [b.complete("s", "u"), b.complete("s", "u")] == ["a", "b"]
    with pytest.raises(LLMBudgetExhausted):
        b.complete("s", "u")
    assert b.calls == 2 and b.model == "scripted"


# ---------------------------------------------------------------- config and plan


def test_main_agents_config_loads():
    assert AGENTS.prefilter.near_duplicate_similarity == 0.9 and AGENTS.critic_model
    assert AGENTS.strategy.every_rounds == 2 and AGENTS.floor_share == 0.2


@pytest.mark.parametrize("patch, match", [
    ({"critic": None}, "missing section 'critic'"),
    ({"prefilter": {"near_duplicate_similarity": 2, "use_model": False}}, r"\[0, 1\]"),
    ({"debate": {"use_model": "yes"}}, "true or false"),
    ({"director": {"floor_share": 1.5}}, "floor_share"),
    ({"max_output": 10}, "max_output"),
])
def test_agents_config_errors(patch, match):
    import yaml
    raw = yaml.safe_load((ROOT / "config" / "agents.yaml").read_text())
    raw.update(patch)
    with pytest.raises(ConfigError, match=match):
        agents_config_from_dict(raw)


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
    monkeypatch.setattr(cli, "make_session", lambda cfg: None)
    monkeypatch.setattr(cli, "RealBrainClient", lambda session, cfg, capture_dir=None: brain)
    monkeypatch.setattr("alpha_research.pipeline.time.sleep", lambda s: None, raising=False)
    agents = yaml.safe_load((ROOT / "config" / "agents.yaml").read_text())
    for name in ("critic", "debate", "repair"):
        agents[name]["use_model"] = False
    agents["strategy"]["every_rounds"] = 0
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
    assert "stopped: simulation budget used up" in out and "pv1/USA/gen-v1: accepted 2" in out


def test_cli_research_needs_a_holdout(tmp_path):
    plan = write_plan(tmp_path, [{"dataset": "pv1", "settings": SETTINGS, "hypotheses": ["h"]}])
    rc = cli.main(["research", "--plan", str(plan), "--max-simulations", "1", "--max-llm-calls", "1",
                   "--config", str(ROOT / "config" / "thresholds.yaml"),
                   "--data-dir", str(tmp_path / "d")])
    assert rc == cli.EXIT_CONFIG
