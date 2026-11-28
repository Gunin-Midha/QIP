"""Arm quality fixes (owner 2026-11-28, after run rs20261128045753: 43 alphas, all rejected,
mean |Sharpe| 0.31; 31 of them on datasets with 2-42 BRAIN alphas while JPN other176 and GLB
pv1, the datasets with our best results, got no slot):

1. the Arm Designer picks thinly used datasets (fewer than ``min_brain_alphas`` BRAIN alphas)
   only as its exploration arm; ``uncrowded`` weighs 0;
2. the Director gives arms with evidence (our best |Sharpe| >= ``evidence_sharpe``) at least
   ``evidence_floor_share`` of a market's slots, and an arm added during a run gets its own
   record from memory, else the market's average record;
3. a thinly used dataset is dead after ``thin_dead_after`` weak alphas."""

import dataclasses

import pytest
import yaml

from alpha_research import arm_designer as ad
from alpha_research.config import ConfigError
from alpha_research.generator import GenerationSettings, Source
from alpha_research.orchestrator import Orchestrator
from alpha_research.research import ArmPlan, agents_config_from_dict
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord

from conftest import ROOT
from test_arm_designer import cands, outcome
from test_rotation import rotating_run

CFG = ad.DesignerConfig(enabled=True, arms_per_batch=3, done_after_alphas=2, explore_arms=1,
                        min_brain_alphas=300, thin_dead_after=4, dead_after_alphas=12,
                        weights={**ad.DesignerConfig().weights, "uncrowded": 0.0})


# ------------------------------------------------------------------ 1. designer usage floor

def test_thin_datasets_only_as_the_exploration_arm(candidate):
    c = {**cands(),
         "rare_news": {"category": "news", "value_score": 9, "alpha_count": 20,
                       "signal_fields": 30},
         "used_news": {"category": "news", "value_score": 3, "alpha_count": 2000,
                       "signal_fields": 30}}
    c = {k: v for k, v in c.items() if v["signal_fields"] >= CFG.min_signal_fields}
    outs = [outcome(candidate, "a1", "rare_model", 0.2), outcome(candidate, "a2", "rare_news", 0.1)]
    picked = ad.choose_datasets(ad.score_datasets(c, outs, "USA", 1, 1.58, CFG), CFG)
    names = [p["dataset"] for p in picked]
    # the never-tried dataset is the exploration arm; every other pick has >= 300 BRAIN alphas
    assert picked[0]["why"].startswith("exploration")
    assert all(p["brain_alphas"] >= 300 for p in picked[1:])
    assert "rare_model" not in names and "rare_news" not in names
    assert "used_news" in names
    # off when min_brain_alphas is 0 (the default)
    assert not ad.thin({"brain_alphas": 1}, ad.DesignerConfig())
    assert ad.thin({"brain_alphas": None}, CFG) and not ad.thin({"brain_alphas": 300}, CFG)


def test_main_config_drops_the_uncrowded_bonus_and_sets_the_floors():
    raw = yaml.safe_load((ROOT / "config" / "agents.yaml").read_text())
    a = agents_config_from_dict(raw)
    assert a.designer.weights["uncrowded"] == 0.0
    assert a.designer.min_brain_alphas == 300 and a.designer.thin_dead_after == 4
    assert a.evidence_floor_share == 0.4 and a.evidence_sharpe == 1.0
    assert a.new_arm_prior == "market"
    raw["director"]["new_arm_prior"] = "best"
    with pytest.raises(ConfigError, match="new_arm_prior"):
        agents_config_from_dict(raw)


# ------------------------------------------------------------------ 3. thin datasets die early

def test_thin_datasets_die_after_four_weak_alphas_used_ones_after_twelve(candidate):
    outs = [outcome(candidate, f"t{i}", "thin_ds", 0.3) for i in range(4)] + \
           [outcome(candidate, f"u{i}", "used_ds", 0.3) for i in range(4)]
    counts = {"thin_ds": 20, "used_ds": 5000}
    dead = ad.dead_datasets(outs, "USA", 1, 1.58, CFG, counts)
    assert set(dead) == {"thin_ds"} and dead["thin_ds"] == (4, 0.3)
    # a thin dataset with one decent alpha stays
    outs[0] = outcome(candidate, "t0", "thin_ds", 0.9)
    assert "thin_ds" not in ad.dead_datasets(outs, "USA", 1, 1.58, CFG, counts)
    # without counts, only the general rule (12)
    assert ad.dead_datasets(outs, "USA", 1, 1.58, CFG) == {}


# ------------------------------------------------------------------ 2. Director

def test_orchestrator_starts_an_arm_from_a_given_record():
    o = Orchestrator([("a", "USA", "pv", "m", 1)], seed=0)
    for _ in range(9):
        o.update(("a", "USA", "pv", "m", 1), False)
    assert o.market_record("USA", 1) == (0.0, 9.0)
    o.add_arm(("b", "USA", "pv", "m", 1), prior=(0.5, 4.0))
    assert o.pass_rate(("b", "USA", "pv", "m", 1)) == pytest.approx(1.5 / 6.5)
    assert o.market_record("EUR") is None


def _store(memory, candidate, cid, dataset, region, sharpe, decision="reject", model="claude-test"):
    from test_rotation import MARKETS
    universe, neut = MARKETS[region]
    memory.put_candidate(candidate(candidate_id=cid, dataset=dataset, region=region,
                                   universe=universe, neutralization=neut,
                                   prompt_version="gen-v18", model=model,
                                   expression=f"rank(ts_mean(volume, {len(cid) + 2}))"))
    memory.put_result(ResultRecord(cid, "complete", sharpe=sharpe))
    memory.put_decision(DecisionRecord(cid, decision, "sharpe", gate_results=[
        GateResult("sharpe", decision == "accept", sharpe, 1.58, "")]))


def _plan(dataset, region="USA"):
    s = GenerationSettings.from_mapping({
        "region": region, "universe": "TOP3000", "delay": 1, "decay": 4,
        "neutralization": "SUBINDUSTRY", "truncation": 0.08, "start_date": "2014-01-01",
        "end_date": "2023-12-31", "dataset": dataset})
    from collections import deque
    return ArmPlan(dataset, s, deque([Source("hypothesis", "idea")]))


def test_an_arm_added_during_a_run_gets_its_own_record_else_the_markets(memory, config,
                                                                      playbook, candidate):
    for i in range(6):     # a dataset rejected six times before, with no arm at the start
        _store(memory, candidate, f"old{i}", "thin_ds", "USA", 0.2)
    run = rotating_run(memory, config, playbook, k=None)
    run.agents = dataclasses.replace(run.agents, new_arm_prior="market")
    pv = next(k for k in run.director.arms if k[1] == "USA")
    for _ in range(4):
        run.director.update(pv, False)
    run.director.update(pv, True)
    old_key = ("thin_ds", "USA", "gen-v18", "claude-test", 1)
    new_key = ("never_ds", "USA", "gen-v18", "claude-test", 1)
    run._add_arm(old_key, _plan("thin_ds"), [])
    run._add_arm(new_key, _plan("never_ds"), [])
    run._seed_new_arms()
    st = run.director.state
    assert (st[old_key].successes, st[old_key].failures) == (0, 6)     # its own record
    assert st[new_key].failures > 0 and st[new_key].successes > 0      # the market's average


def test_arms_with_evidence_get_the_floor_share(memory, config, playbook, candidate):
    _store(memory, candidate, "good1", "pv1", "USA", 1.4)            # evidence: |Sharpe| >= 1
    run = rotating_run(memory, config, playbook, k=None)
    run.agents = dataclasses.replace(run.agents, evidence_floor_share=0.4, evidence_sharpe=1.0)
    usa_pv = [k for k in run.director.arms if k[1] == "USA"]
    others = []
    for i in range(5):     # five untried arms with blank records
        key = (f"new{i}", "USA", "gen-v18", "claude-test", 1)
        run._add_arm(key, _plan(f"new{i}"), [])
        others.append(key)
    for k in usa_pv:       # the evidence arm has a poor record
        for _ in range(8):
            run.director.update(k, False)
    arms = usa_pv + others
    got = run._director_allocate(10, arms)
    assert sum(got[k] for k in usa_pv) >= 4 and sum(got.values()) == 10
    run.agents = dataclasses.replace(run.agents, evidence_floor_share=None)
    run._evidence = None
    assert run._evidence_keys(arms) == set()
