"""Seeds: earlier alphas re-simulated first in a research run, then repaired."""

import dataclasses

import pytest
import yaml

from alpha_research.config import ConfigError
from alpha_research.research import SeedSpec, load_seeds
from alpha_research.schemas import DecisionRecord, ResultRecord

from conftest import ROOT
from test_research import AGENTS, StubBrain, make_run
from test_rotation import rotating_run

V2 = ROOT / "config" / "research_plan.v2.yaml"


def write(tmp_path, seeds, regions=None):
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({
        "defaults": {"truncation": 0.08, "start_date": "2014-01-01", "end_date": "2023-12-31"},
        "regions": regions or {"USA": {"settings": {"universe": "TOP3000",
                                                    "neutralization": "SUBINDUSTRY", "decay": 4},
                                       "arms": [{"dataset": "pv1", "delay": 1,
                                                 "hypotheses": ["h"]}]}},
        "seeds": seeds}))
    return p


def test_the_v2_plan_seeds_both_earlier_alphas():
    iv, skew = load_seeds(V2)
    assert iv == SeedSpec(from_memory="rs20261103184506r01a0_002_r1x")
    s = skew.settings
    assert (s.region, s.universe, s.delay, s.decay, s.truncation, s.neutralization, s.dataset) == \
        ("USA", "TOP3000", 1, 0, 0.01, "INDUSTRY", "option8")
    assert skew.expression.startswith("skew = winsorize(") and "\n" not in skew.expression


def test_inline_seed_inherits_region_settings(tmp_path):
    [s] = load_seeds(write(tmp_path, [{"dataset": "pv1", "region": "USA", "delay": 1,
                                       "expression": "rank(-ts_delta(close, 5))"}]))
    assert (s.settings.decay, s.settings.neutralization, s.settings.truncation) == \
        (4, "SUBINDUSTRY", 0.08)


def test_a_plan_without_seeds_has_none(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump({"arms": []}))
    assert load_seeds(p) == []


@pytest.mark.parametrize("seeds, match", [
    ({"from_memory": "x"}, "must be a list"),
    ([{"from_memory": ""}], "candidate id"),
    ([{"dataset": "pv1", "region": "USA", "delay": 1}], "'from_memory' or an 'expression'"),
    ([{"dataset": "pv1", "region": "MARS", "delay": 1, "expression": "rank(close)"}], "MARS"),
    ([{"dataset": "pv1", "region": "GLB", "delay": 0, "expression": "rank(close)"}], "delay 0"),
])
def test_seed_errors(tmp_path, seeds, match):
    with pytest.raises(ConfigError, match=match):
        load_seeds(write(tmp_path, seeds))


def _old_accept(memory, candidate):
    old = candidate(candidate_id="old_acc", expression="rank(-ts_delta(close, 13))",
                    decay=10, start_date="2016-01-01", prompt_version="gen-v1", model="gemini-x",
                    parent_id="old_parent", generation=1, lever="liquidity_tilt")
    memory.put_candidate(old)
    memory.put_result(ResultRecord("old_acc", "complete", sharpe=1.38))
    memory.put_decision(DecisionRecord("old_acc", "accept", "none"))


def test_a_memory_seed_is_simulated_first_as_a_new_root(memory, config, playbook, candidate):
    _old_accept(memory, candidate)
    brain = StubBrain()
    run = make_run(memory, config, playbook, brain=brain)
    run.seed_specs = [SeedSpec(from_memory="old_acc")]
    report = run.run(1)
    first = brain.submitted[0]
    assert first.candidate_id == "old_acc_seed" and first.expression == "rank(-ts_delta(close, 13))"
    assert (first.parent_id, first.generation, first.lever, first.decay) == (None, 0, None, 10)
    assert (first.prompt_version, first.model) == ("gen-v5", "claude-test")   # its arm's
    assert first.start_date == run.arms[next(iter(run.arms))].settings.start_date
    assert report["seeds"][0]["status"].startswith(("accept", "reject", "repair", "pending"))
    assert run.log[0]["arms"][0]["seed"] == "old_acc_seed"
    # the Director is not credited with the seed's outcome
    key = next(iter(run.arms))
    s = run.director.state[key]
    assert s.successes + s.failures == sum(
        1 for a in run.log[0]["arms"][1:] for o in a.get("outcomes", [])
        if o["decision"] in ("accept", "reject", "repair"))


def test_a_seed_simulated_before_is_not_simulated_again(memory, config, playbook, candidate):
    _old_accept(memory, candidate)
    first = make_run(memory, config, playbook)
    first.seed_specs = [SeedSpec(from_memory="old_acc")]
    first.run(1)
    brain = StubBrain()
    again = make_run(memory, config, playbook, brain=brain)
    again.seed_specs = [SeedSpec(from_memory="old_acc")]
    report = again.run(1)
    assert "old_acc_seed" not in [c.candidate_id for c in brain.submitted]
    assert report["seeds"] == [{"seed": "old_acc", "candidate_id": "old_acc_seed",
                                "status": "simulated in an earlier run"}]


def test_a_seed_that_is_the_same_experiment_is_reported(memory, config, playbook, candidate):
    old = candidate(candidate_id="same", expression="rank(-ts_delta(close, 13))")
    memory.put_candidate(old)                       # same settings and dates as the arm
    memory.put_result(ResultRecord("same", "complete"))
    run = make_run(memory, config, playbook)
    run.seed_specs = [SeedSpec(from_memory="same")]
    assert run.run(1)["seeds"][0]["status"].startswith("same experiment already simulated")


def test_seed_problems_are_reported_not_simulated(memory, config, playbook, tmp_path):
    p = write(tmp_path, [{"from_memory": "nobody"},
                         {"dataset": "pv1", "region": "USA", "delay": 1,
                          "expression": "rank(opt6_ivhvxernratio)"},
                         {"dataset": "pv1", "region": "EUR", "delay": 1,
                          "settings": {"universe": "TOP2500", "neutralization": "INDUSTRY",
                                       "decay": 4},
                          "expression": "rank(close)"}])
    brain = StubBrain()
    run = make_run(memory, config, playbook, brain=brain)
    run.seed_specs = load_seeds(p)
    run.markets = {("USA", 1): (["close", "volume"], {})}
    run._mtyped = {("USA", 1): {"close": "MATRIX", "volume": "MATRIX"}}
    report = run.run(1)
    notes = report["seeds"]
    assert notes[0] == {"seed": "nobody", "status": "not in memory"}
    assert notes[1]["status"].startswith("invalid: unknown datafield 'opt6_ivhvxernratio'")
    assert notes[2]["status"] == "no arm in EUR delay 1 in this plan"
    assert not any(c.candidate_id.startswith("seed_") for c in brain.submitted)


def test_a_seed_waits_for_a_run_with_its_region(memory, config, playbook, candidate):
    old = candidate(candidate_id="jpn_acc", region="JPN", universe="TOP1600",
                    neutralization="INDUSTRY", expression="rank(-ts_delta(close, 13))")
    memory.put_candidate(old)
    brain = StubBrain()
    run = rotating_run(memory, config, playbook, brain=brain)   # USA and EUR this run
    run.seed_specs = [SeedSpec(from_memory="jpn_acc")]
    report = run.run(1)
    assert "jpn_acc_seed" not in [c.candidate_id for c in brain.submitted]
    assert report["seeds"][0]["status"] == "queued"
    later = rotating_run(memory, config, playbook, brain=(b2 := StubBrain()), run_id="rs2")
    later.seed_specs = [SeedSpec(from_memory="jpn_acc")]
    assert "JPN" in later.run_regions
    later.run(1)
    assert b2.submitted[0].candidate_id == "jpn_acc_seed"
