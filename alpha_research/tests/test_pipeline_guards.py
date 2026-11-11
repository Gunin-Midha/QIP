"""BRAIN checks as gates, the overfitting guard, distance to accept, lineage budgets and the
diversity safeguards (owner 2026-11-04)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from alpha_research.auditor import Auditor, backtest_years
from alpha_research.critic import Critic
from alpha_research.evaluator import evaluate
from alpha_research.fastexpr import FastExprSpec
from alpha_research.lever_stats import RepairLearningConfig, distance, family_budget
from alpha_research.orchestrator import Orchestrator
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord
from alpha_research.strategy import summarize

from conftest import ROOT

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")


def passing(**kw):
    base = dict(candidate_id="c1", status="complete", sharpe=1.68, fitness=1.11, turnover=0.2578,
                returns=0.1119, sub_universe_sharpe=1.07, sub_universe_cutoff=0.73,
                self_correlation=0.1)
    base.update(kw)
    return ResultRecord(**base)


# ---------------------------------------------------------------- BRAIN checks are gates

def test_a_failed_brain_check_is_a_hard_failure(config, playbook):
    d = evaluate(passing(checks={"CONCENTRATED_WEIGHT": False, "LOW_SHARPE": True}), config, playbook)
    assert (d.decision, d.failure_class) == ("repair", "concentration")
    g = next(g for g in d.gate_results if g.name == "brain:CONCENTRATED_WEIGHT")
    assert g.passed is False and g.note == "BRAIN check FAIL"
    # an unknown failed check is class "other": no playbook entry, so reject
    d = evaluate(passing(checks={"NEW_BRAIN_TEST": False}), config, playbook)
    assert (d.decision, d.failure_class) == ("reject", "other")
    assert evaluate(passing(checks={"CONCENTRATED_WEIGHT": True}), config, playbook).decision == "accept"


def test_brain_checks_can_be_switched_off(config, playbook):
    off = dataclasses.replace(config, gates=dataclasses.replace(config.gates, use_brain_checks=False))
    d = evaluate(passing(checks={"CONCENTRATED_WEIGHT": False}), off, playbook)
    assert d.decision == "accept" and not any(g.name.startswith("brain:") for g in d.gate_results)


def test_consultant_checks_map_to_classes(config, playbook):
    # live consultant checks (2026-11-10): recent-years Sharpe and correlation with
    # production alphas are classes of their own, each with its own levers
    d = evaluate(passing(checks={"LOW_2Y_SHARPE": False}), config, playbook)
    assert d.failure_class == "recent_sharpe"
    d = evaluate(passing(checks={"IS_LADDER_SHARPE": False}), config, playbook)
    assert d.failure_class == "recent_sharpe"
    d = evaluate(passing(checks={"PROD_CORRELATION": False}), config, playbook)
    assert d.failure_class == "prod_correlation"


def test_ignored_brain_checks_are_not_gates(config, playbook):
    # REGULAR_SUBMISSION is the daily submission quota: failing it says nothing about the alpha
    ignoring = dataclasses.replace(config, gates=dataclasses.replace(
        config.gates, brain_checks_ignored=("REGULAR_SUBMISSION",)))
    d = evaluate(passing(checks={"REGULAR_SUBMISSION": False, "LOW_SHARPE": True}), ignoring, playbook)
    assert d.decision == "accept"
    assert not any(g.name == "brain:REGULAR_SUBMISSION" for g in d.gate_results)
    d = evaluate(passing(checks={"REGULAR_SUBMISSION": False}), config, playbook)
    assert d.decision == "reject"


def test_main_config_ignores_submission_only_checks():
    from alpha_research.config import load_config
    g = load_config(ROOT / "config" / "thresholds.yaml").gates
    assert g.sharpe_min == 1.58 and g.use_brain_checks
    assert {"REGULAR_SUBMISSION", "MATCHES_THEMES", "OSMOSIS_ALLOCATION"} <= set(g.brain_checks_ignored)
    for alpha_check in ("LOW_SHARPE", "LOW_2Y_SHARPE", "SELF_CORRELATION", "PROD_CORRELATION"):
        assert alpha_check not in g.brain_checks_ignored


# ---------------------------------------------------------------- overfitting guard

def lineage(memory, candidate, n_children, root_id="root"):
    memory.put_candidate(candidate(candidate_id=root_id))
    memory.put_result(ResultRecord(root_id, "complete", sharpe=1.0))
    parent = root_id
    for i in range(n_children):
        cid = f"{root_id}_k{i}"
        memory.put_candidate(candidate(candidate_id=cid, parent_id=parent, lever="hump",
                                       generation=1, decay=10 + i))
        memory.put_result(ResultRecord(cid, "complete", sharpe=1.0))
    return f"{root_id}_k{n_children - 1}" if n_children else root_id


def pnl(years):
    return pd.Series(np.zeros(int(252 * years)), index=pd.bdate_range("2019-01-01", periods=int(252 * years)))


def test_deflated_sharpe_counts_trials_by_scope(memory, candidate, config):
    last = lineage(memory, candidate, 39)
    lone = lineage(memory, candidate, 0, root_id="solo")
    memory.put_candidate(candidate(candidate_id="bad", dataset="other"))
    memory.put_result(ResultRecord("bad", "invalid"))                # invalid: never a trial
    lin = dataclasses.replace(config, auditor=dataclasses.replace(config.auditor,
                                                                  trials_scope="lineage"))
    aud = Auditor(memory, lin)
    assert aud.trials(last) == aud.trials("root") == 40 and aud.trials(lone) == 1
    weak = aud.check_accept(last, passing(candidate_id=last, sharpe=1.3, daily_pnl=pnl(5)))
    assert weak.passed is False and "40 simulation(s) (scope lineage)" in weak.note
    ok = aud.check_accept(lone, passing(candidate_id=lone, sharpe=1.3, daily_pnl=pnl(5)))
    assert ok.passed is True and ok.value == 1.3   # one trial: nothing to deflate
    # default scope "dataset": the lone alpha shares pv1 with the 40 others
    by_ds = Auditor(memory, config)
    assert config.auditor.trials_scope == "dataset" and by_ds.trials(lone) == 41
    assert by_ds.check_accept(lone, passing(candidate_id=lone, sharpe=1.3,
                                            daily_pnl=pnl(5))).passed is False
    assert Auditor(memory, dataclasses.replace(config, auditor=dataclasses.replace(
        config.auditor, trials_scope="all"))).trials(lone) == 41


def test_overfit_accept_is_rejected_by_the_pipeline(memory, candidate, config, playbook):
    from alpha_research.pipeline import Pipeline
    last = lineage(memory, candidate, 39)
    p = Pipeline(config, playbook, client=None, memory=memory)
    d = DecisionRecord(last, "accept", "none", reasons=["all hard gates passed"])
    out = p._overfitting(memory.get_candidate(last), passing(candidate_id=last, sharpe=1.3,
                                                             daily_pnl=pnl(5)), d)
    assert (out.decision, out.failure_class) == ("reject", "other")
    assert any(r.startswith("overfitting: deflated Sharpe") for r in out.reasons)
    off = dataclasses.replace(config, auditor=dataclasses.replace(config.auditor,
                                                                  min_deflated_sharpe=None))
    kept = Pipeline(off, playbook, client=None, memory=memory)._overfitting(
        memory.get_candidate(last), passing(candidate_id=last, sharpe=1.3, daily_pnl=pnl(5)), d)
    assert kept.decision == "accept"


def test_backtest_years():
    assert backtest_years(passing(daily_pnl=pnl(5))) == pytest.approx(5.0)
    assert backtest_years(passing()) is None


# ---------------------------------------------------------------- distance to accept

def gate(name, passed, value, threshold):
    return GateResult(name, passed, value, threshold, "")


def test_distance_sums_normalized_shortfalls():
    d = DecisionRecord("c", "repair", "sub_universe", gate_results=[
        gate("sharpe", False, 1.0, 1.25), gate("sub_universe", False, 0.5, 1.0),
        gate("brain:LOW_SUB_UNIVERSE_SHARPE", False, 0.5, 1.0),
        gate("turnover_range", False, 0.84, (0.01, 0.7))])
    r = ResultRecord("c", "complete", checks={"CONCENTRATED_WEIGHT": False})
    # sharpe 0.2 + sub-universe 0.5 (counted once) + turnover 0.2 + concentration penalty 0.5
    assert distance(d, r) == pytest.approx(1.4)
    assert distance(DecisionRecord("c", "accept", "none"), r) == 0.0
    assert distance(d, ResultRecord("c", "timeout")) is None


# ---------------------------------------------------------------- lineage budget

def store(memory, cand, decision, gates, sharpe=1.0):
    memory.put_candidate(cand)
    memory.put_result(ResultRecord(cand.candidate_id, "complete", sharpe=sharpe))
    memory.put_decision(DecisionRecord(cand.candidate_id, decision, "sharpe", gate_results=gates))


def test_family_budget_depth_trials_and_patience(memory, candidate):
    cfg = RepairLearningConfig(max_depth=2, max_trials_per_root=3, patience=2)
    root = candidate(candidate_id="r")
    store(memory, root, "repair", [gate("sharpe", False, 1.0, 1.25)])
    left, why = family_budget(root, memory.candidate_outcomes(), cfg)
    assert (left, why) == (3, None)
    assert family_budget(candidate(candidate_id="x", generation=2), [], cfg)[1].startswith(
        "repair depth limit")
    # two children that do not get closer than the root (distance 0.2) close the lineage
    for i, v in enumerate((0.9, 0.95)):
        store(memory, candidate(candidate_id=f"r_c{i}", parent_id="r", lever="hump", generation=1,
                                decay=20 + i), "repair", [gate("sharpe", False, v, 1.25)])
    left, why = family_budget(root, memory.candidate_outcomes(), cfg)
    assert left == 0 and "stopped improving" in why
    # one that improves resets patience; the third repair simulation uses up the budget
    store(memory, candidate(candidate_id="r_c2", parent_id="r", lever="hump", generation=1,
                            decay=30), "repair", [gate("sharpe", False, 1.2, 1.25)])
    left, why = family_budget(root, memory.candidate_outcomes(), cfg)
    assert left == 0 and "used its repair budget: 3 of 3" in why


def test_repair_agent_respects_the_budget(memory, candidate, playbook):
    from alpha_research.repair_agent import RepairAgent
    cfg = RepairLearningConfig(max_trials_per_root=1)
    root = candidate(candidate_id="r", decay=4)
    store(memory, root, "repair", [gate("turnover_range", False, 0.9, (0.01, 0.7))])
    store(memory, candidate(candidate_id="r_r1", parent_id="r", lever="increase_decay",
                            generation=1, decay=8), "repair",
          [gate("turnover_range", False, 0.8, (0.01, 0.7))])
    d = DecisionRecord("r_r1", "repair", "turnover")
    out = RepairAgent(playbook, SPEC, memory, learning=cfg).repair(memory.get_candidate("r_r1"), d)
    assert out.child is None and "used its repair budget" in out.notes[0]


# ---------------------------------------------------------------- diversity

def test_director_caps_one_arms_share_of_a_round():
    orch = Orchestrator(["hot", "cold1", "cold2"], seed=0)
    for _ in range(30):
        orch.update("hot", True)
        orch.update("cold1", False)
        orch.update("cold2", False)
    assert orch.allocate(10, 0.0)["hot"] >= 9                    # no cap: it takes the round
    capped = orch.allocate(10, 0.0, max_share=0.4)
    assert capped["hot"] == 4 and sum(capped.values()) == 10


def test_strategy_summary_shows_untried_datasets_and_working_fixes(memory, candidate):
    store(memory, candidate(candidate_id="a", dataset="option8"), "accept", [])
    s = summarize(memory.candidate_outcomes(), known_datasets=["option8", "news12", "analyst4"])
    assert s["datasets"] == {"option8": 1}
    assert s["least_tried"] == ["analyst4", "news12", "option8"]
    assert s["what_fixes_worked"] == []


def test_critic_flags_neutralization_none_without_blocking(candidate):
    from alpha_research.critic import ReviewOutcome
    reviews = Critic(SPEC).review([candidate(neutralization="NONE", expression="rank(close)")],
                                  {}, ReviewOutcome())
    r = reviews["c1"]
    assert r.verdict == "simulate" and any("neutralization NONE" in i for i in r.issues)
