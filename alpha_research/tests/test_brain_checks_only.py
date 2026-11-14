"""A pass is BRAIN's checks alone (owner, 2026-11-14): our gates take BRAIN's verdict for the
matching check, and our overfitting, pool-diversity and risk checks are recorded, not applied."""

from alpha_research.config import load_config
from alpha_research.evaluator import evaluate
from alpha_research.pipeline import Pipeline

from conftest import MAIN_CONFIG
from test_pipeline_guards import lineage, passing, pnl


def brain_only(make_config):
    return make_config(gates__brain_checks_only=True, gates__use_brain_checks=True,
                       gates__limits_from_brain=True)


def test_brain_verdict_decides_at_the_boundary(make_config, playbook):
    # hand_20261114_gbr_3: Sharpe 1.58, BRAIN's LOW_SHARPE (limit 1.58) PASS, our "> 1.58" FAIL
    r = passing(sharpe=1.58, checks={"LOW_SHARPE": True},
                check_values={"LOW_SHARPE": {"value": 1.58, "limit": 1.58}})
    d = evaluate(r, brain_only(make_config), playbook)
    g = next(g for g in d.gate_results if g.name == "sharpe")
    assert g.passed is True and "LOW_SHARPE PASS decides" in g.note
    assert d.decision == "accept"
    off = evaluate(r, make_config(gates__use_brain_checks=True, gates__limits_from_brain=True),
                   playbook)
    assert next(g for g in off.gate_results if g.name == "sharpe").passed is False


def test_brain_fail_beats_our_pass_and_unreported_checks_keep_our_threshold(make_config,
                                                                         playbook):
    r = passing(fitness=1.2, checks={"LOW_FITNESS": False, "LOW_TURNOVER": True})
    d = evaluate(r, brain_only(make_config), playbook)
    gates = {g.name: g for g in d.gate_results}
    assert gates["fitness"].passed is False
    # turnover needs both LOW_ and HIGH_TURNOVER from BRAIN; with one, our range decides
    assert gates["turnover_range"].note.startswith("0.01 <= turnover")


def test_our_own_checks_are_recorded_not_applied(memory, candidate, make_config, playbook):
    last = lineage(memory, candidate, 39)    # 40 trials: deflated Sharpe 1.3 fails
    result = passing(candidate_id=last, sharpe=1.3, daily_pnl=pnl(5))
    strict = Pipeline(make_config(), playbook, client=None, memory=memory)
    assert strict._evaluate(memory.get_candidate(last), result).decision == "reject"
    loose = Pipeline(brain_only(make_config), playbook, client=None, memory=memory)
    out = loose._evaluate(memory.get_candidate(last), result)
    assert out.decision == "accept"
    assert any(r.startswith("recorded, not applied (gates.brain_checks_only): overfitting")
               for r in out.decision_record.reasons)
    assert any(g.name == "deflated_sharpe" for g in out.decision_record.gate_results)


def test_main_config_passes_on_brain_checks_only():
    g = load_config(MAIN_CONFIG).gates
    assert g.brain_checks_only and g.use_brain_checks and g.limits_from_brain
