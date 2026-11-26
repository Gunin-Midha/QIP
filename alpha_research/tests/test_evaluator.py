import json

import pytest

from alpha_research.evaluator import classify, decide, evaluate, run_gates
from alpha_research.risk import assess
from alpha_research.schemas import GateResult, ResultRecord
from alpha_research.simulator import normalize
from conftest import BRAIN_DIR


def passing(**kw):
    base = dict(candidate_id="c1", status="complete", sharpe=1.68, fitness=1.11, turnover=0.2578,
                returns=0.1119, sub_universe_sharpe=1.07, sub_universe_cutoff=0.73,
                self_correlation=0.1)
    base.update(kw)
    return ResultRecord(**base)


def test_all_pass_accepts(config, playbook):
    d = evaluate(passing(), config, playbook)
    assert (d.decision, d.failure_class) == ("accept", "none")


def test_gate_order(config):
    names = [g.name for g in run_gates(passing(), config)]
    assert names == ["sharpe", "fitness", "turnover_range", "self_correlation", "sub_universe",
                     "weight_concentration", "drawdown"]


def test_sharpe_far_miss_rejects(config, playbook):
    d = evaluate(passing(sharpe=0.50), config, playbook)
    assert (d.decision, d.failure_class) == ("reject", "sharpe")
    assert any("too far from 1.25 to repair" in r for r in d.reasons)


@pytest.mark.parametrize("sharpe", [1.10, 0.75, -0.90])
def test_sharpe_near_miss_repairs(config, playbook, sharpe):
    # near_miss_share.sharpe = 0.6 -> |sharpe| >= 0.75 goes to repair (a sign flip for -0.90)
    d = evaluate(passing(sharpe=sharpe), config, playbook)
    assert (d.decision, d.failure_class) == ("repair", "sharpe")


def test_negative_sharpe_goes_to_the_sign_flip_before_turnover(config, playbook):
    # run rs20261112054223: Sharpe -1.68 with turnover 109% was smoothed, never flipped
    d = evaluate(passing(sharpe=-0.90, turnover=0.85), config, playbook)
    assert (d.decision, d.failure_class) == ("repair", "sharpe")
    assert any(r.startswith("sign first") for r in d.reasons)
    assert "sent to sharpe levers ahead of turnover" in d.reasons
    # a positive near miss keeps the usual priority (turnover first)
    assert evaluate(passing(sharpe=1.10, turnover=0.85), config, playbook).failure_class == \
        "turnover"
    # a far Sharpe miss is not rerouted, and it is rejected even though turnover (the class
    # repaired first) has no near-miss limit (owner 2026-11-25)
    d = evaluate(passing(sharpe=-0.30, turnover=0.85), config, playbook)
    assert (d.decision, d.failure_class) == ("reject", "sharpe")
    assert any("too far" in r for r in d.reasons)


def test_every_failing_class_must_be_a_near_miss(config, playbook):
    # rs20261125130750r01a1_003: Sharpe -0.59, fitness -0.12, turnover 78% was
    # "repair / turnover" because only turnover's (absent) limit was checked
    d = evaluate(passing(sharpe=0.40, fitness=-0.12, turnover=0.78), config, playbook)
    assert (d.decision, d.failure_class) == ("reject", "fitness")   # checked first by name
    # near misses on every limited class: still repaired, primary class first
    d = evaluate(passing(sharpe=1.20, fitness=0.80, turnover=0.78), config, playbook)
    assert (d.decision, d.failure_class) == ("repair", "turnover")


def rn(sharpe):
    return {"risk_neutralized": {"sharpe": sharpe}}


def test_risk_neutralized_sharpe_counts_for_the_near_miss(make_config, playbook):
    config = make_config(repair__risk_neutralized_gap=0.5)
    assert evaluate(passing(sharpe=0.30), config, playbook).decision == "reject"
    d = evaluate(passing(sharpe=0.30, alt_stats=rn(1.20)), config, playbook)
    assert (d.decision, d.failure_class) == ("repair", "sharpe")
    assert any(r.startswith("factor neutralization first") for r in d.reasons)
    # negative raw, far better risk-neutralized: factors first, not the sign
    d = evaluate(passing(sharpe=-0.30, turnover=0.85, alt_stats=rn(1.20)), config, playbook)
    assert d.failure_class == "sharpe"
    assert any(r.startswith("factor neutralization first") for r in d.reasons)
    assert not any(r.startswith("sign first") for r in d.reasons)
    # a raw near miss keeps the usual priority (turnover first); the factor route only takes
    # over when the risk-neutralized Sharpe is what makes the alpha repairable
    d = evaluate(passing(sharpe=1.0, turnover=0.85, alt_stats=rn(2.0)), config, playbook)
    assert d.failure_class == "turnover"
    d = evaluate(passing(sharpe=1.0, alt_stats=rn(2.0)), config, playbook)
    assert any(r.startswith("factor neutralization first") for r in d.reasons)
    # gap off (null): the risk-neutralized value still counts for the near miss, no factor route
    d = evaluate(passing(sharpe=0.30, alt_stats=rn(1.20)), make_config(), playbook)
    assert d.decision == "repair"
    assert not any(r.startswith("factor neutralization first") for r in d.reasons)


def test_fitness_near_miss(config, playbook):
    assert evaluate(passing(fitness=0.73), config, playbook).decision == "repair"
    assert evaluate(passing(fitness=0.40), config, playbook).decision == "reject"


def test_no_near_miss_limit_repairs_every_failure(make_config, playbook):
    config = make_config(repair__near_miss_share={})
    assert evaluate(passing(sharpe=0.10), config, playbook).decision == "repair"


def test_turnover_failure_repairs(config, playbook):
    d = evaluate(passing(turnover=0.85), config, playbook)
    assert (d.decision, d.failure_class) == ("repair", "turnover")


def test_turnover_lower_bound_matters(config, playbook):
    d = evaluate(passing(turnover=0.005), config, playbook)
    assert (d.decision, d.failure_class) == ("repair", "turnover")


def test_self_correlation_failure(config, playbook):
    d = evaluate(passing(self_correlation=0.8), config, playbook)
    assert d.failure_class == "self_correlation"
    assert d.decision == "repair"      # playbook has a self_correlation entry


def test_self_correlation_without_playbook_entry_rejects(config):
    from alpha_research.repair import Playbook
    d = evaluate(passing(self_correlation=0.8), config, Playbook({}))
    assert (d.decision, d.failure_class) == ("reject", "self_correlation")


def test_sub_universe_failure(config, playbook):
    d = evaluate(passing(sub_universe_sharpe=0.5, sub_universe_cutoff=0.73), config, playbook)
    assert d.failure_class == "sub_universe"
    assert d.decision == "repair"


def test_self_correlation_none_is_pending(config, playbook):
    d = evaluate(passing(self_correlation=None), config, playbook)
    assert (d.decision, d.failure_class) == ("pending", "none")
    sc = next(g for g in d.gate_results if g.name == "self_correlation")
    assert sc.passed is None


def test_definite_failure_beats_pending(config, playbook):
    d = evaluate(passing(sharpe=0.5, self_correlation=None), config, playbook)
    assert (d.decision, d.failure_class) == ("reject", "sharpe")
    d = evaluate(passing(sharpe=1.0, self_correlation=None), config, playbook)
    assert (d.decision, d.failure_class) == ("repair", "sharpe")   # near miss


def test_timeout_rejects_incomplete(config, playbook):
    d = evaluate(ResultRecord("c1", "timeout"), config, playbook)
    assert (d.decision, d.failure_class) == ("reject", "incomplete")


def test_priority_order(config, playbook):
    # turnover and sharpe both fail: turnover comes first in the priority list
    d = evaluate(passing(sharpe=1.0, turnover=0.9), config, playbook)
    assert d.failure_class == "turnover"


def test_null_gate_threshold_is_not_configured(make_config, playbook):
    cfg = make_config(gates__sharpe_min=None)
    d = evaluate(passing(), cfg, playbook)
    g = next(g for g in d.gate_results if g.name == "sharpe")
    assert g.passed is None and g.note == "not configured"
    assert d.decision == "pending"


def test_risk_veto_rejects_otherwise_passing(config, playbook):
    r = passing(weight_concentration=0.3)
    veto, checks, reasons = assess(r, config)
    d = evaluate(r, config, playbook, checks, veto, reasons)
    assert (d.decision, d.failure_class) == ("reject", "concentration")
    assert d.risk_veto


def test_screenshot_fixture_accepts_before_risk(make_config, playbook):
    raw = json.loads((BRAIN_DIR / "screenshot_aggregate.json").read_text())
    r = normalize(raw, "shot", "screenshot_aggregate")
    r.self_correlation = 0.1   # not in the screenshot; supplied as in spec 7.4
    cfg = make_config(risk__max_drawdown=None, risk__weight_concentration_max=None)
    veto, checks, reasons = assess(r, cfg)
    assert veto is False
    d = evaluate(r, cfg, playbook, checks, veto, reasons)
    assert d.decision == "accept"


def test_classify_none_when_nothing_failed():
    assert classify([GateResult("sharpe", True)]) == "none"


def test_factor_first_alphas_are_judged_on_risk_neutralized_fitness(playbook):
    # rs20261114174347r01a27_001_c1, the EUR accept's ancestor: raw Sharpe 0.91 / fitness
    # 0.41 / 2-year Sharpe -0.47, risk-neutralized 2.28 / 0.96; its STATISTICAL child reached
    # 2.71 / 1.30. The every-class rule must not reject it.
    from alpha_research.evaluator import decide
    from alpha_research.schemas import GateResult, ResultRecord
    gates = [GateResult("brain:LOW_SHARPE", False, 0.91, 1.58, ""),
             GateResult("brain:LOW_FITNESS", False, 0.41, 1.0, ""),
             GateResult("brain:LOW_2Y_SHARPE", False, -0.47, 1.58, "")]
    shares = {"sharpe": 0.6, "fitness": 0.5, "recent_sharpe": 0.6}
    r = ResultRecord("c1", "complete", sharpe=0.91, fitness=0.41,
                     alt_stats={"risk_neutralized": {"sharpe": 2.28, "fitness": 0.96}})
    d = decide(r, gates, playbook, near_miss_share=shares, risk_neutralized_gap=0.5)
    assert (d.decision, d.failure_class) == ("repair", "sharpe")
    # without BRAIN's risk-neutralized edge the same raw numbers are a far miss
    plain = ResultRecord("c1", "complete", sharpe=0.91, fitness=0.41)
    assert decide(plain, gates, playbook, near_miss_share=shares,
                  risk_neutralized_gap=0.5).decision == "reject"
