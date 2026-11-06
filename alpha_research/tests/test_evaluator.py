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
