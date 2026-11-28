"""Failure review item 1 (2026-11-27, built 2026-11-28): LOW_RETURNS had no repair class (25
CHN failures) and fitness failures got only turnover cuts even when returns were the weak
part (EUR Sharpe 1.99 / fitness 0.82, returns 3.85%, turnover 23%). Now LOW_RETURNS ->
fitness; when returns are weak the turnover cuts are left out and ``emphasize_tails``
(signed_power(rank(x) - 0.5, 2)) is offered; when turnover is the problem it is not."""

from alpha_research import feasibility
from alpha_research.evaluator import check_class
from alpha_research.feasibility import FeasibilityConfig
from alpha_research.repair_agent import MECHANICAL, LeverNotApplicable, lever_kind
from alpha_research.schemas import CandidateRecord, DecisionRecord, GateResult, ResultRecord

from test_rescue_lessons import SPEC

ON = FeasibilityConfig(enabled=True)


def cand(expr):
    return CandidateRecord(candidate_id="c", expression=expr, dataset="d", region="EUR",
                           universe="TOP2500", delay=1, decay=0, neutralization="INDUSTRY",
                           truncation=0.08, start_date=None, end_date=None, hypothesis="h")


def fit_case(sharpe, fitness, returns, turnover, low_returns=None):
    checks = {} if low_returns is None else {"LOW_RETURNS": not low_returns}
    r = ResultRecord("c", "complete", sharpe=sharpe, fitness=fitness, returns=returns,
                     turnover=turnover, checks=checks,
                     check_values={"LOW_FITNESS": {"value": fitness, "limit": 1.0}})
    d = DecisionRecord("c", "repair", "fitness",
                       gate_results=[GateResult("fitness", False, fitness, 1.0, "")])
    return d, r


def blocked(name, d, r, failing=("fitness",)):
    return feasibility.blocked(name, mechanical=True, decision=d, result=r,
                               failing=set(failing), sign_first=False, config=ON)


def test_low_returns_is_a_fitness_failure():
    assert check_class("brain:LOW_RETURNS") == "fitness"
    assert lever_kind("emphasize_tails") == "shape"


def test_emphasize_tails_centres_ranks_and_ranks_other_values():
    f = MECHANICAL["emphasize_tails"]
    a = f(cand("rank(ts_delta(close, 5))"), {"power": 2}, SPEC)
    assert a.expression.replace(" ", "") == "signed_power(rank(ts_delta(close,5))-0.5,2)"
    b = f(cand("x = ts_delta(close, 5); -x"), {"power": 2}, SPEC)
    assert "signed_power(rank(-x)-0.5,2)" in b.expression.replace(" ", "")
    try:
        f(cand("signed_power(rank(close) - 0.5, 2)"), {}, SPEC)
        raise AssertionError("expected not applicable")
    except LeverNotApplicable:
        pass


def test_returns_weak_leaves_out_turnover_cuts_and_offers_tails():
    # EUR case: best turnover cut 1.99 x sqrt(3.85 / 12.5) = 1.10 < 1.2 x bar 1.0
    d, r = fit_case(1.99, 0.82, 0.0385, 0.23)
    assert feasibility.returns_weak(d, r, 1.2)
    for name in ("increase_decay", "wrap_in_smoothing", "hump", "lengthen_ts_window"):
        assert "returns are fitness's weak part" in blocked(name, d, r)
    assert blocked("emphasize_tails", d, r) is None
    assert blocked("neutralize_slow", d, r) is None
    # LOW_RETURNS failing decides it whatever the arithmetic says
    d, r = fit_case(2.5, 0.9, 0.06, 0.40, low_returns=True)
    assert "LOW_RETURNS" in blocked("increase_decay", d, r)


def test_turnover_weak_keeps_turnover_cuts_and_leaves_out_tails():
    # high turnover, good returns: 1.6 x sqrt(12 / 12.5) = 1.57 >= 1.2
    d, r = fit_case(1.6, 0.8, 0.12, 0.60)
    assert not feasibility.returns_weak(d, r, 1.2)
    assert blocked("increase_decay", d, r) is None
    assert "turnover, not returns" in blocked("emphasize_tails", d, r)
    # not a fitness failure at all
    assert "fitness does not fail" in blocked("emphasize_tails", d, r, failing=("sharpe",))
