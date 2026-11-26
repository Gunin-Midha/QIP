"""Fixes from run rs20261126082238 (owner 2026-11-26): a flipped child is where repair goes
on; a negative Sharpe whose risk-neutralized value is only less negative is flipped first;
rank_transform does not re-rank a combination of ranks."""

import pytest

from alpha_research.evaluator import FACTOR_FIRST, SIGN_FIRST, sharpe_route
from alpha_research.lever_stats import distance
from alpha_research.repair_agent import MECHANICAL, LeverNotApplicable
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord

from test_repair_waves import agents
from test_research import make_run
from test_rescue_lessons import SPEC

SHARES = {"sharpe": 0.6, "fitness": 0.5}


def low(v):
    return GateResult("brain:LOW_SHARPE", False, v, 1.58, "")


def rn(raw, neutral):
    return ResultRecord("c1", "complete", sharpe=raw,
                        alt_stats={"risk_neutralized": {"sharpe": neutral}})


def test_less_negative_risk_neutralized_sharpe_is_flipped_first():
    # GBR r01a202_001_c1_r2d: raw -1.71, risk-neutralized -1.11 (its flip: 1.71 / 1.61)
    assert sharpe_route([low(-1.71)], rn(-1.71, -1.11), SHARES, 0.5).startswith(SIGN_FIRST)
    # a risk-neutralized Sharpe that turns positive is still a factor problem
    assert sharpe_route([low(-1.0)], rn(-1.0, 1.2), SHARES, 0.5).startswith(FACTOR_FIRST)
    # positive raw: unchanged
    assert sharpe_route([low(1.0)], rn(1.0, 1.6), SHARES, 0.5).startswith(FACTOR_FIRST)
    # where a flip does not mirror (CHN) nothing changes
    assert not (sharpe_route([low(-1.71)], rn(-1.71, -1.11), SHARES, 0.5, sign_mirrors=False)
                or "").startswith(SIGN_FIRST)


def test_next_wave_starts_from_the_flipped_child(memory, make_config, playbook, candidate):
    run = make_run(memory, make_config(), playbook, agents=agents())
    gates = [low(-1.54)]
    pc = candidate(candidate_id="p")
    pd = DecisionRecord("p", "repair", "sharpe", gate_results=gates,
                        reasons=[f"{SIGN_FIRST}: Sharpe -1.54 is negative"])
    pr = ResultRecord("p", "complete", sharpe=-1.54)
    kc = candidate(candidate_id="p_r1a", lever="flip_sign", parent_id="p")
    kd = DecisionRecord("p_r1a", "repair", "sharpe", gate_results=[low(1.54)])
    kr = ResultRecord("p_r1a", "complete", sharpe=1.54)
    assert distance(kd, kr) == distance(pd, pr)          # a flip mirrors: same distance
    c, *_ = run._next_start((pc, pd, pr), [(kc, kd, kr)])
    assert c.candidate_id == "p_r1a"
    # another lever at the same distance does not take over from the parent
    oc = candidate(candidate_id="p_r1b", lever="hump", parent_id="p")
    c, *_ = run._next_start((pc, pd, pr), [(oc, kd, kr)])
    assert c.candidate_id == "p"


@pytest.mark.parametrize("expr", [
    "rank(ts_backfill(a, 20)) + 0.5 * rank(ts_backfill(b, 60))",
    "x = group_rank(close, sector); y = rank(volume); -(x + y) / 2",
    "rank(close) - rank(open)",
])
def test_rank_transform_skips_combinations_of_ranks(candidate, expr):
    with pytest.raises(LeverNotApplicable, match="already ranks"):
        MECHANICAL["rank_transform"](candidate(expression=expr), {}, SPEC)


def test_rank_transform_still_ranks_raw_values(candidate):
    out = MECHANICAL["rank_transform"](candidate(expression="rank(close) * volume"), {}, SPEC)
    assert out.expression == "rank(rank(close) * volume)"
