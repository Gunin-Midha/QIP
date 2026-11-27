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


def test_sign_first_wave_is_the_flip_alone_and_nothing_once_it_was_tried(memory, playbook,
                                                                          candidate):
    # run rs20261127123332: after the flip of a USA short-interest alpha (Sharpe -1.03,
    # fitness -0.20) was tried, the next wave on the negative parent added smoothing,
    # rank_transform and a combo from the fitness/sub-universe classes (all came back negative)
    from alpha_research.lever_stats import LeverStats
    from alpha_research.repair_agent import RepairAgent
    gates = [low(-1.03), GateResult("brain:LOW_FITNESS", False, -0.2, 1.0, ""),
             GateResult("brain:LOW_SUB_UNIVERSE_SHARPE", False, -0.84, -0.45, "")]
    d = DecisionRecord("p", "repair", "sharpe", gate_results=gates,
                       reasons=[f"{SIGN_FIRST}: Sharpe -1.03 is negative"])
    r = ResultRecord("p", "complete", sharpe=-1.03, fitness=-0.2)
    c = candidate(candidate_id="p", expression="rank(close)")
    agent = RepairAgent(playbook, SPEC, memory)
    stats = LeverStats.from_events([], agent.learning.prior)
    first = agent._wave_options(c, d, r, stats, set(), ())
    assert [lv.name for _, lv in first] == ["flip_sign"]
    assert agent._wave_options(c, d, r, stats, {"flip_sign"}, ()) == []


@pytest.mark.parametrize("two_year, decision", [(-1.0, "repair"), (1.0, "reject")])
def test_sign_first_alpha_is_judged_as_after_the_flip(playbook, two_year, decision):
    # owner 2026-11-27: a negative alpha must not be rejected on a value the flip changes.
    # Fitness -1.2 is 1.2 after the flip (a near miss); a 2-year Sharpe of -1.0 becomes 1.0
    # (near), but +1.0 becomes -1.0, which |value| used to count as near
    from alpha_research.evaluator import decide
    gates = [low(-1.8), GateResult("brain:LOW_FITNESS", False, -1.2, 1.0, ""),
             GateResult("brain:LOW_2Y_SHARPE", False, two_year, 1.58, "")]
    shares = {"sharpe": 0.6, "fitness": 0.5, "recent_sharpe": 0.6}
    d = decide(ResultRecord("c1", "complete", sharpe=-1.8), gates, playbook,
               near_miss_share=shares, risk_neutralized_gap=0.5)
    assert d.decision == decision
    assert any(r.startswith(SIGN_FIRST) for r in d.reasons)
    if decision == "reject":
        assert d.failure_class == "recent_sharpe"
        assert any("after the sign flip" in r for r in d.reasons)
