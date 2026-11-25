"""BRAIN's region guides built in (2026-11-25). CHN: short selling is restricted to
eligible blue chips, so the opposite of an alpha does not flip its performance: a negative
Sharpe is no near miss by its size, no "sign first" route, no flip_sign there; and a CHN
lesson for the Generator (region_lessons)."""

import pytest

from alpha_research.evaluator import SIGN_FIRST, decide, near_miss, sharpe_route
from alpha_research.generator import lessons_for, load_generator_config
from alpha_research.repair import Playbook
from alpha_research.repair_agent import MECHANICAL, LeverNotApplicable
from alpha_research.schemas import SHORT_RESTRICTED_REGIONS, GateResult, ResultRecord

from conftest import PLAYBOOK, ROOT
from test_rescue_lessons import SPEC

SHARES = {"sharpe": 0.6, "fitness": 0.5}
LOW = GateResult("brain:LOW_SHARPE", False, -1.68, 2.07, "")


def test_chn_is_short_restricted():
    assert SHORT_RESTRICTED_REGIONS == {"CHN"}


def test_a_negative_sharpe_is_a_near_miss_only_where_a_flip_mirrors():
    r = ResultRecord("c1", "complete", sharpe=-1.68)
    assert near_miss([LOW], "sharpe", SHARES, r)[0]
    ok, why = near_miss([LOW], "sharpe", SHARES, r, sign_mirrors=False)
    assert not ok and "does not mirror" in why
    # a positive risk-neutralized Sharpe still counts with its sign
    r2 = ResultRecord("c1", "complete", sharpe=-0.3, alt_stats={"risk_neutralized": {"sharpe": 1.5}})
    assert near_miss([GateResult("brain:LOW_SHARPE", False, -0.3, 2.07, "")], "sharpe", SHARES,
                     r2, sign_mirrors=False)[0]


def test_no_sign_first_route_and_reject_where_short_selling_is_restricted():
    r = ResultRecord("c1", "complete", sharpe=-1.68)
    turnover = GateResult("brain:HIGH_TURNOVER", False, 0.8, 0.7, "")
    assert sharpe_route([LOW, turnover], r, SHARES, 0.5, "turnover").startswith(SIGN_FIRST)
    assert sharpe_route([LOW, turnover], r, SHARES, 0.5, "turnover", sign_mirrors=False) is None
    pb = Playbook.load(PLAYBOOK)
    d = decide(r, [LOW], pb, near_miss_share=SHARES, risk_neutralized_gap=0.5)
    assert (d.decision, d.failure_class) == ("repair", "sharpe")
    d = decide(r, [LOW], pb, near_miss_share=SHARES, risk_neutralized_gap=0.5,
               sign_mirrors=False)
    assert d.decision == "reject"


def test_flip_sign_does_not_apply_in_chn(candidate):
    r = ResultRecord("c1", "complete", sharpe=-1.68)
    flipped = MECHANICAL["flip_sign"](candidate(region="USA"), {}, SPEC, r)
    assert flipped.expression == "-rank(-ts_delta(close, 5))"
    with pytest.raises(LeverNotApplicable, match="restricted in CHN"):
        MECHANICAL["flip_sign"](candidate(region="CHN", universe="TOP2000U",
                                          neutralization="INDUSTRY"), {}, SPEC, r)


def test_pipeline_decides_chn_with_signed_values(config, playbook, memory, candidate):
    from alpha_research.pipeline import Pipeline
    from alpha_research.brain_client import FixtureBrainClient
    from conftest import FIXTURES
    pipe = Pipeline(config, playbook, FixtureBrainClient(FIXTURES / "brain_results"), memory)
    r = ResultRecord("c1", "complete", sharpe=-1.68, fitness=-1.0, turnover=0.2,
                     checks={"LOW_SHARPE": False},
                     check_values={"LOW_SHARPE": {"value": -1.68, "limit": 2.07}})
    usa = pipe._evaluate(candidate(region="USA"), r).decision_record
    chn = pipe._evaluate(candidate(region="CHN", universe="TOP2000U",
                                   neutralization="INDUSTRY"), r).decision_record
    assert usa.decision == "repair" and chn.decision == "reject"


def test_generator_gets_the_chn_region_lesson():
    cfg = load_generator_config(ROOT / "config" / "generator.yaml")
    chn = lessons_for(cfg, "analyst", "CHN")
    assert len(chn) == 3 and "does not mirror" in chn[1]
    assert len(lessons_for(cfg, "analyst", "USA")) == 2
    assert "2.07" not in " ".join(chn)   # BRAIN's limits come from the market line only
