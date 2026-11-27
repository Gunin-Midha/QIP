"""Repair feasibility, part dropping, partner checks, sub-universe scoring and dead datasets
(owner 2026-11-27, after run rs20261127085437). Numbers are the EUR/JPN cases of that run."""

import dataclasses

import pytest

from alpha_research import arm_designer, blend_planner, breadth, feasibility, legs
from alpha_research.fastexpr import analyze
from alpha_research.feasibility import FeasibilityConfig
from alpha_research.lever_stats import RepairLearningConfig, distance, slack
from alpha_research.repair import Lever, Playbook
from alpha_research.repair_agent import (LeverNotApplicable, RepairAgent, RepairOutcome,
                                         build_blend)
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord

from conftest import PLAYBOOK
from test_rescue_lessons import SPEC

ON = FeasibilityConfig(enabled=True)
META = {
    "news_pos": {"dataset": "news1", "category": "news", "type": "MATRIX", "alpha_count": 20,
                 "coverage": 0.9, "description": "positive words"},
    "val_score": {"dataset": "model38", "category": "model", "type": "MATRIX",
                  "alpha_count": 300, "coverage": 1.0, "description": "value"},
}
for item in ("eps", "sales"):
    for part in ("4wk_up", "4wk_dn", "numest"):
        META[f"anl7_best_{item}_{part}"] = {
            "dataset": "analyst7", "category": "analyst", "type": "MATRIX", "alpha_count": 10,
            "coverage": 0.9, "description": f"{item} estimate revisions {part}"}
BASE = "rank(group_rank(ts_delta(news_pos, 20), subindustry) - group_rank(val_score, subindustry))"
PARTNER = "(anl7_best_eps_4wk_up - anl7_best_eps_4wk_dn) / max(anl7_best_eps_numest, 1)"


def result(cid, sharpe, sub=None, cutoff=None, rn=None, **kw):
    alt = {"risk_neutralized": {"sharpe": rn}} if rn is not None else {}
    return ResultRecord(cid, "complete", sharpe=sharpe, fitness=kw.pop("fitness", 1.0),
                        turnover=0.2, sub_universe_sharpe=sub, sub_universe_cutoff=cutoff,
                        alt_stats=alt, check_values={"LOW_SHARPE": {"value": sharpe,
                                                                     "limit": 1.58}}, **kw)


def sub_decision(cid, sub, cutoff, cls="sub_universe"):
    return DecisionRecord(cid, "repair", cls, gate_results=[
        GateResult("sharpe", True, 2.6, 1.58, ""),
        GateResult("sub_universe", False, sub, cutoff, "")])


def sharpe_decision(cid, s):
    return DecisionRecord(cid, "repair", "sharpe", gate_results=[
        GateResult("sharpe", False, s, 1.58, "")])


def store(memory, c, r=None, d=None):
    memory.put_candidate(c)
    if r is not None:
        memory.put_result(dataclasses.replace(r, candidate_id=c.candidate_id))
    if d is not None:
        memory.put_decision(dataclasses.replace(d, candidate_id=c.candidate_id))


def agent(memory, feas=ON, **kw):
    learning = dataclasses.replace(RepairLearningConfig(), leg_tests=3, max_depth=4,
                                   max_trials_per_root=20, patience=10, feasibility=feas)
    return RepairAgent(Playbook.load(PLAYBOOK), SPEC, memory, field_meta={None: META},
                       learning=learning, **kw)


# ------------------------------------------------------------------ validator (earlier fix)

def test_a_variable_named_after_a_datafield_is_an_error():
    a = analyze("sales = ts_mean(val_score, 5); rank(sales)", SPEC,
                datafields=["val_score", "sales"])
    assert any("datafield name" in e for e in a.errors)
    b = analyze("cap = ts_mean(val_score, 5); rank(cap)", SPEC)   # a base field
    assert any("datafield name" in e for e in b.errors)
    ok = analyze("s = ts_mean(val_score, 5); rank(s)", SPEC, datafields=["val_score"])
    assert ok.ok


# ------------------------------------------------------------------ item 3: feasibility

def blocked(name, mechanical, d, r, failing):
    return feasibility.blocked(name, mechanical=mechanical, decision=d, result=r,
                               failing=failing, sign_first=False, config=ON)


def test_a_far_sharpe_gap_leaves_out_mechanical_levers_but_not_blends():
    d, r = sharpe_decision("p", 1.10), result("p", 1.10, rn=0.5)     # needs x1.44
    assert "never raised Sharpe" in blocked("increase_decay", True, d, r, {"sharpe"})
    assert "risk-neutralized" in blocked("neutralize_statistical", True, d, r, {"sharpe"})
    assert blocked("recipe_blend", True, d, r, {"sharpe"}) is None
    assert blocked("model_repair", False, d, r, {"sharpe"}) is None
    # BRAIN's risk-neutralized Sharpe supports the neutralization
    assert blocked("neutralize_statistical", True, d, result("p", 1.10, rn=1.6),
                   {"sharpe"}) is None
    # a small gap (x1.05) keeps every lever
    assert blocked("increase_decay", True, sharpe_decision("p", 1.5), result("p", 1.5),
                   {"sharpe"}) is None
    # off by default
    assert feasibility.blocked("increase_decay", mechanical=True, decision=d, result=r,
                               failing={"sharpe"}, sign_first=False,
                               config=FeasibilityConfig()) is None


def test_tilts_need_a_sub_universe_sharpe_that_can_support_a_passing_alpha():
    # EUR blend: Sharpe 2.73, sub-universe 0.65, limit 1.42 (share 0.52, floor 0.82)
    d, r = sub_decision("p", 0.65, 1.42), result("p", 2.73, sub=0.65, cutoff=1.42)
    for name in ("size_tilt", "liquidity_tilt", "change_universe",
                 "combo:relative_price_change+size_tilt"):
        assert "never adds large-cap Sharpe" in blocked(name, True, d, r, {"sub_universe"})
    # reshapes too (gap 0.17 > 0.10); neutralizations and blends stay
    assert "reshaping levers" in blocked("rank_transform", True, d, r, {"sub_universe"})
    assert blocked("neutralize_statistical", True, d, r, {"sub_universe"}) is None
    assert blocked("rank_transform", True, sub_decision("p", 0.77, 1.48),
                   result("p", 2.85, sub=0.77, cutoff=1.48), {"sub_universe"}) is None
    # EUR news blend: sub-universe 1.11 >= floor 0.82: a tilt can trade Sharpe for ratio
    d2, r2 = sub_decision("q", 1.11, 1.37), result("q", 2.64, sub=1.11, cutoff=1.37)
    assert blocked("size_tilt", True, d2, r2, {"sub_universe"}) is None


def test_no_blend_onto_an_alpha_without_a_large_cap_part():
    # EUR word-count x price-to-cash-flow: ratio 0.11 against limit share 0.52
    d, r = sub_decision("p", 0.17, 0.80), result("p", 1.54, sub=0.17, cutoff=0.80)
    assert "no large-cap part" in blocked("recipe_blend", True, d, r, {"sub_universe"})
    # ratio 0.30 is enough to try
    d2, r2 = sub_decision("q", 0.54, 0.92), result("q", 1.78, sub=0.54, cutoff=0.92)
    assert blocked("recipe_blend", True, d2, r2, {"sub_universe"}) is None


def test_wave_leaves_out_infeasible_levers_with_a_note(memory, candidate):
    p = candidate(candidate_id="p", expression="rank(ts_delta(val_score, 20))", region="EUR",
                  universe="TOP2500", dataset="model38")
    d, r = sharpe_decision("p", 1.10), result("p", 1.10, rn=0.4)
    store(memory, p, r, d)
    out = agent(memory).repair_wave(p, d, r, size=2)
    assert not out.children                     # every mechanical lever is out, no model
    assert any("not feasible" in n for n in out.notes)
    assert not [c for c in out.children if c.lever == legs.LEG_LEVER]


# ------------------------------------------------------------------ item 4: scoring

def test_sub_universe_is_scored_on_the_sub_universe_sharpe_itself():
    before = (sub_decision("p", 0.65, 1.42), result("p", 2.73, sub=0.65, cutoff=1.42))
    # a size tilt: BRAIN's gap shrinks (cutoff follows the lower Sharpe), sub-universe falls
    tilt = (DecisionRecord("c", "repair", "sharpe", gate_results=[
        GateResult("sharpe", False, 1.08, 1.58, ""),
        GateResult("sub_universe", False, 0.48, 0.56, "")]),
        result("c", 1.08, sub=0.48, cutoff=0.56))
    s0, s1 = slack("sub_universe", *before), slack("sub_universe", *tilt)
    assert s0 == pytest.approx(0.65 - 0.52 * 1.58, abs=0.01)
    assert s1 < s0                               # worse, not "improved"
    # distance counts the floor too
    assert distance(*tilt) > 0.3
    # at or above the floor only the ratio is left: slack 0
    ok = (sub_decision("q", 1.11, 1.37), result("q", 2.64, sub=1.11, cutoff=1.37))
    assert slack("sub_universe", *ok) == 0


# ------------------------------------------------------------------ item 1: parts

def blend_of(base, partner=PARTNER, weight=1.0):
    return build_blend(base, partner, SPEC, weight=weight)


def test_blend_partner_reads_back_partner_and_weight():
    expr = blend_of(BASE, weight=2.0)
    got = legs.blend_partner(expr, SPEC)
    assert got is not None and got[1] == 2.0
    assert feasibility.signal_fields(got[0], SPEC) == feasibility.signal_fields(PARTNER, SPEC)
    assert legs.blend_partner(BASE, SPEC) is None
    tilted = blend_of("group_rank(val_score, industry) * (0.25 + rank(cap))")
    assert legs.blend_partner(tilted, SPEC) is not None


def lineage(memory, candidate, news_ratio=0.51, value_ratio=0.19):
    """EUR news - value base (Sharpe 1.78), its parts measured alone, its recipe blend
    (Sharpe 2.64, sub-universe 1.11 vs 1.37)."""
    base = candidate(candidate_id="b", expression=BASE, region="EUR", universe="TOP2500",
                     dataset="news1", neutralization="STATISTICAL")
    store(memory, base, result("b", 1.78, sub=0.54, cutoff=0.92),
          sub_decision("b", 0.54, 0.92, "fitness"))
    parts = legs.split_legs(BASE, SPEC)
    assert len(parts) == 2
    for i, (leg, s, rat) in enumerate(zip(parts, (0.86, 1.82), (news_ratio, value_ratio))):
        c = dataclasses.replace(base, candidate_id=f"b_r1{'cd'[i]}", expression=leg,
                                parent_id="b", generation=1, lever=legs.LEG_LEVER)
        store(memory, c, result(c.candidate_id, s, sub=s * rat, cutoff=0.52 * s))
    blend = dataclasses.replace(base, candidate_id="b_r1a", expression=blend_of(BASE),
                                parent_id="b", generation=1, lever=legs.RECIPE_LEVER)
    d, r = sub_decision("b_r1a", 1.11, 1.37), result("b_r1a", 2.64, sub=1.11, cutoff=1.37)
    store(memory, blend, r, d)
    return blend, d, r, parts


def test_drop_weak_legs_rebuilds_the_blend_without_the_small_cap_part(memory, candidate):
    blend, d, r, parts = lineage(memory, candidate)
    a = agent(memory)
    child = a._apply(blend, Lever(legs.DROP_LEVER), d, r, None, [], RepairOutcome("x"))
    assert "val_score" not in child.expression          # value part: ratio 0.19 < 0.6 x 0.52
    assert "news_pos" in child.expression and "anl7_best_eps_4wk_up" in child.expression
    assert child.hypothesis.startswith("[drop parts 2")
    assert analyze(child.expression, SPEC).ok


def test_drop_weak_legs_needs_a_strong_part(memory, candidate):
    blend, d, r, _ = lineage(memory, candidate, news_ratio=0.2, value_ratio=0.1)
    with pytest.raises(LeverNotApplicable, match="every measured part is weak"):
        agent(memory)._apply(blend, Lever(legs.DROP_LEVER), d, r, None, [],
                             RepairOutcome("x"))


def test_drop_weak_legs_for_sharpe_keeps_the_part_that_carries_the_alpha():
    rows = [{"part": "a", "result": result("a", 0.37)}, {"part": "b", "result": result("b", 1.8)}]
    weak, _ = legs.part_weakness(rows, sub_mode=False, limit_share=None, whole_sharpe=1.54,
                                 config=ON)
    assert weak == [0]
    # the best part alone is far below the whole: nothing to drop
    assert legs.part_weakness(rows, sub_mode=False, limit_share=None, whole_sharpe=2.5,
                              config=ON)[0] == []


def test_measured_parts_go_first_in_the_wave(memory, candidate):
    blend, d, r, _ = lineage(memory, candidate)
    out = agent(memory).repair_wave(blend, d, r, size=2)
    assert out.children and out.children[0].lever == legs.DROP_LEVER
    # the EUR recipe partner never measured alone: one partner-alone test
    tests = [c for c in out.children if c.lever == blend_planner.PARTNER_ALONE]
    assert len(tests) == 1 and "val_score" not in tests[0].expression
    assert tests[0].expression.startswith("rank(") or "rank(" in tests[0].expression


# ------------------------------------------------------------------ item 2: partners

def test_a_partner_measured_with_little_large_cap_sharpe_is_refused(memory, candidate):
    alone = candidate(candidate_id="z", expression=f"rank({PARTNER})", region="EUR",
                      universe="TOP2500", dataset="analyst7")
    store(memory, alone, result("z", 1.2, sub=0.2, cutoff=0.62))
    r = result("p", 2.64, sub=1.11, cutoff=1.37)
    why = feasibility.partner_check(PARTNER, "EUR", 1, memory.candidate_outcomes(), SPEC, r)
    assert "ratio 0.17" in why
    good = dataclasses.replace(alone, candidate_id="z2")
    store(memory, good, result("z2", 1.17, sub=0.68, cutoff=0.61))
    assert feasibility.partner_check(PARTNER, "EUR", 1, memory.candidate_outcomes(), SPEC,
                                     r) is None


def test_planner_scores_partners_by_their_sub_universe_ratio():
    p = blend_planner.PartnerCandidate("analyst7", "analyst", [])
    low, high = dataclasses.replace(p), dataclasses.replace(p)
    low.evidence, high.evidence = {"sub_universe_ratio": 0.2}, {"sub_universe_ratio": 0.6}
    s_low = blend_planner._score(low, {"sub_universe"}, 1.58, limit_share=0.52)
    s_high = blend_planner._score(high, {"sub_universe"}, 1.58, limit_share=0.52)
    assert s_high - s_low == pytest.approx(2.0)


def breadth_agent(memory):
    a = agent(memory)
    a.breadth = breadth.BreadthConfig(enabled=True, min_item_alphas=0, items_per_variant=2)
    return a


def test_breadth_never_measured_in_the_market_is_tested_alone_first(memory, candidate):
    p = candidate(candidate_id="j", expression="rank(ts_delta(val_score, 20))", region="JPN",
                  universe="TOP1600", dataset="model38")
    d = DecisionRecord("j", "repair", "recent_sharpe", gate_results=[
        GateResult("brain:LOW_2Y_SHARPE", False, 1.4, 1.58, "")])
    r = result("j", 2.0)
    store(memory, p, r, d)
    out = breadth_agent(memory).repair_wave(p, d, r, size=2)
    kinds = [c.lever for c in out.children]
    assert blend_planner.PARTNER_ALONE in kinds and breadth.BREADTH_LEVER not in kinds
    # measured weak in the market: no blend, no test
    t = next(c for c in out.children if c.lever == blend_planner.PARTNER_ALONE)
    store(memory, t, result(t.candidate_id, 0.09))
    again = breadth_agent(memory).repair_wave(p, d, r, size=2)
    assert not [c for c in again.children
                if c.lever in (breadth.BREADTH_LEVER, blend_planner.PARTNER_ALONE)]
    assert any("not blended" in n for n in again.notes)


# ------------------------------------------------------------------ dead datasets

def test_dead_datasets_and_their_designs(candidate):
    cfg = arm_designer.DesignerConfig(dead_after_alphas=5, dead_sharpe_share=0.3)
    rows = []
    for i, s in enumerate((0.09, 0.36, -0.04, -0.21, 0.28)):
        c = candidate(candidate_id=f"m{i}", region="EUR", dataset="model307")
        rows.append({"candidate": c, "result": result(c.candidate_id, s), "decision": None})
    dead = arm_designer.dead_datasets(rows, "EUR", 1, 1.58, cfg)
    assert dead == {"model307": (5, 0.36)}
    assert arm_designer.dead_datasets(rows[:4], "EUR", 1, 1.58, cfg) == {}
    big = dict(rows[0], result=result("m0", -2.15))       # a strong negative is not dead
    assert arm_designer.dead_datasets([big] + rows[1:], "EUR", 1, 1.58, cfg) == {}
    done, lines = arm_designer.batch_done([{"dataset": "model307"}], {}, cfg, dead)
    assert done and lines == ["model307 dead"]
    assert arm_designer.dead_datasets(rows, "EUR", 1, 1.58,
                                      arm_designer.DesignerConfig()) == {}   # off by default
