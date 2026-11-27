"""Lessons of the EUR reversal rescues (2026-11-15) built into the agents: size_tilt and
blend_other_dataset levers, the playbook order, the Critic's price/volume-only advisory,
lineages closed by an accept, and the Generator lessons."""

import json

import pytest

from alpha_research.critic import Critic, PRICE_ONLY_ISSUE, ReviewOutcome, price_volume_only
from alpha_research.fastexpr import FastExprSpec, analyze
from alpha_research.generator import lessons_for, load_generator_config
from alpha_research.lever_stats import RepairLearningConfig, family_budget
from alpha_research.llm import ScriptedLLMClient
from alpha_research.repair import Playbook
from alpha_research.repair_agent import (MECHANICAL, LeverNotApplicable, RepairAgent,
                                         build_blend, partner_fields)
from alpha_research.research import load_agents_config
from alpha_research.schemas import DecisionRecord, ResultRecord

from conftest import PLAYBOOK, ROOT

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
REVERSAL = ("intraday = close / open; overnight = open / ts_delay(close, 1); "
            "diff = intraday - overnight; group_rank(-ts_sum(diff, 10), industry) * (0.25 + rank(cap))")
META = {
    "anl_up": {"dataset": "analyst9", "category": "analyst", "type": "MATRIX", "alpha_count": 30,
               "coverage": 0.9, "description": "Estimates raised in 4 weeks"},
    "anl_dn": {"dataset": "analyst9", "category": "analyst", "type": "MATRIX", "alpha_count": 20,
               "coverage": 0.9, "description": "Estimates lowered in 4 weeks"},
    "anl_hot": {"dataset": "analyst9", "category": "analyst", "type": "MATRIX", "alpha_count": 900,
                "coverage": 0.9, "description": "Headline rating"},
    "anl_thin": {"dataset": "analyst9", "category": "analyst", "type": "MATRIX", "alpha_count": 5,
                 "coverage": 0.2, "description": "Rarely filled"},
    "pv_turn": {"dataset": "pv71", "category": "pv", "type": "MATRIX", "alpha_count": 40,
                "coverage": 1.0, "description": "Share turnover"},
    "fnd_margin": {"dataset": "fundamental3", "category": "fundamental", "type": "MATRIX",
                   "alpha_count": 60, "coverage": 0.8, "description": "Operating margin"},
}


# ---------------------------------------------------------------- size_tilt

def test_size_tilt_multiplies_by_offset_plus_cap_rank(candidate):
    c = candidate(expression="group_rank(-ts_sum(returns, 10), industry)")
    out = MECHANICAL["size_tilt"](c, {"field": "cap", "offset": 0.25}, SPEC)
    assert out.expression == "group_rank(-ts_sum(returns, 10), industry) * (0.25 + rank(cap))"
    with pytest.raises(LeverNotApplicable, match="already tilted"):
        MECHANICAL["size_tilt"](out, {"field": "cap", "offset": 0.25}, SPEC)
    with pytest.raises(LeverNotApplicable, match="not a base field"):
        MECHANICAL["size_tilt"](c, {"field": "nosuch", "offset": 0.25}, SPEC)


def test_playbook_orders_the_rescue_levers_first():
    pb = Playbook.load(PLAYBOOK)
    sub = [lv.name for lv in pb.entries["sub_universe"]]
    # drop_weak_legs first (2026-11-27), then the size tilt (feasibility gates it)
    assert sub[:2] == ["drop_weak_legs", "size_tilt"]
    assert sub.index("liquidity_tilt") > sub.index("rank_transform")
    assert sub.index("change_universe") > sub.index("liquidity_tilt")
    assert pb.entries["prod_correlation"][0].name == "blend_other_dataset"


# ---------------------------------------------------------------- blend_other_dataset

def test_build_blend_keeps_the_tilt_outside_and_ranks_the_partner_in_the_alphas_group():
    expr = build_blend(REVERSAL, "(anl_up - anl_dn) / 2", SPEC)
    assert expr.endswith(
        "blend_a = group_rank(-ts_sum(diff, 10), industry); "
        "blend_b = group_rank((anl_up - anl_dn) / 2, industry); "
        "blend_bf = if_else(is_nan(blend_b), 0.5, blend_b); "
        "(blend_a + blend_bf) * (0.25 + rank(cap))")
    assert analyze(expr, SPEC).ok


def test_build_blend_ranks_a_plain_alpha_and_renames_partner_variables():
    expr = build_blend("blend_a = ts_mean(close, 5); close / blend_a",
                       "x = anl_up - anl_dn; ts_mean(x, 3)", SPEC, weight=0.5)
    assert "partner_x = anl_up - anl_dn" in expr and "ts_mean(partner_x, 3)" in expr
    assert "blend_a2 = rank(close / blend_a)" in expr and "blend_b = rank(ts_mean(partner_x, 3))" in expr
    assert expr.endswith("blend_a2 + 0.5 * blend_bf")
    assert analyze(expr, SPEC).ok


def test_partner_fields_leave_out_own_dataset_price_volume_crowded_and_thin_fields():
    got = [f for f, _ in partner_fields(META, {"pv1"}, SPEC, crowded={"anl_hot"},
                                        exclude_categories=["pv"], min_coverage=0.5)]
    assert got == ["anl_dn", "anl_up", "fnd_margin"]     # no topics: least used first
    assert [f for f, _ in partner_fields(META, {"pv1"}, SPEC, crowded={"anl_hot"},
                                         exclude_categories=["pv"], topics="operating margin",
                                         per_dataset=1)] == ["fnd_margin", "anl_dn"]
    assert [f for f, _ in partner_fields(META, {"analyst9"}, SPEC, exclude_categories=["pv"])] \
        == ["fnd_margin"]
    assert [f for f, _ in partner_fields(META, set(), SPEC, role=lambda f: "id")] == []
    dated = {"d": {**META["fnd_margin"], "description": "Current fiscal period end"}}
    assert partner_fields(dated, set(), SPEC) == []


def prod_decision(cid="c1"):
    return DecisionRecord(cid, "repair", "prod_correlation",
                          reasons=["brain:PROD_CORRELATION failed: value 0.7392 vs 0.7"])


BLEND_ONLY = Playbook.from_dict({"prod_correlation": [
    {"lever": "blend_other_dataset", "step": {"weight": 1.0, "exclude_categories": ["pv"]}}]})


def test_blend_lever_asks_for_a_partner_and_builds_the_blend(memory, candidate):
    llm = ScriptedLLMClient([json.dumps({"expression": "(anl_up - anl_dn) / 2",
                                         "explanation": "analyst revision breadth"})])
    agent = RepairAgent(BLEND_ONLY, SPEC, memory, llm=llm, field_meta={None: META})
    out = agent.repair(candidate(expression=REVERSAL, dataset="pv1"), prod_decision())
    assert out.lever == "blend_other_dataset"
    assert "blend_b = group_rank((anl_up - anl_dn) / 2, industry)" in out.child.expression
    assert out.child.expression.endswith("(blend_a + blend_bf) * (0.25 + rank(cap))")
    user = llm.calls[0]["user"]
    assert "<partner_fields>" in user and "anl_up | dataset analyst9 (analyst)" in user
    assert "pv_turn" not in user          # price/volume: the alpha's own kind of information


def test_blend_lever_refuses_a_partner_without_a_listed_field(memory, candidate):
    llm = ScriptedLLMClient([json.dumps({"expression": "rank(volume / adv20)"})])
    agent = RepairAgent(BLEND_ONLY, SPEC, memory, llm=llm, field_meta={None: META})
    out = agent.repair(candidate(expression=REVERSAL), prod_decision())
    assert out.child is None and any("none of the listed fields" in n for n in out.notes)


def test_blend_lever_does_not_apply_without_a_catalog(memory, candidate):
    agent = RepairAgent(BLEND_ONLY, SPEC, memory, llm=ScriptedLLMClient([]))
    out = agent.repair(candidate(expression=REVERSAL), prod_decision())
    assert out.child is None and any("no partner field" in n for n in out.notes)


# ---------------------------------------------------------------- lineage closed by an accept

def row(cid, parent, decision):
    from alpha_research.schemas import CandidateRecord
    c = CandidateRecord.from_dict(dict(
        candidate_id=cid, expression="rank(close)", region="USA", universe="TOP3000", delay=1,
        decay=4, neutralization="SUBINDUSTRY", truncation=0.08, start_date="2019-01-01",
        end_date="2023-12-31", parent_id=parent, generation=0 if parent is None else 1))
    return {"candidate": c, "decision": DecisionRecord(cid, decision, "none" if decision == "accept"
                                                       else "sub_universe"),
            "result": ResultRecord(cid, "complete")}


def test_a_lineage_with_an_accepted_member_gets_no_more_repairs():
    outcomes = [row("root", None, "repair"), row("root_r1", "root", "accept"),
                row("root_r1b", "root", "repair")]
    left, why = family_budget(outcomes[2]["candidate"], outcomes, RepairLearningConfig())
    assert left == 0 and "already has an accepted alpha (root_r1)" in why


# ---------------------------------------------------------------- Critic advisory

def test_price_volume_only():
    assert price_volume_only(REVERSAL, SPEC, {"pv_turn"})
    assert price_volume_only("rank(pv_turn)", SPEC, {"pv_turn"})
    assert not price_volume_only("rank(close) + rank(anl_up)", SPEC, {"pv_turn"})


def test_critic_flags_fresh_price_only_alphas(candidate):
    critic = Critic(SPEC, price_fields={None: {"pv_turn"}})
    reviews = critic.review([candidate(candidate_id="a", expression=REVERSAL),
                             candidate(candidate_id="b", expression="rank(close) + rank(anl_up)"),
                             candidate(candidate_id="c", expression=REVERSAL, lever="size_tilt",
                                       parent_id="a", generation=1)],
                            {}, ReviewOutcome())
    assert PRICE_ONLY_ISSUE in reviews["a"].issues and reviews["a"].verdict == "simulate"
    assert PRICE_ONLY_ISSUE not in reviews["b"].issues + reviews["c"].issues


def test_agents_config_reads_the_price_volume_categories():
    assert load_agents_config(ROOT / "config" / "agents.yaml").price_categories == ("pv",)


# ---------------------------------------------------------------- Generator lessons

def test_generator_lessons_cover_price_volume_and_production_correlation():
    cfg = load_generator_config(ROOT / "config" / "generator.yaml")
    pv = " ".join(lessons_for(cfg, "pv"))
    assert "production" in pv and "another dataset" in pv and "rank(cap)" in pv
    assert "new information" in lessons_for(cfg, None)[0]
