"""Part tests and recipes (legs.py, owner 2026-11-26): a combined alpha's parts are simulated
alone in a repair wave, the model sees them (reweight_legs), and accepted blends are reused
as recipes."""

import dataclasses
import json

import pytest

from alpha_research.lever_stats import RepairLearningConfig
from alpha_research.legs import (LEG_LEVER, RECIPE_LEVER, REWEIGHT_LEVER, part_results,
                                 recipes, split_legs)
from alpha_research.llm import ScriptedLLMClient
from alpha_research.repair import Lever, Playbook
from alpha_research.repair_agent import LeverNotApplicable, RepairAgent, RepairOutcome
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord

from conftest import PLAYBOOK
from test_rescue_lessons import SPEC

IND = ("disp = (est_high - est_low) / (abs(ts_mean(est_high, 20)) + 1.0); "
       "main = rank(zscore(ts_decay_linear(disp, 21))); "
       "other = ts_backfill(group_zscore(returns, sector), 21); "
       "rank(-0.7 * ts_backfill(main, 1) - 0.3 * other)")
META = {f: {"dataset": "analyst4", "category": "analyst", "type": "MATRIX", "alpha_count": 50,
            "coverage": 0.9, "description": f} for f in ("est_high", "est_low", "rev_up", "rev_dn")}
META.update({"val_score": {"dataset": "model38", "category": "model", "type": "MATRIX",
                           "alpha_count": 300, "coverage": 1.0, "description": "value"}})
for f in ("rev_up", "rev_dn"):
    META[f]["dataset"] = "analyst7"


def test_split_legs_sum_product_and_single():
    legs = split_legs(IND, SPEC)
    assert len(legs) == 2
    assert legs[0].endswith("-rank(ts_backfill(main, 1))") and "other" not in legs[0]
    assert legs[1] == "-rank(ts_backfill(group_zscore(returns, sector), 21))"
    prod = "rank(ts_delta(est_high, 15)) * rank(ts_delta(val_score, 15))"
    assert split_legs(prod, SPEC) == ["rank(ts_delta(est_high, 15))", "rank(ts_delta(val_score, 15))"]
    # a size tilt is not a part; a single signal has no parts
    assert split_legs("group_rank(-ts_sum(returns, 10), industry) * (0.25 + rank(cap))", SPEC) == []
    assert split_legs("rank(est_high) + 0.5 * rank(est_low)", SPEC) == ["rank(est_high)", "rank(est_low)"]


def ind_alpha(candidate, **kw):
    return candidate(candidate_id="p", expression=IND, region="IND", universe="TOP500",
                     neutralization="STATISTICAL", dataset="analyst4", **kw)


def failing(cid="p", value=0.74):
    d = DecisionRecord(cid, "repair", "sub_universe", gate_results=[
        GateResult("brain:LOW_ROBUST_UNIVERSE_SHARPE", False, value, 1.0, "")])
    return d, ResultRecord(cid, "complete", sharpe=2.02, fitness=1.51, turnover=0.13)


def agent(memory, llm=None, **repair):
    learning = dataclasses.replace(RepairLearningConfig(), leg_tests=3, max_depth=3,
                                   max_trials_per_root=12, patience=8, **repair)
    return RepairAgent(Playbook.load(PLAYBOOK), SPEC, memory, llm=llm,
                       field_meta={None: META}, learning=learning)


def store(memory, c, result=None, decision=None):
    memory.put_candidate(c)
    if result is not None:
        memory.put_result(dataclasses.replace(result, candidate_id=c.candidate_id))
    if decision is not None:
        memory.put_decision(dataclasses.replace(decision, candidate_id=c.candidate_id))


def test_wave_adds_one_test_per_part_once_per_lineage(memory, candidate):
    p = ind_alpha(candidate)
    d, r = failing()
    store(memory, p, r, d)
    out = agent(memory).repair_wave(p, d, r, size=1)
    tests = [c for c in out.children if c.lever == LEG_LEVER]
    assert len(tests) == 2 and {c.expression for c in tests} == set(split_legs(IND, SPEC))
    assert len(out.children) == 3          # one fix + two part tests
    assert all(c.parent_id == "p" for c in out.children)
    for c in out.children:
        store(memory, c, ResultRecord(c.candidate_id, "complete", sharpe=1.0, fitness=0.5))
    again = agent(memory).repair_wave(p, d, r, size=1)
    assert not [c for c in again.children if c.lever == LEG_LEVER]   # once per lineage


def test_sign_first_alphas_get_no_part_tests(memory, candidate):
    p = ind_alpha(candidate)
    d, r = failing()
    d = dataclasses.replace(d, reasons=["sign first: Sharpe -1.5 is negative"])
    store(memory, p, r, d)
    out = agent(memory).repair_wave(p, d, r, size=2)
    assert not [c for c in out.children if c.lever == LEG_LEVER]


def test_reweight_legs_sees_the_parts_and_goes_first(memory, candidate):
    p = ind_alpha(candidate)
    d, r = failing()
    store(memory, p, r, d)
    with pytest.raises(LeverNotApplicable, match="fewer than two parts"):
        agent(memory, llm=ScriptedLLMClient([]))._apply(
            p, Lever(REWEIGHT_LEVER), d, r, None, [], RepairOutcome("p"))
    for i, (leg, robust) in enumerate(zip(split_legs(IND, SPEC), (1.2, 0.3))):
        c = dataclasses.replace(p, candidate_id=f"p_r1{'ab'[i]}", expression=leg,
                                parent_id="p", generation=1, lever=LEG_LEVER)
        res = ResultRecord(c.candidate_id, "complete", sharpe=1.9 - i, fitness=1.2,
                           turnover=0.1, checks={"LOW_ROBUST_UNIVERSE_SHARPE": robust >= 1},
                           check_values={"LOW_ROBUST_UNIVERSE_SHARPE": {"value": robust,
                                                                        "limit": 1.0}})
        store(memory, c, res)
    rows = part_results(p, memory.candidate_outcomes(), SPEC)
    assert [x["result"].sharpe for x in rows] == pytest.approx([1.9, 0.9])
    reply = json.dumps({"expression": "disp = (est_high - est_low) / (abs(ts_mean(est_high, 20)) + 1.0); "
                                      "main = rank(zscore(ts_decay_linear(disp, 21))); -main"})
    llm = ScriptedLLMClient([reply])
    out = agent(memory, llm=llm).repair_wave(p, d, r, size=2)
    model_kids = [c for c in out.children if c.lever == REWEIGHT_LEVER]
    assert len(model_kids) == 1
    prompt = llm.calls[0]["user"]
    assert "<parts>" in prompt and "LOW_ROBUST_UNIVERSE_SHARPE 0.3 (limit 1.0)" in prompt
    assert out.children[0].lever == REWEIGHT_LEVER   # the model slot went to reweight_legs


def test_recipes_and_recipe_blend(memory, candidate):
    # an accepted IND blend whose partner is analyst7 revision breadth
    parent = candidate(candidate_id="acc", expression="rank(val_score)", region="IND",
                       universe="TOP500", dataset="model38")
    store(memory, parent, ResultRecord("acc", "complete", sharpe=1.4),
          DecisionRecord("acc", "repair", "prod_correlation"))
    acc = candidate(candidate_id="acc_r1", region="IND", universe="TOP500", dataset="model38",
                    parent_id="acc", generation=1, lever="blend_other_dataset",
                    expression="blend_a = rank(val_score); blend_b = rank(rev_up - rev_dn); "
                               "blend_bf = if_else(is_nan(blend_b), 0.5, blend_b); blend_a + blend_bf")
    store(memory, acc, ResultRecord("acc_r1", "complete", sharpe=2.1, fitness=1.3),
          DecisionRecord("acc_r1", "accept", "none"))
    rows = recipes(memory.candidate_outcomes(), lambda r, d: META, SPEC)
    assert len(rows) == 1 and rows[0]["partner"] == "rev_up - rev_dn"
    assert rows[0]["partner_dataset"] == "analyst7" and rows[0]["fixed"] == "prod_correlation"
    # a failing analyst4 alpha in IND gets the partner blended in, no model call
    p = ind_alpha(candidate)
    d, r = failing()
    store(memory, p, r, d)
    a = agent(memory)
    child = a._apply(p, Lever(RECIPE_LEVER, {"weight": 1.0}), d, r, None, [], RepairOutcome("p"))
    assert "rev_up - rev_dn" in child.expression and "[recipe " in child.hypothesis
    # the same partner is not blended twice along a lineage
    kid = a._child(p, child, RECIPE_LEVER)
    store(memory, kid)
    with pytest.raises(LeverNotApplicable, match="no untried partner"):
        a._apply(kid, Lever(RECIPE_LEVER, {"weight": 1.0}), d, r, None, [], RepairOutcome("p"))
    # another market has no recipe
    other = candidate(candidate_id="q", region="USA")
    with pytest.raises(LeverNotApplicable):
        a._apply(other, Lever(RECIPE_LEVER, {"weight": 1.0}), d, r, None, [], RepairOutcome("q"))


def test_blend_lever_is_offered_for_more_classes():
    pb = Playbook.load(PLAYBOOK)
    for cls in ("sharpe", "fitness", "sub_universe", "recent_sharpe"):
        names = [lv.name for lv in pb.entries[cls]]
        assert {REWEIGHT_LEVER, RECIPE_LEVER, "blend_other_dataset"} <= set(names)
        assert names[-1] == "model_repair"
