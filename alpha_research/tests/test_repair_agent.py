import json

import pytest

from alpha_research.fastexpr import FastExprSpec
from alpha_research.llm import ScriptedLLMClient
from alpha_research.repair import Lever, Playbook
from alpha_research.repair_agent import (MECHANICAL, LeverNotApplicable, RepairAgent)
from alpha_research.schemas import DecisionRecord

from conftest import PLAYBOOK, ROOT

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")


def apply(name, cand, **step):
    return MECHANICAL[name](cand, step, SPEC)


def test_increase_decay(candidate):
    assert apply("increase_decay", candidate(decay=4), to=8).decay == 8
    with pytest.raises(LeverNotApplicable, match="already 8"):
        apply("increase_decay", candidate(decay=8), to=8)


def test_tighten_truncation(candidate):
    assert apply("tighten_truncation", candidate(truncation=0.08), to=0.05).truncation == 0.05
    with pytest.raises(LeverNotApplicable):
        apply("tighten_truncation", candidate(truncation=0.05), to=0.05)


def test_lengthen_ts_window_keeps_statements(candidate):
    c = candidate(expression="a = ts_mean(close, 10); rank(-ts_delta(a, 5)) + rank(volume)")
    assert apply("lengthen_ts_window", c, factor=2).expression == \
        "a = ts_mean(close, 20); rank(-ts_delta(a, 10)) + rank(volume)"
    with pytest.raises(LeverNotApplicable, match="no integer"):
        apply("lengthen_ts_window", candidate(expression="rank(close)"), factor=2)


def test_speed_up_shortens_smoothing_and_decay_only(candidate):
    c = candidate(decay=5, expression="u = ts_delta(close, 20); group_rank(ts_decay_linear(rank(u), 5), sector)")
    out = apply("speed_up", c, decay_to=2, window_to=2)
    assert out.decay == 2
    # the smoothing window shortens; ts_delta's window defines the signal and stays
    assert out.expression == "u = ts_delta(close, 20); group_rank(ts_decay_linear(rank(u), 2), sector)"
    assert apply("speed_up", candidate(decay=5, expression="rank(close)"), decay_to=2,
                 window_to=2).decay == 2                      # decay alone is enough
    with pytest.raises(LeverNotApplicable, match="already at most"):
        apply("speed_up", candidate(decay=1, expression="rank(ts_mean(close, 2))"),
              decay_to=2, window_to=2)
    with pytest.raises(LeverNotApplicable, match="needs integer"):
        apply("speed_up", c, decay_to=2)


def test_playbook_tries_speed_up_first_for_recent_sharpe():
    assert Playbook.load(PLAYBOOK).entries["recent_sharpe"][0].name == "speed_up"


def test_wrapping_levers(candidate):
    c = candidate(expression="rank(-ts_delta(close, 5))")
    assert apply("wrap_in_smoothing", c, operator="ts_mean", window=5).expression == \
        "ts_mean(rank(-ts_delta(close, 5)), 5)"
    assert apply("hump", c, threshold=0.01).expression == "hump(rank(-ts_delta(close, 5)), hump=0.01)"
    assert apply("volatility_scale", c, window=20).expression == \
        "rank(-ts_delta(close, 5)) / ts_std_dev(returns, 20)"
    assert apply("rank_transform", candidate(expression="close - open")).expression == "rank(close - open)"
    with pytest.raises(LeverNotApplicable, match="already a rank"):
        apply("rank_transform", c)
    with pytest.raises(LeverNotApplicable, match="already a rank"):  # rank(-rank(x)) = -rank(x)
        apply("rank_transform", candidate(expression="-rank(ts_delta(close, 5))"))
    with pytest.raises(LeverNotApplicable, match="not in the operator list"):
        apply("wrap_in_smoothing", c, operator="no_such_op", window=5)


def decision(cid, failure_class):
    return DecisionRecord(cid, "repair", failure_class, reasons=["turnover failed: 0.85 vs 0.7"])


def test_repair_applies_first_untried_lever_and_sets_lineage(memory, playbook, candidate):
    agent = RepairAgent(playbook, SPEC, memory)
    parent = candidate(candidate_id="p1", decay=4)
    out = agent.repair(parent, decision("p1", "turnover"))
    child = out.child
    assert out.lever == "increase_decay" and child.decay == 8
    assert (child.candidate_id, child.parent_id, child.generation, child.lever) == \
        ("p1_r1", "p1", 1, "increase_decay")
    assert child.expression == parent.expression


def test_factor_first_decision_tries_a_factor_neutralization_not_the_sign(memory, playbook,
                                                                          candidate):
    from alpha_research.schemas import ResultRecord
    agent = RepairAgent(playbook, SPEC, memory)
    parent = candidate(candidate_id="p1")
    res = ResultRecord("p1", "complete", sharpe=-0.3,
                       alt_stats={"risk_neutralized": {"sharpe": 1.2}})
    d = DecisionRecord("p1", "repair", "sharpe", reasons=[
        "factor neutralization first: risk-neutralized Sharpe 1.2 vs raw -0.3"])
    out = agent.repair(parent, d, res)
    assert out.lever.startswith("neutralize_") and out.child.expression == parent.expression
    # without the reason, the negative Sharpe is flipped first as before
    plain = agent.repair(parent, DecisionRecord("p1", "repair", "sharpe"), res)
    assert plain.lever == "flip_sign"


def test_repair_skips_tried_and_inapplicable_levers(memory, playbook, candidate):
    agent = RepairAgent(playbook, SPEC, memory)
    root = candidate(candidate_id="p1", decay=4)
    memory.put_candidate(root)
    first = agent.repair(root, decision("p1", "turnover")).child
    memory.put_candidate(first)
    second = agent.repair(first, decision(first.candidate_id, "turnover"))
    assert second.lever == "lengthen_ts_window"
    assert second.child.expression == "rank(-ts_delta(close, 10))"
    assert second.child.candidate_id == "p1_r1_r2" and second.child.decay == 8


def test_repair_exhausted(memory, candidate):
    pb = Playbook({"turnover": [Lever("increase_decay", {"to": 8})]})
    out = RepairAgent(pb, SPEC, memory).repair(candidate(decay=8), decision("c1", "turnover"))
    assert out.child is None and "repair exhausted" in out.notes[-1]
    assert out.notes[0] == "lever order: increase_decay 0.5"
    assert "increase_decay: not applicable" in out.notes[1]


ORTHO = Playbook({"self_correlation": [Lever("orthogonalize_against_pool")]})


def test_model_lever_without_model_is_skipped(memory, candidate):
    out = RepairAgent(ORTHO, SPEC, memory).repair(candidate(), decision("c1", "self_correlation"))
    assert out.child is None and any("needs the model" in n for n in out.notes)


def test_model_lever_with_valid_reply(memory, playbook, candidate):
    llm = ScriptedLLMClient([json.dumps({"expression": "group_neutralize(rank(-ts_delta(close, 5)), sector)",
                                         "explanation": "removes the sector driver"})])
    agent = RepairAgent(playbook, SPEC, memory, llm=llm)
    out = agent.repair(candidate(), decision("c1", "self_correlation"),
                       pool_expressions=["rank(-ts_delta(close, 3))"])
    assert out.child.expression == "group_neutralize(rank(-ts_delta(close, 5)), sector)"
    assert out.child.lever == "orthogonalize_against_pool" and out.llm_calls == 1
    assert out.child.model == "scripted"
    user = llm.calls[0]["user"]
    assert "<pool>" in user and "rank(-ts_delta(close, 3))" in user and "Failure class: self_correlation" in user


@pytest.mark.parametrize("reply, match", [
    ({"expression": "rank(nosuchop(close))"}, "failed validation"),
    ({"expression": "rank(-ts_delta(close, 5))"}, "unchanged"),
    ({"explanation": "no expression"}, "no expression"),
])
def test_model_lever_bad_replies(memory, candidate, reply, match):
    agent = RepairAgent(ORTHO, SPEC, memory, llm=ScriptedLLMClient([json.dumps(reply)]))
    out = agent.repair(candidate(), decision("c1", "self_correlation"))
    assert out.child is None and any(match in n for n in out.notes)


def test_playbook_without_entry(memory, playbook, candidate):
    out = RepairAgent(playbook, SPEC, memory).repair(candidate(), decision("c1", "weak_year"))
    assert out.child is None and "no playbook entry" in out.notes[0]
