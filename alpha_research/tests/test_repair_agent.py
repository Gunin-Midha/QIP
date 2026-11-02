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
    assert "increase_decay: not applicable" in out.notes[0]


def test_model_lever_without_model_is_skipped(memory, playbook, candidate):
    out = RepairAgent(playbook, SPEC, memory).repair(candidate(), decision("c1", "self_correlation"))
    assert out.child is None and "needs the model" in out.notes[0]


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
def test_model_lever_bad_replies(memory, playbook, candidate, reply, match):
    agent = RepairAgent(playbook, SPEC, memory, llm=ScriptedLLMClient([json.dumps(reply)]))
    out = agent.repair(candidate(), decision("c1", "self_correlation"))
    assert out.child is None and any(match in n for n in out.notes)


def test_playbook_without_entry(memory, playbook, candidate):
    out = RepairAgent(playbook, SPEC, memory).repair(candidate(), decision("c1", "weak_year"))
    assert out.child is None and "no playbook entry" in out.notes[0]
