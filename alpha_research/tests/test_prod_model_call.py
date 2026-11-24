"""An alpha failing production correlation always gets a model call in its repair wave
(owner 2026-11-25): that class's model levers take the wave's model slot first, even when
production correlation is a secondary failure; a lever that never reached the model does
not spend the slot; blend_other_dataset may repeat along a lineage, and the model sees the
partners already tried with BRAIN's production correlation."""

import json

from alpha_research.llm import ScriptedLLMClient
from alpha_research.repair import Playbook
from alpha_research.repair_agent import RepairAgent, tried_partners
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord

from test_rescue_lessons import META, REVERSAL, SPEC

PARTNER = json.dumps({"expression": "(anl_up - anl_dn) / 2", "explanation": "breadth"})
OTHER = json.dumps({"expression": "rank(-ts_delta(close, 5))", "explanation": "new idea"})


def playbook(**prod_step):
    return Playbook.from_dict({
        "sharpe": [{"lever": "increase_decay", "step": {"add": 2}},
                   {"lever": "wrap_in_smoothing", "step": {"window": 5}},
                   {"lever": "model_repair", "step": {}}],
        "prod_correlation": [
            {"lever": "blend_other_dataset",
             "step": {"weight": 1.0, "exclude_categories": ["pv"], **prod_step}},
            {"lever": "differentiate_from_production", "step": {}}]})


def decision(primary="sharpe"):
    gates = [GateResult("brain:LOW_SHARPE", False, 1.4, 1.58, ""),
             GateResult("brain:PROD_CORRELATION", False, 0.74, 0.7, "")]
    return DecisionRecord("c1", "repair", primary, gate_results=gates,
                          reasons=["brain:PROD_CORRELATION failed: value 0.74 vs 0.7"])


def result(cid="c1", prod=0.74):
    return ResultRecord(cid, "complete", sharpe=1.4, fitness=1.1,
                        check_values={"PROD_CORRELATION": {"value": prod, "limit": 0.7}})


def test_secondary_prod_failure_takes_the_model_slot_before_mechanical_levers(memory, candidate):
    llm = ScriptedLLMClient([PARTNER])
    agent = RepairAgent(playbook(), SPEC, memory, llm=llm, field_meta={None: META})
    out = agent.repair_wave(candidate(expression=REVERSAL, dataset="pv1"), decision(),
                            result(), size=2)
    assert [c.lever for c in out.children][0] == "blend_other_dataset"
    assert len(llm.calls) == 1 and out.llm_calls == 1
    assert not any(c.lever == "model_repair" for c in out.children)   # one model slot


def test_a_lever_that_never_asked_the_model_does_not_spend_the_slot(memory, candidate):
    llm = ScriptedLLMClient([OTHER])
    agent = RepairAgent(playbook(), SPEC, memory, llm=llm)   # no catalog: blend can't apply
    out = agent.repair_wave(candidate(expression=REVERSAL, dataset="pv1"),
                            decision("prod_correlation"), result(), size=2)
    assert any("blend_other_dataset: not applicable" in n for n in out.notes)
    assert "differentiate_from_production" in [c.lever for c in out.children]
    assert len(llm.calls) == 1


def test_no_model_configured_is_noted(memory, candidate):
    agent = RepairAgent(playbook(), SPEC, memory, llm=None, field_meta={None: META})
    out = agent.repair_wave(candidate(expression=REVERSAL, dataset="pv1"),
                            decision("prod_correlation"), result(), size=2)
    assert any("needs a model call; no model is configured" in n for n in out.notes)
    assert any("no model call for production correlation" in n for n in out.notes)


def test_blend_repeats_along_a_lineage_and_sees_the_partners_tried(memory, candidate):
    root = candidate(candidate_id="c1", expression=REVERSAL, dataset="pv1")
    memory.put_candidate(root)
    memory.put_result(result("c1", 0.80))
    memory.put_decision(decision("prod_correlation"))
    llm = ScriptedLLMClient([PARTNER, json.dumps({"expression": "rank(fnd_margin)"})])
    agent = RepairAgent(playbook(repeatable=True), SPEC, memory, llm=llm,
                        field_meta={None: META})
    child = agent.repair_wave(root, decision("prod_correlation"), result("c1", 0.80),
                              size=1).children[0]
    assert child.lever == "blend_other_dataset"
    memory.put_candidate(child)
    memory.put_result(result(child.candidate_id, 0.76))
    d2 = DecisionRecord(child.candidate_id, "repair", "prod_correlation",
                        gate_results=decision().gate_results)
    memory.put_decision(d2)
    lines = tried_partners(child, memory.candidate_outcomes(), SPEC)
    assert lines == ["anl_dn, anl_up -> production correlation 0.76, Sharpe 1.4, fitness 1.1"]
    # still eligible (the learner may rank another production-correlation model lever
    # first after a partial gain; priority makes it the one tried here)
    again = agent.repair_wave(child, d2, result(child.candidate_id, 0.76), size=1,
                              priority=["blend_other_dataset"])
    assert [c.lever for c in again.children] == ["blend_other_dataset"], again.notes
    user = llm.calls[-1]["user"]
    assert "<tried_partners>" in user and "anl_dn, anl_up -> production correlation 0.76" in user


def test_repeatable_blend_stays_in_the_wave_order(memory, candidate):
    root = candidate(candidate_id="c1", expression=REVERSAL, dataset="pv1")
    memory.put_candidate(root)
    llm = ScriptedLLMClient([PARTNER, OTHER])
    agent = RepairAgent(playbook(repeatable=True), SPEC, memory, llm=llm,
                        field_meta={None: META})
    child = agent.repair_wave(root, decision("prod_correlation"), result(), size=1).children[0]
    memory.put_candidate(child)
    again = agent.repair_wave(child, decision("prod_correlation"), result(child.candidate_id),
                              size=1)
    order = next(n for n in again.notes if n.startswith("wave order:"))
    assert "blend_other_dataset" in order and again.llm_calls == 1


def test_without_repeatable_a_lineage_blends_once(memory, candidate):
    root = candidate(candidate_id="c1", expression=REVERSAL, dataset="pv1")
    memory.put_candidate(root)
    llm = ScriptedLLMClient([PARTNER, OTHER])
    agent = RepairAgent(playbook(), SPEC, memory, llm=llm, field_meta={None: META})
    child = agent.repair_wave(root, decision("prod_correlation"), result(), size=1).children[0]
    memory.put_candidate(child)
    again = agent.repair_wave(child, decision("prod_correlation"), result(child.candidate_id),
                              size=1)
    assert [c.lever for c in again.children] == ["differentiate_from_production"]


def test_real_playbook_marks_the_blend_repeatable():
    from conftest import PLAYBOOK
    step = Playbook.load(PLAYBOOK).entries["prod_correlation"][0].step
    assert step["repeatable"] is True
