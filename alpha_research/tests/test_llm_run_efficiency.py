"""Run rs20261125111614 (2026-11-25): 60 model calls for 24 alphas and no repair. Owner:
alphas 0.4-0.6 from accept get mechanical fixes only, farther ones none; an arm given slots
gets a batch (min_batch); no required model-call cap, with a stall guard; requests fit the
model's per-minute token limit (gpt-oss was lost to one 9,195-token request)."""

import dataclasses

import pytest

from alpha_research.lever_stats import RepairLearningConfig, mechanical_only, wave_size
from alpha_research.llm import RequestFitClient, ScriptedLLMClient
from alpha_research.orchestrator import consolidate
from alpha_research.repair import Playbook
from alpha_research.repair_agent import RepairAgent
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord

from conftest import PLAYBOOK

TIERS = {"wave_tiers": [{"max_distance": 0.15, "fixes": 4}, {"max_distance": 0.4, "fixes": 2},
                        {"max_distance": 0.6, "fixes": 2, "mechanical_only": True},
                        {"max_distance": None, "fixes": 0}]}


def decision(sharpe, limit=1.58):
    return DecisionRecord("c1", "repair", "sharpe",
                          gate_results=[GateResult("brain:LOW_SHARPE", False, sharpe, limit, "")],
                          reasons=[])


@pytest.mark.parametrize("sharpe, fixes, mech", [
    (1.50, 4, False),    # 0.05 away
    (1.10, 2, False),    # 0.30 away
    (0.85, 2, True),     # 0.46 away: mechanical only
    (0.50, 0, False),    # 0.68 away: nothing
])
def test_tiers_by_distance(sharpe, fixes, mech):
    cfg = RepairLearningConfig.from_dict(TIERS)
    pb = Playbook.load(PLAYBOOK)
    d, r = decision(sharpe), ResultRecord("c1", "complete", sharpe=sharpe)
    assert wave_size(d, r, cfg, pb)[0] == fixes
    assert mechanical_only(d, r, cfg) is mech


def test_a_mechanical_only_wave_asks_no_model_even_for_production_correlation(memory, candidate):
    from test_rescue_lessons import META, REVERSAL, SPEC
    cfg = dataclasses.replace(RepairLearningConfig.from_dict(TIERS), model_levers_per_wave=1)
    llm = ScriptedLLMClient(['{"expression": "rank(fnd_margin)"}'])
    agent = RepairAgent(Playbook.load(PLAYBOOK), SPEC, memory, llm=llm, field_meta={None: META},
                        learning=cfg)
    gates = [GateResult("brain:LOW_SHARPE", False, 0.85, 1.58, ""),
             GateResult("brain:PROD_CORRELATION", False, 0.74, 0.7, "")]
    d = DecisionRecord("c1", "repair", "prod_correlation", gate_results=gates, reasons=[])
    out = agent.repair_wave(candidate(region="EUR", universe="TOP2500", neutralization="COUNTRY",
                                      expression=REVERSAL, dataset="pv1"),
                            d, ResultRecord("c1", "complete", sharpe=0.85))
    assert llm.calls == [] and out.llm_calls == 0
    assert out.children and all(c.lever not in ("blend_other_dataset", "model_repair",
                                                "differentiate_from_production")
                                for c in out.children)
    assert any("mechanical fixes only" in n for n in out.notes)


def test_consolidate_gives_each_arm_a_batch():
    alloc = {"a": 1, "b": 1, "c": 1, "d": 1, "e": 3, "f": 0}
    out = consolidate(alloc, 3)
    assert sum(out.values()) == 7 and out["f"] == 0
    assert all(n == 0 or n >= 3 for n in out.values())
    assert out["e"] >= 3                                   # the Director's favourite stays
    assert consolidate({"a": 1, "b": 1}, 3) == {"a": 2, "b": 0}   # fewer than a batch: one arm
    assert consolidate(alloc, None) == alloc and consolidate(alloc, 1) == alloc


def test_request_fit_lowers_the_output_cap_to_the_token_limit():
    inner = ScriptedLLMClient(["ok"])
    fit = RequestFitClient(inner, 8000)
    system = "x" * 16500                                   # ~5,000 tokens at 3.3 characters
    assert fit.fit(system, "", 4096) == 8000 - 5001 - 200
    assert fit.fit("short", "", 4096) == 4096              # fits: unchanged
    assert fit.fit("x" * 30000, "", 4096) == 1024          # never below min_output
    fit.complete(system, "", max_output=4096)
    assert inner.calls[0]["max_output"] == 8000 - 5001 - 200


def test_cli_research_needs_no_model_call_cap():
    from alpha_research.cli import build_parser
    args = build_parser().parse_args(["research", "--plan", "p.yaml", "--max-simulations", "5",
                                      "--per-round", "5"])
    assert args.max_llm_calls is None


def test_rounds_with_no_simulation_stop_the_run(memory, config, playbook):
    from test_research import AGENTS, make_run
    agents = dataclasses.replace(AGENTS, stop_after_idle_rounds=2)
    run = make_run(memory, config, playbook, agents=agents)
    run._round = lambda r: None                            # a round that sends nothing
    report = run.run(5)
    assert report["stop_reason"] == "2 round(s) in a row sent no alpha to BRAIN"
