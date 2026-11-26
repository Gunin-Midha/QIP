"""Blend Planner (owner 2026-11-26): model-planned repair waves built on blends, with partners
chosen by fit to the alpha, a stated mechanism, a fixed direction, a reviewer from another
maker, and rule checks before anything reaches BRAIN."""

import dataclasses
import json

import numpy as np
import pandas as pd

from alpha_research import blend_planner as bp
from alpha_research.fastexpr import FastExprSpec
from alpha_research.generator import load_generator_config
from alpha_research.llm import ScriptedLLMClient
from alpha_research.repair_agent import RepairAgent
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord, YearRow

from conftest import ROOT

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
GEN = load_generator_config(ROOT / "config" / "generator.yaml")
CFG = bp.BlendConfig(enabled=True, review=True)
DAYS = pd.bdate_range("2019-01-01", periods=600)


def pnl(seed):
    return pd.Series(np.random.default_rng(seed).normal(0, 1, len(DAYS)), index=DAYS)


def meta():
    m = {}
    for ds, cat, n in (("own", "model", 3), ("anl", "analyst", 4), ("alt", "other", 4),
                       ("prices", "pv", 3)):
        for i in range(n):
            m[f"{ds}_f{i}"] = {"dataset": ds, "category": cat, "coverage": 0.8,
                               "alpha_count": i, "type": "MATRIX",
                               "description": f"{cat} measure {i}",
                               "dataset_alpha_count": 100}
    m["anl_crowded"] = {"dataset": "anl", "category": "analyst", "coverage": 0.9,
                        "alpha_count": 5000, "type": "MATRIX", "description": "crowded"}
    return m


def years(vals):
    return [YearRow(2019 + i, sharpe=v) for i, v in enumerate(vals)]


def outcome(candidate, cid, ds, sharpe, *, lever=None, parent=None, daily=None, yearly=(),
            expr="rank(own_f0)", hyp=None, decision="reject", gates=(), prod=None):
    c = candidate(candidate_id=cid, dataset=ds, lever=lever, parent_id=parent,
                  expression=expr, hypothesis=hyp, region="USA", delay=1)
    cv = {"PROD_CORRELATION": {"value": prod, "limit": 0.7}} if prod is not None else {}
    r = ResultRecord(cid, "complete", sharpe=sharpe, fitness=0.9, turnover=0.2,
                     daily_pnl=daily, yearly=list(yearly), check_values=cv)
    return {"candidate": c, "result": r,
            "decision": DecisionRecord(cid, decision, "prod_correlation", gate_results=list(gates))}


def parent_setup(candidate):
    gates = [GateResult("brain:PROD_CORRELATION", False, 0.74, 0.7, "")]
    par = outcome(candidate, "p1", "own", 2.0, daily=pnl(1), yearly=years([2, 2, 2, 0.5, 0.4]),
                  decision="repair", gates=gates, prod=0.74)
    ev = outcome(candidate, "e1", "anl", -1.1, daily=pnl(2), yearly=years([1, 1, 1, -1.5, -1.6]),
                 expr="rank(anl_f0)")
    return par, [par, ev]


def test_candidates_are_scored_by_fit_and_proven_by_evidence(candidate):
    par, outs = parent_setup(candidate)
    cands = bp.partner_candidates(par["candidate"], par["result"], {"prod_correlation"}, outs,
                                  meta(), SPEC, GEN, {}, CFG, 1.58)
    by = {p.dataset: p for p in cands}
    assert set(by) == {"anl", "alt"}                       # own dataset and pv left out
    assert by["anl"].proven and not by["alt"].proven
    assert "anl_crowded" not in {f for f, _ in by["anl"].fields}
    e = by["anl"].evidence
    assert e["sharpe"] == -1.1 and abs(e["pnl_corr_with_alpha"]) < 0.2
    # weak years of the parent (below its median): the evidence alpha, sign-adjusted
    assert e["sharpe_in_weak_years"] == round(np.mean([1.5, 1.6]), 2)
    assert cands[0].dataset == "anl"                        # proven first


def plan_reply(**over):
    reply = {"variants": [
        {"dataset": "anl", "partner": "-rank(anl_f1)", "weight": 2,
         "mechanism": "analyst downgrades lead the drift this alpha trades; the two disagree rarely"},
        {"dataset": "alt", "partner": "rank(alt_f0)", "weight": 1,
         "mechanism": "alternative demand data confirms the fundamental trend in the alpha"},
        {"dataset": "anl", "partner": "rank(anl_f2)", "weight": 3,
         "mechanism": "analyst revisions confirm the alpha's signal at a different horizon"},
        {"dataset": "anl", "partner": "rank(anl_f3)", "weight": 1, "mechanism": "short"},
        {"dataset": "anl", "partner": "rank(alt_f1)", "weight": 1,
         "mechanism": "reads another dataset's field, which the checks must refuse here"},
    ], "partner_alone": [
        {"dataset": "alt", "partner": "rank(alt_f0)",
         "mechanism": "tests whether demand data predicts returns before any blend uses it"},
    ]}
    reply.update(over)
    return reply


def planner(reviews=None):
    llm = ScriptedLLMClient([json.dumps(plan_reply())])
    rev = ScriptedLLMClient([json.dumps({"reviews": reviews})]) if reviews is not None else None
    return bp.BlendPlanner(SPEC, CFG, llm=llm, reviewer=rev), llm


def run_plan(candidate, reviews):
    par, outs = parent_setup(candidate)
    cands = bp.partner_candidates(par["candidate"], par["result"], {"prod_correlation"}, outs,
                                  meta(), SPEC, GEN, {}, CFG, 1.58)
    pl, _ = planner(reviews)
    return pl.plan(par["candidate"], par["decision"], par["result"], {"prod_correlation"}, cands,
                   catalog=list(meta()) , outcomes=outs, meta=meta(), max_variants=4,
                   metrics={}, tried=[])


def test_rule_checks_before_brain(candidate):
    plan = run_plan(candidate, [{"index": 0, "keep": True, "reason": "ok"},
                                {"index": 1, "keep": True, "reason": "ok"}])
    kinds = [(v.dataset, v.weight) for v in plan.variants]
    assert kinds == [("anl", 2.0), ("alt", None)]
    why = " | ".join(plan.dropped)
    assert "untested partner" in why and "weight 3" in why and "mechanism" in why
    assert "listed fields of anl only" in why
    assert plan.llm_calls == 2                                # plan + review


def test_reviewer_drops_what_it_does_not_keep(candidate):
    plan = run_plan(candidate, [{"index": 0, "keep": False, "reason": "same idea twice"},
                                {"index": 1, "keep": True, "reason": "fine"}])
    assert [(v.dataset, v.weight) for v in plan.variants] == [("alt", None)]
    assert any("same idea twice" in d for d in plan.dropped)


def test_a_partner_that_failed_alone_is_never_reused_or_flipped(candidate):
    par, outs = parent_setup(candidate)
    outs.append(outcome(candidate, "x_alone", "alt", -0.9, lever=bp.PARTNER_ALONE,
                        expr="rank(alt_f0)"))
    cands = bp.partner_candidates(par["candidate"], par["result"], {"prod_correlation"}, outs,
                                  meta(), SPEC, GEN, {}, CFG, 1.58)
    assert not {p.dataset: p for p in cands}["alt"].proven
    pl, _ = planner(None)
    pl.config = dataclasses.replace(CFG, review=False)
    plan = pl.plan(par["candidate"], par["decision"], par["result"], {"prod_correlation"}, cands,
                   catalog=list(meta()), outcomes=outs, meta=meta(), max_variants=4,
                   metrics={}, tried=[])
    assert ("alt", None) not in [(v.dataset, v.weight) for v in plan.variants]
    assert any("failed alone" in d for d in plan.dropped)


def test_eligible_only_close_leads_with_an_information_failure(candidate):
    par, _ = parent_setup(candidate)
    ok, why = bp.eligible(par["decision"], par["result"], {"prod_correlation"}, CFG)
    assert ok, why
    assert not bp.eligible(par["decision"], par["result"], {"turnover"}, CFG)[0]
    far = DecisionRecord("p1", "repair", "sharpe",
                         gate_results=[GateResult("sharpe", False, 0.5, 1.58, "")])
    assert not bp.eligible(far, par["result"], {"sharpe"}, CFG)[0]


def test_stability_flag():
    base = ResultRecord("b", "complete", yearly=years([1, 1, 1, 1, 1]))
    good = ResultRecord("c", "complete", yearly=years([1.2, 1.1, 1.3, 0.9, 1.4]))
    lucky = ResultRecord("c", "complete", yearly=years([3.0, 0.8, 0.9, 0.7, 0.9]))
    assert bp.stability_flag(good, base, CFG) is None
    assert "1 of 5 years" in bp.stability_flag(lucky, base, CFG)


def test_repair_wave_builds_blends_on_the_base_alpha(memory, playbook, candidate):
    par, outs = parent_setup(candidate)
    for o in outs:
        memory.put_candidate(o["candidate"])
        memory.put_result(o["result"])
        memory.put_decision(o["decision"])
    llm = ScriptedLLMClient([json.dumps(plan_reply())])
    agent = RepairAgent(playbook, SPEC, memory, llm=llm, generator_config=GEN,
                        field_meta={("USA", 1): meta()}, markets={("USA", 1): list(meta())})
    agent.blend = bp.BlendPlanner(SPEC, dataclasses.replace(CFG, review=False), llm=llm)
    wave = agent.repair_wave(par["candidate"], par["decision"], par["result"], size=4)
    levers = [c.lever for c in wave.children]
    assert bp.BLEND_PLAN in levers and bp.PARTNER_ALONE in levers
    blend = next(c for c in wave.children if c.lever == bp.BLEND_PLAN)
    assert "rank(own_f0)" in blend.expression and "anl_f1" in blend.expression
    assert blend.hypothesis.startswith("[blend partner=anl weight=2]")
    alone = next(c for c in wave.children if c.lever == bp.PARTNER_ALONE)
    assert alone.expression == "rank(alt_f0)" and alone.dataset == "alt"
    # the planning call is the wave's model call: no other model lever ran
    assert wave.llm_calls == 1


def test_a_blend_is_never_built_on_a_blend(candidate):
    root = candidate(candidate_id="r", lever=None)
    b1 = candidate(candidate_id="r_r1", lever=bp.BLEND_PLAN, parent_id="r")
    fix = candidate(candidate_id="r_r1_r2", lever="increase_decay", parent_id="r_r1")
    by = {c.candidate_id: {"candidate": c} for c in (root, b1, fix)}
    assert bp.blend_base(b1, by).candidate_id == "r"
    assert bp.blend_base(fix, by).candidate_id == "r_r1_r2"   # a mechanical fix is its own base
    assert bp.lineage_root(fix, by) == "r"


def test_research_run_wires_a_reviewer_from_another_maker(memory, config, playbook):
    from test_research import make_run
    run = make_run(memory, config, playbook)
    blend = run.repairer.blend
    assert blend is not None and blend.reviewer is not None   # repair claude-, critic gemini-
    assert run.report()["blend_flags"] == []


def test_brain_correlation_checks_count_as_lower_is_better():
    # until 2026-11-26 a PROD_CORRELATION of 0.93 and one of 0.71 both gave distance 0
    from alpha_research.lever_stats import distance
    def d(v):
        dec = DecisionRecord("x", "repair", "prod_correlation", gate_results=[
            GateResult("brain:PROD_CORRELATION", False, v, 0.7, "")])
        return distance(dec, ResultRecord("x", "complete", sharpe=2.0))
    assert d(0.71) == round(0.01 / 0.7, 6) and d(0.93) == round(0.23 / 0.7, 6)
