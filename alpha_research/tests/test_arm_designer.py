"""Arm Designer (owner 2026-11-26): the pipeline writes its own arms, per market, by rules,
and only when the market's previous batch of arms is done."""

import dataclasses
import json
from collections import deque

from alpha_research import arm_designer as ad
from alpha_research.research import ArmPlan, Budget, ResearchRun
from alpha_research.generator import GenerationSettings, Source
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord
from alpha_research.simulator import VirtualClock

from test_research import (AGENTS, GEN_CONFIG, SETTINGS, SPEC, RouterLLM, StubBrain)

CFG = ad.DesignerConfig(enabled=True, arms_per_batch=2, done_after_alphas=2, explore_arms=1)


def cands():
    return {
        "crowded_pv": {"category": "pv", "value_score": 1, "alpha_count": 90000,
                       "signal_fields": 20},
        "rare_model": {"category": "model", "value_score": 8, "alpha_count": 10,
                       "signal_fields": 12},
        "mid_analyst": {"category": "analyst", "value_score": 4, "alpha_count": 500,
                        "signal_fields": 8},
        "tiny": {"category": "other", "value_score": 10, "alpha_count": 1,
                 "signal_fields": 2},       # too few signal fields
    }


def outcome(candidate, cid, dataset, sharpe, decision="reject", prod=None, **kw):
    c = candidate(candidate_id=cid, dataset=dataset, **kw)
    gates = [GateResult("brain:PROD_CORRELATION", False, prod, 0.7, "")] if prod else []
    return {"candidate": c, "result": ResultRecord(cid, "complete", sharpe=sharpe),
            "decision": DecisionRecord(cid, decision, "sharpe", gate_results=gates)}


def test_scores_reward_value_few_alphas_and_our_evidence(candidate):
    outs = [outcome(candidate, "a1", "mid_analyst", 1.5),
            outcome(candidate, "a2", "crowded_pv", 0.3, prod=0.9)]
    rows = {r["dataset"]: r for r in ad.score_datasets(cands(), outs, "USA", 1, 1.58, CFG)}
    assert rows["rare_model"]["score"] > rows["crowded_pv"]["score"]
    assert rows["mid_analyst"]["parts"]["evidence"] == round(1.5 / 1.58, 3)
    assert rows["mid_analyst"]["parts"]["novelty"] == 0.0
    assert rows["crowded_pv"]["parts"]["prod"] == -1.0
    assert rows["rare_model"]["parts"]["value"] == 0.8   # 8 of the market's highest, 10


def test_batch_explores_first_then_one_per_category(candidate):
    outs = [outcome(candidate, "a1", "mid_analyst", 1.5)]
    c = {k: v for k, v in cands().items() if v["signal_fields"] >= CFG.min_signal_fields}
    picked = ad.choose_datasets(ad.score_datasets(c, outs, "USA", 1, 1.58, CFG),
                                dataclasses.replace(CFG, arms_per_batch=3))
    assert picked[0]["why"].startswith("exploration") and picked[0]["our_alphas"] == 0
    assert len({p["category"] for p in picked}) == 3


def test_settings_follow_near_misses_else_category_default(candidate):
    cats = {"m1": "model", "m2": "model"}
    near = []
    for i in range(3):
        o = outcome(candidate, f"n{i}", "m1", 1.4, decision="accept", decay=10,
                    neutralization="SECTOR")
        near.append(o)
    cfg = dataclasses.replace(CFG, decay_by_category={"model": 6, "default": 4})
    assert ad.design_settings("model", near, "USA", 1, "INDUSTRY", cats, cfg)[:2] == \
        (10, "SECTOR")
    assert ad.design_settings("model", near[:2], "USA", 1, "INDUSTRY", cats, cfg)[:2] == \
        (6, "INDUSTRY")
    # a factor neutralization is never a starting setting
    fac = [outcome(candidate, f"f{i}", "m1", 1.4, decision="accept", decay=3,
                   neutralization="STATISTICAL") for i in range(3)]
    assert ad.design_settings("model", fac, "USA", 1, "INDUSTRY", cats, cfg)[1] == "INDUSTRY"


def test_briefs_name_no_catalog_field():
    hyp = ad.briefs("Store reviews", "Reviews of rating and price for items.", "other", "o1",
                    {"rating", "price"}, ["Average rating of the item"])
    assert len(hyp) == 2
    words = {w.strip(".,;:").lower() for h in hyp for w in h.split()}
    assert not words & {"rating", "price"}
    assert "Store reviews" in hyp[0] and "Its fields measure" in hyp[0]


def test_batch_is_done_when_every_arm_has_its_alphas():
    batch = [{"dataset": "a"}, {"dataset": "b"}]
    assert ad.batch_done(batch, {"a": 2, "b": 1}, CFG)[0] is False
    done, lines = ad.batch_done(batch, {"a": 3, "b": 2}, CFG)
    assert done and lines == ["a 2/2", "b 2/2"]


# ------------------------------------------------------------------ in a research run


def _catalog(tmp_path):
    fields = []
    for ds, cat, n, alphas in (("pv1", "pv", 6, 50000), ("rare_model", "model", 6, 10),
                               ("analyst9", "analyst", 6, 300)):
        for i in range(n):
            fields.append({"id": f"{ds}_f{i}", "type": "MATRIX", "coverage": 0.9,
                           "alphaCount": i, "dataset": {"id": ds}, "category": {"id": cat},
                           "description": f"{cat} measure number {i}"})
    path = tmp_path / "datafields_usa_top3000_d1.json"
    path.write_text(json.dumps(fields))
    (tmp_path / "datasets_usa_top3000_d1.json").write_text(json.dumps([
        {"id": "pv1", "name": "Price Volume", "description": "Prices.", "valueScore": 1,
         "alphaCount": 50000, "category": {"id": "pv"}},
        {"id": "rare_model", "name": "Rare Model", "description": "A model.", "valueScore": 8,
         "alphaCount": 10, "category": {"id": "model"}},
        {"id": "analyst9", "name": "Analyst Nine", "description": "Estimates.",
         "valueScore": 4, "alphaCount": 300, "category": {"id": "analyst"}}]))
    return path


def _run(memory, config, playbook, path, run_id):
    from alpha_research.generator import load_datafields, load_datafield_meta
    plan = [ArmPlan("pv1", GenerationSettings.from_mapping({**SETTINGS, "dataset": "pv1"}),
                    deque([Source("hypothesis", "short-term reversal")]), catalog=path)]
    agents = dataclasses.replace(AGENTS, designer=dataclasses.replace(
        CFG, min_signal_fields=3))
    vc = VirtualClock()
    return ResearchRun(plan, agents, Budget(max_simulations=4, per_round=4, max_llm_calls=50),
                       llm_factory=lambda m, r=RouterLLM(): r, client=StubBrain(),
                       config=config, playbook=playbook, spec=SPEC,
                       generator_config=GEN_CONFIG, memory=memory,
                       markets={("USA", 1): (load_datafields(path), load_datafield_meta(path))},
                       seed=0, sleep=vc.sleep, clock=vc.clock, run_id=run_id)


def test_run_designs_arms_and_continues_until_the_batch_is_done(memory, config, playbook,
                                                              tmp_path):
    path = _catalog(tmp_path)
    run = _run(memory, config, playbook, path, "rs1")
    run._design_arms()
    first = memory.arm_designs("USA", 1)
    assert len(first) == 2 and run.designed
    assert first[0]["why"].startswith("exploration")
    assert ("pv1", "USA", GEN_CONFIG.prompt_version, "claude-test", 1) in run.inactive \
        or "pv1" in {d["dataset"] for d in first}
    designed = {d["dataset"] for d in first}
    live = {k[0] for k in run._live_arms()}
    assert live == designed
    # nothing simulated yet: the next run continues the same batch
    again = _run(memory, config, playbook, path, "rs2")
    again._design_arms()
    assert len(memory.arm_designs("USA", 1)) == 2
    # two new alphas on each designed dataset reach BRAIN: the batch is done, a new one
    for ds in designed:
        for i in range(2):
            c = dataclasses.replace(run.arms[next(k for k in run.arms if k[0] == ds)].settings)
            from alpha_research.schemas import CandidateRecord
            cand = CandidateRecord(candidate_id=f"x_{ds}_{i}", expression="rank(close)",
                                   region="USA", universe="TOP3000", delay=1, decay=4,
                                   neutralization="INDUSTRY", truncation=0.08,
                                   start_date=c.start_date, end_date=c.end_date, dataset=ds,
                                   family="f", generation=0)
            memory.put_candidate(cand)
            memory.put_result(ResultRecord(cand.candidate_id, "complete", sharpe=0.5))
    third = _run(memory, config, playbook, path, "rs3")
    third._design_arms()
    batches = {d["batch_id"] for d in memory.arm_designs("USA", 1)}
    assert len(batches) == 2
    newest = ad.latest_batch(memory.arm_designs("USA", 1))
    # datasets designed before are left out while others remain
    assert not {d["dataset"] for d in newest} & designed or len(designed) == 3


def test_a_round_runs_on_the_designed_arms(memory, config, playbook, tmp_path):
    path = _catalog(tmp_path)
    run = _run(memory, config, playbook, path, "rs1")
    report = run.run(1)
    designed = {d["dataset"] for d in report["designed_arms"]}
    sims = [o for r in report["rounds"] for a in r["arms"] for o in a.get("outcomes", [])]
    assert sims and designed
    assert all(a["arm"].split("/")[0] in designed
               for r in report["rounds"] for a in r["arms"] if a.get("outcomes"))


def test_batch_keeps_the_dataset_with_our_best_near_miss(candidate):
    # GBR model38 reached Sharpe 1.52 while the designer would explore little-used data
    outs = [outcome(candidate, "a1", "mid_analyst", 1.52)]
    c = {k: v for k, v in cands().items() if v["signal_fields"] >= CFG.min_signal_fields}
    cfg = dataclasses.replace(CFG, arms_per_batch=3, exploit_arms=1, exploit_min_share=0.6)
    out = ad.design_batch(region="USA", delay=1, candidates=c, dataset_text={},
                          field_ids=set(), outcomes=outs, bar=1.58,
                          base_settings={"neutralization": "INDUSTRY"},
                          earlier={"mid_analyst"}, config=cfg)
    assert out[0]["dataset"] == "mid_analyst" and out[0]["why"].startswith("exploit")
    assert out[1]["why"].startswith("exploration")
    # too weak to exploit: not kept
    weak = [outcome(candidate, "a1", "mid_analyst", 0.4)]
    out = ad.design_batch(region="USA", delay=1, candidates=c, dataset_text={},
                          field_ids=set(), outcomes=weak, bar=1.58,
                          base_settings={"neutralization": "INDUSTRY"},
                          earlier={"mid_analyst"}, config=cfg)
    assert "mid_analyst" not in {d["dataset"] for d in out}
