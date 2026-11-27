"""Breadth blends (owner 2026-11-27): five of our seven accepts were the near miss blended with
analyst net revision breadth, all by hand. The pipeline now finds revision families in the
catalog, builds the partner by rule and puts those blends first in a close lead's repair
wave; the Blend Planner lists families first and scores datasets by their fields' usage;
repairs found by a recheck of BRAIN's late checks are repaired first."""

import dataclasses

import pytest

from alpha_research import blend_planner as bp
from alpha_research import breadth as b
from alpha_research.config import ConfigError
from alpha_research.fastexpr import FastExprSpec, analyze
from alpha_research.generator import load_generator_config
from alpha_research.pipeline import Outcome
from alpha_research.repair_agent import RepairAgent, RepairOutcome
from alpha_research.research import load_agents_config
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord

from conftest import ROOT

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
GEN = load_generator_config(ROOT / "config" / "generator.yaml")
CFG = b.BreadthConfig(enabled=True)


def f(ds, cat="analyst", n=10, cov=1.0, typ="MATRIX"):
    return {"dataset": ds, "category": cat, "coverage": cov, "alpha_count": n, "type": typ,
            "description": "x"}


def meta():
    m = {"own_f0": f("own", "model", 50), "own_f1": f("own", "model", 60)}
    # analyst69 style: <item>_4wk_up / _4wk_dn, count = <item>_numest (one part dropped)
    for item, n in (("eps", 40), ("ptp", 6), ("sales", 8), ("bps", 0)):
        m[f"anl_best_{item}_4wk_up"] = f("anl", n=n)
        m[f"anl_best_{item}_4wk_dn"] = f("anl", n=n)
        m[f"anl_best_{item}_numest"] = f("anl", n=n)
    m["anl_market_status"] = f("anl", n=0)
    # analyst7 style: raisednum / lowerednum, count = <item>_num
    for item, n in (("pre", 500), ("opr", 30)):
        m[f"est_12m_{item}_raisednum_4wks"] = f("est", n=n)
        m[f"est_12m_{item}_lowerednum_4wks"] = f("est", n=n)
        m[f"est_12m_{item}_num"] = f("est", n=n)
    # not revisions: an upside/downside beta in a fundamental dataset
    m["fnd_beta_up"] = f("fnd", "fundamental", 50)
    m["fnd_beta_down"] = f("fnd", "fundamental", 50)
    return m


def test_families_by_field_id():
    fams = b.families(meta(), CFG, skip=SPEC.base_fields)
    assert set(fams) == {"anl", "est"}                     # the beta pair is not analyst
    anl = fams["anl"]
    assert [x.item for x in anl] == ["anl_best_ptp_4wk", "anl_best_sales_4wk",
                                     "anl_best_eps_4wk", "anl_best_bps_4wk"]  # 0-alpha last
    assert anl[0].count == "anl_best_ptp_numest"
    est = {x.item: x for x in fams["est"]}
    assert est["est_12m_opr_4wks"].count == "est_12m_opr_num"
    assert est["est_12m_opr_4wks"].down == "est_12m_opr_lowerednum_4wks"


def test_partner_expression_backfills_sparse_and_averages_vectors():
    fam = b.Family("anl", "anl_x_4wk", "anl_x_4wk_up", "anl_x_4wk_dn", "anl_x_numest", 3.0,
                   0.4, True, "analyst")
    assert b.item_expression(fam, CFG) == (
        "(ts_backfill(vec_avg(anl_x_4wk_up), 20) - ts_backfill(vec_avg(anl_x_4wk_dn), 20)) "
        "/ max(ts_backfill(vec_avg(anl_x_numest), 20), 1)")
    nocount = dataclasses.replace(fam, count=None, coverage=1.0, vector=False)
    assert b.item_expression(nocount, CFG) == \
        "(anl_x_4wk_up - anl_x_4wk_dn) / (anl_x_4wk_up + anl_x_4wk_dn + 1)"
    two = b.partner_expression([nocount, nocount], CFG)
    assert two.endswith(") / 2") and analyze(two, SPEC).ok


def test_variants_order_grid_and_tried():
    fams = b.families(meta(), CFG, skip=SPEC.base_fields)
    vs = b.variants(fams, {"own"}, CFG)
    # anl has more families, so first; least-used items at each weight, then the next ones
    assert [(ds, w, len(items)) for ds, items, w in vs[:3]] == [
        ("anl", 1.0, 3), ("anl", 2.0, 3)] + [("est", 1.0, 2)]
    # a dataset one of our accepted alphas read (both up and down) goes first
    proven = b.accepted_datasets(fams, ["rank(est_12m_pre_raisednum_4wks - "
                                        "est_12m_pre_lowerednum_4wks)"])
    assert proven == {"est"}
    assert b.variants(fams, {"own"}, CFG, proven=proven)[0][0] == "est"
    # combinations tried along the lineage are skipped; the own dataset never
    first = b.tag(*[vs[0][0], vs[0][1], vs[0][2]])
    again = b.variants(fams, {"own"}, CFG, tried=b.tried_tags([f"{first} rest"]))
    assert again[0][2] == 2.0
    assert not b.variants(fams, {"anl", "est"}, CFG)


def test_config_from_dict():
    c = b.BreadthConfig.from_dict({"enabled": True, "weights": [1, 1.5], "pairs": [["up", "dn"]],
                                   "categories": ["analyst"], "min_item_alphas": 3})
    assert c.enabled and c.weights == (1.0, 1.5) and c.pairs == (("up", "dn"),)
    for bad in ({"weights": [0]}, {"pairs": [["up"]]}, {"nope": 1}, {"items_per_variant": 0}):
        with pytest.raises(ConfigError):
            b.BreadthConfig.from_dict(bad)
    agents = load_agents_config(ROOT / "config" / "agents.yaml")
    assert agents.breadth.enabled and agents.recheck_every_rounds == 2


def lead(candidate, cid="p1", lever=None, parent=None, prod=0.74, expr="rank(own_f0)"):
    c = candidate(candidate_id=cid, dataset="own", lever=lever, parent_id=parent,
                  expression=expr, region="USA", delay=1,
                  generation=0 if parent is None else 1)
    r = ResultRecord(cid, "complete", sharpe=2.0, fitness=1.5, turnover=0.2,
                     check_values={"PROD_CORRELATION": {"value": prod, "limit": 0.7}})
    d = DecisionRecord(cid, "repair", "prod_correlation", gate_results=[
        GateResult("brain:PROD_CORRELATION", False, prod, 0.7, "")])
    return c, r, d


def agent(memory, playbook, rows):
    for c, r, d in rows:
        memory.put_candidate(c)
        memory.put_result(r)
        memory.put_decision(d)
    a = RepairAgent(playbook, SPEC, memory, llm=None, generator_config=GEN,
                    field_meta={("USA", 1): meta()}, markets={("USA", 1): list(meta())})
    a.breadth = CFG
    return a


def test_repair_wave_puts_breadth_blends_first_without_a_model(memory, playbook, candidate):
    c, r, d = lead(candidate)
    wave = agent(memory, playbook, [(c, r, d)]).repair_wave(c, d, r, size=4)
    kids = [k for k in wave.children if k.lever == b.BREADTH_LEVER]
    assert len(kids) == 2 and wave.children[:2] == kids and wave.llm_calls == 0
    assert kids[0].hypothesis.startswith("[breadth partner=anl items=anl_best_ptp_4wk,"
                                         "anl_best_sales_4wk,anl_best_eps_4wk weight=1]")
    assert "weight=2]" in kids[1].hypothesis
    e = kids[0].expression
    assert "rank(own_f0)" in e and "anl_best_ptp_4wk_up" in e and "max(anl_best_ptp_numest, 1)" in e
    assert "if_else(is_nan(" in e                           # no partner value -> 0.5
    assert "anl_market_status" not in e


def test_breadth_blends_only_for_close_leads_and_within_the_lineage_cap(memory, playbook,
                                                                        candidate):
    far = lead(candidate, prod=0.95)                         # distance 0.36 -> still close
    far_c, far_r, far_d = far
    assert b.eligible(far_d, far_r, {"prod_correlation"}, CFG, bp.distance)[0]
    farther = lead(candidate, cid="p2", prod=1.2)
    assert not b.eligible(farther[2], farther[1], {"prod_correlation"}, CFG, bp.distance)[0]
    # four breadth children already in the lineage: none more
    rows = [lead(candidate, cid="q1")]
    for i in range(4):
        k = lead(candidate, cid=f"q1_r1{'abcd'[i]}", lever=b.BREADTH_LEVER, parent="q1",
                 expr=f"rank(own_f0) + {i + 1} * rank(own_f1)")
        rows.append(k)
    a = agent(memory, playbook, rows)
    c, r, d = rows[0]
    out = RepairOutcome("q1")
    assert a._breadth_blends(c, d, r, memory.candidate_outcomes(), 4, out) == []
    assert any("already in this lineage" in n for n in out.notes)


def test_planner_lists_families_first_and_scores_by_field_usage(candidate):
    c, r, d = lead(candidate)
    outs = [{"candidate": c, "result": r, "decision": d}]
    cfg = bp.BlendConfig(enabled=True)
    cands = {p.dataset: p for p in bp.partner_candidates(
        c, r, {"prod_correlation"}, outs, meta(), SPEC, GEN, {}, cfg, 1.58)}
    anl = cands["anl"]
    assert anl.families == 4
    assert [x for x, _ in anl.fields[:3]] == ["anl_best_ptp_4wk_up", "anl_best_ptp_4wk_dn",
                                             "anl_best_ptp_numest"]
    assert anl.fields[0][1].startswith("revision family anl_best_ptp_4wk (up)")
    listed = [x for x, _ in anl.fields]
    assert "anl_market_status" not in listed[:9]            # 0-alpha fields after families
    # an accepted alpha (another dataset) whose blend read anl proves it
    acc = candidate(candidate_id="acc", dataset="zzz", region="USA", delay=1,
                    expression="rank(own_f0) + rank(anl_best_eps_4wk_up)")
    outs.append({"candidate": acc, "result": ResultRecord("acc", "complete", sharpe=2.0),
                 "decision": DecisionRecord("acc", "accept", None)})
    again = {p.dataset: p for p in bp.partner_candidates(
        c, r, {"prod_correlation"}, outs, meta(), SPEC, GEN, {}, cfg, 1.58)}
    assert again["anl"].proven and again["anl"].in_accept
    assert again["anl"].score > anl.score


def test_recheck_during_a_run_starts_the_wave(memory, config, playbook, candidate):
    from test_research import make_run
    run = make_run(memory, config, playbook)
    key = next(iter(run.arms))
    c, r, d = lead(candidate, cid="late1")
    memory.put_candidate(c)
    run.pipeline.recheck_pending = lambda: ([Outcome("late1", "repair", "prod_correlation",
                                                     result=r, decision_record=d)], [])
    run._arm_of = lambda cand, model=None: key
    started = []
    run._start_wave = lambda k, cand, dec, res, priority=(): (
        started.append((k, cand.candidate_id, dec.failure_class)) or RepairOutcome("late1"))
    run._recheck_pending()                                    # run start: resume takes it
    assert started == [] and run.rechecked[-1]["decision"] == "repair"
    run._recheck_pending(start_waves=True)                    # during a run: at once
    assert started == [(key, "late1", "prod_correlation")]


def test_resume_takes_rechecked_repairs_first(memory, config, playbook, candidate):
    from test_research import make_run
    run = make_run(memory, config, playbook)
    run.agents = dataclasses.replace(run.agents, repair=dataclasses.replace(
        run.agents.repair, max_resume=1))
    key = next(iter(run.arms))
    close = lead(candidate, cid="close1", prod=0.71)
    late = lead(candidate, cid="late2", prod=0.80)
    for row in (close, late):
        memory.put_candidate(row[0])
        memory.put_result(row[1])
        memory.put_decision(row[2])
    run._arm_of = lambda cand, model=None: key
    started = []

    def fake(k, cand, dec, res, priority=()):
        started.append(cand.candidate_id)
        out = RepairOutcome(cand.candidate_id)
        out.children = [dataclasses.replace(cand, lever="breadth_blend")]
        return out
    run._start_wave = fake
    run.rechecked = [{"candidate_id": "late2", "decision": "repair",
                      "failure_class": "prod_correlation"}]
    run._resume_repairs()
    assert started == ["late2"]

