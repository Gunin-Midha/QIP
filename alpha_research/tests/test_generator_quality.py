"""Run rs20261125130750 (2026-11-25): 43 simulations, no repair child, fresh alphas built on one
field or on heavily mined ones. Owner: leads get mechanical repairs, part of each batch is a
cross-dataset blend, the Generator must combine two sources (one not crowded), sees its
arm's history and partner fields with shown signal; GLB region checks are repairable; each
run reports how plain or crowded its fresh alphas were."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from alpha_research.evaluator import check_class
from alpha_research.fastexpr import FastExprSpec
from alpha_research.generator import (AlphaGenerator, GenerationSettings, Source, blend_count,
                                      build_system_prompt, heavily_used, load_generator_config,
                                      signal_profile, structure_errors, _generate_prompt)
from alpha_research.lever_stats import RepairLearningConfig, mechanical_only, wave_size
from alpha_research.quality import generation_quality, quality_lines
from alpha_research.repair import Playbook
from alpha_research.schemas import CandidateRecord, DecisionRecord, GateResult, ResultRecord
from alpha_research.config import ConfigError
from alpha_research.llm import ScriptedLLMClient
from alpha_research.strategy import arm_history, signal_evidence

from conftest import PLAYBOOK

ROOT = Path(__file__).resolve().parents[1]
SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
CFG = load_generator_config(ROOT / "config" / "generator.yaml")

META = {
    "opt_skew": {"dataset": "option8", "category": "option", "type": "MATRIX", "alpha_count": 1900},
    "opt_term": {"dataset": "option8", "category": "option", "type": "MATRIX", "alpha_count": 700},
    "anl_up": {"dataset": "analyst69", "category": "analyst", "type": "MATRIX", "alpha_count": 12},
    "anl_dn": {"dataset": "analyst69", "category": "analyst", "type": "MATRIX", "alpha_count": 9},
    "loan_x": {"dataset": "short3", "category": "shortinterest", "type": "MATRIX", "alpha_count": 3},
    "mdl_rank": {"dataset": "model39", "category": "model", "type": "MATRIX", "alpha_count": 948},
    "sector": {"dataset": "pv1", "type": "GROUP", "alpha_count": 99999},
}
CATALOG = sorted(META)


def settings(dataset="option8"):
    return GenerationSettings.from_mapping({
        "region": "USA", "universe": "TOP3000", "delay": 1, "decay": 4,
        "neutralization": "SUBINDUSTRY", "truncation": 0.08,
        "start_date": "2014-01-01", "end_date": "2023-12-31", "dataset": dataset})


def profile(expr, crowded=(), heavy=None):
    from alpha_research.fastexpr import analyze
    heavy = heavily_used(META, CFG.heavy_alphas, SPEC.base_fields) if heavy is None else heavy
    return signal_profile(analyze(expr, SPEC, datafields=CATALOG).fields_used,
                          SPEC.base_fields, META, CFG, crowded, heavy)


# ------------------------------------------------------------------ crowding and structure

def test_heavily_used_is_an_absolute_count_without_groups_or_base_fields():
    assert heavily_used(META, 500, SPEC.base_fields) == {"opt_skew", "opt_term", "mdl_rank"}
    assert heavily_used(META, None) == set()
    assert CFG.heavy_alphas == 500


def test_heavy_alphas_must_be_a_positive_integer(tmp_path):
    for bad in ("0", "true", "1.5"):
        p = tmp_path / "g.yaml"
        text = (ROOT / "config" / "generator.yaml").read_text()
        p.write_text(text.replace("heavy_alphas: 500", f"heavy_alphas: {bad}"))
        with pytest.raises(ConfigError):
            load_generator_config(p)


@pytest.mark.parametrize("expr, n_errors, needle", [
    ("rank(ts_delta(close, 5))", 1, "only price/volume"),
    ("rank(anl_up)", 1, "one signal source (anl_up)"),
    ("rank(anl_up) * rank(-returns)", 0, None),                  # field + price = 2 sources
    ("rank(anl_up - anl_dn)", 0, None),                          # two fields, one dataset
    ("rank(opt_skew) + rank(mdl_rank)", 1, "crowded or heavily used"),
    ("rank(opt_skew) + rank(anl_up)", 0, None),                  # heavy part + fresh part
    ("group_rank(opt_skew, sector)", 2, "one signal source"),    # groups are no source
])
def test_structure_rule(expr, n_errors, needle):
    errs = structure_errors(profile(expr), META, CFG)
    assert len(errs) == n_errors
    if needle:
        assert needle in errs[0]


def test_structure_rule_off_when_not_configured():
    off = replace(CFG, min_signal_sources=None)
    assert structure_errors(profile("rank(anl_up)"), META, off) == []


def test_profile_counts_datasets_and_most_used_field():
    p = profile("rank(opt_skew) + rank(anl_up) * rank(-returns)")
    assert p["datasets"] == ["analyst69", "option8", "price/volume"]
    assert p["sources"] == 3 and p["fresh"] == ["anl_up"] and p["max_alphas"] == 1900


# ------------------------------------------------------------------ the Generator

def test_a_single_field_alpha_is_sent_back_once_then_kept_when_fixed():
    first = json.dumps({"alphas": [{"expression": "rank(opt_skew)"},
                                   {"expression": "rank(opt_term) * rank(anl_up)"}]})
    fixed = json.dumps({"alphas": [{"expression": "rank(opt_skew) + rank(anl_dn)"}]})
    llm = ScriptedLLMClient([first, fixed])
    gen = AlphaGenerator(llm, SPEC, settings(), CFG, datafields=CATALOG, field_meta=META)
    res = gen.generate(Source("hypothesis", "skew and revisions"), 2, batch_id="b")
    assert [c.expression for c in res.candidates] == ["rank(opt_term) * rank(anl_up)",
                                                      "rank(opt_skew) + rank(anl_dn)"]
    assert "one signal source (opt_skew)" in llm.calls[1]["user"]


def test_a_still_plain_alpha_is_dropped_and_without_field_facts_the_rule_is_off():
    plain = json.dumps({"alphas": [{"expression": "rank(opt_skew)"}]})
    gen = AlphaGenerator(ScriptedLLMClient([plain, plain]), SPEC, settings(), CFG,
                         datafields=CATALOG, field_meta=META)
    res = gen.generate(Source("hypothesis", "skew"), 1, batch_id="b")
    assert res.candidates == [] and res.rejected
    bare = AlphaGenerator(ScriptedLLMClient([plain]), SPEC, settings(), CFG, datafields=CATALOG)
    assert len(bare.generate(Source("hypothesis", "skew"), 1, batch_id="b").candidates) == 1


def test_prompt_tags_heavy_fields_states_the_rule_and_offers_partners():
    gen = AlphaGenerator(ScriptedLLMClient([]), SPEC, settings(), CFG, datafields=CATALOG,
                         field_meta=META)
    gen.evidence = {"anl_up": 1.3, "loan_x": 1.1, "mdl_rank": 1.6, "opt_term": 2.0,
                    "anl_dn": 0.4}
    gen._crowded, gen._heavy = set(), heavily_used(META, 500, SPEC.base_fields)
    # own dataset, heavily used and weak fields are no partners; strongest first
    assert gen.partners() == [("anl_up", 1.3), ("loan_x", 1.1)]
    system = build_system_prompt(SPEC, CATALOG, "skew", CFG, meta=META, dataset="option8",
                                 partners=gen.partners())
    assert "heavily used" in system and "at least 2 independent sources" in system
    assert "anl_up [cov" not in system and "|Sharpe| 1.30" in system


def test_blend_count_and_request_text():
    assert [blend_count(k, 0.25) for k in (1, 2, 3, 4, 8)] == [0, 1, 1, 1, 2]
    assert blend_count(4, 0.0) == 0
    user = _generate_prompt("h", None, 4, blends=1, history=["- rank(x) | Sharpe 0.72"])
    assert "At least 1 of the 4 must be a blend across datasets" in user
    assert "<history>\n- rank(x) | Sharpe 0.72\n</history>" in user
    assert "<history>" not in _generate_prompt("h", None, 4)


def test_generator_sends_history_and_notes_missing_blends():
    # two fresh analyst fields, one dataset: valid, but no cross-dataset blend
    reply = json.dumps({"alphas": [{"expression": "rank(anl_up - anl_dn)"},
                                   {"expression": "rank(anl_dn / (anl_up + 1))"}]})
    llm = ScriptedLLMClient([reply])
    gen = AlphaGenerator(llm, SPEC, settings("analyst69"), CFG, datafields=CATALOG,
                         field_meta=META)
    gen.history = ["- rank(opt_skew) | Sharpe 0.72, reject / sharpe"]
    res = gen.generate(Source("hypothesis", "revisions"), 2, batch_id="b")
    assert "- rank(opt_skew) | Sharpe 0.72" in llm.calls[0]["user"]
    assert "At least 1 of the 2 must be a blend" in llm.calls[0]["user"]
    assert len(res.candidates) == 2
    assert "0 of 2 alphas read two datasets (1 asked)" in res.notes
    # a heavy field with price only is still all-heavy: sent back
    llm2 = ScriptedLLMClient([json.dumps({"alphas": [{"expression": "rank(opt_term) * rank(-returns)"}]}),
                              json.dumps({"alphas": [{"expression": "rank(opt_term) * rank(anl_up)"}]})])
    res2 = AlphaGenerator(llm2, SPEC, settings(), CFG, datafields=CATALOG,
                          field_meta=META).generate(Source("hypothesis", "term"), 1, batch_id="b")
    assert "crowded or heavily used" in llm2.calls[1]["user"]
    assert [c.expression for c in res2.candidates] == ["rank(opt_term) * rank(anl_up)"]


# ------------------------------------------------------------------ memory evidence

def _row(cid, expr, sharpe, dataset="option8", region="USA", prod=None, decision="reject",
         cls="sharpe", lever=None, rn=None):
    c = CandidateRecord.from_dict(dict(
        candidate_id=cid, expression=expr, region=region, universe="TOP3000", delay=1,
        decay=4, neutralization="SUBINDUSTRY", truncation=0.08, start_date="2014-01-01",
        end_date="2023-12-31", dataset=dataset, lever=lever,
        parent_id="p" if lever and lever != "critic_revision" else None))
    r = ResultRecord(cid, "complete", sharpe=sharpe, fitness=0.3, turnover=0.2)
    if prod is not None:
        r.check_values["PROD_CORRELATION"] = {"value": prod, "limit": 0.7}
    if rn is not None:
        r.alt_stats["risk_neutralized"] = {"sharpe": rn}
    return {"candidate": c, "result": r,
            "decision": DecisionRecord(cid, decision, cls, gate_results=[], reasons=[])}


def test_signal_evidence_takes_the_best_abs_sharpe_per_field_in_the_market():
    rows = [_row("a", "rank(anl_up) * rank(-returns)", -1.2),
            _row("b", "rank(anl_up)", 0.5, rn=1.5),
            _row("c", "rank(loan_x)", 2.0, region="EUR")]
    ev = signal_evidence(rows, "USA", 1, SPEC, CATALOG)
    assert ev == {"anl_up": 1.5}


def test_arm_history_newest_first_with_brain_numbers():
    rows = [_row("a", "rank(opt_skew)", 0.72),
            _row("b", "rank(anl_up)", 1.1, dataset="analyst69"),
            _row("c", "rank(opt_term)", 1.39, prod=0.91, cls="prod_correlation",
                 decision="repair")]
    lines = arm_history(rows, "option8", "USA", 1, 5)
    assert lines == ["- rank(opt_term) | Sharpe 1.39, fitness 0.30, turnover 20%, PROD 0.91, "
                     "repair / prod_correlation",
                     "- rank(opt_skew) | Sharpe 0.72, fitness 0.30, turnover 20%, reject / sharpe"]
    assert arm_history(rows, "option8", "USA", 1, 0) == []


def test_generation_quality_counts_fresh_alphas_only():
    rows = [_row("rs1r01a1_001", "rank(opt_skew)", 0.7),
            _row("rs1r01a1_002", "rank(opt_skew) + rank(anl_up)", 1.2, prod=0.65),
            _row("rs1r01a1_003", "rank(ts_delta(close, 5))", -0.3),
            _row("rs1r01a1_002_r1a", "rank(opt_skew) + rank(anl_up)", 1.5, lever="size_tilt"),
            _row("rs2r01a1_001", "rank(anl_up)", 2.0)]
    q = generation_quality(rows, "rs1", SPEC, CFG, lambda r, d: (CATALOG, META))
    assert q["fresh_alphas"] == 3 and q["profiled"] == 3
    assert q["one_source_or_less"] == round(2 / 3, 3)
    assert q["price_volume_only"] == round(1 / 3, 3)
    assert q["all_fields_crowded_or_heavy"] == round(1 / 3, 3)
    assert q["two_or_more_datasets"] == round(1 / 3, 3)
    assert q["best_abs_sharpe_at_least_1"] == 1 and q["prod_correlations"] == [0.65]
    assert q["mean_abs_sharpe"] == round((0.7 + 1.2 + 0.3) / 3, 3)
    assert len(quality_lines(q)) == 4


# ------------------------------------------------------------------ repairs and GLB checks

TIERS = {"wave_tiers": [{"max_distance": 0.15, "fixes": 4}, {"max_distance": 0.4, "fixes": 2},
                        {"max_distance": 0.6, "fixes": 2, "mechanical_only": True},
                        {"max_distance": None, "fixes": 0}],
         "value_sharpe_share": 0.8, "low_value_fixes": 0, "blocked_fixes": 0}


def _far(sharpe, extra=()):
    gates = [GateResult("brain:LOW_SHARPE", False, sharpe, 1.58, ""),
             GateResult("brain:LOW_FITNESS", False, 0.4, 1.0, ""),
             GateResult("brain:CONCENTRATED_WEIGHT", False, 0.8, 0.1, ""), *extra]
    return DecisionRecord("c1", "repair", "sharpe", gate_results=gates, reasons=[])


@pytest.mark.parametrize("sharpe, rn, fixes", [
    (1.39, None, 2),     # a lead far from accept: 2 mechanical fixes
    (0.80, 1.10, 2),     # risk-neutralized Sharpe makes it a lead
    (0.90, None, 0),     # below 0.63 x 1.58
])
def test_leads_get_mechanical_fixes_however_far(sharpe, rn, fixes):
    cfg = RepairLearningConfig.from_dict({**TIERS, "lead_sharpe_share": 0.63, "lead_fixes": 2})
    pb = Playbook.load(PLAYBOOK)
    r = ResultRecord("c1", "complete", sharpe=sharpe)
    if rn is not None:
        r.alt_stats["risk_neutralized"] = {"sharpe": rn}
    n, why = wave_size(_far(sharpe), r, cfg, pb)
    assert n == fixes
    assert mechanical_only(_far(sharpe), r, cfg) is (fixes > 0)
    if fixes:
        assert "lead" in why
    off = RepairLearningConfig.from_dict(TIERS)
    assert wave_size(_far(sharpe), r, off, pb)[0] == 0


def test_a_lead_failing_a_check_no_lever_fixes_gets_nothing():
    cfg = RepairLearningConfig.from_dict({**TIERS, "lead_sharpe_share": 0.63, "lead_fixes": 2})
    d = _far(1.4, [GateResult("brain:SOMETHING_NEW", False, 0.1, 1.0, "")])
    assert wave_size(d, ResultRecord("c1", "complete", sharpe=1.4), cfg,
                     Playbook.load(PLAYBOOK))[0] == 0


def test_agents_config_turns_leads_on():
    import yaml
    raw = yaml.safe_load((ROOT / "config" / "agents.yaml").read_text())["repair"]
    cfg = RepairLearningConfig.from_dict(raw)
    assert (cfg.lead_sharpe_share, cfg.lead_fixes) == (0.63, 2)


@pytest.mark.parametrize("check", ["LOW_GLB_AMER_SHARPE", "LOW_GLB_APAC_SHARPE",
                                   "LOW_GLB_EMEA_SHARPE"])
def test_glb_region_checks_are_sub_universe(check):
    assert check_class(f"brain:{check}") == "sub_universe"
