"""Consultant level, per region (2026-11-11): BRAIN's check values and limits, the new failure
classes and their levers, per-region limits, lever learning, Director quotas and prompts."""

import copy
import json

import pytest

from alpha_research import prompts
from alpha_research.brain_api import translate
from alpha_research.config import ConfigError, config_from_dict, load_config
from alpha_research.evaluator import evaluate, run_gates
from alpha_research.fastexpr import FastExprSpec
from alpha_research.lever_stats import (LeverStats, RepairEvent, RepairLearningConfig,
                                        rank_levers, successful_examples)
from alpha_research.llm import ScriptedLLMClient
from alpha_research.orchestrator import Orchestrator
from alpha_research.repair import Playbook
from alpha_research.repair_agent import MECHANICAL, NEUTRALIZE_LEVERS, LeverNotApplicable, _metrics
from alpha_research.schemas import FAILURE_CLASSES, DecisionRecord, ResultRecord, SchemaError
from alpha_research.simulator import normalize
from alpha_research.strategy import (StrategyConfig, StrategyEvolution, region_breakdown,
                                     region_notes)

from conftest import FIXTURES, MAIN_CONFIG, ROOT

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
CONSULTANT = FIXTURES / "brain_api_capture_consultant"


def _capture(name):
    return json.loads((CONSULTANT / f"{name}.json").read_text())


def consultant_result():
    return normalize(translate(_capture("simulation"), _capture("alpha"), check=_capture("check")),
                     "acc_01", "s")


def passing(**kw):
    base = dict(candidate_id="c1", status="complete", sharpe=1.68, fitness=1.11, turnover=0.2578,
                returns=0.1119, sub_universe_sharpe=1.07, sub_universe_cutoff=0.73,
                self_correlation=0.1)
    base.update(kw)
    return ResultRecord(**base)


# ---------------------------------------------------------------- check values and alt stats

def test_translate_keeps_brain_check_values_and_risk_neutralized_stats():
    r = consultant_result()
    assert r.check_values["LOW_2Y_SHARPE"] == {"value": 1.05, "limit": 1.58}
    assert r.check_values["PROD_CORRELATION"] == {"value": None, "limit": 0.7}
    assert "CONCENTRATED_WEIGHT" not in r.check_values       # PASS with no value or limit
    assert r.alt_stats["risk_neutralized"]["sharpe"] == 2.39
    assert r.alt_stats["investability_constrained"]["sharpe"] == 1.25
    back = ResultRecord.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back.check_values == r.check_values and back.alt_stats == r.alt_stats


def test_result_rejects_malformed_check_values():
    with pytest.raises(SchemaError):
        ResultRecord("c1", "complete", check_values={"LOW_SHARPE": {"value": "high"}}).validate()
    with pytest.raises(SchemaError):
        normalize({"status": "complete", "alt_stats": {"risk_neutralized": 2.0}}, "c1", "s")


def test_old_payloads_without_the_new_fields_still_load():
    d = passing().to_dict()
    del d["check_values"], d["alt_stats"]
    assert ResultRecord.from_dict(d).check_values == {}


# ---------------------------------------------------------------- new classes and gates

def test_recent_sharpe_and_prod_correlation_have_levers(config, playbook):
    for check, cls in (("LOW_2Y_SHARPE", "recent_sharpe"), ("IS_LADDER_SHARPE", "recent_sharpe"),
                       ("PROD_CORRELATION", "prod_correlation")):
        d = evaluate(passing(checks={check: False}), config, playbook)
        assert (d.decision, d.failure_class) == ("repair", cls)


def test_playbook_classes_are_known_failure_classes(playbook):
    assert set(playbook.entries) <= set(FAILURE_CLASSES)
    assert {"recent_sharpe", "prod_correlation"} <= set(playbook.entries)


def test_brain_gate_carries_brains_value_and_limit(config, playbook):
    r = passing(checks={"LOW_2Y_SHARPE": False},
                check_values={"LOW_2Y_SHARPE": {"value": 1.05, "limit": 1.58}})
    g = next(g for g in run_gates(r, config) if g.name == "brain:LOW_2Y_SHARPE")
    assert (g.value, g.threshold) == (1.05, 1.58)


def test_recent_sharpe_far_miss_is_rejected(make_config, playbook):
    cfg = make_config(repair__near_miss_share={"recent_sharpe": 0.6})
    near = passing(checks={"LOW_2Y_SHARPE": False},
                   check_values={"LOW_2Y_SHARPE": {"value": 1.05, "limit": 1.58}})
    far = passing(checks={"LOW_2Y_SHARPE": False},
                  check_values={"LOW_2Y_SHARPE": {"value": 0.3, "limit": 1.58}})
    assert evaluate(near, cfg, playbook).decision == "repair"
    assert evaluate(far, cfg, playbook).decision == "reject"


def test_limits_from_brain_replace_ours(make_config, playbook):
    # BRAIN's limit for this alpha is 2.0 (another region or delay): 1.68 fails it
    r = passing(check_values={"LOW_SHARPE": {"value": 1.68, "limit": 2.0}})
    assert evaluate(r, make_config(gates__limits_from_brain=False), playbook).decision == "accept"
    d = evaluate(r, make_config(gates__limits_from_brain=True), playbook)
    assert (d.decision, d.failure_class) == ("repair", "sharpe")
    assert next(g for g in d.gate_results if g.name == "sharpe").threshold == 2.0


def test_main_config_takes_limits_from_brain_and_fails_acc01(playbook):
    d = evaluate(consultant_result(), load_config(MAIN_CONFIG), playbook)
    assert d.decision in ("repair", "reject")
    assert next(g for g in d.gate_results if g.name == "sharpe").threshold == 1.58


# ---------------------------------------------------------------- region overrides

def test_region_overrides_apply_by_region_then_delay(raw_config):
    raw = copy.deepcopy(raw_config)
    raw["regions"] = {"EUR": {"sharpe_min": 1.4}, "EUR/D0": {"sharpe_min": 2.0, "fitness_min": 1.3}}
    cfg = config_from_dict(raw)
    assert cfg.for_settings("USA", 1) is cfg
    assert cfg.for_settings("EUR", 1).gates.sharpe_min == 1.4
    d0 = cfg.for_settings("EUR", 0).gates
    assert (d0.sharpe_min, d0.fitness_min, d0.turnover_max) == (2.0, 1.3, cfg.gates.turnover_max)


@pytest.mark.parametrize("regions, match", [
    ({"europe": {"sharpe_min": 1}}, "USA or USA/D0"),
    ({"EUR": {"max_drawdown": 0.2}}, "may set only"),
    ({"EUR": {"turnover_max": 2.0}}, "turnover_max"),
])
def test_region_overrides_are_validated(raw_config, regions, match):
    raw = copy.deepcopy(raw_config)
    raw["regions"] = regions
    with pytest.raises(ConfigError, match=match):
        config_from_dict(raw)


def test_pipeline_uses_the_candidates_region(raw_config, playbook, memory, candidate):
    from alpha_research.brain_client import FixtureBrainClient
    from alpha_research.pipeline import Pipeline
    from conftest import BRAIN_DIR
    raw = copy.deepcopy(raw_config)
    raw["regions"] = {"EUR": {"sharpe_min": 2.0}}
    client = FixtureBrainClient(BRAIN_DIR)
    pipe = Pipeline(config_from_dict(raw), playbook, client, memory, sleep=lambda s: None,
                    clock=lambda: 0.0)
    usa = candidate(candidate_id="u1")
    eur = candidate(candidate_id="e1", region="EUR", universe="TOP2500")
    for c in (usa, eur):
        client.register(c.candidate_id, "sim_acc_01")
    assert pipe.process(usa).decision == "accept"
    d = pipe.process(eur)
    assert (d.decision, d.failure_class) == ("repair", "sharpe")


# ---------------------------------------------------------------- levers

def apply(name, cand, **step):
    return MECHANICAL[name](cand, step, SPEC, None)


def test_factor_neutralization_levers_follow_the_region(candidate):
    assert set(NEUTRALIZE_LEVERS) <= set(MECHANICAL)
    usa = candidate()
    assert apply("neutralize_statistical", usa).neutralization == "STATISTICAL"
    with pytest.raises(LeverNotApplicable, match="not allowed in USA"):
        apply("neutralize_country", usa)
    eur = candidate(region="EUR", universe="TOP2500")
    assert apply("neutralize_country", eur).neutralization == "COUNTRY"
    with pytest.raises(LeverNotApplicable, match="already"):
        apply("neutralize_crowding", candidate(neutralization="CROWDING"))


def test_change_universe_is_region_aware(candidate):
    assert apply("change_universe", candidate(), to="TOP1000").universe == "TOP1000"
    eur = candidate(region="EUR", universe="TOP2500")
    assert apply("change_universe", eur, to="TOP1000").universe == "TOPCS1600"   # next smaller
    with pytest.raises(LeverNotApplicable, match="no known smaller"):
        apply("change_universe", candidate(region="GLB", universe="TOP3000",
                                           neutralization="COUNTRY"), to="TOP1000")


def test_repair_metrics_show_failed_checks_and_risk_neutralized_sharpe():
    m = _metrics(consultant_result())
    assert m["failed_brain_checks"]["LOW_2Y_SHARPE"] == {"value": 1.05, "limit": 1.58}
    assert "LOW_TURNOVER" not in m["failed_brain_checks"]          # passed
    assert m["risk_neutralized"]["sharpe"] == 2.39


# ---------------------------------------------------------------- lever learning per region

def event(region, lever, reward, cls="sharpe"):
    return RepairEvent("p", "c", cls, lever, reward, reward == 1.0, False, (), None, None,
                       "a", "b", {}, region=region)


def test_region_estimate_starts_at_the_global_one_and_moves_with_its_own_evidence():
    events = [event("USA", "rank_transform", 1.0)] * 6 + [event("EUR", "rank_transform", 0.0)] * 6
    stats = LeverStats.from_events(events, region_strength=2.0)
    glob = stats.mean("sharpe", "rank_transform")
    assert stats.mean("sharpe", "rank_transform", "JPN") == glob        # no JPN evidence
    assert stats.mean("sharpe", "rank_transform", "USA") > glob > \
        stats.mean("sharpe", "rank_transform", "EUR")
    rows = stats.region_table()
    assert {r["region"] for r in rows} == {"USA", "EUR"}


def test_lever_order_differs_by_region():
    pb = Playbook.from_dict({"sharpe": [{"lever": "rank_transform"}, {"lever": "neutralize_statistical"}]})
    events = ([event("USA", "rank_transform", 1.0)] * 4 + [event("EUR", "neutralize_statistical", 1.0)] * 4
              + [event("EUR", "rank_transform", 0.0)] * 4)
    stats = LeverStats.from_events(events)
    cfg = RepairLearningConfig()
    usa = [lv.name for lv, _ in rank_levers(pb, "sharpe", [], stats, cfg, region="USA")]
    eur = [lv.name for lv, _ in rank_levers(pb, "sharpe", [], stats, cfg, region="EUR")]
    assert usa[0] == "rank_transform" and eur[0] == "neutralize_statistical"


def test_past_repairs_from_the_same_region_come_first():
    events = [event("USA", "a", 1.0), event("EUR", "b", 0.5)]
    assert [e["lever"] for e in successful_examples(events, "sharpe", 2, region="EUR")] == ["b", "a"]


def test_region_prior_strength_is_validated():
    assert RepairLearningConfig.from_dict({"region_prior_strength": 5}).region_prior_strength == 5
    with pytest.raises(ConfigError):
        RepairLearningConfig.from_dict({"region_prior_strength": -1})


# ---------------------------------------------------------------- Director per region

def test_region_floor_keeps_a_region_without_accepts_in_play():
    arms = [("pv1", "USA", "g", "m"), ("pv2", "USA", "g", "m"), ("pv1", "EUR", "g", "m")]
    orch = Orchestrator(arms, seed=0)
    for _ in range(30):
        orch.update(arms[0], True)
        orch.update(arms[1], True)
        orch.update(arms[2], False)
    plain = orch.allocate(6, floor_share=0.0)
    assert plain[arms[2]] == 0
    alloc = Orchestrator(arms, seed=0)
    alloc.state = orch.state
    got = alloc.allocate(6, floor_share=0.0, region_floor_share=0.5)
    assert got[arms[2]] >= 1 and sum(got.values()) == 6


def test_region_cap_spreads_slots():
    arms = [("pv1", "USA", "g", "m"), ("pv1", "EUR", "g", "m")]
    orch = Orchestrator(arms, seed=1)
    for _ in range(30):
        orch.update(arms[0], True)
    got = orch.allocate(10, floor_share=0.0, max_region_share=0.6)
    assert got[arms[0]] == 6 and got[arms[1]] == 4


# ---------------------------------------------------------------- prompts and Strategy

def test_brain_context_lists_regions_and_overrides(raw_config):
    raw = copy.deepcopy(raw_config)
    raw["regions"] = {"EUR/D0": {"sharpe_min": 2.0}}
    text = prompts.brain_context(config_from_dict(raw), ["USA", "EUR"])
    assert "Region USA: universes TOP3000" in text and "COUNTRY" in text
    assert "Region EUR:" in text and "Our limits for delay 0: sharpe_min 2" in text
    assert "LOW_2Y_SHARPE" in text and "PROD_CORRELATION" in text and "STATISTICAL" in text


def test_repair_prompt_explains_the_failure_class(candidate):
    from alpha_research.repair import Lever
    d = DecisionRecord("c1", "repair", "recent_sharpe", reasons=["brain:LOW_2Y_SHARPE"])
    text = prompts.repair_user(candidate(), Lever("restore_recent_performance"), d, {}, None,
                               SPEC.operators)
    assert "Failure class: recent_sharpe (Sharpe of the last in-sample years" in text
    assert prompts.LEVER_INSTRUCTIONS["restore_recent_performance"] in text


def _outcome(candidate, cid, region, neut, decision, cls="none"):
    c = candidate(candidate_id=cid, region=region, neutralization=neut,
                  universe="TOP3000" if region == "USA" else "TOP2500")
    return {"candidate": c, "decision": DecisionRecord(cid, decision, cls), "result": None}


def test_region_breakdown_and_notes(candidate):
    rows = [_outcome(candidate, "a", "USA", "SUBINDUSTRY", "accept"),
            _outcome(candidate, "b", "USA", "STATISTICAL", "reject", "sharpe"),
            _outcome(candidate, "c", "EUR", "COUNTRY", "reject", "sub_universe")]
    by = region_breakdown(rows)
    assert by["USA"]["accepted"] == 1 and by["USA"]["neutralizations"]["STATISTICAL"] == [1, 0]
    assert by["EUR"]["failures"] == {"sub_universe": 1}
    notes = region_notes(rows, "EUR")
    assert "In EUR so far: 0 of 1" in notes and "COUNTRY 0/1" in notes
    assert region_notes(rows, "JPN") is None


def test_region_notes_show_what_neutralization_repairs_did(candidate):
    # the Generator chooses neutralizations from this evidence (gen-v5)
    from alpha_research.schemas import GateResult, ResultRecord
    parent = candidate(candidate_id="p", neutralization="SUBINDUSTRY")
    child = candidate(candidate_id="p_r1", neutralization="STATISTICAL", parent_id="p",
                      lever="neutralize_statistical", generation=1)
    rows = [{"candidate": parent, "result": ResultRecord("p", "complete", sharpe=1.2),
             "decision": DecisionRecord("p", "repair", "sharpe", gate_results=[
                 GateResult("sharpe_min", False, 1.2, 1.58, "")])},
            {"candidate": child, "result": ResultRecord("p_r1", "complete", sharpe=1.9),
             "decision": DecisionRecord("p_r1", "accept", "none")}]
    notes = region_notes(rows, "USA")
    assert "Neutralization changes by the Repair Agent there" in notes
    assert "neutralize_statistical 1/0/1 (sharpe)" in notes


def test_strategy_region_guidance_goes_only_to_that_region(candidate):
    rows = [_outcome(candidate, "a", "USA", "SUBINDUSTRY", "reject", "sharpe")]
    reply = json.dumps({"guidance": ["g"], "region_guidance": {"EUR": ["use COUNTRY"]},
                        "hypotheses": []})
    llm = ScriptedLLMClient([reply])
    evo = StrategyEvolution(StrategyConfig(max_hypotheses=2), llm=llm).evolve(rows, "gen-v3")
    assert evo.region_guidance == {"EUR": "- use COUNTRY"}
    assert evo.guidance_for("EUR").endswith("In EUR:\n- use COUNTRY")
    assert evo.guidance_for("USA") == evo.guidance
    assert '"regions"' in llm.calls[0]["user"]
