"""BRAIN's tips for production correlation built in (2026-11-25, owner chose A + B).
A: the production-correlation playbook has mechanical levers next to the model's: factor
neutralizations (incl. SLOW, FAST) and a smaller, more liquid universe, so one wave tries
new information, a neutralization and a universe at once.
B: BRAIN's dataset value score and alpha count (the dataset lists saved next to the field
catalogs) steer the Generator's field list, Strategy Evolution's least-tried datasets and
new-dataset arms, and the blend partner's shortlist toward less crowded data."""

import dataclasses
import json

from alpha_research.generator import (_datasets_line, dataset_appeal, dataset_facts,
                                      load_datafield_meta, select_datafields)
from alpha_research.repair import Playbook
from alpha_research.repair_agent import RepairAgent, _partner_line, partner_fields
from alpha_research.research import Budget, ResearchRun
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord
from alpha_research.simulator import VirtualClock
from alpha_research.strategy import Evolution, summarize

from conftest import PLAYBOOK
from test_rescue_lessons import SPEC
from test_research import AGENTS, GEN_CONFIG, TWO_MODELS, RouterLLM, StubBrain
from test_rotation import region_plans


# ---------------------------------------------------------------- A: playbook levers

def prod_decision(cid="c1"):
    gates = [GateResult("brain:PROD_CORRELATION", False, 0.76, 0.7, "")]
    return DecisionRecord(cid, "repair", "prod_correlation", gate_results=gates,
                          reasons=["brain:PROD_CORRELATION failed: value 0.76 vs 0.7"])


def test_prod_correlation_entry_has_factor_neutralizations_and_universe():
    names = [lv.name for lv in Playbook.load(PLAYBOOK).entries["prod_correlation"]]
    assert names[0] == "blend_other_dataset"
    for lever in ("neutralize_statistical", "change_universe", "neutralize_slow",
                  "neutralize_fast", "neutralize_slow_and_fast"):
        assert lever in names
    assert names.index("neutralize_statistical") < names.index("change_universe")
    assert names[-1] == "model_repair"


def test_a_prod_wave_tries_a_neutralization_and_a_smaller_universe(memory, candidate):
    agent = RepairAgent(Playbook.load(PLAYBOOK), SPEC, memory, llm=None)
    c = candidate(region="EUR", universe="TOP2500", neutralization="INDUSTRY",
                  expression="group_rank(-ts_sum(returns, 10), industry)")
    out = agent.repair_wave(c, prod_decision(), ResultRecord("c1", "complete", sharpe=2.0),
                            size=4)
    by_lever = {ch.lever: ch for ch in out.children}
    assert by_lever["neutralize_statistical"].neutralization == "STATISTICAL"
    assert by_lever["change_universe"].universe == "TOPCS1600"   # EUR's next smaller
    assert len(out.children) == 4


def test_single_universe_regions_skip_the_universe_lever(memory, candidate):
    agent = RepairAgent(Playbook.load(PLAYBOOK), SPEC, memory, llm=None)
    c = candidate(region="IND", universe="TOP500", neutralization="SECTOR",
                  expression="group_rank(-ts_sum(returns, 10), industry)")
    out = agent.repair_wave(c, prod_decision(), ResultRecord("c1", "complete", sharpe=2.0),
                            size=4)
    assert "change_universe" not in [ch.lever for ch in out.children]
    assert out.children   # other levers fill the wave


# ---------------------------------------------------------------- B: dataset facts

FIELD_META = {
    "pv_a": {"dataset": "pv1", "category": "pv", "type": "MATRIX", "alpha_count": 500,
             "dataset_value_score": 1.0, "dataset_alpha_count": 400000},
    "ins_a": {"dataset": "insiders9", "category": "insiders", "type": "MATRIX", "alpha_count": 3,
              "dataset_value_score": 5.0, "dataset_alpha_count": 150},
    "ins_b": {"dataset": "insiders9", "category": "insiders", "type": "MATRIX", "alpha_count": 2,
              "dataset_value_score": 5.0, "dataset_alpha_count": 150},
    "ins_c": {"dataset": "insiders9", "category": "insiders", "type": "MATRIX", "alpha_count": 1,
              "dataset_value_score": 5.0, "dataset_alpha_count": 150},
    "anl_a": {"dataset": "analyst4", "category": "analyst", "type": "MATRIX", "alpha_count": 40,
              "dataset_value_score": 2.0, "dataset_alpha_count": 90000},
    "anl_b": {"dataset": "analyst4", "category": "analyst", "type": "MATRIX", "alpha_count": 30,
              "dataset_value_score": 2.0, "dataset_alpha_count": 90000},
}


def test_field_catalog_picks_up_the_dataset_list_next_to_it(tmp_path):
    (tmp_path / "datafields_eur_top2500_d1.json").write_text(json.dumps([
        {"id": "f1", "dataset": {"id": "macro27"}, "category": {"id": "macro"},
         "type": "MATRIX", "alphaCount": 4, "coverage": 0.9},
        {"id": "f2", "dataset": {"id": "unlisted"}, "type": "MATRIX"}]))
    (tmp_path / "datasets_eur_top2500_d1.json").write_text(json.dumps([
        {"id": "macro27", "category": {"id": "macro"}, "valueScore": 5.0, "alphaCount": 177,
         "userCount": 20}]))
    meta = load_datafield_meta(tmp_path / "datafields_eur_top2500_d1.json")
    assert meta["f1"]["dataset_value_score"] == 5.0 and meta["f1"]["dataset_alpha_count"] == 177
    assert meta["f2"]["dataset_value_score"] is None
    # no dataset list: same fields, no facts
    (tmp_path / "datasets_eur_top2500_d1.json").unlink()
    assert load_datafield_meta(tmp_path / "datafields_eur_top2500_d1.json")["f1"][
        "dataset_value_score"] is None


def test_appeal_puts_high_value_then_little_used_datasets_first():
    facts = dataset_facts(FIELD_META)
    assert facts["insiders9"]["fields"] == 3
    order = sorted(facts, key=lambda d: dataset_appeal(facts[d]))
    assert order == ["insiders9", "analyst4", "pv1"]
    assert dataset_appeal(None) > dataset_appeal(facts["pv1"])   # unknown last


def test_exploration_draws_less_crowded_datasets_first_and_twice_per_turn():
    # all of a 3-field list is exploration: insiders9 (value 5, above the median 2) gives two
    # fields in its turn, then analyst4 one; pv1 (value 1) does not get in
    relevant, explore = select_datafields(list(FIELD_META), "unrelated words", 3,
                                          explore_share=1.0, meta=FIELD_META)
    assert relevant == []
    assert [FIELD_META[f]["dataset"] for f in explore] == ["insiders9", "insiders9", "analyst4"]
    # relevance ties (no word matches) also go to the least crowded dataset
    relevant, _ = select_datafields(list(FIELD_META), "unrelated words", 2, meta=FIELD_META)
    assert {FIELD_META[f]["dataset"] for f in relevant} == {"insiders9"}


def test_generator_prompt_lists_datasets_with_value_and_usage():
    line = _datasets_line(list(FIELD_META), FIELD_META)
    assert line.index("insiders9 (insiders, value 5, 150 BRAIN alphas)") < line.index("pv1")
    assert _datasets_line(["x"], {"x": {"dataset": "d"}}) is None


def test_strategy_least_tried_is_ordered_by_appeal_and_shows_facts():
    info = dataset_facts(FIELD_META)
    s = summarize([], known_datasets=["pv1", "analyst4", "insiders9"], dataset_info=info)
    assert s["least_tried"] == ["insiders9", "analyst4", "pv1"]
    assert s["least_tried_facts"]["insiders9"] == {
        "our_trials": 0, "category": "insiders", "value_score": 5.0, "brain_alphas": 150}
    assert "least_tried_facts" not in summarize([], known_datasets=["a", "b"])


def test_partner_shortlist_prefers_less_crowded_datasets_and_shows_their_facts():
    meta = {f: {**m, "coverage": 0.9, "alpha_count": max(m["alpha_count"], 5)}
            for f, m in FIELD_META.items()}
    # topics favour analyst4; per_dataset 2, 4 slots: plain = analyst4 x2 + insiders9 x2
    # (topic ties); with half kept for value, the topic half is analyst4's two and the value
    # half insiders9's two: the topic order's picks stay, value adds what topics would not
    topic = "analyst estimate"
    meta["anl_a"]["description"] = meta["anl_b"]["description"] = "analyst estimate"
    meta["ins_a"]["description"] = "insider buying"
    plain = partner_fields(meta, {"pv1"}, SPEC, exclude_categories=["pv"], per_dataset=2,
                           max_fields=2, topics=topic)
    valued = partner_fields(meta, {"pv1"}, SPEC, exclude_categories=["pv"], per_dataset=2,
                            max_fields=4, topics=topic, value_share=0.5)
    assert [meta[f]["dataset"] for f, _ in plain] == ["analyst4", "analyst4"]
    assert [meta[f]["dataset"] for f, _ in valued] == ["analyst4", "analyst4",
                                                       "insiders9", "insiders9"]
    assert "value 5, 150 BRAIN alphas" in _partner_line("ins_a", meta["ins_a"])
    assert Playbook.load(PLAYBOOK).entries["prod_correlation"][0].step["value_share"] == 0.25


def test_new_dataset_arms_go_to_the_least_crowded_dataset_first(memory, config, playbook):
    meta = {"close": {"dataset": "pv1"},
            "flow": {"dataset": "newds", "dataset_value_score": 1.0,
                     "dataset_alpha_count": 5000},
            "insd": {"dataset": "insiders9", "dataset_value_score": 5.0,
                     "dataset_alpha_count": 400},
            # thinly used (owner 2026-11-28: designer.min_brain_alphas 300): no new arm
            "thin": {"dataset": "thin9", "dataset_value_score": 9.0,
                     "dataset_alpha_count": 50}}
    markets = {(r, 1): (list(meta), meta) for r in ("USA", "EUR", "JPN")}
    agents = dataclasses.replace(
        AGENTS, models=TWO_MODELS, regions_per_run=2, region_balance_since=None,
        strategy=dataclasses.replace(AGENTS.strategy, max_new_arms=1))
    vc = VirtualClock()
    run = ResearchRun(region_plans(), agents, Budget(max_simulations=8, per_round=4,
                                                     max_llm_calls=50),
                      llm_factory=lambda m, r=RouterLLM(): r, client=StubBrain(), config=config,
                      playbook=playbook, spec=SPEC, generator_config=GEN_CONFIG, memory=memory,
                      markets=markets, seed=0, sleep=vc.sleep, clock=vc.clock, run_id="rs1")
    run.log.append({"round": 1, "arms": []})
    seen = {}

    def fake_evolve(*a, **k):
        seen.update(k.get("dataset_info") or {})
        return Evolution("gen-v13+evo1", "guidance",
                         [("newds", "flow idea"), ("insiders9", "insider idea"),
                          ("thin9", "thin idea")])

    run.strategist.evolve = fake_evolve
    run._evolve(1)
    assert seen["insiders9"]["value_score"] == 5.0
    assert [k[0] for k in run.arms if k[0] in ("newds", "insiders9", "thin9")] == ["insiders9"]
