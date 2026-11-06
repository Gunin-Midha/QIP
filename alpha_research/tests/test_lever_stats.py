"""New mechanical levers, lever learning from memory, learned lever order and variants."""

import pytest

from alpha_research.config import ConfigError
from alpha_research.fastexpr import FastExprSpec
from alpha_research.lever_stats import (LeverRecord, LeverStats, RepairLearningConfig,
                                        failing_classes,
                                        rank_levers, repair_events, slack,
                                        successful_examples)
from alpha_research.repair import Lever, Playbook
from alpha_research.repair_agent import MECHANICAL, LeverNotApplicable, RepairAgent
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord

from conftest import ROOT

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")


def apply(name, cand, result=None, **step):
    return MECHANICAL[name](cand, step, SPEC, result)


# ---------------------------------------------------------------- new mechanical levers

def test_flip_sign_only_for_negative_sharpe(candidate):
    c = candidate(expression="a = rank(close); a - rank(open)")
    assert apply("flip_sign", c, ResultRecord("c1", "complete", sharpe=-0.9)).expression == \
        "a = rank(close); -(a - rank(open))"
    neg = candidate(expression="-rank(close)")
    assert apply("flip_sign", neg, ResultRecord("c1", "complete", sharpe=-1.0)).expression == "rank(close)"
    with pytest.raises(LeverNotApplicable, match="not negative"):
        apply("flip_sign", c, ResultRecord("c1", "complete", sharpe=0.9))
    with pytest.raises(LeverNotApplicable, match="needs BRAIN's Sharpe"):
        apply("flip_sign", c)


def test_set_neutralization_only_moves_to_a_finer_group(candidate):
    assert apply("set_neutralization", candidate(neutralization="NONE"),
                 to="SUBINDUSTRY").neutralization == "SUBINDUSTRY"
    with pytest.raises(LeverNotApplicable, match="already SUBINDUSTRY"):
        apply("set_neutralization", candidate(neutralization="SUBINDUSTRY"), to="INDUSTRY")


def test_backfill_fields_wraps_only_catalog_fields(candidate):
    c = candidate(expression="s = divide(implied_volatility_call_60, add(close, 1)); "
                             "group_zscore(ts_decay_linear(s, 10), subindustry) + ts_backfill(assets, 5)")
    out = apply("backfill_fields", c, lookback=63).expression
    assert out == ("s = divide(ts_backfill(implied_volatility_call_60, 63), add(close, 1)); "
                   "group_zscore(ts_decay_linear(s, 10), subindustry) + ts_backfill(assets, 5)")
    with pytest.raises(LeverNotApplicable, match="no datafield"):
        apply("backfill_fields", candidate(expression="rank(close)"), lookback=63)


def test_liquidity_tilt_winsorize_and_universe(candidate):
    c = candidate(expression="rank(-ts_delta(close, 5))")
    tilted = apply("liquidity_tilt", c, field="adv20")
    assert tilted.expression == "rank(-ts_delta(close, 5)) * rank(adv20)"
    with pytest.raises(LeverNotApplicable, match="already tilted"):
        apply("liquidity_tilt", tilted, field="adv20")
    with pytest.raises(LeverNotApplicable, match="not a base field"):
        apply("liquidity_tilt", c, field="nonsense")
    assert apply("winsorize", c, std=3).expression == "winsorize(rank(-ts_delta(close, 5)), std=3)"
    assert apply("change_universe", c, to="TOP1000").universe == "TOP1000"
    with pytest.raises(LeverNotApplicable, match="not larger"):
        apply("change_universe", candidate(universe="TOP500"), to="TOP1000")


# ---------------------------------------------------------------- evidence from memory

def gate(name, passed, value, threshold):
    return GateResult(name, passed, value, threshold, "")


def store(memory, cand, decision, failure_class, gates, result):
    memory.put_candidate(cand)
    memory.put_result(result)
    memory.put_decision(DecisionRecord(cand.candidate_id, decision, failure_class,
                                       gate_results=gates))


def test_failing_classes_include_brain_checks():
    d = DecisionRecord("c1", "repair", "sub_universe",
                       gate_results=[gate("sub_universe", False, 0.55, 0.95),
                                     gate("sharpe", True, 2.2, 1.25)])
    r = ResultRecord("c1", "complete", checks={"CONCENTRATED_WEIGHT": False, "LOW_SHARPE": True})
    assert failing_classes(d, r) == {"sub_universe", "concentration"}
    assert slack("sub_universe", d, r) == pytest.approx(-0.40)
    assert slack("concentration", d, ResultRecord("c1", "complete", weight_concentration=0.15)) == -0.15


def history(memory, candidate):
    """sub_universe parent; one child fixed it (liquidity_tilt), one made it worse (rank)."""
    parent = candidate(candidate_id="p1")
    store(memory, parent, "repair", "sub_universe",
          [gate("sharpe", True, 2.2, 1.25), gate("sub_universe", False, 0.55, 0.95)],
          ResultRecord("p1", "complete", sharpe=2.2))
    good = candidate(candidate_id="p1_r1a", parent_id="p1", lever="liquidity_tilt", generation=1,
                     expression="rank(-ts_delta(close, 5)) * rank(adv20)")
    store(memory, good, "accept", "none",
          [gate("sharpe", True, 2.0, 1.25), gate("sub_universe", True, 1.1, 0.86)],
          ResultRecord("p1_r1a", "complete", sharpe=2.0))
    bad = candidate(candidate_id="p1_r1b", parent_id="p1", lever="rank_transform", generation=1,
                    expression="rank(rank(-ts_delta(close, 5)))")
    store(memory, bad, "reject", "sharpe",
          [gate("sharpe", False, 1.0, 1.25), gate("sub_universe", False, 0.3, 0.43)],
          ResultRecord("p1_r1b", "complete", sharpe=1.0))
    broken = candidate(candidate_id="p1_r1c", parent_id="p1", lever="backfill_fields",
                       generation=1, expression="rank(ts_backfill(x, 63))")
    store(memory, broken, "reject", "incomplete", [], ResultRecord("p1_r1c", "failed"))


def test_repair_events_reward_fixes_improvements_and_collateral(memory, candidate):
    history(memory, candidate)
    ev = {e.lever: e for e in repair_events(memory.candidate_outcomes())}
    assert ev["liquidity_tilt"].reward == 1.0 and ev["liquidity_tilt"].fixed
    # sub-universe slack improved (-0.40 -> -0.13), but Sharpe now fails and the alpha as a
    # whole is further from accept: 0.5 x 0.5 (class improved) + 0.5 x 0 (no overall gain)
    assert ev["rank_transform"].improved and ev["rank_transform"].collateral == ("sharpe",)
    assert ev["rank_transform"].reward == 0.25
    assert ev["backfill_fields"].reward == 0.0 and not ev["backfill_fields"].improved
    stats = LeverStats.from_events(ev.values())
    assert stats.mean("sub_universe", "liquidity_tilt") == pytest.approx(2 / 3)
    assert stats.mean("sub_universe", "rank_transform") == pytest.approx(1.25 / 3)
    assert stats.mean("sub_universe", "never_tried") == 0.5
    rows = stats.table()
    assert rows[0]["lever"] == "liquidity_tilt" and rows[0]["fixed"] == 1
    ex = successful_examples(ev.values(), "sub_universe", 3)
    assert [e["lever"] for e in ex] == ["liquidity_tilt"] and "rank(adv20)" in ex[0]["after"]


PB = Playbook({
    "sub_universe": [Lever("rank_transform"), Lever("backfill_fields", {"lookback": 63}),
                     Lever("liquidity_tilt", {"field": "adv20"})],
    "concentration": [Lever("backfill_fields", {"lookback": 63})],
})


def test_rank_levers_keeps_playbook_order_without_evidence():
    ranked = rank_levers(PB, "sub_universe", {"sub_universe"}, LeverStats(),
                         RepairLearningConfig())
    assert [lv.name for lv, _ in ranked] == ["rank_transform", "backfill_fields", "liquidity_tilt"]


def test_rank_levers_prefers_a_lever_that_also_fixes_another_failure():
    ranked = rank_levers(PB, "sub_universe", {"sub_universe", "concentration"}, LeverStats(),
                         RepairLearningConfig())
    assert ranked[0][0].name == "backfill_fields" and ranked[0][1] == 0.75


def test_flip_sign_first_and_model_repair_last_whatever_the_scores():
    pb = Playbook.from_dict({
        "sharpe": [{"lever": "rank_transform"}, {"lever": "model_repair"}, {"lever": "flip_sign"}],
        "fitness": [{"lever": "model_repair"}], "sub_universe": [{"lever": "model_repair"}]})
    stats = LeverStats(records={("sharpe", "model_repair"): LeverRecord(n=10, reward=10)})
    for order in ("learned", "thompson", "playbook"):
        ranked = rank_levers(pb, "sharpe", {"sharpe", "fitness", "sub_universe"}, stats,
                             RepairLearningConfig(order=order))
        assert [lv.name for lv, _ in ranked] == ["flip_sign", "rank_transform", "model_repair"]


def test_rank_levers_learns_from_evidence(memory, candidate):
    history(memory, candidate)
    stats = LeverStats.from_events(repair_events(memory.candidate_outcomes()))
    ranked = rank_levers(PB, "sub_universe", {"sub_universe"}, stats, RepairLearningConfig())
    assert [lv.name for lv, _ in ranked] == ["liquidity_tilt", "rank_transform", "backfill_fields"]
    fixed = rank_levers(PB, "sub_universe", {"sub_universe"}, stats,
                        RepairLearningConfig(order="playbook"))
    assert fixed[0][0].name == "rank_transform"


def test_learning_config_parsing():
    assert RepairLearningConfig.from_dict({"use_model": True}) == RepairLearningConfig()
    cfg = RepairLearningConfig.from_dict({"order": "thompson", "prior": [2, 3]})
    assert cfg.order == "thompson" and cfg.prior == (2.0, 3.0)
    for bad in ({"order": "random"}, {"prior": [0, 1]}, {"secondary_weight": 2},
                {"resume_from_memory": "yes"}, {"examples_in_prompt": -1}):
        with pytest.raises(ConfigError):
            RepairLearningConfig.from_dict(bad)


# ---------------------------------------------------------------- the agent

def test_agent_uses_what_worked_before(memory, candidate):
    history(memory, candidate)
    agent = RepairAgent(PB, SPEC, memory)
    d = DecisionRecord("n1", "repair", "sub_universe",
                       gate_results=[gate("sub_universe", False, 0.4, 0.9)])
    out = agent.repair(candidate(candidate_id="n1"), d)
    assert out.lever == "liquidity_tilt" and out.notes[0].startswith("lever order: liquidity_tilt")


def test_variants_one_child_per_applicable_lever(memory, candidate, playbook):
    agent = RepairAgent(playbook, SPEC, memory)
    parent = candidate(candidate_id="iv", neutralization="SUBINDUSTRY", decay=10,
                       expression="s = implied_volatility_call_60 - implied_volatility_put_60; "
                                  "group_zscore(ts_decay_linear(s, 10), subindustry)")
    d = DecisionRecord("iv", "repair", "sub_universe",
                       gate_results=[gate("sub_universe", False, 0.55, 0.95)])
    r = ResultRecord("iv", "complete", sharpe=2.2, checks={"CONCENTRATED_WEIGHT": False})
    kids, notes = agent.variants(parent, d, r, ignore_budget=True)
    levers = [k.lever for k in kids]
    # backfill_fields addresses both failures, so it comes first
    assert levers[0] == "backfill_fields"
    assert {"liquidity_tilt", "rank_transform", "change_universe", "winsorize",
            "tighten_truncation"} <= set(levers)
    assert len({k.candidate_id for k in kids}) == len(kids)
    assert all(k.parent_id == "iv" and k.candidate_id.startswith("iv_r1") for k in kids)
    assert any("model lever" in n for n in notes)
    assert len(agent.variants(parent, d, r, limit=2)[0]) == 2
    # the lineage budget (max_trials_per_root 4) caps a hand rescue too
    assert len(agent.variants(parent, d, r)[0]) == 4


def test_model_repair_may_change_settings(memory, candidate):
    import json
    from alpha_research.llm import ScriptedLLMClient
    pb = Playbook({"sharpe": [Lever("model_repair")]})
    llm = ScriptedLLMClient([json.dumps({"expression": "rank(-ts_delta(close, 5))",
                                         "neutralization": "INDUSTRY", "decay": 7})])
    out = RepairAgent(pb, SPEC, memory, llm=llm).repair(
        candidate(neutralization="NONE"), DecisionRecord("c1", "repair", "sharpe"))
    assert out.child.neutralization == "INDUSTRY" and out.child.decay == 7
    assert out.child.lever == "model_repair"


def test_cli_repair_and_levers(tmp_path, candidate, capsys):
    import json
    from alpha_research import cli
    from alpha_research.memory import MemoryStore
    from alpha_research.pipeline import load_candidate_file
    data = tmp_path / "data"
    mem = MemoryStore(data)
    history(mem, candidate)
    mem.close()
    common = ["--data-dir", str(data), "--agents", str(ROOT / "config" / "agents.yaml")]
    out = tmp_path / "rescue.json"
    rc = cli.main(["repair", "--candidate-id", "p1", "--out", str(out), "--limit", "3",
                   "--ignore-budget",
                   "--playbook", str(ROOT / "config" / "repair_playbook.yaml"),
                   "--fastexpr", str(ROOT / "config" / "fastexpr.yaml"), *common])
    assert rc == 0
    kids = [e.candidate for e in load_candidate_file(out)]
    # p1's liquidity_tilt child is already in memory, so that experiment is skipped; rank and
    # backfill do not apply to rank(-ts_delta(close, 5)); ids never collide with memory
    assert [k.lever for k in kids] == ["change_universe"] and kids[0].parent_id == "p1"
    assert kids[0].candidate_id == "p1_r1d"   # p1_r1a, b, c are in memory
    # within budget p1's lineage is closed: its last two repairs did not get closer
    capsys.readouterr()
    assert cli.main(["repair", "--candidate-id", "p1", "--out", str(out),
                     "--playbook", str(ROOT / "config" / "repair_playbook.yaml"),
                     "--fastexpr", str(ROOT / "config" / "fastexpr.yaml"), *common]) == 6
    assert "stopped improving" in capsys.readouterr().out
    assert cli.main(["repair", "--candidate-id", "nope", "--out", str(out),
                     "--playbook", str(ROOT / "config" / "repair_playbook.yaml"),
                     "--fastexpr", str(ROOT / "config" / "fastexpr.yaml"), *common]) == 2
    capsys.readouterr()
    assert cli.main(["levers", *common]) == 0
    table = json.loads(capsys.readouterr().out)
    assert table["lever_stats"][0]["lever"] == "liquidity_tilt" and len(table["events"]) == 3


# ---------------------------------------------------------------- secondary credit, combinations

def iv_history(memory, candidate):
    """Mirrors the 2026-11-04 rescue: parent fails sub-universe and BRAIN's concentration
    check; backfill clears concentration only; the combination fixes both."""
    parent = candidate(candidate_id="iv", expression="group_zscore(ts_mean(iv_call, 10), subindustry)")
    store(memory, parent, "repair", "sub_universe",
          [gate("sharpe", True, 2.2, 1.25), gate("sub_universe", False, 0.55, 0.95)],
          ResultRecord("iv", "complete", sharpe=2.2, weight_concentration=0.15,
                       checks={"CONCENTRATED_WEIGHT": False}))
    bf = candidate(candidate_id="iv_r1a", parent_id="iv", lever="backfill_fields", generation=1,
                   expression="group_zscore(ts_mean(ts_backfill(iv_call, 63), 10), subindustry)")
    store(memory, bf, "repair", "sub_universe",
          [gate("sharpe", True, 2.0, 1.25), gate("sub_universe", False, 0.54, 0.88)],
          ResultRecord("iv_r1a", "complete", sharpe=2.0, checks={"CONCENTRATED_WEIGHT": True}))
    combo = candidate(candidate_id="iv_r1x", parent_id="iv", generation=1,
                      lever="combo:backfill_fields+rank_transform+liquidity_tilt",
                      expression="rank(group_zscore(ts_mean(ts_backfill(iv_call, 63), 10), "
                                 "subindustry)) * rank(adv20)")
    store(memory, combo, "accept", "none",
          [gate("sharpe", True, 1.38, 1.25), gate("sub_universe", True, 0.81, 0.60)],
          ResultRecord("iv_r1x", "complete", sharpe=1.38, checks={"CONCENTRATED_WEIGHT": True}))


def test_secondary_classes_and_combo_parts_get_credit(memory, candidate):
    iv_history(memory, candidate)
    events = repair_events(memory.candidate_outcomes())
    got = {(e.failure_class, e.lever, e.role): (e.reward, e.weight) for e in events}
    # distance to accept: parent 0.40/0.95 + 0.5 (concentration, unmeasured) = 0.921;
    # backfill child 0.34/0.88 = 0.386, so the overall part is (0.921 - 0.386) / 0.921
    gain = (0.4 / 0.95 + 0.5 - 0.34 / 0.88) / (0.4 / 0.95 + 0.5)
    # backfill cleared concentration as a side effect: class part 1, secondary weight
    assert got[("concentration", "backfill_fields", "secondary")] == \
        (pytest.approx(0.5 + 0.5 * gain, abs=1e-5), 0.5)
    assert got[("sub_universe", "backfill_fields", "target")] == \
        (pytest.approx(0.25 + 0.5 * gain, abs=1e-5), 1.0)
    # each part of the accepted combination is credited with a third of the weight
    assert got[("sub_universe", "liquidity_tilt", "part")] == (1.0, pytest.approx(1 / 3))
    assert got[("concentration", "rank_transform", "part")] == (1.0, pytest.approx(1 / 6))
    stats = LeverStats.from_events(events)
    assert stats.mean("sub_universe", "liquidity_tilt") > stats.mean("sub_universe", "never") == 0.5
    # only target events become prompt examples
    assert all(e["lever"].startswith("combo:") or e["lever"] == "backfill_fields"
               for e in successful_examples(events, "sub_universe", 5))


def test_a_combination_that_worked_becomes_a_lever(memory, candidate, playbook):
    iv_history(memory, candidate)
    stats = LeverStats.from_events(repair_events(memory.candidate_outcomes()))
    name = "combo:backfill_fields+rank_transform+liquidity_tilt"
    assert [lv.name for lv in stats.learned_combos("sub_universe")] == [name]
    ranked = rank_levers(playbook, "sub_universe", {"sub_universe", "concentration"}, stats,
                         RepairLearningConfig())
    assert ranked[0][0].name == name
    # the agent applies it to a new alpha, with each part's step taken from the playbook
    new = candidate(candidate_id="n1", expression="group_zscore(ts_mean(opt_x, 5), subindustry)")
    d = DecisionRecord("n1", "repair", "sub_universe",
                       gate_results=[gate("sub_universe", False, 0.3, 0.8)])
    out = RepairAgent(playbook, SPEC, memory).repair(new, d)
    assert out.lever == name
    assert out.child.expression == \
        "rank(group_zscore(ts_mean(ts_backfill(opt_x, 63), 5), subindustry)) * rank(adv20)"


def test_combination_skips_parts_that_do_not_apply(memory, candidate):
    pb = Playbook({"sub_universe": [Lever("combo:backfill_fields+liquidity_tilt",
                                          {"backfill_fields": {"lookback": 21}})],
                   "sharpe": [Lever("liquidity_tilt", {"field": "cap"})]})
    agent = RepairAgent(pb, SPEC, memory)
    d = DecisionRecord("c1", "repair", "sub_universe")
    out = agent.repair(candidate(expression="rank(opt_x)"), d)
    assert out.child.expression == "rank(ts_backfill(opt_x, 21)) * rank(cap)"
    out = agent.repair(candidate(expression="rank(close)"), d)   # nothing to backfill
    assert out.child.expression == "rank(close) * rank(cap)"
    assert any("skipped part backfill_fields" in n for n in out.notes)
    tilted = candidate(expression="rank(close) * rank(cap)")
    out = agent.repair(tilted, d)                                  # no part applies
    assert out.child is None and any("part liquidity_tilt" in n for n in out.notes)
