"""Lesson Writer (lessons.py): evidence from memory, rule and model lessons, storage, and
the agents reading the latest lessons."""

import json

import pytest

from alpha_research import cli
from alpha_research.config import ConfigError
from alpha_research.fastexpr import FastExprSpec
from alpha_research.lessons import (LessonsConfig, evidence, for_agent, learn, metrics,
                                    rule_lessons)
from alpha_research.llm import ScriptedLLMClient
from alpha_research.schemas import DecisionRecord, ResultRecord

from conftest import ROOT

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
META = {"anl_up": {"dataset": "analyst9", "category": "analyst"}}
CFG = LessonsConfig(min_evidence=3)


def result(cid, sharpe, sub, recent=None, prod=None, fitness=1.0):
    cv = {}
    if recent is not None:
        cv["LOW_2Y_SHARPE"] = {"value": recent, "limit": 1.58}
    if prod is not None:
        cv["PROD_CORRELATION"] = {"value": prod, "limit": 0.7}
    return ResultRecord(cid, "complete", sharpe=sharpe, fitness=fitness, turnover=0.2,
                        sub_universe_sharpe=sub, check_values=cv)


def put(memory, cand, decision, cls, res):
    memory.put_candidate(cand)
    memory.put_result(res)
    memory.put_decision(DecisionRecord(cand.candidate_id, decision, cls))


def tilt_history(memory, candidate, region="EUR"):
    """Three sub-universe failures each fixed by size_tilt, which also costs fitness."""
    for i in range(3):
        p = candidate(candidate_id=f"p{i}", region=region, universe="TOP2500",
                      neutralization="INDUSTRY", expression=f"rank(-ts_sum(returns, {5 + i}))")
        put(memory, p, "repair", "sub_universe", result(p.candidate_id, 2.5, 1.0, fitness=1.2))
        c = candidate(candidate_id=f"p{i}_r1", region=region, universe="TOP2500",
                      neutralization="INDUSTRY", parent_id=f"p{i}", generation=1,
                      lever="size_tilt",
                      expression=f"rank(-ts_sum(returns, {5 + i})) * (0.25 + rank(cap))")
        put(memory, c, "accept", "none", result(c.candidate_id, 2.2, 1.2, fitness=1.0))


def test_metrics_take_brain_check_values():
    m = metrics(result("x", 2.0, 1.0, recent=1.4, prod=0.74))
    assert m["sub_ratio"] == 0.5 and m["recent_sharpe"] == 1.4 and m["prod_correlation"] == 0.74
    assert metrics(ResultRecord("x", "failed")) is None
    assert metrics(result("x", -1.0, 0.5))["sub_ratio"] is None   # no ratio on a negative Sharpe


def test_change_effects_become_repair_lessons(memory, candidate):
    tilt_history(memory, candidate)
    ev = evidence(memory.candidate_outcomes(), SPEC, lambda r, d: META)
    rows = {(r["scope"], r["lever"]): r for r in ev["change_effects"]}
    row = rows[("all", "size_tilt")]
    assert (row["n"], row["fixed"], row["failure_class"]) == (3, 3, "sub_universe")
    assert row["median_change_sharpe"] == -0.3 and row["median_change_fitness"] == -0.2
    assert row["median_change_sub_ratio"] == pytest.approx(0.15, abs=0.01)
    assert ("EUR", "size_tilt") in rows
    texts = [x["text"] for x in rule_lessons(ev) if x["audience"] == "repair"]
    assert any(t.startswith("size_tilt on sub_universe failures (all regions, 3 repairs): "
                            "fixed the class in 3; median change Sharpe -0.3, fitness -0.2")
               for t in texts)


def test_groups_below_min_evidence_are_left_out(memory, candidate):
    tilt_history(memory, candidate)
    ev = evidence(memory.candidate_outcomes(), SPEC, lambda r, d: META, min_evidence=4)
    assert ev["change_effects"] == []            # 3 repairs per group
    assert [r["n"] for r in ev["failure_profile"]] == [6]   # parents and children together
    assert evidence(memory.candidate_outcomes(), SPEC, lambda r, d: META,
                    min_evidence=7)["failure_profile"] == []


def test_production_correlation_by_data_read(memory, candidate):
    for i in range(3):
        a = candidate(candidate_id=f"pv{i}", expression=f"rank(-ts_sum(returns, {5 + i}))")
        put(memory, a, "repair", "prod_correlation", result(a.candidate_id, 2.4, 1.3, prod=0.75))
        b = candidate(candidate_id=f"mix{i}", expression=f"rank(returns) + rank(anl_up) + {i}")
        put(memory, b, "accept", "none", result(b.candidate_id, 2.2, 1.2, prod=0.64))
    ev = evidence(memory.candidate_outcomes(), SPEC, lambda r, d: META)
    rows = {(r["scope"], r["data"]): r for r in ev["production_correlation_by_data"]}
    assert rows[("all", "pv")] == {"scope": "all", "data": "pv", "n": 3, "mean": 0.75,
                                   "above_limit": 3}
    assert rows[("all", "analyst+pv")]["above_limit"] == 0
    text = next(x["text"] for x in rule_lessons(ev) if x["scope"] == "all"
                and x["text"].startswith("Production correlation"))
    assert "analyst+pv 0.64 mean over 3 (0 above the limit); pv 0.75 mean over 3 (3 above" in text
    profile = {r["scope"]: r for r in ev["failure_profile"]}   # no pv1 meta: dataset name
    assert profile["USA/pv1"]["n"] == 6 and profile["USA/pv1"]["accepted"] == 3
    assert profile["USA/pv1"]["top_failures"] == {"prod_correlation": 3}


def test_model_lessons_are_checked_and_rules_are_the_fallback(memory, candidate):
    tilt_history(memory, candidate)
    outcomes = memory.candidate_outcomes()
    good = {"audience": "repair", "scope": "EUR",
            "text": "Try size_tilt first for sub-universe in EUR: fixed 3 of 3, fitness -0.2."}
    bad = [{"audience": "boss", "scope": "EUR", "text": "x 1"},
           {"audience": "generator", "scope": "MARS", "text": "fixed 3"},
           {"audience": "generator", "scope": "all", "text": "no numbers here"}]
    out = learn(outcomes, SPEC, lambda r, d: META, CFG,
                llm=ScriptedLLMClient([json.dumps({"lessons": [good, *bad]})]))
    assert out.source == "model" and out.lessons == [good] and out.llm_calls == 1
    out = learn(outcomes, SPEC, lambda r, d: META, CFG,
                llm=ScriptedLLMClient([json.dumps({"lessons": bad})]))
    assert out.source == "rules" and out.lessons and "passed the checks" in out.notes[0]
    out = learn(outcomes, SPEC, lambda r, d: META, LessonsConfig(use_model=False),
                llm=ScriptedLLMClient([]))
    assert out.source == "rules" and out.llm_calls == 0


def test_no_evidence_no_lessons(memory):
    out = learn(memory.candidate_outcomes(), SPEC, lambda r, d: {}, CFG)
    assert out.lessons == [] and "3 or more" in out.notes[0]


def test_memory_keeps_every_write_and_serves_the_latest(memory):
    memory.put_lessons("b1", [{"audience": "generator", "scope": "all", "text": "old 1"}], "rules")
    memory.put_lessons("b2", [{"audience": "generator", "scope": "all", "text": "new 2"},
                              {"audience": "repair", "scope": "EUR", "text": "new 3"}], "model",
                       {"change_effects": []})
    rows = memory.latest_lessons()
    assert [r["text"] for r in rows] == ["new 2", "new 3"] and rows[0]["source"] == "model"
    assert memory.conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] == 3
    stored = [r[0] for r in memory.conn.execute(
        "SELECT evidence_json FROM lessons WHERE batch_id = 'b2' ORDER BY id")]
    assert stored == ['{"change_effects": []}', "{}"]      # evidence once per write


def test_for_agent_picks_audience_and_scopes_general_first():
    rows = [{"audience": "generator", "scope": "EUR/pv", "text": "c"},
            {"audience": "generator", "scope": "all", "text": "a"},
            {"audience": "generator", "scope": "EUR", "text": "b"},
            {"audience": "generator", "scope": "USA", "text": "x"},
            {"audience": "repair", "scope": "all", "text": "r"}]
    assert for_agent(rows, "generator", "EUR", "pv") == ["a", "b", "c"]
    assert for_agent(rows, "repair", "EUR", "pv") == ["r"]


@pytest.mark.parametrize("raw, match", [
    ("yes", "must be a mapping"), ({"use_model": "yes"}, "use_model"),
    ({"min_evidence": 0}, "min_evidence"), ({"catalog_plan": 3}, "catalog_plan")])
def test_lessons_config_is_checked(raw, match):
    with pytest.raises(ConfigError, match=match):
        LessonsConfig.from_dict(raw)
    assert LessonsConfig.from_dict(None).after_every_batch is False


def test_research_run_writes_lessons_and_the_next_run_reads_them(memory, config, playbook,
                                                                  candidate):
    from test_research import make_run
    tilt_history(memory, candidate, region="USA")
    report = make_run(memory, config, playbook).run(1)
    assert report["lessons"]["written"] > 0 and report["lessons"]["source"] == "rules"
    written = memory.latest_lessons()
    assert written[0]["batch_id"] == "rs1"
    run = make_run(memory, config, playbook)
    assert run.lessons == written
    key = next(iter(run.arms))
    texts = for_agent(written, "generator", key[1], "pv")
    assert run._generator(key).learned == texts
    assert run.repairer.lessons == written


def test_cli_run_writes_rule_lessons_unless_told_not_to(tmp_path, capsys):
    args = ["run", "--candidates", str(ROOT / "tests/fixtures/candidates/hand_written.json"),
            "--config", str(ROOT / "tests/fixtures/thresholds_test.yaml"),
            "--fixtures-dir", str(ROOT / "tests/fixtures/brain_results"),
            "--playbook", str(ROOT / "config/repair_playbook.yaml"),
            "--agents", str(ROOT / "config/agents.yaml"),
            "--fastexpr", str(ROOT / "config/fastexpr.yaml")]
    assert cli.main([*args, "--data-dir", str(tmp_path / "a"), "--no-lessons"]) == 0
    assert "lessons:" not in capsys.readouterr().out
    assert cli.main([*args, "--data-dir", str(tmp_path / "b")]) == 0
    assert "lessons: " in capsys.readouterr().out
    assert cli.main(["lessons", "--data-dir", str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("lessons of batch") or out.startswith("no lessons")


def test_hand_variants_go_to_the_model_as_described_changes(memory, candidate):
    p = candidate(candidate_id="h0")
    put(memory, p, "repair", "prod_correlation", result("h0", 2.3, 1.2, prod=0.74))
    c = candidate(candidate_id="h1", parent_id="h0", generation=1,
                  expression="rank(returns) + rank(anl_up)",
                  hypothesis="Blend with analyst revisions at half weight.")
    put(memory, c, "accept", "none", result("h1", 2.2, 1.2, prod=0.65))
    ev = evidence(memory.candidate_outcomes(), SPEC, lambda r, d: META)
    assert ev["change_effects"] == []            # no lever: the rules cannot group it
    d = ev["described_changes"][0]
    assert d["author_note_unverified"] == "Blend with analyst revisions at half weight."
    assert d["fixed"] and d["child_outcome"] == "accept / none" and d["child_prod_correlation"] == 0.65
    assert d["measured_change_prod_correlation"] == -0.09 and d["parent_failure"] == "prod_correlation"


def test_an_unmeasured_production_correlation_is_not_a_fix(memory, candidate):
    p = candidate(candidate_id="u0")
    put(memory, p, "repair", "prod_correlation", result("u0", 2.3, 1.2, prod=0.74))
    c = candidate(candidate_id="u1", parent_id="u0", generation=1, expression="rank(open)",
                  hypothesis="Slower horizon.")
    put(memory, c, "repair", "sub_universe", result("u1", 2.2, 0.9))   # PROD not computed
    d = evidence(memory.candidate_outcomes(), SPEC, lambda r, d: META)["described_changes"][0]
    assert d["fixed"] is False


def test_model_failure_keeps_rule_lessons_and_evidence_is_trimmed(memory, candidate):
    from alpha_research.lessons import trimmed
    from alpha_research.llm import LLMUnavailable

    class TooLarge:
        model = "big"

        def complete(self, system, user, *, max_output=4096):
            raise LLMUnavailable("model call failed: HTTP 413: Request too large")

    tilt_history(memory, candidate)
    out = learn(memory.candidate_outcomes(), SPEC, lambda r, d: META, CFG, llm=TooLarge())
    assert out.source == "rules" and out.lessons and "unavailable" in out.notes[0]
    ev = {"change_effects": [{"scope": "all", "x": 1}, {"scope": "EUR", "x": "y" * 50}],
          "failure_profile": [{"scope": "EUR/pv", "n": 9}, {"scope": "USA/pv", "n": 3}],
          "described_changes": [{"change": "z" * 100}] * 5}
    small = trimmed(ev, 150)
    assert small["described_changes"] == [] and small["change_effects"][0]["scope"] == "all"
    assert len(ev["described_changes"]) == 5           # the original is untouched


def test_model_path_with_described_changes(memory, candidate):
    tilt_history(memory, candidate)
    p = candidate(candidate_id="h0", region="CHN")
    put(memory, p, "repair", "prod_correlation", result("h0", 2.3, 1.2, prod=0.74))
    c = candidate(candidate_id="h1", region="CHN", parent_id="h0", generation=1,
                  expression="rank(returns) + rank(anl_up)", hypothesis="Blend with analysts.")
    put(memory, c, "accept", "none", result("h1", 2.2, 1.2, prod=0.65))
    lesson = {"audience": "generator", "scope": "CHN",
              "text": "Blending with analyst data cut production correlation by 0.09 once."}
    out = learn(memory.candidate_outcomes(), SPEC, lambda r, d: META, CFG,
                llm=ScriptedLLMClient([json.dumps({"lessons": [lesson]})]))
    assert out.source == "model" and out.lessons == [lesson]   # CHN known from the region


def test_too_large_request_halves_the_evidence_and_asks_again(memory, candidate, monkeypatch):
    # run rs20261126082238: the evidence did not fit any model's per-minute limit
    from alpha_research import lessons as lessons_mod
    from alpha_research.llm import RequestTooLarge

    monkeypatch.setattr(lessons_mod, "trimmed", lambda ev, chars: {"evidence": "y" * chars})

    class Fits:
        model = "groq:fake/model"

        def __init__(self, limit):
            self.limit, self.sizes = limit, []

        def complete(self, system, user, *, max_output=4096):
            self.sizes.append(len(user))
            if len(user) > self.limit:
                raise RequestTooLarge(self.model, "too big")
            return json.dumps({"lessons": [{"audience": "repair", "scope": "EUR",
                                            "text": "size_tilt fixed 3 of 3 in EUR."}]})

    tilt_history(memory, candidate)
    outcomes = memory.candidate_outcomes()
    cfg = LessonsConfig(max_prompt_chars=12000)
    llm = Fits(limit=6000 + 3000)    # fits once the evidence is cut to 6,000 characters
    out = learn(outcomes, SPEC, lambda r, d: META, cfg, llm=llm)
    assert out.source == "model" and len(llm.sizes) == 2 and llm.sizes[1] < llm.sizes[0]
    assert any("evidence cut to 6000" in n for n in out.notes)
    # never fits: the rules stand (stops at lessons.MIN_PROMPT_CHARS)
    small = Fits(limit=10)
    out = learn(outcomes, SPEC, lambda r, d: META, cfg, llm=small)
    assert out.source == "rules" and "unavailable" in out.notes[-1] and len(small.sizes) == 3
