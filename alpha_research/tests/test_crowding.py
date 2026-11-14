"""Crowded fields (2026-11-14, OQ-21): the Generator lists a dataset's most used fields last
(A), and fields of our alphas that failed BRAIN's PROD_CORRELATION count as crowded in that
market (C)."""

from dataclasses import replace

import pytest
import yaml

from alpha_research.config import ConfigError
from alpha_research.fastexpr import FastExprSpec
from alpha_research.generator import (_annotate, _rank_datafields, build_system_prompt,
                                      crowded_fields, load_generator_config, select_datafields)
from alpha_research.schemas import DecisionRecord, ResultRecord
from alpha_research.strategy import prod_correlation_fields

from conftest import ROOT
from test_research import make_run

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
GEN = load_generator_config(ROOT / "config" / "generator.yaml")


def _meta(counts: dict[str, int], dataset: str = "anl") -> dict:
    return {f: {"dataset": dataset, "alpha_count": n, "coverage": 1.0} for f, n in counts.items()}


def test_crowded_is_the_top_of_each_dataset_above_a_floor():
    meta = {**_meta({f"anl_f{i}": i for i in range(10)} | {"anl_hot": 800}),
            **_meta({"opt_a": 1, "opt_b": 2}, "opt")}
    assert crowded_fields(meta, 0.9, 50) == {"anl_hot"}       # top 10% of anl, >= 50 alphas
    # nearest rank: the top 2 of anl's 11 fields, the top 1 of opt's 2; each dataset alone
    assert crowded_fields(meta, 0.9, 0) == {"anl_hot", "anl_f9", "opt_b"}
    assert crowded_fields(meta, None, 50) == set()             # off
    assert crowded_fields({"x": {"dataset": "d"}}, 0.9, 0) == set()  # no counts, no flag
    # groups and base fields are in nearly every alpha: neither counted nor flagged
    meta["sector"] = {"dataset": "anl", "alpha_count": 9000, "type": "GROUP"}
    meta["close"] = {"dataset": "anl", "alpha_count": 9000}
    assert crowded_fields(meta, 0.9, 50, exclude={"close"}) == {"anl_hot"}


def test_crowded_fields_rank_last_and_are_not_explored():
    meta = {**_meta({"book_hot": 900, "book_cold": 3, "anl_x": 1}),
            **_meta({"opt_hot": 900, "opt_a": 0}, "opt")}
    assert _rank_datafields(list(meta), "book", 5, meta, {"book_hot"})[-1] == "book_hot"
    rel, exp = select_datafields(list(meta), "book", 4, 0.5, meta, crowded={"book_hot", "opt_hot"})
    assert "book_hot" not in rel and "opt_hot" not in exp
    # a field the task names stays first, crowded or not
    rel, _ = select_datafields(list(meta), "use book_hot", 2, 0.0, meta, crowded={"book_hot"})
    assert rel[0] == "book_hot"


def test_prompt_marks_crowded_and_production_failed_fields():
    catalog = ["anl_hot", "anl_cold", "anl_mid"]
    meta = _meta({"anl_hot": 900, "anl_cold": 2, "anl_mid": 5})
    cfg = replace(GEN, crowded_percentile=0.9, crowded_min_alphas=50)
    prompt = build_system_prompt(SPEC, catalog, "anl_hot breadth", cfg, meta=meta,
                                 prod_failed={"anl_mid": 2, "anl_cold": 0})
    assert "anl_hot [cov 100%, 900 alphas, crowded]" in prompt          # named, still shown
    assert "anl_mid [cov 100%, 5 alphas, crowded, 2 of ours failed production correlation]" \
        in prompt
    assert "anl_cold [cov 100%, 2 alphas]" in prompt                    # 0 failures: no flag
    assert "do not write its common form" in prompt
    # nothing crowded listed: no extra paragraph
    plain = build_system_prompt(SPEC, ["anl_cold"], "x", cfg, meta=meta)
    assert "crowded =" not in plain
    assert _annotate("anl_cold", meta) == "anl_cold [cov 100%, 2 alphas]"


def test_crowding_config(tmp_path):
    assert (GEN.crowded_percentile, GEN.crowded_min_alphas) == (0.9, 50)
    raw = yaml.safe_load((ROOT / "config" / "generator.yaml").read_text())
    for bad in ({"percentile": 1.5}, {"percentile": 0.9, "min_alphas": -1}, {"cap": 1}, 3):
        raw["generator"]["crowding"] = bad
        p = tmp_path / "g.yaml"
        p.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError):
            load_generator_config(p)
    raw["generator"]["crowding"] = {"percentile": None}
    p.write_text(yaml.safe_dump(raw))
    assert load_generator_config(p).crowded_percentile is None


def _row(candidate, cid, region, expr, prod):
    checks = {"LOW_SHARPE": True} if prod is None else {"PROD_CORRELATION": prod}
    return {"candidate": candidate(candidate_id=cid, region=region, expression=expr),
            "decision": None, "result": ResultRecord(cid, "complete", sharpe=2.0, checks=checks)}


def test_prod_correlation_fields_from_memory(candidate):
    rows = [_row(candidate, "a", "USA", "group_rank(est_up / est_n, sector) * rank(adv20)", False),
            _row(candidate, "b", "USA", "rank(est_up)", False),
            _row(candidate, "c", "USA", "rank(other_f)", True),      # passed
            _row(candidate, "d", "USA", "rank(third_f)", None),      # not checked
            _row(candidate, "e", "EUR", "rank(est_up)", False)]      # other market
    rows.append({"candidate": candidate(candidate_id="f", expression="rank(x_f)"),
                 "decision": None, "result": None})                  # not simulated
    assert prod_correlation_fields(rows, "USA", SPEC) == {"est_up": 2, "est_n": 1}
    assert prod_correlation_fields(rows, "JPN", SPEC) == {}


def test_research_run_gives_each_generator_its_markets_failures(memory, config, playbook,
                                                                candidate):
    c = candidate(candidate_id="old", expression="rank(est_up)")
    memory.put_candidate(c)
    memory.put_result(ResultRecord("old", "complete", sharpe=2.1,
                                   checks={"PROD_CORRELATION": False}))
    memory.put_decision(DecisionRecord("old", "repair", "prod_correlation"))
    run = make_run(memory, config, playbook)
    gen = run._generator(next(iter(run.arms)))
    assert gen.prod_failed == {"est_up": 1}


def test_plain_words_for_base_fields_and_groups_are_not_named_fields():
    catalog = ["high", "returns", "sector", "anl_up"]
    meta = {**_meta({"anl_up": 3, "high": 9000, "returns": 9000}),
            "sector": {"dataset": "anl", "type": "GROUP", "alpha_count": 9000}}
    text = "stocks priced high relative to returns, ranked within sector; anl_up rising"
    prompt = build_system_prompt(SPEC, catalog, text, GEN, meta=meta)
    named = next(line for line in prompt.splitlines() if line.startswith("Fields named"))
    assert "anl_up" in named and "high" not in named and "sector" not in named


# ------------------------------------------------------------------ ranker (gen-v6)

def test_ranker_matches_whole_words_in_ids_and_descriptions():
    meta = {"pre_news_high_price": {"dataset": "opt", "description": "High price before news"},
            "opt_iv_cp": {"dataset": "opt",
                          "description": "Implied volatility of calls minus puts"}}
    # "spread" contains "pre", "priced" contains "price": substrings must not count
    text = "the implied volatility spread between calls and puts, priced by informed traders"
    assert _rank_datafields(list(meta), text, 2, meta)[0] == "opt_iv_cp"
    # stems: "raised" in the text matches the id part "raisednum"
    meta2 = {"est_sal_raisednum": {"dataset": "anl"}, "est_sal_num": {"dataset": "anl"}}
    assert _rank_datafields(list(meta2), "analysts raised sales forecasts", 1, meta2) == \
        ["est_sal_raisednum"]


def test_ranker_weights_rare_words_above_common_ones():
    meta = {f"f_q{i}": {"dataset": "fnd", "description": "Quarterly balance item"}
            for i in range(20)}
    meta["f_acc"] = {"dataset": "fnd", "description": "Total accruals"}
    meta["f_both"] = {"dataset": "fnd", "description": "Quarterly balance sheet total"}
    text = "low accruals in quarterly balance data"
    # one rare word ("accruals") outweighs two words every field shares
    assert _rank_datafields(list(meta), text, 1, meta) == ["f_acc"]


def test_ranker_lists_the_arms_dataset_first():
    meta = {"news_sentiment": {"dataset": "news", "description": "analyst sentiment"},
            "mdl_quality": {"dataset": "mdl", "description": "quality component"}}
    text = "analyst sentiment"
    assert _rank_datafields(list(meta), text, 2, meta) == ["news_sentiment", "mdl_quality"]
    assert _rank_datafields(list(meta), text, 2, meta, dataset="mdl") == \
        ["mdl_quality", "news_sentiment"]
    # the Generator passes its arm's dataset through to the prompt's listing
    prompt = build_system_prompt(SPEC, list(meta), text, GEN, meta=meta, dataset="mdl")
    listed = prompt.split("Most relevant to this task:")[1]
    assert listed.index("mdl_quality") < listed.index("news_sentiment")
