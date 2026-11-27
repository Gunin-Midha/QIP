"""Category codes (owner 2026-11-27): BRAIN fields whose numbers name a class (news-topic
codes, return-based cluster labels, currency or fiscal-period codes) are not amounts. Run
rs20261127123332 ranked `news_category_tag` and a cluster label as signals. They are not
listed as signals, a few are offered as groupings, and an alpha that ranks or averages one is
sent back. Quantile buckets stay signals (owner: keep that exception)."""

import pytest

from alpha_research.generator import (AlphaGenerator, Source, build_system_prompt,
                                      category_fields, category_misuse, field_role,
                                      select_datafields, signal_profile)
from alpha_research.llm import ScriptedLLMClient

from test_generator import HYP, alphas, cfg, settings, spec  # noqa: F401 (fixtures)

META = {
    # descriptions as BRAIN's catalogs give them (2026-11-27)
    "news_category_tag": {"dataset": "news29", "type": "VECTOR",
                          "description": "Integer code representing a classification tag for "
                                         "the news headline (e.g., M&A, earnings)"},
    "grp_cluster": {"dataset": "news29", "type": "MATRIX",
                    "description": "Cluster label assignment (1-2) for each stock and date"},
    "ind_code": {"dataset": "news29", "type": "MATRIX",
                 "description": "Numeric code identifying the Level 4 industry classification"},
    "tm_q2": {"dataset": "news29", "type": "MATRIX",
              "description": "Assignment of each stock-date to one of 2 quantile buckets (0 or "
                             "1) representing market-neutral 10-day returns"},
    "q_label": {"dataset": "news29", "type": "MATRIX",
                "description": "Integer label of the return quintile the stock falls in"},
    "parent_flag": {"dataset": "news29", "type": "MATRIX",
                    "description": "Indicator if the entity is a parent company where 1 "
                                   "indicates parent and 0 otherwise"},
    "nws_sent": {"dataset": "news29", "type": "VECTOR",
                 "description": "Sentiment score of the news story between -1 and 1"},
    "anl_rev": {"dataset": "anl69", "type": "MATRIX",
                "description": "Number of upward EPS revisions in the last 4 weeks"},
}


def test_codes_and_cluster_labels_are_categories_quantiles_and_flags_are_not(cfg):
    roles = {f: field_role(f, META, cfg) for f in META}
    assert roles == {"news_category_tag": "category", "grp_cluster": "category",
                     "ind_code": "category", "tm_q2": "signal", "q_label": "signal",
                     "parent_flag": "signal", "nws_sent": "signal", "anl_rev": "signal"}


CATS = ["news_category_tag", "grp_cluster", "ind_code"]


@pytest.mark.parametrize("expr", [
    "group_rank(anl_rev, densify(grp_cluster))",
    "group_neutralize(rank(anl_rev), grp_cluster)",
    "g = densify(ind_code); group_zscore(anl_rev, g)",
    "if_else(vec_avg(nws_sent) > 0, anl_rev, 0) * (ind_code != 45)",
    "rank(anl_rev) * (is_nan(grp_cluster) == 0)",
])
def test_codes_as_groups_or_code_tests_are_fine(spec, expr):
    assert category_misuse(expr, CATS, spec) == []


@pytest.mark.parametrize("expr, field", [
    ("rank(ts_zscore(vec_avg(news_category_tag), 21)) + 0.5 * rank(anl_rev)",
     "news_category_tag"),
    ("rank(add(rank(ts_backfill(grp_cluster, 20)), rank(anl_rev)))", "grp_cluster"),
    ("g = densify(ind_code); rank(g) + rank(anl_rev)", "ind_code"),
    ("group_rank(ind_code, densify(grp_cluster))", "ind_code"),   # code in the value slot
])
def test_codes_used_as_amounts_are_errors(spec, expr, field):
    errs = category_misuse(expr, CATS, spec)
    assert len(errs) == 1 and errs[0].startswith(repr(field))


def test_codes_are_not_listed_as_signals_but_offered_as_groupings(spec, cfg):
    text = "news category tags and revisions"
    rel, _ = select_datafields(list(META), text, 8, 0.0, META, cfg)
    assert not set(rel) & set(CATS) and "tm_q2" in rel
    # named by the task: listed, with the category hint
    rel, _ = select_datafields(list(META), "use news_category_tag", 8, 0.0, META, cfg)
    assert rel[0] == "news_category_tag"
    # groupings: matrix codes only (a vector field cannot be a group), at most category_groups
    groups = category_fields(list(META), text, META, cfg, "news29")
    assert set(groups) == {"grp_cluster", "ind_code"}
    prompt = build_system_prompt(spec, list(META), text, cfg, meta=META)
    line = next(l for l in prompt.splitlines() if l.startswith("Category codes"))
    assert "grp_cluster" in line and "news_category_tag" not in line
    named = build_system_prompt(spec, list(META), "use news_category_tag", cfg, meta=META)
    assert "category code: no order" in named


def test_a_code_is_not_a_signal_source(spec, cfg):
    p = signal_profile(["anl_rev", "grp_cluster"], spec.base_fields, META, cfg)
    assert p["fields"] == ["anl_rev"]


def test_an_alpha_ranking_a_code_is_sent_back(spec, settings, cfg):
    bad = "rank(ts_zscore(vec_avg(news_category_tag), 21)) + 0.5 * rank(anl_rev)"
    good = "group_rank(anl_rev, densify(grp_cluster))"
    llm = ScriptedLLMClient([alphas(bad), alphas(good)])
    gen = AlphaGenerator(llm, spec, settings, cfg, datafields=list(META), field_meta=META)
    res = gen.generate(Source("hypothesis", HYP), 1, batch_id="b1")
    assert [c.expression for c in res.candidates] == [good]
    assert "'news_category_tag' is a category code" in llm.calls[1]["user"]
