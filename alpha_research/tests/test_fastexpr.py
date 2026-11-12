import json

import pytest
import yaml

from alpha_research.config import ConfigError
from alpha_research.fastexpr import (ComplexityLimits, FastExprError, FastExprSpec, analyze,
                                     catalog, parse)

from conftest import CANDIDATES_FILE, ROOT

SPEC_FILE = ROOT / "config" / "fastexpr.yaml"


@pytest.fixture(scope="module")
def spec():
    return FastExprSpec.load(SPEC_FILE)


# ---------------------------------------------------------------- spec file

def test_shipped_spec_loads_and_derives_arity(spec):
    assert spec.operators["ts_mean"].min_args == 2 and spec.operators["ts_mean"].max_args == 2
    assert spec.operators["ts_rank"].min_args == 2 and spec.operators["ts_rank"].max_args == 3
    assert spec.operators["rank"].min_args == 1 and spec.operators["rank"].max_args == 2
    assert spec.operators["max"].min_args == 2 and spec.operators["max"].max_args is None
    assert "subindustry" in spec.groups and "close" in spec.base_fields


@pytest.mark.parametrize("bad", [
    "not a mapping",
    {"operators": ["ts_mean(x, d)"], "groups": [], "base_fields": []},  # constants missing
    {"operators": [], "groups": [], "base_fields": [], "constants": []},
    {"operators": ["ts_mean x d"], "groups": [], "base_fields": [], "constants": []},
    {"operators": ["f(x), g(y)"], "groups": [], "base_fields": [], "constants": []},
    {"operators": ["f(x)", "f(y)"], "groups": [], "base_fields": [], "constants": []},
    {"operators": ["f(a=1, b)"], "groups": [], "base_fields": [], "constants": []},
    {"operators": ["f(..., a)"], "groups": [], "base_fields": [], "constants": []},
    {"operators": ["f(x)"], "groups": "sector", "base_fields": [], "constants": []},
    {"operators": ["f(x)"], "groups": [1], "base_fields": [], "constants": []},
])
def test_malformed_spec_raises_config_error(bad):
    with pytest.raises(ConfigError):
        FastExprSpec.from_dict(bad)


def test_unreadable_spec_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        FastExprSpec.load(tmp_path / "missing.yaml")
    p = tmp_path / "bad.yaml"
    p.write_text("operators: [unclosed")
    with pytest.raises(ConfigError):
        FastExprSpec.load(p)


# ---------------------------------------------------------------- valid input

def test_every_valid_hand_written_expression_passes(spec):
    """The seed operator list must cover the project's own hand-written candidates."""
    entries = json.loads(CANDIDATES_FILE.read_text())
    skipped = {"dup_01", "inv_01"}  # upper-case operator names; unbalanced parentheses
    checked = 0
    for e in entries:
        if e["candidate_id"] in skipped:
            continue
        a = analyze(e["expression"], spec)
        assert a.ok, (e["candidate_id"], a.errors)
        checked += 1
    assert checked == len(entries) - 2


def test_simple_expression_details(spec):
    a = analyze("rank(-ts_delta(close, 5))", spec)
    assert a.ok and a.warnings == []
    assert a.operators_used == ("rank", "ts_delta")
    assert a.fields_used == ("close",)
    assert a.n_calls == 2


def test_keyword_arguments_and_bare_constants(spec):
    assert analyze("rank(ts_decay_linear(returns, 10, dense=false), rate=2)", spec).ok
    assert analyze("quantile(rank(close), driver=gaussian, sigma=1.0)", spec).ok
    assert analyze('ts_backfill(vwap, 60, ignore="NAN")', spec).ok


def test_multi_statement_with_variables(spec):
    expr = """
        mom = ts_delta(close, 20);
        vol = ts_std_dev(returns, 20);
        rank(mom / vol)
    """
    a = analyze(expr, spec)
    assert a.ok, a.errors
    assert a.fields_used == ("close", "returns")


def test_comparisons_and_logic_inside_trade_when(spec):
    a = analyze("trade_when(volume > adv20 && returns < 0, rank(-returns), -1)", spec)
    assert a.ok, a.errors


def test_scientific_and_leading_dot_numbers(spec):
    assert analyze("rank(close * 0.001 + .5)", spec).ok
    # BRAIN rejects scientific notation ("Unexpected character 'e'", run rs20261111162604)
    bad = analyze("rank(close / max(volume, 1e-08))", spec)
    assert not bad.ok and "plain decimal (0.00000001)" in " ".join(bad.errors)


def test_group_arguments_are_not_datafields(spec):
    a = analyze("group_neutralize(rank(close), subindustry)", spec, datafields=["close"])
    assert a.ok and a.fields_used == ("close",)


def test_variadic_operators(spec):
    assert analyze("max(close, open, vwap)", spec).ok
    assert not analyze("max(close)", spec).ok


def test_operators_from_brain_account_list(spec):
    # added after `cli brain operators` on our account (OQ-10)
    for expr in ["add(close, open, vwap)", "subtract(close, open, filter=true)",
                 "if_else(greater(close, open), 1, -1)", "and(less(close, open), not(is_nan(volume)))",
                 "kth_element(close, 5, 2)", "rank(close) * ts_step(1)", "ts_backfill(close, 20)",
                 "bucket(rank(cap), range=\"0.1, 1, 0.1\", skipBoth=False)"]:
        assert analyze(expr, spec).ok, expr


# not on our account at consultant level either (2026-11-10); ts_max, ts_min and
# vector_neut were in this list at user level and came with the upgrade
@pytest.mark.parametrize("op", ["ts_skewness", "exp", "regression_neut"])
def test_operators_not_on_brain_account_are_rejected(spec, op):
    assert op not in spec.operators


# ---------------------------------------------------------------- syntax errors

@pytest.mark.parametrize("expr, fragment", [
    ("rank(ts_delta(close, 5)", "expected ')'"),
    ("rank(close))", "unmatched ')'"),
    ("", "empty expression"),
    ("   ;  ", "empty expression"),
    ("rank(close) rank(open)", "expected ';'"),
    ("rank(close ? 1 : 2)", "unexpected character '?'"),
    ("rank(close[1])", "unexpected character '['"),
    ("rank(close,)", "unexpected ')'"),
    ("ts_mean(x=close, 5)", "positional argument after keyword"),
    ("rank(close) +", "end of expression"),
])
def test_syntax_errors_are_reported_not_raised(spec, expr, fragment):
    a = analyze(expr, spec)
    assert not a.ok
    assert fragment in a.errors[0]


def test_parse_raises_for_direct_callers():
    with pytest.raises(FastExprError):
        parse("rank(")
    with pytest.raises(FastExprError):
        parse(None)


def test_deeply_nested_input_does_not_crash(spec):
    a = analyze("rank(" * 5000 + "close" + ")" * 5000, spec)
    assert not a.ok and "nested too deeply" in a.errors[0]


# ---------------------------------------------------------------- semantic errors

def test_unknown_operator_with_suggestion(spec):
    a = analyze("rank(ts_deltaa(close, 5))", spec)
    assert any("unknown operator 'ts_deltaa'" in e and "ts_delta" in e for e in a.errors)


def test_uppercase_operator_hints_lowercase(spec):
    a = analyze("RANK(close)", spec)
    assert any("lowercase" in e and "rank" in e for e in a.errors)


@pytest.mark.parametrize("expr, fragment", [
    ("ts_mean(close)", "ts_mean takes 2 arguments, got 1"),
    ("ts_mean(close, 5, 3)", "ts_mean takes 2 arguments, got 3"),
    ("rank(close, 2, 3)", "rank takes 1 to 2 arguments, got 3"),
    ("max(close)", "max takes at least 2 arguments, got 1"),
])
def test_arity_errors(spec, expr, fragment):
    assert any(fragment in e for e in analyze(expr, spec).errors)


def test_last_statement_may_not_be_an_assignment(spec):
    a = analyze("x = rank(close)", spec)
    assert any("last statement" in e for e in a.errors)


def test_variable_used_before_definition_is_an_unknown_field(spec):
    a = analyze("rank(m); m = ts_delta(close, 5); rank(m)", spec, datafields=["close"])
    assert any("unknown datafield 'm'" in e for e in a.errors)


def test_variable_may_not_shadow_an_operator(spec):
    a = analyze("rank = close; rank(close)", spec)
    assert any("same name as an operator" in e for e in a.errors)


def test_constant_only_expression_is_rejected(spec):
    a = analyze("rank(5)", spec)
    assert any("no datafield" in e for e in a.errors)


# ---------------------------------------------------------------- datafield catalog

def test_catalog_makes_unknown_fields_errors(spec):
    a = analyze("rank(ts_mean(book_value, 20))", spec, datafields=["assets"])
    assert any("unknown datafield 'book_value'" in e for e in a.errors)


def test_catalog_accepts_listed_fields_and_base_fields(spec):
    a = analyze("rank(book_value / close)", spec, datafields=["book_value"])
    assert a.ok and a.warnings == []


def test_no_catalog_gives_a_warning_not_an_error(spec):
    a = analyze("rank(ts_mean(book_value, 20))", spec)
    assert a.ok
    assert a.warnings == ["datafield 'book_value' not verified (no datafield catalog supplied)"]


def test_catalog_suggests_close_matches(spec):
    a = analyze("rank(book_valu)", spec, datafields=["book_value"])
    assert any("book_value" in e for e in a.errors)


# BRAIN's errors in run rs20261104163055: these reached BRAIN and failed there
@pytest.mark.parametrize("expr, fragment", [
    ("group_rank(winsorize(close - open, 4), subindustry)",
     "winsorize: optional arguments must be written by name (std=...)"),
    ("rank(ts_backfill(vwap, 60, 1))", "ts_backfill: optional arguments must be written by name (k=...)"),
])
def test_optional_arguments_must_be_named(spec, expr, fragment):
    assert any(fragment in e for e in analyze(expr, spec).errors)
    named = expr.replace(", 4)", ", std=4)").replace("60, 1)", "60, k=1)")
    assert analyze(named, spec).ok, named


def test_event_fields_only_inside_vec_operators(spec):
    cat = catalog(["pe_news", "assets"], {"pe_news": {"type": "VECTOR"}, "assets": {"type": "MATRIX"}})
    assert cat == {"pe_news": "VECTOR", "assets": "MATRIX"}
    bad = analyze("rank(-ts_delta(pe_news, 30))", spec, datafields=cat)
    assert any("'pe_news' is an event (vector) field" in e for e in bad.errors)
    assert analyze("rank(-ts_delta(vec_avg(pe_news), 30))", spec, datafields=cat).ok
    assert analyze("x = vec_sum(pe_news); rank(x / assets)", spec, datafields=cat).ok
    # a catalog without types cannot check this, so it passes as before
    assert analyze("rank(-ts_delta(pe_news, 30))", spec, datafields=["pe_news"]).ok


# ---------------------------------------------------------------- complexity limits

def test_limits_none_means_not_configured(spec):
    deep = "rank(" * 30 + "close" + ")" * 30
    assert analyze(deep, spec, limits=ComplexityLimits()).ok


def test_limits_are_enforced_when_set(spec):
    expr = "rank(ts_mean(ts_delta(close, 5), 10))"
    assert analyze(expr, spec, limits=ComplexityLimits(max_depth=2)).errors
    assert analyze(expr, spec, limits=ComplexityLimits(max_depth=3)).ok
    assert analyze(expr, spec, limits=ComplexityLimits(max_calls=2)).errors
    assert analyze(expr, spec, limits=ComplexityLimits(max_calls=3)).ok
    assert analyze(expr, spec, limits=ComplexityLimits(max_chars=10)).errors
    assert analyze(expr, spec, limits=ComplexityLimits(max_chars=len(expr))).ok


def test_spec_round_trips_through_yaml_text():
    raw = yaml.safe_load(SPEC_FILE.read_text())
    again = FastExprSpec.from_dict(yaml.safe_load(yaml.safe_dump(raw)))
    assert set(again.operators) == set(FastExprSpec.from_dict(raw).operators)


# ------------------------------------------------------------------ render

@pytest.mark.parametrize("expr, text", [
    ("rank(-ts_delta(close,5))", "rank(-ts_delta(close, 5))"),
    ("a=close/open;rank(a-(b-c))*-1", "a = close / open; rank(a - (b - c)) * -1"),
    ("(a+b)*c", "(a + b) * c"),
    ("a-b-c", "a - b - c"),
    ("a/(b*c)", "a / (b * c)"),
    ("-(a+b)", "-(a + b)"),
    ("x||y&&z", "x || y && z"),
    ("(x||y)&&z", "(x || y) && z"),
    ("(a<b)+1", "(a < b) + 1"),
    ('ts_backfill(x,20,k=1,ignore="NAN")', 'ts_backfill(x, 20, k=1, ignore="NAN")'),
])
def test_render_round_trips(expr, text):
    from alpha_research.fastexpr import parse, render
    assert render(parse(expr)) == text
    assert parse(text) == parse(expr)


def test_render_round_trips_every_hand_written_expression():
    from alpha_research.fastexpr import parse, render
    for c in json.loads(CANDIDATES_FILE.read_text()):
        try:
            tree = parse(c["expression"])
        except FastExprError:
            continue
        assert parse(render(tree)) == tree
