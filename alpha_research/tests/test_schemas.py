import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from alpha_research.schemas import (
    REGION_SETTINGS, CandidateRecord, DecisionRecord, GateResult, ResultRecord, SchemaError,
    YearRow, experiment_hash, normalize_expression,
)


def test_candidate_from_dict_parses_and_strips(candidate):
    c = candidate(expression="  rank(close)  ")
    assert c.expression == "rank(close)"
    assert c.start_date == date(2019, 1, 1)
    c.validate()


@pytest.mark.parametrize("missing", ["candidate_id", "expression", "region", "delay", "start_date"])
def test_candidate_missing_required_field(candidate, missing):
    d = candidate().to_dict()
    del d[missing]
    with pytest.raises(SchemaError):
        CandidateRecord.from_dict(d)


def test_candidate_wrong_type_is_structural_error(candidate):
    d = candidate().to_dict()
    d["delay"] = "1"
    with pytest.raises(SchemaError):
        CandidateRecord.from_dict(d)


@pytest.mark.parametrize("field,value", [
    ("region", "XYZ"), ("universe", "TOP9"), ("delay", 2), ("decay", -1),
    ("neutralization", "COUNTRY"), ("truncation", 0.0), ("truncation", 1.5),
    ("end_date", "2018-01-01"), ("expression", "   "),
])
def test_candidate_domain_rules(candidate, field, value):
    c = candidate(**{field: value})  # structurally fine
    with pytest.raises(SchemaError):
        c.validate()


@pytest.mark.parametrize("universe", ["TOP2000", "TOPSP500"])
def test_universes_from_brain_options_are_valid(candidate, universe):
    candidate(universe=universe).validate()


def test_region_settings_match_brain_options():
    # the consultant account's OPTIONS /simulations response (cli brain options, 2026-11-10)
    opts = json.loads((Path(__file__).parent / "fixtures" / "brain_api_capture_consultant"
                       / "simulation_options.json").read_text())["common"]
    eq = lambda key: opts[key]["choices"]["instrumentType"]["EQUITY"]
    regions = [c["value"] for c in eq("region")]
    assert set(regions) == set(REGION_SETTINGS)
    for r in regions:
        s = REGION_SETTINGS[r]
        assert [c["value"] for c in eq("universe")["region"][r]] == list(s["universes"])
        assert sorted(c["value"] for c in eq("delay")["region"][r]) == list(s["delays"])
        assert {c["value"] for c in eq("neutralization")["region"][r]} == set(s["neutralizations"])


@pytest.mark.parametrize("region,universe,delay,neutralization,ok", [
    ("EUR", "TOP2500", 0, "COUNTRY", True),
    ("GLB", "TOPDIV3000", 1, "STATISTICAL", True),
    ("USA", "TOP3000", 1, "SLOW_AND_FAST", True),
    ("USA", "TOP3000", 1, "COUNTRY", False),      # COUNTRY only in multi-country regions
    ("GLB", "TOP3000", 0, "MARKET", False),       # GLB has delay 1 only
    ("EUR", "TOP3000", 1, "MARKET", False),       # TOP3000 is not an EUR universe
    ("AMR", "TOP600", 1, "CROWDING", False),      # AMR has no factor neutralizations
])
def test_settings_are_checked_per_region(candidate, region, universe, delay, neutralization, ok):
    c = candidate(region=region, universe=universe, delay=delay, neutralization=neutralization)
    if ok:
        c.validate()
    else:
        with pytest.raises(SchemaError):
            c.validate()


def test_candidate_roundtrip(candidate):
    c = candidate(parent_id="p", generation=2, lever="increase_decay")
    assert CandidateRecord.from_dict(c.to_dict()) == c


def test_result_roundtrip_keeps_none_and_series():
    s = pd.Series([1.0, -0.5], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    r = ResultRecord(candidate_id="c1", status="complete", sharpe=1.2, turnover=None,
                     yearly=[YearRow(2020, sharpe=1.0)], daily_pnl=s, checks={"LOW_SHARPE": True})
    back = ResultRecord.from_dict(r.to_dict())
    assert back.turnover is None
    assert back.sharpe == 1.2
    assert back.yearly[0].year == 2020
    pd.testing.assert_series_equal(back.daily_pnl, s, check_freq=False, check_names=False)


def test_result_bad_status():
    with pytest.raises(SchemaError):
        ResultRecord(candidate_id="c1", status="done").validate()


def test_decision_validation_and_roundtrip():
    d = DecisionRecord("c1", "accept", "none", [GateResult("sharpe", True, 1.5, 1.25, "")])
    d.validate()
    assert DecisionRecord.from_dict(d.to_dict()).decision == "accept"
    with pytest.raises(SchemaError):
        DecisionRecord("c1", "maybe", "none").validate()
    with pytest.raises(SchemaError):
        DecisionRecord("c1", "reject", "bogus").validate()


def test_normalize_expression_lowercases_operators_only():
    assert normalize_expression(" RANK( -Ts_Delta(Close, 5) ) ") == "rank(-ts_delta(Close,5))"


def test_hash_depends_on_settings(candidate):
    assert experiment_hash(candidate()) != experiment_hash(candidate(decay=5))
