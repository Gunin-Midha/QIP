from datetime import date

import pandas as pd
import pytest

from alpha_research.schemas import (
    CandidateRecord, DecisionRecord, GateResult, ResultRecord, SchemaError, YearRow,
    experiment_hash, normalize_expression,
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
