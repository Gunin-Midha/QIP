import numpy as np
import pandas as pd
import pytest

from alpha_research.config import NOT_CONFIGURED
from alpha_research.risk import (
    assess, drawdown_timing, leg_balance, top_pnl_removal, weak_year,
)
from alpha_research.schemas import ResultRecord, YearRow
from datetime import date


def all_null(make_config):
    return make_config(risk__max_drawdown=None, risk__weight_concentration_max=None)


def test_leg_balance_vetoes(make_config):
    cfg = make_config(risk__leg_balance_min_ratio=0.3)
    r = ResultRecord("c", "complete", long_pnl=10.0, short_pnl=0.5)
    g = leg_balance(r, cfg)
    assert g.passed is False and g.value == pytest.approx(0.05)
    veto, _, reasons = assess(r, cfg)
    assert veto and any("leg_balance failed" in x for x in reasons)


def test_leg_balance_needs_both_legs(make_config):
    cfg = make_config(risk__leg_balance_min_ratio=0.3)
    assert leg_balance(ResultRecord("c", "complete", long_pnl=10.0), cfg).passed is None


def test_top_pnl_removal_fails_when_few_days_carry_profit(make_config):
    cfg = make_config(risk__min_sharpe_after_top_pnl_removal=0.5)
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, 1000)
    pnl[rng.choice(1000, 10, replace=False)] = 60.0      # ten huge days
    s = pd.Series(pnl, index=pd.bdate_range("2019-01-01", periods=1000))
    full = s.mean() / s.std() * np.sqrt(252)
    assert full > 1.0
    g = top_pnl_removal(ResultRecord("c", "complete", daily_pnl=s), cfg)
    assert g.passed is False and g.value < 0.5


def test_top_pnl_removal_passes_for_steady_pnl(make_config):
    cfg = make_config(risk__min_sharpe_after_top_pnl_removal=0.5)
    s = pd.Series(np.random.default_rng(1).normal(0.15, 1, 1000),
                  index=pd.bdate_range("2019-01-01", periods=1000))
    assert top_pnl_removal(ResultRecord("c", "complete", daily_pnl=s), cfg).passed is True


def test_all_null_thresholds_no_veto(make_config):
    cfg = all_null(make_config)
    s = pd.Series(np.random.default_rng(1).normal(0.1, 1, 300),
                  index=pd.bdate_range("2019-01-01", periods=300))
    r = ResultRecord("c", "complete", max_drawdown=0.5, weight_concentration=0.9,
                     long_pnl=10.0, short_pnl=0.1, daily_pnl=s,
                     yearly=[YearRow(2020, sharpe=-3.0), YearRow(2021, sharpe=4.0)])
    veto, checks, reasons = assess(r, cfg)
    assert veto is False
    assert all(c.passed is None for c in checks)
    for name in ("drawdown_depth", "leg_balance", "top_pnl_removal", "weight_concentration", "weak_year"):
        assert any(x.startswith(f"{name} not evaluated") and NOT_CONFIGURED in x for x in reasons), name


def test_drawdown_depth_and_concentration(config):
    veto, checks, _ = assess(ResultRecord("c", "complete", max_drawdown=0.3,
                                          weight_concentration=0.02), config)
    by = {c.name: c for c in checks}
    assert veto and by["drawdown_depth"].passed is False
    assert by["weight_concentration"].passed is True


def test_drawdown_timing_is_informational(make_config):
    cfg = make_config(risk__stress_windows=[["2020-02-15", "2020-04-30"]])
    g = drawdown_timing(ResultRecord("c", "complete", drawdown_date=date(2020, 3, 16)), cfg)
    assert g.passed is None and "stress window" in g.note and "outside" not in g.note


def test_weak_year_halves(make_config):
    rows = [YearRow(2019, sharpe=1.4), YearRow(2020, sharpe=0.11), YearRow(2022, sharpe=3.59)]
    r = ResultRecord("c", "complete", yearly=rows)
    assert weak_year(r, make_config(risk__weak_year_sharpe_floor=0.5)).passed is False
    assert weak_year(r, make_config(risk__weak_year_sharpe_floor=0.0)).passed is True
    assert weak_year(r, make_config(risk__max_yearly_sharpe_spread=2.0)).passed is False
    assert weak_year(ResultRecord("c", "complete"), make_config(risk__weak_year_sharpe_floor=0.5)).passed is None


def test_sub_universe_is_referenced_not_rechecked(config):
    from alpha_research.schemas import GateResult
    veto, checks, reasons = assess(ResultRecord("c", "complete"), config,
                                   [GateResult("sub_universe", False, 0.5, 0.73)])
    assert "sub_universe" not in {c.name for c in checks}
    assert any(r.startswith("sub_universe: see evaluator gate") for r in reasons)
    assert veto is False
