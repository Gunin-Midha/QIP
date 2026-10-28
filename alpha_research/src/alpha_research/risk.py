"""Risk manager (spec section 8).

Each check returns a GateResult. ``assess`` collects them and returns
``(veto, checks, reasons)``. Veto is True if any check failed. Checks that return None
never veto; they are listed in reasons as "not evaluated".
"""

from __future__ import annotations

import math

import pandas as pd

from .config import NOT_CONFIGURED, Config
from .schemas import GateResult, ResultRecord

TRADING_DAYS = 252


def annualized_sharpe(pnl: pd.Series) -> float | None:
    pnl = pnl.dropna()
    if len(pnl) < 2:
        return None
    std = float(pnl.std())
    if std == 0 or math.isnan(std):
        return None
    return float(pnl.mean()) / std * math.sqrt(TRADING_DAYS)


def drawdown_depth(result: ResultRecord, config: Config) -> GateResult:
    limit = config.risk.max_drawdown
    if limit is None:
        return GateResult("drawdown_depth", None, result.max_drawdown, None, NOT_CONFIGURED)
    if result.max_drawdown is None:
        return GateResult("drawdown_depth", None, None, limit, "max_drawdown missing")
    return GateResult("drawdown_depth", result.max_drawdown <= limit, result.max_drawdown, limit,
                      f"<= {limit}")


def drawdown_timing(result: ResultRecord, config: Config) -> GateResult:
    """Informational only (passed is always None)."""
    d = result.drawdown_date
    if d is None:
        return GateResult("drawdown_timing", None, None, None, "informational: drawdown_date missing")
    hits = [f"{s}..{e}" for s, e in config.risk.stress_windows if s <= d <= e]
    note = (f"informational: drawdown in stress window {hits}" if hits
            else "informational: drawdown outside configured stress windows")
    return GateResult("drawdown_timing", None, d, None, note)


def leg_balance(result: ResultRecord, config: Config) -> GateResult:
    floor = config.risk.leg_balance_min_ratio
    lp, sp = result.long_pnl, result.short_pnl
    if lp is None or sp is None:
        return GateResult("leg_balance", None, None, floor, "long_pnl or short_pnl missing")
    big = max(abs(lp), abs(sp))
    if big == 0:
        return GateResult("leg_balance", None, None, floor, "both legs have zero PnL")
    ratio = min(abs(lp), abs(sp)) / big
    if floor is None:
        return GateResult("leg_balance", None, ratio, None, NOT_CONFIGURED)
    return GateResult("leg_balance", ratio >= floor, ratio, floor, f">= {floor}")


def top_pnl_removal(result: ResultRecord, config: Config) -> GateResult:
    n = config.risk.top_n_pnl_days_removed
    floor = config.risk.min_sharpe_after_top_pnl_removal
    if result.daily_pnl is None:
        return GateResult("top_pnl_removal", None, None, floor, "daily_pnl missing")
    if n is None:
        return GateResult("top_pnl_removal", None, None, floor,
                          f"{NOT_CONFIGURED} (top_n_pnl_days_removed)")
    pnl = result.daily_pnl.dropna()
    if len(pnl) <= n + 1:
        return GateResult("top_pnl_removal", None, None, floor,
                          f"only {len(pnl)} days, cannot remove {n}")
    top = pnl.abs().sort_values(ascending=False, kind="stable").index[:n]
    sharpe = annualized_sharpe(pnl.drop(top))
    if floor is None:
        return GateResult("top_pnl_removal", None, sharpe, None, NOT_CONFIGURED)
    if sharpe is None:
        return GateResult("top_pnl_removal", None, None, floor, "sharpe undefined after removal")
    return GateResult("top_pnl_removal", sharpe >= floor, sharpe, floor,
                      f"sharpe after removing top {n} |PnL| days >= {floor}")


def weight_concentration(result: ResultRecord, config: Config) -> GateResult:
    limit = config.risk.weight_concentration_max
    v = result.weight_concentration
    if limit is None:
        return GateResult("weight_concentration", None, v, None, NOT_CONFIGURED)
    if v is None:
        return GateResult("weight_concentration", None, None, limit, "weight_concentration missing")
    return GateResult("weight_concentration", v <= limit, v, limit, f"<= {limit}")


def weak_year(result: ResultRecord, config: Config) -> GateResult:
    floor = config.risk.weak_year_sharpe_floor
    spread_max = config.risk.max_yearly_sharpe_spread
    threshold = {"floor": floor, "max_spread": spread_max}
    sharpes = [r.sharpe for r in result.yearly if r.sharpe is not None]
    if not sharpes:
        return GateResult("weak_year", None, None, threshold, "no yearly rows")
    lo, hi = min(sharpes), max(sharpes)
    value = {"min": lo, "spread": hi - lo}

    halves, notes = [], []
    if floor is None:
        notes.append(f"floor {NOT_CONFIGURED}")
    else:
        halves.append(lo >= floor)
        notes.append(f"min yearly sharpe {lo:.2f} vs floor {floor}")
    if spread_max is None:
        notes.append(f"spread {NOT_CONFIGURED}")
    else:
        halves.append(hi - lo <= spread_max)
        notes.append(f"spread {hi - lo:.2f} vs max {spread_max}")

    passed = None if not halves else all(halves)
    if not halves:
        notes = [NOT_CONFIGURED]
    return GateResult("weak_year", passed, value, threshold, "; ".join(notes))


CHECKS = (drawdown_depth, drawdown_timing, leg_balance, top_pnl_removal, weight_concentration,
          weak_year)


def assess(result: ResultRecord, config: Config,
           gate_results: list[GateResult] | None = None) -> tuple[bool, list[GateResult], list[str]]:
    checks = [fn(result, config) for fn in CHECKS]
    reasons = []
    for c in checks:
        if c.passed is False:
            reasons.append(f"{c.name} failed: value {c.value} vs {c.threshold}")
        elif c.passed is None and not c.note.startswith("informational"):
            reasons.append(f"{c.name} not evaluated: {c.note}")
    # Sub-universe is decided by the Evaluator gate; reference it rather than re-check.
    for g in gate_results or []:
        if g.name == "sub_universe":
            reasons.append(f"sub_universe: see evaluator gate (passed={g.passed})")
    veto = any(c.passed is False for c in checks)
    return veto, checks, reasons
