"""Metric validator (spec section 3.3).

Pure functions used as a tripwire on every result. They return notes and never raise on a
mismatch. They raise only on malformed input (non-numeric values).

Notes are plain strings that start with a code, e.g. ``"fitness_mismatch: ..."``.
"""

from __future__ import annotations

import math
from typing import Iterable

from .schemas import ResultRecord, YearRow

FITNESS_TOLERANCE = 0.01
TURNOVER_FLOOR = 0.125          # part of BRAIN's fitness formula, not a policy threshold
TRADING_DAYS = 252
MARGIN_RATIO_BAND = (0.8, 1.25)


def _num(x, name: str) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be a number or None, got {x!r}")
    if math.isnan(x):
        return None
    return float(x)


def expected_fitness(sharpe, returns, turnover) -> float | None:
    """``sharpe * sqrt(abs(returns) / max(turnover, 0.125))``. Inputs as fractions.

    Returns None if any input is missing.
    """
    s, r, t = _num(sharpe, "sharpe"), _num(returns, "returns"), _num(turnover, "turnover")
    if s is None or r is None or t is None:
        return None
    return s * math.sqrt(abs(r) / max(t, TURNOVER_FLOOR))


def check_fitness(sharpe, returns, turnover, reported_fitness, label: str = "aggregate") -> list[str]:
    reported = _num(reported_fitness, "fitness")
    if reported is None:
        return []  # skip, do not fail
    expected = expected_fitness(sharpe, returns, turnover)
    if expected is None:
        return [f"fitness_unchecked: {label} missing sharpe, returns or turnover"]
    diff = abs(expected - reported)
    if diff > FITNESS_TOLERANCE:
        return [f"fitness_mismatch: {label} reported {reported:.4f}, expected {expected:.4f}"]
    return [f"fitness_ok: {label} reported {reported:.4f}, expected {expected:.4f}"]


def check_yearly_consistency(aggregate_sharpe, yearly: Iterable[YearRow]) -> list[str]:
    """Information only: the aggregate Sharpe should lie within the range of yearly Sharpes."""
    agg = _num(aggregate_sharpe, "sharpe")
    sharpes = [_num(r.sharpe, f"yearly[{r.year}].sharpe") for r in yearly]
    sharpes = [s for s in sharpes if s is not None]
    if agg is None or not sharpes:
        return []
    lo, hi = min(sharpes), max(sharpes)
    if lo <= agg <= hi:
        return [f"yearly_consistent: aggregate sharpe {agg:.2f} within [{lo:.2f}, {hi:.2f}]"]
    return [f"yearly_inconsistent: aggregate sharpe {agg:.2f} outside [{lo:.2f}, {hi:.2f}]"]


def naive_margin_bps(returns, turnover) -> float | None:
    r, t = _num(returns, "returns"), _num(turnover, "turnover")
    if r is None or t is None or t == 0:
        return None
    return r / (t * TRADING_DAYS) * 10_000


def margin_check(result: ResultRecord) -> list[str]:
    """Record BRAIN's margin and the naive margin side by side. Never fails.

    TODO(open-question) OQ-1: BRAIN's reported margin is about half the naive value. Until
    the definition is confirmed, margin is recorded and not gated.
    """
    reported = _num(result.margin_bps, "margin_bps")
    naive = naive_margin_bps(result.returns, result.turnover)
    if reported is None or naive is None:
        return [f"margin_recorded: reported={reported} naive={naive}"]
    notes = [f"margin_recorded: reported={reported:.2f}bps naive={naive:.2f}bps"]
    if reported == 0:
        notes.append("margin_mismatch: reported margin is 0, ratio undefined (OQ-1)")
        return notes
    ratio = naive / reported
    lo, hi = MARGIN_RATIO_BAND
    if not (lo <= ratio <= hi):
        notes.append(f"margin_mismatch: naive/reported={ratio:.2f} outside [{lo}, {hi}] (OQ-1)")
    return notes


def validate_result(result: ResultRecord) -> list[str]:
    """Run every check on a result and return the notes."""
    notes = check_fitness(result.sharpe, result.returns, result.turnover, result.fitness)
    for row in result.yearly:
        notes += check_fitness(row.sharpe, row.returns, row.turnover, row.fitness,
                               label=f"year {row.year}")
    notes += check_yearly_consistency(result.sharpe, result.yearly)
    notes += margin_check(result)
    return notes
