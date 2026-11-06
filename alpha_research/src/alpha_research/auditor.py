"""Overfitting auditor (spec section 10).

Deflation method ("expected_max_normal")
----------------------------------------
Deflate the best observed Sharpe in a family by the Sharpe that N zero-edge strategies
would be expected to reach by chance.

* N = number of variants simulated in the family (every simulation counts, including
  failures and repairs); T = years of data in the backtest.
* Standard error of an annualized Sharpe estimate (using the candidate's own Sharpe S):
      se = sqrt((1 + 0.5 * S^2) / T)
* Expected maximum of N independent standard normal draws:
      E_max(N) = (1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N * e)),  g = 0.5772
  with Phi^-1 the inverse normal CDF (scipy.stats.norm.ppf).
* Deflated Sharpe = S - se * E_max(N). For N = 1, E_max = 0 and deflated equals raw.

Citation: Bailey, D. H. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality", Journal of
Portfolio Management. TODO(citation): confirm the exact reference used in the write-up.

This is an approximation: it treats variants as independent, although variants in one
family are usually correlated, so N overstates the effective number of trials. The
write-up should say so.

Accept-time guard (``check_accept``, owner 2026-11-04: "ensure no overfitting"). N is the
number of simulations that count as trials against the accepted alpha, by
``auditor.trials_scope``:
  lineage  its repair family: the original candidate and every descendant that reached
           BRAIN (``research_loop.lineage_members``); catches over-repairing one idea
  dataset  every simulation on the same dataset (default); also catches searching one
           dataset with many generated ideas, so hammering one dataset raises its bar,
           which keeps the search spread out
  all      every simulation in memory (most conservative)
Invalid results do not count. T comes from the daily PnL (trading days / 252), else from
the yearly rows. The accept stands only if the deflated Sharpe is above
``auditor.min_deflated_sharpe``; null = not checked. With five years of data the
deflation is modest until N is large (Sharpe 1.3 needs about 40 trials to fall below 0),
so the repair budget and an out-of-sample test (OQ-12) remain the main guards.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from .config import NOT_CONFIGURED, Config
from .memory import MemoryStore
from .research_loop import lineage_members, lineage_root, simulated
from .schemas import GateResult, ResultRecord

TRADING_DAYS = 252

EULER_GAMMA = 0.5772
UNKNOWN_FAMILY = "__unknown__"


def expected_max_normal(n: int) -> float:
    if n < 1:
        raise ValueError("number of variants must be >= 1")
    if n == 1:
        return 0.0
    g = EULER_GAMMA
    return float((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e)))


def sharpe_standard_error(sharpe: float, years: float) -> float:
    if years <= 0:
        raise ValueError("years must be positive")
    return math.sqrt((1 + 0.5 * sharpe ** 2) / years)


def deflated_sharpe(sharpe: float, n_variants: int, years: float) -> float:
    return sharpe - sharpe_standard_error(sharpe, years) * expected_max_normal(n_variants)


class Auditor:
    def __init__(self, memory: MemoryStore, config: Config):
        self.memory = memory
        self.config = config

    def register_variant(self, family: str | None, candidate_id: str) -> int:
        self.memory.register_variant(family or UNKNOWN_FAMILY, candidate_id)
        return self.variant_count(family)

    def variant_count(self, family: str | None) -> int:
        return self.memory.variant_count(family or UNKNOWN_FAMILY)

    def deflate(self, sharpe: float, family: str | None, years: float) -> float:
        return deflated_sharpe(sharpe, max(self.variant_count(family), 1), years)

    def lineage_trials(self, candidate_id: str) -> int:
        """Simulations in the candidate's repair lineage, the candidate included."""
        return self.trials(candidate_id, "lineage")

    def trials(self, candidate_id: str, scope: str | None = None) -> int:
        """Simulations that count as trials against the candidate (module doc)."""
        scope = scope or self.config.auditor.trials_scope
        outcomes = self.memory.candidate_outcomes()
        cands = {r["candidate"].candidate_id: r["candidate"] for r in outcomes}
        if scope == "lineage":
            rows = lineage_members(outcomes, lineage_root(candidate_id, cands))
        elif scope == "dataset":
            ds = cands[candidate_id].dataset if candidate_id in cands else None
            rows = [r for r in outcomes if r["candidate"].dataset == ds]
        else:
            rows = outcomes
        return max(1, sum(1 for r in rows if simulated(r)))

    def check_accept(self, candidate_id: str, result: ResultRecord) -> GateResult:
        floor = self.config.auditor.min_deflated_sharpe
        scope = self.config.auditor.trials_scope
        n = self.trials(candidate_id)
        years = backtest_years(result)
        if floor is None:
            return GateResult("deflated_sharpe", None, None, None, NOT_CONFIGURED)
        if result.sharpe is None or years is None:
            return GateResult("deflated_sharpe", None, None, floor,
                              "Sharpe or backtest length missing")
        d = deflated_sharpe(result.sharpe, n, years)
        return GateResult("deflated_sharpe", d > floor, round(d, 4), floor,
                          f"Sharpe {result.sharpe} deflated for {n} simulation(s) "
                          f"(scope {scope}) over {years:.2f} years")

    def record_os_is(self, alpha_id: str, family: str | None, is_sharpe: float | None,
                     os_sharpe: float | None) -> float | None:
        self.memory.record_os_is(alpha_id, family or UNKNOWN_FAMILY, is_sharpe, os_sharpe)
        return os_is_ratio(is_sharpe, os_sharpe)

    def family_ratio(self, family: str | None) -> float | None:
        ratios = [os_is_ratio(r["is_sharpe"], r["os_sharpe"])
                  for r in self.memory.os_is_rows(family or UNKNOWN_FAMILY)]
        ratios = [r for r in ratios if r is not None]
        return sum(ratios) / len(ratios) if ratios else None

    def flag_family(self, family: str | None) -> bool | None:
        floor = self.config.auditor.os_is_ratio_floor
        if floor is None:
            return None  # OQ-8: not configured
        ratio = self.family_ratio(family)
        if ratio is None:
            return None
        return ratio < floor


def os_is_ratio(is_sharpe: float | None, os_sharpe: float | None) -> float | None:
    if is_sharpe is None or os_sharpe is None or is_sharpe <= 0:
        return None
    return os_sharpe / is_sharpe


def backtest_years(result: ResultRecord) -> float | None:
    if result.daily_pnl is not None and len(result.daily_pnl) > 1:
        return len(result.daily_pnl) / TRADING_DAYS
    if result.yearly:
        return float(len(result.yearly))
    return None

