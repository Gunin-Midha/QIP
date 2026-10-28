"""Diversity manager (spec section 9). Works on daily PnL Series indexed by date."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import NOT_CONFIGURED, Config
from .schemas import GateResult

TRADING_DAYS = 252
MIN_OVERLAP_DAYS = 60


class InsufficientHistoryError(ValueError):
    pass


def pnl_matrix(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Inner-join on dates and drop rows with any NaN. Needs >= 60 overlapping days."""
    if not series_dict:
        return pd.DataFrame()
    df = pd.concat({k: v.astype(float) for k, v in series_dict.items()}, axis=1, join="inner")
    df = df.dropna(how="any").sort_index()
    if len(df) < MIN_OVERLAP_DAYS:
        raise InsufficientHistoryError(
            f"only {len(df)} overlapping days, need at least {MIN_OVERLAP_DAYS}"
        )
    return df


def _sharpe(pnl: pd.Series) -> float | None:
    std = float(pnl.std())
    if len(pnl) < 2 or std == 0 or math.isnan(std):
        return None
    return float(pnl.mean()) / std * math.sqrt(TRADING_DAYS)


def candidate_correlations(candidate_pnl: pd.Series, pool_matrix: pd.DataFrame) -> dict:
    if pool_matrix is None or pool_matrix.shape[1] == 0:
        return {"max": None, "mean": None, "per_alpha": {}}
    joined = pnl_matrix({"__candidate__": candidate_pnl, **{c: pool_matrix[c] for c in pool_matrix}})
    cand = joined.pop("__candidate__")
    per = {c: float(cand.corr(joined[c])) for c in joined.columns}
    vals = [v for v in per.values() if not math.isnan(v)]
    return {
        "max": max(vals) if vals else None,
        "mean": float(np.mean(vals)) if vals else None,
        "per_alpha": per,
    }


def pool_sharpe(matrix: pd.DataFrame) -> float | None:
    """Equal-weight combined PnL (row mean), annualized Sharpe."""
    if matrix is None or matrix.shape[1] == 0:
        return None
    return _sharpe(matrix.mean(axis=1))


def marginal_gain(candidate_pnl: pd.Series, pool_matrix: pd.DataFrame) -> float | None:
    if pool_matrix is None or pool_matrix.shape[1] == 0:
        return _sharpe(candidate_pnl.dropna())
    joined = pnl_matrix({"__candidate__": candidate_pnl, **{c: pool_matrix[c] for c in pool_matrix}})
    with_cand = pool_sharpe(joined)
    without = pool_sharpe(joined.drop(columns="__candidate__"))
    if with_cand is None or without is None:
        return None
    return with_cand - without


def effective_n(matrix: pd.DataFrame) -> float | None:
    """N_eff = (sum eigenvalues)^2 / sum(eigenvalues^2) of the column correlation matrix.
    Eigenvalues of a correlation matrix sum to its dimension, so the formula is valid."""
    if matrix is None or matrix.shape[1] == 0:
        return None
    corr = np.nan_to_num(matrix.corr().to_numpy(), nan=1.0)
    eig = np.linalg.eigvalsh(corr)
    denom = float(np.sum(eig ** 2))
    return float(np.sum(eig)) ** 2 / denom if denom > 0 else None


def admission_check(candidate_pnl: pd.Series | None, pool_matrix: pd.DataFrame | None,
                    config: Config) -> tuple[bool | None, list[GateResult]]:
    """Returns (admit, gates). admit is None when it cannot be evaluated (no PnL)."""
    max_corr = config.diversity.max_pool_correlation
    min_gain = config.diversity.min_marginal_pool_sharpe_gain
    if candidate_pnl is None:
        return None, [GateResult("pool_correlation", None, None, max_corr, "daily_pnl missing")]

    empty = pool_matrix is None or pool_matrix.shape[1] == 0
    gates = []
    if empty:
        gates.append(GateResult("pool_correlation", True, None, max_corr, "pool is empty"))
    else:
        corr = candidate_correlations(candidate_pnl, pool_matrix)
        if max_corr is None:
            gates.append(GateResult("pool_correlation", None, corr["max"], None, NOT_CONFIGURED))
        elif corr["max"] is None:
            gates.append(GateResult("pool_correlation", None, None, max_corr, "correlation undefined"))
        else:
            gates.append(GateResult("pool_correlation", corr["max"] <= max_corr, corr["max"], max_corr,
                                    f"max correlation to pool <= {max_corr}"))

    gain = marginal_gain(candidate_pnl, pool_matrix)
    if min_gain is None:
        gates.append(GateResult("marginal_pool_sharpe_gain", None, gain, None, NOT_CONFIGURED))
    elif gain is None:
        gates.append(GateResult("marginal_pool_sharpe_gain", None, None, min_gain, "gain undefined"))
    else:
        gates.append(GateResult("marginal_pool_sharpe_gain", gain >= min_gain, gain, min_gain,
                                f">= {min_gain}"))

    if any(g.passed is False for g in gates):
        return False, gates
    corr_gate = gates[0]
    if corr_gate.passed is None:
        return None, gates
    return True, gates
