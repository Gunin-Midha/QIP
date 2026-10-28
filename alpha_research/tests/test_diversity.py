import numpy as np
import pandas as pd
import pytest

from alpha_research.diversity import (
    InsufficientHistoryError, admission_check, candidate_correlations, effective_n,
    marginal_gain, pnl_matrix, pool_sharpe,
)
from conftest import gaussian_series


def pool(n_cols=3, n=500):
    return pnl_matrix({f"a{i}": gaussian_series(n, mean=0.08, seed=i) for i in range(n_cols)})


def test_near_duplicate_rejected(config):
    m = pool()
    dup = m["a0"] + gaussian_series(500, std=0.1, seed=99)
    corr = candidate_correlations(dup, m)
    assert corr["max"] > 0.99 - 0.01
    admit, gates = admission_check(dup, m, config)
    assert admit is False
    assert gates[0].name == "pool_correlation" and gates[0].passed is False


def test_independent_series_admitted(config):
    m = pool()
    admit, gates = admission_check(gaussian_series(500, mean=0.08, seed=42), m, config)
    assert admit is True
    gain = next(g for g in gates if g.name == "marginal_pool_sharpe_gain")
    assert gain.passed is None and gain.note == "not configured"


def test_gain_threshold_applies(make_config):
    cfg = make_config(diversity__min_marginal_pool_sharpe_gain=10.0)
    admit, _ = admission_check(gaussian_series(500, seed=42), pool(), cfg)
    assert admit is False


def test_empty_pool_gain_is_own_sharpe(config):
    s = gaussian_series(500, mean=0.1, seed=3)
    assert marginal_gain(s, pd.DataFrame()) == pytest.approx(s.mean() / s.std() * np.sqrt(252))
    admit, gates = admission_check(s, pd.DataFrame(), config)
    assert admit is True


def test_effective_n_identical_and_independent():
    s = gaussian_series(1000, seed=0)
    same = pnl_matrix({f"c{i}": s for i in range(10)})
    assert effective_n(same) == pytest.approx(1.0, abs=1e-6)
    indep = pnl_matrix({f"c{i}": gaussian_series(1000, seed=i) for i in range(10)})
    assert effective_n(indep) == pytest.approx(10.0, abs=0.5)


def test_insufficient_history():
    with pytest.raises(InsufficientHistoryError):
        pnl_matrix({"a": gaussian_series(59, seed=0), "b": gaussian_series(59, seed=1)})
    with pytest.raises(InsufficientHistoryError):   # 100 days each, only 40 overlap
        pnl_matrix({"a": gaussian_series(100, seed=0, start="2019-01-01"),
                    "b": gaussian_series(100, seed=1, start="2019-04-12")})


def test_pnl_matrix_drops_nan_rows():
    a = gaussian_series(100, seed=0)
    b = gaussian_series(100, seed=1)
    b.iloc[:5] = np.nan
    assert len(pnl_matrix({"a": a, "b": b})) == 95


def test_pool_sharpe_equal_weight():
    m = pool()
    combined = m.mean(axis=1)
    assert pool_sharpe(m) == pytest.approx(combined.mean() / combined.std() * np.sqrt(252))


def test_missing_pnl_not_evaluable(config):
    admit, gates = admission_check(None, pool(), config)
    assert admit is None and gates[0].passed is None
