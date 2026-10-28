import numpy as np
import pandas as pd

from alpha_research.data_scientist import audit, nan_pattern


def frame(n_days=300, n_inst=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(rng.normal(size=(n_days, n_inst)), index=idx,
                        columns=[f"I{i}" for i in range(n_inst)])


def test_dense_frame_is_usable():
    a = audit(frame())
    assert a.n_days == 252
    assert a.coverage_median == 1.0 and a.coverage_min == 1.0
    assert a.nan_pattern == "random"
    assert a.recommendation == "usable"
    assert a.update_gap_median_days is not None and a.update_gap_median_days >= 1
    assert 0 <= a.tail_fraction < 0.01


def test_sparse_frame_is_flagged():
    df = frame()
    rng = np.random.default_rng(1)
    sparse = df.where(rng.random(df.shape) < 0.15)
    a = audit(sparse)
    assert a.coverage_median < 0.2
    assert a.recommendation in ("not_usable", "usable_with_backfill")


def test_copy_of_other_field_is_redundant():
    df = frame()
    a = audit(df, others={"copy": df.copy(), "independent": frame(seed=9)})
    assert a.redundancy == ["copy"]
    assert a.recommendation == "not_usable"


def test_nan_grouped_by_date():
    df = frame()
    df.iloc[::5, :] = np.nan        # every 5th day missing entirely
    assert nan_pattern(df.iloc[-252:]) == "clustered_by_date"
    a = audit(df)
    assert a.nan_pattern == "clustered_by_date"
    assert a.recommendation == "usable_with_backfill"


def test_nan_grouped_by_instrument():
    df = frame()
    df.iloc[:, :30] = np.nan
    assert nan_pattern(df) == "clustered_by_instrument"


def test_slow_updating_field_gap():
    df = frame(n_days=260, n_inst=10)
    slow = df.iloc[::20].reindex(df.index).ffill()   # value changes every 20 business days
    a = audit(slow)
    assert 26 <= a.update_gap_median_days <= 30     # calendar days
