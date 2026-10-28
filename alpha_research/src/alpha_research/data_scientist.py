"""Datafield audit (spec section 6).

Input: a DataFrame for one datafield in one region, dates as index, instruments as columns,
NaN for missing. Only the last 252 rows are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WINDOW = 252
TAIL_SIGMA = 3.0
REDUNDANCY_CORR = 0.9
COVERAGE_NOT_USABLE = 0.5
COVERAGE_BACKFILL = 0.8
# nan_pattern: observed variance must exceed this multiple of the random-mask variance
# before a NaN pattern counts as clustered. A statistical constant, not a policy threshold.
NAN_CLUSTER_VARIANCE_RATIO = 4.0
MIN_RANK_PAIRS = 3


@dataclass
class AuditRecord:
    n_days: int
    n_instruments: int
    coverage_median: float | None
    coverage_min: float | None
    update_gap_median_days: float | None
    tail_fraction: float | None
    nan_pattern: str
    redundancy: list[str] = field(default_factory=list)
    redundancy_scores: dict[str, float] = field(default_factory=dict)
    recommendation: str = "usable"
    reasons: list[str] = field(default_factory=list)


def _window(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0 or df.shape[0] == 0:
        raise ValueError("audit needs a non-empty DataFrame (dates x instruments)")
    return df.iloc[-WINDOW:].astype(float)


def coverage(df: pd.DataFrame) -> tuple[float, float]:
    per_day = df.notna().sum(axis=1) / df.shape[1]
    return float(per_day.median()), float(per_day.min())


def update_gap_median_days(df: pd.DataFrame) -> float | None:
    """Per instrument: gaps (in days) between consecutive value changes; take the median
    gap. Then the median across instruments. Instruments with fewer than two changes are
    skipped. With a DatetimeIndex gaps are calendar days, otherwise row counts."""
    per_inst = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        changed = s.ne(s.shift())
        changed.iloc[0] = False
        idx = s.index[changed.to_numpy()]
        if len(idx) < 2:
            continue
        if isinstance(idx, pd.DatetimeIndex):
            gaps = np.diff(idx.values).astype("timedelta64[D]").astype(float)
        else:
            positions = df.index.get_indexer(idx)
            gaps = np.diff(positions).astype(float)
        per_inst.append(float(np.median(gaps)))
    return float(np.median(per_inst)) if per_inst else None


def tail_fraction(df: pd.DataFrame) -> float | None:
    """Per day: share of non-NaN values more than 3 cross-sectional std from the
    cross-sectional median. Averaged across days with at least 2 values."""
    med = df.median(axis=1)
    std = df.std(axis=1)
    dev = df.sub(med, axis=0).abs()
    tails = dev.gt(TAIL_SIGMA * std, axis=0) & df.notna()
    counts = df.notna().sum(axis=1)
    valid = (counts >= 2) & std.gt(0)
    if not valid.any():
        return None
    return float((tails.sum(axis=1)[valid] / counts[valid]).mean())


def nan_pattern(df: pd.DataFrame) -> str:
    """Classify the NaN mask as ``random``, ``clustered_by_date`` or ``clustered_by_instrument``.

    Method: let p be the overall NaN rate. If NaNs were scattered independently at rate p,
    the per-day NaN fraction would have variance p(1-p)/n_instruments and the
    per-instrument NaN fraction would have variance p(1-p)/n_days. Divide each observed
    variance by its random-mask expectation. If neither ratio exceeds
    NAN_CLUSTER_VARIANCE_RATIO the pattern is ``random``; otherwise it is the axis with
    the larger ratio. A mask with no NaNs (or all NaNs) is reported as ``random``.
    """
    mask = df.isna().to_numpy()
    n_days, n_inst = mask.shape
    p = mask.mean()
    if p == 0 or p == 1:
        return "random"
    base = p * (1 - p)
    day_ratio = mask.mean(axis=1).var() / (base / n_inst) if n_days > 1 else 0.0
    inst_ratio = mask.mean(axis=0).var() / (base / n_days) if n_inst > 1 else 0.0
    if max(day_ratio, inst_ratio) <= NAN_CLUSTER_VARIANCE_RATIO:
        return "random"
    return "clustered_by_date" if day_ratio >= inst_ratio else "clustered_by_instrument"


def rank_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float | None:
    """Mean over days of the Spearman correlation of the two cross-sections, using only
    instruments present in both on that day."""
    days = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    if len(days) == 0 or len(cols) < MIN_RANK_PAIRS:
        return None
    a, b = a.loc[days, cols], b.loc[days, cols]
    both = a.notna() & b.notna()
    ra = a.where(both).rank(axis=1)
    rb = b.where(both).rank(axis=1)
    corr = ra.corrwith(rb, axis=1)
    corr = corr[both.sum(axis=1) >= MIN_RANK_PAIRS].dropna()
    return float(corr.mean()) if len(corr) else None


def audit(df: pd.DataFrame, others: dict[str, pd.DataFrame] | None = None) -> AuditRecord:
    w = _window(df)
    cov_med, cov_min = coverage(w)
    pattern = nan_pattern(w)

    scores: dict[str, float] = {}
    for name, other in (others or {}).items():
        c = rank_correlation(w, _window(other))
        if c is not None:
            scores[name] = c
    # Absolute value: a sign-flipped copy carries the same information.
    redundant = sorted(n for n, c in scores.items() if abs(c) > REDUNDANCY_CORR)

    reasons = []
    if cov_med < COVERAGE_NOT_USABLE or redundant:
        rec = "not_usable"
        if cov_med < COVERAGE_NOT_USABLE:
            reasons.append(f"median coverage {cov_med:.2f} < {COVERAGE_NOT_USABLE}")
        if redundant:
            reasons.append(f"redundant with {redundant} (|rank corr| > {REDUNDANCY_CORR})")
    elif cov_med < COVERAGE_BACKFILL or pattern != "random":
        rec = "usable_with_backfill"
        if cov_med < COVERAGE_BACKFILL:
            reasons.append(f"median coverage {cov_med:.2f} < {COVERAGE_BACKFILL}")
        if pattern != "random":
            reasons.append(f"NaN pattern is {pattern}")
    else:
        rec = "usable"
        reasons.append(f"median coverage {cov_med:.2f} >= {COVERAGE_BACKFILL}, NaNs random, "
                       "no redundant field")

    return AuditRecord(
        n_days=w.shape[0],
        n_instruments=w.shape[1],
        coverage_median=cov_med,
        coverage_min=cov_min,
        update_gap_median_days=update_gap_median_days(w),
        tail_fraction=tail_fraction(w),
        nan_pattern=pattern,
        redundancy=redundant,
        redundancy_scores=scores,
        recommendation=rec,
        reasons=reasons,
    )
