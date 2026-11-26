"""How plain or crowded a run's fresh alphas were, to tell whether the Generator improves.

Method (owner 2026-11-25): for every fresh alpha of a run (a Generator alpha or its Critic
revision; repairs and seeds are left out) take its signal profile (``generator.signal_profile``:
non-base signal fields, whether it reads price/volume, their datasets, crowded and heavily
used fields from BRAIN's catalog facts of its market) and BRAIN's result. Reported shares:
alphas with one source of information or none, price/volume only, every signal field
crowded or heavily used, two or more datasets; the median over alphas of the most-used
signal field's BRAIN alpha count; mean |Sharpe| and how many reached |Sharpe| >= 1 (raw or
risk-neutralized); the production correlations BRAIN measured. Baseline: run
rs20261125130750 (46% of 52 fresh alphas with one source or none, 29% only crowded or
heavily used fields, mean |Sharpe| 0.53).
Read-only; no thresholds decide anything here.
"""

from __future__ import annotations

import statistics
from typing import Callable

from .fastexpr import analyze
from .generator import GeneratorConfig, crowd_sets, signal_profile

FRESH_LEVERS = (None, "critic_revision")


def _best(r) -> float | None:
    if r is None or r.status != "complete":
        return None
    vals = [abs(r.sharpe)] if isinstance(r.sharpe, (int, float)) else []
    rn = ((r.alt_stats or {}).get("risk_neutralized") or {}).get("sharpe")
    if isinstance(rn, (int, float)):
        vals.append(abs(rn))
    return max(vals) if vals else None


def generation_quality(outcomes: list[dict], run_id: str, spec, config: GeneratorConfig,
                       catalog_of: Callable[[str, int], tuple]) -> dict:
    """Quality of ``run_id``'s fresh alphas (module doc). ``catalog_of(region, delay)`` ->
    (datafields or None, field meta). Alphas of a market without field facts count in the
    totals and the Sharpe numbers only."""
    sets: dict[tuple, tuple] = {}
    rows = [r for r in outcomes if r["candidate"].candidate_id.startswith(run_id)
            and r["candidate"].lever in FRESH_LEVERS]
    profiled = one = pv_only = all_crowded = two_ds = 0
    max_alphas, sharpes, prods = [], [], []
    strong = complete = 0
    for row in rows:
        c, r = row["candidate"], row.get("result")
        best = _best(r)
        if best is not None:
            complete += 1
            sharpes.append(abs(r.sharpe) if isinstance(r.sharpe, (int, float)) else best)
            strong += best >= 1.0
            prod = ((r.check_values or {}).get("PROD_CORRELATION") or {}).get("value")
            if isinstance(prod, (int, float)):
                prods.append(round(prod, 3))
        fields, meta = catalog_of(c.region, c.delay)
        if not meta:
            continue
        key = (c.region, c.delay)
        if key not in sets:
            sets[key] = crowd_sets(meta, config, spec.base_fields)
        crowded, heavy = sets[key]
        a = analyze(c.expression, spec, datafields=fields)
        p = signal_profile(a.fields_used, spec.base_fields, meta, config, crowded, heavy)
        profiled += 1
        one += p["sources"] < 2
        pv_only += not p["fields"]
        all_crowded += bool(p["fields"]) and not p["fresh"]
        two_ds += len(p["datasets"]) >= 2
        if p["max_alphas"] is not None:
            max_alphas.append(p["max_alphas"])

    def share(n: int) -> float | None:
        return round(n / profiled, 3) if profiled else None

    return {
        "run_id": run_id,
        "fresh_alphas": len(rows),
        "complete": complete,
        "profiled": profiled,
        "one_source_or_less": share(one),
        "price_volume_only": share(pv_only),
        "all_fields_crowded_or_heavy": share(all_crowded),
        "two_or_more_datasets": share(two_ds),
        "median_most_used_field_alphas": (statistics.median(max_alphas) if max_alphas else None),
        "mean_abs_sharpe": round(statistics.fmean(sharpes), 3) if sharpes else None,
        "best_abs_sharpe_at_least_1": strong,
        "prod_correlations": sorted(prods),
    }


def quality_lines(q: dict) -> list[str]:
    """Plain-text summary of ``generation_quality``."""
    def pct(v):
        return "-" if v is None else f"{v * 100:.0f}%"
    return [
        f"{q['run_id']}: {q['fresh_alphas']} fresh alphas, {q['complete']} complete",
        f"  one source or none {pct(q['one_source_or_less'])}, price/volume only "
        f"{pct(q['price_volume_only'])}, only crowded/heavily used fields "
        f"{pct(q['all_fields_crowded_or_heavy'])}, two+ datasets {pct(q['two_or_more_datasets'])}",
        f"  median most-used field: {q['median_most_used_field_alphas']} BRAIN alphas; "
        f"mean |Sharpe| {q['mean_abs_sharpe']}; |Sharpe| >= 1: {q['best_abs_sharpe_at_least_1']}",
        f"  production correlation measured: {q['prod_correlations'] or 'none'}",
    ]
