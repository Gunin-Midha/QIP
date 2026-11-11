"""Regenerate the committed test fixtures. Deterministic (fixed seed).

    python tests/fixtures/generate_fixtures.py

Writes:
* brain_results/<fixture_id>.json  (our fixture format, percents for turnover/returns/drawdown)
* pnl/<fixture_id>.csv             (synthetic daily PnL, Gaussian, independent per alpha)
* candidates/hand_written.json     (candidates with expected_decision / expected_failure_class)

screenshot_aggregate.json is written from the IS summary screenshot values supplied by the
owner, not generated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BRAIN = ROOT / "brain_results"
PNL = ROOT / "pnl"
CANDS = ROOT / "candidates"

BASE = {
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 4,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "start_date": "2019-01-01",
    "end_date": "2023-12-31",
    "parent_id": None,
    "prompt_version": None,
    "model": None,
}
DATES = pd.bdate_range("2019-01-01", "2023-12-29")
RNG = np.random.default_rng(20261027)


def fitness(sharpe: float, returns_pct: float, turnover_pct: float) -> float:
    return round(sharpe * math.sqrt(abs(returns_pct / 100) / max(turnover_pct / 100, 0.125)), 4)


def write_pnl(name: str, values: np.ndarray) -> str:
    PNL.mkdir(exist_ok=True)
    pd.DataFrame({"date": DATES.strftime("%Y-%m-%d"), "pnl": np.round(values, 6)}).to_csv(
        PNL / f"{name}.csv", index=False)
    return f"pnl/{name}.csv"


def gaussian_pnl(sharpe: float) -> np.ndarray:
    return RNG.normal(sharpe / math.sqrt(252), 1.0, len(DATES))


def metrics(sharpe, turnover_pct, returns_pct, *, sub=1.10, cutoff=0.80, drawdown_pct=6.0,
            conc=0.03, self_corr=None, long_pnl=6.0e6, short_pnl=4.5e6, margin=None):
    m = {
        "status": "complete",
        "sharpe": sharpe,
        "fitness": fitness(sharpe, returns_pct, turnover_pct),
        "turnover": turnover_pct,
        "returns": returns_pct,
        "drawdown": drawdown_pct,
        "margin": margin if margin is not None else round(returns_pct / (turnover_pct * 252) * 1e4 / 2, 2),
        "sub_universe_sharpe": sub,
        "sub_universe_cutoff": cutoff,
        "weight_concentration": conc,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
    }
    if self_corr is not None:
        m["self_correlation"] = self_corr
    return m


# EUR settings our account offers (TOP3000 is a USA/GLB universe, not EUR)
EUR = {"region": "EUR", "universe": "TOP2500"}


def cand(cid, expression, dataset, family, hypothesis, fixture_id, expected, failure_class,
         **overrides):
    c = dict(BASE)
    c.update(candidate_id=cid, expression=expression, dataset=dataset, family=family,
             hypothesis=hypothesis)
    c.update(overrides)
    c.update(fixture_id=fixture_id, expected_decision=expected,
             expected_failure_class=failure_class)
    return c


def main() -> None:
    BRAIN.mkdir(exist_ok=True)
    CANDS.mkdir(exist_ok=True)
    fixtures: dict[str, dict] = {}
    cands: list[dict] = []

    # ---- 5 accepts ------------------------------------------------------------------
    accepts = [
        ("acc_01", "rank(-ts_delta(close, 5))", "pv1", "price_volume",
         "Short-term losers revert over the next week.", 1.85, 22.0, 14.0, 0.21),
        ("acc_02", "group_rank(ts_rank(operating_income / assets, 252), industry)", "fundamental6",
         "fundamentals", "Improving profitability within industry predicts outperformance.",
         1.62, 9.5, 10.5, 0.33),
        ("acc_03", "rank(ts_mean(est_eps_revision, 20))", "analyst4", "analyst",
         "Upward analyst revisions lead price.", 1.74, 15.0, 12.0, 0.18),
        ("acc_04", "-rank(ts_corr(rank(volume), rank(close), 10))", "pv1", "price_volume",
         "Price-volume divergence signals exhaustion.", 1.51, 31.0, 16.0, 0.44),
        ("acc_05", "rank(ts_zscore(news_sentiment, 60))", "news12", "sentiment",
         "Persistent positive news tone is under-reacted to.", 1.95, 18.0, 15.5, 0.27),
    ]
    acc_pnl = {}
    for cid, expr, ds, fam, hyp, s, t, r, sc in accepts:
        fid = f"sim_{cid}"
        acc_pnl[cid] = gaussian_pnl(s)
        m = metrics(s, t, r, self_corr=sc)
        m["daily_pnl_file"] = write_pnl(fid, acc_pnl[cid])
        fixtures[fid] = m
        cands.append(cand(cid, expr, ds, fam, hyp, fid, "accept", "none",
                          **(EUR if ds == "news12" else {})))

    # ---- 5 sharpe failures: near misses (|sharpe| >= 0.6 x 1.25, repair.near_miss_share in
    # thresholds_test.yaml) go to repair, far misses are rejected ------------------------
    sharpe_fails = [
        ("shp_01", "rank(ts_delta(close, 20))", "pv1", "price_volume", 1.10, 20.0, 9.0),
        ("shp_02", "rank(cashflow_op / enterprise_value)", "fundamental6", "fundamentals", 0.95, 8.0, 7.0),
        ("shp_03", "rank(ts_std_dev(returns, 20))", "pv1", "price_volume", 0.72, 25.0, 6.0),
        ("shp_04", "rank(est_eps / close)", "analyst4", "analyst", 1.25, 12.0, 9.5),   # boundary: not > 1.25
        ("shp_05", "rank(-ts_mean(news_volume, 5))", "news12", "sentiment", 0.40, 30.0, 3.0),
    ]
    for cid, expr, ds, fam, s, t, r in sharpe_fails:
        fid = f"sim_{cid}"
        fixtures[fid] = metrics(s, t, r)
        cands.append(cand(cid, expr, ds, fam, None, fid,
                          "repair" if abs(s) >= 0.6 * 1.25 else "reject", "sharpe",
                          **(EUR if ds == "news12" else {})))

    # ---- 5 turnover failures (playbook has turnover -> repair) ----------------------
    turnover_fails = [
        ("tvr_01", "rank(-returns)", "pv1", 1.70, 85.0, 16.0),
        ("tvr_02", "rank(-ts_delta(vwap, 1))", "pv1", 1.55, 95.0, 18.0),
        ("tvr_03", "rank(ts_rank(assets, 504))", "fundamental6", 1.40, 0.5, 9.0),   # below 1% floor
        ("tvr_04", "rank(close / ts_delay(close, 2))", "pv1", 1.60, 72.0, 15.0),
        ("tvr_05", "rank(ts_mean(book_value, 756))", "fundamental6", 1.35, 0.8, 8.0),  # below 1% floor
    ]
    for cid, expr, ds, s, t, r in turnover_fails:
        fid = f"sim_{cid}"
        fixtures[fid] = metrics(s, t, r, self_corr=0.2)
        fam = "price_volume" if ds == "pv1" else "fundamentals"
        cands.append(cand(cid, expr, ds, fam, None, fid, "repair", "turnover"))

    # ---- 3 concentration / drawdown failures (risk veto -> reject) ------------------
    risk_fails = [
        ("con_01", "rank(market_cap)", "fundamental6", "fundamentals", "concentration",
         dict(conc=0.25)),
        ("con_02", "rank(1 / sales)", "fundamental6", "fundamentals", "concentration",
         dict(conc=0.18)),
        ("ddn_01", "rank(ts_mean(est_ptp, 10))", "analyst4", "analyst", "drawdown",
         dict(drawdown_pct=30.0)),
    ]
    for cid, expr, ds, fam, fc, extra in risk_fails:
        fid = f"sim_{cid}"
        fixtures[fid] = metrics(1.60, 20.0, 12.0, self_corr=0.3, **extra)
        cands.append(cand(cid, expr, ds, fam, None, fid, "reject", fc))

    # ---- 3 sub-universe failures (playbook entry -> repair) -------------------------
    sub_fails = [
        ("sub_01", "rank(ts_delta(est_eps, 60))", "analyst4", "analyst", 0.50, 0.73),
        ("sub_02", "rank(-ts_std_dev(returns, 60))", "pv1", "price_volume", 0.40, 0.81),
        ("sub_03", "rank(news_buzz)", "news12", "sentiment", 0.65, 0.78),
    ]
    for cid, expr, ds, fam, sub, cutoff in sub_fails:
        fid = f"sim_{cid}"
        fixtures[fid] = metrics(1.58, 21.0, 12.5, sub=sub, cutoff=cutoff, self_corr=0.25)
        cands.append(cand(cid, expr, ds, fam, None, fid, "repair", "sub_universe",
                          **(EUR if ds == "news12" else {})))

    # ---- 3 self-correlation failures (fixture provides the correlation) -------------
    for cid, expr, ds, fam, sc in [
        ("scr_01", "rank(-ts_delta(close, 4))", "pv1", "price_volume", 0.80),
        ("scr_02", "rank(ts_mean(est_eps_revision, 15))", "analyst4", "analyst", 0.75),
        ("scr_03", "group_rank(ts_rank(operating_income / assets, 200), industry)",
         "fundamental6", "fundamentals", 0.92),
    ]:
        fid = f"sim_{cid}"
        fixtures[fid] = metrics(1.70, 20.0, 13.0, self_corr=sc)
        cands.append(cand(cid, expr, ds, fam, None, fid, "repair", "self_correlation"))

    # ---- 1 pool-correlation rejection (diversity admission) -------------------------
    fid = "sim_div_01"
    m = metrics(1.80, 22.0, 14.0, self_corr=0.35)
    m["daily_pnl_file"] = write_pnl(fid, acc_pnl["acc_01"] + RNG.normal(0, 0.1, len(DATES)))
    fixtures[fid] = m
    cands.append(cand("div_01", "rank(-ts_delta(close, 6))", "pv1", "price_volume",
                      "Reversal over six days.", fid, "reject", "self_correlation"))

    # ---- 2 duplicates (same expression + settings, different whitespace/case) -------
    cands.append(cand("dup_01", "  RANK( -TS_DELTA(close,5) )", "pv1", "price_volume", None,
                      "sim_acc_01", "duplicate", None))
    cands.append(cand("dup_02", "rank(ts_delta( close , 20 ))", "pv1", "price_volume", None,
                      "sim_shp_01", "duplicate", None))

    # ---- 2 invalid ------------------------------------------------------------------
    cands.append(cand("inv_01", "rank(ts_delta(close, 5)", "pv1", "price_volume", None,
                      None, "reject", "incomplete"))
    cands.append(cand("inv_02", "rank(close)", "pv1", "price_volume", None, None, "reject",
                      "incomplete", region="XYZ"))

    # ---- 2 timeouts -----------------------------------------------------------------
    fixtures["sim_to_01"] = {"responses": [{"status": "running"}]}
    fixtures["sim_to_02"] = {"responses": [{"raise": "TimeoutError"}]}
    cands.append(cand("to_01", "rank(ts_av_diff(close, 30))", "pv1", "price_volume", None,
                      "sim_to_01", "reject", "incomplete"))
    cands.append(cand("to_02", "rank(ts_mean(est_eps, 5))", "analyst4", "analyst", None,
                      "sim_to_02", "reject", "incomplete"))

    for fid, data in fixtures.items():
        (BRAIN / f"{fid}.json").write_text(json.dumps(data, indent=2) + "\n")
    (CANDS / "hand_written.json").write_text(json.dumps(cands, indent=2) + "\n")

    (BRAIN / "screenshot_aggregate.json").write_text(json.dumps(SCREENSHOT, indent=2) + "\n")
    print(f"wrote {len(fixtures) + 1} fixtures and {len(cands)} candidates")


# Values from the IS summary screenshot (supplied by the owner). Percent units as shown.
SCREENSHOT = {
    "status": "complete",
    "sharpe": 1.68,
    "fitness": 1.11,
    "turnover": 25.78,
    "returns": 11.19,
    "drawdown": 7.04,
    "margin": 8.68,
    "sub_universe_sharpe": 1.07,
    "sub_universe_cutoff": 0.73,
    "yearly": [
        {"year": 2019, "sharpe": 1.40, "turnover": 23.05, "fitness": 0.73, "returns": 6.21,
         "drawdown": 3.33, "margin": 5.39, "long_count": 1437, "short_count": 1273},
        {"year": 2020, "sharpe": 0.11, "turnover": 26.99, "fitness": 0.02, "returns": 0.80,
         "drawdown": 7.04, "margin": 0.59, "long_count": 1387, "short_count": 1329},
        {"year": 2021, "sharpe": 1.83, "turnover": 27.72, "fitness": 1.28, "returns": 13.61,
         "drawdown": 4.28, "margin": 9.82, "long_count": 1491, "short_count": 1365},
        {"year": 2022, "sharpe": 3.59, "turnover": 25.64, "fitness": 3.79, "returns": 28.51,
         "drawdown": 5.11, "margin": 22.24, "long_count": 1654, "short_count": 1318},
        {"year": 2023, "sharpe": 1.01, "turnover": 25.59, "fitness": 0.49, "returns": 5.95,
         "drawdown": 4.78, "margin": 4.65, "long_count": 1576, "short_count": 1393},
    ],
}


if __name__ == "__main__":
    main()
