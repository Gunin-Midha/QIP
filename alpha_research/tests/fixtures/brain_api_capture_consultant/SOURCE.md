# Live BRAIN responses from our account after the consultant upgrade

Saved on 2026-11-10 during the consultant discovery run (one BRAIN session, account level GOLD).
The simulation is acc_01 again, the same alpha as `../brain_api_capture/`, so the two captures
can be compared: simulation KZi083UV4Tcad27VuxcXSm, alpha MP6POJNn,
`rank(-ts_delta(close, 5))`, USA/TOP3000/delay 1/decay 4/SUBINDUSTRY/truncation 0.08. BRAIN
ran it on its new default in-sample period, 2014-01-01..2023-12-31.

- `simulation.json`, `alpha.json`, `yearly_stats.json`, `daily_pnl.json`, `self_correlation.json`:
  the same calls as the user-level capture.
- `check.json`: `GET /alphas/MP6POJNn/check`, the submission check. Unlike the checks inside
  `alpha.json` (provisional; LOW_SHARPE is a WARNING there) it gives the final results, including
  the new checks (LOW_2Y_SHARPE, PROD_CORRELATION, DATA_DIVERSITY, ...).
- `check_iv_spread.json`: the same check for the IV-spread pool alpha (alpha 3qoOOb0Q,
  `rs20261103184506r01a0_002_r1x`).
- `correlations_prod.json`: `GET /alphas/MP6POJNn/correlations/prod` (histogram of production
  alphas by correlation).
- `simulation_options.json`: the output of `cli brain options` (`OPTIONS /simulations`): regions,
  universes, delays and neutralizations allowed for the account. `schemas.REGION_SETTINGS` follows it.

Copied from `captures/consultant_20261110/` with two redactions: `alpha.json` `author` is replaced
with "REDACTED", and the alpha names in `self_correlation.json` are set to null. Nothing else is
changed. Don't hand-edit; replace the files from a new capture if BRAIN's format changes.
