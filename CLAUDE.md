# CLAUDE.md: QIP Alpha Research System

Context for any new Claude session in this folder. Read this first, then `BUILD_SPEC.md` for the full requirements and `alpha_research/REPORT.md` for the detailed build report.

## What this project is

This is a WorldQuant BRAIN alpha research system for the IIT Madras Quant Guild (QIP). It implements the multi-agent architecture from Gunin Midha's (NA24B035) WorldQuant Project Member application, Question 8: Director, Research Loop, Data Scientist, Generator, Critic, Debate, Pre-Filter, BRAIN Simulator, Risk, Diversity, Overfitting Auditor, Repair, Pool, Memory and Strategy Evolution.

- `BUILD_SPEC.md` is the authoritative spec for Phases 1 and 2.
- `alpha_research/` holds the implementation.

## Current status (as of 2026-10-28)

- **Phase 1 and Phase 2: DONE.** 164 tests pass, none fail, none are skipped.
- **Phase 3 (LLM agents): NOT STARTED.** Don't begin it unless the user asks.
- This is not a git repo, so nothing has been committed.

## Commands (run from `alpha_research/`)

```bash
PYTHONPATH=src python3 -m pytest -q                       # full suite (~5s warm)
PYTHONPATH=src python3 -m alpha_research.cli run \
    --candidates tests/fixtures/candidates/hand_written.json \
    --config tests/fixtures/thresholds_test.yaml --data-dir <tmp dir>
PYTHONPATH=src python3 -m alpha_research.cli bandit \
    --candidates tests/fixtures/candidates/hand_written.json   # Phase 2 driver
python3 tests/fixtures/generate_fixtures.py               # regenerate fixtures (deterministic)
```

- The environment is system Python 3.12.4 with pandas, numpy, scipy, pyyaml and pytest already installed. There is no venv.
- The first import after a cold start can take a minute or two (bytecode compilation).
- Running `cli run` with the default `config/thresholds.yaml` exits with code 2 **on purpose**: the holdout is null (OQ-9).
- CLI exit codes: 0 ok, 2 config error, 3 schema error, 4 real client not implemented.

## Hard rules (from BUILD_SPEC section 0, which still apply)

1. No LLM or external API calls. No network code. Only the `RealBrainClient` stub may mention HTTP, and it must raise `NotImplementedError`.
2. Don't invent BRAIN response fields. Use only the fields in spec 3.2. If something is unclear, leave it null and add `TODO(open-question)`.
3. Never hardcode a policy threshold. Thresholds live in `config/thresholds.yaml`. A `null` threshold means the check returns `passed=None` with note `"not configured"`. It never passes or fails silently.
4. No credentials anywhere in the repo.
5. A missing value is `None`, never `0`.
6. Every component needs pytest tests, and a component isn't done until they pass.
7. Use only the plain stack: dataclasses, pandas, numpy, scipy, sqlite3, pyyaml, pytest. No heavy frameworks, and no new dependencies (which is why PnL is stored as CSV, not Parquet).
8. Memory is append-only for candidates, results, decisions and variants. Never add update or delete methods to `MemoryStore`; a test checks for this. `os_is` is the only table that may be upserted.
9. Stop and ask the user about genuinely ambiguous requirements. Don't resolve open questions silently.

## Layout

```
QIP/
  BUILD_SPEC.md                 # requirements (Phases 1-2)
  CLAUDE.md                     # this file
  alpha_research/
    REPORT.md                   # detailed build report
    README.md                   # setup / run / open questions
    pyproject.toml              # pytest pythonpath = src
    config/thresholds.yaml      # real config; OQ thresholds null, holdout null
    config/repair_playbook.yaml # turnover, concentration, self_correlation, sub_universe, drawdown
    src/alpha_research/
      schemas.py        # contracts + normalize_expression + experiment_hash
      config.py         # load/validate thresholds; ConfigError; require_holdout()
      brain_client.py   # BrainClient protocol, FixtureBrainClient, RealBrainClient stub
      simulator.py      # validate -> submit/resume -> poll/retry -> normalize -> notes -> persist; VirtualClock
      metric_validator.py
      data_scientist.py
      evaluator.py      # run_gates, classify, decide, evaluate
      risk.py           # assess() -> (veto, checks, reasons)
      diversity.py      # pnl_matrix, correlations, pool_sharpe, marginal_gain, effective_n, admission_check
      auditor.py        # variant counts, deflated Sharpe, OS/IS
      memory.py         # SQLite MemoryStore
      pool.py           # admitted alphas, PnL CSVs under data/pnl/
      repair.py         # Playbook.next_lever
      orchestrator.py   # Thompson sampling with floor quotas; run_rounds driver
      research_loop.py  # experiment_id, duplicates, lineage, queue, decide_next/route
      pipeline.py       # end-to-end wiring; load_candidate_file
      cli.py            # `run` and `bandit` subcommands
    tests/
      conftest.py                       # shared fixtures: config, make_config(section__key=...), memory, candidate()
      fixtures/thresholds_test.yaml     # TEST-ONLY placeholders: holdout 2024-01-01..2025-12-31, max_drawdown 0.15, weight_concentration_max 0.10
      fixtures/generate_fixtures.py     # writes brain_results/*.json, pnl/*.csv, candidates/hand_written.json
      fixtures/brain_results/           # 28 fixtures incl. screenshot_aggregate.json
      fixtures/candidates/hand_written.json  # 31 candidates with expected_decision / expected_failure_class
      test_*.py                         # one per module + test_pipeline.py acceptance test
```

## Key behaviours and decisions (don't silently change these)

**Confirmed by the user:**
- **Failure beats pending.** A definitely failed hard gate decides the outcome even when another gate is `None`, and so does a risk veto. Only when nothing failed does a `None` hard gate give `pending`.
- **Invalid candidates.** A domain error in one candidate (unknown region, unbalanced parens, bad delay, holdout overlap) becomes a `status="invalid"` result and then `reject / incomplete`, and the run continues. A structural error in the candidate file (bad JSON, missing keys, wrong types) raises `SchemaError`, and the CLI exits with code 3.
- **Holdout.** The main config keeps the holdout null; tests and the CLI demo use `tests/fixtures/thresholds_test.yaml`.
- **Screenshot yearly rows**, as supplied by the user (stored in percent in the fixture):
  2019: 1.40 / 23.05% / 0.73 / 6.21% / 3.33% / 5.39bps / 1437 / 1273
  2020: 0.11 / 26.99% / 0.02 / 0.80% / 7.04% / 0.59bps / 1387 / 1329
  2021: 1.83 / 27.72% / 1.28 / 13.61% / 4.28% / 9.82bps / 1491 / 1365
  2022: 3.59 / 25.64% / 3.79 / 28.51% / 5.11% / 22.24bps / 1654 / 1318
  2023: 1.01 / 25.59% / 0.49 / 5.95% / 4.78% / 4.65bps / 1576 / 1393
  The aggregate is sharpe 1.68, fitness 1.11, turnover 25.78%, returns 11.19%, drawdown 7.04%, margin 8.68bps, sub-universe 1.07 against a cutoff of 0.73.

**Builder choices:**

*Decisions and pipeline flow*
- Decision precedence: status not complete → reject/incomplete; hard-gate fail → repair if the playbook has the class, else reject; risk veto → reject; any None → pending; else accept, then diversity.
- Failure priority: self_correlation, turnover, sharpe, fitness, sub_universe, concentration, drawdown, leg_imbalance, weak_year, other. `top_pnl_removal` maps to `other`, and pool correlation maps to `self_correlation`.
- A risk veto always rejects, even when the playbook has levers for that class.
- If the diversity admission fails, the decision becomes `reject / self_correlation`. If it can't be evaluated (no PnL, or less than 60 days of overlap), the decision becomes `pending`.
- Duplicates are skipped before simulation and reported as `"duplicate"`, which is outside the DecisionRecord enum. A pending candidate is routed back to the queue but not re-run in the same pass.

*Simulator and fixtures*
- Variants are registered for **every** simulated candidate (spec 10.1), not just accepts (14.1's wording). Invalid and duplicate candidates are not registered.
- Retries: `TimeoutError` and `ConnectionError` are retried with backoff of 1, 2, 4… seconds. When retries run out, a TimeoutError gives `timeout` and a ConnectionError gives `failed`.
- Fixture runs use `VirtualClock`, so nothing really sleeps. Simulator and Pipeline take `sleep` and `clock` arguments.
- The fixture format is ours, not BRAIN's (TODO OQ-2). Turnover, returns and drawdown are in percent and become fractions on ingest. Margin is in bps. Weight concentration is stored as given. A fixture may have a `responses` list (the last item repeats), a `{"raise": "TimeoutError"}` entry, or a `daily_pnl_file` path relative to `tests/fixtures/`.

*Schema and storage additions*
- `CandidateRecord.generation` and `CandidateRecord.lever` were added for lineage and repair. `ResultRecord.notes` was added for validator notes.
- The SQLite candidates table has extra `expression_hash`, `family` and `parent_id` columns.
- Experiment hash = sha256 of the normalized expression plus region, universe, delay, decay, neutralization, truncation and dates. Normalizing removes all whitespace and lowercases operator names (an identifier followed by `(`). Datafield case is kept.

*Other components*
- Data Scientist: redundancy uses |rank corr| > 0.9. A NaN pattern counts as clustered when its variance ratio is above 4× a random mask.
- Orchestrator: each arm's floor is `floor(floor_share*budget/n_arms)`; remaining slots go one at a time by re-sampling Beta posteriors. The bandit driver defaults to budget 50 and skips arms with an invalid region.
- Allowed universes are only TOP3000, TOP1000, TOP500 and TOP200 (TODO to extend).

## Open questions (owner must answer, never guess)

| ID | Topic | Where it shows up |
|---|---|---|
| OQ-1 | Margin definition (BRAIN reports about half of naive) | `metric_validator.margin_check`, recorded but not gated |
| OQ-2 | Real BRAIN response sample | `RealBrainClient` stub; `simulator._AGGREGATE_MAP` key names; weight_concentration unit |
| OQ-3 | Concurrent sim limit | `simulator.max_concurrent` (no threading yet) |
| OQ-4 | Max drawdown and weight concentration limits | `risk.max_drawdown`, `risk.weight_concentration_max` |
| OQ-5 | Leg balance ratio and min Sharpe after top-PnL removal | `risk.*` |
| OQ-6 | Weak-year floor and max yearly spread | `risk.*` |
| OQ-7 | Min marginal pool Sharpe gain | `diversity.min_marginal_pool_sharpe_gain` |
| OQ-8 | OS/IS ratio floor | `auditor.os_is_ratio_floor` |
| OQ-9 | Holdout dates | `holdout.*`; the simulator raises ConfigError until set |

Two more TODOs exist: the full universe list (`schemas.py`) and the deflated-Sharpe citation (`auditor.py`).

## Working conventions

- Match the existing style: plain dataclasses, small pure functions, module docstrings that state the method used, and `GateResult(name, passed, value, threshold, note)` for every check.
- When you add or change behaviour, update the tests, `README.md`, `REPORT.md` and this file.
- If you change fixtures, edit `generate_fixtures.py` and rerun it. Don't hand-edit the generated JSON or CSV. Keep each candidate's `expected_decision` and `expected_failure_class` in sync, because `test_pipeline.py::test_pipeline_acceptance` asserts all 31.
- Before saying a change is done, run the full suite and the definition-of-done greps:
  `grep -rniE "requests|http|anthropic|openai" src tests config --exclude-dir=pnl` (expect only the stub comments) and
  `grep -rniE "password|token|api_key" src tests config` (expect nothing).
- Runtime data (`data/`, `memory.sqlite`, `data/pnl/`) is git-ignored and must never be committed. Use tmp dirs in tests.

## Natural next steps (only when the user asks)

1. Owner sets OQ-9 (holdout) and the OQ-4 limits in `config/thresholds.yaml`.
2. Owner supplies a raw BRAIN response (OQ-2); then implement `RealBrainClient` and remap fields in `simulator.py`.
3. Phase 3: LLM agents (Generator, Critic, Debate judge, Cheap Pre-Filter, Strategy Evolution) built on the existing contracts, and repair levers that actually rewrite expressions.
