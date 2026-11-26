# Build Report: Alpha Research System, Phases 1 and 2

**Status:** Phases 1 and 2 are complete, and so is Phase 3a (the Alpha Generator, section 9). 284 tests pass (164 from Phases 1 and 2, 120 from Phase 3a), none fail and none are skipped. The rest of Phase 3 (critic, debate, pre-filter, strategy evolution, repair levers that rewrite expressions) has not started.
**Source of truth for the requirements:** [`../BUILD_SPEC.md`](../BUILD_SPEC.md)
**Architecture this implements:** the multi-agent design in the WorldQuant Project Member application (Gunin Midha, NA24B035), Question 8 and the architecture diagram.

---

## 1. What this is, in one paragraph

The application proposes a team of agents that generate alpha ideas, criticise them, simulate them on BRAIN, review the results and learn from them. Several of those agents can be pure rules rather than LLMs: the evaluator, risk manager, diversity manager, overfitting auditor, memory, repair playbook, research director and research loop. This build implements all of those as deterministic Python. It also builds the simulator wrapper that will later talk to BRAIN. For now that wrapper runs against recorded JSON fixtures, not the real platform. The result is a pipeline you can test end to end. It takes candidate alphas, gets a "simulation" result for each, applies the submission gates and risk checks, checks each candidate against the existing pool, decides accept, repair, reject or pending, and stores everything in an append-only database. When the LLM agents are added in Phase 3, they will plug into this core rather than replace it.

What is deliberately **not** here:
- No LLM calls and no network calls of any kind.
- No connection to the real BRAIN platform. `RealBrainClient` is a stub that raises `NotImplementedError`.
- No repair levers applied to expressions. The playbook only names the next lever to try.
- No made-up policy thresholds. Anything the owner hasn't decided is `null` in config and reported as "not configured".

---

## 2. Mapping to the architecture diagram

| Agent / box in the diagram | Module | Status |
|---|---|---|
| Research Director (Orchestrator) | `orchestrator.py` | Built: Thompson sampling with a minimum number of slots (floor quota) per arm |
| Research Loop Agent | `research_loop.py` | Built: experiment IDs, duplicate check, lineage, FIFO queue, routing |
| Datafield Analysis Agent | `data_scientist.py` | Built: coverage, update gaps, tails, NaN pattern, redundancy |
| Alpha Generator Agent | `generator.py`, `fastexpr.py`, `llm.py` | Built (Phase 3a); see section 9 |
| Critic / Debate Agents, Cheap Pre-Filter | none | Phase 3 |
| BRAIN Simulator | `brain_client.py`, `simulator.py` | Built against fixtures; real client is a stub |
| Quantitative Evaluation | `evaluator.py`, `metric_validator.py` | Built |
| Risk Assessment Agent | `risk.py` | Built |
| Diversity Agent | `diversity.py` | Built |
| Overfitting Auditor | `auditor.py` | Built |
| Accept / Repair / Reject | `evaluator.decide`, `pipeline.py` | Built |
| Repair Agent | `repair.py` + `config/repair_playbook.yaml` | Lever lookup only |
| Alpha Pool | `pool.py` | Built |
| Memory Agent | `memory.py` (SQLite) | Built |
| Strategy Evolution Agent | none | Phase 3 |
| Submission & OS Tracking | `auditor.record_os_is` | Storage built; no OS data source yet |

---

## 3. How one candidate flows through the pipeline

```
candidate JSON
   │
   ▼
1. Duplicate check ─────────── same expression+settings already has a result? → "duplicate", skip
   │
   ▼
2. Simulator wrapper
   ├─ local validation ─────── bad region / brackets / delay / holdout overlap → status "invalid"
   ├─ submit (or resume) ───── brain_sim_id saved BEFORE polling (crash-safe)
   ├─ poll with retries ────── honours retry_after, exponential backoff, max_poll_seconds
   ├─ normalize ────────────── percents → fractions, missing → None (never 0)
   ├─ metric validator ─────── fitness / yearly / margin notes (never fails)
   └─ persist result
   │
   ▼
3. Auditor registers a variant (every simulated candidate, including failures)
   │
   ▼
4. Evaluator hard gates: sharpe > 1.25, fitness > 1.0, 1% ≤ turnover ≤ 70%,
                         self_correlation < 0.70, sub-universe sharpe ≥ cutoff
   │
   ▼
5. Risk manager (only if status complete): drawdown, timing, leg balance,
                                           top-PnL-day removal, concentration, weak year
   │
   ▼
6. Decision ── accept → Diversity admission vs pool (max corr ≤ 0.70)
   │               fail → reject / self_correlation; not evaluable → pending
   ▼
7. Decision stored; accepted alphas admitted to the pool with their daily PnL
   │
   ▼
8. One line printed: candidate_id | decision | failure_class | sharpe | fitness | turnover
```

### Decision rules

The order of precedence is fixed:

1. Result status is not `complete` → **reject**, class `incomplete`.
2. Any hard gate definitely failed → **repair** if the playbook has levers for that failure class, otherwise **reject**.
3. Risk veto → **reject**, with the first failed risk item as the class.
4. Any hard gate cannot be evaluated (e.g. self-correlation not computed yet) → **pending**.
5. Otherwise → **accept**, then the diversity admission check.

A definite failure always beats "not evaluable". This was an owner decision: a candidate that already fails on Sharpe shouldn't sit waiting for its self-correlation.

When several things fail, the failure class is the first in this order: self_correlation → turnover → sharpe → fitness → sub_universe → concentration → drawdown → leg_imbalance → weak_year → other.

The playbook has entries for turnover, concentration, self_correlation, sub_universe and drawdown, so a Sharpe failure is rejected while a turnover failure goes to repair. Risk vetoes always reject, even when the class (e.g. concentration) has levers. That follows spec 7.3.

---

## 4. Component by component

**`schemas.py`: data contracts.** `CandidateRecord`, `ResultRecord`, `YearRow`, `GateResult`, `DecisionRecord`. Parsing happens in two steps. `from_dict` checks structure (keys present, types, dates) and a failure there is a broken file. `validate()` checks domain rules (allowed regions and neutralizations, delay 0 or 1, truncation in (0, 1], dates ordered), and a failure there makes only that candidate invalid. This module also holds the expression normalizer and the experiment hash.

**`config.py`: thresholds.** Loads `config/thresholds.yaml` and checks every key, type and range. Missing keys raise `ConfigError`; `null` values are allowed and mean "not configured". `require_holdout()` blocks any simulation until the holdout period is set (OQ-9).

**`brain_client.py` + `simulator.py`: backtester wrapper.**
- `FixtureBrainClient` replays JSON files. A fixture can hold a sequence of poll responses (`running`, then `complete`), a `retry_after` value, or injected `TimeoutError` / `ConnectionError`, which is how the retry and timeout paths are tested.
- Time is injected, so fixture runs use a `VirtualClock` and never actually sleep.
- Crash recovery works like this: the candidate and its `brain_sim_id` are written to `pending_submissions` before polling starts. On the next run the simulator finds the pending entry and resumes polling instead of resubmitting.
- When retries run out, a `TimeoutError` gives status `timeout` and a `ConnectionError` gives `failed`.

**`metric_validator.py`: tripwires.** These checks only produce notes; they never block a candidate.
- **Fitness:** checks `sharpe·√(|returns| / max(turnover, 0.125))` against the reported fitness, within 0.01, for the aggregate and for every year. It reproduces the screenshot exactly: 1.68 → 1.11, and 2022's 3.59 → 3.79.
- **Yearly consistency:** checks that the aggregate Sharpe lies inside the range of the yearly Sharpes.
- **Margin:** records BRAIN's margin next to the naive `returns / (turnover·252)` value and flags any ratio outside [0.8, 1.25]. On the screenshot, BRAIN reports 8.68 bps against a naive 17.22 bps, which is the open question OQ-1.

**`data_scientist.py`: datafield audit.** Works on the last 252 days of one datafield and returns:
- coverage (median and minimum across days)
- median update gap
- fraction of values that are extreme outliers (beyond 3σ)
- NaN pattern: `random`, clustered by date, or clustered by instrument. Each axis is compared with the variance a random mask would give, with a 4× cut-off.
- redundancy: fields whose |rank correlation| with this one is above 0.9
- a recommendation: `usable`, `usable_with_backfill` or `not_usable`, with reasons

**`evaluator.py`: gates and the decision** (see section 3). Weight concentration and drawdown are recorded as soft checks here, and the risk manager makes the call on them.

**`risk.py`: risk manager.** Six checks: drawdown depth, drawdown timing (informational, against optional stress windows), leg balance (min/max of long and short PnL), Sharpe after removing the top-N days by absolute PnL, weight concentration, and weak year (a floor and a spread). Any failure is a veto. Any `null` threshold makes that check "not evaluated" and lists it in the reasons. Sub-universe is not checked again here; the risk manager points to the evaluator's gate result instead.

**`diversity.py`: diversity manager.**
- `pnl_matrix` aligns the daily PnL series and needs at least 60 overlapping days.
- `candidate_correlations` measures the candidate against each alpha in the pool.
- `pool_sharpe` uses equal weights.
- `marginal_gain` is how much the pool's Sharpe changes when the candidate is added.
- `effective_n` is the effective number of independent alphas, computed from the eigenvalues of the correlation matrix.
- `admission_check` rejects a candidate whose max correlation to the pool is above 0.70.

**`auditor.py`: overfitting auditor.**
- **Variant count:** counts every simulation per family.
- **Deflated Sharpe:** `S − se·E_max(N)`, where `se = √((1 + ½S²)/T)` and `E_max` is the Bailey & López de Prado expected maximum of N normal draws. A test shows the best of 200 zero-edge strategies loses at least 0.5 Sharpe after deflation.
- **OS/IS tracking:** stores the OS/IS Sharpe ratio per alpha and flags a family whose mean ratio falls below the floor (OQ-8).

**`memory.py`: append-only SQLite.** Tables: candidates, results, decisions, variants, os_is and pending_submissions.
- No update or delete methods exist (a test asserts this).
- `os_is` is the only table that can be upserted, which the spec explicitly allows.
- Queries: by failure class, by family, by experiment hash, pending submissions, and the levers tried along a candidate's lineage.
- The database lives at `$ALPHA_RESEARCH_DATA/memory.sqlite`, by default `./data`.

**`pool.py`: alpha pool.** Admits an alpha only if the decision is accept and diversity passed. Each alpha's daily PnL is stored as `data/pnl/<id>.csv`. It also provides the PnL matrix, a count, and a breakdown by family.

**`repair.py`: playbook.** `next_lever(failure_class, tried)` returns the first lever not yet tried on this lineage, one at a time, or `None` when they are exhausted. Levers are descriptions only.

**`orchestrator.py`: research director.** Arms are `(dataset, region, prompt_version)` with a Beta(1, 1) prior. Every arm first gets `floor(floor_share · budget / n_arms)` slots. The remaining slots go one at a time: every arm's posterior is sampled and the slot goes to the highest draw. The random seed is fixed so runs are reproducible.

**`research_loop.py`: research loop.** Normalizes each expression and hashes it into an experiment ID. Normalizing removes whitespace and lowercases operator names, but leaves datafield names as written. It also:
- runs the duplicate check
- assigns lineage (`parent_id`, `generation`)
- holds the FIFO queue
- routes each decision: accept → pool, repair → repair, reject → memory, pending → back on the queue

**`pipeline.py` + `cli.py`: wiring.** `cli run` executes the whole flow. `cli bandit` is the Phase 2 driver.

---

## 5. Test evidence

| Test file | Tests | What it proves |
|---|---|---|
| test_schemas | 22 | contracts, structural vs domain errors, round-trips, hashing |
| test_config | 19 | missing keys/sections, bad types/ranges, holdout enforcement |
| test_simulator | 19 | % → fraction, None not 0, invalid without calling BRAIN, retry_after, timeout, backoff, crash resume, stub |
| test_evaluator | 16 | every case from spec 7.4, gate order, priority, failure beats pending, screenshot accepts |
| test_metric_validator | 13 | all five screenshot years' fitness, margin mismatch, yearly consistency |
| test_orchestrator | 10 | best arm wins, floor holds, sums to budget, reproducible |
| test_diversity | 9 | near-duplicate rejected, independent admitted, N_eff ≈ 1 and ≈ 10, the 60-day rule |
| test_memory | 9 | 50-row failure-class query, crash survival, no update/delete API, lineage |
| test_pipeline | 9 | **acceptance test on all 31 candidates**, CLI exit codes, rerun = all duplicates, bandit driver |
| test_risk | 9 | leg-balance veto, top-PnL removal, all-null gives no veto |
| test_auditor | 8 | N = 1 gives no deflation, best of 200 deflated ≥ 0.5, OS/IS flags |
| test_data_scientist | 6 | dense, sparse, redundant copy, NaNs grouped by date or instrument |
| test_repair | 6 | lever order, exhaustion, unknown class |
| test_research_loop | 6 | whitespace-invariant hash, duplicates, generations, routing |
| test_pool | 3 | admission rules, PnL matrix |
| **Total** | **164** | 164 passed, 0 failed, 0 skipped |

### The hand-written candidate set (31 candidates)

| Group | Count | Expected outcome |
|---|---|---|
| Accepts | 5 | accept / none (all admitted to the pool) |
| Sharpe failures (one exactly at 1.25) | 5 | reject / sharpe |
| Turnover failures (two below the 1% floor) | 5 | repair / turnover |
| Concentration (2) and drawdown (1) | 3 | reject / concentration or drawdown (risk veto) |
| Sub-universe failures | 3 | repair / sub_universe |
| Self-correlation failures | 3 | repair / self_correlation |
| Pool-correlation (near-copy of acc_01's PnL) | 1 | reject / self_correlation (diversity) |
| Duplicates (whitespace or operator case changed) | 2 | duplicate |
| Invalid (unbalanced brackets, region XYZ) | 2 | reject / incomplete |
| Timeouts (never completes; poll keeps timing out) | 2 | reject / incomplete |

### Phase 2 driver output (10 rounds, budget 50, 4 arms, minimum 2 slots per arm)

The news12/EUR arm has the highest fixture pass rate, and by round 4 it gets most of the slots. The minimum of 2 slots per arm held in every round.

### Definition-of-done checks

- **Network grep** (`requests|http|anthropic|openai`): the only hits are two comments in the `RealBrainClient` stub.
- **Credential grep** (`password|token|api_key`): no hits.

---

## 6. Decisions taken during the build

Confirmed by the owner:
1. The 2019–2023 screenshot yearly rows were supplied and stored in `screenshot_aggregate.json`.
2. The main config's holdout was `null` until the owner chose BRAIN's default on 2026-11-03 (2024-01-01..2026-11-03). The test config uses a placeholder holdout of 2024-01-01 to 2025-12-31, plus placeholder risk limits of drawdown 0.15 and weight concentration 0.10.
3. A definite failure beats pending.
4. A domain error makes only that candidate invalid; a structural file error stops the CLI with exit code 3.

Made by the builder:
- **Fields added to the schemas:**
  - `CandidateRecord.generation` and `CandidateRecord.lever`, needed for lineage and repair
  - `ResultRecord.notes`, for the metric-validator notes
  - `candidates.expression_hash`, `family` and `parent_id` columns in SQLite, for queries
- **Variant counting:** variants are registered for every simulated candidate, following spec 10.1 rather than 14.1's "if accept" wording.
- **Duplicates** are reported as a sixth outcome, `duplicate`, outside the decision enum.
- **Diversity:** a candidate with no PnL, or with less than 60 days of overlap with the pool, becomes pending.
- **Redundancy** uses absolute correlation. The NaN-cluster cut-off is a 4× variance ratio.
- **PnL storage:** stored as CSV, not Parquet, because Parquet would need an extra dependency.
- **Bandit driver:** budget 50; arms from an invalid region are left out.
- **Python:** the code targets 3.11 or newer, and was built and tested on 3.12.4.

---

## 7. Open questions for the owner

| ID | Question | Effect until answered |
|---|---|---|
| OQ-1 | BRAIN's margin definition (it reports about half the naive value) | margin is recorded, not gated |
| OQ-2 | A raw BRAIN response sample | confirmed by a live capture on 2026-11-03 (`tests/fixtures/brain_api_capture/`); only the weight-concentration unit is open (section 10) |
| OQ-3 | Concurrent simulation limit | single-threaded, `max_concurrent: 1` |
| OQ-4 | Max drawdown and weight concentration limits | checks not configured |
| OQ-5 | Leg-balance ratio and Sharpe floor after removing the top PnL days | not configured |
| OQ-6 | Weak-year Sharpe floor and yearly spread | not configured |
| OQ-7 | Minimum marginal pool Sharpe gain | not configured (correlation is still enforced) |
| OQ-8 | OS/IS ratio floor | family flags return None |
| OQ-9 | Holdout dates | done 2026-11-03: 2024-01-01..2026-11-03, after BRAIN's default in-sample period |
| OQ-10 | Operator list in `config/fastexpr.yaml` | done 2026-11-03: matches the account's 66 operators (removed ts_min, ts_max, ts_skewness, exp, regression_neut, vector_neut; added add, subtract, multiply, divide, and, or, not, the six comparison functions, bucket, kth_element, ts_step) |
| OQ-11 | Expression complexity limits | not checked |
| OQ-12 | How the holdout is enforced on BRAIN, which ignores candidate dates | done 2026-11-03: BRAIN's default period ends 2023-12-31; `simulator.holdout_leak` makes any result whose BRAIN period overlaps the holdout `invalid` |
| OQ-13 | Agent builder defaults: near-duplicate similarity 0.9, evolve every 2 rounds | used as shipped in `config/agents.yaml` |
| OQ-14 | Raw data source for the Data Scientist | the Data Scientist is not in the research loop |

---

## 8. Suggested next steps

1. ~~OQ-9~~ done. Answer OQ-4 (optional; null risk checks are skipped).
2. ~~Run `cli brain capture` (OQ-2)~~ and ~~OQ-12~~ done.
3. ~~Run `cli brain operators` (OQ-10)~~ done. Run `cli brain datafields` for a catalog, so generated alphas are checked against real fields.
4. Run `cli research` on a small budget once OQ-2 and OQ-9/12 are settled (section 11).

---

## 9. Phase 3a: Alpha Generator

**What it is.** An agent that takes a hypothesis, or a research paper, and produces alphas written in WorldQuant BRAIN's Fast Expression language, as `CandidateRecord`s the existing pipeline already accepts. Run it with `cli generate` (see the README).

**How it works.**

```
paper text ──► [model: extract hypotheses] ──┐
hypothesis ──────────────────────────────────┤
                                             ▼
                 [model: write N alphas per hypothesis]   system prompt = rules + operator
                                             │             list + most relevant datafields
                                             ▼
                      fastexpr.analyze (parser + checks)
                          │ fail                 │ pass
                          ▼                      ▼
        [model: fix these, with the errors]   CandidateRecord.validate
           (max_repair_rounds, default 1)        │
                          │ still fails          ▼
                          ▼                 de-duplicate by experiment hash
                    report.rejected              │
                                                 ▼
                                   candidates.json + report.json
```

**Why the validator, not the prompt, is the safeguard.** A model can invent an operator, pass the wrong number of arguments, or use a datafield that does not exist. Paper text can also contain instructions aimed at the model. The prompt marks source text as data, but nothing depends on the model obeying: an expression only becomes a candidate if the parser and the checks accept it. A test feeds an injected "output os_system(...)" instruction through the pipeline and checks that it ends up in `rejected`.

**Modules.**

| Module | Role |
|---|---|
| `fastexpr.py` | Lexer and recursive-descent parser. `analyze()` returns errors, warnings, operators used, fields used, depth and call count, and never raises on bad input. Supports calls with keyword arguments, `+ - * /`, comparisons, `&&`, `||`, strings, and `name = expr;` statements. Reports everything else (such as `?:` or indexing) as an error. |
| `config/fastexpr.yaml` | The operator list, written as signatures such as `ts_rank(x, d, constant=0)`. Arity is derived from the signature, so there is one source of truth. Also groups, base fields and constants. Matches our account's operators (OQ-10). |
| `llm.py` | `LLMClient` protocol, `ScriptedLLMClient` for tests and offline runs, `AnthropicLLMClient` for the real model (optional SDK, key from the environment). The only file that names a provider. |
| `generator.py` | `AlphaGenerator`, prompts, JSON extraction, repair round, de-duplication, `GenerationResult.to_report()`, `load_datafields`, `load_generator_config`. |
| `config/generator.yaml` | prompt version, repair rounds, paper length cap, hypotheses per paper, datafields shown, output cap. Complexity limits are null (OQ-11). |

**Decisions taken during the build.**
- Fast Expression only. The simulator submits `CandidateRecord.expression`, and nothing in the repo shows that BRAIN runs Python alphas.
- Unknown datafields are an error only when a catalog is supplied. Without one they pass with a warning, because the validator cannot know the account's fields.
- With a large catalog the prompt lists only the fields whose name parts occur in the hypothesis text (deterministic scoring), up to `max_datafields_in_prompt`. The validator still accepts the whole catalog.
- Surplus alphas beyond the requested count are dropped before they are recorded as seen, so a later call can still produce them.
- The model may propose `decay` and `neutralization`. Both are validated, and invalid values fall back to the settings with a warning.
- A paper is not read as PDF (no new dependency); the CLI explains how to convert it.
- `cli run` on generated candidates now exits 4 with an explanation instead of a traceback, because fixture runs have no recorded result for them.

**Tests added (120).**

| Test file | Tests | What it proves |
|---|---|---|
| test_fastexpr | 49 | spec parsing and arity, every valid hand-written expression passes, syntax and arity errors, unknown operators and fields, variables, limits, deep nesting does not crash |
| test_generator | 48 | hypothesis and paper flows, repair round and what the model sees, rejects, duplicates, surplus handling, JSON in fences or prose, retry on bad JSON, prompt-injection text, catalog behaviour, config loaders |
| test_llm | 7 | scripted client, missing key, missing SDK, request shape, provider errors, empty reply |
| test_cli_generate | 16 | end-to-end files, settings merge, paper and file inputs, catalog filtering, every exit code, `run` on generated candidates |

**Not built, on purpose.** The critic, debate judge, cheap pre-filter, strategy evolution and expression-rewriting repair levers; any BRAIN network code; PDF parsing.

**Checks.** Full suite 284 passed. The definition-of-done greps run with `--exclude='*llm*.py'` find only the two `RealBrainClient` stub comments and no credentials. A search for literal key strings finds none.

---

## 10. Real BRAIN client

**What it is.** `brain_api.py` replaces the `RealBrainClient` stub, at the owner's request. It is the only module that talks to BRAIN, and it uses only the standard library (`urllib`, `http.cookiejar`), so there is no new dependency.

**Status: matches BRAIN's web app and a live capture from our account (2026-11-03).** The API documentation is behind a login. The first version followed the API shape used by community client code. It was then checked against BRAIN's own web app, by reading the public JavaScript bundle of platform.worldquantbrain.com (section 10.1). A live capture from our account then matched every mapped field (sharpe, fitness, fractions, margin in bps, sub-universe, self-correlation from `correlations/self` max, yearly rows, daily PnL as daily values); it is the fixture `tests/fixtures/brain_api_capture/`, and the `brain_mapping_unverified` note was removed.

**Flow.**

```
candidate ──► POST /simulations ──► Location: .../simulations/<id>   (submit)
                                        │
             GET /simulations/<id> ◄────┘  Retry-After → "running"  (poll, simulator sleeps)
                     │ COMPLETE / WARNING            │ ERROR / FAIL
                     ▼                               ▼
   GET /alphas/<alpha>                         status "failed" + message
   GET /alphas/<alpha>/recordsets/yearly-stats
   GET /alphas/<alpha>/recordsets/daily-pnl    (each waits on Retry-After, bounded;
   GET /alphas/<alpha>/recordsets/pnl            pnl only if daily-pnl is missing)
   GET /alphas/<alpha>/correlations/self
                     │
                     ▼
   brain_api.translate ──► simulator.normalize ──► ResultRecord
```

**Field mapping (spec 3.2 fields only).**

| ResultRecord field | From BRAIN | Conversion |
|---|---|---|
| sharpe, fitness | `is.sharpe`, `is.fitness` | none |
| turnover, returns, max_drawdown | `is.turnover`, `is.returns`, `is.drawdown` | already fractions (`units: fraction`) |
| margin_bps | `is.margin` | × 10,000 |
| sub_universe_sharpe / cutoff | `LOW_SUB_UNIVERSE_SHARPE` check value / limit | none |
| weight_concentration | `CONCENTRATED_WEIGHT` check value | unit unknown (OQ-2); a PASS carries no value, so it is None |
| self_correlation | `correlations/self` `max`, else `SELF_CORRELATION` check value | none |
| checks | `is.checks[]` PASS → true, FAIL → false | PENDING / WARNING left out, noted |
| yearly | `yearly-stats` rows with stage `IS` | other stages excluded, noted |
| daily_pnl | `daily-pnl` recordset, else `pnl` | `pnl` is cumulative, so it is differenced |
| long_pnl, short_pnl, drawdown_date | not in BRAIN's response | `None` |
| (notes only) | `settings.testPeriod`, `alpha.test`, `is.startDate` | recorded, not gated |

**Behaviour.**
- Login is HTTP Basic from `BRAIN_EMAIL` / `BRAIN_PASSWORD`, done lazily and redone once on a 401. A biometric-login request is reported as an error, not handled.
- 429 responses wait on `Retry-After` up to `rate_limit_max_wait_seconds`, then become a `ConnectionError`. 5xx and network failures become `ConnectionError` / `TimeoutError`, which the simulator already retries.
- A 4xx on submit raises `SubmissionRejected` (new, in `brain_client.py`); the simulator records a `failed` result with the message instead of crashing.
- A recordset that is missing or does not become ready in `recordset_max_wait_seconds` becomes `None` with a note. It never fails the whole result.
- With `--capture-dir` (or `cli brain capture`) every raw body is written to disk. Credentials and cookies are never written.
- Candidate start and end dates are not sent, because BRAIN has no such setting in the known API. The notes record the period BRAIN used as `brain_period: start..end`, and the simulator makes a result `invalid` if that period overlaps the holdout (OQ-12).
- Nothing submits alphas for competition.

**Other changes.**
- `simulator.normalize` accepts `units: "fraction"` (skips the percent conversion) and copies a raw `notes` list onto the result. Fixtures are unchanged.
- `cli run --client real` now works; it checks the holdout before logging in. New `cli brain auth | capture | operators | datafields`. Exit code 7 is a BRAIN login or API error. Exit code 4 now only means "no fixture for this candidate".
- `config/brain.yaml` holds connection settings and the fixed simulation settings (not policy thresholds). `captures/` is git-ignored.
- Hard rules 1 and 4 now allow network code in `brain_api.py` only, with credentials from the environment only.

**Tests added (58).** `test_brain_api.py` (56) drives everything through a fake transport: BRAIN's own sample responses, config validation, payload, translation with the screenshot numbers, cumulative PnL, check handling, failures, login and re-login, wrong credentials without echoing the secret, biometric error, rate limits, 5xx, urllib error mapping, simulator and pipeline runs on the real client, rejected submissions, missing and slow recordsets, captures without secrets, datafield pagination, operators, and every `brain` CLI command. `test_simulator.py` replaces the stub test with two tests for the new raw keys; `test_pipeline.py` replaces the stub CLI test with two real-client CLI tests.

### 10.1 Checking the API against BRAIN's web app

The web app at platform.worldquantbrain.com is a single-page app whose JavaScript is public. Its main bundle and lazy chunks were downloaded and read (never executed). What was found, and what changed as a result:

| Found in the bundle | Effect on the client |
|---|---|
| Every request sends `Accept: application/json;version=2.0` (some endpoints use 3.0 or 4.0) | `ACCEPT` header added |
| Poll handler: `details` → error; `status` ERROR (with `message`), TIMEOUT, FAIL → error; `detail` → error; `progress >= 0` → running; `children` → multi-simulation; WARNING → done with a message; otherwise done, alpha id in `alpha` | `poll` and `translate` follow the same order; TIMEOUT added; `progress` without Retry-After counts as running |
| A rejected POST returns `{"settings": {field: [messages]}}`; 403, 429 and 503 are handled separately | rejection messages are flattened into the result note |
| Self-correlation is read as `correlations/self` → `max` / `min` | confirmed |
| `OPTIONS /simulations` gives `actions.POST.settings.children` plus per-type settings under `type.choices` | `simulation_options()` and `cli brain options` |
| The editor walkthrough mocks the API with sample responses for a simulation, an alpha, the recordset list, yearly stats and PnL | saved under `tests/fixtures/brain_api_samples/` (see SOURCE.md); tests translate them |
| Recordsets: `pnl` (cumulative; its last value is within 0.02% of `is.pnl`), `daily-pnl`, `sharpe`, `turnover`, `yearly-stats`; schema types `percent` (fractions) and `permyriad` (margin) | `daily-pnl` preferred, `pnl` differenced as a fallback; the `pnl_recordset_is_cumulative` flag was removed |
| Alpha settings include `testPeriod` (e.g. `P1Y0M0D`), and the alpha has a `test` stats block | recorded in notes; offered as the way to settle OQ-12 |
| Checks include `PROD_CORRELATION`, `REGULAR_SUBMISSION`, `MATCHES_COMPETITION`, `IS_LADDER_SHARPE` | kept in `checks` when PASS or FAIL; not used by any gate |
| `CONCENTRATED_WEIGHT` sample: value 0.2, limit 1 | the unit is still unclear (OQ-2) |

**Checks.** Full suite 342 passed. The definition-of-done greps, run with `--exclude='*llm*.py' --exclude='*brain_api*.py'`, find nothing.

---

## 11. Remaining agents and the research loop

**What was built.** The agents that do not need owner input, and the loop that connects all of them. `cli research` runs it.

| Module | Agent | Model use | Without the model |
|---|---|---|---|
| `prefilter.py` | Cheap Pre-Filter | optional batch score 0-10 | validator, exact duplicate, near duplicate (term-set Jaccard) |
| `critic.py` | Critic | one batch review: simulate / revise / drop, optional revised expression | rule checks: non-positive window blocks, missing normalization and very long windows are flagged |
| `critic.py` | Debate judge | rules on contested candidates, weighing the Generator's rationale against the Critic | the Critic's verdict stands |
| `repair_agent.py` | Repair Agent | levers with no mechanical form (orthogonalize, liquid fields, winsorize) | seven mechanical levers rewrite the parsed expression or the settings |
| `strategy.py` | Strategy Evolution | guidance and new hypotheses from a memory summary | guidance from failure counts |
| `research.py` | Research loop | none itself | all of the above wired together |

**How they connect.** The Director (existing Thompson-sampling orchestrator, now with `add_arm`) splits each round's slots across arms. Queued repairs take an arm's slots first, and the Generator fills the rest. Candidates then pass through the Pre-Filter, the Critic and the Debate judge, and the survivors go through the existing pipeline. Each accept, reject or repair updates the Director. Each repair decision goes to the Repair Agent, and its child is queued for the same arm. Every N rounds Strategy Evolution creates a new prompt version with its guidance and hypotheses, added as new arms, so the bandit tests whether the guidance helps. The Director warm-starts from memory.

**Safety properties.**
- Nothing a model writes reaches BRAIN without passing `fastexpr.analyze`. That includes critic revisions and model repairs.
- A blocking rule issue (a look-ahead window) can be overruled by neither the Critic model nor the judge.
- Candidate and memory text goes to every model inside tags marked as data.
- Budgets: `--max-simulations` and `--max-llm-calls` (all agents together, through `BudgetedLLM`) are required flags. A model failure or a used-up budget stops the run cleanly with a report.
- None of the new settings changes an accept or reject decision. Those stay with `config/thresholds.yaml`.
- Nothing submits alphas for competition.

**Supporting changes.**
- `fastexpr.render` turns a parsed expression back into text (round-trip tested on every hand-written expression).
- `generator.ask_json` is the shared JSON request used by every agent.
- `AlphaGenerator(guidance=...)` adds Strategy Evolution's guidance to the system prompt.
- `MemoryStore.candidate_outcomes()` is a read-only query.
- `Orchestrator.add_arm`.
- `config/agents.yaml` and `config/research_plan.example.yaml`.
- The playbook comments now describe how levers are applied.

**Decisions taken during the build.**
- Repairs are exempt from the near-duplicate check: they are deliberate variants, and the auditor counts them.
- A critic revision is a child (`<id>_c1`, lever `critic_revision`). The original is not simulated or stored.
- Slots emptied by the Pre-Filter or Critic are not refilled within the round. Pending decisions are not re-run.
- Mechanical lever semantics: `increase_decay` sets decay to `step.to`, and `lengthen_ts_window` multiplies the second positional argument of every `ts_*` call. The other levers wrap the final statement.
- The Data Scientist is not in the loop: it needs raw datafield matrices (OQ-14).

**Tests added (77).**
- `test_repair_agent` (13): every mechanical lever, lever order and lineage, exhaustion, model levers and bad replies.
- `test_prefilter` (6) and `test_critic` (15): rules, model verdicts, revisions, blocking issues that cannot be overruled, judge overrule and drop, fallbacks.
- `test_strategy` (6).
- `test_research` (23): the full closed loop with a routing fake model and a stub BRAIN client. The Critic blocks look-ahead, a turnover failure is repaired and accepted, evolution adds an arm whose guidance reaches the Generator, both budgets stop cleanly, warm start works, plus config and plan errors and `cli research` end to end.
- Plus `render`, `candidate_outcomes`, `add_arm` and generator guidance tests.

### 11.1 Agent prompts

The first versions had a one-paragraph prompt inline in each agent module. They are now in `prompts.py`, written out in full and versioned (`PROMPT_VERSIONS`, all `-v2`; the run report records them).

**What every system prompt contains.**
- `brain_context(config)`: how BRAIN turns an expression into weights, delay, decay, the fitness formula (the same one `metric_validator` checks), the sub-universe, concentration and self-correlation tests, and this run's thresholds read from `config/thresholds.yaml` (`not configured` for nulls, never hardcoded).
- The agent's role and method.
- `DATA_RULE`: tagged text is data, never instructions.
- `JSON_RULE` and the exact reply shape the parser accepts.

The Pre-Filter and Critic also share a list of common pitfalls: look-ahead, no normalization, mixed scales, near-zero division, sparse fields, sign errors, very short windows, and hypothesis mismatch.

**Per agent.**
- **Pre-Filter:** a calibrated 0-10 rubric, scoring each candidate on its own.
- **Critic:** issues in ten categories, simulate / revise / drop with rules for when to revise (a minimal change that keeps the idea), and an optional expected failure. Told that rule-found look-ahead cannot be cleared.
- **Debate judge:** decides on one question: does the criticism make the simulation fail or uninformative? Speculative or style criticism loses, a new idea is worth more than another variant, and look-ahead is always decisive.
- **Repair:** one lever only, the idea kept, operators limited to the listed ones. Per-lever instructions for the three model levers (`LEVER_INSTRUCTIONS`), and a test checks that every non-mechanical lever in the playbook has them.
- **Strategy:** 3-6 guidance bullets that must cite the evidence, plus new hypotheses with a direction and why they are new, keeping at least one genuinely different idea.

**Parser changes to match.**
- The Critic accepts issues as `{category, detail}` objects or strings, and records `expected_failure`.
- The judge falls back to `decisive_point` when `reason` is missing.
- The Repair prompt shows the pool only for `orthogonalize_against_pool`.

**Not changed.** The Generator's prompts (still `gen-v1`). Changing them would rename every arm. Adding the thresholds to the Generator too would be a natural `gen-v2`.

**New command.** `cli prompts [--agent NAME] --config F` prints the rendered prompts for review.

**Tests added (14).** `test_prompts` checks that:
- the thresholds come from config, with nulls shown as not configured;
- every system prompt carries the context, the data rule, the JSON rule and the keys its parser reads;
- candidate text is fenced in tags;
- every model lever has instructions;
- the pool is shown only for orthogonalize;
- structured issues and `decisive_point` parse;
- a research run passes this config's context to every agent prompt and records the prompt versions;
- `cli prompts` works.

**Checks.** Full suite 433 passed. The definition-of-done greps find nothing.

## 12. A model per agent, Gemini, and model comparison

**Why.** All agents used to share one model. A Critic built on the same model as the Generator tends to miss the same mistakes, and the owner wanted to compare models agent by agent. The owner chose: Anthropic plus Google Gemini, generator models compared as bandit arms, and critic models compared as shadows.

**What was built.**
- `llm.py`: `GeminiLLMClient` calls Gemini's `generateContent` REST endpoint with `urllib` (no new dependency; the key comes from `GEMINI_API_KEY`, sent only in the `x-goog-api-key` header). `provider_of(model)` infers the provider from the id (`claude-*`, `gemini-*`), and `make_real_client` picks the client. Thought parts are skipped. An HTTP error, a blocked prompt or an empty reply raises `LLMUnavailable`; the empty-reply message says to raise `max_output` when thinking used the whole output cap.
- `config/agents.yaml` has a new `models` section: `default`, `generator` (list), `critic` (list, in preference order), `shadow_critics`, and `prefilter` / `debate` / `repair` / `strategy` (null means the default). `--model` replaces the default only.
- `research.py`:
  - Arms are now `(dataset, region, prompt_version, generator model)`. Each generator-model arm gets its own copy of the plan's sources, so the models see the same hypotheses in the same order.
  - The primary critic for an arm is the first listed critic from a different provider. `check_models()` refuses a config where a generator has no such critic, or a model id has an unknown provider.
  - Shadow critics review the same batch after the primary; they never change what is simulated.
  - One `CallCounter` is shared by every model's `BudgetedLLM`, so `--max-llm-calls` still covers everything, and the report adds `model_calls_by_model`.
- `memory.py`: an append-only `critic_reviews` table (`put_critic_review`, `critic_reviews()`) holds every verdict, primary and shadow, with the arm's generator model.
- `compare.py`: per-generator-model accept rates, and per-critic-model verdict counts, rule fallbacks, agreement with the primary, cleared-accept rate and flag precision. Every candidate is credited to its arm's generator model through the critic records and lineage, so repair children and critic revisions count for the arm they came from. The research report has a `comparison` section, and `cli compare` prints the same tables across all runs in memory.

**Decisions taken during the build.**
- The provider is read from the model id prefix, so config files never name a provider and the definition-of-done grep stays clean.
- Critics are scored only on candidates whose own expression was simulated and decided. Pending decisions are left out. A primary's drops never reach BRAIN, so its false drops show up only through shadows.
- The Debate judge, Repair, Pre-Filter and Strategy agents each take one model. No provider rule is enforced on them.
- The Gemini model id in the config (`gemini-3.1-pro-preview`) comes from Google's documentation as of 2026-11-03, which listed several current ids. Check it before a real run.

**Tests added (28).**
- `test_llm` (10): provider inference, the Gemini request shape, thought parts skipped, six error cases, client selection.
- `test_research` (11): two generator models become two arms with other-provider primary critics and shadows recorded; shadows don't change what is simulated; warm start credits each generator arm; `--model` replaces only the default; `used_models`; six config errors including the same-provider critic.
- `test_compare` (6): arm attribution through lineage and critic revisions, both comparison tables, rule fallbacks, `cli compare`.
- `test_memory` (1): `critic_reviews` is append-only and reads back.

**Checks.** Full suite 461 passed (470 after the operator and universe update from our account on 2026-11-03). The definition-of-done greps find nothing.

### Free models and Qwen (2026-11-03)

The owner asked to use free API keys only, and to add Qwen.
- First real Gemini call: `cli generate --model gemini-3.6-flash` (free tier) returned 3 valid candidates from 1 model call, none rejected.
- Free availability on 2026-11-03: Gemini Flash models are free (Gemini 3.1 Pro is paid-only); Claude has no free tier; OpenRouter's live model list had 14 free models but no Qwen; Groq's free plan includes `qwen/qwen3.8-27b`.
- `llm.ChatCompletionsLLMClient` calls a hosted chat-completions endpoint with `urllib`. `groq:<maker>/<model>` goes to Groq (`GROQ_API_KEY`), a plain `<maker>/<model>` to OpenRouter (`OPENROUTER_API_KEY`). The provider is the maker (`qwen`), not the host, so the cross-provider critic rule still means "a different model family". Hosted ids are routed first, so `google/gemini-x` goes to OpenRouter, not Google's API. Inline `<think>` blocks are stripped, since `extract_json` would otherwise pick up a JSON draft from the thinking. An HTTP 200 carrying an `error` object is treated as a failure. The first real Groq call failed with `HTTP 403` and no message: Cloudflare in front of Groq refuses urllib's default `User-Agent` (`error code: 1010`, reproduced without a key). Hosted calls now send `User-Agent: qip-alpha-research/0.1`, and a non-JSON error body is shown in the message. After the fix, `cli generate --model groq:qwen/qwen3.8-27b` returned 3 valid candidates from 1 call, none rejected.
- `config/agents.yaml`: generators `gemini-3.6-flash` and `groq:qwen/qwen3.8-27b`, critics in the opposite order, shadow critics off (to save the free daily quota), strategy and default `gemini-3.6-flash`. Claude can be added back by listing a `claude-*` id.
- Tests: hosted provider inference, request shape for both hosts, missing keys, seven error cases, thinking stripped, client selection. 487 passed; greps clean.

### First closed research run (2026-11-03)

Run `rs20261103171304`: `cli research` with the example plan, `--max-simulations 6 --max-llm-calls 20 --per-round 3 --rounds 2`, free models only.
- Round 1 gave 3 slots across 4 arms (no history, so the split was random). Gemini generated 2 candidates and Qwen, as primary critic, cleared both. BRAIN simulated both.
- `group_neutralize(-ts_sum(returns, 5) / (ts_std_dev(returns, 5) + 0.001), subindustry)`: Sharpe 0.84, fitness 0.39, turnover 46.6%, self-correlation 0.73, so repair / self_correlation (Sharpe and fitness also failed). The Repair Agent's model lever wrote an orthogonalized child, queued for the next round.
- `group_rank(ts_delta(cash_flow_from_operations / max(assets, 1), 252), subindustry)`: Sharpe -0.76, so reject / sharpe. Qwen had noted "expected to fail: sharpe" but did not block it.
- The run stopped on Groq's free limit of 7,000 input tokens per minute: the generator prompt is about 3,350 tokens, about 2,400 of them the 300 listed datafields. 2 simulations and 7 model calls used.
- Built after this run: the Gemini and hosted clients now wait and retry on HTTP 429 when the host says how long (`try again in 7.954s`, `1m30s`, `450ms`, or Gemini's `retryDelay`), if the wait is at most 60 s, up to 3 times, logging each wait (`llm.rate_limit_wait`, `_post_json`). A longer wait (a daily quota) or a 429 with no wait still stops the run. Retries are inside one `complete` call, so the budget counts one call. Tests: parsing, retry then success for Groq and Gemini, no sleep for long or unknown waits, retries running out.
- Still proposed, not built: ~~pause a rate-limited model instead of stopping the run~~ (done 2026-11-04, model failover); `max_datafields_in_prompt` 300 to about 80; more free hosts (Mistral; NVIDIA NIM as a second Qwen host). Owner question: should a candidate that also fails Sharpe or fitness still go to repair for a higher-priority failure class?
- `SETUP.md` at the repo root explains how teammates set the system up on their own laptops.

### Second research run (2026-11-03)

Run `rs20261103175655`, same command, after the 429 wait-and-retry. It reached the simulation budget (6 simulations, 10 model calls: Gemini 6, Qwen 4) without stopping.
- Round 1 (one slot each to three arms): two reversal alphas (Sharpe 0.51 reject / sharpe; Sharpe 0.88 with self-correlation 0.80, repair / self_correlation) and one 6-month cash-flow change (Sharpe 0.25, reject).
- Round 2: the Director gave all three slots to the only untried arm (fundamental6, Qwen); the floor rounds to 0 at 3 slots over 4 arms, so the queued repair in another arm never ran.
- Best candidate: `rank(ts_corr(returns, cash_flow_from_operations / assets, 21)) * sign(cash_flow_from_operations / assets)`, Sharpe 1.14, fitness 1.36, sub-universe 0.12 against 0.49, drawdown 30%. Gemini's critic proposed wrapping it in `rank`; the Debate judge kept the original. Rulings are not in the report.
- `rank(ts_delta(cash_flow_from_operations, 21) / assets)` with neutralization NONE: returns 38%, drawdown 67%, fitness 1.45 above Sharpe 0.83, which points to a market or concentration bet. It was rejected for Sharpe; with `risk.max_drawdown` null (OQ-4) the drawdown itself was not checked.
- All hypotheses came from the example plan, so every idea was a reversal or cash-flow variant.

### Research plan v1 and third run (2026-11-04)

`config/research_plan.v1.yaml` replaces the example plan's reversal and cash-flow ideas with less crowded data and documented signals: options (call-put implied-volatility spread, rising put fear, put/call volume), analyst estimates (EPS revisions, forward earnings yield) and risk/links (idiosyncratic volatility, customer momentum, earnings-news sentiment). Every field the hypotheses name exists in our catalog; neutralization is SUBINDUSTRY or INDUSTRY.

Run `rs20261103184506` (`--max-simulations 18 --max-llm-calls 60 --per-round 6 --rounds 3`) stopped after 2 simulations when Gemini returned HTTP 503 ("high demand"), which the 429 retry does not cover. Both simulated alphas came from the IV-spread hypothesis:
- `spread = divide(subtract(implied_volatility_call_60, implied_volatility_put_60), add(implied_volatility_mean_60, 0.001)); group_zscore(ts_decay_linear(spread, 10), subindustry)`: Sharpe 2.20, fitness 1.43, turnover 21.5%, self-correlation 0.31, drawdown 4.2%, yearly Sharpe 0.95 / 1.91 / 3.19 / 2.98 / 1.55. Failed sub-universe (0.55 against 0.95) and BRAIN's CONCENTRATED_WEIGHT check (0.1538). Decision repair / sub_universe; the repair call hit the 503, so no child was queued.
- `group_rank(ts_mean(implied_volatility_call_30 - implied_volatility_put_30, 5), subindustry)`: Sharpe 1.64, fitness 0.73, sub-universe 0.37 against 0.71. Reject / fitness.

Paused to-do, in order: retry HTTP 500/502/503/504 with growing waits; rescue the first alpha by hand with `cli run --client real` (backfilled IV fields and `rank` for concentration; a liquidity tilt or TOP1000 for sub-universe); set `risk.weight_concentration_max`.

### Why the close alphas were not repaired, and the learning Repair Agent (2026-11-04)

Diagnosis from memory (10 real simulations over three runs). No repair child ever reached BRAIN:
- Run 1: the reversal alpha's orthogonalized child was queued, then the run stopped on Groq's rate limit. The queue lived in memory only, so the child was lost.
- Run 2: a child was queued, but the Director gave every round-2 slot to another arm (floors round to 0 at 3 slots over 4 arms), so it never ran.
- Run 3: the IV-spread alpha (Sharpe 2.20) went to repair / sub_universe. That class had one lever, a model lever (`restrict_to_liquid_fields`), and the model call hit Gemini's 503.
- The other two near misses never reached the Repair Agent. The playbook had no `sharpe` or `fitness` entry, so Qwen's cash-flow alpha (Sharpe 1.14 against 1.25) and the 30-day IV sibling (fitness 0.73 against 1.0) were rejected outright.
- Concentration was invisible: BRAIN failed CONCENTRATED_WEIGHT (0.154) on the IV alpha, but `risk.weight_concentration_max` is null (OQ-4), and the agent only looked at the single highest-priority class.
- Nothing learned: the lever order was fixed, and no lever's past results were used.

Built:
- Model clients retry HTTP 500/502/503/504 after 10 s, 30 s and 60 s (`llm.SERVER_ERROR_WAITS_SECONDS`).
- `repair.near_miss_share` (thresholds.yaml, OQ-15): Sharpe and fitness failures go to repair when |value| >= 0.6 / 0.5 of the threshold. Three fixture candidates (shp_01, shp_02, shp_04) moved from reject to repair.
- Six new mechanical levers: `flip_sign`, `set_neutralization`, `backfill_fields`, `liquidity_tilt`, `winsorize` (now mechanical) and `change_universe`. There is also a `model_repair` lever at the end of every class. The playbook now covers sharpe, fitness, turnover, concentration, self_correlation, sub_universe and drawdown with 4 to 6 levers each.
- `lever_stats.py`: lever evidence from parent -> child pairs in memory (reward 1 fixed / 0.5 improved / 0, minus 0.5 for collateral failures), a Beta posterior per (class, lever), and a ranking that adds weight for the alpha's other failing classes, including BRAIN's own failed checks. Past successful repairs go into the repair prompt (`repair-v4`).
- Research loop: queued repairs take up to half of each round's slots first. Repair decisions whose child never ran are re-repaired at the start of a run. The report has `resumed_repairs` and `lever_stats`.
- `cli repair` (one child per applicable lever, for hand rescues) and `cli levers` (the learned table).

Hand rescue, ready to run: `candidates/rescue_20261104.json`, 10 single-lever or combination children of the three alphas. IV spread: backfill, liquidity tilt, TOP1000, and backfill+rank+tilt. 30-day IV: decay 10, window 10, and window 20+tilt. Cash flow: SUBINDUSTRY, INDUSTRY, and SUBINDUSTRY+backfill. Their results will be the first evidence for the learner.

### Hand rescue, round 1 results and learner fixes (2026-11-04)

`candidates/rescue_20261104.json` on real BRAIN, 10 simulations, 1 accept:
- IV spread. Backfill: Sharpe 2.03, fitness 1.67, turnover 12%, CONCENTRATED_WEIGHT passes, sub-universe 0.54 against 0.88. Liquidity tilt: 2.08 / 1.50, sub-universe 0.66 against 0.90, concentration still fails. TOP1000: sub-universe passes but Sharpe 0.99. **Backfill + rank + tilt: accepted** with Sharpe 1.38, fitness 1.01, sub-universe 0.81 against 0.60, self-correlation 0.26 and drawdown 4.4%; it is in the pool. The accept is thin: fitness is barely over 1.0, yearly Sharpe is 1.84 / 3.26 / 1.05 / 0.72 / 0.74, and BRAIN's self-correlation is pending until submission. BRAIN's sub-universe cutoff is about 0.43 x the alpha's own Sharpe, so part of the pass came from the lower Sharpe.
- 30-day IV sibling. Decay 10: fitness 0.89 with Sharpe 1.66. Window 10: 0.78. Window 20 + tilt: Sharpe 1.14. Sub-universe still fails on all three.
- Cash-flow alpha. Neutralization SUBINDUSTRY / INDUSTRY / SUBINDUSTRY+backfill: Sharpe 0.23 to 0.39. The original was a long market bet (800 long / 120 short) and is dropped.

Learner gaps found, now fixed:
- Combinations taught the playbook nothing. Each part of `combo:a+b+c` is now credited with weight 1/3, and a combination with positive evidence for a class becomes a choosable lever for that class (parts already in place are skipped).
- Side effects were not credited. A child now also counts, at `secondary_weight`, for every other class its parent failed. Backfill now gets credit for clearing concentration.

Round 2 (`candidates/rescue_20261104_round2.json`, 5 candidates, children of the best round-1 children):
- IV backfill + tilt by adv20 without rank, and IV backfill + tilt by cap.
- Sibling decay 10 + ts_mean 10, decay 10 + ts_decay_linear smoothing, and decay 15.

### Rescue round 2 and pipeline guards (2026-11-04)

Round 2 (5 simulations, children of the best round-1 children): no accepts. IV backfill × rank(adv20): Sharpe 1.86, fitness 1.62, sub-universe 0.59 against 0.81. IV backfill × rank(cap): 1.70 / 1.18 / 0.48 against 0.74. Sibling decay 10 + window 10: sub-universe passes (0.68 against 0.67) but fitness 0.92. Sibling decay 10 + smoothing, and decay 15: fitness 1.09, sub-universe 0.63 / 0.57 against 0.73 / 0.76. Every child passed BRAIN's concentration check. The IV family moves between fitness and sub-universe; its edge sits in small names. Totals: 25 simulations, 1 accept; failures sharpe 11, sub_universe 7, fitness 4, self_correlation 2.

Owner direction: improve the pipeline, not individual alphas; use BRAIN's checks only; ensure no overfitting; keep the search diverse. Built:
- BRAIN's PASS/FAIL checks as hard gates (`gates.use_brain_checks`).
- An accept-time deflated-Sharpe gate (`auditor.min_deflated_sharpe` 0.0, `trials_scope` dataset). The pool alpha passes it: 1.38 deflates to 0.51 / 0.28 / 0.12 with lineage / dataset / all trials.
- Distance to accept, used for the learner's reward (half class fix, half overall progress), for resume order, and for a per-lineage repair budget (depth 2, 4 repair simulations, patience 2). Under it, all 9 open IV-family repairs are closed (budget used or depth reached); only the two old reversal repairs from runs 1-2 remain open, and they are on gen-v1 arms, so they are not resumed.
- Diversity:
  - the Director caps one arm at 40% of a round;
  - the generator field list keeps the most relevant fields and adds a 30% exploration sample across datasets, annotated with coverage and crowding (`gen-v2`; skips SYMBOL/UNIVERSE fields);
  - Strategy Evolution (`strategy-v3`) gets least-tried datasets and the fixes that worked, with rules against narrowing;
  - the Critic flags neutralization NONE (advisory).

### Fourth research run and fixes (2026-11-04, `rs20261104073248`)

First closed-loop run under the pipeline guards: 14 simulations, 34 model calls (Gemini 22, Qwen 12), no accept. Fresh ideas: 9; 7 failed Sharpe far from the bar (Sharpe -0.37 to 0.21) and were rejected. Near misses all came from the option8 IV-spread idea again (fitness 0.71, 0.59; Sharpe 1.44, 1.61) and their repairs improved fitness only to 0.76 and 0.81, then the depth limit closed the family. The run stopped on Gemini's free-tier daily limit of 20 requests, with Qwen still working and 6 simulations left.

The run showed three faults, now fixed:
- **`model_repair` ranked first.** Every class lists it, so on an alpha failing several checks it collected the secondary bonus from its prior alone and outranked levers with better evidence (scores 0.88-1.0 against 0.41-0.44 posterior). `rank_levers` now places it last by rule, the catch-all after the mechanical levers.
- **`flip_sign` was not tried on negative Sharpe.** Two near misses had Sharpe -1.05 and -1.18; increase_decay and model_repair came first, and the model rewrite reached Sharpe 0.41. `flip_sign` is now placed first by rule (it applies only when Sharpe < 0). On a copy of memory, `cli repair` now proposes the flip first for both.
- **The Critic rewrote repair children** (`..._r1_c1`), so a simulation measured the lever plus the Critic's change, and the learner credited the lever with both. Repair children now get the Critic's rule checks only (a blocking issue still drops them); no model review, no judge, no shadow critics.

And the stop: **a failed model no longer stops the run.** `BudgetedLLM` marks a model whose call fails after the client's retries as down for the rest of the run (`llm.ModelDown`), and later calls to it fail at once without spending budget. Its generator arms get no more slots (`Orchestrator.allocate(only=...)`; their queued repairs still run); the primary critic becomes the next critic from another provider that is up, else the rules-only Critic; the judge, Pre-Filter, Repair and Strategy agents use `FallbackLLM` (their own model, then the generator models, then the rest) and fall back to their rules if all are down. The run stops only when every generator model is down. The report has `models_down`.

Note for the next runs: Gemini's free tier allows 20 requests a day, and its 503 "high demand" answers were frequent. Once Gemini is down, the Gemini arms stop and Qwen's arms are reviewed by the rules-only Critic (the critic must come from another provider), so the run keeps going but with a weaker review until a third free provider is added.

### More free models: GPT, Llama, DeepSeek (2026-11-04)

Owner: no Mistral; add ChatGPT, Llama and DeepSeek, so one host's quota cannot stall the loop. ChatGPT's own API has no free tier, so the GPT model is `gpt-oss-120b` (the open-weight GPT model) on Groq's free plan, which the existing Groq key covers; GitHub Models (hosted GPT models, about 50 calls a day with a GitHub token) is wired as an optional host. Llama and DeepSeek come from NVIDIA's free API (build.nvidia.com, about 40 calls a minute). Both new hosts speak the same chat-completions format, so each is one `llm.HOSTS` entry. A hosted id may leave out the maker when the name says it (`groq:gpt-oss-120b`); `llm.BARE_MODEL_MAKERS` gives the provider and the namespace the host expects, which also keeps the provider's name out of config files. `cli models` lists a host's models or checks every configured model with one tiny call. The roster now has five makers, so every arm keeps a critic from another maker even when two are down, and the agents that run every round default to Llama on NVIDIA instead of Gemini. Ids checked against NVIDIA's list on 2026-11-04 (`cli models`): its catalog has no Llama 3.3 70B, so Llama is `meta/llama-3.2-90b-vision-instruct` (text works); DeepSeek is `deepseek-ai/deepseek-v4.1-flash`.

Update the same day: `cli models --check` passed for Qwen and gpt-oss on Groq, but both NVIDIA models timed out, and so did a plain curl call to NVIDIA's chat endpoint (0 bytes in 120 s), while its model list answered. The stall is on NVIDIA's free endpoint, not in our client, so the NVIDIA models are out of the config for now and gpt-oss on Groq is the default for the agents that run every round. Groq's free plan has no Llama chat model. Llama and DeepSeek wait for a free host that answers (next to try: OpenRouter's free models).

OpenRouter's free list (2026-11-04) has no Llama or DeepSeek either. Owner chose a stand-in: `nvidia/nemotron-3-ultra-550b-a55b:free` as a fourth generator maker, generator only (OpenRouter's free models share about 50 calls a day, too few for an agent that runs every round). Llama and DeepSeek go back in when NVIDIA's chat endpoint answers.


### Fifth research run and fixes (2026-11-04, `rs20261104163055`)

First closed-loop run to finish every round: 4 rounds, 19 simulations, 41 model calls (gpt-oss 25, Qwen 9, Nemotron 6, Gemini 1). Gemini's daily quota was used up all run; failover kept the run going on the other three generators. No accept. Near misses: three resumed repairs (IV spread Sharpe 1.12 / fitness 0.72; analyst EPS revision after the sign flip 1.05 / 0.47; customer momentum 1.18 with self-correlation 0.73), all then closed by the repair depth limit. Nine new ideas, all far misses (Sharpe -0.42 to 0.27, best gpt-oss's news P/E change at 0.90 with turnover 83%, queued for repair).

Why some models looked weak with the same prompts: samples of 5-14 per model, and the models were not tested on equal terms (Gemini's lead is one IV family found while the plan's ideas were still unused; the later models got the harder, explored-out arms); gpt-oss lost 4 of 9 ideas to BRAIN errors the validator let through (below); Nemotron chose decay 30-60 on slow fields; no generator sees past results; no temperature is set, so each host's default applies; and the Director learns from accepts only, so with 0 accepts it cannot tell a Sharpe 1.6 idea from a 0.1 one. Proposals (not built): show each generator its arm's best and worst results, one temperature for every model, a Director reward by distance to accept, model comparison within a dataset.

Fixed:
1. **Validator gaps (5 of 19 simulations were BRAIN errors).** `fastexpr.analyze` now requires optional arguments by name (`winsorize(x, 4)` and `ts_backfill(x, 60, 1)` gave "Invalid number of inputs : 2, should be exactly 1") and allows an event (VECTOR) field only inside `vec_*` (`ts_delta(<vector field>, 30)` gave "does not support event inputs"). Field types come from the catalog (`fastexpr.catalog(ids, meta)`), passed to the Generator, Pre-Filter, Critic and Repair Agent. Replayed over every expression in memory: all 5 failures caught, no complete simulation flagged.
2. **`rank_transform` ignored a leading minus**: `-rank(x)` became `rank(-rank(x))`, the same alpha (identical results). It now counts as already ranked.
3. **Slots of a model found down mid-round were lost** (3 to Gemini in round 3). They now go back to the Director within the round for live arms; refill batches get their own ids (`...a<i>f<n>`).
4. **Ideas were credited to the wrong dataset.** Strategy's hypotheses on a dataset no arm covered went to every arm, so news12 ideas ran and were stored as analyst4/model51 (and Strategy's per-dataset counts never saw news12 as tried). A hypothesis on a catalog dataset no arm covers now gets its own arms (one per generator model, settings of the existing arms); a candidate's `dataset` is the dataset most of its non-base fields come from (`generator.signal_dataset`, the arm's on a tie), with a warning when it differs.
5. **The Critic dropped fixable ideas** (two gpt-oss ideas for a missing `ts_backfill`). `critic-v3` / `judge-v3`: drop only for a look-ahead or an untestable idea; a fixable defect is revise (or simulate). Slots emptied by the Generator, Pre-Filter or Critic go back to the Director for `director.refill_passes` extra passes per round (config/agents.yaml: 1).

559 tests pass.

### Consultant upgrade (2026-11-10..11)

**Why.** The account moved to consultant level (`level` GOLD). Settings, operators, limits, the in-sample period and the checks all changed, and BRAIN now asks for biometric verification on every new login.

**What we found** (one discovery session, `captures/consultant_20261110/`, git-ignored): 11 regions with their own universes and delays; the factor neutralizations (REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING, FAST, SLOW, SLOW_AND_FAST) and COUNTRY in GLB/EUR/ASI/AMR; 82 operators (+16); in-sample 2014-01-01..2023-12-31; LOW_SHARPE limit 1.58; new checks (LOW_2Y_SHARPE, IS_LADDER_SHARPE, PROD_CORRELATION, POWER_POOL_CORRELATION, DATA_DIVERSITY, REGULAR_SUBMISSION, CLUSTER_TEST, MATCHES_PYRAMID/THEMES, OSMOSIS_ALLOCATION, POWER_POOL_DESCRIPTION_*). The checks inside a fresh simulation are provisional (LOW_SHARPE WARNING, SELF_CORRELATION PENDING); `GET /alphas/<id>/check` gives the final ones. USA/TOP3000/D1 has 289 datasets and more than 10,000 fields, and BRAIN returns nothing past offset 10,000.

**What was built.**
- `brain_api.py`: biometric login (`BrainSession(on_biometric=)`; the CLI prints the link and waits for Enter in a terminal); `RealBrainClient` fetches the submission check (`fetch_submission_check`, `check_max_wait_seconds`) and `translate(check=)` prefers it, noting WARNING value/limit and the new `is.investabilityConstrained` / `is.riskNeutralized` stats; `list_datafields` goes one dataset at a time above `DATAFIELD_OFFSET_CAP`, with progress lines; `list_datasets`.
- `schemas.REGION_SETTINGS` and `neutralizations_for(region)` (a test checks them against `brain options`); `GROUP_NEUTRALIZATIONS`. The Generator and Repair Agent accept the region's set; `set_neutralization` leaves factor neutralizations alone.
- Prompts: the Generator system prompt (`gen-v3`) and the Repair request (`repair-v5`) list the neutralizations allowed in the region, groups and factor ones named apart; the reply shapes point at that list instead of the five old values.
- `config/fastexpr.yaml` 82 operators; `gates.sharpe_min` 1.58; `gates.brain_checks_ignored` (submission-only checks, OQ-17); the evaluator maps the new checks to failure classes; candidate start dates 2014-01-01 in the plans and settings.
- `cli brain datasets` (datasets with field counts) and `scripts/brain_discovery.py` (every region and delay in one login, per-dataset retries on 502, rerunnable; plain progress lines). Capped 2026-11-11 after the uncapped run took ~3 h for USA D0 alone: `--max-datasets` (20) chosen by `pick_datasets` (top per category, then most used, 30% least explored), per-dataset part files so nothing is lost on Ctrl+C (`tests/test_brain_discovery.py`). Run completed 2026-11-11: all 18 region/delay catalogs, no gaps (ALL D1 and USA D0 in full from the uncapped run; the other 16 with 20 datasets each, 688 to 4,021 fields).
- Fixture `tests/fixtures/brain_api_capture_consultant/` (acc_01 again, its submission check, the IV-spread pool alpha's check, prod correlations, `brain options`; redacted; SOURCE.md).

**Not done yet.** The full per-region catalogs (the first discovery hit the 10,000 cap in ALL, USA D0/D1, EUR D1, ASI, and 502s on GLB D1 and EUR D1); `captures/datafields_usa_top3000_d1.json` is still the capped 10,000-field catalog. The pool alpha (Sharpe 1.38) and the lever statistics were judged at user-level limits on 2019-2023, and its 1.38 is below the new 1.58 limit; it should be re-simulated on the new period before it is trusted.

579 tests pass.

### Regions, new checks and repairs (2026-11-11)

**Why.** The owner asked that the Repair Agent repair BRAIN's new checks and use the new neutralizations, that every agent know the consultant features, and that each agent be tuned per region, because what works in one region may not work in another. Owner's choices: new failure classes for the new checks; build region lever learning, region limits, region-aware prompts and region quotas in the Director now; write research plan v2 once the full catalogs are downloaded.

**What we found first.** The neutralization lever only moved along NONE < MARKET < SECTOR < INDUSTRY < SUBINDUSTRY, so the factor neutralizations and COUNTRY were never tried; and since lever learning and the lineage skip go by lever name, one `set_neutralization` covering all of them would have allowed only one neutralization repair per lineage. LOW_2Y_SHARPE / IS_LADDER_SHARPE were mapped to `sharpe` (whose levers, flip_sign and backfill, target another problem) and PROD_CORRELATION to `self_correlation` (whose main lever rewrites against our own pool, not BRAIN's production alphas). Their values were kept only in notes, so near-miss limits could not apply. BRAIN's risk-neutralized Sharpe (acc_01: 2.39 against 1.34) was a note too. `change_universe` knew only USA's universes.

**What was built.**
- `schemas`: `ResultRecord.check_values` and `alt_stats` (from `translate`, through `normalize`, stored in the result payload; old payloads load); failure classes `recent_sharpe`, `prod_correlation`; `GROUP_NEUTRALIZATIONS`.
- `evaluator`: the new mapping and priority; `brain:<CHECK>` gates take BRAIN's value and limit; `gates_for` uses BRAIN's limit per check when `gates.limits_from_brain`.
- `config`: `gates.limits_from_brain`; optional `regions` section of gate overrides (`USA` or `USA/D0`), `Config.for_settings`; `recent_sharpe` near-miss share. The pipeline evaluates each candidate with its region's config.
- `repair_agent`: `neutralize_<name>` levers (7), region-aware `change_universe`, repair metrics with BRAIN's failed checks and alternate stats. Playbook: `recent_sharpe`, `prod_correlation` and factor levers as priors elsewhere. Prompts: `restore_recent_performance`, `differentiate_from_production`, `CLASS_MEANING`.
- `lever_stats`: per-region evidence with shrinkage to the all-region estimate (`region_prior_strength`), region-aware ranking and examples, `region_table`; reports and `cli levers` show it.
- `orchestrator`: `region_floor_share`, `max_region_share`.
- `prompts.brain_context(config, regions)`, Generator market line and per-round `region_notes`, Strategy `regions` breakdown and `region_guidance`; `cli prompts --regions`.

**Not built (suggestions).** See "Per-region tuning: further ideas" in CLAUDE.md.

609 tests pass.

### Research plan v2 (2026-11-11, draft)

**Why.** The owner asked for a plan over every region and delay the account allows, with region blocks whose defaults the arms inherit, datasets per region from the downloaded catalogs, and the new neutralizations.

**What blocked it.** The loader keyed arms by (dataset, region), so one dataset at delay 0 and delay 1 in the same region was refused, and the Director's arm had no delay. The run took one datafield catalog, so a USA field would pass validation in a CHN alpha and fail only at BRAIN. Strategy Evolution gave a new dataset's arms the settings of an arbitrary arm per model, which in a multi-region plan could be a region without that dataset.

**What was built.**
- `research.load_plan`: v2 layout (`defaults`, `regions.<R>.enabled/settings/catalogs/arms`), settings merged defaults <- region <- arm, checked against `REGION_SETTINGS`; arms unique per (dataset, region, delay); `ArmPlan.catalog`. v1 plans load as before.
- Arm key `(dataset, region, prompt_version, model, delay)` (delay last, so the other indexes keep their meaning); `arm_label` for reports.
- `ResearchRun(markets=)`: catalog per (region, delay); each arm's Generator gets its market's fields and meta; `fastexpr.catalog_for` lets the Pre-Filter, Critic and Repair Agent validate each candidate against its own market. `cli research` reads each plan catalog once (`_plan_catalogs`).
- `_evolve`: templates per (model, region, delay); a new dataset's arms only where its market's catalog has it.
- `config/research_plan.v2.yaml`: 37 arms over 18 region/delays (11 regions). Datasets chosen from the 20 downloaded per region/delay for coverage, low alpha and user counts, a high pyramid multiplier and fields that carry a signal; option8 kept in USA D1. Every field named in a hypothesis was checked against that market's catalog. Group neutralization by default (COUNTRY in multi-country regions, SECTOR in the 500-name universes), CROWDING / STATISTICAL / REVERSION_AND_MOMENTUM on three arms, lower decay at delay 0.

**Open.** 37 arms x 4 generator models = 148 Director arms, far more than a free-tier run simulates; per-region `enabled` runs a subset. An offline round over the real catalogs (fake model, stub BRAIN) ran 10 simulations across regions.

631 tests pass.

### Region rotation (2026-11-11)

**Why.** The owner asked to cap each run at about 20 alphas, with a run covering only some regions and the next run the others, so that across runs every region is covered equally.

**What was built.** `director.regions_per_run` (default 3) and `director.region_balance_since` (default 2026-11-11, so the earlier USA-only runs do not count) in `config/agents.yaml`. `MemoryStore.simulations_by_region(since)` (read-only) counts candidates that reached BRAIN per region. At the start of a run `ResearchRun._pick_regions` chooses the least-simulated regions (then the longest since the last simulation, then plan order); only their arms are live, only their repairs are taken or resumed. Each allocation splits the slots across those regions with `orchestrator.balance_slots` (fewest simulations so far, memory plus this run), then the Director allocates within each region. Report: `regions_this_run`, `simulations_by_region`. The run size stays the `--max-simulations` flag (20).

**Checked.** Three offline 20-simulation runs over the v2 plan (fake model, stub BRAIN) took USA/ALL/GLB, then EUR/ASI/CHN, then JPN/AMR/IND, 5 to 7 simulations per region each.

**Open (OQ-19).** The unit balanced is the region, not region/delay (delays inside a region follow the Director); defaults 3 regions per run and the 2026-11-11 start date.

643 tests pass.

### Seeds: earlier alphas to improve (2026-11-11)

**Why.** The owner asked to include the two earlier accepted alphas, which fail consultant-level checks: the IV spread from memory (`rs20261103184506r01a0_002_r1x`; consultant check LOW_SHARPE 1.03, LOW_FITNESS 0.62, IS_LADDER_SHARPE 0.73) and a skew + volatility-risk-premium alpha accepted in the BRAIN web app (USA/TOP3000/D1, decay 0, truncation 0.01, INDUSTRY; BRAIN TRAIN view Sharpe 1.72, fitness 1.15, risk-neutralized Sharpe 1.50 / fitness 0.71).

**What was built.** `research.load_seeds` (`seeds:` in the plan: `from_memory` or inline expression + settings, settings checked like arms via `_entry_settings`); `ResearchRun(seeds=)` / `_prepare_seeds`: a new root candidate on the arm's dates (so it is a new experiment, and BRAIN runs its current checks), the arm chosen by region/delay then dataset then model, validated against the market catalog; simulated first in the round once its region is in the run, with `credit=False` (no Director update); report `seeds`. Skipped with a note: not in memory, invalid, no arm, already simulated, same experiment as one in memory. `scripts/brain_discovery.py --datasets` fetches named datasets as well as the picked ones.

**Status.** The skew/VRP seed uses `opt6_ivhvxernratio` (option6), which is not in our USA D1 catalog (option6 was not among the 20 datasets fetched), so it is skipped as invalid until `scripts/brain_discovery.py --regions USA --datasets option6` is run. Offline check on a copy of memory: the IV seed was the first simulation of the run, on 2014-01-01..2023-12-31.

657 tests pass.

### First plan v2 run (2026-11-11, `rs20261111162604`)

**Result.** The skew + VRP seed was accepted after two repairs: `seed_774d71e6` (Sharpe 1.65, fitness 1.07, failed only PROD_CORRELATION 0.81 > 0.70) -> `neutralize_statistical` (PROD_CORRELATION gone, fitness 0.90) -> `wrap_in_smoothing` (ts_decay_linear 10 around the whole signal): `seed_774d71e6_r1_r2`, Sharpe 1.84, fitness 1.35, turnover 13.2%, sub-universe 1.00, self-correlation 0.13, PROD_CORRELATION 0.655 < 0.70; every BRAIN check passes. The IV spread seed reached Sharpe 1.14 / fitness 0.52 (rank_transform) and hit the repair depth limit. Fresh ideas: GLB and USA far misses (Sharpe -0.26..0.26).

**Problems found and fixed.**
- BRAIN refuses regular alphas in region ALL (400 "Region ALL is not available for simulation type REGULAR"): 7 of 20 simulations wasted. `schemas.NO_REGULAR_SIMULATION`; the plan loader refuses ALL arms and seeds; ALL is `enabled: false` in plan v2 (35 arms).
- BRAIN's parser rejects scientific notation (`1e-08`: "Unexpected character 'e'"); the lexer now raises with the plain decimal to use.
- A GLB simulation hit `max_poll_seconds` 600; raised to 1200.
- The console printed every arm's allocation (140+ zeros) and every new evolution arm; now only arms with slots and the first 6 new arms plus a count.
- Seen, not changed: Nemotron (OpenRouter) gave an empty reply and was marked down for the run (failover worked); Gemini 503s were retried; Groq per-minute limits were waited out.

### Why GLB looked weak, and general guards (2026-11-11, gen-v4)

**Diagnosis.** Four GLB results, all Sharpe -0.26..0.26 (BRAIN's GLB limits are not stricter: sub-universe limit 0.08-0.16). Both news20 alphas used `nws20_rp_story_event_count`, an identifier ("Identifies which content set the record came from"): the hypothesis named the sentiment fields, but the relevance ranking scores name parts ("story", "event", "count") and did not favour named fields. Both analyst69 revision alphas started under REVERSION_AND_MOMENTUM (my plan setting), which removes much of a revision signal. analyst69 has no GLB data before mid-2015.

**Owner: no region-specific fixes, general ones.** Built: field roles (`generator.field_role`, patterns in `config/generator.yaml` matched at the start of BRAIN's description; checked on all 17 catalogs: 621 id, 1,291 time of ~78,000; keyword matching anywhere in the text was tried first and mislabelled real signals such as currency-adjusted estimates and implied volatilities, so it was dropped); id fields left out of the prompt unless named; time fields tagged; an alpha using only id fields is sent back to the model; fields named in the hypothesis listed first as the ones to build on (`named_fields`); new alphas start from a group neutralization (plan loader, Generator), factor neutralizations and NONE stay Repair Agent levers; plan v2's three factor-neutralization overrides removed. Generator prompt `gen-v4`.

674 tests pass.

### Model-chosen neutralization, prompt-version resume, BRAIN-side losses (2026-11-12)

**Model decides neutralization (owner decision 2026-11-11), gen-v5 / critic-v5.** The Generator may choose any neutralization the region allows. A factor neutralization or NONE needs a one-line `neutralization_reason`; without it the arm's default is used (warning in the generation report). The reason is appended to the rationale (`Neutralization X: ...`) so the Critic sees it. The Critic gets a rule advisory for a fresh alpha's factor neutralization (repair children are exempt: their lever comes from results), a prompt paragraph on when a factor neutralization fits and when it removes the signal, and may revise only the neutralization (reply key `neutralization`, checked against the region). NONE remains an advisory. Evidence: `strategy.region_notes` adds the Repair Agent's neutralization levers in the region (fixed/improved/tried per failure class) to the per-neutralization accept counts it already showed. Plan defaults stay group neutralizations (`research._check_start_neutralization`).

**Open repairs after a prompt version bump.** `_warm_start` and `_resume_repairs` keyed on the candidate's `prompt_version`, so gen-v3 -> gen-v4 orphaned earlier runs' open repairs and warm-start evidence. `ResearchRun._arm_of` now uses the candidate's own arm if it exists, else the same (dataset, region, model, delay) under the current prompt version.

**BRAIN-side losses (run rs20261111174623: 2 of 3 simulations).** (1) Timeouts: `memory.pending()` skips any candidate with a result, and a timeout is stored as one, so timed-out submissions were never looked at again. New read-only `MemoryStore.unfinished_submissions()` (latest result `timeout` with a simulation id, plus pending submissions without any result); at run start `ResearchRun._recover_submissions` polls each once (`Simulator.recover`, `Pipeline.recover`); a finished one is decided like any other (no simulation budget, Director credited, before resumed repairs so a recovered near miss can be repaired), one still running waits for the next run, one BRAIN no longer knows is skipped. Report `recovered_results`. (2) BRAIN's generic error: `brain.yaml` `retry_error_messages`; `RealBrainClient.poll` marks such a failure `retryable` and the simulator submits once more (the new simulation id is on the result; `pending_submissions` keeps the first, since it is append-only).

687 tests pass at that point.

### Why non-USA results were poor, and pipeline guards (2026-11-12)

**Diagnosis (33 non-USA candidates, runs rs20261111162604, rs20261111174623, rs20261112054223).** 12 lost to infrastructure (region ALL 7, scientific notation 1, BRAIN error 1, timeouts 2, one in flight), all fixed since. 21 completed: Sharpe -1.68..0.48, median |Sharpe| ~0.2, no accept (USA: 78 completed, median 0.93, 2 accepts). Causes: (1) plan v2 arms are all news/sentiment/short-interest/alternative datasets, several with almost no BRAIN alphas in that market, while each market's workhorse datasets (pv1, model110/26/38, analyst69, fundamental6/17/23, risk70; fields downloaded) are unused; (2) half the arms are delay 0, where BRAIN's bar is Sharpe 2.69 / fitness 1.5 (JPN, AMR) or 3.49 / 1.5 / returns 12% (CHN; CHN delay 1: 2.07, returns 8%), and the prompts said 1.58; the current run spent 14 of 18 simulations at delay 0; (3) story-based sentiment fields fed to ts_zscore without backfill: CONCENTRATED_WEIGHT 0.5, turnover 140-160%, investability-constrained Sharpe <= 0; (4) most risk-neutralized Sharpes were below the raw ones (factor exposure); (5) the best lead, CHN D0 retail flow at Sharpe -1.68 (risk-neutralized -2.62), went to turnover (priority) and was smoothed and rejected without a sign flip; a sibling at -0.32 raw / +1.52 risk-neutralized was rejected on the raw value alone.

**Built (owner: pipeline fixes first).** `evaluator.sharpe_route` (sign first / factor neutralization first, `repair.risk_neutralized_gap` 0.5, OQ-20) and near-miss on max(|raw|, |risk-neutralized|); `RepairAgent.ranked` puts `neutralize_*` first and drops `flip_sign` on a factor-first decision; `critic.SparseFieldRule` / `sparse_unfilled` / `backfilled` with a rule revision for fresh alphas (critic-v6 prompt line); `load_datafield_meta` keeps the dataset category; `strategy.brain_limits` + `prompts.bar_text` show BRAIN's limits per region/delay in `brain_context` and each Generator arm's guidance. Replay over memory: the only decision changed by the routing is the CHN retail-flow alpha (now repair / sharpe, sign first). Next (proposed): research plan v3 on each market's most-used datasets, delay 1 first, the flipped CHN retail-flow alpha as a seed.

701 tests pass.

### Crowded fields (2026-11-14, gen-v6)

**Why.** Run rs20261114135300: `rs20261114135300r01a104_001_c1` (DEU TOP500 D1, analyst7, gpt-oss) had Sharpe 2.16, fitness 2.12 and passed every BRAIN check but PROD_CORRELATION (0.80 vs 0.70); its risk-neutralized Sharpe was 1.15, so about half of it was a known factor. It was built on `est_12m_ebi_raisednum_4wks`, used by 831 BRAIN alphas (436 users) while the median analyst7 field has 1 and 263 of 608 have none. The field ranking used crowding only to break relevance ties, and nothing fed PROD_CORRELATION failures back.

**Built (owner chose A and C).** (A) `generator.crowded_fields`: per dataset, fields at or above the `crowding.percentile` (0.9) quantile of BRAIN alpha counts and >= `min_alphas` (50), groups and base fields excluded; `_rank_datafields` lists them after every other field, `select_datafields` never draws them for exploration; named fields stay first, marked `crowded`, with one prompt line asking for a form other than the common one. On the downloaded catalogs this flags 91 of 1,423 DEU D1 fields, 419 of 4,154 USA D1. (C) `strategy.prod_correlation_fields`: per region, the non-base fields of our simulated alphas whose latest result failed PROD_CORRELATION; `ResearchRun._generator` refreshes `AlphaGenerator.prod_failed` before each batch and those fields count as crowded, annotated `N of ours failed production correlation`. The Director already scores a prod_correlation failure as a non-accept, so it is unchanged. Prompt version gen-v6. `tests/test_crowding.py`.

**Plan v4 (owner chose option 1: hypotheses name no fields).** `config/research_plan.v4.yaml` is plan v3 with the 60 hypotheses rewritten to describe the idea, the mechanism and the form to avoid, with no datafield ids (same arms, settings, seeds; test checks both and that no hypothesis names a signal field in its catalog). Plain words that are base fields or groups (high, low, returns, sector) no longer count as named fields (`named_fields(exclude=)`, gen-v6); before, "high relative to" made `high` a field to "build the signal on".

**Offline test of the Generator's listing, v3 vs v4 (60 hypotheses, real catalogs).** Crowded fields named: 72 -> 0; crowded in the top 15 listed: 1.2 -> 0; but top-15 fields from the arm's own dataset: 6.8 -> 3.7, median BRAIN alpha count 5 -> 2. Cause: `_rank_datafields` scores id parts as substrings of the text, so without field names it lists off-topic fields (USA option8: pre_news_high_price first). **Ranker rebuilt (owner chose to build it).** `_rank_datafields` order: crowded fields last; the arm's own dataset (`GenerationSettings.dataset`, passed by `AlphaGenerator` to `build_system_prompt`/`select_datafields`) ahead of other datasets, which the 30% exploration share still covers; then the hypothesis's content words (`_words`: 3+ letters, stopwords removed, first 6 letters) found in the field id or BRAIN's description, each weighted by inverse document frequency over the catalog. Without the IDF weight, USA fundamental6's accrual hypothesis listed noncontrolling-interest fields first (matched only "quarterly balance sheet"). Plan v4, offline, 60 hypotheses: 14.4 of each top 15 from the arm's dataset (3.7 with the old ranker), none crowded, exploration across a median 19 datasets, 0.03 s per prompt; plan v3 under the new ranker also lists 14.4 own-dataset fields. Examples: GLB model110 lists the model's components (quality, tree, growth, value) instead of the headline score; GBR analyst7 lists downgrade counts on less used items (0-41 alphas). pv1 has ~20 fields, so its listing is complete either way.

713 tests pass.

### Simulations 3 at a time (2026-11-14, OQ-3)

**Why.** Run rs20261114135300 spaced its 18 results 2-4 minutes apart: one simulation at a time, mostly waiting on BRAIN.

**Built (owner: option 1, 3 at a time).** `Simulator.run_many(candidates, max_concurrent)`: validate all, then submit until `max_concurrent` are running (a resumed pending submission takes a slot), poll each running one in turn, sleep the shortest wait any of them asked for, submit the next as soon as one finishes; each simulation times out on its own clock; a BRAIN error is resubmitted once, first in line. One thread, since BRAIN does the work; SQLite and `VirtualClock` stay as they were. `run` is `run_many` of one, so single-candidate behaviour is unchanged (all earlier tests pass untouched). `Pipeline.process_many`: duplicates of memory or of an earlier candidate in the batch are skipped, the rest simulated together and then evaluated one at a time in order (pool admission order unchanged); `Pipeline.run` (and `cli run`) goes in chunks of `max_concurrent`. `ResearchRun`: `_simulate` queues a screened alpha; `_flush` sends the queue when it holds `max_concurrent` alphas and at the end of every round (also when a model failure stopped the round), then does each alpha's Director credit and repair child in queue order; budget checks count queued alphas (`_committed`). Over the account's real limit BRAIN answers 429 on submit and `BrainSession` waits (up to `rate_limit_max_wait_seconds`, 900 s). Fixture test: 5 simulations of 3 polls with limit 3 take 40 virtual seconds instead of 100.

721 tests pass.

### Unused variables, 8 at a time, bounded evolution (2026-11-14)

**Unused variables.** BRAIN refused `trial3_20261114_both_subind` with 'Found unused variables "liq"'. `fastexpr.analyze` now errors on a variable no later statement reads (redefinitions that read the old value count as reads). Replay over memory: 1 hit, that alpha; no other candidate affected.

**8 at a time.** Round 2's 8 hand alphas (`--max-concurrent 8`) were all submitted within 6 s with no 429 and all results were back in 3.8 minutes, so `simulator.max_concurrent` is now 8. `cli run --max-concurrent N` overrides it for one run; the 31 hand-written fixtures give identical decisions at 1 and 8.

**Arms.** Run rs20261114135300's first evolution added 140 arms (all 30 plan arms x 4 models copied in every region, the originals kept, plus 20 new-dataset arms) to a 20-simulation run, leaving 237 arms with no results. Now: only live arms of the run's regions evolve; each evolved arm replaces its parent (`successor`; the parent gets no new slots, its queued repairs still run and its outcomes credit the latest successor) and starts from the parent's Beta posterior (`Orchestrator.add_arm(like=)`); arms for datasets no arm covers are capped at `strategy.max_new_arms` (8), one per (dataset, market of the run) with the generator models in turn. With plan v3/v4 and 3 regions per run, an evolution now changes about 36 live arms in place and adds at most 8.

**DEU/GBR rescues.** Round 1 (`rescue_20261114_8.json`): leaving EBIT out cost 2022-23. Round 2 (`_round2_8.json`): adding recommendation breadth lifted DEU to Sharpe 1.82-1.94 / fitness 1.77-1.99 with sub-universe passing and risk-neutralized Sharpe 1.30-1.45; GBR net breadth on pre-tax and net income reached Sharpe 1.58. All fail only LOW_2Y_SHARPE (best 1.17 of 1.58); PROD_CORRELATION stays PENDING until that passes. Round 3 (`_round3_8.json`, owner: tune toward 2022-23) tries 1-week and 1-month revision counts, faster smoothing, industry neutralization with truncation 0.05, and EBIT at a quarter weight.

**Pass = BRAIN's checks alone (owner).** `gates.brain_checks_only: true`: each of our five gates takes BRAIN's verdict for the matching check whenever BRAIN reported it (turnover needs both LOW_ and HIGH_TURNOVER), our thresholds deciding only when BRAIN did not (fixtures, PENDING); the deflated-Sharpe, pool-diversity and risk results are added to the decision as "recorded, not applied" and never reject or hold an alpha pending. Found by `hand_20261114_gbr_3`: Sharpe 1.58, BRAIN's LOW_SHARPE PASS (limit 1.58), our gate FAIL (strictly greater). Replay over the 138 results with BRAIN checks: 11 decisions change, all to BRAIN's verdict (9 user-level USA results that passed BRAIN's 1.25 Sharpe limit then; the pool's IV alpha; gbr_3, now reject / recent_sharpe). Memory is not rewritten.

**Operator limit.** Three round-3 alphas failed with "Expression with 66 operators exceeds limit of 64". BRAIN counts every function call plus every arithmetic or comparison operator and unary minus; that count reproduces its 66, 66 and 71. `config/fastexpr.yaml` `max_operators: 64`, checked in `fastexpr.analyze` for every alpha; replay over memory: exactly those 3. Round 3: `deu_11_fast` (2-day smoothing) Sharpe 2.06 / fitness 2.05 / 2-year Sharpe 1.32 (best yet; deu_7 1.17), `deu_9` (1-week counts) 1.21; GBR best 2-year 0.82 (`gbr_8`, Sharpe 1.57). Round 4 (`rescue_20261114_round4_8.json`) combines and extends those, all under 64 operators.

Round 5 (pairing with other signals): `deu_19` (EBIT breadth at half weight, no smoothing) Sharpe 2.32, fitness 2.52, 2-year Sharpe 1.43, the best yet but still short; pairing with a low-volume dip raised 2023 (up to 1.30) but cut 2022 to 0.33-0.52 and turnover rose to ~49%, so the recent checks got worse; GBR's overnight reversal lifted the overall Sharpe (1.82) but not 2023 (0.23). The hand rescues of this family stop here.

### General improvements from the rescues (2026-11-14)

**Submission-only checks (owner: count them, but only at that stage).** `gates.submission_checks` (quota, theme/pyramid/competition matches, Osmosis, Power Pool description and correlation) become `submit:<CHECK>` gates that `evaluator.decide` looks at only after every other hard gate passed; a failure then gives pending (the alpha is good, BRAIN would not take it as it is), never reject or repair. Replay over 160 results: no decision changes.

**Lessons by dataset category (gen-v7).** `generator.category_lessons` in `config/generator.yaml`, shown to the Generator by the BRAIN category of its arm's dataset (`dataset_category`, from the catalog), plus `all` for every arm: keep the speeds of blended parts alike (the dip blends), a part helps a blend only where it is strong in the blend's weak years and not weak elsewhere; for analyst data, net revision breadth over several less-used items plus recommendation breadth, little smoothing, and the note that EBIT upgrades correlate with production.

**Repair lever `speed_up`.** Mechanical: decay down to `decay_to` and the windows of smoothing operators (`ts_decay_linear`, `ts_mean`) down to `window_to`; windows that define the signal (`ts_delta`, ...) stay. First lever for `recent_sharpe` (playbook order; lever learning re-ranks with evidence). In the rescues faster smoothing raised the 2-year Sharpe 1.17 -> 1.32 -> 1.35 where no field change did.

The owner submitted `seed_774d71e6_r1_r2` (BRAIN alpha `3q0wOPoQ`) on 2026-11-14. `regions_per_run` stays 3 (owner).

### First plan v4 run, progress lines, memory split (2026-11-14)

**Run `rs20261114174347` (GLB, EUR, ASI; stopped by the owner in round 3, no report file).** 10 simulations in two batches of 5. The 2026-11-12 factor routing paid off: an EUR pv1 overnight-vs-intraday reversal written by gpt-oss had raw Sharpe 0.91 but risk-neutralized 2.28, was sent "factor neutralization first", and its `neutralize_statistical` child reached Sharpe 2.71, fitness 1.30, every year positive, failing only sub-universe and the 2-year Sharpe (1.45). ASI model110 component alphas (Sharpe 1.10-1.25, built as the v4 hypotheses ask, from the model's components) fail ASI-only checks (LOW_ASI_JPN_SHARPE, LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO, LOW_ROBUST_UNIVERSE_RETURNS), which have no failure class yet. In ASI, BRAIN reported LOW_SHARPE as WARNING (provisional, other checks failed); our gate then uses BRAIN's own limit, so the outcome matches. Strategy Evolution updated 36 arms in place and added none. BRAIN refused a variable named `delta`: `fastexpr.yaml` `reserved_names` (replay: exactly that alpha).

**Progress lines.** With several alphas at once the decisions print only when the whole batch is done, and a finished alpha's recordsets and /check can take minutes after BRAIN shows it complete, so the terminal looked idle. `Simulator.run_many(on_done=)` and `Pipeline._progress` now print `BRAIN: <id> finished (k of N), ...` as each finishes.

**Memory split (owner).** `scripts/split_memory.py --since 2026-11-10` moved `data/` to `data_archive/pre_consultant_20261114/` and rebuilt memory from the consultant-era rows (114 of 192 candidates, pool = the skew+VRP alpha only, 3 run reports). MemoryStore stays append-only.

740 tests pass.

### EUR reversal rescue and what the agents learned from it (2026-11-15)

**The rescue (hand rounds, 38 variants).** The EUR pv1 overnight-vs-intraday reversal (Sharpe 2.71 after `neutralize_statistical`) failed sub-universe and the 2-year Sharpe. Industry peers instead of country fixed the recent years. BRAIN's sub-universe cutoff is about 0.52 x the alpha's own Sharpe; an uncentred size tilt `x (0.25 + rank(cap))` lifted the sub-universe/full-Sharpe ratio from 0.43 to 0.54 and kept the recent years (2-year Sharpe 2.61), while traded-value (adv20) tilts cost fitness and recent years, centring the rank before the tilt lowered the ratio to 0.44, a smaller universe cut the Sharpe 2.80 -> 1.03, and truncation and decay 6 changed nothing. That variant (eur_26) then failed only PROD_CORRELATION, 0.739. Subindustry peers raised it to 0.760, so the large-stock lean was not the cause; only new information lowered it: analyst69 net revision breadth (cash flow, pre-tax profit, EBITDA, operating profit; each under 100 BRAIN alphas) at a third of the weight gave 0.720, at half 0.645. **Accepted: `hand_20261115_eur_37_blend_half_analyst`** (BRAIN `omaA0nmE`): Sharpe 2.25, fitness 1.07, every BRAIN check passes, deflated Sharpe over 42 trials passes. The analyst part alone was weak (Sharpe 1.17, near zero before 2022) and carries most of the blend's 2022-23 strength, so its holdout years are worth watching. The DEU/GBR analyst7 lineages (2026-11-14) are closed: 2023 never recovered without the crowded EBIT field.

**Built into the agents (owner: the lessons belong in the pipeline, not in a rescue log):**
- Repair lever `blend_other_dataset`, first for `prod_correlation`. The model writes only a partner signal, from a shortlist of fields of other datasets in the alpha's market (`repair_agent.partner_fields`: not price/volume or risk-model categories, not crowded, not id/date fields, at least 5 BRAIN alphas, ordered by word match to the playbook's `topics`, at most 8 per dataset); `build_blend` makes `(rank(alpha) + weight x rank(partner))`, ranked within the alpha's group when it has one, the partner's missing values at 0.5, a size/liquidity tilt kept outside. A partner that uses none of the listed fields is refused. Repair prompt `repair-v6` (`<partner_fields>`). Peer-group, tilt, horizon and neutralization levers moved behind it.
- Mechanical lever `size_tilt` (`x (offset + rank(cap))`, offset 0.25), first for `sub_universe`; `liquidity_tilt` and `change_universe` moved down.
- Critic advisory for a fresh alpha that reads only base fields and price/volume datasets (`critic.price_volume_categories: [pv]` in agents.yaml): the crowding check never flags base fields, so a pure price signal looked uncrowded.
- A lineage with an accepted member gets no more repairs (`lever_stats.family_budget`): later variants would fail SELF_CORRELATION.
- Generator lessons (gen-v8): a `pv` lesson (pair price ideas with another dataset at about equal weight, rank within industry, cap tilt over traded value) and in `all` that production correlation needs new information, not a reshaped signal; `analyst` adds sector peers, no revision size, `vec_avg` for list fields, revision breadth as a blend part.
- Plan v4 EUR: the overnight/intraday hypothesis is replaced by a price idea paired with another dataset; new analyst69 arm (SECTOR, decay 2) with two revision-breadth hypotheses.

753 tests pass.

### Lesson Writer: lessons after every batch (2026-11-15)

Owner: the lessons of a batch should be written automatically so the pipeline improves itself. `lessons.py` runs after every research run and every `cli run` batch and always reads all of memory, so a lesson that later evidence contradicts is replaced at the next write. Evidence (no model): (a) change effects per (lever, parent's failure class), over all regions and per region: how often the class was fixed and the median change in Sharpe, fitness, sub-universe ratio (only at Sharpe >= 0.5, where the ratio is stable; one parent near Sharpe 0 gave a mean ratio change of -7.79 before this), recent-years Sharpe, production correlation and turnover; production and self correlation count as fixed only on an accept or a measured passing value, since BRAIN computes them last; (b) production correlation by the dataset categories an alpha reads; (c) failures per region and data category; (d) the latest 40 hand-written variants with their hypothesis (what the author changed) and the same deltas, which only the model can read. Rule lessons are always written; with `lessons.use_model` the strategy model condenses the evidence (one call within the run's budget), and its lessons are kept only with a known audience (generator / repair), a scope from the evidence, a number and at most 400 characters. Stored append-only (`lessons` table); the next run's Generator (after the configured category lessons) and Repair Agent (`<lessons>`, treated as data) read the latest write for their market and data. Nothing changes a decision. On a copy of memory: 15 rule lessons (2 change-effect groups, production correlation by data, 11 failure profiles) and 40 described changes. 770 tests pass.


### Repair waves and gain credit (2026-11-23)

Owner: the Repair Agent should try four different fixes at once and, with eight simulations per BRAIN batch, repair two alphas at a time; when no fix works, the next best four go out together. A repair that improves an alpha a lot should count as a good repair even when it does not cross the limit, since the same change may carry an alpha that needs less. BRAIN's checks decide; overfitting checks only raise a flag. Simulations may be higher; model calls are the limit (free APIs).

- **Waves** (`RepairAgent.repair_wave`, `ResearchRun._start_wave` / `_finish_wave` / `_next_start`). Up to `repair.fixes_per_wave` (4) children per parent: levers of every class the alpha fails, one per kind of change first (smoothing, shape, neutralization, universe/tilt, data, sign, each model lever), skipping levers already tried along the lineage or on the same parent; `model_repair` last; at most one model-written fix per wave. Waves run whole and first in a round (up to 2). When a wave is in: an accept ends the lineage; else the next wave starts from the child closest to accept if it beat the parent (the levers that helped its siblings first), else from the parent with its next levers. One Director credit per wave, so four sibling failures do not count as four arm failures.
- **Gain credit** (`lever_stats.gap_closed`, `LeverStats.estimate`). The class part of a repair's reward is the share of the gap closed (Sharpe 1.20 -> 1.50 against 1.58: 0.79) instead of a flat 0.5. Each (class, lever) keeps its past slack gains; for an alpha that misses a class by `need`, the estimate is (1 - 0.5) x posterior mean + 0.5 x P(gain >= need) once there are two gains.
- **Budget relaxed** (owner): depth 3, 12 repair simulations per original alpha, patience 8 children (two waves); it is a cost limit now. Deflated Sharpe stays recorded on the decision and is printed as a flag (`overfitting_flags`).
- Suggested run size: `--max-simulations 60 --max-llm-calls 60 --per-round 12`.

### IND momentum/reversal rescue and `cli run` deciding as alphas finish (2026-11-24)

**The rescue (owner's plan, 17 hand/lever variants).** IND `rs20261123190057r01a118_001` (model39 global price-momentum percentile minus the 10-day price change in rupees, SECTOR, TOP500; Sharpe 3.14, fitness 2.20) was rejected only for BRAIN's robust-universe Sharpe (0.95 of 1.0), then mapped to `other`. After `cli reevaluate` (-> repair / sub_universe), round 1 (`candidates/rescue_20261124_ind.json`): backfilling the model field (`r1c`) gave Sharpe 3.11, fitness 2.17, robust-universe 1.01, every year >= 1.7, and failed only PROD_CORRELATION 0.730; the units fix (n-day return instead of a rupee change) cost a third of the Sharpe (2.07), and size and liquidity tilts cost most of it (0.85, 1.08). Round 2 (`_r2.json`, `cli repair` levers): CROWDING neutralization (Sharpe 2.09, robust 0.59) and longer windows (1.56, robust 0.76) broke the robust universe; REVERSION_AND_MOMENTUM and STATISTICAL timed out on BRAIN (30 min; recovered at the next research start). Round 3 (`_r3.json`, 8 blends built with `repair_agent.build_blend`, lever `blend_other_dataset`): analyst partners lowered the production correlation and raised the robust universe, every other partner broke it:

| partner (1:1 unless noted) | Sharpe | fitness | robust | PROD | decision |
|---|---|---|---|---|---|
| annual EPS consensus, 60-day z-score (analyst4) | **4.10** | **3.44** | **1.36** | **0.687** | **accept** (BRAIN `LLKQ7Kpe`) |
| quarterly net-profit consensus, 60-day z-score | 3.48 | 2.65 | 1.20 | 0.684 | accept (`wporjRQ1`, in the pool) |
| same at half weight | 3.43 | 2.52 | 1.16 | 0.712 | repair / prod_correlation |
| net-profit trend + net-margin change (three-way) | 3.30 | 2.51 | 0.97 | - | repair / sub_universe |
| net-margin change, half weight (fundamental17) | 3.20 | 2.37 | 0.86 | - | repair / sub_universe |
| net-margin change | 2.98 | 2.31 | 0.73 | - | repair / sub_universe |
| positive news sentiment, 5-day (news85) | 2.43 | 1.37 | 0.67 | - | repair / sub_universe |
| net-margin level | 2.09 | 1.46 | 0.53 | - | repair / sub_universe |

The two accepts correlate 0.80 (the EPS one was accepted but not pooled: our pool check is informational under `brain_checks_only`); only one can be submitted, and the EPS blend is better on every measure (risk-neutralized Sharpe 2.97, investability-constrained 2.52, worst year 2.1). Its production correlation (0.687) is close to BRAIN's 0.70, which BRAIN measures against the production pool at submission time. Deflated Sharpe over 22 lineage trials passes. The rupee-unit price leg stays (BRAIN UNITS warning, not counted).

**Into the agents:** the blend lever's instruction (repair-v7) prefers analyst partners with good large-stock coverage (net revision breadth or a consensus-estimate trend against its own history) and steers away from slow fundamentals and news sentiment; generator category lessons (gen-v9): `all` adds the IND evidence and that half weight left the correlation at 0.71; `analyst` now separates revision size as a signal alone (hurt, DEU) from a consensus-trend blend part (worked, IND). The Lesson Writer's stored write after the batch measures `blend_other_dataset` on prod_correlation in IND: fixed 2 of 8, median production correlation -0.043.

**`cli run` decides as alphas finish.** One variant stuck on BRAIN held a 5-alpha batch's decisions for 30 minutes. `Pipeline.run` now uses the streaming path of `process_many` (as research runs do since 2026-11-23): each alpha is decided the moment its result is in, and the next alpha in the file goes into a freed slot at once; outcomes are still returned in file order; a new batch starts only when none of the current one reached BRAIN (all duplicate or invalid). `tests/test_concurrency.py` (2 new tests). 815 tests pass.

### CHN production correlation, BRAIN's slow checks and stuck simulations (2026-11-24)

**CHN rescue, rounds 2-3.** Lineage `rs20261115045859r02a52_001` (other561 retail net buyers). The retail part alone fails only PROD_CORRELATION (0.93). Round 2: measuring the direction of retail flow (net-buyer share, imbalance, portfolio flow) lost the Sharpe (-0.03 to 0.81); blends kept it: retail + ARM 1:1 (Sharpe 3.83, PROD 0.793) and retail + 2x (ARM + EPS consensus trend) (4.02, 0.791). Doubling the partner did not lower the correlation, so the partner itself resembles production (`star_arm_score` is in 146 BRAIN alphas). Round 3 (`candidates/rescue_20261124_chn_r3.json`): partner = analyst69 net revision breadth, (up - down) / (up + down + 1) over 4 weeks for EPS, net income, sales and pretax profit (fields in 0-19 alphas each; alone Sharpe 0.76, risk-neutralized 1.38). Results: CROWDING neutralization on the round-2 best 0.762 (Sharpe 4.00); 20-day retail + breadth 0.842 (worse); STATISTICAL on the round-2 best Sharpe 4.59; breadth blends Sharpe 3.38-3.71, every other check passed; their PROD_CORRELATION was still uncomputed by BRAIN 2.5 h later.

**Why checks took hours.** For alphas that passed BRAIN's other checks, PROD_CORRELATION came back within 2-12 min on 2026-11-11..15 and this morning, but today only the first 2-4 alphas of each batch of 8 got it within minutes; the rest took hours (round 2: about 5 h). Our client waited up to 5 min (self-correlation) + 10 min (`/check`) per alpha, one alpha at a time, so 8 alphas took about 2 h. `/check` answers an empty body with Retry-After 1 s while BRAIN computes.

**Stuck simulations.** Peer feedback: simulations of runs stopped with Ctrl-C stay on BRAIN. Of our 13 unfinished submissions, 7 sat at "running, 90%" (up to 10 days; left by our 30-min timeout and by stopped runs), 3 answered HTTP 500, 3 were gone. `DELETE /simulations/<id>` cancels (200, then 404); 10 cleared; the 3 HTTP-500 ones refuse the cancel too.

**Built.** `cli recheck` and run-start recheck (alphas pending on a BRAIN check with no verdict are fetched again and decided; a new result is stored only when the checks changed). An unfinished alpha check (PENDING / ERROR) is an unknown gate, so such an alpha is pending (one was accepted with PROD_CORRELATION ERROR; recheck re-decides such accepts). `RealBrainClient.cancel`, `Simulator.cancel` (stores a failed result so recovery stops asking), Ctrl-C cancels the batch's running simulations, `simulator.cancel_stuck_after_hours` 6 (run start cancels what is still unfinished; recovery saved 3 late results since 2026-11-10, all within hours, so no cancel at our timeout), `cli brain cancel`. BRAIN waits: `check_max_wait_seconds` 60 (was 600), `self_correlation_max_wait_seconds` 60; `simulator.max_concurrent` 4 (was 8). Tests `test_recheck.py`, `test_cancel.py`.

**Repair agent, observed.** Of 81 pipeline repair children 1 was accepted; of 18 hand children 2 (both `blend_other_dataset`). The pipeline never ran that lever: `cli repair` skips model levers, PROD_CORRELATION arrived late (pending), waves allow 1 model lever, and the partner filter's `min_coverage` 0.5 excludes fields like the breadth counts. Mechanical levers reshape a signal (decay, smoothing, windows, neutralization); production correlation is about the information in it, which only a new partner changes. Proposed next: a model-planned wave for information failures, built from the family's evidence.

### CHN round 3 results, pool follows decisions, partner lessons (2026-11-25)

**Round 3 values (BRAIN, read 2026-11-25).** PROD_CORRELATION: retail + 2x net revision breadth 0.698 (accepted: `hand_20261124_chn3_breadth_2x`, BRAIN `ZYXRwXVx`, Sharpe 3.38, fitness 3.28, every year >= 1.97, turnover 17.6%; submitted by the owner 2026-11-25), + 2x (breadth + EPS consensus trend) 0.731, + 2x (ARM + EPS trend) under CROWDING 0.762 / STATISTICAL 0.819, + 2x (breadth + ARM) 0.804, 1:1 breadth 0.839, 20-day retail + breadth 0.842. So: partner weight matters when the parent is far above the limit (1:1 0.839 -> 2:1 0.698), and heavily used rating-model scores (ARM) raise the correlation.

**Pool.** breadth_2x was accepted but not admitted: our pool-diversity check (recorded only under `brain_checks_only`) failed at 0.93 against its sibling breadth_arm, which had been accepted while PROD_CORRELATION was ERROR and stayed in the pool after recheck re-decided it (0.804, repair). Built: under `brain_checks_only` an accept joins the pool; a decision other than accept retires the alpha (append-only `pool_events`: retire / restore; every pool reader counts only active alphas); `Pool.sync` and `cli pool [--sync]`; `scripts/split_memory.py` copies `pool_events` when present. Applied to memory: breadth_2x admitted, breadth_arm retired.

**Partner lessons into the Repair Agent.** All four breadth items had coverage 0.33-0.46, below the partner filter's 0.5, so `blend_other_dataset` could never have offered them. Playbook step: `min_coverage` 0.3 (the blend fills a missing partner with the middle rank), `prefer_low_usage` (fields used by at most the median number of BRAIN alphas listed first, topic order within each half), `weight_by_prod` [[0.78, 1.0], [1.0, 2.0]] (`repair_agent.blend_weight`: parents at 0.73-0.74 passed with 1:1 in EUR and IND, the CHN part at 0.93 needed 2:1). Prompt repair-v8 (prefer less-used fields; rating-model partners raised CHN's correlation; `_up` fields have `_dn` counterparts), analyst category lesson gen-v10. Tests `test_pool_sync.py`, `test_recheck.py`.

**Pool = submitted alphas (owner, same day).** The pool held every accepted alpha, including IND `wporjRQ1` (accepted, not submitted; 0.80 correlated with the submitted `LLKQ7Kpe`). Owner: add alphas only when told they were submitted. `diversity.pool_admits` (thresholds.yaml `submitted`; optional, default `accept` for older and test configs): in `submitted` mode the pipeline never adds or retires pool alphas; `cli pool --submitted <candidate or BRAIN id>` adds (event `submitted`; `Pool.add_submitted` from the latest complete result), `--remove <id> [--reason]` retires. Applied: `wporjRQ1` removed; pool = USA `3q0wOPoQ`, EUR `omaA0nmE`, IND `LLKQ7Kpe`, CHN `ZYXRwXVx`.

**Production correlation always gets a model call (owner, same day).** The wave mechanism stays; a model is now always asked when the alpha fails PROD_CORRELATION. Gaps found: the one model slot could go to another class's model lever or be filled by mechanical levers first when production correlation was secondary; a blend with no partner field spent the slot without a call; a lineage could blend only once (the CHN blend children could never blend again). Built in `RepairAgent.repair_wave`: production-correlation model levers first; the slot counted only when the model was asked; `blend_other_dataset` `repeatable: true`, its prompt lists the lineage's tried partners with BRAIN's production correlation (`repair_agent.tried_partners`, repair-v9); notes when no model is configured or none applied. Wave sizes for the 22 production-correlation alphas in memory: 21 get 4 fixes, 1 gets 2; all their lineages are closed (accepted) or out of repair budget, which still applies. Far-off Sharpe stays unrepaired (near-miss share and wave tiers, unchanged). Tests `test_prod_model_call.py`.

### BRAIN's production-correlation tips into the pipeline (2026-11-25)

**Tips and our evidence.** BRAIN's platform tips for PROD_CORRELATION: less crowded regions, less-used data (risk, insiders) or datasets with high value scores, small liquid universes, delay 0 or factor neutralizations (SLOW, FAST). Our memory holds 28 alphas with a PROD_CORRELATION value (6 under 0.7): by region IND median 0.700, EUR 0.720, USA 0.732, DEU 0.795, CHN 0.835 (15 values); by neutralization SECTOR 0.712, STATISTICAL 0.730 (3 of 8 under), INDUSTRY 0.837; the lowest groups are small universes (IND TOP500, EUR TOP1200). BRAIN's dataset lists (`datasets_<market>.json`, `valueScore`) show risk data is not little used (EUR 215k alphas, USA 161k); short interest, insiders, institutions and earnings are. Owner chose A (playbook) and B (dataset value); delay 0 and crowding-weighted region rotation not built.

**A: mechanical levers for production correlation.** The playbook's `prod_correlation` entry now has, after the model levers, `neutralize_statistical`, `change_universe` (next smaller universe of the region; only USA, EUR, JPN, ASI have one, CHN/IND/DEU/GBR/AMR have a single universe), `neutralize_slow`, `neutralize_fast`, `neutralize_slow_and_fast`, then CROWDING, REVERSION_AND_MOMENTUM, a longer window and `model_repair`. A wave takes one lever per kind, so a 4-fix wave is blend + a neutralization + a universe + a horizon. Overfitting guard: BRAIN's sub-universe and robust-universe checks and the deflated-Sharpe flag.

**B: BRAIN's dataset value.** `generator.load_datafield_meta` now reads the dataset list next to a field catalog (`dataset_catalog_for`, `load_dataset_meta`) and copies `dataset_value_score` / `dataset_alpha_count` onto each field; `dataset_facts` and `dataset_appeal` (value high, then few BRAIN alphas) order datasets. Generator (gen-v11): relevance ties and exploration go to less crowded datasets first (above-median value: two fields per turn), and the prompt lists the listed fields' datasets with value and usage. Strategy Evolution (strategy-v5): `least_tried` was ordered alphabetically after our trial count; now by appeal, with `least_tried_facts`; the prompt asks for high-value, little-used datasets; new-dataset arms are ordered by appeal before the `max_new_arms` cap. Blend partner (repair-v10): `value_share` 0.25 of the 48-field shortlist from the least crowded datasets, after the topic/usage order; partner lines show the dataset's value and alpha count. A reorder by value alone was tried first and dropped analyst69 (value 1), the partner of the EUR accept, so value only adds fields: on the real catalogs EUR keeps analyst69 and gains option1 / institutions6, CHN gains sentiment27, IND gains insiders1 / shortinterest5.

**Plan v5** (`config/research_plan.v5.yaml`, DRAFT, not run): plan v4 plus one arm per region on its least crowded dataset among short interest, insiders and institutions (USA/ASI/JPN/AMR shortinterest3, GLB shortinterest2, EUR institutions6, CHN fund_holdings_panel, IND insiders1, DEU/GBR insider_agg_matrix), decay 6-8, hypotheses that name no fields. 41 arms. Tests `test_prod_tips.py`, `test_plan_v2.py::test_plan_v5_...`. 854 tests pass.

### Checked against BRAIN's API and simulation-limit docs (2026-11-25)

**Matches.** POST /authentication with basic auth; biometric login (401, `WWW-Authenticate: persona`, Location joined to the response URL, POST to it after verifying); POST /simulations (201 + Location); polling GET /simulations/<id> on Retry-After; the alpha id on COMPLETE/WARNING; recordsets `yearly-stats`, `daily-pnl`, `pnl` and their schema/records encoding; failures on ERROR/FAIL/TIMEOUT with `message`. Our settings (instrumentType, pasteurization, unitHandling, nanHandling, language, visualization) are the docs' keys; no testPeriod (OQ-12).

**Live facts** (read-only GET /authentication with the saved session): a login has at most 4 h (12,199 s left at the time; the docs cap `expiry` at 14,400 s); permissions include MULTI_SIMULATION and QUICK_MODE. Saved discovery captures show request limits on other endpoints: /alphas 2,000 per hour, correlations 60 per minute (`x-ratelimit-*-hour`, `-minute`); a run's ~6 GETs per alpha are far below them.

**Fixed.** (1) Daily simulation limit: POST /simulations sends `X-Ratelimit-Limit/-Remaining/-Reset` (resets midnight US Eastern); we ignored them, and a 429 for the limit would have been waited out for 15 min and retried as if it were a concurrency limit. Now `brain_api.simulation_quota` reads them on every submit (`RealBrainClient.quota`, report `brain_quota`, printed at the end of `cli research`); a research run stops before a round when at most `quota_reserve` (50) are left; a 429 with 0 left raises `SimulationQuotaExhausted` at once and the run stops (running simulations are recovered next start). (2) Statuses: WAITING and SIMULATING without Retry-After are running (were read as "finished without an alpha"); CANCELLED is a failure. (3) An ERROR's `location` (line, columns) is in the failure note. (4) `cli research` renews a login with less than `relogin_within_seconds` (3 h) left at its start, so the biometric check comes while the owner is at the terminal. (5) Validator: a matrix field, base field or constant as a `vec_*` argument is an error (the docs' "Invalid data field" / "must be an event data"); replay over memory flags one alpha, which BRAIN failed, and no completed one.

**Built after (same day, owner: multi-simulation and QUICK mode not relevant).** `cli brain diversity` and `brain_api.submission_diversity` (GET `/users/self/activities/diversity`, grouped by region, delay and category; 5 submissions: CHN 2, EUR, IND, USA 1 each, every category row with DATA_DIVERSITY PASS at limit 0.3), fetched once at each research run's start and given to Strategy Evolution as `our_brain_submissions` (strategy-v6 prefers categories with few submissions in the run's regions). `neutralize_slow_and_fast` joins the sharpe playbook entry and goes first when a Sharpe is routed factor-first, since BRAIN's risk-neutralized stats are that neutralization. BRAIN's search-space advice is in the Generator's `all` lesson (gen-v12). 868 tests pass.

**Proposed earlier.** Multi-simulation (2-10 alphas per POST, same type/region/delay/language; children count toward the limit): could raise throughput if BRAIN's concurrency limit counts the parent once. `simulationMode: QUICK` (permission held): faster, but what it leaves out is undocumented; worth one side-by-side trial. `/users/<id>/activities/diversity` (submissions by region, delay, category): input for DATA_DIVERSITY and region choice. The simulation-limit guide says the risk-handled stats are SLOW_AND_FAST neutralization: a factor-first repair could try `neutralize_slow_and_fast` first, since BRAIN has already shown its result.

### BRAIN's region guides: CHN (2026-11-25)

**What the guide says and what we had.** CHN's bar is higher (trading costs): our gates already take BRAIN's own limits per check (`limits_from_brain`), and the prompts show them per market. Risk-handled (SLOW_AND_FAST) neutralization to judge uniqueness and robust-universe retention: already used (factor-first route now tries SLOW_AND_FAST first; LOW_ROBUST_UNIVERSE_* map to sub_universe). New to us: short selling is restricted to eligible blue chips, so "the opposite alphas will not flip the performance". Our repair logic assumed a flip mirrors: near misses by |Sharpe|, a SIGN_FIRST route and `flip_sign`. Outside CHN that held exactly in memory (GBR -0.98 -> 0.98; USA -1.05 -> 1.05, -1.18 -> 1.18); we have no CHN flip.

**Built.** `schemas.SHORT_RESTRICTED_REGIONS` = {CHN} (a market fact from BRAIN, not a policy threshold). There a negative Sharpe or fitness counts with its sign (no near miss), no SIGN_FIRST route, and `flip_sign` does not apply. Replay over memory: of 6 CHN alphas with negative Sharpe, only the retail flow at -1.68 changes (repair/turnover -> reject); its idea later became the CHN accept as a rewritten alpha in the other direction with other parts, which this rule does not stop (the Generator can still write the other direction). Generator `region_lessons` (gen-v13) with a CHN entry: prefer slower, low-turnover, strong-return signals; direction from the mechanism; the 5-10% daily price limits distort price changes over limit days; keep robust-universe Sharpe and returns; judge uniqueness by the risk-handled result.

**Datasets.** BRAIN recommends model14, 25, 26, 31, 53, 122; fundamental5, 17, 27; analyst14, 17. Our CHN field catalog has only model26 (the 20 datasets downloaded per market). In BRAIN's CHN TOP2000U list but not downloaded: model14 (Quant Model Data, value 2, 2,121 alphas), model31 (Earnings Quality Model, 3, 265), fundamental5 (China Fundamentals, 3, 547), fundamental27 (A-shares Fundamental Data, 3, 887), analyst14 (Estimations of Key Fundamentals, 3, 961). Not in our CHN list: model25, model53, model122, fundamental17, analyst17.

**Downloaded and added (owner, same day).** Fields for model14, model31, fundamental5, fundamental27 and analyst14 (`brain_discovery.py --regions CHN --delays 1 --datasets ...`; `--delays` is new): the CHN delay-1 catalog grew from 2,313 to 6,027 fields over 25 datasets. Plan v5 adds two CHN arms, model31 (148 earnings-quality fields, median coverage 0.98, at most 41 alphas per field) and fundamental5 (957 China fundamentals, 629 VECTOR), decay 6, two hypotheses each that name no fields. CHN's catalog has a fundamentals dataset (fundamental6) whose field ids are plain words (cash, sales, assets, debt, inventory, receivable, revenue), so the hypotheses avoid those words: a named field is listed first whatever its dataset. Offline, every one of the top 15 fields listed for each new arm comes from that arm's dataset.

### Run 4 (`rs20261125111614`) and the run-efficiency fixes (2026-11-25)

**Run.** Plan v4 on the code from before the day's changes (prompt gen-v10), regions GLB, ASI, JPN. Stopped on the 60-call model budget after 24 simulations. No accept: 16 rejects (all Sharpe; GLB/JPN fundamental17 gave -0.36 to 0.85) and 8 repair decisions that got no fix (distance 0.66-3.7, above the 0.4 tier). Leads, all with BRAIN's risk-neutralized Sharpe well above the raw one: ASI pv1 `group_rank(ts_mean(vwap - close, 5), country)` (Sharpe 1.95, risk-neutralized 3.06; fitness 0.74 at 51% turnover with decay 0; fails LOW_ASI_JPN_SHARPE 0.55 of 1.0 and investability-constrained Sharpe 1.09 of 1.36); JPN other176 earnings-quality blend (1.27, risk-neutralized 1.87, PROD_CORRELATION 0.7004); ASI model110 growth + quality (0.78, 1.61). The three simulations that answered HTTP 500 were cancelled at run start (31-41 h old).

**Why the calls ran out.** 2.5 model calls per simulated alpha: the Director spread 12 slots over 12 arms, so most arms wrote one alpha, each a Generator call plus a Critic call. gpt-oss was down from its first failure: a 9,195-token request (prompt about 5,100 plus the 4,096 output cap) against Groq's free 8,000 tokens a minute (HTTP 413).

**Built (owner).** Repair tiers: 0.4-0.6 from accept gets 2 mechanical fixes only (no model call, also for production correlation), farther nothing. `director.min_batch` 3: every arm given slots gets at least 3 (`orchestrator.consolidate`, total unchanged, the Director's top arms keep theirs), so one Generator and one Critic call serve three alphas. `--max-llm-calls` is optional; without it a run ends when every generator model is down (its free limit), at the simulation budget, after the rounds, or after `director.stop_after_idle_rounds` (2) rounds with no simulation. `llm.RequestFitClient` lowers a call's output cap so prompt plus cap fit the model's per-minute token limit (from `models.pace`; floor 1,024). LOW_ASI_JPN_SHARPE maps to sub_universe (a sub-universe question like the robust-universe checks), so it no longer blocks ASI repairs; the ASI lead itself is 0.91 away and still gets no fix under the tiers.

### Repair only when every limited class is a near miss (2026-11-25)

In plan v5's first round, `r01a1_003` (Sharpe -0.59, fitness -0.12, turnover 78%) was decided "repair / turnover". The decision picks the class to repair first by priority (turnover before Sharpe) and checked the near-miss limit of that class only; turnover has none, so the far Sharpe and fitness never counted. No fix was made (its distance gave a wave of 0), but the label was wrong. Now every failing class with a near-miss share must pass it; the first that does not rejects the alpha under that class. Fitness, like Sharpe, counts the better of raw and BRAIN's risk-neutralized value, and an alpha routed "factor neutralization first" is not rejected on recent-years Sharpe: the EUR accept descends from such an alpha (raw fitness 0.41 and 2-year Sharpe -0.47, risk-neutralized fitness 0.96; its STATISTICAL child reached Sharpe 2.71, fitness 1.30), which a raw-only rule would have dropped. Replay over memory: 46 of 141 repair decisions become rejects; their 37 children never led to an accept; no accept's lineage is blocked.

### Run 5 (`rs20261125130750`) and the Generator quality guards (2026-11-25)

Plan v5 on USA/AMR/GLB, round 5 of 6 at the time of writing, 52 fresh alphas: no accept, no repair child at all. The fresh alphas were plain: 46% used one source of information or none (one non-base field, or price/volume only), 29% only crowded or heavily used fields (USA option8 skew used by 1,350-2,000 BRAIN alphas; AMR model39's value-momentum rank by 948, production correlation 0.91), mean |Sharpe| 0.53. The relative crowding rule (top 10% of a dataset) flagged none of them: in a mined dataset every field is heavily used. The opposite extreme failed too: loan and short-interest fields with 1-40 BRAIN alphas gave Sharpe near 0, so "least used" alone is no guide. Leads at Sharpe 1.05-1.39 got no fix because their other failures put them more than 0.6 from accept. Every GLB alpha also failed BRAIN's LOW_GLB_AMER/APAC/EMEA_SHARPE checks, mapped to `other` (no levers), so no GLB alpha could be repaired.

Built (owner chose these): heavily used fields (`generator.crowding.heavy_alphas` 500: tagged, kept in place, never explored or offered as partners); a structure rule for fresh alphas (`generator.structure`: two sources, one neither crowded nor heavily used; sent back once, then dropped); blend slots (`generator.blend`: a quarter of each batch must read two datasets, with partner fields of other datasets that reached |Sharpe| >= 1 in our alphas in the market, `strategy.signal_evidence`); the arm's latest 4 results in the request (`strategy.arm_history`); leads get 2 mechanical fixes (`repair.lead_sharpe_share` 0.63); the three GLB checks map to `sub_universe`; `quality.py`, report key `generation_quality`, `cli quality`. Prompt gen-v14 (about 900 tokens longer).

Checks: all five accepts pass the structure rule. In memory the rule does not pick out higher-Sharpe fresh alphas (mean |Sharpe| 0.60 passing, 0.64 failing; production correlation measured on 4 only), so it is a novelty guard whose value must show in `cli quality` and accepts over the next runs; about two thirds of past fresh alphas would have been sent back, which costs one extra model call in most batches. CHN's LOW_RETURNS check (13 failures in memory) is still `other`.

### Run 5 result and request-size fixes (2026-11-25)

`rs20261125130750` finished on its 60-simulation budget: 53 model calls, no accept, no repair wave (39 reject / Sharpe, 6 incomplete, 15 "repair" with wave size 0). Four of five models went down: gpt-oss (HTTP 413, 11,002 tokens against 8,000 a minute) and Qwen (11,644 input tokens against 7,000), almost certainly one Strategy Evolution request (about 23,500 characters, indented JSON) passed from one to the other by the fallback; Gemini's 20 free requests a day; Nemotron Lightning's empty reply. Rounds 5-6 ran on Ernie alone, with the Critic on rules.

Fixed: a too-large request (`llm.RequestTooLarge`) no longer marks a model down (agents fall back for that call, the fallback tries the next model); the size client learns each model's characters per token from the host's own counts, retries a 413 once with an output cap that fits, and refuses before sending once it knows no cap fits; report keys `too_large_calls` and `prompt_tokens`; Strategy records sent as compact JSON and cut to `strategy.max_prompt_chars` (15,000; 23,500 -> 14,800 on current memory). The run's model-written lessons (one advised deprioritizing GLB on evidence from plain alphas when GLB alphas could not be repaired) were superseded by a rule write.



## Run rs20261125173606 and three fixes (2026-11-26)

Run `rs20261125173606` (ASI/GBR/JPN, gen-v14, 17:36-20:01 UTC) wrote no report: the process stopped (no Ctrl-C cancel record) with `r02a74_002` still on BRAIN. 21 alphas reached BRAIN, no accept, 7 timed out at 30 minutes (4 ASI, 2 JPN other176, 1 GBR model38). Leads: ASI `r02a69f2_001` (close-below-VWAP gap x current-asset quality tilt, Sharpe 2.56, fitness 1.03; fails LOW_ASI_JPN_SHARPE 0.53 and investability-constrained 1.78 vs 1.79) and GBR `r01a202_001_c1` (value, Sharpe -1.54: sign-first route). Fresh alphas' mean |Sharpe| about 1.18 (run 5: 0.53).

Fixed:
- **Coverage 0 = not reported.** BRAIN's GBR and DEU catalogs list every price/volume field and nearly every fundamental field at coverage 0.0, so the Critic's sparse rule asked for `ts_backfill(close, 20)`, the Generator showed "cov 0%", and blend partners were filtered out. `generator.load_datafield_meta` reads 0 as None (every consumer treats None as unknown); `SparseFieldRule.fields(meta, exempt=)` never counts base fields.
- **Group in the wrong slot.** BRAIN refused `group_rank(industry, x)` ("Incompatible unit ... Unit[Group:1]"). `fastexpr` now checks that in an operator with a `group`/`g1`/`g2` argument a group name sits only in a group slot (`OperatorSpec.params`, `GROUP_PARAMS`). Replay over 529 expressions (memory + archive): one hit, the alpha BRAIN refused.
- **Freed slots take new alphas.** New alphas waited for the repair batch twice (18:06-18:07 and 19:14-19:46) while 2 of 4 slots were idle. `_round` now allocates before the repairs go to BRAIN (`_set_fresh`); `_top_up` asks `_fresh_for_freed` after ready waves: the next arm of the allocation is generated whole (one Generator call), screened, and joins the running batch (extras wait in the simulator queue). `_run_fresh` works through what is left; slots lost or emptied mid-batch go to the round's refill. `director.fill_freed_slots` (true). `tests/test_repair_waves.py` (2 new), `test_critic.py`, `test_fastexpr.py`. 931 tests pass.

- **Stuck-cancel line.** At the next run start six simulations were cancelled (BRAIN refused with HTTP 500): a direct poll showed each at `{"progress": 0.9}` 10-24 h after submission. The run now prints what its poll found before cancelling (`Simulator.last_recover`: still running at N%, no answer, not found, checks unfinished).

## Run rs20261126060436 and five fixes (2026-11-26)

Plan v5, GBR/JPN/ASI, 06:04-08:05 UTC, stopped by the owner: round 2 had looped through 9 refill passes printing "request too large". About 31 simulations, no accept, no 30-minute timeouts until round 2. Ernie wrote every new alpha until Qwen's first batch at 07:34; Gemini was out of its free daily calls (resets 08:00 UTC). Best: GBR model38 intrinsic value / price `r01a204f1_002` Sharpe 1.52, fitness 1.05 (fails LOW_SHARPE 1.52 vs 1.58 and the IS ladder); ASI VWAP-gap lineage best 2.01 / 0.82 (the fixes trade fitness against the ASI-Japan check).

Fixed:
- **Refill loop.** A too-large request does not take a model down (by design), so its arms got the round's lost slots back forever. Each too-large generator request now halves the fields that model's prompts list (`ResearchRun.field_cap`, all arms of the model, rest of the run) and retries at once; at `MIN_FIELDS_WHEN_TOO_LARGE` (20) the model writes no alphas this run (`no_fit`, out of `_live_arms`). Refill passes for lost slots are also capped at the number of generator models + 1. Report `generator_field_cap`, `generator_no_fit`. `tests/test_too_large_generator.py`.
- **A server error on one recordset no longer fails the result.** `RealBrainClient._fetch` turns a ConnectionError (5xx after retries) into None plus a note, like a timeout; only the alpha itself (`required`) still raises. `memory.unfinished_submissions` now includes a failed result whose note is `poll_error:` (BRAIN finished it, our poll failed), so recovery fetches it again: the two alphas of this run plus four older ones (two IND, two CHN hand variants).
- **Sign-first distance.** `lever_stats.distance` measures a sign-first alpha as if flipped (Sharpe/fitness/returns-type gates change sign, a negative threshold too): the GBR value alpha -1.54 went from 6.65 to 0.03, so resume order and wave size see it. Its wave is the flip alone (`RepairAgent.ranked`, `_wave_options`): other fixes of the unflipped alpha keep the wrong sign.
- **Backfill no-op.** `backfill_fields` treats a field inside `group_backfill` as filled (the ASI child re-ran the same alpha; BRAIN gave identical numbers).
- Run start prints what the poll found before cancelling a stuck simulation (above).

938 tests pass.

## Unattended runs end by themselves (2026-11-26, owner)

Runs go on while nobody is at the terminal, so a loop or a lost login must end them cleanly (queued alphas still go out, report and lessons written), never hang:
- **Loop guard** (`agents.yaml` `director.stall_minutes` 60, `stall_model_calls` 40): `ResearchRun._stalled` stops the run when no alpha went to or came back from BRAIN for 60 min, or 40 model calls were spent since the last alpha went out. Checked at every generation, refill pass and freed-slot top-up, and before each round; progress (`_mark_progress`) = an alpha queued for BRAIN, joining a running batch, or decided. Stop reason `loop guard: ...`.
- **Login guard** (`brain.yaml` `login_reserve_seconds` 2700): before each round `RealBrainClient.login_stop` asks BRAIN how long the login has left and stops the run under 45 min (a new login needs the biometric check). `cli research` switches off the biometric and credential prompts after its start-up login (`cli._unattended`), so a login BRAIN ends mid-run raises `BrainLoginRequired` and the run stops with `BRAIN login lost: ...` (running alphas are recovered at the next start) instead of waiting at a prompt.
`tests/test_unattended.py` (7). 945 tests pass.

## Arm Designer (2026-11-26, owner)

Owner: arms should be written by the pipeline, a new batch only when the previous arms are done; build steps 1 (dataset choice by rules), 2 (settings from evidence) and 4 (guards, storage, report), not step 3 (model-written hypotheses). `arm_designer.py`; `ResearchRun._design_arms` at run start for each market of the run; `MemoryStore.put_arm_designs` / `arm_designs` / `fresh_simulations` (append-only table `arm_designs`, also copied by `scripts/split_memory.py`); `load_dataset_meta` keeps BRAIN's dataset name and description; agents.yaml `designer:` (OQ-24 builder defaults: 4 arms a batch, done after 6 new alphas, 1 exploration arm, 5 signal fields minimum, weights, decay by category, 3 near misses before their settings are used, near = distance <= 0.6). Plan arms of a designed market go to `ResearchRun.inactive` (no slots, no evolution; their queued repairs still run).

Dry run on memory and the catalogs (offline): GBR -> other250 (exploration), model307, insider_agg_matrix, analyst48; JPN -> model77, sentiment27, socialmedia32, institutions14. Two problems found and fixed before any run: GBR other250's BRAIN description is of an ID mapping while its fields are product ratings, prices and bestseller ranks (the brief now also quotes three of its least used signal fields' descriptions); BRAIN value scores run 1-10 (scaled by the market's highest). `tests/test_arm_designer.py` (7). 952 tests pass.

## Prompt budget and operator spotlight (2026-11-26, owner; gen-v15)

Owner: the same instructions go with every API call; more data breadth must not mean more context. The catalog stays local; per call the Generator sends a slice chosen by rules. Built: `max_datafields_in_prompt` 120 -> 60; `prompt_budget` (`max_prompt_chars` 14000, fields cut by 20% steps to `MIN_PROMPT_FIELDS` 20; `max_lesson_chars` 1800 with `lessons_for(specific_first=True)` and `fit_lessons`); `operator_spotlight` (`spotlight_ops`: rules by the arm's latest failure classes, `strategy.arm_failures`, plus "event" for event categories or VECTOR fields, then operators our simulated alphas never used, `ResearchRun.unused_ops`, rotated per call; BRAIN's descriptions in `config/operators_brain.json`). Measured on the real catalogs: GBR 16,800 -> 12,382 characters, USA 13,932, JPN 12,817, spotlight included; a 2,000-dataset-field catalog gives the same size as a 200 one (`tests/test_prompt_budget.py`). Too-large refusals now start halving from 60 fields. 957 tests pass.

- **Designer keeps an exploit arm** (owner 2026-11-26): each batch first takes the dataset with our best |Sharpe| in the market (>= `exploit_min_share` 0.6 x the bar), designed before or not (`arm_designer.exploit_rows`); all five accepts came from near misses. Dry run: GBR analyst7 (1.82), JPN other176 (1.87), ASI fundamental23 (3.22) lead their batches. 958 tests pass.

## Blend Planner (2026-11-26, owner; prompts blend-v1, blend-review-v1)

Owner: blending is how most accepts came; partners must suit the alpha they fix and have economic backing, not be a random blend. Built `blend_planner.py` (module doc has the method), wired into `RepairAgent.repair_wave` (`_planned_blends`; the planning call is the wave's model call, the rest of the wave mechanical, `blend_other_dataset` skipped that wave), `ResearchRun` (planner with the repair model, reviewer = critic models of another maker; `_pool_pnl` from the active pool by region; `_blend_flag` after BRAIN), prompts `blend_plan_system/user`, `blend_review_system/user`; agents.yaml `blend_plan:` (OQ-25). Partner checks happen before simulation (the owner's rule that an alpha passes on BRAIN's checks alone stays: the year-stability test is a flag, not a gate).

Dry run on memory (no calls): for the EUR reversal leads the rules rank analyst69 as a proven partner (PnL correlation with the alpha 0.07-0.08); analyst69 is the partner of the accepted EUR blend.

Bug found on the way (older): BRAIN's `brain:PROD_CORRELATION` / `SELF_CORRELATION` / `POWER_POOL_CORRELATION` gates were scored higher-is-better in `lever_stats._shortfall` and `_gate_slack`, so every production-correlation failure had distance 0 to accept (0.93 like 0.71). `_lower_is_better`. Replay: 26 of 173 repair distances change, 10 move wave tier (e.g. CHN retail 0.93: 0 -> 0.33). `tests/test_blend_planner.py` (10). 970 tests pass.
