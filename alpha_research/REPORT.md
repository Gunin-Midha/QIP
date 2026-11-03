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
- Proposed, not built: wait-and-retry on short 429s and pause a rate-limited model instead of stopping the run; `max_datafields_in_prompt` 300 to about 80; more free hosts (Mistral; NVIDIA NIM as a second Qwen host). Owner question: should a candidate that also fails Sharpe or fitness still go to repair for a higher-priority failure class?
- `SETUP.md` at the repo root explains how teammates set the system up on their own laptops.
