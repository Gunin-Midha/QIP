# Build Report: Alpha Research System, Phases 1 and 2

**Status:** Phases 1 and 2 are complete. 164 tests pass, none fail and none are skipped. Phase 3 (LLM generation) has not started.
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
| Alpha Generator Agent | none | Phase 3 (LLM) |
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
2. The main config keeps the holdout `null`. The test config uses a placeholder holdout of 2024-01-01 to 2025-12-31, plus placeholder risk limits of drawdown 0.15 and weight concentration 0.10.
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
| OQ-2 | A raw BRAIN response sample | `RealBrainClient` stays a stub; key names and the weight-concentration unit are unconfirmed |
| OQ-3 | Concurrent simulation limit | single-threaded, `max_concurrent: 1` |
| OQ-4 | Max drawdown and weight concentration limits | checks not configured |
| OQ-5 | Leg-balance ratio and Sharpe floor after removing the top PnL days | not configured |
| OQ-6 | Weak-year Sharpe floor and yearly spread | not configured |
| OQ-7 | Minimum marginal pool Sharpe gain | not configured (correlation is still enforced) |
| OQ-8 | OS/IS ratio floor | family flags return None |
| OQ-9 | Holdout dates | the simulator refuses to run under the main config |

---

## 8. Suggested next steps

1. Answer OQ-9 and OQ-4 so the main config can run end to end.
2. Get one raw BRAIN response (OQ-2), then implement `RealBrainClient` and map its field names in `simulator.py`.
3. Phase 3: add the LLM agents (generator, critic, debate, pre-filter, strategy evolution) on top of the existing contracts. Repair levers can start rewriting expressions at the same time.
