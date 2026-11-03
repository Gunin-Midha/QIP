# CLAUDE.md: QIP Alpha Research System

Context for any new Claude session in this folder. Read this first, then `BUILD_SPEC.md` for the full requirements and `alpha_research/REPORT.md` for the detailed build report.

## What this project is

This is a WorldQuant BRAIN alpha research system for the IIT Madras Quant Guild (QIP). It implements the multi-agent architecture from Gunin Midha's (NA24B035) WorldQuant Project Member application, Question 8: Director, Research Loop, Data Scientist, Generator, Critic, Debate, Pre-Filter, BRAIN Simulator, Risk, Diversity, Overfitting Auditor, Repair, Pool, Memory and Strategy Evolution.

- `BUILD_SPEC.md` is the authoritative spec for Phases 1 and 2.
- `alpha_research/` holds the implementation.

## Current status (as of 2026-11-02)

- **Phase 1 and Phase 2: DONE.**
- **Phase 3a, Alpha Generator: DONE.** `generator.py`, `fastexpr.py`, `llm.py` and `cli generate`. 284 tests pass (164 from Phases 1-2, 120 new), none fail, none are skipped.
- **Phase 3b, remaining agents + research loop: DONE.** `prefilter.py` (Cheap Pre-Filter), `critic.py` (Critic + Debate judge), `repair_agent.py` (levers that rewrite expressions), `strategy.py` (Strategy Evolution), `research.py` (closed loop, `cli research`), `prompts.py` (all agent prompts, versioned; `cli prompts` prints them). 433 tests pass. Not run against real BRAIN or the real model yet.
- **Model per agent + Gemini + model comparison: DONE (2026-11-03).** `llm.GeminiLLMClient` (stdlib `urllib`), `models` section in `config/agents.yaml`, generator models as bandit arms, primary critic always from another provider, shadow critics, `critic_reviews` table, `compare.py`, `cli compare`. 461 tests pass. **First real Gemini call 2026-11-03** (`cli generate --model gemini-3.6-flash`, free tier): 3 valid candidates, 0 rejected, 1 model call.
- **Free models + Qwen (2026-11-03, owner: free API keys only for now):** `llm.ChatCompletionsLLMClient` (stdlib `urllib`) serves hosted open models: `groq:<maker>/<model>` on Groq (`GROQ_API_KEY`), plain `<maker>/<model>` on OpenRouter (`OPENROUTER_API_KEY`). `config/agents.yaml` now uses only free models: generators `gemini-3.6-flash` and `groq:qwen/qwen3.8-27b`, each the other's primary critic, shadow critics off, strategy/default Gemini Flash. Gemini 3.1 Pro is paid-only; Claude has no free tier; OpenRouter had no free Qwen on 2026-11-03 (its live list), so Qwen comes from Groq's free plan. Inline `<think>` blocks are stripped from hosted replies. Hosted calls send `User-Agent: qip-alpha-research/0.1`: Cloudflare in front of Groq answers urllib's default agent with `403 error code: 1010` (first real Qwen call hit this). After the fix, first real Qwen call (`cli generate --model groq:qwen/qwen3.8-27b`): 3 valid candidates, 0 rejected, 1 model call. 488 tests pass.
- **Not built:** the Data Scientist is not in the loop (needs raw datafield matrices, OQ-14). Parallel simulations (OQ-3).
- **Real BRAIN on our account (2026-11-03):** `brain auth`, `brain options` and `brain operators` work. `config/fastexpr.yaml` now matches the account's 66 operators exactly (OQ-10 done); the account allows region USA only, universes TOP3000/2000/1000/500/200/TOPSP500 (added to `schemas.UNIVERSES`). Repair prompt `repair-v3` (orthogonalize no longer names operators the account lacks). `brain datafields` gave 4,367 fields for USA/TOP3000/D1. `brain capture` ran one real simulation (acc_01: Sharpe 1.45, fitness 0.78); every mapped field matched, so OQ-2 is closed apart from the weight-concentration unit (BRAIN sends no value on a PASS). The capture is the regression fixture `tests/fixtures/brain_api_capture/` (author and alpha names redacted). `UNVERIFIED_NOTE` removed; margin bps rounded to 6 places. Holdout set to BRAIN's default (OQ-9, OQ-12 done) with a guard against BRAIN period overlap. 474 tests pass.
- **Real BRAIN client: BUILT and verified by one capture.** `brain_api.py`, `config/brain.yaml`, `cli run --client real`, `cli brain auth|options|capture|operators|datafields`. 342 tests pass. The API docs are behind a login, so the client follows BRAIN's own web app (its public JS bundle, read 2026-11-02) and the sample responses it ships, saved in `tests/fixtures/brain_api_samples/` (see SOURCE.md there; don't hand-edit).
- Git repo: https://github.com/Gunin-Midha/QIP (branch `main`). The repo root is `QIP/`.

## Commands (run from `alpha_research/`)

```bash
PYTHONPATH=src python3 -m pytest -q                       # full suite (~5s warm)
PYTHONPATH=src python3 -m alpha_research.cli run \
    --candidates tests/fixtures/candidates/hand_written.json \
    --config tests/fixtures/thresholds_test.yaml --data-dir <tmp dir>
PYTHONPATH=src python3 -m alpha_research.cli bandit \
    --candidates tests/fixtures/candidates/hand_written.json   # Phase 2 driver
python3 tests/fixtures/generate_fixtures.py               # regenerate fixtures (deterministic)

# Phase 3a: hypothesis or paper -> candidates (real model needs ANTHROPIC_API_KEY and `pip install -e ".[llm]"`)
PYTHONPATH=src python3 -m alpha_research.cli generate --hypothesis "<text>" \
    --settings config/generation_settings.example.yaml --n 10 --out generated/batch1.json
#   also: --hypothesis-file F | --paper F.txt [--paper-title T], --datafields F, --dataset D,
#   setting flags (--region ... --end-date) that override --settings, --model M.
#   Offline / no key: --llm scripted --llm-script replies.json (a JSON list of reply strings).
#   Writes batch1.json (loadable by `run`) and batch1.report.json (rejects, rationales, warnings).

# Closed research loop: all agents connected (budgets required; needs BRAIN + model credentials)
PYTHONPATH=src python3 -m alpha_research.cli research --plan config/research_plan.example.yaml \
    --config <thresholds with holdout> --max-simulations 20 --max-llm-calls 60 --per-round 5 --rounds 4

PYTHONPATH=src python3 -m alpha_research.cli prompts --agent all --config config/thresholds.yaml   # rendered agent prompts
PYTHONPATH=src python3 -m alpha_research.cli compare --data-dir data [--out F]       # generator + critic models side by side, from memory

# Real BRAIN (needs BRAIN_EMAIL and BRAIN_PASSWORD in the environment; never in files)
PYTHONPATH=src python3 -m alpha_research.cli brain auth
PYTHONPATH=src python3 -m alpha_research.cli brain options                          # allowed settings (OPTIONS /simulations)
PYTHONPATH=src python3 -m alpha_research.cli brain capture --candidates F --limit 1   # raw responses -> captures/<sim id>/
PYTHONPATH=src python3 -m alpha_research.cli brain operators                        # diff vs config/fastexpr.yaml (OQ-10)
PYTHONPATH=src python3 -m alpha_research.cli brain datafields --region USA --universe TOP3000 --delay 1 --out F
PYTHONPATH=src python3 -m alpha_research.cli run --client real --candidates F --config <config with holdout>
```

- The environment is system Python 3.12.4 with pandas, numpy, scipy, pyyaml and pytest already installed. There is no venv.
- The first import after a cold start can take a minute or two (bytecode compilation).
- `config/thresholds.yaml` now has a holdout (owner's choice 2026-11-03: follow BRAIN's default): the holdout is 2024-01-01..2026-11-03, the time after BRAIN's default in-sample period (2019-01-01..2023-12-31 on our account). A config with a null holdout still makes `run` and `research` exit 2.
- CLI exit codes: 0 ok, 2 config error, 3 schema error, 4 no fixture for a candidate, 5 model unavailable or unusable reply, 6 `generate` produced no valid candidate, 7 BRAIN login or API error.

## Hard rules (from BUILD_SPEC section 0, which still apply)

1. Network code lives only in `llm.py` (the models: Anthropic SDK; Gemini, Groq and OpenRouter over standard-library `urllib`) and `brain_api.py` (BRAIN, standard-library `urllib` only). `llm.py` is the only module that may import a model provider SDK (optional, lazily imported) or name a provider; every other module depends on the `LLMClient` or `BrainClient` protocols. Tests never touch the network: they use `ScriptedLLMClient`, `FixtureBrainClient` or a fake transport for `brain_api`. Agents that use the model (Generator, Pre-Filter, Critic, Debate judge, Repair, Strategy) take an `LLMClient` and go through `generator.ask_json`; every model-written expression must pass `fastexpr.analyze` before it can reach BRAIN.
2. Don't invent BRAIN response fields. Use only the fields in spec 3.2. If something is unclear, leave it null and add `TODO(open-question)`.
3. Never hardcode a policy threshold. Thresholds live in `config/thresholds.yaml`. A `null` threshold means the check returns `passed=None` with note `"not configured"`. It never passes or fails silently.
4. No credentials anywhere in the repo. The model keys are read from the `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY` and `OPENROUTER_API_KEY` environment variables in `llm.py`, and the BRAIN login from `BRAIN_EMAIL` / `BRAIN_PASSWORD` in `brain_api.py`, and nowhere else. Captures (`captures/`, git-ignored) never contain credentials or cookies.
5. A missing value is `None`, never `0`.
6. Every component needs pytest tests, and a component isn't done until they pass.
7. Use only the plain stack: dataclasses, pandas, numpy, scipy, sqlite3, pyyaml, pytest. No heavy frameworks, and no new required dependencies (which is why PnL is stored as CSV, not Parquet). The only addition is the optional `llm` extra (`anthropic`), needed only for the real model.
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
    config/thresholds.yaml      # real config; OQ thresholds null, holdout 2024-01-01..2026-11-03
    config/repair_playbook.yaml # turnover, concentration, self_correlation, sub_universe, drawdown
    config/fastexpr.yaml        # operator list matching our account (OQ-10), groups, base fields, constants
    config/generator.yaml       # generator limits; complexity limits null (OQ-11)
    config/generation_settings.example.yaml  # example --settings file
    config/brain.yaml           # BRAIN connection settings + fixed simulation settings (not thresholds)
    config/agents.yaml          # per-agent model on/off, Pre-Filter + Strategy settings (OQ-13 defaults)
    config/research_plan.example.yaml  # arms (dataset, settings, hypotheses, papers) for `cli research`
    src/alpha_research/
      schemas.py        # contracts + normalize_expression + experiment_hash
      config.py         # load/validate thresholds; ConfigError; require_holdout()
      brain_client.py   # BrainClient protocol, FixtureBrainClient, SubmissionRejected
      brain_api.py      # only BRAIN network code: BrainSession, RealBrainClient, translate(), catalogs
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
      cli.py            # `run`, `bandit`, `generate`, `research` and `brain ...` subcommands
      fastexpr.py       # Phase 3a: Fast Expression lexer/parser, analyze() against config/fastexpr.yaml
      llm.py            # Phase 3a: LLMClient protocol, ScriptedLLMClient, AnthropicLLMClient, GeminiLLMClient, ChatCompletionsLLMClient (Groq/OpenRouter), provider_of (only provider code)
      generator.py      # Phase 3a: AlphaGenerator (hypothesis/paper -> validated CandidateRecords); ask_json
      prefilter.py      # Phase 3b: Cheap Pre-Filter (validator, duplicates, near duplicates, optional model score)
      critic.py         # Phase 3b: Critic (rules + model) and Debate judge; review_and_debate()
      repair_agent.py   # Phase 3b: RepairAgent applies playbook levers (7 mechanical, others via model)
      strategy.py       # Phase 3b: Strategy Evolution (memory summary -> guidance + hypotheses, new prompt version)
      research.py       # Phase 3b: ResearchRun closed loop, BudgetedLLM, agents config, research plan
      compare.py        # model comparison: generator accept rates per model, critic scoring (primary + shadow) from memory
      prompts.py        # Phase 3b: every agent prompt (system + user builders), PROMPT_VERSIONS, brain_context(config)
    tests/
      conftest.py                       # shared fixtures: config, make_config(section__key=...), memory, candidate()
      fixtures/thresholds_test.yaml     # TEST-ONLY placeholders: holdout 2024-01-01..2025-12-31, max_drawdown 0.15, weight_concentration_max 0.10
      fixtures/generate_fixtures.py     # writes brain_results/*.json, pnl/*.csv, candidates/hand_written.json
      fixtures/brain_results/           # 28 fixtures incl. screenshot_aggregate.json
      fixtures/candidates/hand_written.json  # 31 candidates with expected_decision / expected_failure_class
      test_*.py                         # one per module + test_pipeline.py acceptance test
                                        # Phase 3a: test_fastexpr, test_generator, test_llm, test_cli_generate
                                        # BRAIN: test_brain_api (fake transport, no network)
                                        # Phase 3b: test_prefilter, test_critic, test_repair_agent, test_strategy,
                                        #   test_research (RouterLLM fake model + StubBrain over fixtures)
                                        # Models: test_compare; Gemini tests in test_llm (fake transport)
      fixtures/brain_api_samples/       # BRAIN web-app sample responses (not generated; SOURCE.md)
      fixtures/brain_api_capture/       # live responses from our account, redacted (SOURCE.md)
```

## Key behaviours and decisions (don't silently change these)

**Confirmed by the user:**
- **Failure beats pending.** A definitely failed hard gate decides the outcome even when another gate is `None`, and so does a risk veto. Only when nothing failed does a `None` hard gate give `pending`.
- **Invalid candidates.** A domain error in one candidate (unknown region, unbalanced parens, bad delay, holdout overlap) becomes a `status="invalid"` result and then `reject / incomplete`, and the run continues. A structural error in the candidate file (bad JSON, missing keys, wrong types) raises `SchemaError`, and the CLI exits with code 3.
- **Holdout.** Owner (2026-11-03): use BRAIN's default, so the main config's holdout is 2024-01-01..2026-11-03 (move the end date forward over time). Tests and the CLI demo still use `tests/fixtures/thresholds_test.yaml`; tests that need a null holdout use the `no_holdout_config` fixture.
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
- Allowed universes are TOP3000, TOP2000, TOP1000, TOP500, TOP200 and TOPSP500 (our account's USA list from `brain options`). Other regions are not enabled on the account.

**Alpha Generator (Phase 3a, builder choices):**
- The target language is **Fast Expression** only. `CandidateRecord.expression` is what the simulator submits, so Python alphas are not generated (nothing in the repo confirms BRAIN runs them).
- Nothing the model writes reaches the output unless `fastexpr.analyze` and `CandidateRecord.validate` accept it. Unknown operators, wrong argument counts, unknown datafields (when a catalog is supplied) and syntax errors are rejected, then sent back once (`max_repair_rounds`) with the exact errors. `rejected` in the report holds the final failures.
- Operators come only from `config/fastexpr.yaml` (checked against our account with `cli brain operators`, OQ-10). Fix the list there, never in code. Operator names must be lowercase.
- Datafields: with `--datafields` a catalog makes unknown names an error. Without one, unknown names pass with a warning in the report. The prompt lists only the catalog fields most relevant to the hypothesis (`max_datafields_in_prompt`).
- Paper and hypothesis text is wrapped in `<paper>`/`<hypothesis>` tags and the prompt calls it data. The validator is the real defence. PDFs are not read; convert to text first.
- The model may suggest `decay` and `neutralization` per alpha (validated; bad values fall back to the settings with a warning). Region, universe, delay, truncation and dates always come from the settings. There are no built-in defaults for them, so a missing one is a config error (exit 2).
- Candidate ids are `<batch_id>_NNN` with `batch_id = gen<UTC timestamp><4 hex of source hash>`. `prompt_version` comes from `config/generator.yaml` (bump it when prompts change; the bandit uses it as an arm key). `model` is the client's model name.
- Duplicates (same experiment hash) inside a run or across calls on one generator are dropped and listed in the report. The pipeline's own duplicate check still applies later.

**Real BRAIN client (builder choices, owner approved building before verification):**
- Every request sends `Accept: application/json;version=2.0`, as the web app does.
- Poll order follows the web app: `details` → failed; status ERROR/FAIL/TIMEOUT → failed (reason in `message`); `detail` → failed; `progress` (or Retry-After) → running; `children` → failed (multi-sim not supported); COMPLETE/WARNING with `alpha` → fetch results.
- `RealBrainClient.poll` returns `{"status": "running", "retry_after": s}` while running, then `translate()` output in the simulator's raw format with `units: "fraction"` (normalize skips the percent conversion) and a `notes` list (copied onto the result).
- Mapping: `is.*` aggregates; margin × 10,000 → bps; sub-universe from the `LOW_SUB_UNIVERSE_SHARPE` check value/limit; weight concentration from the `CONCENTRATED_WEIGHT` check value; self-correlation from `correlations/self` `max`, else the `SELF_CORRELATION` check; PASS/FAIL checks only; daily PnL from the `daily-pnl` recordset, else by differencing the cumulative `pnl` recordset; yearly rows with stage `IS` only. `testPeriod`, `alpha.test` and `is.startDate` go into notes, not gates. Long/short PnL and drawdown date are not in BRAIN's response and stay None.
- A missing or slow recordset becomes None plus a note, never a failed result. A 4xx on submit raises `SubmissionRejected` → `failed` result. 401 → one re-login; 429 → wait up to `rate_limit_max_wait_seconds`; 5xx/network → ConnectionError/TimeoutError (simulator retries).
- Candidate start/end dates are not sent and no `testPeriod` is sent (OQ-12). `translate` adds a `brain_period: start..end` note from the alpha's settings; `simulator.holdout_leak` turns a complete result whose BRAIN period overlaps the holdout into `invalid` (then reject / incomplete), so a shift in BRAIN's default period can't leak the holdout. Biometric login is not supported. Nothing submits alphas for competition.
- `cli run --client real` checks the holdout before logging in, and uses real time (no VirtualClock).

**Agents and research loop (Phase 3b, builder choices):**
- Round order: Director allocates slots → queued repairs take an arm's slots first → Generator fills the rest → Pre-Filter → Critic → Debate judge (contested only) → Pipeline → Director.update on accept/reject/repair → Repair Agent child queued for the same arm → every `strategy.every_rounds` rounds Strategy Evolution adds arms `(dataset, region, <base>+evoN)` with guidance and new hypotheses.
- Arms are `(dataset, region, prompt_version)`; children keep the parent's prompt_version. The Director warm-starts from `MemoryStore.candidate_outcomes()`.
- Budgets are required CLI flags: `--max-simulations` (counts candidates that reached BRAIN; invalid and duplicate do not count) and `--max-llm-calls` (all agents, via `BudgetedLLM`). Any `LLMUnavailable` stops the run cleanly with a report.
- Critic rule checks: a `ts_*` window <= 0 is blocking (neither the critic model nor the judge can overrule it); no top-level rank/zscore/scale/normalize/quantile/winsorize/group_* and windows > 512 are advisory. A revision becomes child `<id>_c1` (lever `critic_revision`); the original is not simulated or stored.
- Pre-Filter: near duplicate = term-set Jaccard >= threshold with memory or earlier in the batch under the same region/universe/delay/neutralization. Repairs (any candidate with a lever) are exempt.
- Repair: playbook order, skip levers in `lineage_levers`, skip inapplicable ones; child id `<parent>_r<generation>`. Mechanical levers: increase_decay (decay → step.to), tighten_truncation, lengthen_ts_window (2nd positional arg of every ts_* × factor), wrap_in_smoothing, hump, rank_transform, volatility_scale (÷ ts_std_dev(returns, w)); others go to the model.
- Prompts: all agent prompts live in `prompts.py` (Generator's stay in `generator.py`, still `gen-v1`). Bump the agent's entry in `PROMPT_VERSIONS` when its text changes. Thresholds enter prompts only via `brain_context(config)`, never typed in. Every system prompt has `DATA_RULE`, `JSON_RULE` and the exact reply shape; `test_prompts` checks this. Every non-mechanical playbook lever needs an entry in `LEVER_INSTRUCTIONS`.
- None of `config/agents.yaml` changes an accept/reject decision. Slots emptied by Pre-Filter/Critic are not refilled within the round; pending decisions are not re-run.

**Model per agent (owner chose: several providers, generator models as bandit arms, shadow critics available; since 2026-11-03 free models only):**
- `models` in `config/agents.yaml`: `default`, `generator` (list), `critic` (list, preference order), `shadow_critics`, and `prefilter`/`debate`/`repair`/`strategy` (null = default). `cli research --model M` replaces only `models.default`.
- The provider comes from the model id via `llm.provider_of`: `claude-*` = anthropic, `gemini-*` = google, a hosted id `[groq:]<maker>/<model>` = its maker (`groq:qwen/...` = qwen), never the host, because the critic rule is about who made the model. `llm.split_hosted` picks the host (`groq:` prefix, else OpenRouter); hosted ids are routed before `gemini-`/`claude-` checks. Config files never name a provider (this keeps the definition-of-done grep clean).
- Arms are `(dataset, region, prompt_version, generator model)`. Each generator-model arm has its own copy of the plan's sources. Warm start credits candidates to their arm's generator model via `compare.arm_models` (critic records, then lineage).
- **The primary critic never shares a provider with the arm's generator** (the owner's requirement): it's the first listed critic from a different provider. `AgentsConfig.check_models()` refuses configs without one (exit 2). Shadow critics review the same batch and are stored in `critic_reviews` (append-only) but never change what is simulated.
- One `CallCounter` is shared by every model's `BudgetedLLM`. The report adds `model_calls_by_model`, `models` and `comparison` (cumulative over memory).
- Model ids change often: check `gemini-3.6-flash` (aistudio.google.com/rate-limit) and `qwen/qwen3.8-27b` (Groq console) before a run. Free tiers have small daily limits (sources vary), so keep `--max-llm-calls` small.

## Open questions (owner must answer, never guess)

| ID | Topic | Where it shows up |
|---|---|---|
| OQ-1 | Margin definition (BRAIN reports about half of naive) | `metric_validator.margin_check`, recorded but not gated |
| OQ-2 | Mostly DONE 2026-11-03 (live capture matches). Open: weight_concentration unit, since a PASS carries no value; daily-pnl confirmed daily, not cumulative | `brain_api.translate`; `tests/fixtures/brain_api_capture/` |
| OQ-3 | Concurrent sim limit | `simulator.max_concurrent` (no threading yet) |
| OQ-4 | Max drawdown and weight concentration limits | `risk.max_drawdown`, `risk.weight_concentration_max` |
| OQ-5 | Leg balance ratio and min Sharpe after top-PnL removal | `risk.*` |
| OQ-6 | Weak-year floor and max yearly spread | `risk.*` |
| OQ-7 | Min marginal pool Sharpe gain | `diversity.min_marginal_pool_sharpe_gain` |
| OQ-8 | OS/IS ratio floor | `auditor.os_is_ratio_floor` |
| OQ-9 | DONE 2026-11-03: 2024-01-01..2026-11-03 (BRAIN's default; end date to be moved forward) | `holdout.*` |
| OQ-10 | DONE 2026-11-03: list matches the account's 66 operators; rerun `brain operators` if the account level changes | `config/fastexpr.yaml` |
| OQ-11 | Expression complexity limits (depth, operator calls, characters) | `config/generator.yaml` (null = not checked) |
| OQ-12 | DONE 2026-11-03: rely on BRAIN's default period (ends 2023-12-31), no `testPeriod`; results whose BRAIN period overlaps the holdout become invalid. Open later: how to test alphas on the holdout itself | `simulator.holdout_leak`, `brain_api.translate` |
| OQ-13 | Agent builder defaults: near-duplicate similarity 0.9, Strategy Evolution every 2 rounds | `config/agents.yaml` |
| OQ-14 | Raw data source for the Data Scientist (BRAIN's API gives no raw matrices) | `data_scientist.py` is not in the research loop |

Two more TODOs exist: the full universe list (`schemas.py`) and the deflated-Sharpe citation (`auditor.py`).

## Working conventions

- Match the existing style: plain dataclasses, small pure functions, module docstrings that state the method used, and `GateResult(name, passed, value, threshold, note)` for every check.
- When you add or change behaviour, update the tests, `README.md`, `REPORT.md` and this file.
- If you change fixtures, edit `generate_fixtures.py` and rerun it. Don't hand-edit the generated JSON or CSV. Keep each candidate's `expected_decision` and `expected_failure_class` in sync, because `test_pipeline.py::test_pipeline_acceptance` asserts all 31.
- Before saying a change is done, run the full suite and the definition-of-done greps:
  `grep -rniE "requests|http|anthropic|openai" src tests config --exclude-dir=pnl --exclude-dir=__pycache__ --exclude='*llm*.py' --exclude='*brain_api*.py'` (expect nothing) and
  `grep -rniE "password|token|api_key" src tests config --exclude-dir=__pycache__ --exclude='*llm*.py' --exclude='*brain_api*.py'` (expect nothing). `llm.py` and `brain_api.py` may name the provider, endpoints and env variables; also check that no literal key is present: `grep -rnE "\bsk-[A-Za-z0-9_-]{16,}" . --exclude-dir=.git` (expect nothing).
- Runtime data (`data/`, `memory.sqlite`, `data/pnl/`) is git-ignored and must never be committed. Use tmp dirs in tests.

## Natural next steps (only when the user asks)

1. ~~OQ-9 holdout~~ done. Owner sets the OQ-4 limits in `config/thresholds.yaml` (optional: null checks are skipped, not pending).
2. ~~`cli brain capture` (OQ-2)~~ and ~~OQ-12~~ done 2026-11-03.
3. ~~`cli brain operators` (OQ-10)~~ done. Owner runs `cli brain datafields` so the Generator can validate operators and fields.
4. First real `cli research` run on a small budget after steps 1-3; then tune OQ-13 from its report.
5. Parallel simulations once the account's concurrent limit is known (OQ-3); refill emptied slots within a round; re-run pending decisions.
