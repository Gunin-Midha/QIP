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
- **First closed research run (2026-11-03, `rs20261103171304`):** all agents ran end to end on real BRAIN with free models. Gemini wrote 2 alphas, Qwen critiqued them, BRAIN simulated both: a 5-day reversal (Sharpe 0.84, fitness 0.39, self-corr 0.73) went to repair / self_correlation and the Repair Agent queued an orthogonalized child; a cash-flow-to-assets change (Sharpe -0.76) was rejected / sharpe. The run then stopped on Groq's free limit of 7,000 input tokens per minute (the generator prompt is ~3,350 tokens, ~2,400 of them the 300 listed datafields). Built since: Gemini and hosted clients wait and retry on HTTP 429 when the host gives a wait (`try again in 7.95s`, Gemini `retryDelay`) of at most `llm.RATE_LIMIT_MAX_WAIT_SECONDS` (60), up to `RATE_LIMIT_RETRIES` (3), logging each wait; longer or unstated waits still stop the run; retries count as one budgeted call. Note: the repair queue is in memory only, so repairs queued in a stopped run are lost (the Director's warm start and the duplicate check do carry over). Proposed next (not built): pause a rate-limited model instead of stopping; persist the repair queue; `max_datafields_in_prompt` 300 -> ~80; more free hosts (Mistral, NVIDIA NIM as a second Qwen host). Open owner question: should a candidate that also fails Sharpe/fitness still go to repair for a higher-priority class?
- **Second research run (2026-11-03, `rs20261103175655`), after the 429 retry:** ran to the simulation budget (6 sims, 10 model calls), no stop. No accepts. Best: Qwen `rank(ts_corr(returns, cash_flow_from_operations / assets, 21)) * sign(cash_flow_from_operations / assets)` Sharpe 1.14, fitness 1.36, failed sub-universe. Red flag: Qwen chose neutralization NONE on a sparse fundamental (`rank(ts_delta(cash_flow_from_operations, 21) / assets)`): returns 38%, drawdown 67%, fitness 1.45 > Sharpe 0.83; only rejected because Sharpe failed, since `risk.max_drawdown` is null (OQ-4). Reversal ideas hit self-correlation again (0.80). Director put all round-2 slots in the one untried arm (floors round to 0 at 3 slots / 4 arms), so the queued repair got no slot. The judge's rulings are not in the report. Owner options: set max_drawdown, restrict model-chosen neutralization, let queued repairs claim a slot, show judge rulings.
- **Research plan v1 + third run (2026-11-04, `rs20261103184506`):** `config/research_plan.v1.yaml` (3 arms: option8 IV spread / put fear / put-call volume; analyst4 EPS revisions / forward earnings yield; model51 idio vol / customer momentum / earnings-news sentiment; every named field checked against the catalog; neutralization SUBINDUSTRY/INDUSTRY). Command: `--plan config/research_plan.v1.yaml --max-simulations 18 --max-llm-calls 60 --per-round 6 --rounds 3`. Best alpha so far (Gemini, option8): `spread = divide(subtract(implied_volatility_call_60, implied_volatility_put_60), add(implied_volatility_mean_60, 0.001)); group_zscore(ts_decay_linear(spread, 10), subindustry)` with SUBINDUSTRY, decay 10: Sharpe 2.20, fitness 1.43, turnover 21.5%, self-corr 0.31, drawdown 4.2%, every year positive; fails sub-universe (0.55 vs 0.95) and BRAIN's CONCENTRATED_WEIGHT (value 0.1538) -> repair / sub_universe. Sibling `group_rank(ts_mean(call_30 - put_30 IV, 5), subindustry)`: Sharpe 1.64, fitness 0.73 -> reject / fitness. Run stopped on Gemini **HTTP 503 "high demand"** (not handled by the 429 retry) while the repair was being written, so the repair was lost; 2 sims, 4 model calls. See "Paused to-do" below.
- **Learning Repair Agent (2026-11-04):** diagnosis: no repair child had ever reached BRAIN (run 1 queue lost on stop, run 2 child got no slot, run 3 repair call hit Gemini 503), and the near misses were rejected because the playbook had no sharpe/fitness entries. Built: 5xx retry (10/30/60 s) in `llm._post_json`; `repair.near_miss_share` in thresholds (OQ-15: Sharpe 0.6, fitness 0.5 of threshold, by |value|); 6 new mechanical levers + `model_repair`; `lever_stats.py` learns lever order from parent->child pairs in memory; queued repairs take up to half of each round's slots first; lost repairs resumed at run start; `cli repair` / `cli levers`. Rescue round 1 (`candidates/rescue_20261104.json`, 10 sims) gave **the first accept**: `rs20261103184506r01a0_002_r1x` (IV spread, backfill+rank+tilt by adv20), Sharpe 1.38, fitness 1.01, sub-universe 0.81/0.60, in the pool; it is thin (fitness 1.01, 2021-23 yearly Sharpe ~0.7-1.05). The cash-flow alpha is a market bet and is dropped. Learner fixes after round 1: secondary-class credit (weight `secondary_weight`), combo parts credited 1/k, learned combos become choosable levers (parts already in place are skipped). Round 2 (5 sims): no accept; the IV family oscillates between fitness and sub-universe.
- **Pipeline guards (2026-11-04, owner: improve the pipeline not single alphas; BRAIN's checks only; no overfitting; keep diversity):** `gates.use_brain_checks` (every BRAIN PASS/FAIL check is a hard gate `brain:<CHECK>`); accept-time deflated-Sharpe gate (`auditor.min_deflated_sharpe` 0.0, `trials_scope` dataset, OQ-16); distance to accept (`lever_stats.distance`) drives the reward (0.5 class + 0.5 overall), resume order (`max_resume` 3) and a per-lineage repair budget (`max_depth` 2, `max_trials_per_root` 4, `patience` 2), all IV-family repairs now closed; Director `max_arm_share` 0.4; generator `gen-v2` lists 120 fields (30% exploration across datasets, coverage/crowding/type hints, never filters); Strategy `strategy-v3` must put half its hypotheses on least-tried datasets; Critic advisory for neutralization NONE. 540 tests pass. Next: a closed-loop `cli research` run, not hand rescues.
- **Fourth run + fixes (2026-11-04, `rs20261104073248`):** 14 sims, 34 model calls, no accept; 7 fresh ideas far from the bar, the IV-spread family again the only near miss (repairs reached fitness 0.81, then depth limit). Stopped on Gemini's free daily limit (20 requests) with Qwen up. Fixed: model failover (`llm.ModelDown`; `BudgetedLLM` marks a model down for the run; `FallbackLLM` for judge/Pre-Filter/Repair/Strategy; critic = next other-provider critic that is up, else rules; arms of a down generator model get no slots via `Orchestrator.allocate(only=)`; stop only when every generator model is down; report `models_down`); repair children get Critic rule checks only (no model review, judge or shadows); `rank_levers` pins `flip_sign` first and `model_repair` last (it was winning on the secondary bonus from its prior). 547 tests pass.
- **More free models (2026-11-04, owner: no Mistral; add GPT, Llama, DeepSeek):** hosts `nvidia` (build.nvidia.com, `NVIDIA_API_KEY`) and `github` (GitHub Models, `GITHUB_MODELS_TOKEN`) in `llm.HOSTS`; bare hosted ids `<host>:<model>` when the name says the maker (`llm.BARE_MODEL_MAKERS`: gpt- is namespaced for the host, llama/deepseek/qwen sent as is), `MAKER_ALIASES` (meta-llama -> meta, deepseek-ai -> deepseek); `llm.list_models(host)`; `cli models --host H [--filter S]` and `cli models --check` (one tiny call per configured model). Config: generators Gemini Flash, Groq Qwen, `groq:gpt-oss-120b`, `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter, generator only: ~50 free calls/day; OpenRouter's free list has no Llama/DeepSeek); critics gpt-oss, Qwen, Gemini last; default and strategy gpt-oss on Groq. NVIDIA left out: its chat endpoint gave 0 bytes in 120 s even to plain curl (2026-11-04) while its model list worked; ids if it recovers: `nvidia:meta/llama-3.2-90b-vision-instruct`, `nvidia:deepseek-ai/deepseek-v4.1-flash`. Groq's free plan has no Llama chat model (only prompt-guard). Next for Llama/DeepSeek: OpenRouter free models (`OPENROUTER_API_KEY`, ~50 calls/day). ChatGPT's own API has no free tier, so GPT = gpt-oss (or GitHub Models). Not run against the real hosts yet. 550 tests pass.
- **Fifth run + fixes (2026-11-04, `rs20261104163055`):** first run to finish all 4 rounds (19 sims, 41 calls; Gemini out of quota all run, failover worked). No accept; resumed repairs reached Sharpe 1.05-1.18 then the depth limit; 9 fresh ideas far misses (best gpt-oss news P/E, Sharpe 0.90, turnover 83%). Fixed: validator requires optional args by name and VECTOR fields inside `vec_*` (5 of 19 sims were these BRAIN errors; replay over memory: 5 caught, 0 false); `rank_transform` sees through a leading minus; unused slots (model went down: always; Generator/Pre-Filter/Critic drops: `director.refill_passes`, 1) go back to the Director in the same round; Strategy hypotheses on an uncovered catalog dataset get their own arms, and a candidate's `dataset` follows its fields (`generator.signal_dataset`); `critic-v3`/`judge-v3` revise fixable defects instead of dropping. Proposed, not built (why some models look weak): show generators their arm's past results, one temperature for all models, Director reward by distance to accept, compare models within a dataset. 559 tests pass.
- **Consultant upgrade (2026-11-10..11, code and docs done; remaining items in "Consultant upgrade: next steps" below):** account is now consultant (`level` GOLD). Discovery run in one BRAIN session saved `alpha_research/captures/consultant_20261110/` (git-ignored; old user-level data in `captures/baseline_user_20261103/`). Found: 11 regions (ALL, USA, GLB, EUR, ASI, CHN, JPN, AMR, IND, DEU, GBR) with their own universes/delays; new neutralizations (REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING, FAST, SLOW, SLOW_AND_FAST; COUNTRY in GLB/EUR/ASI/AMR); 82 operators (+16); in-sample now **2014-01-01..2023-12-31**; LOW_SHARPE limit **1.58** (was 1.25); new checks LOW_2Y_SHARPE, IS_LADDER_SHARPE, PROD_CORRELATION, POWER_POOL_CORRELATION, DATA_DIVERSITY, REGULAR_SUBMISSION (quota 4), CLUSTER_TEST/UNITS (WARNING), MATCHES_PYRAMID/THEMES, OSMOSIS_ALLOCATION, POWER_POOL_DESCRIPTION_*; checks inside a fresh alpha are provisional (LOW_SHARPE WARNING there, FAIL in `GET /alphas/<id>/check`); new `is.investabilityConstrained` / `is.riskNeutralized`; options also offer language PYTHON, simulationMode QUICK, maxTrade/maxPosition; USA/TOP3000/D1 has 289 datasets and >10,000 fields (BRAIN stops at offset 10,000). BRAIN asks for biometric verification on every new login. Done so far: biometric login (`BrainSession(on_biometric=)`, CLI prompts in a terminal); `schemas.REGION_SETTINGS` per region + `neutralizations_for` (Generator, Repair Agent use it; `set_neutralization` skips factor neutralizations); fastexpr.yaml 82 ops; `RealBrainClient` fetches `/check` (`brain.yaml` `fetch_submission_check`, `check_max_wait_seconds`) and `translate(check=)` prefers its checks, notes WARNING value/limit and investability/risk-neutralized stats; `list_datafields` goes per dataset above `DATAFIELD_OFFSET_CAP` with progress lines, `list_datasets`; evaluator maps the new checks; `gates.brain_checks_ignored` (config); thresholds sharpe_min 1.58; candidate start dates 2014-01-01 in plans/settings; EUR fixtures now TOP2500; fixture `tests/fixtures/brain_api_capture_consultant/` (redacted, SOURCE.md). 2026-11-11: tests read the Sharpe limit from config (1.58); `schemas.GROUP_NEUTRALIZATIONS`; Generator (`gen-v3`) and Repair (`repair-v5`) prompts list the region's neutralizations (groups and factor ones apart); `cli brain datasets`; `scripts/brain_discovery.py` (every region/delay, largest universe, one login, per-dataset retries, rerunnable, output `captures/consultant_catalogs/`); README/REPORT/SETUP updated; OQ-17. 579 tests pass. Catalogs for all 18 region/delays downloaded 2026-11-11 (capped at 20 datasets each; see next steps item 2).
- **Regions, new checks and repairs (2026-11-11):** owner asked: repair the new checks, use the new neutralizations, every agent knows the consultant features, each agent tuned per region. Owner chose new failure classes and all four region pieces now; research plan v2 waits for the full catalogs. Built: `ResultRecord.check_values` / `alt_stats` (BRAIN's value+limit per check, risk-neutralized and investability-constrained stats); classes `recent_sharpe` (LOW_2Y_SHARPE, IS_LADDER_SHARPE) and `prod_correlation` (PROD_CORRELATION); 7 `neutralize_<name>` levers, region-aware `change_universe`, model levers `restore_recent_performance` / `differentiate_from_production`; `gates.limits_from_brain` + `regions:` overrides (`Config.for_settings`, pipeline uses the candidate's region); per-region lever stats with shrinkage (`region_prior_strength`); Director `region_floor_share` / `max_region_share`; `brain_context(config, regions)`, Generator market line + `region_notes` per round, Strategy `regions` + `region_guidance`. Prompt versions prefilter-v3, critic-v4, judge-v4, repair-v5, strategy-v4, gen-v3. OQ-18. 609 tests pass.
- **Research plan v2 (2026-11-11, DRAFT, not committed, not run on BRAIN):** `config/research_plan.v2.yaml`, 37 arms over all 18 region/delays (datasets from the 20 downloaded per market, every named field checked against that market's catalog; option8 kept in USA D1). Loader v2 layout: `defaults`, `regions.<R>.enabled/settings/catalogs/arms` (settings defaults <- region <- arm, checked against `REGION_SETTINGS`; arms unique per (dataset, region, delay)). Arm key is now `(dataset, region, prompt_version, model, delay)`, reported by `research.arm_label` as `ds/REGION/D<delay>/pv/model`. Catalog per market: `ResearchRun(markets=)`, `fastexpr.catalog_for` in Pre-Filter/Critic/Repair, Generator per arm; `cli research` reads plan catalogs (`_plan_catalogs`), `--datafields` is the fallback. Strategy Evolution adds new-dataset arms only in markets whose catalog has the dataset. `tests/test_plan_v2.py`. 631 tests pass. Open: 148 Director arms with 4 generator models is far beyond a free-tier run (use `enabled`).
- **Region rotation (2026-11-11, owner: ~20 alphas per run, runs take turns over regions, all regions equal across runs; not committed):** `director.regions_per_run` 3 and `region_balance_since` 2026-11-11 (agents.yaml, OQ-19); `MemoryStore.simulations_by_region(since)`; `ResearchRun._pick_regions` (fewest simulations, then oldest last simulation, then plan order); only chosen regions' arms live and repairs taken/resumed; `orchestrator.balance_slots` splits each allocation across chosen regions by fewest simulations, then the Director allocates within each; report `regions_this_run`, `simulations_by_region`. Run size = `--max-simulations 20`. Offline check: 3 runs took USA/ALL/GLB, EUR/ASI/CHN, JPN/AMR/IND, 5-7 sims each. `tests/test_rotation.py`. 643 tests pass.
- **Seeds (2026-11-11, not committed):** plan `seeds:` (`from_memory: id` or inline expression + dataset/region/delay/settings), `research.load_seeds`, `ResearchRun(seeds=)` / `_prepare_seeds`: new root `<id>_seed` / `seed_<hash>` on the arm's dates, simulated first once its region is in the run, `credit=False` (no Director update), then normal repair; report `seeds`. v2 plan seeds the IV spread (memory) and the owner's skew+VRP alpha (BRAIN web app: USA/TOP3000/D1, decay 0, truncation 0.01, INDUSTRY; Sharpe 1.72, fitness 1.15). The latter needs `opt6_ivhvxernratio` (option6) in the USA D1 catalog: `scripts/brain_discovery.py --regions USA --datasets option6` (new `--datasets` flag); until then it is skipped as invalid. `tests/test_seeds.py`. 657 tests pass.
- **Teammate setup:** `SETUP.md` at the repo root (venv, keys via `read -s`, BRAIN checks, model smoke tests, first research run, troubleshooting).
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
PYTHONPATH=src python3 -m alpha_research.cli repair --candidate-id ID --datafields captures/datafields_usa_top3000_d1.json --out candidates/F.json  # hand-rescue variants
PYTHONPATH=src python3 -m alpha_research.cli levers --data-dir data                  # what each repair lever fixed so far
PYTHONPATH=src python3 -m alpha_research.cli models --check                          # one tiny call to every configured model (--host H lists a host's models)

# Real BRAIN (needs BRAIN_EMAIL and BRAIN_PASSWORD in the environment; never in files)
PYTHONPATH=src python3 -m alpha_research.cli brain auth
PYTHONPATH=src python3 -m alpha_research.cli brain options                          # allowed settings (OPTIONS /simulations)
PYTHONPATH=src python3 -m alpha_research.cli brain capture --candidates F --limit 1   # raw responses -> captures/<sim id>/
PYTHONPATH=src python3 -m alpha_research.cli brain operators                        # diff vs config/fastexpr.yaml (OQ-10)
PYTHONPATH=src python3 -m alpha_research.cli brain datafields --region USA --universe TOP3000 --delay 1 --out F
PYTHONPATH=src python3 -m alpha_research.cli brain datasets --region USA --universe TOP3000 --delay 1 --out F   # field counts
PYTHONPATH=src python3 scripts/brain_discovery.py [--max-datasets 20]  # all regions/delays, one login -> captures/consultant_catalogs/
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
4. No credentials anywhere in the repo. The model keys are read from the `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `NVIDIA_API_KEY`, `GITHUB_MODELS_TOKEN` and `OPENROUTER_API_KEY` environment variables in `llm.py`, and the BRAIN login from `BRAIN_EMAIL` / `BRAIN_PASSWORD` in `brain_api.py`, and nowhere else. Captures (`captures/`, git-ignored) never contain credentials or cookies.
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
    scripts/brain_discovery.py  # all-region catalog download, one BRAIN login (hand-run)
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
    config/research_plan.v2.yaml       # every region/delay: region blocks, catalog per delay (draft)
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
      repair_agent.py   # Phase 3b: RepairAgent applies playbook levers (13 mechanical, others via model); variants() for hand rescues
      lever_stats.py    # lever learning: repair events from memory, Beta posterior per (class, lever), rank_levers
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
- Failure priority: self_correlation, prod_correlation, turnover, sharpe, fitness, recent_sharpe, sub_universe, concentration, drawdown, leg_imbalance, weak_year, other. `top_pnl_removal` maps to `other`, and pool correlation maps to `self_correlation`. BRAIN's LOW_2Y_SHARPE/IS_LADDER_SHARPE map to `recent_sharpe`, PROD_CORRELATION/POWER_POOL_CORRELATION to `prod_correlation`, DATA_DIVERSITY and unknown checks to `other`.
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
- Arms are `(dataset, region, prompt_version, generator model, delay)` (Phase 2 bandit: `(dataset, region, prompt_version)`); children keep the parent's prompt_version. The Director warm-starts from `MemoryStore.candidate_outcomes()`.
- Budgets are required CLI flags: `--max-simulations` (counts candidates that reached BRAIN; invalid and duplicate do not count) and `--max-llm-calls` (all agents, via `BudgetedLLM`). A model that fails after its retries is down for the rest of the run (`ModelDown`; see failover above); the run stops cleanly with a report on budget exhaustion or when every generator model is down.
- Validator (`fastexpr.analyze`): optional arguments (`name=default` in the signature) must be written by name; with a typed catalog (`fastexpr.catalog`, types from `load_datafield_meta`) a VECTOR field may appear only inside a `vec_*` operator. Both rules come from BRAIN errors, not guesses. Unknown keyword names are allowed (our signatures can be incomplete, e.g. `ts_backfill(..., ignore="NAN")`).
- Critic rule checks: a `ts_*` window <= 0 is blocking (neither the critic model nor the judge can overrule it); no top-level rank/zscore/scale/normalize/quantile/winsorize/group_* and windows > 512 are advisory. A revision becomes child `<id>_c1` (lever `critic_revision`); the original is not simulated or stored. Repair children (any lever) get the rule checks only.
- Pre-Filter: near duplicate = term-set Jaccard >= threshold with memory or earlier in the batch under the same region/universe/delay/neutralization. Repairs (any candidate with a lever) are exempt.
- Repair: levers ranked by `lever_stats.rank_levers` (posterior mean of past rewards per (class, lever), ties in playbook order; + `secondary_weight` x estimate for each other failing class that lists the lever, BRAIN's failed checks included; `flip_sign` pinned first, `model_repair` pinned last), skip levers in `lineage_levers`, skip inapplicable ones; child id `<parent>_r<generation>`. Reward: 1 fixed/accepted, 0.5 improved slack, 0 else, -0.5 collateral (floor 0); incomplete child = 0. Each child is evidence for its target class (weight 1), each other class the parent failed (weight `secondary_weight`), and each part of a `combo:a+b+c` lever (1/k of those). `LeverStats.learned_combos(cls)` (positive evidence) are added to the class's options; `RepairAgent._combo` applies parts with their playbook step and skips parts that don't apply. Mechanical levers: increase_decay, tighten_truncation, lengthen_ts_window, wrap_in_smoothing, hump, rank_transform, volatility_scale, flip_sign (only Sharpe < 0), set_neutralization (only to finer), backfill_fields (non-base fields), liquidity_tilt (× rank(adv20)), winsorize, change_universe (only to smaller); others go to the model, whose reply may set decay/neutralization. `variants()` gives one child per applicable mechanical lever with ids free in memory, skipping experiments already in memory.
- BRAIN checks: `gates.use_brain_checks` adds a hard gate `brain:<CHECK>` per PASS/FAIL check (`evaluator.check_class` maps it to a class, unknown = other). Accept then passes `Pipeline._overfitting` (deflated Sharpe over `auditor.trials_scope` trials, `research_loop.lineage_root/lineage_members`) before diversity.
- Repair budget: `lever_stats.family_budget` (depth, trials per lineage root, patience on `distance`) is checked in `RepairAgent.repair` and `variants` (`--ignore-budget` in `cli repair`).
- Sharpe/fitness failures are repaired only if |value| >= `repair.near_miss_share` x threshold (thresholds.yaml, optional section; missing = no limit).
- Research round: queued repairs (any arm, round-robin) take up to ceil(`repair.max_share_per_round` x slots) first, then the Director allocates the rest. `resume_from_memory`: at run start, repair decisions with no child in memory and an arm in the plan are re-repaired.
- Prompts: all agent prompts live in `prompts.py` (Generator's stay in `generator.py`, still `gen-v1`). Bump the agent's entry in `PROMPT_VERSIONS` when its text changes. Thresholds enter prompts only via `brain_context(config)`, never typed in. Every system prompt has `DATA_RULE`, `JSON_RULE` and the exact reply shape; `test_prompts` checks this. Every non-mechanical playbook lever needs an entry in `LEVER_INSTRUCTIONS`.
- None of `config/agents.yaml` changes an accept/reject decision. Unused slots go back to the Director within the round: a generator model's slots when it goes down (always), slots emptied by the Generator, Pre-Filter or Critic for `director.refill_passes` extra passes; refill batch ids end `f<n>`. Pending decisions are not re-run.
- Strategy Evolution: a hypothesis whose dataset is in the catalog but has no arm gets new arms `(dataset, region, evo version, model, delay)` per generator model in each market whose catalog has the dataset (settings of that market's arms); unknown datasets go to every arm. A generated candidate's `dataset` is the dataset most of its non-base fields belong to (the arm's on a tie).

**Model per agent (owner chose: several providers, generator models as bandit arms, shadow critics available; since 2026-11-03 free models only):**
- `models` in `config/agents.yaml`: `default`, `generator` (list), `critic` (list, preference order), `shadow_critics`, and `prefilter`/`debate`/`repair`/`strategy` (null = default). `cli research --model M` replaces only `models.default`.
- The provider comes from the model id via `llm.provider_of`: `claude-*` = anthropic, `gemini-*` = google, a hosted id `[<host>:]<maker>/<model>` or `<host>:<model>` = its maker (`groq:qwen/...` = qwen, `groq:gpt-oss-120b` = the GPT maker, `nvidia:meta/...` = meta), never the host, because the critic rule is about who made the model. `llm.split_hosted` picks the host (`groq:` prefix, else OpenRouter); hosted ids are routed before `gemini-`/`claude-` checks. Config files never name a provider (this keeps the definition-of-done grep clean).
- Arms are `(dataset, region, prompt_version, generator model, delay)`. Each generator-model arm has its own copy of the plan's sources. Warm start credits candidates to their arm's generator model via `compare.arm_models` (critic records, then lineage).
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
| OQ-15 | Repair near-miss shares (Sharpe 0.6, fitness 0.5) and lever-learning settings, builder defaults 2026-11-04 | `config/thresholds.yaml` `repair`, `config/agents.yaml` `repair` |
| OQ-16 | Overfitting and diversity defaults: min_deflated_sharpe 0.0, trials_scope dataset, repair max_depth 2 / max_trials_per_root 4 / patience 2, max_arm_share 0.4, datafield_exploration_share 0.3 | `config/thresholds.yaml` `auditor`, `config/agents.yaml`, `config/generator.yaml` |
| OQ-17 | Consultant checks (builder defaults 2026-11-10): `gates.brain_checks_ignored` (submission-only checks, esp. POWER_POOL_CORRELATION); WARNING checks not gated; CLUSTER_TEST meaning unknown; BRAIN limits other than USA D1 unknown | `config/thresholds.yaml` `gates`, `evaluator.check_class` |
| OQ-18 | Region/new-check builder defaults (2026-11-11): factor-lever priors in the playbook, `near_miss_share.recent_sharpe` 0.6, `region_prior_strength` 2, `region_floor_share` 0.5, `max_region_share` 0.6, `limits_from_brain` true, per-region universe size order; DATA_DIVERSITY meaning unknown (class other) | `config/repair_playbook.yaml`, `thresholds.yaml`, `agents.yaml`, `repair_agent.UNIVERSE_ORDER_BY_REGION` |
| OQ-19 | Region rotation builder defaults (2026-11-11): `regions_per_run` 3, `region_balance_since` 2026-11-11 (earlier USA-only runs not counted), balance unit = region (not region/delay) | `config/agents.yaml` `director` |
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

## Consultant upgrade: next steps (updated 2026-11-11)

1. ~~Fix `tests/test_prompts.py`~~, ~~consultant fixture SOURCE.md~~, ~~`cli brain datasets`~~, ~~prompts list the new neutralizations~~ (operators were already rendered from fastexpr.yaml; regions come from the settings), ~~docs and OQ-17~~: done 2026-11-11.
2. ~~Owner runs `scripts/brain_discovery.py`~~ **done 2026-11-11**: all 18 region/delay catalogs complete, no gaps, in `captures/consultant_catalogs/` (git-ignored). The uncapped run did ALL D1 (141 datasets, 29,254 fields) and USA D0 (124, 16,919; ~3 h) in full, then was stopped in USA D1 (~80,600 fields) and rerun with the cap: the script now saves every full dataset list but fields for only `--max-datasets` (20) datasets per region/delay (`pick_datasets`: top per category, most used, 30% least explored), one part file per dataset (`parts/<tag>/`), so a rerun or a larger cap only fetches what is missing; old full catalogs (no parts dir) are skipped. Datasets / fields (20 chosen datasets each): USA D1 289 / 4,021; GLB D1 137 / 3,844; EUR D0 69 / 1,659, D1 173 / 2,918; ASI D1 160 / 1,803; CHN D0 28 / 2,768, D1 75 / 2,313; JPN D0 64 / 745, D1 206 / 753; AMR D0 82 / 2,439, D1 203 / 2,827; IND D1 141 / 908; DEU D0 70 / 1,210, D1 172 / 1,423; GBR D0 73 / 688, D1 177 / 1,360. Keep the old `captures/datafields_usa_top3000_d1.json` (10,000-capped but has fields outside the 20 chosen datasets) next to the new USA D1 file.
3. ~~Full suite + greps, commit~~ done 2026-11-11 (609 tests).
4. **Research plan v2** drafted 2026-11-11 (see status; owner to review datasets/hypotheses and arm count before the first run) (owner 2026-11-11: big overhaul, after the catalogs): every region and delay the account allows, region blocks with default settings that arms inherit, datasets per region from `captures/consultant_catalogs/`, the new neutralizations and operators. Later (owner): steer toward less-explored regions, datafields and operators (e.g. exploration quotas from catalog `alphaCount`/`userCount`/`pyramidMultiplier`, operator-usage counts from memory).
5. Owner questions: re-simulate the pool alpha (Sharpe 1.38, judged on 2019-2023 at user-level limits) under consultant settings? New research plan v2 using the new datasets/regions? OQ-17.

## Per-region tuning: further ideas (suggested 2026-11-11, not built)

- Generator few-shot per region: show the region's best accepted expressions (not other regions') as examples.
- Critic/Pre-Filter calibration per region: track how often each model's "simulate" was right per region and weight its score by that.
- Region-specific datafield selection: rank catalog fields by what passed in that region (fields' accept rate), with the exploration share kept.
- Default settings learned per region: decay/neutralization/truncation that passed most often there become the plan's region defaults (shown to the model, never forced).
- Cross-region transfer as an explicit lever: re-simulate an accepted alpha in another region (field availability checked against that region's catalog), so a good idea is tested where it has not been, and its result is new evidence for both.
- Per-region prompt versions: arms already carry the region, so Strategy can evolve prompts in one region without touching another.

## Paused to-do (owner, 2026-11-04: "make a note, continue later")

1. ~~Retry temporary server errors~~ done 2026-11-04.
2. **Rescue by hand**: round 1 done (1 accept). Round 2 file written (`candidates/rescue_20261104_round2.json`); owner runs it, then `cli levers`.
3. **Set `risk.weight_concentration_max`** (OQ-4) so our Risk Manager catches concentration itself; BRAIN reported 0.1538 for the best alpha (it sends a value only when its check fails).
4. Later, from earlier runs: ~~pause a rate-limited or failing model instead of stopping the run~~ done 2026-11-04 (failover); ~~persist the repair queue~~ and ~~queued repair slot~~ done 2026-11-04 (resume from memory, repairs first); show the Debate judge's rulings in the report; optionally stop models choosing neutralization NONE; `max_datafields_in_prompt` 300 -> ~80; more free hosts (Mistral, NVIDIA NIM). Owner questions still open: should a candidate that also fails Sharpe/fitness go to repair; `director.floor_share` for small runs.

## Natural next steps (only when the user asks)

1. ~~OQ-9 holdout~~ done. Owner sets the OQ-4 limits in `config/thresholds.yaml` (optional: null checks are skipped, not pending).
2. ~~`cli brain capture` (OQ-2)~~ and ~~OQ-12~~ done 2026-11-03.
3. ~~`cli brain operators` (OQ-10)~~ done. Owner runs `cli brain datafields` so the Generator can validate operators and fields.
4. First real `cli research` run on a small budget after steps 1-3; then tune OQ-13 from its report.
5. Parallel simulations once the account's concurrent limit is known (OQ-3); ~~refill emptied slots within a round~~ done 2026-11-04; re-run pending decisions.
