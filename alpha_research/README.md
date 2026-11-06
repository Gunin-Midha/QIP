# alpha_research: deterministic core (Phases 1 and 2) and the LLM agents (Phase 3)

This is the multi-agent alpha research system: the rule-based backbone, the LLM agents
(Generator, Pre-Filter, Critic, Debate judge, Repair, Strategy Evolution), the research
loop that connects them, and a real BRAIN API client. Only two modules reach the
network: `llm.py` (the model) and `brain_api.py` (BRAIN). Tests and fixture runs use a
fixture-backed BRAIN client and never touch the network. The real client follows the API
as BRAIN's own web app uses it, and was confirmed against a live capture from our account
on 2026-11-03 (OQ-2, see "Real BRAIN client" below).

| Module | Role in the architecture |
|---|---|
| `schemas.py` | CandidateRecord / ResultRecord / DecisionRecord contracts |
| `config.py` | loads and validates `config/thresholds.yaml` |
| `brain_client.py`, `simulator.py` | Backtester: validate, submit/resume, poll, retry, normalize, persist |
| `brain_api.py` | the only module that talks to BRAIN: login, `RealBrainClient`, response translation, operator and datafield catalogs |
| `metric_validator.py` | fitness / yearly / margin tripwires on every result |
| `data_scientist.py` | Datafield Analysis Agent (coverage, update gaps, tails, NaN pattern, redundancy) |
| `evaluator.py` | Quantitative Evaluation: hard gates, failure class, decision |
| `risk.py` | Risk Assessment Agent: drawdown, leg balance, top-PnL removal, concentration, weak year |
| `diversity.py`, `pool.py` | Diversity Agent and the Alpha Pool |
| `auditor.py` | Overfitting Auditor: variant counts, deflated Sharpe, OS/IS tracking |
| `memory.py` | Memory Agent: append-only SQLite store |
| `repair.py` | Repair Agent playbook (lever lookup only; `repair_agent.py` applies them) |
| `lever_stats.py` | lever learning: what each repair lever fixed so far (from memory) ranks the levers |
| `orchestrator.py` | Research Director: Thompson sampling with floor quotas |
| `research_loop.py` | Research Loop Agent: experiment ids, duplicates, lineage, queue, routing |
| `pipeline.py`, `cli.py` | deterministic end-to-end wiring |
| `generator.py` | Alpha Generator Agent: hypothesis or paper to validated candidates (Phase 3a) |
| `fastexpr.py` | Fast Expression parser and validator that gates everything the generator writes |
| `llm.py` | the only module that talks to a model: `LLMClient` protocol, scripted client, real client |
| `prefilter.py` | Cheap Pre-Filter: drops invalid, already-simulated and near-duplicate candidates before BRAIN |
| `critic.py` | Critic (rule checks plus model review) and Debate judge |
| `repair_agent.py` | Repair Agent: applies playbook levers (mechanical rewrites, or the model) |
| `strategy.py` | Strategy Evolution: learns from memory, writes guidance and new hypotheses as a new prompt version |
| `research.py` | the research loop that connects every agent (`cli research`) |
| `prompts.py` | every prompt for the Pre-Filter, Critic, Debate judge, Repair and Strategy agents |

## Setup

Setting this up on a new laptop? Follow `../SETUP.md` (step by step, free keys only).

Python 3.11 or newer. Dependencies: pandas, numpy, scipy, pyyaml, pytest.

```bash
cd alpha_research
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

Without installing, prefix commands with `PYTHONPATH=src`.

The real models need keys in the environment, and Claude also needs the optional extra.
Tests and `--llm scripted` need neither. Set only the keys for the providers your
`config/agents.yaml` uses:

```bash
# Each key is read by llm.py only and never written anywhere. `read -s` keeps it off the
# screen and out of shell history: run it, paste the key, press Enter.
read -s GEMINI_API_KEY && export GEMINI_API_KEY          # gemini-* ids (free tier: Flash models)
read -s GROQ_API_KEY && export GROQ_API_KEY              # groq:... ids (free plan: Qwen, gpt-oss)
read -s NVIDIA_API_KEY && export NVIDIA_API_KEY          # nvidia:... ids (build.nvidia.com: Llama, DeepSeek)
read -s GITHUB_MODELS_TOKEN && export GITHUB_MODELS_TOKEN  # github:... ids (optional: hosted GPT models, ~50 calls/day)
read -s OPENROUTER_API_KEY && export OPENROUTER_API_KEY  # plain <maker>/<model> ids (free ones end in :free)
pip install -e ".[llm]"                                  # only for Claude (claude-* ids, paid)
read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```

`config/agents.yaml` uses free models only (no Mistral, owner 2026-11-04): generators
`gemini-3.6-flash`, `groq:qwen/qwen3.8-27b`, `groq:gpt-oss-120b` (the open-weight GPT
model; ChatGPT's own API has no free tier), `nvidia:deepseek-ai/deepseek-v4.1-flash` and
`nvidia:meta/llama-3.2-90b-vision-instruct`. Each arm's critic is the first listed model from
another maker; Llama on NVIDIA is the default for the agents that run every round, so
Gemini's 20 calls a day go to its own arm. Keys come from aistudio.google.com (Gemini),
console.groq.com (Groq), build.nvidia.com (NVIDIA), github.com/settings/tokens (a
fine-grained token with the Models read permission) and openrouter.ai (OpenRouter).

Model ids change. Before a run:

```bash
PYTHONPATH=src python3 -m alpha_research.cli models --host nvidia --filter deepseek   # what a host serves
PYTHONPATH=src python3 -m alpha_research.cli models --check   # one tiny call to every configured model
```

A hosted id is `<host>:<maker>/<model>`, or `<host>:<model>` when the name says the maker
(`groq:gpt-oss-120b`, `groq:llama-3.3-70b-versatile`; `llm.BARE_MODEL_MAKERS`). A model
that fails during a run is dropped for the rest of it, so a wrong id costs one call.

## Run the tests

```bash
pytest -q
```

## Run the pipeline on the hand-written candidates

`config/thresholds.yaml` sets the holdout to 2024-01-01..2026-11-03, the time after
BRAIN's default in-sample period on our account (OQ-9). A config whose holdout is null makes
the simulator refuse to run (exit code 2), as the spec requires. The demo below uses the
test config, which sets placeholder holdout dates and two placeholder risk limits so the
concentration and drawdown fixtures can fail:

```bash
python -m alpha_research.cli run \
    --candidates tests/fixtures/candidates/hand_written.json \
    --config tests/fixtures/thresholds_test.yaml
```

Options: `--data-dir` (default `$ALPHA_RESEARCH_DATA` or `./data`), `--client fixture|real`
(see "Real BRAIN client"), `--brain-config`, `--capture-dir`, `--fixtures-dir`, `--playbook`.
Exit codes: 0 ok, 2 config error, 3 schema error, 4 no fixture for a candidate, 5 model error,
6 `generate` produced nothing, 7 BRAIN login or API error. Each candidate prints one line:
`candidate_id | decision | failure_class | sharpe | fitness | turnover`.

Fixture runs use a virtual clock, so timeouts resolve instantly. Running the command again
on the same data dir skips everything as duplicates; delete `data/` to start over (it is
git-ignored).

## Phase 2 driver: bandit allocation rounds

```bash
python -m alpha_research.cli bandit --candidates tests/fixtures/candidates/hand_written.json
```

Arms are `(dataset, region, prompt_version)` built from the hand-written candidates.
`prompt_version` is the placeholder `pv0`. Each round allocates `--budget` slots (default
50), feeds each arm's fixture outcomes (accept = pass) and prints the slot counts. It exits
non-zero if the floor quota fails in any round.

## Alpha Generator: make alphas from a hypothesis or a paper

```bash
python -m alpha_research.cli generate \
    --hypothesis "Stocks whose analysts keep raising EPS estimates keep outperforming." \
    --settings config/generation_settings.example.yaml \
    --datafields my_datafields.json --n 10 --out generated/batch1.json
```

Input is one of `--hypothesis TEXT`, `--hypothesis-file F`, or `--paper F` (text or
markdown; convert PDFs first, e.g. `pdftotext paper.pdf paper.txt`). Simulation settings come
from `--settings` and/or flags (`--region --universe --delay --decay --neutralization
--truncation --start-date --end-date --dataset`); flags win. There are deliberately no
built-in defaults for them. The example settings file matches the hand-written fixtures,
so replace it with your account's values.

What it does:

1. For a paper, one model call extracts up to `max_hypotheses_per_paper` testable hypotheses.
2. For each hypothesis, one model call writes alphas in **Fast Expression**. The prompt lists
   the allowed operators, groups and the datafields most relevant to the hypothesis.
3. Every expression is parsed and checked (`fastexpr.py`): known operators with the right
   argument counts, valid syntax, and known datafields when a catalog is given. Failures go
   back to the model once with the exact errors. What still fails is reported, never emitted.
4. Survivors are written as `CandidateRecord`s with `prompt_version` and `model` filled in.
   `batch1.json` loads straight into `cli run`. `batch1.report.json` holds the rationale for
   each alpha, warnings (for example unverified datafields), rejects with their errors,
   duplicates, and the number of model calls.

Datafield catalog (`--datafields`): a JSON list of ids (or of objects with an `id`), or a text
file with one id per line. Without it, unknown field names pass with a warning because they
cannot be checked, and the model may invent fields. Supply one for real use.

Offline: `--llm scripted --llm-script replies.json` replays a JSON list of reply strings, which
is how the tests and demos run without a key.

Limits: the generator stops at candidates. Fixture runs have no result for them, so
`cli run` on generated candidates exits 4 with an explanation; simulate them with
`--client real`. Python alphas are not generated: the pipeline submits Fast Expressions.

## Real BRAIN client

`brain_api.py` talks to `api.worldquantbrain.com` with the standard library only. Log in
details come from the environment and are never stored, logged or captured:

```bash
export BRAIN_EMAIL=you@example.com
export BRAIN_PASSWORD=...
python -m alpha_research.cli brain auth                      # check the login
python -m alpha_research.cli brain options                   # allowed regions, universes, neutralizations...
python -m alpha_research.cli brain capture --candidates generated/batch1.json --limit 1
python -m alpha_research.cli brain operators                 # diff against config/fastexpr.yaml (OQ-10)
python -m alpha_research.cli brain datafields --region USA --universe TOP3000 --delay 1 \
    --out captures/datafields_usa.json                       # then: generate --datafields ...
python -m alpha_research.cli run --client real --candidates generated/batch1.json \
    --config <config with a holdout set>
```

**Where the API shape comes from.** BRAIN's API docs are behind a login, so the client
was built by reading the public JavaScript of platform.worldquantbrain.com: its endpoint
paths, its `Accept: application/json;version=2.0` header, how it polls simulations, and the
sample responses its editor walkthrough ships with. Those samples are kept in
`tests/fixtures/brain_api_samples/` and the tests translate them. A live capture from our
account (2026-11-03) matched the mapping; it is kept, redacted, in
`tests/fixtures/brain_api_capture/` and a test translates it. If BRAIN's format changes, run
`brain capture` again and compare `captures/<sim id>/` with `translated.json`.
`captures/` is git-ignored.

How it works: `submit` posts the candidate's settings and expression to `/simulations` and
keeps the simulation id from the `Location` header. `poll` reports `running` while BRAIN
sends `Retry-After` or a `progress` value, treats `ERROR`, `FAIL`, `TIMEOUT` and
`detail`/`details` bodies as failures (as the web app does), then fetches the alpha's IS
stats, yearly stats, daily PnL and self-correlation. A missing piece becomes `None` with a note, never zero. A 401 triggers
one fresh login, a 429 waits (up to `rate_limit_max_wait_seconds`), and 5xx or network
errors are retried by the simulator. A submission BRAIN rejects (400) becomes a `failed`
result. Accounts that require biometric login are not supported; the error says so.

Mapping (matches BRAIN's samples): `is.sharpe/fitness/turnover/returns/drawdown` as
fractions, `is.margin` times 10,000 into bps, sub-universe Sharpe and cutoff from the
`LOW_SUB_UNIVERSE_SHARPE` check's value and limit, weight concentration from the
`CONCENTRATED_WEIGHT` check's value (unit still unclear), self-correlation from
`correlations/self` `max` (or the `SELF_CORRELATION` check), daily PnL from the `daily-pnl`
recordset, or else by differencing the cumulative `pnl` recordset. Yearly rows are kept
only for stage `IS`. Checks that are neither PASS nor FAIL are left out. Long and short
PnL and the drawdown date are not in BRAIN's response and stay `None`. The test-period
stats (`alpha.test`) are recorded in the notes, not gated.

BRAIN does not take start and end dates; it simulates its own period (2019-01-01 to
2023-12-31 on our account). The result notes record it as `brain_period: start..end`, and
the simulator marks a result `invalid` if that period overlaps the holdout, so a change in
BRAIN's default period cannot leak the holdout (OQ-12). BRAIN's `testPeriod` setting is
not used. Fixed settings sent with every simulation (instrument type,
pasteurization, unit and NaN handling, language) live in `config/brain.yaml`.
Simulations run one at a time (OQ-3). Alphas are never submitted for competition.

## Research loop: all agents connected

```bash
# BRAIN_EMAIL, BRAIN_PASSWORD and the model keys set as above
python -m alpha_research.cli research --plan config/research_plan.example.yaml \
    --config <thresholds with a holdout> --max-simulations 20 --max-llm-calls 60 \
    --per-round 5 --rounds 4 --datafields captures/datafields_usa.json
```

Each round:

1. **Research Director** splits `--per-round` simulation slots across the arms
   (dataset × region × prompt version × generator model) by Thompson sampling, with a
   floor for every arm.
   It starts from the outcomes already in memory, so a second run continues the first.
2. **Repair queue first**: queued repair children from any arm take up to
   `repair.max_share_per_round` (0.5) of the round's slots before the Director allocates
   the rest, so a repair is never stranded in an arm that gets no slot. Then children
   waiting for an arm take that arm's slots. At the start of a run, every `repair`
   decision in memory whose child never reached BRAIN (the run stopped, or no slot) is
   repaired again (`repair.resume_from_memory`).
3. **Alpha Generator** fills the remaining slots from the arm's next hypothesis or paper
   (the plan's sources are used in turn).
4. **Cheap Pre-Filter** drops invalid expressions, experiments already simulated, and near
   duplicates of anything in memory (term-set similarity, `config/agents.yaml`).
   Optionally the model scores the batch.
5. **Critic** checks rules (a non-positive window, which reads the future, always blocks;
   a missing cross-sectional normalization is flagged), then the model reviews the batch:
   simulate, revise or drop. **Debate judge** rules on the contested ones, weighing the
   Generator's rationale against the Critic. A revision becomes a child candidate. The
   deciding (primary) Critic always comes from a different provider than the arm's
   Generator; the other critic models review the same batch as **shadows** (recorded, never
   acted on).
6. **Pipeline** simulates on BRAIN and decides (gates, risk, diversity, pool, memory). The
   Director learns from each accept, reject or repair.
7. **Repair Agent** turns each `repair` decision into a child with one playbook lever
   (`config/repair_playbook.yaml`), queued for the same arm. The lever order is learned
   (see "Repair levers that learn" below).
8. Every `strategy.every_rounds` rounds, **Strategy Evolution** summarizes memory, writes
   guidance for the Generator and proposes new hypotheses. They are released as a new
   prompt version (`gen-v1+evo1`, ...) on new arms, so the Director measures whether the
   guidance helps.

The run stops at `--max-simulations` (candidates that reached BRAIN), at `--max-llm-calls`
(all agents together), or after `--rounds`. A JSON report (`--report`, default
`<data-dir>/<run id>.json`) lists every round: allocation, what each agent dropped and why,
decisions, repairs and evolutions. Nothing submits alphas for competition.

`config/agents.yaml` turns the model on or off per agent (off = rules only) and holds the
Pre-Filter and Strategy settings. Its two builder defaults (near-duplicate similarity 0.9,
evolve every 2 rounds) are OQ-13. None of these settings changes an accept or reject
decision. A research plan lists arms with a dataset label, simulation settings,
hypotheses and paper text files; see `config/research_plan.example.yaml`.

**The models behind the agents.** Each agent has its own model, set in the `models`
section of `config/agents.yaml`. The provider comes from the model id: `claude-*` is
Anthropic, `gemini-*` is Google (Gemini's REST API over the standard library). A null
entry uses `models.default`, and `--model M` replaces `models.default`. `use_model: false`
still switches an agent to rules only. All models share one `--max-llm-calls` budget, and
the report lists calls per model.

- **Generators are compared by the bandit.** Every model in `models.generator` is its own
  arm, with its own copy of the plan's hypotheses, so the models work through the same
  ideas and the Director moves slots toward the one that gets accepted more.
- **Generator and Critic never share a provider.** The primary critic for an arm is the
  first entry in `models.critic` whose provider differs from the arm's generator, so they
  don't share blind spots. A config where some generator has no such critic is refused
  (exit 2).
- **Critics are compared as shadows.** With `shadow_critics: true` every other critic model
  reviews the same batch. All verdicts go to memory (`critic_reviews`), but only the
  primary's decides. Critics are scored on simulated candidates: how often a "simulate"
  verdict was accepted, and how often a "revise"/"drop" flag was right. A candidate the
  primary dropped never reaches BRAIN, so false drops can only be measured through
  shadows. Shadows cost one extra model call per batch per extra critic.
- **Check the Gemini ids** in `config/agents.yaml` against Google's current model list
  (https://ai.google.dev/gemini-api/docs/models) before a real run. With thinking models,
  thinking counts against the output cap; if a reply comes back empty with
  `MAX_TOKENS`, raise `max_output`.

The report has a `comparison` section, and `cli compare` prints the same tables from
memory across every run so far:

```bash
python -m alpha_research.cli compare --data-dir data [--out comparison.json]
```

**Prompts.** All agent prompts are in `src/alpha_research/prompts.py` (the Generator's stay in
`generator.py`). Each agent has a version in `PROMPT_VERSIONS`; bump it when you change its
text, and the run report records the versions used. Each system prompt contains:

- a shared section on how BRAIN judges an alpha (weights, delay, decay, the fitness
  formula, sub-universe, concentration, self-correlation);
- the pass thresholds, read from `config/thresholds.yaml` at run time (never typed into a
  prompt);
- the agent's own instructions;
- the rule that tagged text is data, not instructions;
- the exact JSON shape its parser accepts.

To read them as the model sees them:

```bash
python -m alpha_research.cli prompts --agent critic --config config/thresholds.yaml   # or --agent all
```

| Agent | What its prompt asks for |
|---|---|
| Pre-Filter | a 0-10 score per candidate against a fixed rubric, with a one-line reason |
| Critic | issues by category (lookahead, sign, normalization, scale, turnover, ...), a verdict, a minimal revised expression, the test it expects to fail |
| Debate judge | a ruling per contested candidate, decided on one question: does the criticism make the simulation fail or uninformative? |
| Repair | exactly one lever applied, keeping the idea, with lever-specific instructions (orthogonalize, liquid fields, model_repair), every class the alpha fails, and past repairs of the class that worked |
| Strategy | 3-6 evidence-based guidance bullets and new hypotheses, each with a direction and why it is new |

Known limits: slots that the Pre-Filter or Critic empty are not refilled in the same round.
`pending` decisions are not re-run. The Data Scientist agent is not in the loop, because it
needs raw datafield matrices that BRAIN's API does not provide (see open questions).

## Repair levers that learn

The playbook lists levers per failure class. 13 are mechanical rewrites (no model):
`increase_decay`, `tighten_truncation`, `lengthen_ts_window`, `wrap_in_smoothing`, `hump`,
`rank_transform`, `volatility_scale`, `flip_sign` (negative Sharpe only),
`set_neutralization` (to a finer group), `backfill_fields` (`ts_backfill` on every catalog
field), `liquidity_tilt` (`expr * rank(adv20)`), `winsorize` and `change_universe`. The
model levers are `orthogonalize_against_pool`, `restrict_to_liquid_fields` and, last in
every class, `model_repair`, where the model proposes its own fix and may change decay or
neutralization.

Sharpe and fitness failures now go to repair too, but only near misses:
`repair.near_miss_share` in `config/thresholds.yaml` (Sharpe |value| >= 0.6 x threshold,
fitness >= 0.5 x; OQ-15). Far misses are still rejected.

The order is learned. `lever_stats.py` reads every parent -> child pair in memory (nothing
new is stored) and scores each repair: 1 if the target failure is fixed (or the child is
accepted), 0.5 if it moved toward passing, 0 otherwise, minus 0.5 if the child broke a test
the parent passed. A child also counts, at `secondary_weight` (0.5), for every other class its parent failed
(so a lever aimed at sub-universe that clears concentration on the way gets credit), and a
combination `combo:a+b+c` credits each part with a third of that weight. Per (failure
class, lever) this updates a Beta posterior. A combination with positive evidence for a
class becomes a lever the agent can choose for that class: its parts are applied in order
with their playbook steps, and parts already in place are skipped. Levers are
tried best first, and a lever that also addresses another class the alpha fails (including
BRAIN's own failed checks, such as `CONCENTRATED_WEIGHT`) gets extra weight. With no
evidence the playbook order is used. Settings: the `repair` section of `config/agents.yaml`
(`order: learned | thompson | playbook`). Repairs that worked are also shown to the repair
model as examples, so `model_repair` learns in context.

```bash
python -m alpha_research.cli repair --data-dir data --candidate-id <id> [--candidate-id <id2>] \
    --datafields captures/datafields_usa_top3000_d1.json --out candidates/rescue.json
python -m alpha_research.cli run --client real --candidates candidates/rescue.json --config config/thresholds.yaml
python -m alpha_research.cli levers --data-dir data     # what each lever has fixed so far
```

`cli repair` writes one child per applicable mechanical lever (every failing class, best
ranked first; experiments already in memory are skipped), with `parent_id` and `lever`
set, so the results of a hand rescue become evidence for the learner too.

## Pipeline guards: BRAIN checks, overfitting, repair budgets, diversity (2026-11-04)

- **BRAIN's checks decide.** With `gates.use_brain_checks: true` every PASS/FAIL check BRAIN
  reports becomes a hard gate `brain:<CHECK>`. A failed CONCENTRATED_WEIGHT is a definite
  concentration failure even though our own limit (OQ-4) stays null; we set no limits of
  our own for what BRAIN tests.
- **Overfitting guard at accept.** The accepted alpha's Sharpe is deflated for the number
  of trials (`auditor.trials_scope`: `dataset` by default = every simulation on its
  dataset; `lineage` or `all`) and must stay above `auditor.min_deflated_sharpe` (0.0),
  else `reject / other`. With 5 years of data this bites only with many trials, so the
  repair budget and an out-of-sample test (OQ-12) remain the main guards.
- **Distance to accept** = the sum of each failing class's normalized shortfall (an
  unmeasured failure adds `repair.unmeasured_failure_penalty`). The learner's reward is half
  "did the target class get fixed" and half "did the whole alpha get closer".
- **Repair budget per original idea** (`config/agents.yaml` `repair`): `max_depth` 2,
  `max_trials_per_root` 4, `patience` 2 (stop after two children that do not get closer),
  and at most `max_resume` 3 lost repairs resumed per run, closest to accept first.
  `cli repair` obeys it too (`--ignore-budget` to override; extra simulations still count
  as trials).
- **Diversity.** The Director gives no arm more than `director.max_arm_share` (0.4) of a
  round while others are below it. The generator lists 120 fields: 70% most relevant
  (crowding breaks ties) and 30% drawn across the other datasets, each annotated with
  coverage, how many BRAIN alphas use it, and vector/group type, as hints, not limits
  (`gen-v2`). Strategy Evolution sees per-dataset trial counts, the least-tried datasets and
  the fixes that worked; its prompt (`strategy-v3`) requires at least half the new
  hypotheses on least-tried datasets and at most one building on an accepted alpha, and a
  hypothesis on a dataset no arm covers goes to every arm. The Critic flags neutralization
  NONE (advisory, never blocking).

## Model failover and repair fixes (2026-11-04, after run `rs20261104073248`)

- **A failed model is dropped, not fatal.** A model call that still fails after the
  client's retries (a free tier's daily quota, an outage) marks that model down for the
  rest of the run (`llm.ModelDown`, report `models_down`). Its generator arms get no more
  slots; critics switch to the next critic from another provider, else to rules only; the
  judge, Pre-Filter, Repair and Strategy agents switch to the next model that is up, else to
  their rules. The run stops only when every generator model is down.
- **Repair children skip the Critic's model and the judge.** They get the rule checks only,
  so what BRAIN measures is exactly what the lever changed.
- **Lever order rules:** `flip_sign` first (applies only to negative Sharpe), `model_repair`
  last (the catch-all); everything between is ordered by the learned scores.

## Validator and round fixes (2026-11-04, after run `rs20261104163055`)

- `fastexpr.analyze` requires optional arguments by name (`winsorize(x, std=4)`, `ts_backfill(x, 60, k=1)`) and, with a typed catalog (`fastexpr.catalog(ids, meta)`), allows an event (VECTOR) field only inside `vec_avg`/`vec_sum`. Both were BRAIN errors in that run (5 of 19 simulations).
- `rank_transform` treats `-rank(x)` as already ranked.
- Slots of a generator model that goes down mid-round, and (for `director.refill_passes` extra passes) slots emptied by the Generator, Pre-Filter or Critic, go back to the Director in the same round.
- A Strategy hypothesis on a catalog dataset no arm covers gets its own arms; a candidate's `dataset` follows its fields (`generator.signal_dataset`).
- Critic and judge (`critic-v3`, `judge-v3`) revise fixable defects instead of dropping them.

## Regenerating fixtures

```bash
python tests/fixtures/generate_fixtures.py
```

The script is deterministic (fixed seed). It writes `brain_results/*.json`, `pnl/*.csv` and
`candidates/hand_written.json`. `screenshot_aggregate.json` holds the IS summary
screenshot values (aggregate plus yearly rows for 2019 to 2023).

## Decision rules worth knowing

- A hard gate that definitely failed beats one that cannot be evaluated. For example,
  Sharpe 0.50 with self-correlation not yet computed gives `reject / sharpe`, not `pending`
  (Sharpe 1.10 gives `repair / sharpe`: a near miss).
  A risk veto also beats `pending`.
- Domain errors in a single candidate (unknown region, unbalanced parentheses, holdout
  overlap) become a `status="invalid"` result and then `reject / incomplete`, and the run
  continues. Structural errors in the candidate file (unreadable JSON, missing required
  keys, wrong types) stop the CLI with exit code 3.
- Duplicates are skipped before simulation and reported as `duplicate`.
- Every candidate that reaches BRAIN registers a variant for the auditor, including
  failures, as section 10.1 requires.

## Open questions (owner to answer)

- **OQ-1 Margin definition.** BRAIN's reported margin is about half the naive
  `returns / (turnover * 252)`. Margin is recorded alongside the naive value and is not gated.
- **OQ-2 Real BRAIN response format.** Confirmed by a live capture (2026-11-03); `daily-pnl`
  holds daily values. Still open: the unit of `weight_concentration`, because BRAIN's
  `CONCENTRATED_WEIGHT` check carries no value when it passes.
- **OQ-3 Concurrent simulation limit.** Account-specific. `max_concurrent` is 1 and no
  threading is implemented.
- **OQ-4 Drawdown and concentration limits.** Policy choice.
- **OQ-5 Leg balance and top-PnL-removal thresholds.** Policy choice.
- **OQ-6 Weak-year floor and yearly Sharpe spread.** Policy choice.
- **OQ-7 Minimum marginal pool Sharpe gain.** Policy choice.
- **OQ-8 OS/IS ratio floor.** Policy choice.
- **OQ-9 Holdout period.** Done: 2024-01-01..2026-11-03, following BRAIN's default period.
  Move the end date forward as time passes.
- **OQ-10 Operator list.** Done (2026-11-03): `config/fastexpr.yaml` matches the 66
  operators `cli brain operators` lists for our account, with BRAIN's argument lists.
  Rerun it if the account level changes; the generator discards anything not listed.
- **OQ-11 Expression complexity limits.** Depth, operator calls and length are `null` (not
  checked) in `config/generator.yaml` until the owner chooses them.
- **OQ-12 Simulation period on BRAIN.** Done: BRAIN's default period ends before the
  holdout, and any result whose BRAIN period overlaps the holdout becomes `invalid`. Still
  open for later: how to test alphas on the holdout itself.
- **OQ-13 Agent defaults.** `config/agents.yaml` ships with builder defaults so the loop
  works: near-duplicate similarity 0.9 and Strategy Evolution every 2 rounds. Review them.
- **OQ-15 Repair near misses.** Builder defaults (2026-11-04): Sharpe and fitness failures go
  to repair when |value| >= 0.6 / 0.5 of the threshold (`repair.near_miss_share` in
  `config/thresholds.yaml`). Review them, and the lever-learning settings in the `repair`
  section of `config/agents.yaml`.
- **OQ-16 Overfitting and repair budget defaults.** `auditor.min_deflated_sharpe` 0.0 with
  `trials_scope: dataset`; repair `max_depth` 2, `max_trials_per_root` 4, `patience` 2;
  `director.max_arm_share` 0.4; generator `datafield_exploration_share` 0.3. Review them.
- **OQ-14 Data Scientist input.** The datafield audit needs a dates × instruments matrix
  per field. BRAIN's API (as far as the web app shows) does not provide raw data, so decide
  the source: BRAIN's per-field statistics, an outside data source, or probe simulations.
