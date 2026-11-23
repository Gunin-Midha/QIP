# How one research run works, step by step

This document follows one `cli research` run from the command line to the last line it
prints, in the order the code does things. A real alpha from memory is followed the whole
way: the EUR overnight-vs-intraday reversal from run `rs20261114174347`, which later led
to the accepted alpha `eur_37`.

Code references are `file:function`, under `src/alpha_research/`. Numbers come from the
current config files (`config/*.yaml`). Change a config file and the behaviour changes
with it.

```
cli research
 └─ 0. Setup: load config, plan, catalogs, models, BRAIN session, memory
 └─ 1. Build the run: agents, arms, Director warm start, region rotation
 └─ 2. Start of run: seeds, recover lost results, resume open repairs
 └─ 3. For each round (up to --rounds):
 │     3.1 Seeds first
 │     3.2 Queued repairs take up to half the slots
 │     3.3 Director allocates the remaining slots to arms
 │     3.4 For each arm with slots:
 │     │     a. Generator writes alphas (model call)
 │     │     b. Pre-Filter
 │     │     c. Critic (model call) + Debate judge (model call, contested only)
 │     │     d. Queue for BRAIN; when 8 are queued (or the round ends): flush
 │     3.5 Flush: simulate together on BRAIN, then for each alpha in order:
 │     │     e. Simulator: submit, poll, fetch results and checks, store
 │     │     f. Evaluate: gates, risk, decision, overfitting, diversity, store
 │     │     g. Director credit
 │     │     h. Repair Agent writes a child (queued for a later round)
 │     3.6 Refill emptied slots (once)
 │     3.7 Every 2nd round: Strategy Evolution
 └─ 4. Lesson Writer (model call)
 └─ 5. Report file + summary lines
```

---

## 0. Setup (`cli.py:_research`)

The command used for plan v4:

```
PYTHONPATH=src python3 -m alpha_research.cli research --plan config/research_plan.v4.yaml \
    --max-simulations 20 --max-llm-calls 60 --per-round 5 --rounds 6
```

In this order:

1. **Thresholds** (`config/thresholds.yaml`). The run stops with exit 2 if no holdout is
   set. This check comes before any login or model call.
2. **Repair playbook** (`config/repair_playbook.yaml`): the list of levers for each
   failure class.
3. **Agents config** (`config/agents.yaml`): the model for each agent, Director settings
   and repair-learning settings.
4. **Research plan** (`config/research_plan.v4.yaml`): regions, the arms in each region
   (a dataset plus settings plus hypotheses) and the seeds.
5. **Fast Expression spec** (`config/fastexpr.yaml`): BRAIN's 82 operators, groups and
   base fields. Every expression written by a model is checked against it.
6. **Generator config** (`config/generator.yaml`): prompt version `gen-v8`, how many
   fields to list, crowding settings and category lessons.
7. **Datafield catalogs**, one per market. The plan's `catalogs:` entries point to
   `captures/consultant_catalogs/datafields_<region>_<universe>_d<delay>.json`. A catalog
   gives each field's dataset, type, coverage, BRAIN alpha count and description.
8. **Budgets**: `--max-simulations` counts alphas that reached BRAIN; `--max-llm-calls`
   counts every model call from every agent.
9. **Model clients**, built before logging in to BRAIN, so a missing API key fails early.
10. **BRAIN session**: the saved cookie in `~/.config/qip/brain_session.txt`, or a fresh
    login (`BRAIN_EMAIL`/`BRAIN_PASSWORD`, plus biometric approval).
11. **Memory**: `data/memory.sqlite`.

## 1. Building the run (`research.py:ResearchRun.__init__`)

1. **Model check.** Each generator model needs a critic from a different provider. The
   run refuses to start otherwise.
2. **Shared call counter.** One `CallCounter` is shared by every model, so
   `--max-llm-calls 60` caps all agents together. A model that fails after its retries is
   marked down for the rest of the run (`ModelDown`).
3. **Latest lessons** are read from memory (`memory.latest_lessons()`). The Lesson Writer
   wrote them at the end of the previous run.
4. **BRAIN context.** `prompts.brain_context` builds one text, shown to every agent's
   model, that explains how BRAIN judges an alpha: fitness formula, sub-universe test,
   2-year Sharpe, production correlation, our thresholds, each region's settings, and the
   limits BRAIN reported on our earlier alphas in each market (`strategy.brain_limits`).
5. **The agents**: Pre-Filter, one Critic per critic model plus a rules-only Critic,
   Debate judge, Repair Agent and Strategy Evolution.
6. **Arms.** An arm is `(dataset, region, prompt_version, generator model, delay)`. Every
   plan arm is copied once per generator model: plan v4's arms × 4 generator models
   (Gemini Flash, Qwen on Groq, gpt-oss on Groq, Nemotron on OpenRouter). Each copy works
   through the same hypotheses in the same order.
7. **Director** (`orchestrator.py:Orchestrator`): each arm starts with a Beta(1, 1) belief
   about its pass rate.
8. **Warm start** (`_warm_start`). Every decided alpha in memory (accept, reject or repair)
   counts as one success (accept) or one failure (anything else) for its arm. An alpha
   whose arm no longer exists, for example after a prompt version bump, is credited to the
   same `(dataset, region, model, delay)` under the current prompt version.
9. **Region rotation** (`_pick_regions`). `director.regions_per_run: 3`: the 3 plan
   regions with the fewest simulations since `region_balance_since` (2026-11-11), ties
   going to the region whose last simulation is oldest. Arms in other regions get no slots
   in this run.

   Console: `--- regions this run: GLB, EUR, ASI (fewest simulations so far; done: USA 31, ...)`

**Example.** Run `rs20261114174347` picked GLB, EUR and ASI. Our alpha's arm was
`pv1 / EUR / gen-v6 / nvidia/nemotron-3-ultra-550b-a55b:free / D1`, the EUR price-volume
dataset written by the Nemotron model.

## 2. Start of the run (`_run`)

Three steps before round 1, in this order:

1. **Seeds** (`_prepare_seeds`). Each plan seed becomes a fresh root candidate on the
   arm's dates, after a validator check and a check that the same experiment is not
   already in memory. It is queued only for regions in this run. (Plan v4's seeds are
   USA, so none ran here.)
2. **Recover lost results** (`_recover_submissions`). An alpha that timed out on our side
   but may have finished on BRAIN (`memory.unfinished_submissions()`) is polled once. If
   it finished, it is decided like any other result and credited to its arm. This costs no
   simulation budget.
3. **Resume open repairs** (`_resume_repairs`). A repair decision in memory with no child
   yet gets a child now: closest to accept first, at most `repair.max_resume` (3), only
   for regions in this run. The child joins its arm's repair queue.

   Console: `--- resumed 2 repair(s) from memory: [...]`

## 3. A round (`_round`)

`slots = min(--per-round, simulations left)`, so 5 per round here.

### 3.1 Seeds first
Queued seeds take one slot each. A seed's outcome does not update the Director
(`credit=False`): a seed is not one of the arm's own ideas.

### 3.2 Queued repairs take up to half the slots
`repair.max_share_per_round: 0.5`, so ceil(0.5 × 5) = 3 of the 5 slots at most. The
Repair Agent's children from earlier rounds (or resumed from memory) go first, taken in
turn from each arm. Near misses are the cheapest chance of an accept, and an arm the
Director gives no slot would otherwise never run its repairs.

### 3.3 The Director allocates the rest (`_allocate`, `orchestrator.py`)
This is how it decides what to work on:

1. **Split by region** (`balance_slots`). Each slot goes, one at a time, to the run's
   region with the fewest simulations so far (memory plus this run). This keeps all
   regions level across runs.
2. **Within each region, Thompson sampling** (`Orchestrator.allocate`):
   - **Floor:** every live arm gets `floor(floor_share × slots / n_arms)` slots
     (`floor_share 0.2`). With a few slots and many arms this rounds to 0.
   - **Remaining slots, one at a time:** draw a random pass rate from each arm's Beta
     (successes + 1, failures + 1) and give the slot to the highest draw. An arm with
     accepts usually draws high. An arm with no history has a wide distribution and
     sometimes draws high, so new arms keep getting explored.
   - **Cap:** no arm gets more than `max_arm_share` (0.4) of the slots while another arm
     is below the cap.
3. **Live arms only:** arms whose generator model is down, arms in regions outside this
   run, and arms that Strategy Evolution replaced get nothing.

Console: `--- round 1: 2 repair(s), then 3 new: pv1/EUR/D1/gen-v6/nemotron 1, ...`

The batch id is `<run id>r<round>a<arm index><refill tag>`. **Example:**
`rs20261114174347r01a27` is round 01, arm number 27 in the Director's list.

### 3.4 One arm's slots (`_arm`)
If the arm has queued repairs, they fill its slots first. The Generator writes the rest.

#### a. Generator (`generator.py:AlphaGenerator.generate`), model call
The arm's next **source** (a hypothesis or a paper) is taken and rotated to the back, so
the arm cycles through its hypotheses. A paper first goes through one extra model call
that extracts hypotheses.

The **system prompt** (`build_system_prompt`) is built fresh for each batch. In order:
1. What an alpha is (cross-sectional, no look-ahead, normalize with rank, zscore or
   group_*).
2. Language rules and **the full operator list** with signatures, the groups and the base
   fields.
3. **Market** line (`EUR TOP2500, delay 1`) and **neutralizations** the region allows:
   group ones, factor ones and NONE. A factor neutralization or NONE needs a
   `neutralization_reason`.
4. **Datafields** (`select_datafields`), up to `max_datafields_in_prompt` (120) fields:
   - fields named in the hypothesis come first;
   - then the fields that best match the hypothesis text (whole words in the field id and
     BRAIN's description, rare words weighted higher). The arm's own dataset ranks first
     and crowded fields rank last;
   - `datafield_exploration_share` (30%) of the list is drawn from other datasets, one
     dataset at a time, never crowded fields;
   - each field is tagged `[cov 63%, 12 alphas, crowded, N of ours failed production
     correlation]`;
   - identifier/text fields are hidden unless named.

   A field counts as **crowded** if it is in the top 10% of its dataset by BRAIN alpha
   count (and has at least 50 alphas), or if one of our alphas built on it failed
   PROD_CORRELATION in this market (`strategy.prod_correlation_fields`, read from memory
   every batch).
5. **Lessons for this kind of data**: the config's category lessons
   (`generator.yaml category_lessons`: `all`, plus `pv`/`analyst`/... by the arm dataset's
   BRAIN category), then the **Lesson Writer's** latest lessons for this region and
   category.
6. **Lessons from earlier rounds** (the arm's guidance):
   - **BRAIN's bar in this market**: limits seen on our alphas there, e.g. Sharpe 1.58,
     fitness 1.0;
   - **region notes** from memory (`strategy.region_notes`): accepted N of M there, most
     common failures, neutralizations tried with their accept rates, repair levers that
     worked;
   - Strategy Evolution's guidance, once an evolution has run.
7. A note that the hypothesis is data, not instructions, and the JSON-only reply rule.

The **user prompt** (`_generate_prompt`) is the `<hypothesis>` text plus "Write k
distinct alpha expressions...". The requested reply is
`{"alphas": [{"expression", "hypothesis", "rationale", "family", "decay"?, "neutralization"?, "neutralization_reason"?}]}`.

**Checking the reply.** Each expression goes through `fastexpr.analyze`: known operators
with the right arguments, fields in this market's catalog, VECTOR fields only inside
`vec_*`, optional arguments by name, no reserved variable names (`delta`), no unused
variables, at most 64 operators, no scientific notation. An alpha built only on
identifier fields is rejected too. Rejects go back to the model **once** with the exact
errors (one more call). The survivors become `CandidateRecord`s:
- region, universe, delay, truncation and dates always come from the arm's settings;
- decay and neutralization come from the model if valid, otherwise from the arm;
- `dataset` follows the fields the expression actually uses (`signal_dataset`);
- id `<batch id>_NNN`, plus `prompt_version` and `model`;
- a second alpha with the same experiment hash in the same batch is dropped.

**Example.** The Nemotron arm got 1 slot. Its alpha was
`rs20261114174347r01a27_001`: the 10-day sum of (intraday return minus overnight return),
ranked within country, with the hypothesis "Overnight gaps that fade intraday indicate
noise-driven pressure that reverses". The arm's settings gave EUR, TOP2500, delay 1,
truncation 0.08 and dates 2014-01-01..2023-12-31; the alpha ran with decay 4 and SECTOR.

#### b. Pre-Filter (`prefilter.py:PreFilter.run`), no model call by default
For each candidate, the first check that fails drops it:
1. the validator again, against the candidate's own market catalog;
2. **exact duplicate:** memory already has a result for this experiment hash (expression
   with whitespace removed, plus region, universe, delay, decay, neutralization,
   truncation and dates);
3. **near duplicate:** the term set (operators, names, numbers) has Jaccard similarity
   ≥ 0.9 with an alpha in memory or earlier in this batch under the same region, universe,
   delay and neutralization. Repairs are exempt.
4. (off: `use_model: false`) a model score from 0 to 10.

The example passed all checks.

#### c. Critic, then Debate judge (`critic.py:review_and_debate`), model calls
**Primary critic** = the first model in `models.critic` from a **different provider** than
the arm's generator that is not down (`_critics_for`). For the example, the generator was
Nemotron (NVIDIA), so the critic was `groq:gpt-oss-120b`.

1. **Rule checks first** (no model):
   - **blocking:** a `ts_*` window ≤ 0. Nothing can overrule it.
   - **advisory:** no top-level rank/zscore/group_*, windows > 512 days, neutralization
     NONE, a fresh alpha with a factor neutralization, a fresh alpha reading only
     price/volume fields (`price_volume_categories: [pv]`).
   - **sparse rule:** a news/sentiment field, or a field with coverage < 50%, used in
     `ts_*` without backfill is rewritten automatically to `ts_backfill(field, 20)`.
2. **Model review** (fresh alphas only; repair children get the rule checks only, so
   each lever's effect is measured on its own). One call for the whole batch. The system
   prompt is `critic_system`: role, the BRAIN context, common pitfalls, how to give
   simulate/revise/drop, when a factor neutralization is justified, JSON shape. The user
   prompt has each candidate's expression, settings, hypothesis, rationale and the rule
   issues so far. A `revise` must include a full revised expression, which is validated
   again. If it fails validation, the revision is thrown away.
3. **Debate judge** (`DebateJudge.judge`), one call, **only for contested candidates**
   (verdict ≠ simulate). It weighs the author's rationale against the critic's issues and
   rules simulate (original), revise (the critic's version) or drop. Drop is allowed only
   for look-ahead or an idea that cannot be tested. It cannot clear a blocking rule issue.
4. A revision becomes a **child** `<id>_c1`, lever `critic_revision`, and the original is
   neither simulated nor stored. Every review is stored in `critic_reviews`.

**Example.** The critic's verdict was `revise`: *"sign: The hypothesis expects a negative
10-day sum to be bullish, but the expression ranks the raw sum, giving high rank to
positive sums."* Its revision negated the sum:

```
intraday = divide(close, open); overnight = divide(open, ts_delay(close,1));
diff = subtract(intraday, overnight); sum10 = ts_sum(diff, 10);
group_rank(multiply(-1, sum10), country)
```

The judge upheld the revision. So `rs20261114174347r01a27_001_c1` (generation 1, lever
`critic_revision`, model `groq:gpt-oss-120b`) went to BRAIN, and `_001` was never stored.

#### d. Queue for BRAIN (`_simulate`)
Alphas wait in `sim_queue`. The queue is flushed when it holds
`simulator.max_concurrent` (8) alphas, and at the end of every round. The budget checks
count queued alphas as used, so a run never queues more than `--max-simulations`.

### 3.5 Flush: BRAIN, then decisions (`_flush` → `pipeline.py:Pipeline.process_many`)
Console: `--- BRAIN: 5 alphas at once`

1. **Duplicate check** again, against memory and within the batch.
2. **Simulator** (`simulator.py:Simulator.run_many`), see e.
3. Then, **one alpha at a time in queue order**: register the variant, evaluate (f),
   credit the Director (g), repair (h).

#### e. Simulator: submit, poll, fetch, store
For each alpha:
1. **Local validation** (region, universe, delay, parentheses, dates against the
   holdout). An error gives an `invalid` result, which is not submitted and costs no
   budget.
2. **Store the candidate** → `candidates` table.
3. **Submit** (`brain_api.py:RealBrainClient.submit`): `POST /simulations` with
   `{"type": "REGULAR", "settings": {region, universe, delay, decay, neutralization,
   truncation, + fixed settings from brain.yaml}, "regular": "<expression>"}`. Dates are
   not sent: BRAIN uses its own in-sample period (2014-01-01..2023-12-31).
   BRAIN's simulation id → `pending_submissions` table.
4. **Poll** until done: up to 8 at once; the next is submitted as soon as one finishes.
   Every 60 s:
   `BRAIN: 5 running, 1 of 5 finished: r01a27_001_c1 6m40s 45%, ...`. A simulation past
   `max_poll_seconds` (1200 s) becomes a `timeout`, which is recovered at a later run's
   start. Network errors are retried 3 times (1, 2, 4 s). A BRAIN-side error listed in
   `brain.yaml retry_error_messages` is resubmitted once.
5. **Fetch** (console `BRAIN: <id> done on BRAIN, fetching its results and checks`): the
   alpha (`/alphas/<id>`: IS aggregates, checks), yearly stats, daily PnL, self-correlation
   and the full `/check` (submission checks, which can take minutes).
6. **Translate** to our `ResultRecord`. Values become fractions, margin becomes bps, and
   the sub-universe values come from the check. Each PASS/FAIL check goes to `checks`, its
   value and limit to `check_values`, the risk-neutralized and investability-constrained
   stats to `alt_stats`, and WARNING/PENDING checks to notes only.
7. **Holdout guard:** if BRAIN's period overlaps our holdout (2024-01-01..), the result
   is marked `invalid`.
8. **Metric validator** notes (fitness recomputed, yearly consistency, margin).
9. **Store the result** → `results` table, including the full daily PnL.

Console: `BRAIN: rs20261114174347r01a27_001_c1 finished (3 of 5), complete: Sharpe 0.91, fitness 0.41, turnover 0.20`

**Example result**, BRAIN alpha `mLa2vm82`:

| | value | BRAIN limit |
|---|---|---|
| Sharpe | 0.91 | 1.58 (FAIL) |
| Fitness | 0.41 | 1.0 (FAIL) |
| Turnover | 20.4% | 1%..70% (PASS) |
| Sub-universe Sharpe | 0.37 | 0.47 (FAIL) |
| 2-year Sharpe | -0.47 | 1.58 (FAIL) |
| **Risk-neutralized Sharpe** | **2.28** | not a gate |
| Investability-constrained Sharpe | 0.48 | not a gate |

Yearly Sharpe was positive from 2014 to 2021 (0.96 to 1.72), then -0.87 in 2022 and 0.13
in 2023. SELF_CORRELATION and PROD_CORRELATION were PENDING, so they were only noted.

#### f. Evaluate (`Pipeline._evaluate`, `evaluator.py`, `risk.py`)
1. **Register the variant** → `variants` table, under the candidate's family label.
   The overfitting check counts trials from all simulated alphas in memory, failures
   included.
2. **Config for this market** (`config.for_settings(region, delay)`): per-region
   overrides from `thresholds.yaml regions:`.
3. **Gates** (`evaluator.run_gates`):
   - our five gates (sharpe, fitness, turnover_range, self_correlation, sub_universe).
     Their limits come from BRAIN's own limit for that alpha when reported
     (`limits_from_brain`), and **BRAIN's PASS/FAIL decides them when reported**
     (`brain_checks_only`);
   - plus one hard gate `brain:<CHECK>` for every PASS/FAIL check BRAIN reported;
   - plus `submit:<CHECK>` gates for submission-only checks (REGULAR_SUBMISSION,
     MATCHES_THEMES, ...), decided last.
4. **Risk** (`risk.assess`): drawdown, leg balance, top-PnL removal, concentration and
   weak year. Most limits are null (OQ-4/5/6), so they record "not configured". Under
   `brain_checks_only` a veto is recorded, not applied.
5. **Decision** (`evaluator.decide`), in this order:
   1. status not complete → **reject / incomplete**;
   2. any hard gate failed → pick the **failure class**:
      - the first failed class in the priority list `self_correlation, prod_correlation,
        turnover, sharpe, fitness, recent_sharpe, sub_universe, concentration, drawdown,
        ...`;
      - **Sharpe routing** (`sharpe_route`): a negative raw Sharpe that is a near miss is
        sent to the Sharpe levers ("sign first"). If BRAIN's risk-neutralized Sharpe is a
        near miss and ≥ 0.5 above raw, it is sent there too ("factor neutralization
        first");
      - no playbook entry for the class → **reject**;
      - not a near miss (`repair.near_miss_share`: |Sharpe| < 0.6 × limit, |fitness| <
        0.5 × limit, or recent Sharpe < 0.6 × limit; for Sharpe the larger of raw and
        risk-neutralized counts) → **reject**;
      - otherwise → **repair / <class>**;
   3. risk veto (only if applied) → reject;
   4. any hard gate unknown → **pending**;
   5. a submission check failed → **pending** (the alpha itself is good);
   6. otherwise → **accept**.
6. **Accept only:** the overfitting auditor (deflated Sharpe over every simulation in the
   same dataset, `auditor.trials_scope`), then pool diversity (PnL correlation with pool
   alphas, marginal pool Sharpe). Under `brain_checks_only` both are recorded, not applied.
7. **Power Pool tag** (`power_pool.candidate_reason`): if the alpha is not accepted but
   POWER_POOL_CORRELATION passed, no other check outside the submission checks failed, and
   at least one regular check is a WARNING, a `power_pool_candidate:` reason is added.
   The decision does not change.
8. **Store the decision** → `decisions` table (decision, class, every gate with value,
   threshold and note, risk reasons, reasons).
9. **Accept only:** admit to the **pool** → `pool` table, with the daily PnL as
   `data/pnl/<id>.csv`.
10. Print the decision line: `<id> | <decision> | <class> | <Sharpe> | <fitness> | <turnover>`

**Example.** The failed classes were sharpe, fitness, recent_sharpe and sub_universe.
Sharpe has the highest priority. Raw Sharpe 0.91 is below the near-miss bar of
0.6 × 1.58 = 0.948, but the risk-neutralized 2.28 is above it and 1.37 higher than raw, so
the factor route applies:

```
rs20261114174347r01a27_001_c1 | repair | sharpe | 0.9100 | 0.4100 | 0.2040
reasons: ... "factor neutralization first: risk-neutralized Sharpe 2.28 vs raw 0.91",
         "playbook has sharpe levers"
```

#### g. Director credit (`_after_simulation`)
`sims_used` and the region's count go up by one. On **accept / reject / repair** the arm
(or its evolved successor) gets one success (accept) or one failure (anything else). The
next allocation's draws reflect it immediately. Pending results and seeds are not credited.

The example's repair counted as one failure for the Nemotron pv1/EUR arm.

#### h. Repair Agent (`repair_agent.py:RepairAgent.repair`), model call only for model levers
Only when the decision is **repair**:
1. **Lineage budget** (`lever_stats.family_budget`). No child if:
   - it would exceed generation `max_depth` (2);
   - the lineage already used `max_trials_per_root` (4) repair simulations;
   - `patience` (2) children in a row got no closer to accept;
   - any member of the lineage was already accepted.
2. **Rank the levers** for the failure class (`rank_levers`): the learned success rate of
   each (class, lever) from every earlier parent→child pair in memory, per region with
   shrinkage toward all regions, plus a bonus for levers that also help the alpha's other
   failing classes. `flip_sign` is pinned first and `model_repair` last. On the "factor
   neutralization first" route, the `neutralize_*` levers go first and `flip_sign` is
   skipped.
3. Levers already used in this lineage are skipped. The first lever that applies makes
   the child:
   - **mechanical levers**, no model: `increase_decay`, `neutralize_statistical`,
     `size_tilt` (× (0.25 + rank(cap))), `speed_up`, `backfill_fields`, `rank_transform`,
     ...;
   - **model levers**, one call: `repair_system` (the BRAIN context, apply exactly one
     lever, keep the idea, JSON shape) plus `repair_user` (expression, metrics, why it
     failed, the lever and its step, `<past_repairs>` that worked, the Lesson Writer's
     `<lessons>`, `<pool>`, `<partner_fields>` for `blend_other_dataset`). The reply is
     validated like any model expression.
4. The child, id `<parent>_r<generation>`, joins **this arm's repair queue**. It runs in
   a later round, ahead of new ideas (3.2).

**Example.** The lever chosen was `neutralize_statistical`. The expression stayed the
same and the neutralization changed from SECTOR to STATISTICAL. The child is
`rs20261114174347r01a27_001_c1_r2` (generation 2). It ran in round 2:

| | parent `_c1` | child `_c1_r2` | limit |
|---|---|---|---|
| Sharpe | 0.91 | **2.71** | 1.58 ✓ |
| Fitness | 0.41 | **1.30** | 1.0 ✓ |
| Sub-universe | 0.37 | 1.23 | 1.41 ✗ |
| 2-year Sharpe | -0.47 | 1.45 | 1.58 ✗ |
| Every year positive | no | yes (1.42..4.52) | |

Decision: **repair / recent_sharpe** (recent_sharpe comes before sub_universe in the
priority list). The lineage then hit the depth limit, since a further child would be
generation 3 > `max_depth` 2. The owner rescued it by hand with `cli repair` and
`cli run` (files `candidates/rescue_20261115_*.json`). After 38 variants,
`hand_20261115_eur_37_blend_half_analyst` was accepted: the reversal blended 1:1 with
analyst69 revision breadth and tilted by size, with PROD_CORRELATION 0.645 (BRAIN
`omaA0nmE`). What those rescues taught was built back into the agents (the
`blend_other_dataset` and `size_tilt` levers, the price/volume advisory). Since then the
Lesson Writer does this automatically after every run.

### 3.6 Refill (`_round`)
Slots lost to a model going down always go back to the Director in the same round.
Slots the Generator, Pre-Filter or Critic left empty go back `refill_passes` (1) time.
Console: `--- round 1: refill 2: ...`, batch ids ending `f1`.

Then the round's queue is flushed (3.5), even if a budget ran out mid-round.

### 3.7 Strategy Evolution, every 2nd round (`_evolve`, `strategy.py`), model call
1. The strategy model (`groq:gpt-oss-120b`) gets a summary of all of memory: accept rates
   per dataset/region/model, failure classes, what repairs did, and BRAIN's limits per
   market.
2. It replies with guidance (overall and per region) and up to 3 new hypotheses. At
   least half must be on the least-tried datasets.
3. Each live arm in this run's regions is **replaced** by an evolved arm with prompt
   version `<base>+evoN`. The new arm keeps the parent's Beta record, gets the new
   guidance, and puts the new hypotheses at the front of its sources. The parent gets no
   new slots; its queued repairs still run.
4. A hypothesis on a catalog dataset no arm covers gets **new arms**: at most
   `max_new_arms` (8), spread over markets and models.

Console: `--- strategy evolution: gen-v8+evo1 (model), 36 arms updated, 0 new-dataset arms`

The run stops when: the rounds are done, the simulation budget is used up, the model-call
budget is used up, or every generator model is down.

## 4. Lesson Writer (`lessons.py:learn`), after the last round, model call

1. **Evidence from all of memory:**
   - for each (lever, parent failure class), in all regions and per region: how often it
     fixed the class, and the median change in Sharpe, fitness, sub-universe ratio, 2-year
     Sharpe, PROD_CORRELATION and turnover;
   - PROD_CORRELATION by the data categories an alpha reads;
   - failure profile per region and data category;
   - the latest 40 hand variants, with the measured changes.
2. **Rule lessons** are always written from that evidence. If the call budget allows, the
   strategy model condenses them (`lessons_system`: measured numbers win over hypotheses).
   Each lesson is checked: it names an audience (generator/repair), a scope that is in the
   evidence, a number, and is at most 400 characters. If any step fails, the rule lessons
   stand.
3. **Store** → `lessons` table (append-only). The next run reads the latest write: the
   Generator gets them in its system prompt (step a.5), the Repair Agent in `<lessons>`.

## 5. Report

`data/<run id>.json` holds:
- stop reason, simulations used, model calls (total and per model), models down;
- regions this run and simulations per region;
- every arm's accepts, non-accepts, posterior pass rate and queued repairs;
- recovered results, resumed repairs, seeds, Power Pool candidates;
- lever stats (all regions and per region), and a model comparison (generator accept
  rates, critic accuracy);
- `rounds`: for every round, each arm's slots, generated/rejected counts, Pre-Filter and
  Critic drops, revisions, each alpha's decision and its repair notes, refills and the
  evolution;
- the lessons written.

The console ends with `stopped: <reason> | simulations N | model calls N | pool N | report <path>` and one line per arm that has results.

A run stopped with Ctrl-C writes **no report and no lessons**. Decisions already printed
are in memory. Run `cli lessons --write` afterwards.

---

## What is stored, and when

All in `data/memory.sqlite`. Every table is append-only except `os_is`.

| Table | Written at | Holds |
|---|---|---|
| `candidates` | e.2, before submit | expression, settings, dates, hypothesis, dataset, family, parent, generation, lever, prompt version, model |
| `pending_submissions` | e.3, after submit | candidate id → BRAIN simulation id (used for recovery) |
| `results` | e.9, when BRAIN finishes (or invalid/timeout/failed) | status, every metric, yearly rows, **daily PnL**, BRAIN checks, check values and limits, alt stats, notes (incl. `brain_alpha_id`) |
| `variants` | f.1 | family label + candidate id, one row per simulation |
| `critic_reviews` | c, after review | critic model, role (primary/shadow), verdict, rules/model, generator model, issues, revised expression |
| `decisions` | f.8 | decision, failure class, every gate (value, threshold, note), risk reasons, reasons |
| `pool` | f.9, accept only | metrics; PnL in `data/pnl/<id>.csv` |
| `lessons` | 4, end of run | audience, scope, text, source (rules/model), evidence |

Not stored:
- the original of a critic revision (only `_c1` is kept, linked through `parent_id`);
- the judge's rulings;
- the Pre-Filter's drops (they are in the run report).

The repair queue lives in memory only, but nothing is lost: at the next start,
`_resume_repairs` rebuilds children for repair decisions that have none.

## What each model call costs (per arm batch)

| Agent | Calls | When |
|---|---|---|
| Generator | 1 (+1 if the validator rejected something, +1 per unparsable reply) | each arm with slots left after repairs |
| Paper extraction | 1 | only for a paper source |
| Pre-Filter | 0 (1 if `use_model`) | each batch |
| Critic | 1 | each batch with fresh alphas |
| Debate judge | 1 | only if a candidate was contested |
| Repair Agent | 0 for mechanical levers, 1 for model levers | each repair decision |
| Strategy Evolution | 1 | every 2nd round |
| Lesson Writer | 1 | end of run |

All of these count against `--max-llm-calls`. A model that fails after its retries
(429 waits ≤ 60 s, 5xx retried after 10/30/60 s) is down for the rest of the run. The
judge, Pre-Filter, Repair and Strategy then fall back to the next model; a generator
model's arms get no more slots.
