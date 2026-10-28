# alpha_research: deterministic core (Phases 1 and 2)

This is the rule-based backbone of the multi-agent alpha research system. It contains no
LLM calls and makes no network calls. BRAIN is replaced by a fixture-backed client;
`RealBrainClient` is a stub that raises `NotImplementedError` (OQ-2).

| Module | Role in the architecture |
|---|---|
| `schemas.py` | CandidateRecord / ResultRecord / DecisionRecord contracts |
| `config.py` | loads and validates `config/thresholds.yaml` |
| `brain_client.py`, `simulator.py` | Backtester: validate, submit/resume, poll, retry, normalize, persist |
| `metric_validator.py` | fitness / yearly / margin tripwires on every result |
| `data_scientist.py` | Datafield Analysis Agent (coverage, update gaps, tails, NaN pattern, redundancy) |
| `evaluator.py` | Quantitative Evaluation: hard gates, failure class, decision |
| `risk.py` | Risk Assessment Agent: drawdown, leg balance, top-PnL removal, concentration, weak year |
| `diversity.py`, `pool.py` | Diversity Agent and the Alpha Pool |
| `auditor.py` | Overfitting Auditor: variant counts, deflated Sharpe, OS/IS tracking |
| `memory.py` | Memory Agent: append-only SQLite store |
| `repair.py` | Repair Agent playbook (lever lookup only; levers are not applied yet) |
| `orchestrator.py` | Research Director: Thompson sampling with floor quotas |
| `research_loop.py` | Research Loop Agent: experiment ids, duplicates, lineage, queue, routing |
| `pipeline.py`, `cli.py` | deterministic end-to-end wiring |

## Setup

Python 3.11 or newer. Dependencies: pandas, numpy, scipy, pyyaml, pytest.

```bash
cd alpha_research
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

Without installing, prefix commands with `PYTHONPATH=src`.

## Run the tests

```bash
pytest -q
```

## Run the pipeline on the hand-written candidates

`config/thresholds.yaml` leaves the holdout null (OQ-9), so the simulator refuses to run
with it, as the spec requires. That command exits with code 2. Until the owner sets the
holdout, use the test config. It sets placeholder holdout dates and two placeholder risk
limits so the concentration and drawdown fixtures can fail:

```bash
python -m alpha_research.cli run \
    --candidates tests/fixtures/candidates/hand_written.json \
    --config tests/fixtures/thresholds_test.yaml
```

Options: `--data-dir` (default `$ALPHA_RESEARCH_DATA` or `./data`), `--client fixture|real`
(`real` exits with code 4), `--fixtures-dir`, `--playbook`. Exit codes: 0 ok, 2 config error,
3 schema error, 4 client not implemented. Each candidate prints one line:
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

## Regenerating fixtures

```bash
python tests/fixtures/generate_fixtures.py
```

The script is deterministic (fixed seed). It writes `brain_results/*.json`, `pnl/*.csv` and
`candidates/hand_written.json`. `screenshot_aggregate.json` holds the IS summary
screenshot values (aggregate plus yearly rows for 2019 to 2023).

## Decision rules worth knowing

- A hard gate that definitely failed beats one that cannot be evaluated. For example,
  Sharpe 1.10 with self-correlation not yet computed gives `reject / sharpe`, not `pending`.
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
- **OQ-2 Real BRAIN response format.** Needed before `RealBrainClient` can be written. The
  raw key names and the unit of `weight_concentration` follow our fixture format.
- **OQ-3 Concurrent simulation limit.** Account-specific. `max_concurrent` is 1 and no
  threading is implemented.
- **OQ-4 Drawdown and concentration limits.** Policy choice.
- **OQ-5 Leg balance and top-PnL-removal thresholds.** Policy choice.
- **OQ-6 Weak-year floor and yearly Sharpe spread.** Policy choice.
- **OQ-7 Minimum marginal pool Sharpe gain.** Policy choice.
- **OQ-8 OS/IS ratio floor.** Policy choice.
- **OQ-9 Holdout period.** The simulator refuses to run until it is set.
