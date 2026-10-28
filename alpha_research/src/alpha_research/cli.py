"""Command line entry point (spec section 14.2).

    python -m alpha_research.cli run --candidates tests/fixtures/candidates/hand_written.json \
        --config tests/fixtures/thresholds_test.yaml
    python -m alpha_research.cli bandit --candidates tests/fixtures/candidates/hand_written.json

Exit codes: 0 success, 2 config error, 3 schema error, 4 client not implemented.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .brain_client import FixtureBrainClient, RealBrainClient
from .config import ConfigError, load_config
from .memory import MemoryStore, default_data_dir
from .orchestrator import run_rounds
from .pipeline import Pipeline, load_candidate_file
from .repair import Playbook
from .schemas import REGIONS, SchemaError
from .simulator import VirtualClock

EXIT_OK, EXIT_CONFIG, EXIT_SCHEMA, EXIT_CLIENT = 0, 2, 3, 4
PLACEHOLDER_PROMPT_VERSION = "pv0"  # Phase 2: prompt_version is a placeholder


def _run(args) -> int:
    config = load_config(args.config)
    playbook = Playbook.load(args.playbook)
    entries = load_candidate_file(args.candidates)

    if args.client == "real":
        raise NotImplementedError(RealBrainClient._MSG)
    client = FixtureBrainClient(args.fixtures_dir, {
        e.candidate.candidate_id: e.meta["fixture_id"]
        for e in entries if e.meta.get("fixture_id")
    })
    vclock = VirtualClock()  # fixture runs never wait in real time
    memory = MemoryStore(args.data_dir)
    try:
        pipeline = Pipeline(config, playbook, client, memory, sleep=vclock.sleep,
                            clock=vclock.clock, out=sys.stdout)
        pipeline.run([e.candidate for e in entries])
    finally:
        memory.close()
    return EXIT_OK


def _bandit(args) -> int:
    entries = load_candidate_file(args.candidates)
    arm_outcomes: dict[tuple, list[bool]] = {}
    for e in entries:
        c, expected = e.candidate, e.meta.get("expected_decision")
        if expected in (None, "duplicate") or c.region not in REGIONS:
            continue  # duplicates never simulate; an invalid region is not a real arm
        arm = (c.dataset or "unknown", c.region, c.prompt_version or PLACEHOLDER_PROMPT_VERSION)
        arm_outcomes.setdefault(arm, []).append(expected == "accept")
    history = run_rounds(arm_outcomes, rounds=args.rounds, budget=args.budget,
                         floor_share=args.floor_share, seed=args.seed)
    arms = list(arm_outcomes)
    print("arms:")
    for i, arm in enumerate(arms):
        print(f"  A{i}: {arm}  fixture pass rate "
              f"{sum(arm_outcomes[arm])}/{len(arm_outcomes[arm])}")
    print("round | " + " | ".join(f"A{i}" for i in range(len(arms))) + " | total | floor | floor_held")
    ok = True
    for h in history:
        counts = " | ".join(f"{h['allocation'][a]:>2}" for a in arms)
        print(f"{h['round']:>5} | {counts} | {h['total']:>5} | {h['floor']:>5} | {h['floor_held']}")
        ok &= h["floor_held"] and h["total"] == args.budget
    return EXIT_OK if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alpha_research")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline on a candidate file")
    run.add_argument("--candidates", required=True)
    run.add_argument("--config", default="config/thresholds.yaml")
    run.add_argument("--playbook", default="config/repair_playbook.yaml")
    run.add_argument("--data-dir", default=str(default_data_dir()))
    run.add_argument("--client", choices=["fixture", "real"], default="fixture")
    run.add_argument("--fixtures-dir", default="tests/fixtures/brain_results")
    run.set_defaults(func=_run)

    bandit = sub.add_parser("bandit", help="Phase 2 driver: allocation rounds over fixture arms")
    bandit.add_argument("--candidates", required=True)
    bandit.add_argument("--rounds", type=int, default=10)
    bandit.add_argument("--budget", type=int, default=50)
    bandit.add_argument("--floor-share", type=float, default=0.2)
    bandit.add_argument("--seed", type=int, default=0)
    bandit.set_defaults(func=_bandit)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except SchemaError as exc:
        print(f"schema error: {exc}", file=sys.stderr)
        return EXIT_SCHEMA
    except NotImplementedError as exc:
        print(f"client error: {exc}", file=sys.stderr)
        return EXIT_CLIENT


if __name__ == "__main__":
    sys.exit(main())
