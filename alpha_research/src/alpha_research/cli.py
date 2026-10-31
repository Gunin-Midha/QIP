"""Command line entry point (spec section 14.2).

    python -m alpha_research.cli run --candidates tests/fixtures/candidates/hand_written.json \
        --config tests/fixtures/thresholds_test.yaml
    python -m alpha_research.cli bandit --candidates tests/fixtures/candidates/hand_written.json
    python -m alpha_research.cli generate --hypothesis "..." --settings my_settings.yaml \
        --out generated/batch1.json

Exit codes: 0 success, 2 config error, 3 schema error, 4 client not implemented,
5 model unavailable or unusable reply, 6 generate produced no valid candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from .brain_client import FixtureBrainClient, FixtureNotFoundError, RealBrainClient
from .config import ConfigError, load_config
from .fastexpr import FastExprSpec
from .generator import (AlphaGenerator, GenerationError, GenerationSettings, Source,
                        load_datafields, load_generator_config)
from .llm import LLMUnavailable, ScriptedLLMClient, make_real_client
from .memory import MemoryStore, default_data_dir
from .orchestrator import run_rounds
from .pipeline import Pipeline, load_candidate_file
from .repair import Playbook
from .schemas import REGIONS, SchemaError
from .simulator import VirtualClock

EXIT_OK, EXIT_CONFIG, EXIT_SCHEMA, EXIT_CLIENT, EXIT_LLM, EXIT_EMPTY = 0, 2, 3, 4, 5, 6
PLACEHOLDER_PROMPT_VERSION = "pv0"  # Phase 2: prompt_version is a placeholder
DEFAULT_MODEL = "claude-sonnet-5-5"
SETTING_FLAGS = ("region", "universe", "delay", "decay", "neutralization", "truncation",
                 "start_date", "end_date", "dataset")


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


def _read_text(path: str, what: str) -> str:
    try:
        return Path(path).read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read {what} {path}: {exc}") from exc


def _generation_settings(args) -> GenerationSettings:
    merged: dict = {}
    if args.settings:
        try:
            loaded = yaml.safe_load(_read_text(args.settings, "settings file"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"settings file {args.settings} is not valid YAML: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"settings file {args.settings} must be a mapping")
        merged.update(loaded)
    merged.update({k: getattr(args, k) for k in SETTING_FLAGS if getattr(args, k) is not None})
    return GenerationSettings.from_mapping(merged)


def _generation_source(args) -> Source:
    try:
        if args.hypothesis is not None:
            return Source("hypothesis", args.hypothesis)
        if args.hypothesis_file:
            return Source("hypothesis", _read_text(args.hypothesis_file, "hypothesis file"))
        if Path(args.paper).suffix.lower() == ".pdf":
            raise ConfigError("PDF papers are not read directly; convert to text first "
                              "(for example: pdftotext paper.pdf paper.txt)")
        return Source("paper", _read_text(args.paper, "paper"), title=args.paper_title)
    except ValueError as exc:  # empty text
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(str(exc)) from exc


def _llm_client(args):
    if args.llm == "scripted":
        if not args.llm_script:
            raise ConfigError("--llm scripted needs --llm-script (a JSON list of reply strings)")
        try:
            replies = json.loads(_read_text(args.llm_script, "script"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"script {args.llm_script} is not valid JSON: {exc}") from exc
        if not isinstance(replies, list) or not all(isinstance(r, str) for r in replies):
            raise ConfigError("script must be a JSON list of strings")
        return ScriptedLLMClient(replies)
    return make_real_client(args.model)


def _generate(args) -> int:
    spec = FastExprSpec.load(args.fastexpr)
    config = load_generator_config(args.generator_config)
    settings = _generation_settings(args)
    source = _generation_source(args)
    datafields = None
    if args.datafields:
        try:
            datafields = load_datafields(args.datafields)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigError(f"cannot read datafield catalog {args.datafields}: {exc}") from exc
    llm = _llm_client(args)

    result = AlphaGenerator(llm, spec, settings, config, datafields=datafields).generate(
        source, args.n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([c.to_dict() for c in result.candidates], indent=2) + "\n")
    report = out.with_suffix(".report.json")
    report.write_text(json.dumps(result.to_report(), indent=2) + "\n")

    print(f"batch {result.batch_id} | model {result.model} | {len(result.candidates)} candidates, "
          f"{len(result.rejected)} rejected, {len(result.duplicates)} duplicates | "
          f"{result.llm_calls} model calls")
    for c in result.candidates:
        print(f"{c.candidate_id} | {c.expression}")
        for w in result.warnings.get(c.candidate_id, []):
            print(f"    warning: {w}")
    for r in result.rejected:
        print(f"rejected | {r.expression} | {r.errors[0]}")
    print(f"wrote {out} and {report}")
    return EXIT_OK if result.candidates else EXIT_EMPTY


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

    gen = sub.add_parser("generate", help="Alpha Generator: hypothesis or paper -> candidates")
    src = gen.add_mutually_exclusive_group(required=True)
    src.add_argument("--hypothesis", help="the hypothesis as text")
    src.add_argument("--hypothesis-file", help="text file holding the hypothesis")
    src.add_argument("--paper", help="paper as a text or markdown file (not PDF)")
    gen.add_argument("--paper-title", default=None)
    gen.add_argument("--n", type=int, default=10, help="alphas to produce (default 10)")
    gen.add_argument("--out", required=True, help="candidate JSON to write; a .report.json "
                     "with rejects, rationales and warnings is written next to it")
    gen.add_argument("--settings", help="YAML with the simulation settings; flags override it")
    gen.add_argument("--region")
    gen.add_argument("--universe")
    gen.add_argument("--delay", type=int)
    gen.add_argument("--decay", type=int)
    gen.add_argument("--neutralization")
    gen.add_argument("--truncation", type=float)
    gen.add_argument("--start-date", dest="start_date")
    gen.add_argument("--end-date", dest="end_date")
    gen.add_argument("--dataset", help="dataset label stored on each candidate")
    gen.add_argument("--datafields", help="datafield catalog: JSON list or one id per line")
    gen.add_argument("--fastexpr", default="config/fastexpr.yaml")
    gen.add_argument("--generator-config", default="config/generator.yaml")
    gen.add_argument("--llm", choices=["real", "scripted"], default="real")
    gen.add_argument("--model", default=os.environ.get("ALPHA_RESEARCH_MODEL", DEFAULT_MODEL))
    gen.add_argument("--llm-script", help="JSON list of replies for --llm scripted (offline)")
    gen.set_defaults(func=_generate)
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
    except FixtureNotFoundError as exc:
        print(f"client error: {exc}. Fixture runs can only simulate candidates that have a "
              "recorded BRAIN result; generated candidates need the real client (OQ-2).",
              file=sys.stderr)
        return EXIT_CLIENT
    except (LLMUnavailable, GenerationError) as exc:
        print(f"model error: {exc}", file=sys.stderr)
        return EXIT_LLM


if __name__ == "__main__":
    sys.exit(main())
