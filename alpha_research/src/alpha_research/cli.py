"""Command line entry point (spec section 14.2).

    python -m alpha_research.cli run --candidates tests/fixtures/candidates/hand_written.json \
        --config tests/fixtures/thresholds_test.yaml
    python -m alpha_research.cli bandit --candidates tests/fixtures/candidates/hand_written.json
    python -m alpha_research.cli generate --hypothesis "..." --settings my_settings.yaml \
        --out generated/batch1.json
    python -m alpha_research.cli brain auth | options | capture | operators | datafields | datasets
    python -m alpha_research.cli research --plan config/research_plan.example.yaml \
        --config <thresholds with holdout> --max-simulations 20 --max-llm-calls 60
    python -m alpha_research.cli compare --data-dir data
    python -m alpha_research.cli repair --data-dir data --candidate-id ID --out rescue.json
    python -m alpha_research.cli levers --data-dir data
    python -m alpha_research.cli models --host nvidia --filter llama | --check

Exit codes: 0 success, 2 config error, 3 schema error, 4 no fixture for a candidate,
5 model unavailable or unusable reply, 6 generate produced no valid candidate,
7 BRAIN login or API error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .brain_api import (EMAIL_ENV_VAR, SECRET_ENV_VAR, BrainApiError, RealBrainClient,
                        UrllibTransport, list_datafields, list_datasets, list_operators,
                        load_brain_config, make_session, prompt_credentials, quota_text,
                        simulation_options, submission_diversity)
from .brain_client import (FixtureBrainClient, FixtureNotFoundError, SimulationQuotaExhausted,
                           SubmissionRejected)
from .compare import model_comparison
from .config import ConfigError, load_config
from .fastexpr import FastExprSpec, catalog
from .generator import (AlphaGenerator, GenerationError, GenerationSettings, Source,
                        load_datafield_meta, load_datafields, load_generator_config)
from .llm import HOSTS, LLMUnavailable, ScriptedLLMClient, list_models, make_real_client
from .memory import MemoryStore, default_data_dir
from .orchestrator import run_rounds
from .pipeline import Pipeline, awaiting_brain, load_candidate_file
from .repair import Playbook
from .lever_stats import LeverStats, repair_events
from .repair_agent import RepairAgent
from . import power_pool
from .pool import Pool
from .quality import generation_quality, quality_lines
from .lessons import LessonsOutcome, learn as learn_lessons, summary as lessons_summary
from .research import (Budget, ResearchRun, load_agents_config, load_plan, load_seeds,
                       write_report)
from . import prompts as agent_prompts
from .schemas import REGIONS, SchemaError
from .simulator import VirtualClock, normalize

EXIT_OK, EXIT_CONFIG, EXIT_SCHEMA, EXIT_CLIENT, EXIT_LLM, EXIT_EMPTY, EXIT_BRAIN = 0, 2, 3, 4, 5, 6, 7
PLACEHOLDER_PROMPT_VERSION = "pv0"  # Phase 2: prompt_version is a placeholder
DEFAULT_MODEL = "claude-sonnet-5-5"
SETTING_FLAGS = ("region", "universe", "delay", "decay", "neutralization", "truncation",
                 "start_date", "end_date", "dataset")


def _run(args) -> int:
    config = load_config(args.config)
    if args.max_concurrent is not None:
        if args.max_concurrent < 1:
            print("error: --max-concurrent must be at least 1", file=sys.stderr)
            return EXIT_CONFIG
        config = dataclasses.replace(config, simulator=dataclasses.replace(
            config.simulator, max_concurrent=args.max_concurrent))
    playbook = Playbook.load(args.playbook)
    entries = load_candidate_file(args.candidates)

    if args.client == "real":
        config.require_holdout()  # fail before logging in, not after the first submit
        brain_config = load_brain_config(args.brain_config)
        _keep_awake(args)
        client = RealBrainClient(_brain_session(brain_config), brain_config,
                                 capture_dir=args.capture_dir)
        sleep, clock = time.sleep, time.monotonic
    else:
        client = FixtureBrainClient(args.fixtures_dir, {
            e.candidate.candidate_id: e.meta["fixture_id"]
            for e in entries if e.meta.get("fixture_id")
        })
        vclock = VirtualClock()  # fixture runs never wait in real time
        sleep, clock = vclock.sleep, vclock.clock
    memory = MemoryStore(args.data_dir)
    try:
        pipeline = Pipeline(config, playbook, client, memory, sleep=sleep,
                            clock=clock, out=sys.stdout)
        pipeline.run([e.candidate for e in entries])
        ids = {e.candidate.candidate_id for e in entries}
        waiting = [r["candidate"].candidate_id for r in memory.candidate_outcomes()
                   if r["candidate"].candidate_id in ids
                   and awaiting_brain(r["decision"], r["result"])]
        if waiting:
            print(f"{len(waiting)} alpha(s) pending until BRAIN finishes their checks "
                  "(PROD_CORRELATION can take hours): run `cli recheck` later")
        if not args.no_lessons:
            _write_lessons(memory, args, "batch" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    finally:
        memory.close()
    return EXIT_OK


def _write_lessons(memory: MemoryStore, args, batch_id: str, llm=None) -> LessonsOutcome | None:
    """Lesson Writer after a batch (lessons.py): rules, or the model when ``llm`` is given;
    datafield catalogs from the agents config's lessons.catalog_plan."""
    agents = load_agents_config(args.agents)
    cfg = agents.lessons
    if not cfg.after_every_batch and llm is None:
        return None
    markets: dict = {}
    if cfg.catalog_plan:
        try:
            markets = _plan_catalogs(load_plan(cfg.catalog_plan))
        except ConfigError as exc:
            print(f"lessons: no datafield catalogs ({exc}); categories unknown", file=sys.stderr)
    out = learn_lessons(memory.candidate_outcomes(), FastExprSpec.load(args.fastexpr),
                        lambda region, delay: markets.get((region, delay), (None, {}))[1], cfg,
                        llm=llm, max_output=agents.max_output)
    if out.lessons:
        memory.put_lessons(batch_id, out.lessons, out.source, out.evidence)
    print(lessons_summary(out))
    return out


def _quality(args) -> int:
    """``quality.generation_quality`` for research runs in memory."""
    spec = FastExprSpec.load(args.fastexpr)
    gen = load_generator_config(args.generator_config)
    markets = _plan_catalogs(load_plan(args.plan))
    memory = MemoryStore(args.data_dir)
    try:
        outcomes = memory.candidate_outcomes()
    finally:
        memory.close()
    runs = args.run
    if not runs:
        ids = [r["candidate"].candidate_id for r in outcomes
               if re.match(r"rs\d{14}", r["candidate"].candidate_id)]
        runs = [max(ids, key=lambda i: i[:16])[:16]] if ids else []
    if not runs:
        print("no research run in memory")
        return EXIT_OK
    for rid in runs:
        q = generation_quality(outcomes, rid, spec, gen,
                               lambda region, delay: markets.get((region, delay), (None, {})))
        print("\n".join(quality_lines(q)))
    return EXIT_OK


def _pool(args) -> int:
    """The pool's active alphas; ``--sync`` first makes it match memory's latest decisions
    (retire alphas no longer accepted; under gates.brain_checks_only admit accepted ones
    that are missing)."""
    config = load_config(args.config)
    memory = MemoryStore(args.data_dir)
    try:
        pool = Pool(memory)
        for ref in args.submitted or ():
            cid = _candidate_by_ref(memory, ref)
            if cid is None:
                print(f"not in memory: {ref}", file=sys.stderr)
                return EXIT_CONFIG
            try:
                print(f"{pool.add_submitted(cid)}: {cid}")
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return EXIT_CONFIG
        for ref in args.remove or ():
            cid = _candidate_by_ref(memory, ref) or ref
            ok = pool.retire(cid, args.reason or "removed by the owner")
            print(f"removed: {cid}" if ok else f"not in the pool: {ref}")
        if args.sync and config.diversity.pool_admits == "submitted":
            print("diversity.pool_admits is 'submitted': the pool holds what you added with "
                  "--submitted, nothing to sync")
        elif args.sync:
            changed = pool.sync(admit_accepted=config.gates.brain_checks_only)
            for cid in changed["admitted"]:
                print(f"admitted: {cid}")
            for cid in changed["retired"]:
                print(f"retired:  {cid} (latest decision is no longer accept)")
            if not changed["admitted"] and not changed["retired"]:
                print("pool already matches the latest decisions")
        ids = pool.alpha_ids()
        print(f"{len(ids)} alpha(s) in the pool:")
        for cid in ids:
            c, r = memory.get_candidate(cid), (memory.results_for(cid) or [None])[-1]
            brain = next((n.split(":", 1)[1].strip() for n in (r.notes or [])
                          if n.startswith("brain_alpha_id:")), "-") if r else "-"
            stats = (f"Sharpe {r.sharpe}, fitness {r.fitness}" if r else "no result")
            print(f"  {cid}  {c.region if c else '?'}  BRAIN {brain}  {stats}")
    finally:
        memory.close()
    return EXIT_OK


def _candidate_by_ref(memory: MemoryStore, ref: str) -> str | None:
    """A candidate id, or the candidate whose result names BRAIN alpha ``ref`` (any case)."""
    if memory.get_candidate(ref) is not None:
        return ref
    want = ref.lower()
    for row in memory.candidate_outcomes():
        r = row["result"]
        for n in (r.notes or []) if r else ():
            if n.startswith("brain_alpha_id:") and n.split(":", 1)[1].strip().lower() == want:
                return row["candidate"].candidate_id
    return None


def _power_pool(args) -> int:
    """Power Pool candidates in memory (power_pool.py), recomputed from every stored
    result, with a description to submit them by hand. Recorded only."""
    config = load_config(args.config)
    submission = config.gates.submission_checks
    memory = MemoryStore(args.data_dir)
    try:
        rows = [r for r in memory.candidate_outcomes()
                if power_pool.candidate_reason(r["result"], r["decision"], submission)]
    finally:
        memory.close()
    if args.region:
        rows = [r for r in rows if r["candidate"].region == args.region]
    if not rows:
        print("no Power Pool candidates in memory")
        return EXIT_OK
    for r in rows:
        c = r["candidate"]
        print(power_pool.describe(c, r["result"], submission))
        text = " ".join((c.hypothesis or "").split())
        ok = "ok" if len(text) >= power_pool.DESCRIPTION_MIN else \
            f"too short for BRAIN's {power_pool.DESCRIPTION_MIN}"
        print(f"  description ({len(text)} characters, {ok}): {text or '(none)'}")
        print(f"  expression: {c.expression}")
    print(f"{len(rows)} candidate(s). BRAIN's Power Pool rules beyond these checks are not "
          "known to the pipeline: check them in BRAIN before submitting.")
    return EXIT_OK


def _lessons(args) -> int:
    """Show the latest lessons; with --write, write new ones from memory first."""
    memory = MemoryStore(args.data_dir)
    try:
        if args.write:
            llm = _llm_client(args) if args.model or args.llm == "scripted" else None
            _write_lessons(memory, args, "manual" + datetime.now(timezone.utc).strftime(
                "%Y%m%d%H%M%S"), llm=llm)
        rows = memory.latest_lessons()
    finally:
        memory.close()
    if not rows:
        print("no lessons in memory yet")
        return EXIT_OK
    print(f"lessons of {rows[0]['batch_id']} ({rows[0]['source']}, {rows[0]['created_at'][:19]}):")
    for r in rows:
        print(f"- [{r['audience']} | {r['scope']}] {r['text']}")
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
    return make_real_client(args.model, _output_caps(args).get(args.model))


DEFAULT_AGENTS_CONFIG = "config/agents.yaml"


def _output_caps(args) -> dict:
    """Per-model output caps (``models.max_output``) from the agents config, if any."""
    path = getattr(args, "agents", None)
    if path is None and Path(DEFAULT_AGENTS_CONFIG).exists():
        path = DEFAULT_AGENTS_CONFIG   # `generate`: use the repo config when run from it
    return load_agents_config(path).models.max_output if path else {}


def _llm_factory(args, models: list[str], caps: dict | None = None,
                 pace: dict | None = None):
    """model id -> client. Scripted: one client answers for every model id. Real: one client
    per model id, all built now so a missing key or package fails before BRAIN is called;
    ``caps`` (model id -> output cap) replaces the agents' cap for those models and ``pace``
    (model id -> per-minute limits) paces their calls."""
    if args.llm == "scripted":
        scripted = _llm_client(args)
        return lambda model: scripted
    caps, pace = caps or {}, pace or {}
    make = lambda m: make_real_client(m, caps.get(m), pace.get(m))
    clients = {m: make(m) for m in models}
    return lambda model: clients[model] if model in clients else make(model)


def _generate(args) -> int:
    spec = FastExprSpec.load(args.fastexpr)
    config = load_generator_config(args.generator_config)
    settings = _generation_settings(args)
    source = _generation_source(args)
    datafields, field_meta = None, {}
    if args.datafields:
        try:
            datafields = load_datafields(args.datafields)
            field_meta = load_datafield_meta(args.datafields)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigError(f"cannot read datafield catalog {args.datafields}: {exc}") from exc
    llm = _llm_client(args)

    result = AlphaGenerator(llm, spec, settings, config, datafields=datafields,
                            field_meta=field_meta).generate(source, args.n)

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


def _biometric_prompt(url: str) -> None:
    print(f"BRAIN asks for biometric verification. Open this link in a browser, verify,\n"
          f"then come back here and press Enter:\n  {url}", file=sys.stderr)
    input()


def _brain_session(config):
    """A BRAIN session that, in a terminal, asks for the login when a new one is needed
    (saved session expired, no BRAIN login in the environment) and for biometric verification."""
    tty = sys.stdin.isatty()
    return make_session(config, on_biometric=_biometric_prompt if tty else None,
                        ask_credentials=prompt_credentials if tty else None)


def _brain_auth(args) -> int:
    """A fresh login (with BRAIN's biometric check), saved for the next commands."""
    config = load_brain_config(args.brain_config)
    _brain_session(config).authenticate()
    if config.session_file is not None:
        print(f"BRAIN login ok; session saved to {config.session_file}. The next commands "
              "reuse it until BRAIN ends it (`brain logout` deletes it).")
    else:
        print("BRAIN login ok (session_file is off in the brain config, so nothing is saved)")
    return EXIT_OK


def _brain_cancel(args) -> int:
    """Cancel our simulations stuck on BRAIN: submissions memory has no final result for
    (our wait timed out, or the run was stopped) that are still unfinished after
    --older-than-hours. Each is asked once first, so a finished one is decided instead.
    A cancelled one is stored as failed (reject / incomplete) and never asked about again."""
    config = load_config(args.config)
    hours = (args.older_than_hours if args.older_than_hours is not None
             else config.simulator.cancel_stuck_after_hours)
    if hours is None:
        print("error: give --older-than-hours (simulator.cancel_stuck_after_hours is off)",
              file=sys.stderr)
        return EXIT_CONFIG
    memory = MemoryStore(args.data_dir)
    try:
        brain_config = load_brain_config(args.brain_config)
        client = RealBrainClient(_brain_session(brain_config), brain_config)
        pipe = Pipeline(config, Playbook.load(args.playbook), client, memory, out=sys.stdout)
        rows = pipe.stuck_submissions(hours)
        if not rows:
            print(f"nothing to cancel: no submission unfinished for {hours:g} h or more")
            return EXIT_OK
        print(f"{len(rows)} submission(s) without a final result for {hours:g} h or more:")
        for r in rows:
            print(f"  {r['candidate_id']}  ({r['hours']:.0f} h, simulation {r['brain_sim_id']})")
        if not args.yes:
            if not sys.stdin.isatty():
                print("error: not a terminal; pass --yes to cancel", file=sys.stderr)
                return EXIT_CONFIG
            if input("ask BRAIN about each and cancel the unfinished ones? [y/N] ").strip().lower() != "y":
                print("nothing cancelled")
                return EXIT_OK
        done = cancelled = kept = 0
        for r in rows:
            c = memory.get_candidate(r["candidate_id"])
            if c is None:
                continue
            try:
                out = pipe.recover(c, r["brain_sim_id"])
            except (BrainApiError, ConnectionError, TimeoutError):
                out = None       # BRAIN answered with an error: cancel it like a stuck one
            if out is not None:
                done += 1
                continue
            if pipe.cancel_submission(c, r["brain_sim_id"],
                                      f"cli brain cancel: unfinished {r['hours']:.0f} h "
                                      "after submission") is None:
                print(f"  {r['candidate_id']}: BRAIN refused the cancel; kept for later")
                kept += 1
            else:
                cancelled += 1
    finally:
        memory.close()
    print(f"cancelled {cancelled}, finished on BRAIN and decided {done}"
          + (f", kept {kept}" if kept else ""))
    return EXIT_OK


def _brain_logout(args) -> int:
    config = load_brain_config(args.brain_config)
    if config.session_file is None:
        print("no saved BRAIN session (session_file is off in the brain config)")
        return EXIT_OK
    gone = UrllibTransport(session_file=config.session_file).forget_session()
    print(f"deleted the saved BRAIN session {config.session_file}" if gone
          else f"no saved BRAIN session at {config.session_file}")
    return EXIT_OK


def _brain_options(args) -> int:
    """Allowed simulation settings for the account (regions, universes, neutralizations...)."""
    options = simulation_options(_brain_session(load_brain_config(args.brain_config)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(options, indent=2) + "\n")
    for name, spec in sorted(options["common"].items()):
        choices = spec.get("choices") if isinstance(spec, dict) else None
        if isinstance(choices, list):
            values = [c.get("value") if isinstance(c, dict) else c for c in choices]
            print(f"{name}: {values}")
        elif choices is not None:
            print(f"{name}: depends on {', '.join(map(str, choices)) if isinstance(choices, dict) else '?'}"
                  " (see the file)")
        else:
            print(f"{name}: {json.dumps(spec)[:120]}")
    print(f"types: {sorted(options['by_type'])}; wrote {out}")
    return EXIT_OK


def _brain_capture(args) -> int:
    """Simulate a few candidates for real and save every raw response, so the field
    mapping in brain_api.translate can be checked against our account (OQ-2)."""
    brain_config = load_brain_config(args.brain_config)
    entries = load_candidate_file(args.candidates)[: args.limit]
    out_dir = Path(args.out_dir)
    client = RealBrainClient(_brain_session(brain_config), brain_config, capture_dir=out_dir)
    for e in entries:
        c = e.candidate
        try:
            c.validate()
            sim_id = client.submit(c)
        except (SchemaError, SubmissionRejected) as exc:
            print(f"{c.candidate_id} | skipped | {exc}")
            continue
        started = time.monotonic()
        raw = client.poll(sim_id)
        while raw["status"] == "running" and time.monotonic() - started < args.max_wait:
            time.sleep(raw["retry_after"])
            raw = client.poll(sim_id)
        if raw["status"] == "running":
            print(f"{c.candidate_id} | {sim_id} | still running after {args.max_wait:g}s")
            continue
        result = normalize(raw, c.candidate_id, sim_id)
        (out_dir / sim_id).mkdir(parents=True, exist_ok=True)
        (out_dir / sim_id / "candidate.json").write_text(json.dumps(c.to_dict(), indent=2) + "\n")
        (out_dir / sim_id / "result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        print(f"{c.candidate_id} | {sim_id} | {result.status} | sharpe {result.sharpe} | "
              f"fitness {result.fitness} | turnover {result.turnover} | saved {out_dir / sim_id}")
    return EXIT_OK


def _brain_operators(args) -> int:
    session = _brain_session(load_brain_config(args.brain_config))
    operators = list_operators(session)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(operators, indent=2) + "\n")
    spec = FastExprSpec.load(args.fastexpr)
    brain = {o["name"]: o.get("definition") for o in operators}
    seed = set(spec.operators)
    print(f"{len(brain)} operators on BRAIN, {len(seed)} in {args.fastexpr}; wrote {out}")
    for name in sorted(seed - set(brain)):
        print(f"in seed list, not on BRAIN: {name}")
    for name in sorted(set(brain) - seed):
        print(f"on BRAIN, not in seed list: {name}  {brain[name] or ''}")
    for name in sorted(seed & set(brain)):
        print(f"check arity: {name}  BRAIN: {brain[name]}")
    return EXIT_OK


def _brain_datafields(args) -> int:
    brain_config = load_brain_config(args.brain_config)
    session = _brain_session(brain_config)
    print(f"Downloading datafields for {args.region}/{args.universe}/delay {args.delay} "
          "(a large catalog can take 10-30 minutes)...", file=sys.stderr)
    fields = list_datafields(
        session, region=args.region, universe=args.universe, delay=args.delay,
        instrument_type=brain_config.settings.get("instrumentType", "EQUITY"), dataset=args.dataset,
        progress=lambda msg: print("  " + msg, file=sys.stderr, flush=True))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fields, indent=2) + "\n")
    print(f"{len(fields)} datafields for {args.region}/{args.universe}/delay {args.delay}"
          f"{' dataset ' + args.dataset if args.dataset else ''}; wrote {out} "
          "(use it with generate --datafields)")
    return EXIT_OK


def _brain_diversity(args) -> int:
    """Our submitted alphas by region and data category, from BRAIN."""
    d = submission_diversity(_brain_session(load_brain_config(args.brain_config)))
    print(f"{d['total']} alpha(s) submitted on BRAIN")
    by_region: dict = {}
    for row in d["categories"]:
        by_region.setdefault(row["region"], []).append(row)
    for region in sorted(set(d["regions"]) | set(by_region)):
        cats = ", ".join(f"{r['category']} {r['alphas']} (D{r['delay']}, DATA_DIVERSITY "
                         f"{r['check']})" for r in sorted(by_region.get(region, []),
                                                           key=lambda r: r["category"]))
        print(f"  {region}: {d['regions'].get(region, '?')} alpha(s); categories: {cats or '-'}")
    return EXIT_OK


def _brain_datasets(args) -> int:
    """Every dataset for these settings, with its field count, so a discovery run can see
    which catalogs are too large for one datafield query."""
    brain_config = load_brain_config(args.brain_config)
    datasets = list_datasets(_brain_session(brain_config), region=args.region,
                             universe=args.universe, delay=args.delay,
                             instrument_type=brain_config.settings.get("instrumentType", "EQUITY"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(datasets, indent=2) + "\n")

    def show(v, fmt="{}"):
        return "?" if v is None else fmt.format(v)

    for d in sorted(datasets, key=lambda d: d["id"]):
        print(f"{d['id']} | {show((d.get('category') or {}).get('id'))} | "
              f"{show(d.get('fieldCount'))} fields | coverage {show(d.get('coverage'), '{:.0%}')} | "
              f"{show(d.get('alphaCount'))} alphas | pyramid {show(d.get('pyramidMultiplier'))}")
    fields = [d.get("fieldCount") for d in datasets]
    total = sum(f for f in fields if isinstance(f, int))
    print(f"{len(datasets)} datasets for {args.region}/{args.universe}/delay {args.delay}, "
          f"{total:,} fields{' (some counts missing)' if None in fields else ''}; wrote {out}")
    return EXIT_OK


def _plan_catalogs(plans) -> dict[tuple, tuple[list[str], dict]]:
    """{(region, delay): (datafields, meta)} from the catalogs a v2 plan names, each file
    read once. Markets without one fall back to --datafields."""
    markets, read = {}, {}
    for p in plans:
        if p.catalog is None:
            continue
        if p.catalog not in read:
            try:
                read[p.catalog] = (load_datafields(p.catalog), load_datafield_meta(p.catalog))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ConfigError(f"cannot read datafield catalog {p.catalog} for "
                                  f"{p.settings.region} delay {p.settings.delay}: {exc} "
                                  "(scripts/brain_discovery.py downloads the catalogs)") from exc
        markets[(p.settings.region, p.settings.delay)] = read[p.catalog]
    if markets:
        print(f"field catalogs: {len(markets)} region/delays, "
              f"{sum(len(f) for f, _ in read.values()):,} fields")
    return markets


def _keep_awake(args) -> None:
    """On macOS, keep the computer awake while this command runs: ``caffeinate -ims -w
    <our pid>`` stops idle, display-idle and (on mains power) system sleep and ends by itself
    when this process ends (owner 2026-11-25: runs take hours). Closing the lid still
    sleeps a laptop. ``--no-caffeinate`` turns it off; other systems are left alone."""
    if getattr(args, "no_caffeinate", False) or sys.platform != "darwin":
        return
    exe = shutil.which("caffeinate")
    if exe is None:
        return
    subprocess.Popen([exe, "-ims", "-w", str(os.getpid())],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("keeping the computer awake while this runs (caffeinate; closing the lid still "
          "sleeps it)", flush=True)


def _fresh_login(session, brain_config) -> None:
    """Start a long command with a login that lasts (BRAIN ends a login after at most 4 h;
    the next one asks for the biometric check, which a run hours in cannot answer)."""
    within = brain_config.relogin_within_seconds
    if within is None or not hasattr(session, "refresh_if_expiring"):
        return
    left = session.refresh_if_expiring(within)
    if left is not None:
        print(f"BRAIN login valid for {left / 3600:.1f} h", flush=True)


def _unattended(session) -> None:
    """After the start-up login, a research run never waits for a person: a login BRAIN
    ends mid-run raises (the run stops cleanly with its report) instead of asking for the
    biometric check or credentials at a terminal nobody may be watching (owner 2026-11-26)."""
    if hasattr(session, "on_biometric"):
        session.on_biometric = None
    if hasattr(session, "ask_credentials"):
        session.ask_credentials = None


def _research(args) -> int:
    config = load_config(args.config)
    config.require_holdout()  # before logging in or calling the model
    playbook = Playbook.load(args.playbook)
    agents = load_agents_config(args.agents)
    plans = load_plan(args.plan)
    seeds = load_seeds(args.plan)
    spec = FastExprSpec.load(args.fastexpr)
    gen_config = load_generator_config(args.generator_config)
    datafields, field_meta = None, {}
    if args.datafields:
        try:
            datafields = load_datafields(args.datafields)
            field_meta = load_datafield_meta(args.datafields)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigError(f"cannot read datafield catalog {args.datafields}: {exc}") from exc
    markets = _plan_catalogs(plans)
    budget = Budget(max_simulations=args.max_simulations, per_round=args.per_round,
                    max_llm_calls=args.max_llm_calls)
    if min(budget.max_simulations, budget.per_round) < 1 or \
            (budget.max_llm_calls is not None and budget.max_llm_calls < 1):
        raise ConfigError("--max-simulations, --per-round and --max-llm-calls must be >= 1")
    if args.model:
        agents = dataclasses.replace(agents, models=agents.models.with_default(args.model))
        agents.check_models()
    llm_factory = _llm_factory(args, agents.used_models(),  # before logging in to BRAIN
                               agents.models.max_output, agents.models.pace)
    brain_config = load_brain_config(args.brain_config)
    session = _brain_session(brain_config)
    _fresh_login(session, brain_config)
    _keep_awake(args)
    _unattended(session)
    client = RealBrainClient(session, brain_config, capture_dir=args.capture_dir)
    memory = MemoryStore(args.data_dir)
    try:
        run = ResearchRun(plans, agents, budget, llm_factory=llm_factory, client=client,
                          config=config,
                          playbook=playbook, spec=spec, generator_config=gen_config,
                          memory=memory, datafields=datafields, field_meta=field_meta,
                          markets=markets, seeds=seeds, seed=args.seed, out=sys.stdout)
        report = run.run(args.rounds, catch_interrupt=True)   # staged Ctrl-C (ResearchRun.run)
        path = write_report(report, args.report or Path(args.data_dir) / f"{report['run_id']}.json")
    finally:
        memory.close()
    print(f"stopped: {report['stop_reason']} | simulations {report['simulations_used']} | "
          f"model calls {report['model_calls']} | pool {report['pool_size']} | report {path}")
    print(f"  model calls by model: {report['model_calls_by_model']}")
    if report.get("dropped_on_stop"):
        print(f"  not sent to BRAIN after Ctrl-C: {len(report['dropped_on_stop'])} screened alpha(s)")
    for line in quality_lines(report["generation_quality"]):
        print(f"  {line}")
    if report.get("brain_quota"):
        print(f"  {quota_text(report['brain_quota'], 'BRAIN daily simulations')}")
    if report["regions_this_run"] is not None:
        print(f"  regions this run: {report['regions_this_run']}; simulations per region so far: "
              f"{report['simulations_by_region']}")
    active = {a: s for a, s in report["arms"].items()
              if s["accepted"] or s["not_accepted"] or s["repairs_queued"]}
    for arm, s in active.items():   # arms with no evidence yet are in the report file only
        print(f"  {arm}: accepted {s['accepted']}, not accepted {s['not_accepted']}, "
              f"pass rate {s['posterior_pass_rate']}, repairs queued {s['repairs_queued']}")
    if len(active) < len(report["arms"]):
        print(f"  ({len(report['arms']) - len(active)} arms with no results yet are in the report)")
    return EXIT_OK


def _compare(args) -> int:
    """Generator and critic models side by side, from everything in memory."""
    memory = MemoryStore(args.data_dir)
    try:
        table = model_comparison(memory.candidate_outcomes(), memory.critic_reviews())
    finally:
        memory.close()
    text = json.dumps(table, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return EXIT_OK


def _reevaluate(args) -> int:
    """Re-decide, from stored results, rejected alphas a playbook lever can now address."""
    config = load_config(args.config)
    playbook = Playbook.load(args.playbook)
    memory = MemoryStore(args.data_dir)
    try:
        outs = Pipeline(config, playbook, None, memory, out=sys.stdout).reevaluate_rejected()
    finally:
        memory.close()
    print(f"re-decided {len(outs)} rejected alpha(s)" if outs else
          "nothing to re-decide: no rejected alpha has a failure a lever can now fix")
    return EXIT_OK


def _recheck(args) -> int:
    """Fetch BRAIN's result again for alphas decided pending because a BRAIN check had no
    verdict yet (PROD_CORRELATION PENDING, SELF_CORRELATION ERROR), and decide them again.
    Simulates nothing; the run start of `research` does the same."""
    config = load_config(args.config)
    playbook = Playbook.load(args.playbook)
    memory = MemoryStore(args.data_dir)
    try:
        brain_config = load_brain_config(args.brain_config)
        client = RealBrainClient(_brain_session(brain_config), brain_config)
        outs, still = Pipeline(config, playbook, client, memory,
                               out=sys.stdout).recheck_pending(
            set(args.candidate_id) if args.candidate_id else None)
    finally:
        memory.close()
    if not outs and not still:
        print("nothing to recheck: no pending alpha is waiting on a BRAIN check")
    else:
        print(f"re-decided {len(outs)} alpha(s)"
              + (f"; still waiting on BRAIN: {', '.join(still)}" if still else ""))
    return EXIT_OK


def _repair(args) -> int:
    """Rescue alphas by hand: one child per applicable mechanical lever for each named
    candidate in memory, written as a candidate file for ``run --client real``."""
    playbook = Playbook.load(args.playbook)
    spec = FastExprSpec.load(args.fastexpr)
    agents = load_agents_config(args.agents)
    datafields = (catalog(load_datafields(args.datafields), load_datafield_meta(args.datafields))
                  if args.datafields else None)
    memory = MemoryStore(args.data_dir)
    out, notes = [], {}
    try:
        agent = RepairAgent(playbook, spec, memory, datafields=datafields,
                            learning=agents.repair)
        if getattr(args, "config", None) and Path(args.config).exists():
            agent.flip_restricted = load_config(args.config).repair.flip_short_restricted
        for cid in args.candidate_id:
            cand = memory.get_candidate(cid)
            decisions = memory.decisions_for(cid)
            if cand is None or not decisions:
                print(f"error: no candidate with a decision named {cid!r} in {args.data_dir}",
                      file=sys.stderr)
                return EXIT_CONFIG
            results = memory.results_for(cid)
            kids, n = agent.variants(cand, decisions[-1], results[-1] if results else None,
                                     limit=args.limit, ignore_budget=args.ignore_budget)
            out += [k.to_dict() for k in kids]
            notes[cid] = n
            print(f"{cid} ({decisions[-1].decision} / {decisions[-1].failure_class}): "
                  f"{len(kids)} variant(s)" + "".join(f"; {x}" for x in n if x.startswith("no repair")))
            for k in kids:
                print(f"  {k.candidate_id}  {k.lever:<20} {k.universe} {k.neutralization} "
                      f"decay {k.decay}  {k.expression}")
    finally:
        memory.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str) + "\n")
    Path(args.out).with_suffix(".notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(f"wrote {len(out)} candidate(s) to {args.out}")
    return EXIT_OK if out else EXIT_EMPTY


def _levers(args) -> int:
    """What each repair lever has done so far, per failure class, from memory."""
    agents = load_agents_config(args.agents)
    memory = MemoryStore(args.data_dir)
    try:
        events = repair_events(memory.candidate_outcomes(), agents.repair)
    finally:
        memory.close()
    stats = LeverStats.from_events(events, agents.repair.prior, agents.repair.region_prior_strength)
    table = {"lever_stats": stats.table(), "lever_stats_by_region": stats.region_table(),
             "events": [dataclasses.asdict(e) for e in events]}
    text = json.dumps(table, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return EXIT_OK


def _models(args) -> int:
    """List a host's models, or check every model in the agents config with one tiny call."""
    if args.check:
        agents = load_agents_config(args.agents)
        failed = 0
        for m in agents.used_models():
            try:
                reply = make_real_client(m, agents.models.max_output.get(m)).complete(
                    "Reply with the single word OK.", "Say OK.", max_output=256)
                print(f"ok      {m}: {reply.strip()[:40]!r}")
            except LLMUnavailable as exc:
                failed += 1
                print(f"FAILED  {m}: {exc}")
        return EXIT_LLM if failed else EXIT_OK
    if not args.host:
        print("give --host or --check", file=sys.stderr)
        return EXIT_CONFIG
    ids = list_models(args.host)
    for i in ids:
        if not args.filter or args.filter.lower() in i.lower():
            print(f"{args.host}:{i}")
    return EXIT_OK


def _prompts(args) -> int:
    """Print an agent's system prompt as the model will see it, thresholds included."""
    config = load_config(args.config)
    context = agent_prompts.brain_context(config, args.regions)
    builders = {"prefilter": agent_prompts.prefilter_system, "critic": agent_prompts.critic_system,
                "judge": agent_prompts.judge_system, "repair": agent_prompts.repair_system,
                "strategy": agent_prompts.strategy_system,
                "lessons": agent_prompts.lessons_system}
    for name in builders if args.agent == "all" else [args.agent]:
        print(f"===== {name} ({agent_prompts.PROMPT_VERSIONS[name]})\n{builders[name](context)}\n")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alpha_research")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline on a candidate file")
    run.add_argument("--candidates", required=True)
    run.add_argument("--no-caffeinate", action="store_true",
                     help="with --client real: do not keep the Mac awake while it runs")
    run.add_argument("--config", default="config/thresholds.yaml")
    run.add_argument("--playbook", default="config/repair_playbook.yaml")
    run.add_argument("--data-dir", default=str(default_data_dir()))
    run.add_argument("--client", choices=["fixture", "real"], default="fixture")
    run.add_argument("--fixtures-dir", default="tests/fixtures/brain_results")
    run.add_argument("--brain-config", default="config/brain.yaml")
    run.add_argument("--capture-dir", default=None,
                     help="with --client real: save every raw BRAIN response here")
    run.add_argument("--max-concurrent", type=int, default=None,
                     help="simulations on BRAIN at once (default: simulator.max_concurrent in "
                          "--config); over the account's limit BRAIN answers 429 and the "
                          "client waits for a slot")
    run.add_argument("--agents", default="config/agents.yaml",
                     help="lessons section: the Lesson Writer runs after the batch")
    run.add_argument("--fastexpr", default="config/fastexpr.yaml")
    run.add_argument("--no-lessons", action="store_true",
                     help="do not write lessons from memory after this batch")
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
    gen.add_argument("--agents", default=None,
                     help=f"agents config (default {DEFAULT_AGENTS_CONFIG} if present); only "
                          "its models.max_output (per-model output caps) is used here")
    gen.add_argument("--llm", choices=["real", "scripted"], default="real")
    gen.add_argument("--model", default=os.environ.get("ALPHA_RESEARCH_MODEL", DEFAULT_MODEL))
    gen.add_argument("--llm-script", help="JSON list of replies for --llm scripted (offline)")
    gen.set_defaults(func=_generate)

    rs = sub.add_parser("research", help="closed research loop: Director, Generator, "
                        "Pre-Filter, Critic, Debate, BRAIN, Repair, Strategy Evolution")
    rs.add_argument("--plan", required=True, help="research plan YAML (arms, hypotheses, papers)")
    rs.add_argument("--max-simulations", type=int, required=True,
                    help="stop after this many candidates reached BRAIN")
    rs.add_argument("--max-llm-calls", type=int, default=None,
                    help="stop after this many model calls (all agents together); default: no "
                         "cap, the run ends when every generator model has hit its own limit "
                         "(free tiers), the simulation budget or the rounds. Set it for paid models")
    rs.add_argument("--per-round", type=int, default=10, help="simulation slots per round")
    rs.add_argument("--rounds", type=int, default=5)
    rs.add_argument("--seed", type=int, default=None)
    rs.add_argument("--no-caffeinate", action="store_true",
                    help="do not keep the Mac awake while the run goes (default: caffeinate)")
    rs.add_argument("--report", default=None, help="report JSON (default: <data-dir>/<run id>.json)")
    rs.add_argument("--config", default="config/thresholds.yaml")
    rs.add_argument("--playbook", default="config/repair_playbook.yaml")
    rs.add_argument("--agents", default="config/agents.yaml")
    rs.add_argument("--fastexpr", default="config/fastexpr.yaml")
    rs.add_argument("--generator-config", default="config/generator.yaml")
    rs.add_argument("--datafields", help="datafield catalog (see brain datafields)")
    rs.add_argument("--data-dir", default=str(default_data_dir()))
    rs.add_argument("--brain-config", default="config/brain.yaml")
    rs.add_argument("--capture-dir", default=None)
    rs.add_argument("--llm", choices=["real", "scripted"], default="real")
    rs.add_argument("--model", default=os.environ.get("ALPHA_RESEARCH_MODEL"),
                    help="replaces models.default in the agents config")
    rs.add_argument("--llm-script", help="JSON list of replies for --llm scripted (offline)")
    rs.set_defaults(func=_research)

    cp = sub.add_parser("compare", help="compare generator and critic models from memory")
    cp.add_argument("--data-dir", default=str(default_data_dir()))
    cp.add_argument("--out", default=None, help="also write the table to this JSON file")
    cp.set_defaults(func=_compare)

    rv = sub.add_parser("reevaluate", help="re-decide rejected alphas a playbook lever can "
                                           "now address, from their stored results (no BRAIN)")
    rv.add_argument("--data-dir", default=str(default_data_dir()))
    rv.add_argument("--config", default="config/thresholds.yaml")
    rv.add_argument("--playbook", default="config/repair_playbook.yaml")
    rv.set_defaults(func=_reevaluate)

    rc = sub.add_parser("recheck", help="fetch BRAIN's checks again for alphas pending on "
                                        "a check BRAIN had not finished (e.g. PROD_CORRELATION)")
    rc.add_argument("--candidate-id", action="append", default=None,
                    help="only these candidates (repeat for several); default every one")
    rc.add_argument("--data-dir", default=str(default_data_dir()))
    rc.add_argument("--config", default="config/thresholds.yaml")
    rc.add_argument("--playbook", default="config/repair_playbook.yaml")
    rc.add_argument("--brain-config", default="config/brain.yaml")
    rc.set_defaults(func=_recheck)

    rp = sub.add_parser("repair", help="write repair variants of candidates in memory "
                                       "(one per applicable mechanical lever) for run --client real")
    rp.add_argument("--candidate-id", action="append", required=True,
                    help="candidate in memory to rescue (repeat for several)")
    rp.add_argument("--out", required=True, help="candidate file to write")
    rp.add_argument("--limit", type=int, default=None, help="at most this many variants each")
    rp.add_argument("--ignore-budget", action="store_true",
                    help="ignore the lineage repair budget (repair.max_depth, max_trials_per_root, "
                         "patience); every extra simulation still counts in the overfitting check")
    rp.add_argument("--data-dir", default=str(default_data_dir()))
    rp.add_argument("--playbook", default="config/repair_playbook.yaml")
    rp.add_argument("--fastexpr", default="config/fastexpr.yaml")
    rp.add_argument("--agents", default="config/agents.yaml")
    rp.add_argument("--datafields", default=None, help="datafield catalog to validate against")
    rp.add_argument("--config", default="config/thresholds.yaml",
                    help="thresholds (repair.flip_short_restricted)")
    rp.set_defaults(func=_repair)

    lv = sub.add_parser("levers", help="what each repair lever has fixed so far, from memory")
    lv.add_argument("--data-dir", default=str(default_data_dir()))
    lv.add_argument("--agents", default="config/agents.yaml")
    lv.add_argument("--out", default=None, help="also write the table to this JSON file")
    lv.set_defaults(func=_levers)

    md = sub.add_parser("models", help="list a host's models, or --check the configured ones")
    md.add_argument("--host", choices=sorted(HOSTS), default=None)
    md.add_argument("--filter", default=None, help="only ids containing this text")
    md.add_argument("--check", action="store_true",
                    help="send one tiny request to every model config/agents.yaml uses")
    md.add_argument("--agents", default="config/agents.yaml")
    md.set_defaults(func=_models)

    pl = sub.add_parser("pool", help="alphas in our pool; --submitted ID adds one you "
                                     "submitted on BRAIN, --remove ID takes one out")
    pl.add_argument("--data-dir", default=str(default_data_dir()))
    pl.add_argument("--config", default="config/thresholds.yaml")
    pl.add_argument("--submitted", action="append", metavar="ID",
                    help="candidate id or BRAIN alpha id you submitted (repeatable)")
    pl.add_argument("--remove", action="append", metavar="ID",
                    help="candidate id or BRAIN alpha id to take out (repeatable)")
    pl.add_argument("--reason", default=None, help="recorded with --remove")
    pl.add_argument("--sync", action="store_true",
                    help="with pool_admits 'accept': match the latest decisions")
    pl.set_defaults(func=_pool)

    pp = sub.add_parser("power-pool", help="Power Pool candidates in memory (recorded only)")
    pp.add_argument("--data-dir", default=str(default_data_dir()))
    pp.add_argument("--config", default="config/thresholds.yaml")
    pp.add_argument("--region", default=None)
    pp.set_defaults(func=_power_pool)

    qu = sub.add_parser("quality", help="how plain or crowded a run's fresh alphas were")
    qu.add_argument("--run", nargs="*", default=None,
                    help="research run ids (default: the latest run in memory)")
    qu.add_argument("--plan", default="config/research_plan.v5.yaml",
                    help="research plan whose catalogs give the field facts")
    qu.add_argument("--data-dir", default=str(default_data_dir()))
    qu.add_argument("--fastexpr", default="config/fastexpr.yaml")
    qu.add_argument("--generator-config", default="config/generator.yaml")
    qu.set_defaults(func=_quality)

    ls = sub.add_parser("lessons", help="the Lesson Writer's latest lessons (--write: new ones now)")
    ls.add_argument("--data-dir", default=str(default_data_dir()))
    ls.add_argument("--agents", default="config/agents.yaml")
    ls.add_argument("--fastexpr", default="config/fastexpr.yaml")
    ls.add_argument("--write", action="store_true", help="write lessons from memory first")
    ls.add_argument("--model", default=None,
                    help="with --write: let this model condense the evidence (else rules)")
    ls.add_argument("--llm", choices=["real", "scripted"], default="real")
    ls.add_argument("--llm-script", default=None, help="with --llm scripted: JSON list of replies")
    ls.set_defaults(func=_lessons)

    pr = sub.add_parser("prompts", help="print the agents' system prompts with this config's thresholds")
    pr.add_argument("--agent", choices=["all", "prefilter", "critic", "judge", "repair", "strategy",
                                         "lessons"],
                    default="all")
    pr.add_argument("--config", default="config/thresholds.yaml")
    pr.add_argument("--regions", nargs="*", default=["USA"],
                    help="regions whose settings and limits the context lists")
    pr.set_defaults(func=_prompts)

    brain = sub.add_parser("brain", help=f"BRAIN account tools (needs {EMAIL_ENV_VAR} and {SECRET_ENV_VAR})")
    bsub = brain.add_subparsers(dest="brain_command", required=True)
    b_auth = bsub.add_parser("auth", help="log in to BRAIN and save the session for later commands")
    b_auth.set_defaults(func=_brain_auth)
    b_out = bsub.add_parser("logout", help="delete the saved BRAIN session")
    b_out.set_defaults(func=_brain_logout)
    b_can = bsub.add_parser("cancel", help="cancel our simulations stuck on BRAIN (no final "
                                           "result for --older-than-hours)")
    b_can.add_argument("--older-than-hours", type=float, default=None,
                       help="default: simulator.cancel_stuck_after_hours in --config")
    b_can.add_argument("--yes", action="store_true", help="do not ask before cancelling")
    b_can.add_argument("--data-dir", default=str(default_data_dir()))
    b_can.add_argument("--config", default="config/thresholds.yaml")
    b_can.add_argument("--playbook", default="config/repair_playbook.yaml")
    b_can.set_defaults(func=_brain_cancel)
    b_opt = bsub.add_parser("options", help="list the simulation settings BRAIN allows")
    b_opt.add_argument("--out", default="captures/simulation_options.json")
    b_opt.set_defaults(func=_brain_options)
    b_cap = bsub.add_parser("capture", help="simulate candidates and save the raw responses (OQ-2)")
    b_cap.add_argument("--candidates", required=True)
    b_cap.add_argument("--limit", type=int, default=1, help="candidates to simulate (default 1)")
    b_cap.add_argument("--out-dir", default="captures")
    b_cap.add_argument("--max-wait", type=float, default=900.0, help="seconds per simulation")
    b_cap.set_defaults(func=_brain_capture)
    b_ops = bsub.add_parser("operators", help="download BRAIN's operator list and diff it "
                            "against the seed list (OQ-10)")
    b_ops.add_argument("--out", default="captures/operators.json")
    b_ops.add_argument("--fastexpr", default="config/fastexpr.yaml")
    b_ops.set_defaults(func=_brain_operators)
    b_df = bsub.add_parser("datafields", help="download a datafield catalog for generate --datafields")
    b_df.add_argument("--region", required=True)
    b_df.add_argument("--universe", required=True)
    b_df.add_argument("--delay", type=int, required=True)
    b_df.add_argument("--dataset", help="limit to one dataset id")
    b_df.add_argument("--out", required=True)
    b_df.set_defaults(func=_brain_datafields)
    b_ds = bsub.add_parser("datasets", help="list the datasets (with field counts) for a region/universe/delay")
    b_ds.add_argument("--region", required=True)
    b_ds.add_argument("--universe", required=True)
    b_ds.add_argument("--delay", type=int, required=True)
    b_ds.add_argument("--out", required=True)
    b_ds.set_defaults(func=_brain_datasets)
    b_div = bsub.add_parser("diversity", help="our submitted alphas by region, delay and "
                                              "data category (BRAIN's count)")
    b_div.set_defaults(func=_brain_diversity)
    for sp in (b_auth, b_out, b_can, b_opt, b_cap, b_ops, b_df, b_ds, b_div):
        sp.add_argument("--brain-config", default="config/brain.yaml")
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
    except FixtureNotFoundError as exc:
        print(f"client error: {exc}. Fixture runs can only simulate candidates that have a "
              "recorded BRAIN result; generated candidates need the real client (--client real).",
              file=sys.stderr)
        return EXIT_CLIENT
    except (LLMUnavailable, GenerationError) as exc:
        print(f"model error: {exc}", file=sys.stderr)
        return EXIT_LLM
    except (BrainApiError, SimulationQuotaExhausted, ConnectionError, TimeoutError) as exc:
        print(f"BRAIN error: {exc}", file=sys.stderr)
        return EXIT_BRAIN


if __name__ == "__main__":
    sys.exit(main())
