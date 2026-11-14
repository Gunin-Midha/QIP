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
import sys
import time
from pathlib import Path

import yaml

from .brain_api import (EMAIL_ENV_VAR, SECRET_ENV_VAR, BrainApiError, RealBrainClient,
                        UrllibTransport, list_datafields, list_datasets, list_operators,
                        load_brain_config, make_session,
                        simulation_options)
from .brain_client import FixtureBrainClient, FixtureNotFoundError, SubmissionRejected
from .compare import model_comparison
from .config import ConfigError, load_config
from .fastexpr import FastExprSpec, catalog
from .generator import (AlphaGenerator, GenerationError, GenerationSettings, Source,
                        load_datafield_meta, load_datafields, load_generator_config)
from .llm import HOSTS, LLMUnavailable, ScriptedLLMClient, list_models, make_real_client
from .memory import MemoryStore, default_data_dir
from .orchestrator import run_rounds
from .pipeline import Pipeline, load_candidate_file
from .repair import Playbook
from .lever_stats import LeverStats, repair_events
from .repair_agent import RepairAgent
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


def _llm_factory(args, models: list[str]):
    """model id -> client. Scripted: one client answers for every model id. Real: one client
    per model id, all built now so a missing key or package fails before BRAIN is called."""
    if args.llm == "scripted":
        scripted = _llm_client(args)
        return lambda model: scripted
    clients = {m: make_real_client(m) for m in models}
    return lambda model: clients[model] if model in clients else make_real_client(model)


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
    """A BRAIN session that prompts for biometric verification in a terminal."""
    return make_session(config, on_biometric=_biometric_prompt if sys.stdin.isatty() else None)


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
    if min(budget.max_simulations, budget.per_round, budget.max_llm_calls) < 1:
        raise ConfigError("--max-simulations, --per-round and --max-llm-calls must be >= 1")
    if args.model:
        agents = dataclasses.replace(agents, models=agents.models.with_default(args.model))
        agents.check_models()
    llm_factory = _llm_factory(args, agents.used_models())  # before logging in to BRAIN
    brain_config = load_brain_config(args.brain_config)
    client = RealBrainClient(_brain_session(brain_config), brain_config, capture_dir=args.capture_dir)
    memory = MemoryStore(args.data_dir)
    try:
        run = ResearchRun(plans, agents, budget, llm_factory=llm_factory, client=client,
                          config=config,
                          playbook=playbook, spec=spec, generator_config=gen_config,
                          memory=memory, datafields=datafields, field_meta=field_meta,
                          markets=markets, seeds=seeds, seed=args.seed, out=sys.stdout)
        report = run.run(args.rounds)
        path = write_report(report, args.report or Path(args.data_dir) / f"{report['run_id']}.json")
    finally:
        memory.close()
    print(f"stopped: {report['stop_reason']} | simulations {report['simulations_used']} | "
          f"model calls {report['model_calls']} | pool {report['pool_size']} | report {path}")
    print(f"  model calls by model: {report['model_calls_by_model']}")
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
                reply = make_real_client(m).complete(
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
                "strategy": agent_prompts.strategy_system}
    for name in builders if args.agent == "all" else [args.agent]:
        print(f"===== {name} ({agent_prompts.PROMPT_VERSIONS[name]})\n{builders[name](context)}\n")
    return EXIT_OK


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
    run.add_argument("--brain-config", default="config/brain.yaml")
    run.add_argument("--capture-dir", default=None,
                     help="with --client real: save every raw BRAIN response here")
    run.add_argument("--max-concurrent", type=int, default=None,
                     help="simulations on BRAIN at once (default: simulator.max_concurrent in "
                          "--config); over the account's limit BRAIN answers 429 and the "
                          "client waits for a slot")
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

    rs = sub.add_parser("research", help="closed research loop: Director, Generator, "
                        "Pre-Filter, Critic, Debate, BRAIN, Repair, Strategy Evolution")
    rs.add_argument("--plan", required=True, help="research plan YAML (arms, hypotheses, papers)")
    rs.add_argument("--max-simulations", type=int, required=True,
                    help="stop after this many candidates reached BRAIN")
    rs.add_argument("--max-llm-calls", type=int, required=True,
                    help="stop after this many model calls (all agents together)")
    rs.add_argument("--per-round", type=int, default=10, help="simulation slots per round")
    rs.add_argument("--rounds", type=int, default=5)
    rs.add_argument("--seed", type=int, default=None)
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

    pr = sub.add_parser("prompts", help="print the agents' system prompts with this config's thresholds")
    pr.add_argument("--agent", choices=["all", "prefilter", "critic", "judge", "repair", "strategy"],
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
    for sp in (b_auth, b_out, b_opt, b_cap, b_ops, b_df, b_ds):
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
    except (BrainApiError, ConnectionError, TimeoutError) as exc:
        print(f"BRAIN error: {exc}", file=sys.stderr)
        return EXIT_BRAIN


if __name__ == "__main__":
    sys.exit(main())
