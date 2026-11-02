"""Research loop: connects the agents into one closed loop (``cli research``).

One round:

    Research Director (Thompson sampling over arms = dataset x region x prompt_version)
        | slots per arm, capped by the simulation budget
        v
    queued repairs for the arm  +  Alpha Generator (next hypothesis or paper of the arm)
        v
    Cheap Pre-Filter            drops invalid, already-simulated and near-duplicate ideas
        v
    Critic -> Debate judge      drops flawed tests, or swaps in a revised expression
        v
    Pipeline (BRAIN simulation, gates, risk, diversity, pool, memory)
        |-- accept / reject / repair  -> Director.update(arm, accepted)
        |-- repair                    -> Repair Agent -> child queued for the same arm
        v
    every N rounds: Strategy Evolution -> guidance + hypotheses as a new prompt version,
                    added as new arms so the Director measures whether it helps

Budgets stop the run cleanly: ``max_simulations`` (candidates that reached BRAIN; invalid
and duplicate ones do not count) and ``max_llm_calls`` (every model call by every agent).
The Director starts from the outcomes already in memory, so a second run continues where
the first stopped. Nothing here submits alphas for competition.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TextIO

import yaml

from .brain_client import BrainClient
from .config import Config, ConfigError
from .critic import Critic, DebateJudge, review_and_debate
from .fastexpr import FastExprSpec
from .generator import (AlphaGenerator, GenerationError, GenerationSettings, GeneratorConfig,
                        Source)
from .llm import LLMClient, LLMUnavailable
from .memory import MemoryStore
from .orchestrator import Orchestrator
from .pipeline import Pipeline
from .prompts import PROMPT_VERSIONS, brain_context
from .prefilter import PreFilter, PrefilterConfig
from .repair import Playbook
from .repair_agent import RepairAgent
from .schemas import CandidateRecord
from .strategy import StrategyConfig, StrategyEvolution

DECIDED = ("accept", "reject", "repair")


class LLMBudgetExhausted(LLMUnavailable):
    pass


class BudgetedLLM:
    """Counts every model call across all agents and stops at ``max_calls``."""

    def __init__(self, inner: LLMClient, max_calls: int | None):
        self.inner = inner
        self.model = getattr(inner, "model", "unknown")
        self.max_calls = max_calls
        self.calls = 0

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise LLMBudgetExhausted(f"model call budget of {self.max_calls} used up")
        self.calls += 1
        return self.inner.complete(system, user, max_output=max_output)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentsConfig:
    prefilter: PrefilterConfig
    critic_model: bool
    debate_model: bool
    repair_model: bool
    strategy: StrategyConfig
    floor_share: float
    max_output: int


def _bool(sec: dict, name: str, key: str) -> bool:
    v = sec.get(key)
    if not isinstance(v, bool):
        raise ConfigError(f"agents config: {name}.{key} must be true or false")
    return v


def _opt_num(sec: dict, name: str, key: str, lo: float, hi: float) -> float | None:
    v = sec.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise ConfigError(f"agents config: {name}.{key} must be null or in [{lo}, {hi}]")
    return float(v)


def _pos_int(sec: dict, name: str, key: str, lo: int = 1) -> int:
    v = sec.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ConfigError(f"agents config: {name}.{key} must be an integer >= {lo}")
    return v


def agents_config_from_dict(raw) -> AgentsConfig:
    if not isinstance(raw, dict):
        raise ConfigError("agents config must be a mapping")
    secs = {}
    for name in ("prefilter", "critic", "debate", "repair", "strategy", "director"):
        sec = raw.get(name)
        if not isinstance(sec, dict):
            raise ConfigError(f"agents config: missing section {name!r}")
        secs[name] = sec
    pf, st = secs["prefilter"], secs["strategy"]
    floor = secs["director"].get("floor_share")
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not 0 <= floor <= 1:
        raise ConfigError("agents config: director.floor_share must be in [0, 1]")
    return AgentsConfig(
        prefilter=PrefilterConfig(
            near_duplicate_similarity=_opt_num(pf, "prefilter", "near_duplicate_similarity", 0, 1),
            use_model=_bool(pf, "prefilter", "use_model"),
            min_model_score=_opt_num(pf, "prefilter", "min_model_score", 0, 10)),
        critic_model=_bool(secs["critic"], "critic", "use_model"),
        debate_model=_bool(secs["debate"], "debate", "use_model"),
        repair_model=_bool(secs["repair"], "repair", "use_model"),
        strategy=StrategyConfig(
            every_rounds=_pos_int(st, "strategy", "every_rounds", lo=0),
            max_hypotheses=_pos_int(st, "strategy", "max_hypotheses", lo=0),
            max_guidance_chars=_pos_int(st, "strategy", "max_guidance_chars")),
        floor_share=float(floor),
        max_output=_pos_int(raw, "agents", "max_output", lo=256),
    )


def load_agents_config(path: str | Path) -> AgentsConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read agents config {path}: {exc}") from exc
    return agents_config_from_dict(raw)


@dataclass
class ArmPlan:
    dataset: str
    settings: GenerationSettings
    sources: deque = field(default_factory=deque)   # Source objects, used round-robin
    guidance: str | None = None


def load_plan(path: str | Path) -> list[ArmPlan]:
    """Research plan YAML: a list of arms, each with a dataset label, simulation settings,
    hypotheses and/or paper text files (paths relative to the plan file)."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read research plan {path}: {exc}") from exc
    arms = raw.get("arms") if isinstance(raw, dict) else None
    if not isinstance(arms, list) or not arms:
        raise ConfigError("research plan needs a non-empty 'arms' list")
    plans, seen = [], set()
    for i, a in enumerate(arms):
        if not isinstance(a, dict) or not isinstance(a.get("dataset"), str) or not a["dataset"]:
            raise ConfigError(f"research plan arm #{i} needs a 'dataset' label")
        if not isinstance(a.get("settings"), dict):
            raise ConfigError(f"research plan arm #{i} needs a 'settings' mapping")
        settings = GenerationSettings.from_mapping({**a["settings"], "dataset": a["dataset"]})
        sources = deque()
        for h in a.get("hypotheses") or []:
            if not isinstance(h, str) or not h.strip():
                raise ConfigError(f"research plan arm #{i}: hypotheses must be non-empty strings")
            sources.append(Source("hypothesis", h.strip()))
        for p in a.get("papers") or []:
            p = (path.parent / p) if not Path(p).is_absolute() else Path(p)
            try:
                sources.append(Source("paper", p.read_text(), title=p.stem))
            except (OSError, ValueError) as exc:
                raise ConfigError(f"research plan arm #{i}: cannot read paper {p}: {exc}") from exc
        if not sources:
            raise ConfigError(f"research plan arm #{i} needs at least one hypothesis or paper")
        key = (a["dataset"], settings.region)
        if key in seen:
            raise ConfigError(f"research plan lists dataset {key[0]!r} in {key[1]} twice")
        seen.add(key)
        plans.append(ArmPlan(a["dataset"], settings, sources))
    return plans


@dataclass(frozen=True)
class Budget:
    max_simulations: int
    per_round: int
    max_llm_calls: int | None = None


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

class ResearchRun:
    def __init__(self, plans: list[ArmPlan], agents: AgentsConfig, budget: Budget, *,
                 llm: LLMClient, client: BrainClient, config: Config, playbook: Playbook,
                 spec: FastExprSpec, generator_config: GeneratorConfig, memory: MemoryStore,
                 datafields: list[str] | None = None, seed: int | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, out: TextIO | None = None,
                 run_id: str | None = None):
        self.budget = budget
        self.agents = agents
        self.llm = BudgetedLLM(llm, budget.max_llm_calls)
        self.memory = memory
        self.spec = spec
        self.generator_config = generator_config
        self.datafields = datafields
        self.out = out
        self.run_id = run_id or "rs" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        self.pipeline = Pipeline(config, playbook, client, memory, sleep=sleep, clock=clock,
                                 out=out)
        limits, mo = generator_config.limits, agents.max_output
        ctx = brain_context(config)  # thresholds from config, shown to every agent's model
        self.prefilter = PreFilter(agents.prefilter, spec, memory,
                                   llm=self.llm if agents.prefilter.use_model else None,
                                   datafields=datafields, limits=limits, max_output=mo, context=ctx)
        self.critic = Critic(spec, llm=self.llm if agents.critic_model else None,
                             datafields=datafields, limits=limits, max_output=mo, context=ctx)
        self.judge = DebateJudge(llm=self.llm if agents.debate_model else None, max_output=mo,
                                 context=ctx)
        self.repairer = RepairAgent(playbook, spec, memory,
                                    llm=self.llm if agents.repair_model else None,
                                    datafields=datafields, limits=limits, max_output=mo,
                                    context=ctx)
        self.strategist = StrategyEvolution(agents.strategy, llm=self.llm, max_output=mo,
                                            context=ctx)

        base_pv = generator_config.prompt_version
        self.arms: dict[tuple, ArmPlan] = {}
        self.generators: dict[tuple, AlphaGenerator] = {}
        for p in plans:
            self.arms[(p.dataset, p.settings.region, base_pv)] = p
        self.director = Orchestrator(self.arms.keys(), seed=seed)
        self.repair_queue: dict[tuple, deque] = {k: deque() for k in self.arms}
        self.sims_used = 0
        self.log: list[dict] = []
        self.stop_reason: str | None = None
        self.warm_started = self._warm_start()

    # ------------------------------------------------------------------ setup

    def _warm_start(self) -> int:
        n = 0
        for row in self.memory.candidate_outcomes():
            c, d = row["candidate"], row["decision"]
            key = (c.dataset, c.region, c.prompt_version)
            if d is not None and d.decision in DECIDED and key in self.director.state:
                self.director.update(key, d.decision == "accept")
                n += 1
        return n

    def _emit(self, text: str) -> None:
        if self.out is not None:
            print(text, file=self.out)

    def _generator(self, key: tuple) -> AlphaGenerator:
        if key not in self.generators:
            plan = self.arms[key]
            cfg = dataclasses.replace(self.generator_config, prompt_version=key[2])
            self.generators[key] = AlphaGenerator(self.llm, self.spec, plan.settings, cfg,
                                                  datafields=self.datafields,
                                                  guidance=plan.guidance)
        return self.generators[key]

    # ------------------------------------------------------------------ run

    def run(self, rounds: int) -> dict:
        for r in range(1, rounds + 1):
            if self.sims_used >= self.budget.max_simulations:
                self.stop_reason = "simulation budget used up"
                break
            try:
                self._round(r)
            except LLMUnavailable as exc:  # budget used up, or the model failed
                self.stop_reason = str(exc)
                break
            if self.stop_reason:
                break
            if self.strategist.due(r):
                try:
                    self._evolve(r)
                except LLMUnavailable as exc:  # budget used up, or the model failed
                    self.stop_reason = str(exc)
                    break
        else:
            self.stop_reason = self.stop_reason or f"completed {rounds} rounds"
        return self.report()

    def _round(self, r: int) -> None:
        slots = min(self.budget.per_round, self.budget.max_simulations - self.sims_used)
        alloc = self.director.allocate(slots, self.agents.floor_share)
        entry = {"round": r, "allocation": {"/".join(map(str, k)): n for k, n in alloc.items()},
                 "arms": []}
        self.log.append(entry)
        self._emit(f"--- round {r}: {slots} slots, allocation {entry['allocation']}")
        for i, (key, n) in enumerate(alloc.items()):
            if n == 0:
                continue
            arm_log = {"arm": "/".join(map(str, key)), "slots": n}
            entry["arms"].append(arm_log)
            self._arm(r, i, key, n, arm_log)
            if self.sims_used >= self.budget.max_simulations:
                self.stop_reason = "simulation budget used up"
                return

    def _arm(self, r: int, i: int, key: tuple, n: int, log: dict) -> None:
        queue = self.repair_queue[key]
        batch = [queue.popleft() for _ in range(min(n, len(queue)))]
        log["repairs_taken"] = len(batch)
        rationales: dict[str, str] = {}
        need = n - len(batch)
        if need > 0:
            plan = self.arms[key]
            source = plan.sources[0]
            plan.sources.rotate(-1)
            try:
                res = self._generator(key).generate(source, need, batch_id=f"{self.run_id}r{r:02d}a{i}")
            except GenerationError as exc:
                log["generation_error"] = str(exc)
                res = None
            if res is not None:
                batch += res.candidates
                rationales.update(res.rationales)
                log["generated"] = len(res.candidates)
                log["generator_rejected"] = len(res.rejected)

        pf = self.prefilter.run(batch)
        log["prefilter_dropped"] = [[c.candidate_id, why] for c, why in pf.dropped]
        rv = review_and_debate(pf.kept, rationales, self.critic, self.judge)
        log["critic_dropped"] = [[c.candidate_id, why] for c, why in rv.dropped]
        log["revised"] = [c.candidate_id for c in rv.to_simulate if c.lever == "critic_revision"]
        log["outcomes"] = []
        for cand in rv.to_simulate:
            if self.sims_used >= self.budget.max_simulations:
                break
            self._simulate(key, cand, log)

    def _simulate(self, key: tuple, cand: CandidateRecord, log: dict) -> None:
        outcome = self.pipeline.process(cand)
        if outcome.result is not None and outcome.result.status != "invalid":
            self.sims_used += 1
        item = {"candidate_id": cand.candidate_id, "decision": outcome.decision,
                "failure_class": outcome.failure_class}
        log["outcomes"].append(item)
        if outcome.decision in DECIDED:
            self.director.update(key, outcome.decision == "accept")
        if outcome.decision == "repair" and outcome.decision_record is not None:
            pool = [c.expression for a in self.pipeline.pool.alpha_ids()
                    if (c := self.memory.get_candidate(a)) is not None]
            rep = self.repairer.repair(cand, outcome.decision_record, outcome.result, pool)
            item["repair"] = rep.notes
            if rep.child is not None:
                self.repair_queue[key].append(rep.child)
                item["repair_child"] = rep.child.candidate_id

    def _evolve(self, r: int) -> None:
        evo = self.strategist.evolve(self.memory.candidate_outcomes(),
                                     self.generator_config.prompt_version)
        if evo is None:
            return
        added = []
        for key, plan in list(self.arms.items()):
            if key[2] != self.generator_config.prompt_version:
                continue
            new_key = (key[0], key[1], evo.prompt_version)
            sources = deque(plan.sources)
            for ds, text in evo.hypotheses:
                if ds is None or ds == key[0]:
                    sources.appendleft(Source("hypothesis", text))
            self.arms[new_key] = ArmPlan(plan.dataset, plan.settings, sources, evo.guidance)
            self.repair_queue[new_key] = deque()
            if self.director.add_arm(new_key):
                added.append("/".join(map(str, new_key)))
        self.log[-1]["evolution"] = {"prompt_version": evo.prompt_version, "source": evo.source,
                                     "guidance": evo.guidance, "hypotheses": evo.hypotheses,
                                     "new_arms": added, "notes": evo.notes}
        self._emit(f"--- strategy evolution: {evo.prompt_version} ({evo.source}), new arms {added}")

    # ------------------------------------------------------------------ report

    def report(self) -> dict:
        arms = {}
        for key in self.director.arms:
            s = self.director.state[key]
            arms["/".join(map(str, key))] = {"accepted": s.successes, "not_accepted": s.failures,
                                             "posterior_pass_rate": round(self.director.pass_rate(key), 4),
                                             "repairs_queued": len(self.repair_queue[key])}
        return {
            "run_id": self.run_id,
            "stop_reason": self.stop_reason,
            "simulations_used": self.sims_used,
            "model_calls": self.llm.calls,
            "model": self.llm.model,
            "prompt_versions": {"generator": self.generator_config.prompt_version,
                                **PROMPT_VERSIONS},
            "warm_start_outcomes": self.warm_started,
            "pool_size": self.pipeline.pool.count(),
            "arms": arms,
            "rounds": self.log,
        }


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return path
