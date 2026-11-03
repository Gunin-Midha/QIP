"""Research loop: connects the agents into one closed loop (``cli research``).

One round:

    Research Director (Thompson sampling over arms =
                       dataset x region x prompt_version x generator model)
        | slots per arm, capped by the simulation budget
        v
    queued repairs for the arm  +  Alpha Generator (next hypothesis or paper of the arm)
        v
    Cheap Pre-Filter            drops invalid, already-simulated and near-duplicate ideas
        v
    Critic -> Debate judge      drops flawed tests, or swaps in a revised expression
      (+ shadow critics)        the other critic models review the same batch; recorded only
        v
    Pipeline (BRAIN simulation, gates, risk, diversity, pool, memory)
        |-- accept / reject / repair  -> Director.update(arm, accepted)
        |-- repair                    -> Repair Agent -> child queued for the same arm
        v
    every N rounds: Strategy Evolution -> guidance + hypotheses as a new prompt version,
                    added as new arms so the Director measures whether it helps

Models (``models`` in config/agents.yaml): each agent has its own model, and the provider
comes from the model id (``llm.provider_of``). Every generator model is a separate arm, so
the Director compares them on accept rate. The primary critic for an arm is the first
listed critic from a different provider than the arm's generator, so the two do not share
blind spots; a config with no such critic is an error. With ``shadow_critics`` the other
critic models review the same batch and their verdicts are stored in memory
(``critic_reviews``) next to the primary's, for ``compare.py``. Shadows never change what
is simulated.

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
from .critic import Critic, DebateJudge, ReviewOutcome, review_and_debate
from .compare import arm_models, model_comparison
from .fastexpr import FastExprSpec
from .generator import (AlphaGenerator, GenerationError, GenerationSettings, GeneratorConfig,
                        Source)
from .llm import LLMClient, LLMUnavailable, provider_of
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


class CallCounter:
    """One model-call budget shared by every agent and every model."""

    def __init__(self, max_calls: int | None):
        self.max_calls = max_calls
        self.calls = 0
        self.by_model: dict[str, int] = {}


class BudgetedLLM:
    """Counts every model call across all agents and stops at ``max_calls``. ``model``
    labels the calls (and the candidates the client writes); it defaults to the inner
    client's model."""

    def __init__(self, inner: LLMClient, max_calls: int | None = None, *,
                 counter: CallCounter | None = None, model: str | None = None):
        self.inner = inner
        self.counter = counter or CallCounter(max_calls)
        self.model = model or getattr(inner, "model", "unknown")
        self.provider = provider_of(self.model)

    @property
    def calls(self) -> int:
        return self.counter.calls

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        c = self.counter
        if c.max_calls is not None and c.calls >= c.max_calls:
            raise LLMBudgetExhausted(f"model call budget of {c.max_calls} used up")
        c.calls += 1
        c.by_model[self.model] = c.by_model.get(self.model, 0) + 1
        return self.inner.complete(system, user, max_output=max_output)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelsConfig:
    """Model id per agent. ``generator`` and ``critic`` are lists: every generator model is
    an arm, and critics are ordered by preference."""
    default: str
    generator: tuple[str, ...]
    critic: tuple[str, ...]
    shadow_critics: bool
    prefilter: str
    debate: str
    repair: str
    strategy: str

    def all_models(self) -> list[str]:
        return list(dict.fromkeys([*self.generator, *self.critic, self.prefilter, self.debate,
                                   self.repair, self.strategy]))

    def primary_critic(self, generator_model: str) -> str | None:
        """First listed critic whose provider differs from the generator's."""
        g = provider_of(generator_model)
        return next((c for c in self.critic if provider_of(c) != g), None)

    def with_default(self, model: str) -> "ModelsConfig":
        """Replace the default model (``--model``); agents that named their own keep it."""
        swap = lambda m: model if m == self.default else m
        return dataclasses.replace(
            self, default=model, generator=tuple(swap(m) for m in self.generator),
            critic=tuple(dict.fromkeys(swap(m) for m in self.critic)),
            prefilter=swap(self.prefilter), debate=swap(self.debate), repair=swap(self.repair),
            strategy=swap(self.strategy))


@dataclass(frozen=True)
class AgentsConfig:
    prefilter: PrefilterConfig
    critic_model: bool
    debate_model: bool
    repair_model: bool
    strategy: StrategyConfig
    floor_share: float
    max_output: int
    models: ModelsConfig

    def used_models(self) -> list[str]:
        """Model ids this config will call (agents with the model switched off left out)."""
        m = self.models
        used = list(m.generator)
        used += [m.strategy] if self.strategy.every_rounds else []
        used += list(m.critic) if self.critic_model else []
        used += [m.prefilter] if self.prefilter.use_model else []
        used += [m.debate] if self.debate_model else []
        used += [m.repair] if self.repair_model else []
        return list(dict.fromkeys(used))

    def check_models(self) -> None:
        """Every model id has a known provider; with the critic model on, every generator
        model has a critic from another provider."""
        for m in self.models.all_models():
            if provider_of(m) is None:
                raise ConfigError(f"agents config: unknown provider for model {m!r} "
                                  "(expected claude-* or gemini-*)")
        if self.critic_model:
            for g in self.models.generator:
                if self.models.primary_critic(g) is None:
                    raise ConfigError(
                        f"agents config: generator model {g!r} needs a critic from a different "
                        f"provider ({provider_of(g)}); add one to models.critic")


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


def _model_list(sec: dict, key: str, default: str) -> tuple[str, ...]:
    v = sec.get(key)
    if v is None:
        return (default,)
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not v or not all(isinstance(m, str) and m for m in v):
        raise ConfigError(f"agents config: models.{key} must be a model id or a list of them")
    return tuple(dict.fromkeys(v))


def models_config_from_dict(sec) -> ModelsConfig:
    if not isinstance(sec, dict):
        raise ConfigError("agents config: missing section 'models'")
    default = sec.get("default")
    if not isinstance(default, str) or not default:
        raise ConfigError("agents config: models.default must be a model id")
    single = {}
    for key in ("prefilter", "debate", "repair", "strategy"):
        v = sec.get(key)
        if v is not None and (not isinstance(v, str) or not v):
            raise ConfigError(f"agents config: models.{key} must be null or a model id")
        single[key] = v or default
    shadows = sec.get("shadow_critics")
    if not isinstance(shadows, bool):
        raise ConfigError("agents config: models.shadow_critics must be true or false")
    return ModelsConfig(default=default, generator=_model_list(sec, "generator", default),
                        critic=_model_list(sec, "critic", default), shadow_critics=shadows,
                        **single)


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
        models=models_config_from_dict(raw.get("models")),
    )


def load_agents_config(path: str | Path) -> AgentsConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read agents config {path}: {exc}") from exc
    agents = agents_config_from_dict(raw)
    agents.check_models()
    return agents


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
                 llm_factory: Callable[[str], LLMClient], client: BrainClient,
                 config: Config, playbook: Playbook,
                 spec: FastExprSpec, generator_config: GeneratorConfig, memory: MemoryStore,
                 datafields: list[str] | None = None, seed: int | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, out: TextIO | None = None,
                 run_id: str | None = None):
        self.budget = budget
        self.agents = agents
        agents.check_models()
        self.models = agents.models
        self.counter = CallCounter(budget.max_llm_calls)
        self._llm_factory = llm_factory
        self._llms: dict[str, BudgetedLLM] = {}
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
        m = self.models
        pf_llm = self.llm(m.prefilter) if agents.prefilter.use_model else None
        self.prefilter = PreFilter(agents.prefilter, spec, memory, llm=pf_llm,
                                   datafields=datafields, limits=limits, max_output=mo, context=ctx)
        # one Critic per critic model; without the critic model, one rules-only Critic
        critic_models = m.critic if agents.critic_model else (None,)
        self.critics = {cm: Critic(spec, llm=self.llm(cm) if cm else None, datafields=datafields,
                                   limits=limits, max_output=mo, context=ctx)
                        for cm in critic_models}
        self.judge = DebateJudge(llm=self.llm(m.debate) if agents.debate_model else None,
                                 max_output=mo, context=ctx)
        self.repairer = RepairAgent(playbook, spec, memory,
                                    llm=self.llm(m.repair) if agents.repair_model else None,
                                    datafields=datafields, limits=limits, max_output=mo,
                                    context=ctx)
        st_llm = self.llm(m.strategy) if agents.strategy.every_rounds else None
        self.strategist = StrategyEvolution(agents.strategy, llm=st_llm, max_output=mo,
                                            context=ctx)

        base_pv = generator_config.prompt_version
        self.arms: dict[tuple, ArmPlan] = {}
        self.generators: dict[tuple, AlphaGenerator] = {}
        for p in plans:
            for gm in m.generator:
                # every generator model is its own arm, with its own copy of the sources so
                # each model works through the same hypotheses in the same order
                self.arms[(p.dataset, p.settings.region, base_pv, gm)] = dataclasses.replace(
                    p, sources=deque(p.sources))
        self.director = Orchestrator(self.arms.keys(), seed=seed)
        self.repair_queue: dict[tuple, deque] = {k: deque() for k in self.arms}
        self.sims_used = 0
        self.log: list[dict] = []
        self.stop_reason: str | None = None
        self.warm_started = self._warm_start()

    # ------------------------------------------------------------------ setup

    def llm(self, model: str) -> BudgetedLLM:
        """The budgeted client for a model id, built once and shared by every agent."""
        if model not in self._llms:
            self._llms[model] = BudgetedLLM(self._llm_factory(model), counter=self.counter,
                                            model=model)
        return self._llms[model]

    def _critics_for(self, key: tuple) -> tuple[str | None, list[str]]:
        """(primary critic model, shadow critic models) for an arm."""
        if not self.agents.critic_model:
            return None, []
        primary = self.models.primary_critic(key[3])
        shadows = [cm for cm in self.critics if cm != primary] if self.models.shadow_critics else []
        return primary, shadows

    def _warm_start(self) -> int:
        n = 0
        outcomes = self.memory.candidate_outcomes()
        models = arm_models(outcomes, self.memory.critic_reviews())
        for row in outcomes:
            c, d = row["candidate"], row["decision"]
            key = (c.dataset, c.region, c.prompt_version, models.get(c.candidate_id))
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
            self.generators[key] = AlphaGenerator(self.llm(key[3]), self.spec, plan.settings,
                                                  cfg, datafields=self.datafields,
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
        primary, shadows = self._critics_for(key)
        log["critic"] = primary
        rv = review_and_debate(pf.kept, rationales, self.critics[primary], self.judge)
        self._record_reviews(key, primary, "primary", rv.reviews)
        for cm in shadows:  # recorded for compare.py; never changes what is simulated
            shadow = ReviewOutcome()
            reviews = self.critics[cm].review(pf.kept, rationales, shadow)
            self._record_reviews(key, cm, "shadow", reviews)
            log.setdefault("shadow_verdicts", {})[cm] = {cid: r.verdict
                                                         for cid, r in reviews.items()}
        log["critic_dropped"] = [[c.candidate_id, why] for c, why in rv.dropped]
        log["revised"] = [c.candidate_id for c in rv.to_simulate if c.lever == "critic_revision"]
        log["outcomes"] = []
        for cand in rv.to_simulate:
            if self.sims_used >= self.budget.max_simulations:
                break
            self._simulate(key, cand, log)

    def _record_reviews(self, key: tuple, critic_model: str | None, role: str,
                        reviews: dict) -> None:
        for cid, r in reviews.items():
            self.memory.put_critic_review(
                run_id=self.run_id, candidate_id=cid, critic_model=critic_model, role=role,
                verdict=r.verdict, source=r.source, generator_model=key[3],
                payload={"issues": r.issues, "blocking": r.blocking,
                         "revised": r.revised.expression if r.revised else None})

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
            new_key = (key[0], key[1], evo.prompt_version, key[3])
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
            "model_calls": self.counter.calls,
            "model_calls_by_model": dict(self.counter.by_model),
            "models": {"generator": list(self.models.generator),
                       "critic": list(self.critics) if self.agents.critic_model else [],
                       "shadow_critics": self.models.shadow_critics,
                       "prefilter": self.models.prefilter, "debate": self.models.debate,
                       "repair": self.models.repair, "strategy": self.models.strategy},
            "prompt_versions": {"generator": self.generator_config.prompt_version,
                                **PROMPT_VERSIONS},
            "warm_start_outcomes": self.warm_started,
            "pool_size": self.pipeline.pool.count(),
            "arms": arms,
            "comparison": model_comparison(self.memory.candidate_outcomes(),
                                           self.memory.critic_reviews()),
            "rounds": self.log,
        }


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return path
