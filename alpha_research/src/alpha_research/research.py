"""Research loop: connects the agents into one closed loop (``cli research``).

One round:

    Research Director (Thompson sampling over arms =
                       dataset x region x prompt_version x generator model x delay)
        | slots per arm, capped by the simulation budget
        v
    queued repairs (any arm, up to repair.max_share_per_round of the slots) first, then
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
                                         (levers ranked by what worked before, lever_stats)
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
the first stopped. With ``repair.resume_from_memory``, a repair decision whose child never
reached BRAIN (the run stopped, or the arm got no slot) is repaired again at the start, for
arms in the current plan. Nothing here submits alphas for competition.

A model that fails after its client's retries (a free tier's daily quota, an outage) is
down for the rest of the run, not a reason to stop it: its generator arms get no more
slots (their queued repairs still run), the Critic and judge fall back to their rules, and
Pre-Filter, Repair and Strategy switch to the next model that is still up (their own model
first, then the generator models, then the others). The run stops only when no generator
model is left.

Slots a round could not use go back to the Director in the same round (``_round``): slots
of a generator model that went down (always, since run rs20261104163055 lost 3 to Gemini's
quota) and slots emptied by the generator, Pre-Filter or Critic (up to
``director.refill_passes`` extra passes, so a drop costs a model call, not a simulation).

Repair children skip the Critic's model review and the judge: a rewrite of a repair would
change two things at once, and the lever learner would credit the lever with the Critic's
change. They still get the rule checks (a blocking issue drops them).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, TextIO

import yaml

from .brain_client import BrainClient, BrainLoginRequired
from .config import Config, ConfigError
from .critic import Critic, DebateJudge, ReviewOutcome, SparseFieldRule, review_and_debate
from .compare import arm_models, model_comparison
from .fastexpr import FastExprSpec, analyze, catalog, catalog_for
from .generator import (AlphaGenerator, GenerationError, GenerationSettings, GeneratorConfig,
                        Source, dataset_category)
from .llm import LLMClient, LLMUnavailable, ModelDown, provider_of
from .memory import MemoryStore
from .orchestrator import Orchestrator, balance_slots
from . import power_pool
from .pipeline import Pipeline
from .prompts import PROMPT_VERSIONS, bar_text, brain_context
from .prefilter import PreFilter, PrefilterConfig
from .repair import Playbook
from .lessons import LessonsConfig, for_agent, learn, summary
from .lever_stats import LeverStats, RepairLearningConfig, distance, repair_events
from .repair_agent import RepairAgent
from .schemas import (GROUP_NEUTRALIZATIONS, NO_REGULAR_SIMULATION, REGION_SETTINGS,
                      CandidateRecord, experiment_hash)
from .strategy import (StrategyConfig, StrategyEvolution, brain_limits,
                       prod_correlation_fields, region_notes)

DECIDED = ("accept", "reject", "repair")


class LLMBudgetExhausted(LLMUnavailable):
    pass


class CallCounter:
    """One model-call budget shared by every agent and every model."""

    def __init__(self, max_calls: int | None):
        self.max_calls = max_calls
        self.calls = 0
        self.by_model: dict[str, int] = {}
        self.down: dict[str, str] = {}   # model -> why it is out for the rest of the run


class BudgetedLLM:
    """Counts every model call across all agents and stops at ``max_calls``. ``model``
    labels the calls (and the candidates the client writes); it defaults to the inner
    client's model. A call that fails after the client's own retries (quota, outage, bad
    key) marks the model down for the rest of the run and raises ``ModelDown``; later calls
    to it raise at once without spending budget."""

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
        if self.model in c.down:
            raise ModelDown(self.model, c.down[self.model])
        if c.max_calls is not None and c.calls >= c.max_calls:
            raise LLMBudgetExhausted(f"model call budget of {c.max_calls} used up")
        c.calls += 1
        c.by_model[self.model] = c.by_model.get(self.model, 0) + 1
        try:
            return self.inner.complete(system, user, max_output=max_output)
        except ModelDown as exc:
            c.down[self.model] = exc.reason
            raise
        except LLMUnavailable as exc:
            c.down[self.model] = str(exc)
            raise ModelDown(self.model, str(exc)) from exc


class FallbackLLM:
    """An agent's model with fallbacks: calls the first model that is not down, in order.
    Raises ``ModelDown`` only when every model in the list is down."""

    def __init__(self, models: list[BudgetedLLM]):
        self.models = models

    def _live(self) -> list[BudgetedLLM]:
        return [m for m in self.models if m.model not in m.counter.down]

    @property
    def model(self) -> str:
        live = self._live()
        return (live or self.models)[0].model

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        for m in self._live():
            try:
                return m.complete(system, user, max_output=max_output)
            except ModelDown:
                continue
        names = ", ".join(m.model for m in self.models)
        raise ModelDown(names, "every model for this agent is down")


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
    # model id -> output cap that replaces the agents' cap on every call to that model
    # (reasoning models think inside the cap; e.g. 16384 for a model that needs room)
    max_output: dict = field(default_factory=dict)
    # model id -> {calls_per_minute, tokens_per_minute}: free-tier limits to pace calls to
    pace: dict = field(default_factory=dict)
    # how an arm's primary critic is chosen among the critics from another maker that are
    # up: "balanced" (the one with the fewest calls so far this run, ties in list order) or
    # "first" (the first listed)
    critic_pick: str = "balanced"

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
    max_arm_share: float | None = None
    refill_passes: int = 0   # extra allocation passes per round for slots emptied by drops
    region_floor_share: float | None = None   # each region's minimum share of a round
    max_region_share: float | None = None     # no region above this share while others are below
    regions_per_run: int | None = None        # rotation: work on this many regions per run
    region_balance_since: str | None = None   # ISO date: count simulations per region from here
    repair: RepairLearningConfig = field(default_factory=RepairLearningConfig)
    sparse: SparseFieldRule | None = None     # critic.sparse_fields; None = rule off
    price_categories: tuple[str, ...] = ()    # critic.price_volume_categories; () = advisory off
    lessons: LessonsConfig = field(default_factory=lambda: LessonsConfig(after_every_batch=False))

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
                                  "(expected claude-*, gemini-* or a hosted id such as groq:qwen/qwen3.8-27b)")
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


def _opt_date(sec: dict, name: str, key: str) -> str | None:
    v = sec.get(key)
    if v is None:
        return None
    try:
        return (v if isinstance(v, date) else date.fromisoformat(str(v))).isoformat()
    except ValueError:
        raise ConfigError(f"agents config: {name}.{key} must be null or an ISO date") from None


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
    caps = sec.get("max_output") or {}
    if not isinstance(caps, dict) or not all(
            isinstance(m, str) and m and isinstance(v, int) and not isinstance(v, bool)
            and v >= 256 for m, v in caps.items()):
        raise ConfigError("agents config: models.max_output must map model ids to "
                          "integers >= 256")
    pace = sec.get("pace") or {}
    keys = {"calls_per_minute", "tokens_per_minute"}
    if not isinstance(pace, dict) or not all(
            isinstance(m, str) and isinstance(v, dict) and v and set(v) <= keys
            and all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in v.values())
            for m, v in pace.items()):
        raise ConfigError("agents config: models.pace must map model ids to "
                          "{calls_per_minute, tokens_per_minute} positive integers")
    pick = sec.get("critic_pick", "balanced")
    if pick not in ("balanced", "first"):
        raise ConfigError("agents config: models.critic_pick must be balanced or first")
    return ModelsConfig(default=default, generator=_model_list(sec, "generator", default),
                        critic=_model_list(sec, "critic", default), shadow_critics=shadows,
                        max_output=dict(caps), pace={m: dict(v) for m, v in pace.items()},
                        critic_pick=pick, **single)


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
            max_guidance_chars=_pos_int(st, "strategy", "max_guidance_chars"),
            max_new_arms=(_pos_int(st, "strategy", "max_new_arms", lo=0)
                          if "max_new_arms" in st else StrategyConfig.max_new_arms)),
        floor_share=float(floor),
        max_output=_pos_int(raw, "agents", "max_output", lo=256),
        models=models_config_from_dict(raw.get("models")),
        repair=RepairLearningConfig.from_dict(secs["repair"]),
        max_arm_share=_opt_num(secs["director"], "director", "max_arm_share", 0, 1),
        region_floor_share=_opt_num(secs["director"], "director", "region_floor_share", 0, 1),
        max_region_share=_opt_num(secs["director"], "director", "max_region_share", 0, 1),
        refill_passes=(_pos_int(secs["director"], "director", "refill_passes", lo=0)
                       if "refill_passes" in secs["director"] else 0),
        regions_per_run=(None if secs["director"].get("regions_per_run") is None
                         else _pos_int(secs["director"], "director", "regions_per_run")),
        region_balance_since=_opt_date(secs["director"], "director", "region_balance_since"),
        sparse=_sparse_rule(secs["critic"]),
        price_categories=_categories(secs["critic"], "price_volume_categories"),
        lessons=LessonsConfig.from_dict(raw.get("lessons")),
    )


def _categories(sec: dict, key: str) -> tuple[str, ...]:
    raw = sec.get(key) or []
    if not isinstance(raw, list) or not all(isinstance(c, str) and c for c in raw):
        raise ConfigError(f"agents config: critic.{key} must be a list of category names")
    return tuple(raw)


def _sparse_rule(sec: dict) -> SparseFieldRule | None:
    try:
        return SparseFieldRule.from_dict(sec.get("sparse_fields"))
    except ValueError as exc:
        raise ConfigError(f"agents config: {exc}") from exc


def load_agents_config(path: str | Path) -> AgentsConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read agents config {path}: {exc}") from exc
    agents = agents_config_from_dict(raw)
    agents.check_models()
    return agents


def _lever_tables(stats: LeverStats) -> dict:
    return {"lever_stats": stats.table(), "lever_stats_by_region": stats.region_table()}


@dataclass
class ArmPlan:
    dataset: str
    settings: GenerationSettings
    sources: deque = field(default_factory=deque)   # Source objects, used round-robin
    guidance: str | None = None
    catalog: Path | None = None   # the region/delay's datafield catalog (plan v2)


def _short_id(candidate_id: str) -> str:
    """A candidate id without its research run prefix (rs + 14 digits), for the console."""
    return re.sub(r"^rs\d{14}", "", candidate_id) or candidate_id


def arm_label(key: tuple) -> str:
    """An arm key (dataset, region, prompt_version, model, delay) as report text:
    ``dataset/REGION/D<delay>/prompt_version/model``."""
    ds, region, pv, model, delay = key
    return f"{ds}/{region}/D{delay}/{pv}/{model}"


def arm_short(key: tuple) -> str:
    """An arm as console text: ``dataset REGION D<delay> (model)``, the model without its
    host and maker prefix, and the prompt version only when evolved."""
    ds, region, pv, model, delay = key
    evo = f" {pv.split('+', 1)[1]}" if "+" in pv else ""
    head, _, rest = model.partition(":")
    name = rest if rest and "/" not in head else model      # drop a "groq:" host prefix
    name = name.split("/")[-1].removesuffix(":free")
    return f"{ds} {region} D{delay}{evo} ({name})"


def _slots_text(alloc: dict) -> str:
    return ", ".join(f"{arm_short(k)}{f' x{n}' if n > 1 else ''}" for k, n in alloc.items() if n)


def load_plan(path: str | Path) -> list[ArmPlan]:
    """Research plan YAML in one of two layouts (paper paths relative to the plan file).

    v1, ``arms``: a list of arms, each with a dataset label, full simulation settings and
    hypotheses and/or papers.
    v2, ``regions``: one block per BRAIN region with ``enabled`` (default true),
    ``settings`` (region defaults), ``catalogs`` ({delay: datafield catalog path}) and
    ``arms`` (dataset, delay, optional settings, hypotheses and/or papers); ``defaults``
    holds settings for every arm. An arm's settings are defaults <- region settings <- arm
    settings, plus its region and delay, and are checked against what BRAIN allows in that
    region (``schemas.REGION_SETTINGS``)."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read research plan {path}: {exc}") from exc
    if isinstance(raw, dict) and "regions" in raw:
        return _load_plan_v2(raw, path)
    arms = raw.get("arms") if isinstance(raw, dict) else None
    if not isinstance(arms, list) or not arms:
        raise ConfigError("research plan needs a non-empty 'arms' list or a 'regions' mapping")
    plans, seen = [], set()
    for i, a in enumerate(arms):
        where = f"research plan arm #{i}"
        if not isinstance(a, dict) or not isinstance(a.get("dataset"), str) or not a["dataset"]:
            raise ConfigError(f"{where} needs a 'dataset' label")
        if not isinstance(a.get("settings"), dict):
            raise ConfigError(f"{where} needs a 'settings' mapping")
        settings = GenerationSettings.from_mapping({**a["settings"], "dataset": a["dataset"]})
        _check_start_neutralization(settings, where)
        plans.append(ArmPlan(a["dataset"], settings, _plan_sources(a, where, path)))
        _check_unique(plans[-1], seen)
    return plans


def _plan_sources(a: dict, where: str, path: Path) -> deque:
    sources = deque()
    for h in a.get("hypotheses") or []:
        if not isinstance(h, str) or not h.strip():
            raise ConfigError(f"{where}: hypotheses must be non-empty strings")
        sources.append(Source("hypothesis", h.strip()))
    for p in a.get("papers") or []:
        p = (path.parent / p) if not Path(p).is_absolute() else Path(p)
        try:
            sources.append(Source("paper", p.read_text(), title=p.stem))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"{where}: cannot read paper {p}: {exc}") from exc
    if not sources:
        raise ConfigError(f"{where} needs at least one hypothesis or paper")
    return sources


def _check_unique(plan: ArmPlan, seen: set) -> None:
    key = (plan.dataset, plan.settings.region, plan.settings.delay)
    if key in seen:
        raise ConfigError(f"research plan lists dataset {key[0]!r} in {key[1]} delay {key[2]} twice")
    seen.add(key)


def _entry_settings(a, region: str, defaults: dict, rsettings: dict,
                    where: str) -> GenerationSettings:
    """An arm's or seed's settings: defaults <- region settings <- its own, plus region and
    delay, checked against what BRAIN allows in the region."""
    allowed = REGION_SETTINGS[region]
    if region in NO_REGULAR_SIMULATION:
        raise ConfigError(f"{where}: BRAIN does not simulate regular alphas in region {region} "
                          "(\"not available for simulation type REGULAR\"); set enabled: false")
    if not isinstance(a, dict) or not isinstance(a.get("dataset"), str) or not a["dataset"]:
        raise ConfigError(f"{where} needs a 'dataset' label")
    if not isinstance(a.get("settings") or {}, dict):
        raise ConfigError(f"{where}: 'settings' must be a mapping")
    delay = a.get("delay")
    if delay not in allowed["delays"]:
        raise ConfigError(f"{where} ({a['dataset']}): delay {delay!r} is not one of "
                          f"{list(allowed['delays'])} for {region}")
    settings = GenerationSettings.from_mapping(
        {**defaults, **rsettings, **(a.get("settings") or {}),
         "region": region, "delay": delay, "dataset": a["dataset"]})
    if settings.universe not in allowed["universes"]:
        raise ConfigError(f"{where} ({a['dataset']}): universe {settings.universe!r} "
                          f"is not allowed in {region}")
    if settings.neutralization not in allowed["neutralizations"]:
        raise ConfigError(f"{where} ({a['dataset']}): neutralization "
                          f"{settings.neutralization!r} is not allowed in {region}")
    _check_start_neutralization(settings, where)
    return settings


def _check_start_neutralization(settings: GenerationSettings, where: str) -> None:
    """A plan's default neutralization is a group one: as a default for every idea a factor
    neutralization can remove the signal itself (run rs20261111162604: revision ideas under
    REVERSION_AND_MOMENTUM). The Generator may still choose a factor one or NONE for one
    alpha with a reason the Critic reviews (gen-v5), and the Repair Agent tries them as
    levers."""
    if settings.neutralization not in GROUP_NEUTRALIZATIONS:
        raise ConfigError(f"{where} ({settings.dataset}): neutralization "
                          f"{settings.neutralization!r} as a starting setting; use one of "
                          f"{', '.join(GROUP_NEUTRALIZATIONS)} (the Generator may pick a factor one or "
                          "NONE per alpha with a reason; the Repair Agent tries them as levers)")


@dataclass(frozen=True)
class SeedSpec:
    """An alpha from outside the plan's ideas to re-simulate first and then repair: one
    from memory by candidate id, or an inline expression with its settings."""
    from_memory: str | None = None
    expression: str | None = None
    settings: GenerationSettings | None = None   # inline seeds (dataset in settings.dataset)


def load_seeds(path: str | Path) -> list[SeedSpec]:
    """The plan's optional ``seeds`` list. Each entry is ``from_memory: <candidate id>`` or
    an inline seed: ``expression``, ``dataset``, ``region``, ``delay`` and optional
    ``settings``, laid over the plan's ``defaults`` and that region's settings."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read research plan {path}: {exc}") from exc
    seeds = (raw or {}).get("seeds") if isinstance(raw, dict) else None
    if seeds is None:
        return []
    if not isinstance(seeds, list):
        raise ConfigError("research plan 'seeds' must be a list")
    defaults = raw.get("defaults") or {}
    regions = raw.get("regions") or {}
    out = []
    for i, s in enumerate(seeds):
        where = f"research plan seed #{i}"
        if isinstance(s, dict) and set(s) == {"from_memory"}:
            if not isinstance(s["from_memory"], str) or not s["from_memory"]:
                raise ConfigError(f"{where}: from_memory must be a candidate id")
            out.append(SeedSpec(from_memory=s["from_memory"]))
            continue
        if not isinstance(s, dict) or not isinstance(s.get("expression"), str) \
                or not s["expression"].strip():
            raise ConfigError(f"{where} needs 'from_memory' or an 'expression'")
        region = s.get("region")
        if region not in REGION_SETTINGS:
            raise ConfigError(f"{where}: region {region!r} is not one of {sorted(REGION_SETTINGS)}")
        rsettings = (regions.get(region) or {}).get("settings") or {}
        settings = _entry_settings(s, region, defaults, rsettings, where)
        out.append(SeedSpec(expression=" ".join(s["expression"].split()), settings=settings))
    return out


def _load_plan_v2(raw: dict, path: Path) -> list[ArmPlan]:
    defaults = raw.get("defaults") or {}
    regions = raw["regions"]
    if not isinstance(defaults, dict):
        raise ConfigError("research plan 'defaults' must be a mapping")
    if not isinstance(regions, dict) or not regions:
        raise ConfigError("research plan 'regions' must be a non-empty mapping")
    plans, seen = [], set()
    for region, block in regions.items():
        if region not in REGION_SETTINGS:
            raise ConfigError(f"research plan region {region!r} is not one of "
                              f"{sorted(REGION_SETTINGS)}")
        if not isinstance(block, dict):
            raise ConfigError(f"research plan region {region} must be a mapping")
        enabled = block.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f"research plan region {region}: 'enabled' must be true or false")
        if not enabled:
            continue
        rsettings, catalogs = block.get("settings") or {}, block.get("catalogs") or {}
        arms = block.get("arms")
        if not isinstance(rsettings, dict) or not isinstance(catalogs, dict):
            raise ConfigError(f"research plan region {region}: 'settings' and 'catalogs' "
                              "must be mappings")
        if not isinstance(arms, list) or not arms:
            raise ConfigError(f"research plan region {region} needs a non-empty 'arms' list")
        for i, a in enumerate(arms):
            where = f"research plan {region} arm #{i}"
            settings = _entry_settings(a, region, defaults, rsettings, where)
            delay = settings.delay
            cat = catalogs.get(delay)
            catalog = None if cat is None else (Path(cat) if Path(cat).is_absolute()
                                                else path.parent / cat)
            plans.append(ArmPlan(a["dataset"], settings, _plan_sources(a, where, path),
                                 catalog=catalog))
            _check_unique(plans[-1], seen)
    if not plans:
        raise ConfigError("research plan has no arms in an enabled region")
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
                 datafields: list[str] | None = None, field_meta: dict | None = None,
                 markets: dict[tuple, tuple[list[str], dict]] | None = None,
                 seeds: list[SeedSpec] | None = None,
                 seed: int | None = None,
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
        self.field_meta = field_meta or {}
        # {(region, delay): (datafields, meta)}: a multi-region plan's catalog per market,
        # ahead of the one --datafields catalog for markets without their own
        self.markets = markets or {}
        typed = catalog(datafields, self.field_meta)  # agents' validators check vec_* use
        mtyped = {k: catalog(f, meta) for k, (f, meta) in self.markets.items()}
        self._typed, self._mtyped = typed, mtyped
        self.seed_specs = list(seeds or [])
        self.seed_queue: list[tuple[tuple, CandidateRecord]] = []
        self.seed_notes: list[dict] = []
        self.out = out
        self.run_id = run_id or "rs" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        self.pipeline = Pipeline(config, playbook, client, memory, sleep=sleep, clock=clock,
                                 out=out)
        limits, mo = generator_config.limits, agents.max_output
        # the Lesson Writer's latest lessons (lessons.py), read once at the start
        self.lessons = memory.latest_lessons()
        # thresholds from config and the plan's regions, shown to every agent's model
        self.seen_limits = brain_limits(memory.candidate_outcomes())   # BRAIN's, per market
        self.ctx = ctx = brain_context(config, sorted({p.settings.region for p in plans}),
                            seen_limits=self.seen_limits)
        m = self.models
        pf_llm = self.fallback(m.prefilter) if agents.prefilter.use_model else None
        self.prefilter = PreFilter(agents.prefilter, spec, memory, llm=pf_llm,
                                   datafields=typed, markets=mtyped, limits=limits,
                                   max_output=mo, context=ctx)
        # one Critic per critic model, plus a rules-only Critic (None) for when they are down
        critic_models = (*m.critic, None) if agents.critic_model else (None,)
        rule = agents.sparse
        sparse = ({None: rule.fields(self.field_meta),
                   **{k: rule.fields(meta) for k, (_, meta) in self.markets.items()}}
                  if rule else {})
        cats = set(agents.price_categories)
        price = ({k: {f for f, x in meta.items() if (x or {}).get("category") in cats}
                  for k, meta in [(None, self.field_meta),
                                  *((k, meta) for k, (_, meta) in self.markets.items())]}
                 if cats else {})
        self.critics = {cm: Critic(spec, llm=self.llm(cm) if cm else None, datafields=typed,
                                   markets=mtyped, limits=limits, max_output=mo, context=ctx,
                                   sparse=sparse,
                                   sparse_lookback=rule.lookback if rule else 20,
                                   price_fields=price)
                        for cm in critic_models}
        self.judge = DebateJudge(llm=self.fallback(m.debate) if agents.debate_model else None,
                                 max_output=mo, context=ctx)
        self.repairer = RepairAgent(playbook, spec, memory,
                                    llm=self.fallback(m.repair) if agents.repair_model else None,
                                    datafields=typed, markets=mtyped, limits=limits,
                                    max_output=mo, context=ctx, learning=agents.repair,
                                    seed=seed,
                                    field_meta={None: self.field_meta,
                                                **{k: meta for k, (_, meta) in self.markets.items()}},
                                    crowding=(generator_config.crowded_percentile,
                                              generator_config.crowded_min_alphas),
                                    generator_config=generator_config,
                                    lessons=self.lessons)
        st_llm = self.fallback(m.strategy) if agents.strategy.every_rounds else None
        self.strategist = StrategyEvolution(agents.strategy, llm=st_llm, max_output=mo,
                                            context=ctx)

        base_pv = generator_config.prompt_version
        self.arms: dict[tuple, ArmPlan] = {}
        self.generators: dict[tuple, AlphaGenerator] = {}
        for p in plans:
            for gm in m.generator:
                # every generator model is its own arm, with its own copy of the sources so
                # each model works through the same hypotheses in the same order
                key = (p.dataset, p.settings.region, base_pv, gm, p.settings.delay)
                self.arms[key] = dataclasses.replace(p, sources=deque(p.sources))
        self.director = Orchestrator(self.arms.keys(), seed=seed)
        self.repair_queue: dict[tuple, deque] = {k: deque() for k in self.arms}
        # Strategy Evolution: arm -> the evolved arm that replaced it. A replaced arm gets no
        # new slots (its queued repairs still run) and its outcomes credit the successor
        self.successor: dict[tuple, tuple] = {}
        self.sims_used = 0
        # alphas screened and waiting for BRAIN: sent simulator.max_concurrent at a time
        # (OQ-3), then decided in order (_flush)
        self.sim_queue: list[tuple[tuple, CandidateRecord, dict, bool]] = []
        self._in_flight = 0   # alphas of the current batch not decided yet (budget counts them)
        self._round_waves = 0  # repair waves started this round (repair.waves_per_round)
        self.max_concurrent = max(1, config.simulator.max_concurrent)
        self.log: list[dict] = []
        self.stop_reason: str | None = None
        self.resumed: list[dict] = []
        self.power_pool: list[dict] = []   # Power Pool candidates this run (recorded only)
        self.recovered: list[dict] = []
        self._resume_rows: list[tuple] = []
        # repair waves (owner 2026-11-23): fixes_per_wave children of one parent simulated
        # together; when all are decided the next wave starts from the best child if it got
        # closer to accept, else from the same parent with its next levers
        self.waves: dict[str, dict] = {}       # wave id -> {arm, parent, children, pending, done}
        self.wave_of: dict[str, str] = {}      # child id -> wave id, until it is decided
        self.wave_log: list[dict] = []
        self.overfitting_flags: list[dict] = []   # deflated-Sharpe misses, recorded only
        self.warm_started = self._warm_start()
        self.region_sims, self.run_regions = self._pick_regions()

    # ------------------------------------------------------------------ setup

    def llm(self, model: str) -> BudgetedLLM:
        """The budgeted client for a model id, built once and shared by every agent."""
        if model not in self._llms:
            self._llms[model] = BudgetedLLM(self._llm_factory(model), counter=self.counter,
                                            model=model)
        return self._llms[model]

    def fallback(self, model: str) -> FallbackLLM:
        """``model`` first, then the generator models, then every other model the run uses."""
        order = dict.fromkeys([model, *self.models.generator, *self.agents.used_models()])
        return FallbackLLM([self.llm(m) for m in order])

    def catalog_of(self, region: str, delay: int) -> tuple[list[str] | None, dict]:
        """(datafields, meta) for a market: its own catalog, else the --datafields one."""
        return self.markets.get((region, delay), (self.datafields, self.field_meta))

    def down(self) -> dict[str, str]:
        """Models out for the rest of the run, with the reason."""
        return dict(self.counter.down)

    def _critics_for(self, key: tuple) -> tuple[str | None, list[str]]:
        """(primary critic model, shadow critic models) for an arm. The primary is a critic
        from another provider that is not down: the one with the fewest model calls so far
        in this run (``models.critic_pick: balanced``, ties in list order, so no single
        model carries every arm's review into its rate limit) or the first listed
        (``first``); with none left, the rules-only critic (None)."""
        if not self.agents.critic_model:
            return None, []
        g = provider_of(key[3])
        eligible = [c for c in self.models.critic
                    if provider_of(c) != g and c not in self.counter.down]
        if self.models.critic_pick == "balanced" and eligible:
            primary = min(eligible, key=lambda c: (self.counter.by_model.get(c, 0),
                                                   eligible.index(c)))
        else:
            primary = eligible[0] if eligible else None
        shadows = ([cm for cm in self.critics
                    if cm not in (primary, None) and cm not in self.counter.down]
                   if self.models.shadow_critics else [])
        return primary, shadows

    def _arm_of(self, c: CandidateRecord, model: str | None) -> tuple | None:
        """The arm a candidate in memory belongs to in this run: its own arm when that still
        exists, else the same (dataset, region, model, delay) under the current prompt
        version, so a prompt version bump (gen-v3 -> gen-v4) or a Strategy version from an
        earlier run does not orphan its outcomes and open repairs. None = no such arm."""
        key = (c.dataset, c.region, c.prompt_version, model, c.delay)
        if key in self.arms:
            return key
        key = (c.dataset, c.region, self.generator_config.prompt_version, model, c.delay)
        return key if key in self.arms else None

    def _warm_start(self) -> int:
        """Credit the Director with every decided alpha in memory, counted the way a run
        counts them: repair children of one parent (its waves) as one result, a success
        when any of them was accepted; every other alpha as itself."""
        n = 0
        outcomes = self.memory.candidate_outcomes()
        models = arm_models(outcomes, self.memory.critic_reviews())
        waves: dict[str, list[tuple[tuple, bool]]] = {}
        for row in outcomes:
            c, d = row["candidate"], row["decision"]
            key = self._arm_of(c, models.get(c.candidate_id))
            if d is None or d.decision not in DECIDED or key is None:
                continue
            if c.parent_id and c.lever and c.lever != "critic_revision":
                waves.setdefault(c.parent_id, []).append((key, d.decision == "accept"))
                continue
            self.director.update(key, d.decision == "accept")
            n += 1
        for members in waves.values():
            self.director.update(members[0][0], any(ok for _, ok in members))
            n += 1
        return n

    def _pick_regions(self) -> tuple[dict[str, int], list[str] | None]:
        """Region rotation (``director.regions_per_run``): simulations per plan region in
        memory since ``director.region_balance_since``, and the regions this run works on:
        the fewest simulations first, then the longest since the last one, then plan order.
        None = every region (no rotation)."""
        regions = list(dict.fromkeys(k[1] for k in self.arms))
        seen = self.memory.simulations_by_region(self.agents.region_balance_since)
        counts = {r: seen.get(r, {}).get("simulations", 0) for r in regions}
        k = self.agents.regions_per_run
        if k is None or k >= len(regions):
            return counts, None
        last = {r: seen.get(r, {}).get("last") or "" for r in regions}
        chosen = sorted(regions, key=lambda r: (counts[r], last[r], regions.index(r)))[:k]
        chosen = [r for r in regions if r in chosen]  # plan order
        done = ", ".join(f"{r} {n}" for r, n in counts.items() if n) or "none yet"
        self._emit(f"--- regions this run: {', '.join(chosen)} (fewest simulations so far; "
                   f"done: {done})")
        return counts, chosen

    def _in_run(self, key: tuple) -> bool:
        return self.run_regions is None or key[1] in self.run_regions

    def _emit(self, text: str) -> None:
        if self.out is not None:
            print(text, file=self.out, flush=True)

    def _generator(self, key: tuple) -> AlphaGenerator:
        plan = self.arms[key]
        if key not in self.generators:
            cfg = dataclasses.replace(self.generator_config, prompt_version=key[2])
            fields, meta = self.catalog_of(plan.settings.region, plan.settings.delay)
            self.generators[key] = AlphaGenerator(self.llm(key[3]), self.spec, plan.settings,
                                                  cfg, datafields=fields, field_meta=meta)
        # the region's own evidence from memory, refreshed every time, then the arm's guidance
        outcomes = self.memory.candidate_outcomes()
        notes = region_notes(outcomes, plan.settings.region)
        self.generators[key].prod_failed = prod_correlation_fields(
            outcomes, plan.settings.region, self.spec,
            self.catalog_of(plan.settings.region, plan.settings.delay)[0])
        bar = bar_text(self.seen_limits.get((plan.settings.region, plan.settings.delay), {}))
        bar = (f"- BRAIN's bar in this market ({plan.settings.region} delay "
               f"{plan.settings.delay}, from its checks on our alphas): {bar}.") if bar else None
        parts = [p for p in (bar, notes, plan.guidance) if p]
        self.generators[key].guidance = "\n".join(parts) or None
        meta = self.catalog_of(plan.settings.region, plan.settings.delay)[1]
        self.generators[key].learned = for_agent(
            self.lessons, "generator", plan.settings.region, dataset_category(plan.dataset, meta))
        return self.generators[key]

    # ------------------------------------------------------------------ run

    def _resume_repairs(self) -> None:
        """Start a repair wave for the open repairs in memory, closest to accept first, at
        most ``repair.max_resume`` waves; the lineage budget still applies to each. An open
        root (a repair decision with no child) starts from itself. Children of one parent
        with no child of their own were a wave that stopped before its next one: the next
        wave starts from the best of them if it got closer than the parent, else from the
        parent again (_next_start)."""
        outcomes = self.memory.candidate_outcomes()
        parents = {row["candidate"].parent_id for row in outcomes}
        by_id = {row["candidate"].candidate_id: row for row in outcomes}
        models = arm_models(outcomes, self.memory.critic_reviews())
        cfg = self.agents.repair
        roots, groups = [], {}
        for row in outcomes:
            c, d = row["candidate"], row["decision"]
            if d is None or c.candidate_id in parents:
                continue
            if c.parent_id and c.parent_id in by_id:
                groups.setdefault(c.parent_id, []).append(row)
            elif d.decision == "repair":
                roots.append((row, row, ()))
        starts = []
        for pid, rows in groups.items():
            if any(r["decision"].decision == "accept" for r in rows):
                continue   # the wave already produced an accept: the lineage is done
            parent = by_id[pid]
            start = self._next_start(
                (parent["candidate"], parent["decision"], parent["result"]),
                [(r["candidate"], r["decision"], r["result"]) for r in rows])
            if start is not None:
                c, d, res, priority, _ = start
                starts.append(({"candidate": c, "decision": d, "result": res}, parent, priority))
        for row, origin, priority in roots + starts:
            c = row["candidate"]
            key = self._arm_of(c, models.get(c.candidate_id) or
                               models.get(origin["candidate"].candidate_id))
            if key is None or not self._in_run(key):
                continue
            dist = distance(row["decision"], row["result"], cfg.unmeasured_failure_penalty)
            starts_row = (float("inf") if dist is None else dist, c.candidate_id, key, row,
                          priority)
            self._resume_rows.append(starts_row)
        self._resume_rows.sort(key=lambda t: (t[0], t[1]))
        for _, _, key, row, priority in self._resume_rows:
            if sum(1 for i in self.resumed if i.get("children")) >= cfg.max_resume:
                break
            c = row["candidate"]
            rep = self._start_wave(key, c, row["decision"], row["result"], priority)
            item = {"parent": c.candidate_id, "arm": arm_label(key), "notes": rep.notes}
            if rep.children:
                item["children"] = [x.candidate_id for x in rep.children]
                item["fixes"] = [x.lever for x in rep.children]
            self.resumed.append(item)
        waves = [i for i in self.resumed if i.get("children")]
        if waves:
            self._emit(f"--- resumed {len(waves)} repair wave(s) from memory:")
            width = max(len(_short_id(i["parent"])) for i in waves)
            for i in waves:
                self._emit(f"    {_short_id(i['parent']):<{width}}  {len(i['fixes'])} fixes: "
                           f"{', '.join(i['fixes'])}")

    def _next_start(self, parent: tuple, siblings: list[tuple]):
        """Where the next wave starts after a wave of ``siblings`` (candidate, decision,
        result) of ``parent``: (candidate, decision, result, priority levers, why), or None.
        The best sibling still in repair that got closer to accept than the parent, with
        the levers of the other siblings that got closer first; else the parent itself,
        whose tried levers the Repair Agent skips; None when the parent is not in repair."""
        pen = self.agents.repair.unmeasured_failure_penalty
        pc, pd, pr = parent
        pdist = distance(pd, pr, pen)
        scored = []
        for c, d, r in siblings:
            dist = distance(d, r, pen)
            if d is not None and d.decision == "repair" and dist is not None:
                scored.append((dist, c.candidate_id, c, d, r))
        closer = sorted(x for x in scored if pdist is None or x[0] < pdist)
        if closer:
            best = closer[0]
            priority = [x[2].lever for x in closer[1:] if x[2].lever]
            return (best[2], best[3], best[4], priority,
                    f"from {best[1]}: distance to accept {pdist} -> {best[0]}")
        if pd is not None and pd.decision == "repair":
            return pc, pd, pr, (), "no child got closer: the next levers on the same parent"
        return None

    def _start_wave(self, key: tuple, cand: CandidateRecord, decision, result,
                    priority=()):
        """Ask the Repair Agent for a wave of fixes of ``cand`` and queue them together."""
        rep = self.repairer.repair_wave(cand, decision, result, self._pool_expressions(),
                                        priority=priority)
        if rep.children:
            wid = f"{cand.candidate_id}/w{len(self.wave_log) + 1}"
            ids = [c.candidate_id for c in rep.children]
            self.waves[wid] = {"arm": key, "parent": (cand, decision, result),
                               "pending": set(ids), "done": []}
            for c in rep.children:
                self.wave_of[c.candidate_id] = wid
                self.repair_queue[key].append(c)
            self.wave_log.append({"wave": wid, "parent": cand.candidate_id, "arm": arm_label(key),
                                  "fixes": {c.candidate_id: c.lever for c in rep.children},
                                  "size": next((n for n in rep.notes
                                                if n.startswith("wave size")), None),
                                  "priority": list(priority)})
        return rep

    def _wave_done(self, cid: str, cand: CandidateRecord | None = None, outcome=None) -> None:
        """A wave child was decided (``outcome``) or dropped before BRAIN (None)."""
        wid = self.wave_of.pop(cid, None)
        wave = self.waves.get(wid) if wid else None
        if wave is None:
            return
        wave["pending"].discard(cid)
        if outcome is not None:
            wave["done"].append((cand, outcome))
        if not wave["pending"]:
            self._finish_wave(wid)

    def _finish_wave(self, wid: str) -> None:
        """Every child of the wave is decided: one Director credit for the wave, then the
        next wave (_next_start) unless a child was accepted."""
        wave = self.waves.pop(wid)
        key, done = wave["arm"], wave["done"]
        entry = next(e for e in self.wave_log if e["wave"] == wid)
        pen = self.agents.repair.unmeasured_failure_penalty
        entry["results"] = {c.candidate_id: {
            "decision": f"{o.decision} / {o.failure_class}",
            "distance": distance(o.decision_record, o.result, pen)} for c, o in done}
        accepted = [c.candidate_id for c, o in done if o.decision == "accept"]
        if any(o.decision in DECIDED for _, o in done):
            self.director.update(self._current(key), bool(accepted))
        parent = _short_id(wave["parent"][0].candidate_id)
        if accepted:
            entry["next"] = f"accepted: {accepted}"
            self._emit(f"  repair of {parent}: accepted {', '.join(map(_short_id, accepted))}")
            return
        if not any(o.result is not None and o.result.status == "complete" for _, o in done):
            # every child timed out or failed on BRAIN: BRAIN may still finish them, and the
            # next run's start recovers them; a new wave now would only guess
            entry["next"] = "none: no child completed (timeouts are recovered next run)"
            self._emit(f"  repair of {parent}: waiting, no fix completed on BRAIN "
                       "(timeouts are recovered at the next run's start)")
            return
        start = self._next_start(wave["parent"],
                                 [(c, o.decision_record, o.result) for c, o in done])
        if start is None:
            entry["next"] = "none: the parent is no longer in repair"
            self._emit(f"  repair of {parent}: stopped, the parent is no longer in repair")
            return
        c, d, r, priority, why = start
        rep = self._start_wave(key, c, d, r, priority)
        entry["next"] = (f"{why}; wave {[x.candidate_id for x in rep.children]}"
                         if rep.children else f"{why}; no wave: {rep.notes[-1]}")
        base = (f"from {_short_id(c.candidate_id)}" if c is not wave["parent"][0]
                else "no fix got closer, same parent")
        if rep.children:
            self._emit(f"  repair of {parent}: next wave {base}, {len(rep.children)} fixes: "
                       f"{', '.join(x.lever for x in rep.children)}")
        else:
            self._emit(f"  repair of {parent}: {base}; stopped ({rep.notes[-1]})")

    def _recover_submissions(self) -> None:
        """Results BRAIN finished after an earlier run stopped waiting (our timeout, or a
        crash): one poll each, and a finished one is decided like any other result and
        credited to its arm. They spend no simulation budget (BRAIN already ran them) and
        come before resumed repairs, so a recovered near miss can be repaired in this run.
        One still running is tried again at the next run's start."""
        rows = self.memory.unfinished_submissions()
        if not rows:
            return
        models = arm_models(self.memory.candidate_outcomes(), self.memory.critic_reviews())
        still = 0
        for p in rows:
            c = self.memory.get_candidate(p["candidate_id"])
            if c is None:
                continue
            try:
                out = self.pipeline.recover(c, p["brain_sim_id"])
            except BrainLoginRequired:
                raise   # every later BRAIN call would fail the same way: stop now
            except Exception as exc:  # a lookup that goes wrong never stops the run
                self._emit(f"  recovering {c.candidate_id} failed: {exc}")
                out = None
            if out is None:
                still += 1
                continue
            key = self._arm_of(c, models.get(c.candidate_id))
            if key is not None and out.decision in DECIDED:
                self.director.update(key, out.decision == "accept")
            self.recovered.append({"candidate_id": c.candidate_id, "decision": out.decision,
                                   "failure_class": out.failure_class})
        if self.recovered or still:
            got = ", ".join(f"{i['candidate_id']} {i['decision']}/{i['failure_class']}"
                            for i in self.recovered) or "none"
            self._emit(f"--- recovered {len(self.recovered)} result(s) BRAIN finished after "
                       f"an earlier run stopped waiting: {got}"
                       + (f"; {still} not ready yet" if still else ""))

    def _prepare_seeds(self) -> None:
        """Turn the plan's seeds into fresh root candidates on BRAIN's current period (the
        arm's dates), queued to simulate before anything else once their region is in the
        run. A seed already simulated in an earlier run is skipped: its repairs resume from
        memory like any other."""
        base_pv = self.generator_config.prompt_version
        for spec in self.seed_specs:
            if spec.from_memory:
                orig = self.memory.get_candidate(spec.from_memory)
                if orig is None:
                    self.seed_notes.append({"seed": spec.from_memory, "status": "not in memory"})
                    continue
                name, new_id = spec.from_memory, f"{spec.from_memory}_seed"
            else:
                st = spec.settings
                digest = hashlib.sha256(f"{spec.expression}|{st}".encode()).hexdigest()[:8]
                name = new_id = f"seed_{digest}"
                orig = CandidateRecord(
                    candidate_id=new_id, expression=spec.expression, region=st.region,
                    universe=st.universe, delay=st.delay, decay=st.decay,
                    neutralization=st.neutralization, truncation=st.truncation,
                    start_date=st.start_date, end_date=st.end_date, dataset=st.dataset,
                    family=st.dataset)
            same = [k for k in self.arms if (k[1], k[4], k[2]) == (orig.region, orig.delay, base_pv)]
            key = next((k for k in same if (k[0], k[3]) == (orig.dataset, orig.model)), None) \
                or next((k for k in same if k[0] == orig.dataset), None) \
                or next(iter(same), None)
            if key is None:
                self.seed_notes.append({"seed": name, "status": f"no arm in {orig.region} "
                                        f"delay {orig.delay} in this plan"})
                continue
            if self.memory.get_candidate(new_id) is not None:
                self.seed_notes.append({"seed": name, "candidate_id": new_id,
                                        "status": "simulated in an earlier run"})
                continue
            arm = self.arms[key].settings
            cand = dataclasses.replace(orig, candidate_id=new_id, start_date=arm.start_date,
                                       end_date=arm.end_date, parent_id=None, generation=0,
                                       lever=None, prompt_version=key[2], model=key[3])
            if self.memory.has_result_for_hash(experiment_hash(cand)):
                self.seed_notes.append({"seed": name, "candidate_id": new_id,
                                        "status": "same experiment already simulated (same "
                                        "settings and dates as the arm's)"})
                continue
            fields = catalog_for(self._mtyped, self._typed, cand.region, cand.delay)
            a = analyze(cand.expression, self.spec, datafields=fields,
                        limits=self.generator_config.limits)
            if not a.ok:
                self.seed_notes.append({"seed": name, "candidate_id": new_id,
                                        "status": "invalid: " + "; ".join(a.errors)})
                continue
            self.seed_queue.append((key, cand))
            self.seed_notes.append({"seed": name, "candidate_id": new_id, "arm": arm_label(key),
                                    "status": "queued"})
        for n in self.seed_notes:   # a seed simulated in an earlier run is not news
            if n["status"] != "simulated in an earlier run":
                self._emit(f"--- seed {n['seed']}: {n['status']}")

    def _pool_expressions(self) -> list[str]:
        return [c.expression for a in self.pipeline.pool.alpha_ids()
                if (c := self.memory.get_candidate(a)) is not None]

    def run(self, rounds: int) -> dict:
        self._run(rounds)
        lessons = self._learn()
        report = self.report()   # after the Lesson Writer, so its model call is counted
        report["lessons"] = lessons
        return report

    def _learn(self) -> dict | None:
        """Lesson Writer after the run (lessons.py): evidence from all of memory, lessons
        written by the strategy model while the budget lasts, else by rules; stored for the
        next run's agents."""
        cfg = self.agents.lessons
        if not cfg.after_every_batch:
            return None
        llm = self.fallback(self.models.strategy) if cfg.use_model else None
        args = (self.memory.candidate_outcomes(), self.spec,
                lambda region, delay: self.catalog_of(region, delay)[1], cfg)
        try:
            out = learn(*args, llm=llm, context=self.ctx,
                        max_output=self.agents.max_output)
        except LLMUnavailable as exc:   # budget used up: the rules still write
            out = learn(*args)
            out.notes.append(f"no model call left ({exc}); rule lessons")
        if out.lessons:
            self.memory.put_lessons(self.run_id, out.lessons, out.source, out.evidence)
        self._emit(summary(out))
        return {"written": len(out.lessons), "source": out.source, "notes": out.notes,
                "lessons": out.lessons}

    def _run(self, rounds: int) -> dict:
        self._prepare_seeds()
        self._recover_submissions()
        if self.agents.repair.resume_from_memory:
            try:
                self._resume_repairs()
            except LLMUnavailable as exc:
                self.stop_reason = str(exc)
                return self.report()
        for r in range(1, rounds + 1):
            if self.sims_used >= self.budget.max_simulations:
                self.stop_reason = "simulation budget used up"
                break
            try:
                self._round(r)
            except LLMUnavailable as exc:  # budget used up, or the model failed
                self.stop_reason = str(exc)
            try:   # alphas already screened still go to BRAIN, whatever stopped the round
                self._flush()
            except LLMUnavailable as exc:
                self.stop_reason = self.stop_reason or str(exc)
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

    def _take_repairs(self, cap: int) -> dict[tuple, list[CandidateRecord]]:
        """Up to ``cap`` queued repairs as whole waves (at most ``repair.waves_per_round``),
        one arm at a time in turn, oldest first. A wave that does not fit what is left of
        ``cap`` waits for the next round, unless nothing was taken yet (then its first
        ``cap`` children go now and the rest later)."""
        taken: dict[tuple, list[CandidateRecord]] = {}
        waves, limit = 0, self.agents.repair.waves_per_round - self._round_waves
        progress = True
        while cap > 0 and waves < limit and progress:
            progress = False
            for key, queue in self.repair_queue.items():
                if not queue or cap <= 0 or waves >= limit or not self._in_run(key):
                    continue
                wid = self.wave_of.get(queue[0].candidate_id)
                size = 1 if wid is None else sum(
                    1 for c in queue if self.wave_of.get(c.candidate_id) == wid)
                if size > cap and taken:
                    continue
                group = [queue.popleft()]
                while (queue and wid is not None and len(group) < cap
                       and self.wave_of.get(queue[0].candidate_id) == wid):
                    group.append(queue.popleft())
                taken.setdefault(key, []).extend(group)
                cap -= len(group)
                waves += 1
                self._round_waves += 1
                progress = True
        return taken

    def _round(self, r: int) -> None:
        self._round_waves = 0
        slots = min(self.budget.per_round, self.budget.max_simulations - self._committed())
        entry = {"round": r, "arms": []}
        self.log.append(entry)
        for key, cand in [(k, c) for k, c in self.seed_queue if self._in_run(k)]:
            if slots <= 0 or self._committed() >= self.budget.max_simulations:
                break
            self.seed_queue.remove((key, cand))
            arm_log = {"arm": arm_label(key), "slots": 1, "seed": cand.candidate_id,
                       "outcomes": []}
            entry["arms"].append(arm_log)
            self._simulate(key, cand, arm_log, credit=False)
            slots -= 1
        if self._committed() >= self.budget.max_simulations:
            self.stop_reason = "simulation budget used up"
            return
        # queued repairs first: a near-miss alpha is the cheapest chance of an accept, and an
        # arm the Director gives no slot would otherwise never run its repairs
        cap = math.ceil(slots * self.agents.repair.max_share_per_round)
        first = self._take_repairs(min(cap, slots))
        for key, cands in first.items():
            arm_log = {"arm": arm_label(key), "slots": len(cands), "repairs_first": True}
            entry["arms"].append(arm_log)
            self._screen_and_simulate(key, cands, {}, arm_log)
            if self._committed() >= self.budget.max_simulations:
                self.stop_reason = "simulation budget used up"
                return
        slots -= sum(len(c) for c in first.values())
        live = self._live_arms()
        if not live:
            self.stop_reason = "every generator model is down: " + "; ".join(
                f"{m}: {why}" for m, why in self.counter.down.items())
            return
        alloc = self._allocate(slots, live) if slots > 0 else {}
        entry["allocation"] = {arm_label(k): n for k, n in alloc.items()}
        repairs = sum(len(c) for c in first.values())
        self._emit(f"--- round {r}: {f'{repairs} repair(s), then ' if repairs else ''}"
                   f"{slots} new: {_slots_text(alloc) or 'none'}")
        lost, emptied = self._run_allocation(r, alloc, entry, "")
        passes = 0
        while not self.stop_reason:
            spare = lost
            if emptied and passes < self.agents.refill_passes:
                spare += emptied
                passes += 1
            spare = min(spare, self.budget.max_simulations - self._committed())
            live = self._live_arms()
            if spare <= 0 or not live:
                return
            alloc = self._allocate(spare, live)
            entry.setdefault("refills", []).append(
                {arm_label(k): n for k, n in alloc.items() if n})
            self._emit(f"--- round {r}: refill {spare}: {_slots_text(alloc)}")
            lost, emptied = self._run_allocation(r, alloc, entry, f"f{len(entry['refills'])}")

    def _allocate(self, slots: int, live: list[tuple]) -> dict:
        a = self.agents
        if self.run_regions is None:
            return self.director.allocate(slots, a.floor_share, a.max_arm_share, only=live,
                                          region_floor_share=a.region_floor_share,
                                          max_region_share=a.max_region_share)
        # rotation: slots to the run's regions with the fewest simulations so far (memory
        # plus this run), then the Director picks arms within each region
        regions = [r for r in self.run_regions if any(k[1] == r for k in live)]
        per_region = balance_slots(slots, {r: self.region_sims[r] for r in regions})
        alloc = {k: 0 for k in self.director.arms}
        for r, n in per_region.items():
            if n:
                got = self.director.allocate(n, a.floor_share, a.max_arm_share,
                                             only=[k for k in live if k[1] == r])
                for k, m in got.items():
                    alloc[k] += m
        return alloc

    def _live_arms(self) -> list[tuple]:
        return [k for k in self.arms if k[3] not in self.counter.down and self._in_run(k)
                and k not in self.successor]

    def _current(self, key: tuple) -> tuple:
        """The arm that now stands for ``key``: itself, or its latest evolved successor."""
        while key in self.successor:
            key = self.successor[key]
        return key

    def _run_allocation(self, r: int, alloc: dict, entry: dict, tag: str) -> tuple[int, int]:
        """Run each arm's slots. Returns (slots lost to a down generator model, slots
        emptied by the generator, Pre-Filter or Critic)."""
        lost = emptied = 0
        for i, (key, n) in enumerate(alloc.items()):
            if n == 0:
                continue
            if key[3] in self.counter.down:  # went down earlier in this pass
                lost += n
                continue
            arm_log = {"arm": arm_label(key), "slots": n}
            if tag:
                arm_log["refill"] = True
            entry["arms"].append(arm_log)
            d, e = self._arm(r, i, key, n, arm_log, tag)
            lost, emptied = lost + d, emptied + e
            if self._committed() >= self.budget.max_simulations:
                self.stop_reason = "simulation budget used up"
                break
        return lost, emptied

    def _arm(self, r: int, i: int, key: tuple, n: int, log: dict,
             tag: str = "") -> tuple[int, int]:
        """Fill ``n`` slots of one arm. Returns (slots lost to a down generator model,
        slots emptied by the generator, Pre-Filter or Critic)."""
        lost = 0
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
                res = self._generator(key).generate(
                    source, need, batch_id=f"{self.run_id}r{r:02d}a{i}{tag}")
            except GenerationError as exc:
                log["generation_error"] = str(exc)
                res = None
            except ModelDown as exc:  # the arm gets no more slots; the run goes on
                log["generation_error"] = str(exc)
                self._emit(f"{exc}; its arms get no more slots, these go to live arms")
                res = None
                lost = need
            if res is not None:
                batch += res.candidates
                rationales.update(res.rationales)
                log["generated"] = len(res.candidates)
                log["generator_rejected"] = len(res.rejected)
        simulated = self._screen_and_simulate(key, batch, rationales, log)
        return lost, max(0, n - lost - simulated)

    def _screen_and_simulate(self, key: tuple, batch: list[CandidateRecord],
                             rationales: dict[str, str], log: dict) -> int:
        """Pre-Filter, Critic and judge, then simulate. Returns how many were simulated."""
        go = self._screen(key, batch, rationales, log)
        simulated = 0
        for i, cand in enumerate(go):
            if self._committed() >= self.budget.max_simulations:
                for c in go[i:]:
                    self._wave_done(c.candidate_id)
                break
            self._simulate(key, cand, log)
            simulated += 1
        return simulated

    def _screen(self, key: tuple, batch: list[CandidateRecord], rationales: dict[str, str],
                log: dict) -> list[CandidateRecord]:
        """Pre-Filter, Critic and judge: the alphas that may go to BRAIN. Wave children
        stopped here close their place in the wave."""
        pf = self.prefilter.run(batch)
        log["prefilter_dropped"] = [[c.candidate_id, why] for c, why in pf.dropped]
        primary, shadows = self._critics_for(key)
        log["critic"] = primary
        rv = review_and_debate(pf.kept, rationales, self.critics[primary], self.judge)
        self._record_reviews(key, primary, "primary", rv.reviews)
        fresh = [c for c in pf.kept if c.lever is None]
        for cm in shadows if fresh else []:  # recorded for compare.py; never changes what is simulated
            shadow = ReviewOutcome()
            reviews = self.critics[cm].review(fresh, rationales, shadow)
            self._record_reviews(key, cm, "shadow", reviews)
            log.setdefault("shadow_verdicts", {})[cm] = {cid: r.verdict
                                                         for cid, r in reviews.items()}
        log["critic_dropped"] = [[c.candidate_id, why] for c, why in rv.dropped]
        log["revised"] = [c.candidate_id for c in rv.to_simulate if c.lever == "critic_revision"]
        log["outcomes"] = []
        going = {c.candidate_id for c in rv.to_simulate}
        for c in batch:   # wave children the Pre-Filter or Critic stopped
            if c.candidate_id in self.wave_of and c.candidate_id not in going:
                self._wave_done(c.candidate_id)
        return list(rv.to_simulate)

    def _record_reviews(self, key: tuple, critic_model: str | None, role: str,
                        reviews: dict) -> None:
        for cid, r in reviews.items():
            self.memory.put_critic_review(
                run_id=self.run_id, candidate_id=cid, critic_model=critic_model, role=role,
                verdict=r.verdict, source=r.source, generator_model=key[3],
                payload={"issues": r.issues, "blocking": r.blocking,
                         "revised": r.revised.expression if r.revised else None})

    def _committed(self) -> int:
        """Simulations used plus alphas queued for or running on BRAIN: what the budget
        checks count."""
        return self.sims_used + len(self.sim_queue) + self._in_flight

    def _simulate(self, key: tuple, cand: CandidateRecord, log: dict, *,
                  credit: bool = True) -> None:
        """Queue an alpha for BRAIN; the queue goes out once it holds
        ``simulator.max_concurrent`` alphas, and at the end of each round. ``credit=False``
        (seeds): the outcome does not update the Director's view of the arm, since the
        alpha is not one of the arm's ideas."""
        self.sim_queue.append((key, cand, log, credit))
        if len(self.sim_queue) >= self.max_concurrent:
            self._flush()

    def _flush(self) -> None:
        """Simulate the queued alphas together and decide each the moment its result is in
        (owner 2026-11-23: a slow alpha must not hold up the others): Director credit,
        repair wave. When a slot frees, ready repair waves are screened and added to the
        running batch (``_top_up``), within the simulation budget. A model failure in one
        repair does not stop the others' bookkeeping; the first is raised at the end."""
        queued, self.sim_queue = self.sim_queue, []
        if not queued:
            return
        if len(queued) > 1:
            self._emit(f"--- BRAIN: simulating {len(queued)} alphas together")
        meta = {c.candidate_id: (key, c, log, credit) for key, c, log, credit in queued}
        failures: list[Exception] = []
        self._in_flight = len(queued)

        def on_outcome(cand: CandidateRecord, outcome) -> None:
            key, c, log, credit = meta.pop(cand.candidate_id)
            self._in_flight -= 1
            try:
                self._after_simulation(key, c, log, credit, outcome)
            except LLMUnavailable as exc:
                failures.append(exc)

        def more(free: int) -> list[CandidateRecord]:
            if failures or self.stop_reason:
                return []
            try:
                added = self._top_up(free)
            except LLMUnavailable as exc:
                failures.append(exc)
                return []
            for key, c, log in added:
                meta[c.candidate_id] = (key, c, log, True)
            self._in_flight += len(added)
            return [c for _, c, _ in added]

        try:
            self.pipeline.process_many([c for _, c, _, _ in queued], on_outcome=on_outcome,
                                       more=more)
        finally:
            self._in_flight = 0
        if failures:
            raise failures[0]

    def _top_up(self, free: int) -> list[tuple[tuple, CandidateRecord, dict]]:
        """Ready repair waves for ``free`` freed BRAIN slots (whole waves when they fit),
        screened like any repair, within the simulation budget: (arm, alpha, arm log)."""
        cap = min(free, self.budget.max_simulations - self._committed())
        if cap <= 0 or not self.agents.repair.fill_freed_slots:
            return []
        out = []
        for key, cands in self._take_repairs(cap).items():
            arm_log = {"arm": arm_label(key), "slots": len(cands), "repairs_mid_batch": True}
            if self.log:
                self.log[-1]["arms"].append(arm_log)
            go = self._screen(key, cands, {}, arm_log)
            room = cap - len(out)
            for c in go[room:]:            # over budget: their place in the wave is closed
                self._wave_done(c.candidate_id)
            out += [(key, c, arm_log) for c in go[:room]]
        if out:
            self._emit(f"--- BRAIN: {len(out)} repair fix(es) join the running batch: "
                       f"{', '.join(_short_id(c.candidate_id) for _, c, _ in out)}")
        return out

    def _after_simulation(self, key: tuple, cand: CandidateRecord, log: dict, credit: bool,
                          outcome) -> None:
        if outcome.result is not None and outcome.result.status != "invalid":
            self.sims_used += 1
            self.region_sims[cand.region] = self.region_sims.get(cand.region, 0) + 1
        item = {"candidate_id": cand.candidate_id, "decision": outcome.decision,
                "failure_class": outcome.failure_class}
        log["outcomes"].append(item)
        rec = outcome.decision_record
        if rec is not None and any(r.startswith(power_pool.PREFIX) for r in rec.reasons):
            item["power_pool_candidate"] = True
            self.power_pool.append({"candidate_id": cand.candidate_id,
                                    "brain_alpha_id": power_pool.brain_alpha_id(outcome.result),
                                    "region": cand.region, "sharpe": outcome.result.sharpe,
                                    "fitness": outcome.result.fitness})
        seed = next((n for n in self.seed_notes if n.get("candidate_id") == cand.candidate_id),
                    None)
        if seed is not None:
            seed["status"] = f"{outcome.decision} / {outcome.failure_class}"
        flag = next((r for r in (rec.reasons if rec else [])
                     if r.startswith("recorded, not applied") and "overfitting" in r), None)
        if flag is not None:   # owner 2026-11-23: overfitting checks raise a flag only
            item["overfitting_flag"] = flag
            self.overfitting_flags.append({"candidate_id": cand.candidate_id, "flag": flag})
            self._emit(f"  flag: {cand.candidate_id}: {flag.split(': ', 1)[-1]}")
        in_wave = cand.candidate_id in self.wave_of
        if outcome.decision in DECIDED and credit and not in_wave:  # a wave: one credit at its end
            self.director.update(self._current(key), outcome.decision == "accept")
        if in_wave:   # the wave decides where repair goes next, once all its children are in
            self._wave_done(cand.candidate_id, cand, outcome)
        elif outcome.decision == "repair" and outcome.decision_record is not None:
            rep = self._start_wave(key, cand, outcome.decision_record, outcome.result)
            item["repair"] = rep.notes
            if rep.children:
                item["repair_wave"] = [c.candidate_id for c in rep.children]

    def _market_datasets(self) -> dict[tuple, set[str]]:
        """{(region, delay): datasets in that market's catalog} for every market with an arm."""
        out = {}
        for key in self.arms:
            market = (key[1], key[4])
            if market not in out:
                _, meta = self.catalog_of(*market)
                out[market] = {m["dataset"] for m in meta.values() if m.get("dataset")}
        return out

    def _evolve(self, r: int) -> None:
        market_ds = self._market_datasets()
        catalog_datasets = set().union(*market_ds.values())
        known = sorted({k[0] for k in self.arms} | catalog_datasets)
        evo = self.strategist.evolve(self.memory.candidate_outcomes(),
                                     self.generator_config.prompt_version, known_datasets=known)
        if evo is None:
            return
        added, replaced = [], []
        # only the live arms of this run's regions evolve: the others get no slots in this
        # run, and each evolved arm replaces its parent (one arm per idea, not two)
        base = [(k, p) for k, p in self.arms.items()
                if k not in self.successor and self._in_run(k)]
        arm_datasets = {k[0] for k in self.arms}
        # a hypothesis on a catalog dataset no arm covers gets arms of its own, so the
        # Director credits that dataset, not the arm it ran under (run rs20261104163055 ran
        # news12 ideas as analyst4 and model51); one on an unknown dataset goes to every arm
        new_ds: dict[str, list[str]] = {}
        for ds, text in evo.hypotheses:
            if ds is not None and ds not in arm_datasets and ds in catalog_datasets:
                new_ds.setdefault(ds, []).append(text)
        for key, plan in base:
            new_key = (key[0], key[1], evo.prompt_version, key[3], key[4])
            sources = deque(plan.sources)
            for ds, text in evo.hypotheses:
                if ds is None or ds == key[0] or (ds not in arm_datasets and ds not in new_ds):
                    sources.appendleft(Source("hypothesis", text))
            if self._add_arm(new_key, ArmPlan(plan.dataset, plan.settings, sources,
                                              evo.guidance_for(plan.settings.region),
                                              catalog=plan.catalog), added, like=key):
                self.successor[key] = new_key
                replaced.append(arm_label(key))
        # new datasets: at most strategy.max_new_arms arms, first one per (dataset, market)
        # with the generator models taken in turn, then further models; only markets of
        # this run whose catalog has the dataset, with that market's settings
        templates = {(k[1], k[4]): p for k, p in base}
        markets = [(ds, region, delay, plan) for ds in new_ds
                   for (region, delay), plan in templates.items()
                   if ds in market_ds.get((region, delay), ())]
        models = [m for m in self.models.generator if m not in self.counter.down]
        picks = [(*mk, models[(i + p) % len(models)]) for p in range(len(models))
                 for i, mk in enumerate(markets)] if models else []
        for ds, region, delay, plan, model in picks[:self.agents.strategy.max_new_arms]:
            settings = dataclasses.replace(plan.settings, dataset=ds)
            sources = deque(Source("hypothesis", t) for t in new_ds[ds])
            self._add_arm((ds, region, evo.prompt_version, model, delay),
                          ArmPlan(ds, settings, sources, evo.guidance_for(region),
                                  catalog=plan.catalog), added)
        self.log[-1]["evolution"] = {"prompt_version": evo.prompt_version, "source": evo.source,
                                     "guidance": evo.guidance, "hypotheses": evo.hypotheses,
                                     "region_guidance": evo.region_guidance,
                                     "new_arms": added, "replaced": replaced,
                                     "notes": evo.notes}
        self._emit(f"--- strategy evolution: {evo.prompt_version} ({evo.source}), "
                   f"{len(replaced)} arms updated, {len(added) - len(replaced)} new-dataset arms")

    def _add_arm(self, key: tuple, plan: ArmPlan, added: list[str],
                 like: tuple | None = None) -> bool:
        """Add an arm (``like``: start from that arm's record). False if it exists."""
        if key in self.arms:
            return False
        self.arms[key] = plan
        self.repair_queue[key] = deque()
        if self.director.add_arm(key, like=like):
            added.append(arm_label(key))
        return True

    # ------------------------------------------------------------------ report

    def report(self) -> dict:
        arms = {}
        for key in self.director.arms:
            s = self.director.state[key]
            arms[arm_label(key)] = {"accepted": s.successes, "not_accepted": s.failures,
                                             "posterior_pass_rate": round(self.director.pass_rate(key), 4),
                                             "repairs_queued": len(self.repair_queue[key])}
        return {
            "run_id": self.run_id,
            "stop_reason": self.stop_reason,
            "power_pool_candidates": self.power_pool,
            "simulations_used": self.sims_used,
            "max_concurrent": self.max_concurrent,
            "model_calls": self.counter.calls,
            "model_calls_by_model": dict(self.counter.by_model),
            "models_down": self.down(),
            "models": {"generator": list(self.models.generator),
                       "critic": [c for c in self.critics if c] if self.agents.critic_model else [],
                       "shadow_critics": self.models.shadow_critics,
                       "prefilter": self.models.prefilter, "debate": self.models.debate,
                       "repair": self.models.repair, "strategy": self.models.strategy},
            "prompt_versions": {"generator": self.generator_config.prompt_version,
                                **PROMPT_VERSIONS},
            "warm_start_outcomes": self.warm_started,
            "pool_size": self.pipeline.pool.count(),
            "regions_this_run": self.run_regions,   # None = every region (no rotation)
            "simulations_by_region": dict(self.region_sims),   # since region_balance_since
            "arms": arms,
            "recovered_results": self.recovered,
            "resumed_repairs": self.resumed,
            "repair_waves": self.wave_log,
            "overfitting_flags": self.overfitting_flags,
            "seeds": self.seed_notes,
            **_lever_tables(LeverStats.from_events(
                repair_events(self.memory.candidate_outcomes(), self.agents.repair),
                self.agents.repair.prior, self.agents.repair.region_prior_strength)),
            "comparison": model_comparison(self.memory.candidate_outcomes(),
                                           self.memory.critic_reviews()),
            "rounds": self.log,
        }


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return path
