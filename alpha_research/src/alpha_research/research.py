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
import logging
import math
import re
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, TextIO

import yaml

from . import arm_designer
from .arm_designer import DesignerConfig
from .blend_planner import BLEND_PLAN, BlendConfig, BlendPlanner, blend_base, stability_flag
from .breadth import BreadthConfig
from .brain_client import BrainClient, BrainLoginRequired, SimulationQuotaExhausted
from .config import Config, ConfigError
from .critic import (Critic, DebateJudge, ReviewOutcome, SparseFieldRule, nan_guarded,
                     review_and_debate)
from .compare import arm_models, model_comparison
from .fastexpr import FastExprSpec, analyze, catalog, catalog_for
from .generator import (AlphaGenerator, GenerationError, GenerationSettings, GeneratorConfig,
                        Source, dataset_appeal, dataset_catalog_for, dataset_category,
                        dataset_facts, field_role, load_dataset_meta)
from .llm import (CallFailed, EmptyReply, LLMClient, LLMUnavailable, ModelDown, RateLimited,
                  RequestTooLarge, provider_of)
from .memory import MemoryStore
from .orchestrator import Orchestrator, balance_slots, consolidate
from . import power_pool
from .pipeline import Pipeline
from .research_loop import assign_lineage
from .prompts import PROMPT_VERSIONS, bar_text, brain_context
from .prefilter import PreFilter, PrefilterConfig
from .repair import Playbook
from .lessons import LessonsConfig, for_agent, learn, summary
from .lever_stats import LeverStats, RepairLearningConfig, distance, repair_events
from .quality import generation_quality
from .repair_agent import RepairAgent
from .schemas import (GROUP_NEUTRALIZATIONS, NO_REGULAR_SIMULATION, REGION_SETTINGS,
                      CandidateRecord, SchemaError, experiment_hash)
from .simulator import EMPTY_ALPHA
from .strategy import (StrategyConfig, StrategyEvolution, brain_limits,
                       prod_correlation_fields, region_notes, signal_evidence, arm_history,
                       arm_failures)

DECIDED = ("accept", "reject", "repair")
FILL_LEVER = "fill_missing_values"   # an empty alpha tried once more (_retry_empty)
# fewest catalog fields a generator prompt lists after too-large refusals (each halves the
# list, from generator.yaml max_datafields_in_prompt); below this the hypothesis has too few
# fields to choose from, so the model writes no alphas for the rest of the run instead
MIN_FIELDS_WHEN_TOO_LARGE = 20


class LLMBudgetExhausted(LLMUnavailable):
    pass


log = logging.getLogger(__name__)

EMPTY_REPLIES_DOWN = 3       # empty replies in a row (after the cap retry) before a model is down
PAUSE_MAX_SECONDS = 1800.0   # a 429 asking for at most this long pauses the model, not downs it


class CallCounter:
    """One model-call budget shared by every agent and every model."""

    def __init__(self, max_calls: int | None, *, clock: Callable[[], float] = time.monotonic,
                 pause_max_seconds: float = PAUSE_MAX_SECONDS):
        self.max_calls = max_calls
        self.calls = 0
        self.by_model: dict[str, int] = {}
        self.down: dict[str, str] = {}   # model -> why it is out for the rest of the run
        self.too_large: dict[str, int] = {}   # model -> calls refused as too large
        self.empty: dict[str, int] = {}       # model -> empty replies (call failed, model up)
        self.empty_streak: dict[str, int] = {}
        self.paused: dict[str, float] = {}    # model -> clock time its pause ends
        self.pauses: dict[str, int] = {}      # model -> pauses so far
        self.stopped: str | None = None   # set by Ctrl-C: no new model calls (lessons aside)
        self.clock = clock
        self.pause_max_seconds = pause_max_seconds

    def pause_left(self, model: str) -> float:
        """Seconds until a paused model may be called again (0 = not paused)."""
        until = self.paused.get(model)
        return max(0.0, until - self.clock()) if until is not None else 0.0

    def is_up(self, model: str) -> bool:
        """Neither down for the run nor paused right now."""
        return model not in self.down and self.pause_left(model) <= 0


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
        if (left := c.pause_left(self.model)) > 0:
            raise RateLimited(self.model, f"paused for {left:.0f}s more after a rate limit",
                              left)
        if c.stopped:
            raise LLMBudgetExhausted(c.stopped)
        if c.max_calls is not None and c.calls >= c.max_calls:
            raise LLMBudgetExhausted(f"model call budget of {c.max_calls} used up")
        c.calls += 1
        c.by_model[self.model] = c.by_model.get(self.model, 0) + 1
        try:
            reply = self.inner.complete(system, user, max_output=max_output)
        except RequestTooLarge:   # the request, not the model: it stays up
            c.too_large[self.model] = c.too_large.get(self.model, 0) + 1
            raise
        except EmptyReply as exc:
            # one empty reply is this call's failure; several in a row mean the model
            # cannot answer at its cap (run rs20261127153052 lost two models to one each)
            c.empty[self.model] = c.empty.get(self.model, 0) + 1
            streak = c.empty_streak[self.model] = c.empty_streak.get(self.model, 0) + 1
            if streak >= EMPTY_REPLIES_DOWN:
                c.down[self.model] = f"{streak} empty replies in a row: {exc.reason}"
                raise ModelDown(self.model, c.down[self.model]) from exc
            raise
        except RateLimited as exc:
            # a stated wait up to pause_max_seconds pauses the model (Groq's daily limit
            # asked for 13 minutes in run rs20261127153052); a longer one (a daily quota)
            # takes it out for the run
            if exc.wait <= c.pause_max_seconds:
                c.paused[self.model] = c.clock() + exc.wait + 1
                c.pauses[self.model] = c.pauses.get(self.model, 0) + 1
                log.warning("%s paused for %.0fs: %s", self.model, exc.wait + 1, exc.reason)
                raise
            c.down[self.model] = exc.reason
            raise ModelDown(self.model, exc.reason) from exc
        except ModelDown as exc:
            c.down[self.model] = exc.reason
            raise
        except LLMUnavailable as exc:
            c.down[self.model] = str(exc)
            raise ModelDown(self.model, str(exc)) from exc
        c.empty_streak[self.model] = 0
        return reply


class FallbackLLM:
    """An agent's model with fallbacks: calls the first model that is not down, in order.
    Raises ``ModelDown`` only when every model in the list is down."""

    def __init__(self, models: list[BudgetedLLM]):
        self.models = models

    def _live(self) -> list[BudgetedLLM]:
        return [m for m in self.models if m.counter.is_up(m.model)]

    @property
    def model(self) -> str:
        live = self._live()
        return (live or self.models)[0].model

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        tried, big, failed = 0, 0, 0
        for m in self._live():
            tried += 1
            try:
                return m.complete(system, user, max_output=max_output)
            except RequestTooLarge:   # try a model with a larger limit; this one stays up
                big += 1
                continue
            except CallFailed:   # empty reply or a pause: this model stays up
                failed += 1
                continue
            except ModelDown:
                continue
        names = ", ".join(m.model for m in self.models)
        if tried and big == tried:
            raise RequestTooLarge(names, "too large for every model of this agent")
        if failed or big:
            raise CallFailed(names, "this call failed on every model of this agent "
                                    "that is up")
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
    min_batch: int | None = None   # an arm given slots gets at least this many (orchestrator.consolidate)
    stop_after_idle_rounds: int | None = None   # stop after this many rounds in a row with no simulation
    fresh_fill_freed_slots: bool = True   # a BRAIN slot freed mid-batch takes the round's new alphas
    # loop guard: stop when no alpha went to or came back from BRAIN for this many minutes,
    # or this many model calls were spent with no alpha going to BRAIN (None = off)
    stall_minutes: float | None = None
    stall_model_calls: int | None = None
    designer: DesignerConfig = field(default_factory=DesignerConfig)   # arm_designer.py
    blend: BlendConfig = field(default_factory=BlendConfig)   # blend_planner.py
    breadth: BreadthConfig = field(default_factory=BreadthConfig)   # breadth.py
    # alphas pending on BRAIN's checks are fetched again every this many rounds during a
    # run (0 = only at the start); one decided repair starts its wave at once
    recheck_every_rounds: int = 0
    # each market's arms whose dataset gave us best |Sharpe| (raw or risk-neutralized) >=
    # evidence_sharpe get at least this share of the market's new-alpha slots (None = off)
    evidence_floor_share: float | None = None
    evidence_sharpe: float = 1.0
    # a new arm (designed or new-dataset) with no record of its own starts from the
    # market's average record ("market") or a blank one ("blank")
    new_arm_prior: str = "blank"
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


def _choice(sec: dict, key: str, allowed: tuple[str, ...]) -> str:
    v = sec.get(key, allowed[0])
    if v not in allowed:
        raise ConfigError(f"agents config: director.{key} must be one of {list(allowed)}")
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
    keys = {"calls_per_minute", "tokens_per_minute", "output_tokens_per_minute"}
    if not isinstance(pace, dict) or not all(
            isinstance(m, str) and isinstance(v, dict) and v and set(v) <= keys
            and all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in v.values())
            for m, v in pace.items()):
        raise ConfigError("agents config: models.pace must map model ids to "
                          "{calls_per_minute, tokens_per_minute, output_tokens_per_minute} "
                          "positive integers")
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
                          if "max_new_arms" in st else StrategyConfig.max_new_arms),
            max_prompt_chars=(_pos_int(st, "strategy", "max_prompt_chars", lo=1000)
                              if st.get("max_prompt_chars") is not None else None)),
        floor_share=float(floor),
        max_output=_pos_int(raw, "agents", "max_output", lo=256),
        models=models_config_from_dict(raw.get("models")),
        repair=RepairLearningConfig.from_dict(secs["repair"]),
        max_arm_share=_opt_num(secs["director"], "director", "max_arm_share", 0, 1),
        region_floor_share=_opt_num(secs["director"], "director", "region_floor_share", 0, 1),
        evidence_floor_share=_opt_num(secs["director"], "director", "evidence_floor_share", 0, 1),
        evidence_sharpe=(_opt_num(secs["director"], "director", "evidence_sharpe", 0, 10)
                         if secs["director"].get("evidence_sharpe") is not None else 1.0),
        new_arm_prior=_choice(secs["director"], "new_arm_prior", ("blank", "market")),
        max_region_share=_opt_num(secs["director"], "director", "max_region_share", 0, 1),
        refill_passes=(_pos_int(secs["director"], "director", "refill_passes", lo=0)
                       if "refill_passes" in secs["director"] else 0),
        min_batch=(None if secs["director"].get("min_batch") is None
                   else _pos_int(secs["director"], "director", "min_batch")),
        stop_after_idle_rounds=(None if secs["director"].get("stop_after_idle_rounds") is None
                                else _pos_int(secs["director"], "director",
                                              "stop_after_idle_rounds")),
        fresh_fill_freed_slots=(_bool(secs["director"], "director", "fill_freed_slots")
                                if "fill_freed_slots" in secs["director"] else True),
        stall_minutes=_opt_num(secs["director"], "director", "stall_minutes", 1, 10000),
        stall_model_calls=(None if secs["director"].get("stall_model_calls") is None
                           else _pos_int(secs["director"], "director", "stall_model_calls")),
        regions_per_run=(None if secs["director"].get("regions_per_run") is None
                         else _pos_int(secs["director"], "director", "regions_per_run")),
        region_balance_since=_opt_date(secs["director"], "director", "region_balance_since"),
        sparse=_sparse_rule(secs["critic"]),
        designer=DesignerConfig.from_dict(raw.get("designer")),
        blend=BlendConfig.from_dict(raw.get("blend_plan")),
        breadth=BreadthConfig.from_dict(raw.get("breadth_blend")),
        recheck_every_rounds=(_pos_int(secs["repair"], "repair", "recheck_every_rounds", lo=0)
                              if "recheck_every_rounds" in secs["repair"] else 0),
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
        self._clock = clock
        self._progress_at: float | None = None   # last alpha to or from BRAIN (loop guard)
        self._progress_calls = 0                  # model calls at that moment
        agents.check_models()
        self.models = agents.models
        self.counter = CallCounter(budget.max_llm_calls, clock=clock)
        self._sleep = sleep
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
        self.client = client   # BRAIN (quota_stop / quota when it is the real client)
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
        base = spec.base_fields
        sparse = ({None: rule.fields(self.field_meta, base),
                   **{k: rule.fields(meta, base) for k, (_, meta) in self.markets.items()}}
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
        self.repairer.flip_restricted = config.repair.flip_short_restricted
        if agents.breadth.enabled:
            self.repairer.breadth = agents.breadth
        if agents.blend.enabled and self.repairer.llm is not None:
            # the reviewer: critic models from another maker than the planner (repair model)
            planner_maker = provider_of(m.repair)
            others = [cm for cm in m.critic if provider_of(cm) != planner_maker]
            self.repairer.blend = BlendPlanner(
                spec, agents.blend, llm=self.repairer.llm,
                reviewer=FallbackLLM([self.llm(cm) for cm in others]) if others else None,
                context=ctx, max_output=mo, limits=limits)
            self.repairer.pool_pnl = self._pool_pnl()
        self.blend_flags: list[dict] = []
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
        # Arm Designer (arm_designer.py): plan arms its designed arms replace in this run's
        # markets (no new slots; their queued repairs still run), and the run's designs
        self.inactive: set[tuple] = set()
        # datasets dead in a market (arm_designer.dead_datasets): {(region, delay): {ds: (n,
        # best |Sharpe|)}}; their arms get no slots
        self.dead: dict[tuple, dict] = {}
        # operators no simulated alpha of ours has used: the Generator's spotlight rotates
        # through them (generator.yaml operator_spotlight)
        used_ops: set[str] = set()
        for o in memory.candidate_outcomes():
            if o["result"] is not None and o["result"].status != "invalid":
                used_ops |= set(analyze(o["candidate"].expression, spec).operators_used)
        self.unused_ops = sorted(set(spec.operators) - used_ops)
        self.designed: list[dict] = []
        self.sims_used = 0
        # alphas screened and waiting for BRAIN: sent simulator.max_concurrent at a time
        # (OQ-3), then decided in order (_flush)
        self.sim_queue: list[tuple[tuple, CandidateRecord, dict, bool]] = []
        self._in_flight = 0   # alphas of the current batch not decided yet (budget counts them)
        # empty alphas tried once more with missing vector values filled (_retry_empty);
        # they join the running batch first, else the next one
        self.fill_queue: list[tuple[tuple, CandidateRecord, dict]] = []
        self.empty_retries: list[dict] = []
        self._round_waves = 0  # repair waves started this round (repair.waves_per_round)
        # the round's new-alpha slots not generated yet, {arm: slots} in allocation order;
        # _run_fresh works through them, and a BRAIN slot freed mid-batch takes the next
        # arm's share early (_fresh_for_freed)
        self._fresh_left: dict[tuple, int] = {}
        # generator models whose prompt was too large: fields listed from now on, and the
        # ones no field count fits (they write no alphas this run; _shrink_fields)
        self.field_cap: dict[str, int] = {}
        self.no_fit: dict[str, str] = {}
        self._fresh_ctx: tuple = (0, {}, "", {})   # (round, round log, batch tag, arm index)
        self._fresh_lost = 0       # slots of fresh_left lost to a down model / emptied by
        self._fresh_emptied = 0    # drops while taken mid-batch (go to the round's refill)
        self.max_concurrent = max(1, config.simulator.max_concurrent)
        self.log: list[dict] = []
        self.stop_reason: str | None = None
        self.stop_requested = False        # first Ctrl-C: finish what is on BRAIN, then stop
        self.interrupts = 0                # Ctrl-C presses during run(catch_interrupt=True)
        self.live = None                   # dashboard: live.LiveStatus (attach_live)
        self._new_arms: list[tuple] = []   # arms added during the run, to credit (_seed_new_arms)
        self._evidence = None              # (round, {(dataset, region, delay): best |Sharpe|})
        self.current_round = 0
        self.max_rounds = None
        self.dropped_on_stop: list[str] = []   # screened alphas not sent to BRAIN after Ctrl-C
        self.resumed: list[dict] = []
        self.power_pool: list[dict] = []   # Power Pool candidates this run (recorded only)
        self.recovered: list[dict] = []
        self.cancelled: list[dict] = []    # stuck BRAIN simulations cancelled at run start
        self._resume_rows: list[tuple] = []
        self.reevaluated: list[dict] = []
        self.rechecked: list[dict] = []
        # repair waves (owner 2026-11-23): fixes_per_wave children of one parent simulated
        # together; when all are decided the next wave starts from the best child if it got
        # closer to accept, else from the same parent with its next levers
        self.waves: dict[str, dict] = {}       # wave id -> {arm, parent, children, pending, done}
        self.wave_of: dict[str, str] = {}      # child id -> wave id, until it is decided
        self.wave_log: list[dict] = []
        self.overfitting_flags: list[dict] = []   # deflated-Sharpe misses, recorded only
        self.warm_started = self._warm_start()
        self.region_sims, self.run_regions = self._pick_regions()
        # arms outside the rotation that may run repairs of a close lead (no new alphas)
        self.lead_arms: set[tuple] = set()
        self.cross_region: list[dict] = []

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

    def token_stats(self) -> dict[str, dict]:
        """{model: RequestFitClient.stats()} for the models whose client measures tokens."""
        out = {}
        for model, client in self._llms.items():
            c = client
            while c is not None and not hasattr(c, "stats"):
                c = getattr(c, "inner", None)
            if c is not None:
                out[model] = c.stats()
        return out

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
                    if provider_of(c) != g and self.counter.is_up(c)]
        if self.models.critic_pick == "balanced" and eligible:
            primary = min(eligible, key=lambda c: (self.counter.by_model.get(c, 0),
                                                   eligible.index(c)))
        else:
            primary = eligible[0] if eligible else None
        shadows = ([cm for cm in self.critics
                    if cm not in (primary, None) and self.counter.is_up(cm)]
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

    def _warm_start(self, only: set | None = None, outcomes: list[dict] | None = None) -> int:
        """Credit the Director with every decided alpha in memory, counted the way a run
        counts them: repair children of one parent (its waves) as one result, a success
        when any of them was accepted; every other alpha as itself. ``only``: credit just
        these arms (arms added during the run: designed or new-dataset arms)."""
        n = 0
        outcomes = outcomes if outcomes is not None else self.memory.candidate_outcomes()
        models = arm_models(outcomes, self.memory.critic_reviews())
        waves: dict[str, list[tuple[tuple, bool]]] = {}
        for row in outcomes:
            c, d = row["candidate"], row["decision"]
            key = self._arm_of(c, models.get(c.candidate_id))
            if d is None or d.decision not in DECIDED or key is None \
                    or (only is not None and key not in only):
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

    def _seed_new_arms(self, outcomes: list[dict] | None = None) -> None:
        """Arms added during the run (designed, new-dataset): credit their own record from
        memory (owner 2026-11-28: added after the warm start, a designed arm used to restart
        every run from a blank record, so a dataset with 12 rejects looked as good as an
        untried one); with none, start from the market's average record
        (``director.new_arm_prior: market``)."""
        keys, self._new_arms = set(self._new_arms), []
        if not keys:
            return
        self._warm_start(only=keys, outcomes=outcomes)
        if self.agents.new_arm_prior != "market":
            return
        for k in keys:
            st = self.director.state.get(k)
            if st is None or st.successes + st.failures > 0:
                continue
            others = [a for a in self.director.arms if a not in keys]
            rows = [self.director.state[a] for a in others if a[1] == k[1] and a[4] == k[4]
                    and self.director.state[a].successes + self.director.state[a].failures > 0]
            if rows:
                st.successes = sum(r.successes for r in rows) / len(rows)
                st.failures = sum(r.failures for r in rows) / len(rows)

    def _evidence_keys(self, live: list[tuple]) -> set[tuple]:
        """Live arms whose dataset gave us best |Sharpe| (raw or risk-neutralized) >=
        ``director.evidence_sharpe`` in their market (any alpha, repairs included)."""
        if self.agents.evidence_floor_share is None:
            return set()
        if self._evidence is None or self._evidence[0] != self.current_round:
            best: dict[tuple, float] = {}
            for o in self.memory.candidate_outcomes():
                c, r = o["candidate"], o["result"]
                if r is None or r.status != "complete":
                    continue
                vals = [abs(r.sharpe)] if r.sharpe is not None else []
                rn = ((r.alt_stats or {}).get("risk_neutralized") or {}).get("sharpe")
                if isinstance(rn, (int, float)) and not isinstance(rn, bool):
                    vals.append(abs(rn))
                if vals:
                    k = (c.dataset, c.region, c.delay)
                    best[k] = max(best.get(k, 0.0), max(vals))
            self._evidence = (self.current_round, best)
        best = self._evidence[1]
        cut = self.agents.evidence_sharpe
        return {k for k in live if best.get((k[0], k[1], k[4]), 0.0) >= cut}

    def _director_allocate(self, n: int, arms: list[tuple], **kw) -> dict:
        """The Director's allocation of ``n`` slots over ``arms``, with at least
        ``director.evidence_floor_share`` of them to the arms with evidence (owner
        2026-11-28: in run rs20261128045753 JPN other176, the dataset of our submitted JPN
        alpha, and GLB pv1 got no slot while untried thin datasets took 31 of 43)."""
        a = self.agents
        ev = [k for k in arms if k in self._evidence_keys(arms)]
        floor = math.ceil(a.evidence_floor_share * n) if ev and a.evidence_floor_share else 0
        floor = min(floor, n)
        out = {k: 0 for k in self.director.arms}
        if floor:
            for k, m in self.director.allocate(floor, a.floor_share, None, only=ev).items():
                out[k] += m
        if n - floor:
            for k, m in self.director.allocate(n - floor, a.floor_share, a.max_arm_share,
                                               only=arms, **kw).items():
                out[k] += m
        return out

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

    def _repair_ok(self, key: tuple) -> bool:
        """The arm may run repair waves: its region is in this run, or it holds a close
        lead from another region (``repair.cross_region_distance``)."""
        return self._in_run(key) or key in self.lead_arms

    def _emit(self, text: str) -> None:
        if self.out is not None:
            print(text, file=self.out, flush=True)
        if self.live is not None:
            self.live.event(text)

    def attach_live(self, live) -> None:
        """Report to the dashboard: console lines, batches, BRAIN's progress, decisions and
        ``live_state`` (live.py)."""
        self.live = live
        live.state = self.live_state
        self.pipeline.live = live

    def live_state(self) -> dict:
        """The run's numbers for the dashboard's live page."""
        c = self.counter
        models = {}
        for m in self.models.all_models():
            if m in c.down:
                st, why = "down", c.down[m]
            elif c.pause_left(m) > 0:
                st, why = "paused", f"{c.pause_left(m):.0f}s left"
            else:
                st, why = "up", None
            models[m] = {"status": st, "reason": why, "calls": c.by_model.get(m, 0),
                         "too_large": c.too_large.get(m, 0), "empty": c.empty.get(m, 0),
                         "generator": m in self.models.generator}
        quota = getattr(self.client, "quota", None)
        quota = quota if isinstance(quota, dict) and quota else None
        return {"round": self.current_round, "rounds": self.max_rounds,
                "regions": self.run_regions,
                "simulations": {"used": self.sims_used, "committed": self._committed(),
                                "max": self.budget.max_simulations},
                "model_calls": {"used": c.calls, "max": c.max_calls},
                "models": models, "stop_reason": self.stop_reason,
                "stop_requested": self.stop_requested,
                "repair_waves_open": sum(1 for w in self.waves.values() if not w.get("done")),
                "cross_region_leads": [i["parent"] for i in self.cross_region
                                       if i.get("children")],
                "brain_quota": quota}

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
        region, delay = plan.settings.region, plan.settings.delay
        fields = self.catalog_of(region, delay)[0]
        gen = self.generators[key]
        cap = self.field_cap.get(key[3])
        if cap is not None and gen.config.max_datafields_in_prompt > cap:
            gen.config = dataclasses.replace(gen.config, max_datafields_in_prompt=cap)
        gen.prod_failed = prod_correlation_fields(outcomes, region, self.spec, fields)
        # blend partners with shown signal, and this arm's latest results (owner 2026-11-25)
        if gen.config.blend_partners:
            gen.evidence = signal_evidence(outcomes, region, delay, self.spec, fields)
        gen.history = arm_history(outcomes, plan.dataset, region, delay,
                                  gen.config.history_in_prompt)
        gen.failures = arm_failures(outcomes, plan.dataset, region, delay,
                                    max(gen.config.history_in_prompt, 4))
        gen.unused_ops = self.unused_ops
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
        outside = []
        for row, origin, priority in roots + starts:
            c = row["candidate"]
            key = self._arm_of(c, models.get(c.candidate_id) or
                               models.get(origin["candidate"].candidate_id))
            if key is None:
                continue
            dist = distance(row["decision"], row["result"], cfg.unmeasured_failure_penalty)
            if not self._in_run(key):
                # a close lead of a region outside the rotation (owner 2026-11-28): the
                # closest few go on now instead of waiting for their region's turn
                if (cfg.cross_region_distance is not None and dist is not None
                        and dist <= cfg.cross_region_distance):
                    outside.append((dist, c.candidate_id, key, row, priority))
                continue
            starts_row = (float("inf") if dist is None else dist, c.candidate_id, key, row,
                          priority)
            self._resume_rows.append(starts_row)
        outside.sort(key=lambda t: (t[0], t[1]))
        leads = 0
        for dist, cid, key, row, priority in outside:
            if leads >= cfg.cross_region_leads:
                break
            c = row["candidate"]
            self.lead_arms.add(key)
            rep = self._start_wave(key, c, row["decision"], row["result"], priority)
            item = {"parent": cid, "arm": arm_label(key), "distance": round(dist, 3),
                    "notes": rep.notes}
            if rep.children:
                item["children"] = [x.candidate_id for x in rep.children]
                item["fixes"] = [x.lever for x in rep.children]
                leads += 1
            else:
                self.lead_arms.discard(key)
            self.cross_region.append(item)
        for i in self.cross_region:
            if i.get("children"):
                self._emit(f"--- close lead outside this run's regions ({i['arm'].split('/')[1]}, "
                           f"distance {i['distance']}): {_short_id(i['parent'])}, "
                           f"{len(i['fixes'])} fixes: {', '.join(i['fixes'])}")
        # repairs a recheck of BRAIN's late checks just decided (a PROD_CORRELATION value
        # that came hours after the simulation) go first: they are the closest leads
        fresh = {i["candidate_id"] for i in self.rechecked if i.get("decision") == "repair"}
        self._resume_rows.sort(key=lambda t: (t[1] not in fresh, t[0], t[1]))
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
        # a flip of a sign-first parent mirrors it, so its distance equals the parent's
        # (lever_stats._mirrored); it is still the alpha to build on, never the negative
        # parent (run rs20261126082238: four fixes went to the unflipped parent, all negative)
        closer = sorted(x for x in scored if pdist is None or x[0] < pdist
                        or (x[2].lever == "flip_sign" and x[0] <= pdist))
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
        if self.stop_requested:
            entry["next"] = "none: run stopped by hand (resumed at the next run's start)"
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

    def _reevaluate(self) -> None:
        """Rejected alphas a lever can now address get a new decision from their stored
        result (``Pipeline.reevaluate_rejected``), before open repairs are resumed, so a
        near miss turned into a repair is resumed like any other (in its region's run)."""
        outs = self.pipeline.reevaluate_rejected()
        self.reevaluated = [{"candidate_id": o.candidate_id, "decision": o.decision,
                             "failure_class": o.failure_class} for o in outs]
        if outs:
            self._emit(f"--- re-decided {len(outs)} rejected alpha(s) a lever can now fix: "
                       + ", ".join(f"{_short_id(o.candidate_id)} -> {o.decision} / "
                                   f"{o.failure_class}" for o in outs))

    def _recheck_pending(self, start_waves: bool = False) -> None:
        """Alphas decided pending because BRAIN had not finished a check (PROD_CORRELATION
        PENDING, SELF_CORRELATION ERROR) are fetched again and decided
        (``Pipeline.recheck_pending``); a decided one credits its arm, as a pending decision
        did not. No simulation budget. Before resumed repairs, so a PROD_CORRELATION
        failure found here is repaired in this run (in its region's run)."""
        try:
            outs, still = self.pipeline.recheck_pending()
        except BrainLoginRequired:
            raise
        except Exception as exc:  # a lookup that goes wrong never stops the run
            self._emit(f"  rechecking pending alphas failed: {exc}")
            return
        models = arm_models(self.memory.candidate_outcomes(), self.memory.critic_reviews()) \
            if outs else {}
        for o in outs:
            c = self.memory.get_candidate(o.candidate_id)
            key = self._arm_of(c, models.get(o.candidate_id)) if c is not None else None
            if key is not None and o.decision in DECIDED:
                self.director.update(key, o.decision == "accept")
            self.rechecked.append({"candidate_id": o.candidate_id, "decision": o.decision,
                                   "failure_class": o.failure_class})
            if start_waves and o.decision == "repair" and key is not None \
                    and self._repair_ok(key):
                # during a run: the repair wave starts now (at the start, _resume_repairs
                # takes these first)
                d = o.decision_record or (self.memory.decisions_for(o.candidate_id) or [None])[-1]
                res = o.result or (self.memory.results_for(o.candidate_id) or [None])[-1]
                rep = self._start_wave(key, c, d, res)
                self.resumed.append({"parent": o.candidate_id, "arm": arm_label(key),
                                     "notes": rep.notes, "rechecked": True,
                                     **({"children": [x.candidate_id for x in rep.children],
                                         "fixes": [x.lever for x in rep.children]}
                                        if rep.children else {})})
                if rep.children:
                    self._emit(f"    {_short_id(o.candidate_id)}: {len(rep.children)} fixes: "
                               + ", ".join(x.lever for x in rep.children))
        if outs or still:
            got = ", ".join(f"{_short_id(o.candidate_id)} -> {o.decision} / {o.failure_class}"
                            for o in outs) or "none"
            self._emit(f"--- rechecked {len(outs) + len(still)} pending alpha(s) BRAIN had not "
                       f"finished checking: {got}"
                       + (f"; {len(still)} still waiting on BRAIN" if still else ""))

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
        after = self.pipeline.config.simulator.cancel_stuck_after_hours
        stuck = ({r["candidate_id"]: r["hours"] for r in self.pipeline.stuck_submissions(after)}
                 if after is not None else {})
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
            if out is None and c.candidate_id in stuck:
                # still unfinished long after submission: BRAIN's stuck simulations sat at
                # 90% for days (2026-11-24); cancel it so it holds no place on BRAIN
                hours = stuck[c.candidate_id]
                seen = getattr(self.pipeline.simulator, "last_recover", None) or "no result"
                self._emit(f"  {_short_id(c.candidate_id)}: {seen} {hours:.0f} h after "
                           f"submission, cancelling")
                try:
                    gone = self.pipeline.cancel_submission(
                        c, p["brain_sim_id"], f"still unfinished {hours:.0f} h after submission "
                                              f"(simulator.cancel_stuck_after_hours {after:g})")
                except BrainLoginRequired:
                    raise
                if gone is not None:
                    self.cancelled.append({"candidate_id": c.candidate_id,
                                           "brain_sim_id": p["brain_sim_id"],
                                           "hours": round(hours, 1)})
                    continue
            if out is None:
                still += 1
                continue
            key = self._arm_of(c, models.get(c.candidate_id))
            if key is not None and out.decision in DECIDED:
                self.director.update(key, out.decision == "accept")
            self.recovered.append({"candidate_id": c.candidate_id, "decision": out.decision,
                                   "failure_class": out.failure_class})
        if self.cancelled:
            self._emit(f"--- cancelled {len(self.cancelled)} simulation(s) stuck on BRAIN: "
                       + ", ".join(f"{_short_id(i['candidate_id'])} ({i['hours']:.0f} h)"
                                   for i in self.cancelled))
        if self.recovered or still:
            got = ", ".join(f"{i['candidate_id']} {i['decision']}/{i['failure_class']}"
                            for i in self.recovered) or "none"
            self._emit(f"--- recovered {len(self.recovered)} result(s) BRAIN finished after "
                       f"an earlier run stopped waiting: {got}"
                       + (f"; {still} not ready yet" if still else ""))

    def _market_bar(self, region: str, delay: int) -> float:
        return (self.seen_limits.get((region, delay), {}).get("LOW_SHARPE")
                or self.pipeline.config.for_settings(region, delay).gates.sharpe_min)

    def _refresh_dead(self, outcomes: list[dict] | None = None) -> None:
        """Datasets dead in the run's markets (arm_designer.dead_datasets); a line for each
        newly dead one."""
        cfg = self.agents.designer
        if cfg.dead_after_alphas <= 0 and cfg.thin_dead_after <= 0:
            return
        outcomes = outcomes if outcomes is not None else self.memory.candidate_outcomes()
        markets = {(k[1], k[4]) for k in self.arms if self._in_run(k)}
        for region, delay in sorted(markets):
            meta = self.catalog_of(region, delay)[1]
            counts = ({ds: f.get("alpha_count") for ds, f in dataset_facts(meta).items()}
                      if meta and cfg.thin_dead_after > 0 else None)
            now = arm_designer.dead_datasets(outcomes, region, delay,
                                             self._market_bar(region, delay), cfg, counts)
            old = self.dead.get((region, delay), {})
            for ds, (n, best) in now.items():
                if ds not in old:
                    self._emit(f"--- {ds} in {region} D{delay}: no more slots ({n} new "
                               f"alphas, best |Sharpe| {best:.2f} < "
                               f"{cfg.dead_sharpe_share:g} x bar)")
            self.dead[(region, delay)] = now

    def _design_arms(self) -> None:
        """Arm Designer (arm_designer.py) for each market of this run: continue its latest
        batch of designed arms while any of them is not done, else write a new batch (owner
        2026-11-26). The designed arms (one per generator model) replace the plan's arms of
        that market for this run; the plan still gives the markets, their catalogs, base
        settings and seeds."""
        cfg = self.agents.designer
        outcomes = self.memory.candidate_outcomes()
        self._refresh_dead(outcomes)
        markets: dict[tuple, list[tuple]] = {}
        for k in list(self.arms):
            if self._in_run(k) and k not in self.successor:
                markets.setdefault((k[1], k[4]), []).append(k)
        base_pv = self.generator_config.prompt_version
        for (region, delay), keys in markets.items():
            plan = self.arms[keys[0]]
            fields, meta = self.catalog_of(region, delay)
            if not meta or plan.catalog is None:
                self._emit(f"--- arms for {region} D{delay}: no field catalog with dataset "
                           f"facts; the plan's arms are used")
                continue
            history = self.memory.arm_designs(region, delay)
            batch = arm_designer.latest_batch(history)
            status = ""
            if batch:
                fresh = self.memory.fresh_simulations(region, delay, batch[0]["created_at"])
                done, lines = arm_designer.batch_done(batch, fresh, cfg,
                                                      self.dead.get((region, delay)))
                status = ", ".join(lines)
            if not batch or done:
                gc = self.generator_config
                cands = arm_designer.market_candidates(
                    meta, lambda f, m=meta: field_role(f, m, gc))
                bar = self._market_bar(region, delay)
                base = {k: getattr(plan.settings, k) for k in (
                    "region", "universe", "delay", "decay", "neutralization", "truncation",
                    "start_date", "end_date")}
                new = arm_designer.design_batch(
                    region=region, delay=delay, candidates=cands,
                    dataset_text=load_dataset_meta(dataset_catalog_for(plan.catalog)),
                    field_ids=set(fields or meta), outcomes=outcomes, bar=bar,
                    base_settings=base, earlier={d["dataset"] for d in history}, config=cfg,
                    dead=self.dead.get((region, delay)))
                if not new:
                    self._emit(f"--- arms for {region} D{delay}: no dataset qualifies; the "
                               f"plan's arms are used")
                    continue
                self.memory.put_arm_designs(f"{self.run_id}_{region}_d{delay}", new)
                batch = arm_designer.latest_batch(self.memory.arm_designs(region, delay))
                self._emit(f"--- arms for {region} D{delay}: new batch"
                           + (f" (previous done: {status})" if status else " (first batch)"))
            else:
                self._emit(f"--- arms for {region} D{delay}: continuing batch "
                           f"{batch[0]['batch_id']} ({status})")
            designed = set()
            for d in batch:
                settings = GenerationSettings.from_mapping(d["settings"])
                arm = ArmPlan(d["dataset"], settings,
                              deque(Source("hypothesis", h) for h in d["hypotheses"]),
                              catalog=plan.catalog)
                for gm in self.models.generator:
                    key = (d["dataset"], region, base_pv, gm, delay)
                    designed.add(key)
                    if key not in self.arms:
                        self._add_arm(key, dataclasses.replace(arm, sources=deque(arm.sources)),
                                      [])
                self._emit(f"    {d['dataset']} ({d['category']}): {d['why']}; score "
                           f"{d['score']:g}; decay {settings.decay}, {settings.neutralization}")
                self.designed.append({k: d[k] for k in (
                    "batch_id", "region", "delay", "dataset", "category", "why", "score",
                    "parts", "facts", "settings_why")})
            self.inactive |= {k for k in keys if k not in designed}
        self._seed_new_arms(outcomes)

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

    def run(self, rounds: int, *, catch_interrupt: bool = False) -> dict:
        """The research run, then the Lesson Writer and the report. With ``catch_interrupt``
        (``cli research``, owner 2026-11-26: a stop by hand must still write the lessons and
        end smoothly) Ctrl-C is staged:

        1. first press: no new model calls, no new alphas to BRAIN, no new repair waves; the
           alphas already on BRAIN finish and are decided, then lessons and the report;
        2. second press: the simulations still on BRAIN are cancelled; rule lessons (no model
           call) and the report are written at once;
        3. third press: quit at once (nothing more is written).

        Open repairs and anything left unfinished are picked up at the next run's start."""
        restore = None
        self.max_rounds = rounds
        if catch_interrupt and threading.current_thread() is threading.main_thread():
            restore = signal.signal(signal.SIGINT, self._on_interrupt)
        finished = False
        try:
            hard = False
            try:
                self._run(rounds)
            except KeyboardInterrupt:
                if self.interrupts >= 3 or not catch_interrupt:
                    raise
                hard = True
                self.stop_reason = ("stopped by hand (Ctrl-C twice): simulations still on "
                                    "BRAIN were cancelled")
                self._emit(f"--- {self.stop_reason}; writing lessons (rules) and the report "
                           "(Ctrl-C again to quit at once)")
            if self.stop_requested and not hard:
                self._emit("--- every alpha on BRAIN is in; writing lessons and the report")
            self.counter.stopped = None   # the Lesson Writer may use its one model call
            lessons = self._learn(use_model=not hard)
            report = self.report()   # after the Lesson Writer, so its model call is counted
            report["lessons"] = lessons
            finished = True
            if self.live is not None:
                self.live.finish("stopped" if (hard or self.stop_requested) else "finished",
                                 self.stop_reason)
            return report
        finally:
            if self.live is not None and not finished:
                self.live.finish("stopped", self.stop_reason or "stopped with an error")
            if restore is not None:
                signal.signal(signal.SIGINT, restore)

    def _on_interrupt(self, signum, frame) -> None:
        """SIGINT handler while ``run(catch_interrupt=True)`` runs (see ``run``)."""
        self.interrupts += 1
        if self.interrupts == 1:
            self.request_stop()
            return
        if self.interrupts >= 3:   # the next one goes straight to Python
            signal.signal(signal.SIGINT, signal.default_int_handler)
        raise KeyboardInterrupt

    def request_stop(self, why: str = "stopped by hand (Ctrl-C)") -> None:
        """Finish smoothly: no new model calls (each raises ``LLMBudgetExhausted``, which
        every caller already treats as the end of the run), no new alphas to BRAIN and no new
        repair waves; the alphas already running on BRAIN are waited for and decided."""
        if self.stop_requested:
            return
        self.stop_requested = True
        self.stop_reason = self.stop_reason or why
        self.counter.stopped = why
        n = self._in_flight
        wait = (f"waiting for the {n} alpha(s) on BRAIN to finish, then "
                if n else "")
        print(f"\n--- {why}: {wait}writing lessons and the report. "
              "Ctrl-C again to cancel what is on BRAIN and stop now.",
              file=self.out or sys.stderr, flush=True)

    def _learn(self, use_model: bool = True) -> dict | None:
        """Lesson Writer after the run (lessons.py): evidence from all of memory, lessons
        written by the strategy model while the budget lasts, else by rules; stored for the
        next run's agents."""
        cfg = self.agents.lessons
        if not cfg.after_every_batch:
            return None
        llm = self.fallback(self.models.strategy) if cfg.use_model and use_model else None
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
        # BRAIN's count of our submissions by region and data category: context for
        # Strategy Evolution (the real client only; None when BRAIN does not answer)
        fetch = getattr(self.client, "submission_diversity", None)
        self.submissions = fetch() if fetch is not None else None
        if self.agents.designer.enabled:
            self._design_arms()
        self._prepare_seeds()
        self._recover_submissions()
        self._recheck_pending()
        self._reevaluate()
        if self.agents.repair.resume_from_memory:
            try:
                self._resume_repairs()
            except LLMUnavailable as exc:
                self.stop_reason = str(exc)
                return self.report()
        idle = 0
        self._mark_progress()   # the loop guard counts from the first round
        for r in range(1, rounds + 1):
            if self.sims_used >= self.budget.max_simulations:
                self.stop_reason = "simulation budget used up"
                break
            if self._stalled():
                break
            login_stop = getattr(self.client, "login_stop", None)
            if login_stop is not None and (why := login_stop()):
                self.stop_reason = why
                break
            before = self._committed()
            quota_stop = getattr(self.client, "quota_stop", None)
            if quota_stop is not None and (why := quota_stop()):
                self.stop_reason = why
                break
            try:
                every = self.agents.recheck_every_rounds
                if every and r > 1 and (r - 1) % every == 0:
                    self._recheck_pending(start_waves=True)
                self._round(r)
            except LLMUnavailable as exc:  # budget used up, or the model failed
                self.stop_reason = str(exc)
            except SimulationQuotaExhausted as exc:
                self.stop_reason = str(exc)
                break
            except BrainLoginRequired as exc:   # nobody to answer a new login: stop cleanly
                self.stop_reason = f"BRAIN login lost: {exc}"
                break
            try:   # alphas already screened still go to BRAIN, whatever stopped the round
                self._flush()
            except BrainLoginRequired as exc:   # running alphas are recovered next start
                self.stop_reason = f"BRAIN login lost: {exc}"
            except LLMUnavailable as exc:
                self.stop_reason = self.stop_reason or str(exc)
            except SimulationQuotaExhausted as exc:
                # BRAIN refused more simulations today: the ones already running stay on
                # BRAIN and are recovered at the next run start
                self.stop_reason = str(exc)
            if self.stop_reason:
                break
            # stall guard: rounds that spend model calls but send nothing to BRAIN (every
            # alpha dropped or rejected before simulation) end the run, so a run without a
            # model-call cap cannot burn the day's free quota on nothing
            idle = idle + 1 if self._committed() == before else 0
            limit = self.agents.stop_after_idle_rounds
            if limit is not None and idle >= limit:
                self.stop_reason = f"{idle} round(s) in a row sent no alpha to BRAIN"
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
                if not queue or cap <= 0 or waves >= limit or not self._repair_ok(key):
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
        self.current_round = r
        if r > 1 or not self.agents.designer.enabled:   # round 1: _design_arms did it
            self._refresh_dead()
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
        slots -= sum(len(c) for c in first.values())
        # the new alphas' allocation is made before the repairs go to BRAIN, so a slot a
        # finished repair frees can take new alphas at once (run rs20261125173606: new
        # alphas waited ~30 min twice for a repair batch's timeouts while slots sat idle)
        live = self._live_arms()
        alloc = self._allocate(slots, live) if slots > 0 and live else {}
        entry["allocation"] = {arm_label(k): n for k, n in alloc.items()}
        repairs = sum(len(c) for c in first.values())
        if live:
            self._emit(f"--- round {r}: {f'{repairs} repair(s), then ' if repairs else ''}"
                       f"{slots} new: {_slots_text(alloc) or 'none'}")
        self._set_fresh(r, alloc, entry, "")
        for key, cands in first.items():
            arm_log = {"arm": arm_label(key), "slots": len(cands), "repairs_first": True}
            entry["arms"].append(arm_log)
            self._screen_and_simulate(key, cands, {}, arm_log)
            if self._committed() >= self.budget.max_simulations:
                self.stop_reason = "simulation budget used up"
                self._fresh_left = {}
                return
        if not live and (wait := self._pause_wait()) is not None:
            # every generator model left is paused on a rate limit: wait it out, then
            # allocate this round's new alphas again
            self._emit(f"--- every generator model is paused on a rate limit; waiting "
                       f"{wait:.0f}s")
            self._sleep(wait)
            live = self._live_arms()
            alloc = self._allocate(slots, live) if slots > 0 and live else {}
            entry["allocation"] = {arm_label(k): n for k, n in alloc.items()}
            self._set_fresh(r, alloc, entry, "")
        if not live:
            self.stop_reason = "every generator model is down: " + "; ".join(
                f"{m}: {why}" for m, why in {**self.counter.down, **self.no_fit}.items())
            return
        lost, emptied = self._run_fresh()
        passes = 0
        # every pass for lost slots follows a model leaving the live arms (down, or no
        # prompt fits), so the loop ends; the cap is a guard against a slot that is lost
        # without that
        lost_passes, max_lost = 0, len(self.models.generator) + 1
        while not self._stalled():
            if lost:
                lost_passes += 1
                if lost_passes > max_lost:
                    return
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
            if a.evidence_floor_share:
                alloc = {k: 0 for k in self.director.arms}
                for r in dict.fromkeys(k[1] for k in live):
                    arms_r = [k for k in live if k[1] == r]
                    n_r = round(slots * len(arms_r) / len(live))
                    for k, m in self._director_allocate(n_r, arms_r).items():
                        alloc[k] += m
                short = slots - sum(alloc.values())
                if short > 0:
                    for k, m in self.director.allocate(short, a.floor_share, a.max_arm_share,
                                                       only=live).items():
                        alloc[k] += m
                return consolidate(alloc, a.min_batch)
            return consolidate(self.director.allocate(
                slots, a.floor_share, a.max_arm_share, only=live,
                region_floor_share=a.region_floor_share,
                max_region_share=a.max_region_share), a.min_batch)
        # rotation: slots to the run's regions with the fewest simulations so far (memory
        # plus this run), then the Director picks arms within each region
        regions = [r for r in self.run_regions if any(k[1] == r for k in live)]
        per_region = balance_slots(slots, {r: self.region_sims[r] for r in regions})
        alloc = {k: 0 for k in self.director.arms}
        for r, n in per_region.items():
            if n:
                got = consolidate(self._director_allocate(n, [k for k in live if k[1] == r]),
                                  a.min_batch)
                for k, m in got.items():
                    alloc[k] += m
        return alloc

    def _live_arms(self) -> list[tuple]:
        return [k for k in self.arms if self.counter.is_up(k[3]) and self._in_run(k)
                and k not in self.successor and k[3] not in self.no_fit
                and k not in self.inactive
                and k[0] not in self.dead.get((k[1], k[4]), {})]

    def _current(self, key: tuple) -> tuple:
        """The arm that now stands for ``key``: itself, or its latest evolved successor."""
        while key in self.successor:
            key = self.successor[key]
        return key

    def _run_allocation(self, r: int, alloc: dict, entry: dict, tag: str) -> tuple[int, int]:
        """Run each arm's slots. Returns (slots lost to a down generator model, slots
        emptied by the generator, Pre-Filter or Critic)."""
        self._set_fresh(r, alloc, entry, tag)
        return self._run_fresh()

    def _set_fresh(self, r: int, alloc: dict, entry: dict, tag: str) -> None:
        self._fresh_left = {k: n for k, n in alloc.items() if n}
        self._fresh_ctx = (r, entry, tag, {k: i for i, k in enumerate(alloc)})
        self._fresh_lost = self._fresh_emptied = 0

    def _run_fresh(self) -> tuple[int, int]:
        """Generate, screen and queue the arms left in ``_fresh_left``, one arm at a time
        (a flush started by one arm may take later arms early, see ``_fresh_for_freed``).
        Returns (slots lost to a down generator model, slots emptied by the generator,
        Pre-Filter or Critic), counting arms taken mid-batch."""
        r, entry, tag, index = self._fresh_ctx
        lost = emptied = 0
        while self._fresh_left and not self._stalled():
            key = next(iter(self._fresh_left))
            n = self._fresh_left.pop(key)   # popped first: a flush inside may take others
            if not self.counter.is_up(key[3]):  # went down or paused earlier in this pass
                lost += n
                continue
            arm_log = {"arm": arm_label(key), "slots": n}
            if tag:
                arm_log["refill"] = True
            entry["arms"].append(arm_log)
            d, e = self._arm(r, index[key], key, n, arm_log, tag)
            lost, emptied = lost + d, emptied + e
            if self._committed() >= self.budget.max_simulations:
                self.stop_reason = "simulation budget used up"
                break
        self._fresh_left = {}
        lost, emptied = lost + self._fresh_lost, emptied + self._fresh_emptied
        self._fresh_lost = self._fresh_emptied = 0
        return lost, emptied

    def _fresh_for_freed(self, room: int,
                         budget: int) -> list[tuple[tuple, CandidateRecord, dict]]:
        """New alphas for BRAIN slots freed mid-batch: the next arms of the round's
        allocation, generated whole (one Generator call per arm, as in ``_arm``) until at
        least ``room`` are ready; extras wait in the simulator's queue for the next free
        slot. At most ``budget`` alphas: (arm, alpha, arm log)."""
        out: list[tuple[tuple, CandidateRecord, dict]] = []
        if not self.agents.fresh_fill_freed_slots:
            return out
        r, entry, tag, index = self._fresh_ctx
        while len(out) < room and self._fresh_left and not self._stalled():
            left = budget - len(out)     # simulations the budget still allows
            if left <= 0:
                break
            key = next(iter(self._fresh_left))
            n = self._fresh_left.pop(key)
            if not self.counter.is_up(key[3]):
                self._fresh_lost += n
                continue
            arm_log = {"arm": arm_label(key), "slots": n, "fresh_mid_batch": True}
            if tag:
                arm_log["refill"] = True
            entry["arms"].append(arm_log)
            lost, go = self._prepare_arm(r, index[key], key, n, arm_log, tag)
            for c in go[left:]:            # over budget: not simulated
                self._wave_done(c.candidate_id)
            go = go[:left]
            self._fresh_lost += lost
            self._fresh_emptied += max(0, n - lost - len(go))
            out += [(key, c, arm_log) for c in go]
        if out:
            self._emit(f"--- BRAIN: {len(out)} new alpha(s) for freed slots: "
                       f"{', '.join(_short_id(c.candidate_id) for _, c, _ in out)}")
        return out

    def _arm(self, r: int, i: int, key: tuple, n: int, log: dict,
             tag: str = "") -> tuple[int, int]:
        """Fill ``n`` slots of one arm. Returns (slots lost to a down generator model,
        slots emptied by the generator, Pre-Filter or Critic)."""
        lost, go = self._prepare_arm(r, i, key, n, log, tag)
        simulated = self._simulate_all(key, go, log)
        return lost, max(0, n - lost - simulated)

    def _pool_pnl(self) -> dict[str, dict]:
        """{region: {alpha id: daily PnL}} of the active pool (our submitted alphas): the
        Blend Planner checks partners against them."""
        out: dict[str, dict] = {}
        try:
            matrix = self.pipeline.pool.matrix()
        except Exception:   # a missing or unreadable PnL file never stops a run
            return out
        for alpha_id in getattr(matrix, "columns", []):
            c = self.memory.get_candidate(alpha_id)
            if c is not None:
                out.setdefault(c.region, {})[alpha_id] = matrix[alpha_id].dropna()
        return out

    def _blend_flag(self, cand: CandidateRecord, result) -> str | None:
        """Stability flag of a planned blend against its base alpha (recorded only)."""
        by_id = {}
        c = cand
        while c.parent_id and (p := self.memory.get_candidate(c.parent_id)) is not None:
            by_id[p.candidate_id] = {"candidate": p}
            c = p
        base = blend_base(cand, by_id)
        if base.candidate_id == cand.candidate_id:
            return None
        rs = self.memory.results_for(base.candidate_id)
        return stability_flag(result, rs[-1] if rs else None, self.agents.blend)

    def _mark_progress(self) -> None:
        """An alpha went to BRAIN or came back from it: the loop guard's clock restarts."""
        self._progress_at = self._clock()
        self._progress_calls = self.counter.calls

    def _pause_wait(self) -> float | None:
        """Seconds until the first paused generator model may be called again, when the
        arms of a paused model would be live then; None if no generator model is paused."""
        left = [self.counter.pause_left(m) for m in self.models.generator
                if m not in self.counter.down and m not in self.no_fit]
        left = [x for x in left if x > 0]
        return min(left) + 1 if left else None

    def _stalled(self) -> bool:
        """The loop guard (owner 2026-11-26, after a round looped through 9 refill passes
        until stopped by hand): no alpha went to or came back from BRAIN for
        ``director.stall_minutes``, or ``director.stall_model_calls`` model calls were spent
        since the last one went out. Sets the stop reason (the run then flushes what is
        queued, writes its report and lessons) and returns True."""
        if self.stop_reason:
            return True
        if self._progress_at is None:
            return False
        a = self.agents
        idle = (self._clock() - self._progress_at) / 60
        calls = self.counter.calls - self._progress_calls
        why = None
        if a.stall_minutes is not None and idle >= a.stall_minutes:
            why = f"{idle:.0f} min"
        elif a.stall_model_calls is not None and calls >= a.stall_model_calls:
            why = f"{calls} model calls"
        if why is None:
            return False
        self.stop_reason = (f"loop guard: {why} with no alpha sent to or decided by BRAIN "
                            f"(director.stall_minutes / stall_model_calls)")
        self._emit(f"--- {self.stop_reason}; stopping")
        return True

    def _shrink_fields(self, model: str, why: str) -> bool:
        """After a too-large generator request: halve the fields this model's prompts list
        (``field_cap``, for every arm of the model, for the rest of the run). False when the
        cap is already at ``MIN_FIELDS_WHEN_TOO_LARGE``: the model then writes no alphas this
        run (``no_fit``), so its arms get no slots and the round's refill ends."""
        now = self.field_cap.get(model, self.generator_config.max_datafields_in_prompt)
        if now > MIN_FIELDS_WHEN_TOO_LARGE:
            self.field_cap[model] = max(MIN_FIELDS_WHEN_TOO_LARGE, now // 2)
            self._emit(f"  {model}: generator prompt too large, listing "
                       f"{self.field_cap[model]} fields instead of {now}")
            return True
        self.no_fit[model] = why
        self._emit(f"  {model}: generator prompt too large even with {now} fields; it writes "
                   f"no alphas this run (its arms' slots go to other models)")
        return False

    def _prepare_arm(self, r: int, i: int, key: tuple, n: int, log: dict,
                     tag: str = "") -> tuple[int, list[CandidateRecord]]:
        """The arm's queued repairs plus new alphas for the rest of ``n`` slots, screened.
        Returns (slots lost to a down generator model, alphas that may go to BRAIN)."""
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
            res = None
            while not self._stalled():
                try:
                    res = self._generator(key).generate(
                        source, need, batch_id=f"{self.run_id}r{r:02d}a{i}{tag}")
                except GenerationError as exc:
                    log["generation_error"] = str(exc)
                except RequestTooLarge as exc:
                    # the prompt does not fit the model's per-minute limit: list fewer
                    # fields and try again; at the floor, the model writes no more alphas
                    # this run (run rs20261126060436 looped through 9 refill passes giving
                    # Groq models slots their prompt could never fit)
                    if self._shrink_fields(key[3], str(exc)):
                        continue
                    log["generation_error"] = str(exc)
                    lost = need
                except CallFailed as exc:
                    # an empty reply or a rate-limit pause: these slots go back to the
                    # Director; the model stays up (it is down only after repeated failures)
                    log["generation_error"] = str(exc)
                    lost = need
                except ModelDown as exc:  # the arm gets no more slots; the run goes on
                    log["generation_error"] = str(exc)
                    self._emit(f"{exc}; its arms get no more slots, these go to live arms")
                    lost = need
                break
            if res is not None:
                batch += res.candidates
                rationales.update(res.rationales)
                log["generated"] = len(res.candidates)
                log["generator_rejected"] = len(res.rejected)
        return lost, self._screen(key, batch, rationales, log)

    def _screen_and_simulate(self, key: tuple, batch: list[CandidateRecord],
                             rationales: dict[str, str], log: dict) -> int:
        """Pre-Filter, Critic and judge, then simulate. Returns how many were simulated."""
        return self._simulate_all(key, self._screen(key, batch, rationales, log), log)

    def _simulate_all(self, key: tuple, go: list[CandidateRecord], log: dict) -> int:
        """Queue screened alphas for BRAIN within the budget. Returns how many."""
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
        return self.sims_used + len(self.sim_queue) + len(self.fill_queue) + self._in_flight

    def _simulate(self, key: tuple, cand: CandidateRecord, log: dict, *,
                  credit: bool = True) -> None:
        """Queue an alpha for BRAIN; the queue goes out once it holds
        ``simulator.max_concurrent`` alphas, and at the end of each round. ``credit=False``
        (seeds): the outcome does not update the Director's view of the arm, since the
        alpha is not one of the arm's ideas."""
        self.sim_queue.append((key, cand, log, credit))
        self._mark_progress()
        if len(self.sim_queue) >= self.max_concurrent:
            self._flush()

    def _flush(self) -> None:
        """Simulate the queued alphas together and decide each the moment its result is in
        (owner 2026-11-23: a slow alpha must not hold up the others): Director credit,
        repair wave. When a slot frees, ready repair waves are screened and added to the
        running batch (``_top_up``), within the simulation budget. A model failure in one
        repair does not stop the others' bookkeeping; the first is raised at the end."""
        fills, self.fill_queue = self.fill_queue, []
        queued, self.sim_queue = [(k, c, lg, True) for k, c, lg in fills] + self.sim_queue, []
        if not queued:
            return
        if self.stop_requested:   # Ctrl-C: nothing new goes to BRAIN
            self.dropped_on_stop += [c.candidate_id for _, c, _, _ in queued]
            self._emit(f"--- stopping: {len(queued)} screened alpha(s) not sent to BRAIN")
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
            if added:
                self._mark_progress()
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
        cap = min(free, self.budget.max_simulations - self._committed() + len(self.fill_queue))
        if cap <= 0:
            return []
        out = []
        while self.fill_queue and len(out) < cap:   # empty-alpha retries first
            out.append(self.fill_queue.pop(0))
        fills = len(out)
        if fills:
            self._emit(f"--- BRAIN: {fills} empty alpha(s) again with missing values filled: "
                       f"{', '.join(_short_id(c.candidate_id) for _, c, _ in out)}")
        if cap - fills <= 0:
            return out
        for key, cands in (self._take_repairs(cap - fills).items()
                           if self.agents.repair.fill_freed_slots else ()):
            arm_log = {"arm": arm_label(key), "slots": len(cands), "repairs_mid_batch": True}
            if self.log:
                self.log[-1]["arms"].append(arm_log)
            go = self._screen(key, cands, {}, arm_log)
            room = cap - len(out)
            for c in go[room:]:            # over budget: their place in the wave is closed
                self._wave_done(c.candidate_id)
            out += [(key, c, arm_log) for c in go[:room]]
        if len(out) > fills:
            self._emit(f"--- BRAIN: {len(out) - fills} repair fix(es) join the running batch: "
                       f"{', '.join(_short_id(c.candidate_id) for _, c, _ in out[fills:])}")
        if len(out) < cap:   # then the round's new alphas not generated yet
            out += self._fresh_for_freed(
                cap - len(out), self.budget.max_simulations - self._committed() - len(out))
        return out

    def _retry_empty(self, key: tuple, cand: CandidateRecord, outcome,
                     log: dict) -> CandidateRecord | None:
        """An alpha BRAIN returned empty (no positions in any year; ``simulator.empty_alpha``)
        goes to BRAIN once more with each vec_* part given its middle value where the vector
        has none (``critic.nan_guarded``), as child ``<id>_n1`` (lever
        ``fill_missing_values``). Alphas that came back with positions are never rewritten:
        on memory the fill would have changed 74 working alphas, among them the submitted
        CHN breadth blend. Returns the queued child, or None."""
        r = outcome.result
        if r is None or r.status != "failed" or cand.lever == FILL_LEVER \
                or self.stop_requested \
                or not any(str(n).startswith(EMPTY_ALPHA) for n in r.notes):
            return None
        expression, filled, unfixed = nan_guarded(cand.expression)
        note = {"candidate_id": cand.candidate_id, "filled": filled, "unfilled": unfixed}
        if not filled:
            note["status"] = "no vec_* part to fill"
            self.empty_retries.append(note)
            return None
        fields = catalog_for(self._mtyped, self._typed, cand.region, cand.delay)
        if not analyze(expression, self.spec, datafields=fields,
                       limits=self.generator_config.limits).ok:
            note["status"] = "filled expression not valid"
            self.empty_retries.append(note)
            return None
        child = assign_lineage(dataclasses.replace(cand, expression=expression), cand,
                               lever=FILL_LEVER)
        child = dataclasses.replace(child, candidate_id=f"{cand.candidate_id}_n1",
                                    hypothesis=f"{cand.hypothesis} [missing values filled: "
                                               f"{', '.join(filled)}]")
        try:
            child.validate()
        except SchemaError as exc:
            note["status"] = f"invalid: {exc}"
            self.empty_retries.append(note)
            return None
        note.update(status="queued", child=child.candidate_id)
        self.empty_retries.append(note)
        self.fill_queue.append((key, child, log))
        self._emit(f"  {_short_id(cand.candidate_id)}: BRAIN held no positions (empty alpha); "
                   f"queued once more with missing values filled ({', '.join(filled)})")
        return child

    def _after_simulation(self, key: tuple, cand: CandidateRecord, log: dict, credit: bool,
                          outcome) -> None:
        self._mark_progress()
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
        if cand.lever == BLEND_PLAN and outcome.result is not None \
                and outcome.result.status == "complete":
            bflag = self._blend_flag(cand, outcome.result)
            if bflag:
                item["blend_flag"] = bflag
                self.blend_flags.append({"candidate_id": cand.candidate_id, "flag": bflag})
                self._emit(f"  flag: {_short_id(cand.candidate_id)}: {bflag}")
        retry = self._retry_empty(key, cand, outcome, log)
        if retry is not None:
            item["empty_retry"] = retry.candidate_id
            credit = False   # an expression bug, not the arm's idea: the retry is credited
        in_wave = cand.candidate_id in self.wave_of
        if outcome.decision in DECIDED and credit and not in_wave:  # a wave: one credit at its end
            self.director.update(self._current(key), outcome.decision == "accept")
        if in_wave:   # the wave decides where repair goes next, once all its children are in
            self._wave_done(cand.candidate_id, cand, outcome)
        elif outcome.decision == "repair" and outcome.decision_record is not None \
                and not self.stop_requested:   # after Ctrl-C: resumed at the next start
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

    def _market_facts(self, markets) -> dict[tuple, dict[str, dict]]:
        """{(region, delay): {dataset: facts}} (``generator.dataset_facts``)."""
        return {mk: dataset_facts(self.catalog_of(*mk)[1]) for mk in markets}

    @staticmethod
    def _run_dataset_info(facts: dict[tuple, dict[str, dict]]) -> dict[str, dict]:
        """One entry per dataset over the run's markets: its least crowded market's facts
        (highest value score, then fewest BRAIN alphas)."""
        out: dict[str, dict] = {}
        for per in facts.values():
            for ds, f in per.items():
                if ds not in out or dataset_appeal(f) < dataset_appeal(out[ds]):
                    out[ds] = f
        return out

    def _evolve(self, r: int) -> None:
        market_ds = self._market_datasets()
        catalog_datasets = set().union(*market_ds.values())
        known = sorted({k[0] for k in self.arms} | catalog_datasets)
        live = {(k[1], k[4]) for k in self.arms if self._in_run(k)}
        facts = self._market_facts(live or market_ds)
        evo = self.strategist.evolve(self.memory.candidate_outcomes(),
                                     self.generator_config.prompt_version, known_datasets=known,
                                     dataset_info=self._run_dataset_info(facts),
                                     submissions=getattr(self, "submissions", None))
        if evo is None:
            return
        added, replaced = [], []
        # only the live arms of this run's regions evolve: the others get no slots in this
        # run, and each evolved arm replaces its parent (one arm per idea, not two)
        base = [(k, p) for k, p in self.arms.items()
                if k not in self.successor and self._in_run(k) and k not in self.inactive]
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
        # least crowded (dataset, market) first, so the cap keeps those (BRAIN's tip for
        # production correlation, 2026-11-25); the hypotheses' order breaks ties
        markets.sort(key=lambda m: dataset_appeal(facts.get((m[1], m[2]), {}).get(m[0])))
        models = [m for m in self.models.generator if m not in self.counter.down]  # paused ok
        picks = [(*mk, models[(i + p) % len(models)]) for p in range(len(models))
                 for i, mk in enumerate(markets)] if models else []
        # thinly used datasets (designer.min_brain_alphas) get no new-dataset arm here: the
        # designer's exploration arm is their one way in (owner 2026-11-28)
        dz = self.agents.designer
        def known_thin(p) -> bool:
            n = (facts.get((p[1], p[2]), {}).get(p[0]) or {}).get("alpha_count")
            return n is not None and arm_designer.thin({"brain_alphas": n}, dz)

        picks = [p for p in picks if not known_thin(p)]
        for ds, region, delay, plan, model in picks[:self.agents.strategy.max_new_arms]:
            settings = dataclasses.replace(plan.settings, dataset=ds)
            sources = deque(Source("hypothesis", t) for t in new_ds[ds])
            self._add_arm((ds, region, evo.prompt_version, model, delay),
                          ArmPlan(ds, settings, sources, evo.guidance_for(region),
                                  catalog=plan.catalog), added)
        self._seed_new_arms()
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
            if like is None:
                self._new_arms.append(key)
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
            "dropped_on_stop": self.dropped_on_stop,
            "empty_retries": self.empty_retries,
            # BRAIN's daily simulation limit after the last submit (X-Ratelimit-* headers)
            "brain_quota": dict(getattr(self.client, "quota", None) or {}),
            "power_pool_candidates": self.power_pool,
            # how plain or crowded this run's fresh alphas were (quality.py; owner 2026-11-25)
            "generation_quality": generation_quality(
                self.memory.candidate_outcomes(), self.run_id, self.spec,
                self.generator_config, self.catalog_of),
            "simulations_used": self.sims_used,
            "max_concurrent": self.max_concurrent,
            "model_calls": self.counter.calls,
            "model_calls_by_model": dict(self.counter.by_model),
            "models_down": self.down(),
            # calls refused as too large per model, and what the size client measured
            # (characters per token, largest prompt) for models with a per-minute token limit
            "too_large_calls": dict(self.counter.too_large),
            "empty_replies": dict(self.counter.empty),
            "rate_limit_pauses": dict(self.counter.pauses),
            "generator_field_cap": dict(self.field_cap),
            "generator_no_fit": dict(self.no_fit),
            "prompt_tokens": self.token_stats(),
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
            "simulations_by_region": dict(self.region_sims),
            "designed_arms": list(self.designed),
            "blend_flags": list(self.blend_flags),   # since region_balance_since
            "arms": arms,
            "recovered_results": self.recovered,
            "cancelled_stuck": self.cancelled,
            "reevaluated": self.reevaluated,
            "rechecked_pending": self.rechecked,
            "resumed_repairs": self.resumed,
            "cross_region_leads": self.cross_region,
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
