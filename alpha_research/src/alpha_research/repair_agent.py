"""Repair Agent: applies a playbook lever to a candidate and returns the repaired child.

``repair.py`` only says which lever comes next. This module carries it out.

Method
1. Levers come from the playbook for the decision's failure class, skipping any lever
   already applied along the parent chain (``MemoryStore.lineage_levers``). Their order is
   learned (lever_stats.py): what each lever did in past repairs, read from memory at
   every repair, ranks the levers, and a lever that also addresses another class the alpha
   fails (a failed gate, or BRAIN's own failed check such as CONCENTRATED_WEIGHT) ranks
   higher. With no evidence the playbook order is kept.
2. Each lever is tried until one applies. A lever that does not apply (decay already at
   the target, no time-series window to lengthen, operator not in the spec) is skipped
   with a note.
3. Mechanical levers rewrite the parsed expression or the settings, with no model:

   ==================  ===========================================================
   increase_decay      decay -> step.to, if the current decay is lower
   tighten_truncation  truncation -> step.to, if the current truncation is higher
   lengthen_ts_window  every integer window (2nd positional argument of a ts_* call)
                       multiplied by step.factor
   wrap_in_smoothing   final expression -> step.operator(expr, step.window)
   hump                final expression -> hump(expr, hump=step.threshold)
   rank_transform      final expression -> rank(expr), unless it already is one
   volatility_scale    final expression -> expr / ts_std_dev(returns, step.window)
   flip_sign           final expression -> -expr, only when BRAIN's Sharpe is negative
   set_neutralization  neutralization -> step.to, if the current one is coarser
                       (NONE < MARKET < SECTOR < INDUSTRY < SUBINDUSTRY)
   backfill_fields     every datafield outside base_fields -> ts_backfill(f, step.lookback)
                       (sparse data leaves few names with values, which concentrates
                       weight and hurts the larger-cap sub-universe)
   liquidity_tilt      final expression -> expr * rank(step.field), tilting weight to
                       liquid names (sub-universe test)
   winsorize           final expression -> winsorize(expr, std=step.std)
   change_universe     universe -> step.to if it is a smaller universe of the alpha's
                       region, else the region's next smaller one
                       (UNIVERSE_ORDER_BY_REGION; regions not listed: never)
   neutralize_<name>   neutralization -> NAME (STATISTICAL, CROWDING,
                       REVERSION_AND_MOMENTUM, FAST, SLOW, SLOW_AND_FAST, COUNTRY),
                       when the region allows it and the alpha does not use it
   ==================  ===========================================================

   Only the final statement is rewritten; ``name = ...;`` statements are kept.

   A combination ``combo:a+b+c`` applies mechanical levers a, b and c in order, each with
   its step from ``step[a]`` or else from its first listing in the playbook. A part that
   does not apply (already backfilled, already ranked, decay already there) is skipped
   with a note; the combination does not apply only when no part does. Combinations come
   from the playbook or are learned: one
   that fixed or improved a class before is offered for that class (lever_stats.py).
4. Any other lever (orthogonalize_against_pool, restrict_to_liquid_fields, model_repair,
   or one added to the playbook later) goes to the model with the metrics, every class
   the alpha fails, the lever, the past repairs of this class that worked (from memory)
   and, for orthogonalization, the pool's expressions. Its expression passes through
   ``fastexpr.analyze`` like everything the Generator writes. The metrics it sees include
   BRAIN's failed checks with value and limit and BRAIN's risk-neutralized stats, and the
   past repairs from the alpha's own region come first. It may also set decay and
   neutralization (validated; bad values are ignored with a note). An invalid reply means
   the lever does not apply. The prompts (with per-lever instructions) are in prompts.py.
5. The child gets ``parent_id``, ``generation + 1`` and ``lever`` (``assign_lineage``) and
   the id ``<parent id>_r<generation>``. Exactly one lever per repair. ``variants``
   instead returns one child per applicable lever (ids ``..._r<generation><a, b, ...>``),
   for rescuing an alpha by hand with ``cli repair``.
"""

from __future__ import annotations

import dataclasses
import random
import string
from dataclasses import dataclass, field
from typing import Collection

from .fastexpr import (Assign, Binary, Call, ComplexityLimits, FastExprError, FastExprSpec,
                       Name, Num, Unary, analyze, catalog, parse, render)
from .generator import GenerationError, ask_json
from .lever_stats import (LeverStats, RepairLearningConfig, combo_parts, failing_classes,
                          family_budget, rank_levers, repair_events, successful_examples)
from .llm import LLMClient, ModelDown
from .memory import MemoryStore
from .prompts import repair_system, repair_user
from .repair import Lever, Playbook
from .research_loop import assign_lineage
from .schemas import (CandidateRecord, DecisionRecord, ResultRecord, SchemaError,
                      experiment_hash, neutralizations_for)

MODEL_LEVERS_NOTE = "lever has no mechanical rewrite; the model was asked"

class LeverNotApplicable(Exception):
    pass


@dataclass
class RepairOutcome:
    parent_id: str
    child: CandidateRecord | None = None
    lever: str | None = None
    notes: list[str] = field(default_factory=list)
    llm_calls: int = 0


# ---------------------------------------------------------------------------
# mechanical levers
# ---------------------------------------------------------------------------

def _final(cand: CandidateRecord) -> tuple[tuple, object]:
    try:
        stmts = parse(cand.expression)
    except FastExprError as exc:
        raise LeverNotApplicable(f"expression does not parse: {exc}") from exc
    return stmts[:-1], stmts[-1]


def _with_final(cand: CandidateRecord, new_final) -> CandidateRecord:
    head, _ = _final(cand)
    return dataclasses.replace(cand, expression=render(head + (new_final,)))


def _need_op(spec: FastExprSpec, name: str) -> None:
    if name not in spec.operators:
        raise LeverNotApplicable(f"operator {name!r} is not in the operator list")


def _num(value) -> Num:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LeverNotApplicable(f"lever step value {value!r} is not a number")
    return Num(repr(value) if isinstance(value, float) else str(value))


def _increase_decay(c, step, spec, result=None):
    to = step.get("to")
    if not isinstance(to, int) or isinstance(to, bool):
        raise LeverNotApplicable("increase_decay needs an integer step.to")
    if c.decay >= to:
        raise LeverNotApplicable(f"decay is already {c.decay} (target {to})")
    return dataclasses.replace(c, decay=to)


def _tighten_truncation(c, step, spec, result=None):
    to = step.get("to")
    if isinstance(to, bool) or not isinstance(to, (int, float)) or not 0 < to <= 1:
        raise LeverNotApplicable("tighten_truncation needs step.to in (0, 1]")
    if c.truncation <= to:
        raise LeverNotApplicable(f"truncation is already {c.truncation} (target {to})")
    return dataclasses.replace(c, truncation=float(to))


def _lengthen(node, factor: int, changed: list):
    if isinstance(node, Call):
        args = [_lengthen(a, factor, changed) for a in node.args]
        if node.name.startswith("ts_") and len(args) >= 2 and isinstance(args[1], Num) \
                and args[1].text.isdigit():
            args[1] = Num(str(int(args[1].text) * factor))
            changed.append(node.name)
        kwargs = tuple((k, _lengthen(v, factor, changed)) for k, v in node.kwargs)
        return Call(node.name, tuple(args), kwargs)
    if isinstance(node, Unary):
        return Unary(node.op, _lengthen(node.operand, factor, changed))
    if isinstance(node, Binary):
        return Binary(node.op, _lengthen(node.left, factor, changed),
                      _lengthen(node.right, factor, changed))
    if isinstance(node, Assign):
        return Assign(node.name, _lengthen(node.value, factor, changed))
    return node


def _lengthen_ts_window(c, step, spec, result=None):
    factor = step.get("factor")
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 2:
        raise LeverNotApplicable("lengthen_ts_window needs an integer step.factor >= 2")
    stmts = parse(c.expression)
    changed: list = []
    new = tuple(_lengthen(s, factor, changed) for s in stmts)
    if not changed:
        raise LeverNotApplicable("no integer time-series window to lengthen")
    return dataclasses.replace(c, expression=render(new))


def _wrap_in_smoothing(c, step, spec, result=None):
    op, window = step.get("operator", "ts_mean"), step.get("window")
    _need_op(spec, op)
    return _with_final(c, Call(op, (_final(c)[1], _num(window)), ()))


def _hump(c, step, spec, result=None):
    _need_op(spec, "hump")
    return _with_final(c, Call("hump", (_final(c)[1],), (("hump", _num(step.get("threshold"))),)))


def _rank_transform(c, step, spec, result=None):
    _need_op(spec, "rank")
    final = _final(c)[1]
    inner = final.operand if isinstance(final, Unary) and final.op == "-" else final
    if isinstance(inner, Call) and inner.name == "rank":  # rank(-rank(x)) = -rank(x)
        raise LeverNotApplicable("expression is already a rank(...)")
    return _with_final(c, Call("rank", (final,), ()))


def _volatility_scale(c, step, spec, result=None):
    _need_op(spec, "ts_std_dev")
    if "returns" not in spec.base_fields:
        raise LeverNotApplicable("'returns' is not a base field")
    vol = Call("ts_std_dev", (Name("returns"), _num(step.get("window"))), ())
    return _with_final(c, Binary("/", _final(c)[1], vol))


def _flip_sign(c, step, spec, result=None):
    if result is None or result.sharpe is None:
        raise LeverNotApplicable("needs BRAIN's Sharpe to know the sign")
    if result.sharpe >= 0:
        raise LeverNotApplicable(f"Sharpe {result.sharpe} is not negative")
    final = _final(c)[1]
    if isinstance(final, Unary) and final.op == "-":
        return _with_final(c, final.operand)
    return _with_final(c, Unary("-", final))


# coarse -> fine; a repair only moves to a finer group. The factor neutralizations
# (STATISTICAL, CROWDING, ...; consultant level) and COUNTRY are not on this scale: each has
# its own lever, neutralize_<name> (below), so the learner scores each one apart.
NEUTRALIZATION_ORDER = ("NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY")
# large -> small per region, only where the sizes are known from the names (TOP<n>;
# TOPSP500 and TOPCS1600 are index universes of about that size; MINVOL10M is the more
# liquid subset of MINVOL1M). Regions not listed have no change_universe.
UNIVERSE_ORDER_BY_REGION = {
    "USA": ("TOP3000", "TOP2000", "TOP1000", "TOP500", "TOPSP500", "TOP200"),
    "EUR": ("TOP2500", "TOPCS1600", "TOP1200", "TOP800", "TOP400"),
    "JPN": ("TOP1600", "TOP1200"),
    "ASI": ("MINVOL1M", "MINVOL10M"),
}
UNIVERSE_ORDER = UNIVERSE_ORDER_BY_REGION["USA"]


def _set_neutralization(c, step, spec, result=None):
    to = step.get("to")
    if to not in NEUTRALIZATION_ORDER:
        raise LeverNotApplicable(f"set_neutralization needs step.to in {list(NEUTRALIZATION_ORDER)}")
    if to not in neutralizations_for(c.region):
        raise LeverNotApplicable(f"neutralization {to} is not allowed in {c.region}")
    if c.neutralization not in NEUTRALIZATION_ORDER:
        raise LeverNotApplicable(f"neutralization {c.neutralization} is not on the group scale")
    if NEUTRALIZATION_ORDER.index(c.neutralization) >= NEUTRALIZATION_ORDER.index(to):
        raise LeverNotApplicable(f"neutralization is already {c.neutralization} (target {to})")
    return dataclasses.replace(c, neutralization=to)


def _change_universe(c, step, spec, result=None):
    """universe -> step.to when it is a smaller universe of the alpha's region; otherwise
    (step.to is another region's, or not given) the next smaller one in the region."""
    order = UNIVERSE_ORDER_BY_REGION.get(c.region)
    if not order or c.universe not in order:
        raise LeverNotApplicable(f"no known smaller universe than {c.universe} in {c.region}")
    at = order.index(c.universe)
    to = step.get("to")
    if to in order:
        if order.index(to) <= at:
            raise LeverNotApplicable(f"universe {c.universe} is not larger than {to}")
    elif at + 1 < len(order):
        to = order[at + 1]
    else:
        raise LeverNotApplicable(f"{c.universe} is already the smallest known in {c.region}")
    return dataclasses.replace(c, universe=to)


def _neutralize_to(target: str):
    """Lever ``neutralize_<target>``: neutralization -> target, when the region allows it
    and the alpha does not use it already. The step is unused."""
    def lever(c, step, spec, result=None):
        if target not in neutralizations_for(c.region):
            raise LeverNotApplicable(f"neutralization {target} is not allowed in {c.region}")
        if c.neutralization == target:
            raise LeverNotApplicable(f"neutralization is already {target}")
        return dataclasses.replace(c, neutralization=target)
    return lever


# one lever per neutralization off the group scale (consultant level, 2026-11-10)
NEUTRALIZE_LEVERS = {f"neutralize_{n.lower()}": n for n in (
    "STATISTICAL", "CROWDING", "REVERSION_AND_MOMENTUM", "FAST", "SLOW", "SLOW_AND_FAST",
    "COUNTRY")}


def _backfill(node, spec, variables: set, lookback: Num, changed: list):
    if isinstance(node, Name):
        n = node.name
        if n in variables or n in spec.groups or n in spec.constants or n in spec.base_fields:
            return node
        changed.append(n)
        return Call("ts_backfill", (node, lookback), ())
    if isinstance(node, Call):
        if node.name == "ts_backfill":
            return node
        return Call(node.name, tuple(_backfill(a, spec, variables, lookback, changed)
                                     for a in node.args),
                    tuple((k, _backfill(v, spec, variables, lookback, changed))
                          for k, v in node.kwargs))
    if isinstance(node, Unary):
        return Unary(node.op, _backfill(node.operand, spec, variables, lookback, changed))
    if isinstance(node, Binary):
        return Binary(node.op, _backfill(node.left, spec, variables, lookback, changed),
                      _backfill(node.right, spec, variables, lookback, changed))
    return node


def _backfill_fields(c, step, spec, result=None):
    _need_op(spec, "ts_backfill")
    lookback = _num(step.get("lookback"))
    stmts, variables, changed, out = parse(c.expression), set(), [], []
    for st in stmts:
        if isinstance(st, Assign):
            out.append(Assign(st.name, _backfill(st.value, spec, variables, lookback, changed)))
            variables.add(st.name)
        else:
            out.append(_backfill(st, spec, variables, lookback, changed))
    if not changed:
        raise LeverNotApplicable("no datafield outside base_fields, or all already backfilled")
    return dataclasses.replace(c, expression=render(tuple(out)))


def _liquidity_tilt(c, step, spec, result=None):
    _need_op(spec, "rank")
    fld = step.get("field", "adv20")
    if fld not in spec.base_fields:
        raise LeverNotApplicable(f"{fld!r} is not a base field")
    tilt = Call("rank", (Name(fld),), ())
    final = _final(c)[1]
    if isinstance(final, Binary) and final.op == "*" and final.right == tilt:
        raise LeverNotApplicable(f"expression is already tilted by rank({fld})")
    return _with_final(c, Binary("*", final, tilt))


def _winsorize(c, step, spec, result=None):
    _need_op(spec, "winsorize")
    final = _final(c)[1]
    if isinstance(final, Call) and final.name == "winsorize":
        raise LeverNotApplicable("expression is already a winsorize(...)")
    return _with_final(c, Call("winsorize", (final,), (("std", _num(step.get("std", 4))),)))


MECHANICAL = {
    "increase_decay": _increase_decay,
    "tighten_truncation": _tighten_truncation,
    "lengthen_ts_window": _lengthen_ts_window,
    "wrap_in_smoothing": _wrap_in_smoothing,
    "hump": _hump,
    "rank_transform": _rank_transform,
    "volatility_scale": _volatility_scale,
    "flip_sign": _flip_sign,
    "set_neutralization": _set_neutralization,
    "backfill_fields": _backfill_fields,
    "liquidity_tilt": _liquidity_tilt,
    "winsorize": _winsorize,
    "change_universe": _change_universe,
    **{name: _neutralize_to(target) for name, target in NEUTRALIZE_LEVERS.items()},
}


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------

def _metrics(result: ResultRecord | None) -> dict:
    if result is None:
        return {}
    keys = ("sharpe", "fitness", "turnover", "returns", "max_drawdown", "margin_bps",
            "sub_universe_sharpe", "sub_universe_cutoff", "weight_concentration",
            "self_correlation")
    out = {k: getattr(result, k) for k in keys if getattr(result, k) is not None}
    # BRAIN's own check values (LOW_2Y_SHARPE, IS_LADDER_SHARPE, ...) and the same stats with
    # BRAIN's risk factors neutralized: evidence for or against a factor neutralization
    failed = {k for k, ok in result.checks.items() if ok is False}
    checks = {k: v for k, v in result.check_values.items() if k in failed}
    if checks:
        out["failed_brain_checks"] = checks
    for name, stats in result.alt_stats.items():
        kept = {k: v for k, v in stats.items() if v is not None}
        if kept:
            out[name] = kept
    return out


class RepairAgent:
    def __init__(self, playbook: Playbook, spec: FastExprSpec, memory: MemoryStore, *,
                 llm: LLMClient | None = None, datafields: Collection[str] | None = None,
                 limits: ComplexityLimits | None = None, max_output: int = 2048,
                 context: str = "", learning: RepairLearningConfig | None = None,
                 seed: int | None = None):
        self.playbook = playbook
        self.context = context  # prompts.brain_context(config)
        self.spec = spec
        self.memory = memory
        self.llm = llm
        self.datafields = catalog(datafields)  # {id: type}: types check vec_* use
        self.limits = limits
        self.max_output = max_output
        self.learning = learning or RepairLearningConfig()
        self.rng = random.Random(seed)

    def evidence(self) -> tuple[list, LeverStats, list]:
        """Repair events, lever statistics and the outcome rows, read fresh from memory."""
        outcomes = self.memory.candidate_outcomes()
        events = repair_events(outcomes, self.learning)
        return events, LeverStats.from_events(events, self.learning.prior,
                                              self.learning.region_prior_strength), outcomes

    def ranked(self, candidate: CandidateRecord, decision: DecisionRecord,
               result: ResultRecord | None, stats: LeverStats) -> list[tuple[Lever, float]]:
        failing = failing_classes(decision, result) | {decision.failure_class}
        return rank_levers(self.playbook, decision.failure_class, failing, stats,
                           self.learning, self.rng, region=candidate.region)

    def repair(self, candidate: CandidateRecord, decision: DecisionRecord,
               result: ResultRecord | None = None,
               pool_expressions: list[str] | None = None) -> RepairOutcome:
        out = RepairOutcome(candidate.candidate_id)
        if not self.playbook.entries.get(decision.failure_class):
            out.notes.append(f"no playbook entry for {decision.failure_class!r}")
            return out
        events, stats, outcomes = self.evidence()
        left, why = family_budget(candidate, outcomes, self.learning)
        if left <= 0:
            out.notes.append(f"no repair: {why}")
            return out
        ranked = self.ranked(candidate, decision, result, stats)
        out.notes.append("lever order: " + ", ".join(f"{lv.name} {sc:g}" for lv, sc in ranked))
        tried = set(self.memory.lineage_levers(candidate.candidate_id))
        for lever, _ in ranked:
            if lever.name in tried:
                continue
            child = self._try(candidate, lever, decision, result, pool_expressions, events, out)
            if child is not None:
                out.child, out.lever = child, lever.name
                out.notes.append(f"{lever.name}: applied")
                return out
        out.notes.append("repair exhausted: every lever for this failure class was tried or "
                         "did not apply")
        return out

    def variants(self, candidate: CandidateRecord, decision: DecisionRecord,
                 result: ResultRecord | None = None, *, limit: int | None = None,
                 use_model: bool = False, ignore_budget: bool = False,
                 pool_expressions: list[str] | None = None) -> tuple[list[CandidateRecord], list[str]]:
        """One child per applicable lever, best ranked first: every lever of every class the
        alpha fails (primary class first), for rescuing an alpha by hand. Model levers are
        skipped unless ``use_model``. The lineage's repair budget caps how many children are
        returned unless ``ignore_budget``."""
        notes: list[str] = []
        events, stats, outcomes = self.evidence()
        if not ignore_budget:
            left, why = family_budget(candidate, outcomes, self.learning)
            if left <= 0:
                return [], [f"no repair: {why}"]
            limit = left if limit is None else min(limit, left)
        failing = failing_classes(decision, result) | {decision.failure_class}
        order = [decision.failure_class] + sorted(failing - {decision.failure_class})
        tried = set(self.memory.lineage_levers(candidate.candidate_id))
        children: list[CandidateRecord] = []
        seen = {(candidate.expression, candidate.universe, candidate.decay,
                 candidate.neutralization, candidate.truncation)}
        names: set[str] = set()
        for cls in order:
            if not self.playbook.entries.get(cls):
                continue
            d = dataclasses.replace(decision, failure_class=cls)
            for lever, _ in self.ranked(candidate, d, result, stats):
                if lever.name in tried or lever.name in names:
                    continue
                mechanical = lever.name in MECHANICAL or (
                    combo_parts(lever.name) and all(x in MECHANICAL for x in combo_parts(lever.name)))
                if not mechanical and not use_model:
                    notes.append(f"{lever.name}: skipped (model lever)")
                    continue
                out = RepairOutcome(candidate.candidate_id)
                child = self._try(candidate, lever, d, result, pool_expressions, events, out)
                notes += out.notes
                key = (child.expression, child.universe, child.decay, child.neutralization,
                       child.truncation) if child else None
                if child is None or key in seen:
                    continue
                seen.add(key)
                if self.memory.has_result_for_hash(experiment_hash(child)):
                    notes.append(f"{lever.name}: skipped (this experiment is already in memory)")
                    continue
                names.add(lever.name)
                children.append(dataclasses.replace(
                    child, candidate_id=self._free_id(child.candidate_id, children)))
                if limit is not None and len(children) >= limit:
                    return children, notes
        return children, notes

    def _free_id(self, base: str, taken: list[CandidateRecord]) -> str:
        """``base`` plus the first letter suffix not used in this batch or in memory."""
        used = {c.candidate_id for c in taken}
        for suffix in (a + b for a in ("", *string.ascii_lowercase)
                       for b in string.ascii_lowercase):
            cid = base + suffix
            if cid not in used and self.memory.get_candidate(cid) is None:
                return cid
        raise RuntimeError(f"no free candidate id for {base}")

    def _try(self, candidate, lever, decision, result, pool, events, out) -> CandidateRecord | None:
        try:
            repaired = self._apply(candidate, lever, decision, result, pool, events, out)
        except LeverNotApplicable as exc:
            out.notes.append(f"{lever.name}: not applicable ({exc})")
            return None
        child = self._child(candidate, repaired, lever.name)
        if child is None:
            out.notes.append(f"{lever.name}: produced an invalid candidate")
        return child

    def _combo(self, c: CandidateRecord, lever: Lever, result, out) -> CandidateRecord:
        applied, skipped = 0, []
        for part in combo_parts(lever.name):
            fn = MECHANICAL.get(part)
            if fn is None:
                raise LeverNotApplicable(f"combination part {part!r} is not a mechanical lever")
            step = lever.step.get(part)
            step = self.playbook.step_for(part) if step is None else step
            if step is None:
                raise LeverNotApplicable(f"no step for combination part {part!r}")
            try:
                c = fn(c, step, self.spec, result)
                applied += 1
            except LeverNotApplicable as exc:
                skipped.append(f"part {part}: {exc}")
        if not applied:
            raise LeverNotApplicable("; ".join(skipped))
        out.notes += [f"{lever.name}: skipped {x}" for x in skipped]
        return c

    def _apply(self, c: CandidateRecord, lever: Lever, decision, result, pool, events, out):
        if combo_parts(lever.name):
            return self._combo(c, lever, result, out)
        fn = MECHANICAL.get(lever.name)
        if fn is not None:
            return fn(c, lever.step, self.spec, result)
        if self.llm is None:
            raise LeverNotApplicable("needs the model, and no model is configured")
        out.notes.append(f"{lever.name}: {MODEL_LEVERS_NOTE}")
        pool = pool if lever.name == "orthogonalize_against_pool" else None
        examples = successful_examples(events, decision.failure_class,
                                       self.learning.examples_in_prompt, region=c.region)
        failing = sorted(failing_classes(decision, result) | {decision.failure_class})
        user = repair_user(c, lever, decision, _metrics(result), pool, self.spec.operators,
                           failing=failing, examples=examples)
        try:
            reply, calls = ask_json(self.llm, repair_system(self.context), user,
                                    max_output=self.max_output)
        except GenerationError as exc:
            out.llm_calls += 2
            raise LeverNotApplicable(f"model reply unusable: {exc}") from exc
        except ModelDown as exc:
            raise LeverNotApplicable(f"no model available: {exc}") from exc
        out.llm_calls += calls
        expr = reply.get("expression") if isinstance(reply, dict) else None
        if not isinstance(expr, str) or not expr.strip():
            raise LeverNotApplicable("model reply had no expression")
        a = analyze(expr.strip(), self.spec, datafields=self.datafields, limits=self.limits)
        if not a.ok:
            raise LeverNotApplicable("model expression failed validation: " + "; ".join(a.errors))
        changes = {"expression": expr.strip(), "model": getattr(self.llm, "model", c.model)}
        decay, neut = reply.get("decay"), reply.get("neutralization")
        if decay is not None:
            if isinstance(decay, int) and not isinstance(decay, bool) and 0 <= decay <= 512:
                changes["decay"] = decay
            else:
                out.notes.append(f"{lever.name}: ignored model decay {decay!r}")
        if neut is not None:
            if neut in neutralizations_for(c.region):
                changes["neutralization"] = neut
            else:
                out.notes.append(f"{lever.name}: ignored model neutralization {neut!r}")
        new = dataclasses.replace(c, **changes)
        if (new.expression, new.decay, new.neutralization) == (c.expression, c.decay,
                                                               c.neutralization):
            raise LeverNotApplicable("model returned the expression unchanged")
        return new

    def _child(self, parent: CandidateRecord, repaired: CandidateRecord,
               lever: str) -> CandidateRecord | None:
        child = assign_lineage(repaired, parent, lever=lever)
        child = dataclasses.replace(child, candidate_id=f"{parent.candidate_id}_r{child.generation}")
        try:
            child.validate()
        except SchemaError:
            return None
        a = analyze(child.expression, self.spec, datafields=self.datafields, limits=self.limits)
        return child if a.ok else None
