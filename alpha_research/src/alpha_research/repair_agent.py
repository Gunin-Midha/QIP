"""Repair Agent: applies a playbook lever to a candidate and returns the repaired child.

``repair.py`` only says which lever comes next. This module carries it out.

Method
1. Levers come from the playbook for the decision's failure class, in playbook order,
   skipping any lever already applied along the parent chain (``MemoryStore.lineage_levers``).
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
   ==================  ===========================================================

   Only the final statement is rewritten; ``name = ...;`` statements are kept.
4. Any other lever (orthogonalize_against_pool, restrict_to_liquid_fields, winsorize, or
   one added to the playbook later) goes to the model with the metrics, the lever and,
   for orthogonalization, the pool's expressions. Its reply passes through
   ``fastexpr.analyze`` like everything the Generator writes; an invalid reply means the
   lever does not apply. The prompts (with per-lever instructions) are in prompts.py.
5. The child gets ``parent_id``, ``generation + 1`` and ``lever`` (``assign_lineage``) and
   the id ``<parent id>_r<generation>``. Exactly one lever per repair.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Collection

from .fastexpr import (Assign, Binary, Call, ComplexityLimits, FastExprError, FastExprSpec,
                       Name, Num, Unary, analyze, parse, render)
from .generator import GenerationError, ask_json
from .llm import LLMClient
from .memory import MemoryStore
from .prompts import repair_system, repair_user
from .repair import Lever, Playbook
from .research_loop import assign_lineage
from .schemas import CandidateRecord, DecisionRecord, ResultRecord, SchemaError

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


def _increase_decay(c, step, spec):
    to = step.get("to")
    if not isinstance(to, int) or isinstance(to, bool):
        raise LeverNotApplicable("increase_decay needs an integer step.to")
    if c.decay >= to:
        raise LeverNotApplicable(f"decay is already {c.decay} (target {to})")
    return dataclasses.replace(c, decay=to)


def _tighten_truncation(c, step, spec):
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


def _lengthen_ts_window(c, step, spec):
    factor = step.get("factor")
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 2:
        raise LeverNotApplicable("lengthen_ts_window needs an integer step.factor >= 2")
    stmts = parse(c.expression)
    changed: list = []
    new = tuple(_lengthen(s, factor, changed) for s in stmts)
    if not changed:
        raise LeverNotApplicable("no integer time-series window to lengthen")
    return dataclasses.replace(c, expression=render(new))


def _wrap_in_smoothing(c, step, spec):
    op, window = step.get("operator", "ts_mean"), step.get("window")
    _need_op(spec, op)
    return _with_final(c, Call(op, (_final(c)[1], _num(window)), ()))


def _hump(c, step, spec):
    _need_op(spec, "hump")
    return _with_final(c, Call("hump", (_final(c)[1],), (("hump", _num(step.get("threshold"))),)))


def _rank_transform(c, step, spec):
    _need_op(spec, "rank")
    final = _final(c)[1]
    if isinstance(final, Call) and final.name == "rank":
        raise LeverNotApplicable("expression is already a rank(...)")
    return _with_final(c, Call("rank", (final,), ()))


def _volatility_scale(c, step, spec):
    _need_op(spec, "ts_std_dev")
    if "returns" not in spec.base_fields:
        raise LeverNotApplicable("'returns' is not a base field")
    vol = Call("ts_std_dev", (Name("returns"), _num(step.get("window"))), ())
    return _with_final(c, Binary("/", _final(c)[1], vol))


MECHANICAL = {
    "increase_decay": _increase_decay,
    "tighten_truncation": _tighten_truncation,
    "lengthen_ts_window": _lengthen_ts_window,
    "wrap_in_smoothing": _wrap_in_smoothing,
    "hump": _hump,
    "rank_transform": _rank_transform,
    "volatility_scale": _volatility_scale,
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
    return {k: getattr(result, k) for k in keys if getattr(result, k) is not None}


class RepairAgent:
    def __init__(self, playbook: Playbook, spec: FastExprSpec, memory: MemoryStore, *,
                 llm: LLMClient | None = None, datafields: Collection[str] | None = None,
                 limits: ComplexityLimits | None = None, max_output: int = 2048,
                 context: str = ""):
        self.playbook = playbook
        self.context = context  # prompts.brain_context(config)
        self.spec = spec
        self.memory = memory
        self.llm = llm
        self.datafields = None if datafields is None else set(datafields)
        self.limits = limits
        self.max_output = max_output

    def repair(self, candidate: CandidateRecord, decision: DecisionRecord,
               result: ResultRecord | None = None,
               pool_expressions: list[str] | None = None) -> RepairOutcome:
        out = RepairOutcome(candidate.candidate_id)
        levers = self.playbook.entries.get(decision.failure_class, [])
        if not levers:
            out.notes.append(f"no playbook entry for {decision.failure_class!r}")
            return out
        tried = set(self.memory.lineage_levers(candidate.candidate_id))
        for lever in levers:
            if lever.name in tried:
                continue
            try:
                repaired = self._apply(candidate, lever, decision, result, pool_expressions, out)
            except LeverNotApplicable as exc:
                out.notes.append(f"{lever.name}: not applicable ({exc})")
                continue
            child = self._child(candidate, repaired, lever.name)
            if child is None:
                out.notes.append(f"{lever.name}: produced an invalid candidate")
                continue
            out.child, out.lever = child, lever.name
            out.notes.append(f"{lever.name}: applied")
            return out
        out.notes.append("repair exhausted: every lever for this failure class was tried or "
                         "did not apply")
        return out

    def _apply(self, c: CandidateRecord, lever: Lever, decision, result, pool, out):
        fn = MECHANICAL.get(lever.name)
        if fn is not None:
            return fn(c, lever.step, self.spec)
        if self.llm is None:
            raise LeverNotApplicable("needs the model, and no model is configured")
        out.notes.append(f"{lever.name}: {MODEL_LEVERS_NOTE}")
        pool = pool if lever.name == "orthogonalize_against_pool" else None
        user = repair_user(c, lever, decision, _metrics(result), pool, self.spec.operators)
        try:
            reply, calls = ask_json(self.llm, repair_system(self.context), user,
                                    max_output=self.max_output)
        except GenerationError as exc:
            out.llm_calls += 2
            raise LeverNotApplicable(f"model reply unusable: {exc}") from exc
        out.llm_calls += calls
        expr = reply.get("expression") if isinstance(reply, dict) else None
        if not isinstance(expr, str) or not expr.strip():
            raise LeverNotApplicable("model reply had no expression")
        a = analyze(expr.strip(), self.spec, datafields=self.datafields, limits=self.limits)
        if not a.ok:
            raise LeverNotApplicable("model expression failed validation: " + "; ".join(a.errors))
        if expr.strip() == c.expression:
            raise LeverNotApplicable("model returned the expression unchanged")
        return dataclasses.replace(c, expression=expr.strip(),
                                   model=getattr(self.llm, "model", c.model))

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
