"""Cheap Pre-Filter: decides which candidates are worth a BRAIN simulation.

BRAIN simulations are the scarce resource, so this runs before any is spent. It never
changes an accept or reject decision; it only removes candidates before simulation.

Checks, in order (the first that fails drops the candidate):
1. ``fastexpr.analyze`` accepts the expression (repairs and critic revisions are
   re-checked here too).
2. Exact duplicate: memory already has a result for the same experiment hash.
3. Near duplicate: the expression's term set (operators, names and numbers in the parsed
   tree) has Jaccard similarity >= ``near_duplicate_similarity`` with a candidate already
   in memory or earlier in the batch, under the same region, universe, delay and
   neutralization. ``null`` turns the check off. Repairs (candidates with a ``lever``)
   are exempt: they are deliberate variants of a known candidate, and the auditor counts
   them as variants.
4. Optional model score (``use_model``): one call scores the whole batch 0-10 for how
   likely each alpha is to pass BRAIN's tests. With ``min_model_score: null`` the score is
   only recorded; with a number, lower scores are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection, Mapping

from .fastexpr import (Assign, Binary, Call, ComplexityLimits, FastExprError, FastExprSpec,
                       Name, Num, Str, Unary, analyze, catalog, catalog_for, parse)
from .generator import GenerationError, ask_json
from .llm import LLMClient, ModelDown
from .memory import MemoryStore
from .prompts import prefilter_system, prefilter_user
from .schemas import CandidateRecord, experiment_hash

@dataclass(frozen=True)
class PrefilterConfig:
    near_duplicate_similarity: float | None = None
    use_model: bool = False
    min_model_score: float | None = None


@dataclass
class PrefilterResult:
    kept: list[CandidateRecord] = field(default_factory=list)
    dropped: list[tuple[CandidateRecord, str]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    llm_calls: int = 0


def terms(expression: str) -> frozenset[str]:
    """Operators, names and numbers in the parsed expression. Empty if it does not parse."""
    try:
        stmts = parse(expression)
    except FastExprError:
        return frozenset()
    out: set[str] = set()

    def walk(n):
        if isinstance(n, Call):
            out.add(n.name.lower() + "(")
            for a in n.args:
                walk(a)
            for k, v in n.kwargs:
                out.add(k + "=")
                walk(v)
        elif isinstance(n, Name):
            out.add(n.name)
        elif isinstance(n, Num):
            out.add(n.text)
        elif isinstance(n, Str):
            out.add(repr(n.value))
        elif isinstance(n, Unary):
            out.add("u" + n.op)
            walk(n.operand)
        elif isinstance(n, Binary):
            out.add(n.op)
            walk(n.left)
            walk(n.right)
        elif isinstance(n, Assign):
            walk(n.value)

    for s in stmts:
        walk(s)
    return frozenset(out)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _universe_key(c: CandidateRecord) -> tuple:
    return (c.region, c.universe, c.delay, c.neutralization)


class PreFilter:
    def __init__(self, config: PrefilterConfig, spec: FastExprSpec, memory: MemoryStore, *,
                 llm: LLMClient | None = None, datafields: Collection[str] | None = None,
                 markets: Mapping[tuple, Mapping] | None = None,
                 limits: ComplexityLimits | None = None, max_output: int = 2048,
                 context: str = ""):
        self.config = config
        self.context = context  # prompts.brain_context(config), shown to the model
        self.spec = spec
        self.memory = memory
        self.llm = llm
        self.datafields = catalog(datafields)  # {id: type}: types check vec_* use
        self.markets = markets  # {(region, delay): typed catalog}, ahead of datafields
        self.limits = limits
        self.max_output = max_output

    def run(self, candidates: list[CandidateRecord]) -> PrefilterResult:
        out = PrefilterResult()
        threshold = self.config.near_duplicate_similarity
        known: list[tuple[str, tuple, frozenset]] = []
        if threshold is None:
            out.notes.append("near-duplicate check not configured")
        else:
            for row in self.memory.candidate_outcomes():
                c = row["candidate"]
                known.append((c.candidate_id, _universe_key(c), terms(c.expression)))

        survivors = []
        for c in candidates:
            reason = self._static_reason(c, known, threshold)
            if reason:
                out.dropped.append((c, reason))
                continue
            survivors.append(c)
            if threshold is not None:
                known.append((c.candidate_id, _universe_key(c), terms(c.expression)))

        if survivors and self.config.use_model and self.llm is not None:
            self._score(survivors, out)
            floor = self.config.min_model_score
            if floor is not None:
                kept = []
                for c in survivors:
                    s = out.scores.get(c.candidate_id)
                    if s is not None and s < floor:
                        out.dropped.append((c, f"model score {s:g} below {floor:g}"))
                    else:
                        kept.append(c)
                survivors = kept
        out.kept = survivors
        return out

    def _static_reason(self, c: CandidateRecord, known, threshold) -> str | None:
        fields = catalog_for(self.markets, self.datafields, c.region, c.delay)
        a = analyze(c.expression, self.spec, datafields=fields, limits=self.limits)
        if not a.ok:
            return "invalid expression: " + "; ".join(a.errors)
        if self.memory.has_result_for_hash(experiment_hash(c)):
            return "duplicate of an experiment already simulated"
        if threshold is not None and c.lever is None:
            toks, key = terms(c.expression), _universe_key(c)
            for other_id, other_key, other_toks in known:
                if other_id != c.candidate_id and other_key == key:
                    sim = jaccard(toks, other_toks)
                    if sim >= threshold:
                        return f"near duplicate of {other_id} (term similarity {sim:.2f})"
        return None

    def _score(self, cands: list[CandidateRecord], out: PrefilterResult) -> None:
        try:
            reply, calls = ask_json(self.llm, prefilter_system(self.context),
                                    prefilter_user(cands), max_output=self.max_output)
        except GenerationError as exc:
            out.llm_calls += 2
            out.notes.append(f"model scoring skipped: {exc}")
            return
        except ModelDown as exc:
            out.notes.append(f"model scoring skipped: {exc}")
            return
        out.llm_calls += calls
        items = reply.get("scores") if isinstance(reply, dict) else reply
        ids = {c.candidate_id for c in cands}
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or item.get("candidate_id") not in ids:
                continue
            s = item.get("score")
            if isinstance(s, (int, float)) and not isinstance(s, bool) and 0 <= s <= 10:
                out.scores[item["candidate_id"]] = float(s)
        missing = ids - set(out.scores)
        if missing:
            out.notes.append(f"model gave no usable score for {sorted(missing)}")
