"""Critic and Debate judge: review candidates before a BRAIN simulation is spent.

Critic
1. Rule checks, no model:
   - blocking: a time-series operator with a window (2nd positional argument) of zero or
     less. A negative ``ts_delay`` reads the future; zero is meaningless.
   - advisory: the final expression is not cross-sectionally normalized (its outermost
     operator, ignoring a leading minus, is not rank, zscore, scale, normalize, quantile,
     winsorize or a group_* operator), or a window is longer than 512 days, or the
     neutralization is NONE (the alpha keeps a market bet; run 2's cash-flow alpha looked
     strong only because of it), or a fresh alpha starts from a factor neutralization (its
     reason is in the rationale). Advisory issues never block: they inform the reviews.
2. Model review (``critic.use_model``): one call per batch sees each candidate, its
   hypothesis and rationale, and the rule issues, and answers simulate, revise or drop,
   with issues and optionally a revised expression and, with revise, a neutralization
   allowed in the region (to undo a factor neutralization or NONE whose reason does not
   hold). A revision must pass
   ``fastexpr.analyze`` or it is discarded. If the model is off or its reply is unusable,
   the verdict comes from the rules: drop on a blocking issue, otherwise simulate.

Debate judge
Candidates the Critic did not clear (revise or drop) go to the judge, who weighs the
Generator's case (hypothesis and rationale) against the Critic's issues and rules simulate,
revise (only offered when a valid revision exists) or drop. One call per batch. Without the
model the Critic's verdict stands, except that revise becomes simulate when no valid
revision exists.

Repair children skip the model review and the judge (``review_and_debate``): they get the
rule checks only, so the lever learner sees exactly what the lever changed.

A revision becomes a new candidate ``<id>_c1`` with ``lever="critic_revision"``, the
original as parent and ``generation + 1``. The original is not simulated.

Generated text is untrusted: candidate fields are wrapped in <candidate> tags marked as data.
The prompts are in prompts.py (critic_system / critic_user, judge_system / judge_user).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Collection, Mapping

from .fastexpr import (Call, ComplexityLimits, FastExprError, FastExprSpec, Num, Unary,
                       analyze, catalog, catalog_for, parse)
from .generator import GenerationError, ask_json
from .llm import LLMClient, ModelDown
from .prompts import critic_system, critic_user, judge_system, judge_user
from .research_loop import assign_lineage
from .schemas import GROUP_NEUTRALIZATIONS, CandidateRecord, SchemaError, neutralizations_for

VERDICTS = ("simulate", "revise", "drop")
NORMALIZERS = {"rank", "zscore", "scale", "normalize", "quantile", "winsorize"}
LONG_WINDOW_DAYS = 512   # about two years of trading days; advisory only

@dataclass(frozen=True)
class Issue:
    severity: str   # "blocking" or "advisory"
    text: str


@dataclass
class Review:
    candidate_id: str
    verdict: str
    issues: list[str] = field(default_factory=list)
    revised: CandidateRecord | None = None
    source: str = "rules"            # "rules" or "model"
    blocking: bool = False           # a blocking rule issue: neither model may clear it


@dataclass
class Ruling:
    candidate_id: str
    ruling: str
    reason: str
    source: str = "rules"


@dataclass
class ReviewOutcome:
    """What goes on to simulation, and why the rest did not."""
    to_simulate: list[CandidateRecord] = field(default_factory=list)
    dropped: list[tuple[CandidateRecord, str]] = field(default_factory=list)
    reviews: dict[str, Review] = field(default_factory=dict)
    rulings: dict[str, Ruling] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    llm_calls: int = 0


# ---------------------------------------------------------------------------
# rule checks
# ---------------------------------------------------------------------------

def _window_value(node) -> float | None:
    if isinstance(node, Num):
        return float(node.text)
    if isinstance(node, Unary) and node.op == "-" and isinstance(node.operand, Num):
        return -float(node.operand.text)
    return None


def setting_issues(c: CandidateRecord) -> list[Issue]:
    if c.neutralization == "NONE":
        return [Issue("advisory", "neutralization NONE leaves a market bet in the alpha: its "
                                  "Sharpe may come from market direction, not stock selection")]
    if c.lever is None and c.neutralization not in GROUP_NEUTRALIZATIONS:
        # the Generator's own choice (a repair lever's is evidence-driven): its reason is in
        # the rationale for the Critic to weigh
        return [Issue("advisory", f"factor neutralization {c.neutralization} chosen at the "
                                  "start: check the author's reason; it can remove the signal "
                                  "itself when the signal is close to the factor")]
    return []


def rule_issues(expression: str) -> list[Issue]:
    try:
        stmts = parse(expression)
    except FastExprError as exc:
        return [Issue("blocking", f"does not parse: {exc}")]
    issues: list[Issue] = []

    def walk(n):
        if isinstance(n, Call):
            if n.name.startswith("ts_") and len(n.args) >= 2:
                w = _window_value(n.args[1])
                if w is not None and w <= 0:
                    issues.append(Issue("blocking", f"{n.name} has window {w:g}: a window must "
                                        "be positive (a negative delay reads future data)"))
                elif w is not None and w > LONG_WINDOW_DAYS:
                    issues.append(Issue("advisory", f"{n.name} window {w:g} is longer than about "
                                        "two years of trading days"))
            for a in n.args:
                walk(a)
            for _, v in n.kwargs:
                walk(v)
        for attr in ("operand", "left", "right", "value"):
            child = getattr(n, attr, None)
            if child is not None and not isinstance(child, str):
                walk(child)

    for s in stmts:
        walk(s)
    final = stmts[-1]
    while isinstance(final, Unary):
        final = final.operand
    if not (isinstance(final, Call) and (final.name in NORMALIZERS or final.name.startswith("group_"))):
        issues.append(Issue("advisory", "final expression is not cross-sectionally normalized "
                            "(no rank/zscore/scale/group_* at the top)"))
    return issues


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

class Critic:
    def __init__(self, spec: FastExprSpec, *, llm: LLMClient | None = None,
                 datafields: Collection[str] | None = None,
                 markets: Mapping[tuple, Mapping] | None = None,
                 limits: ComplexityLimits | None = None, max_output: int = 4096,
                 context: str = ""):
        self.spec = spec
        self.context = context  # prompts.brain_context(config)
        self.llm = llm
        self.datafields = catalog(datafields)  # {id: type}: types check vec_* use
        self.markets = markets  # {(region, delay): typed catalog}, ahead of datafields
        self.limits = limits
        self.max_output = max_output

    def revision(self, original: CandidateRecord, expression,
                 neutralization=None) -> CandidateRecord | None:
        """The revised candidate, or None. ``neutralization`` (optional) must be allowed in
        the region; a revision may change only the neutralization."""
        neut = original.neutralization
        if isinstance(neutralization, str) and neutralization.upper() in \
                neutralizations_for(original.region):
            neut = neutralization.upper()
        if not isinstance(expression, str) or not expression.strip():
            expression = original.expression
        expression = expression.strip()
        if expression == original.expression and neut == original.neutralization:
            return None
        fields = catalog_for(self.markets, self.datafields, original.region, original.delay)
        if not analyze(expression, self.spec, datafields=fields, limits=self.limits).ok:
            return None
        if any(i.severity == "blocking" for i in rule_issues(expression)):
            return None
        child = assign_lineage(dataclasses.replace(original, expression=expression,
                                                   neutralization=neut),
                               original, lever="critic_revision")
        child = dataclasses.replace(child, candidate_id=f"{original.candidate_id}_c1",
                                    model=getattr(self.llm, "model", original.model))
        try:
            child.validate()
        except SchemaError:
            return None
        return child

    def review(self, candidates: list[CandidateRecord], rationales: dict[str, str],
               out: ReviewOutcome, *, use_model: bool = True) -> dict[str, Review]:
        reviews: dict[str, Review] = {}
        for c in candidates:
            issues = rule_issues(c.expression) + setting_issues(c)
            blocking = any(i.severity == "blocking" for i in issues)
            reviews[c.candidate_id] = Review(c.candidate_id, "drop" if blocking else "simulate",
                                             [i.text for i in issues], blocking=blocking)
        if self.llm is None or not use_model or not candidates:
            return reviews
        user = critic_user(candidates, rationales,
                           {cid: r.issues for cid, r in reviews.items()}, self.spec.operators)
        try:
            reply, calls = ask_json(self.llm, critic_system(self.context), user,
                                    max_output=self.max_output)
        except GenerationError as exc:
            out.llm_calls += 2
            out.notes.append(f"critic model reply unusable, rule verdicts used: {exc}")
            return reviews
        except ModelDown as exc:
            out.notes.append(f"critic model down, rule verdicts used: {exc}")
            return reviews
        out.llm_calls += calls
        by_id = {c.candidate_id: c for c in candidates}
        items = reply.get("reviews") if isinstance(reply, dict) else reply
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or item.get("candidate_id") not in by_id:
                continue
            cid, verdict = item["candidate_id"], item.get("verdict")
            if verdict not in VERDICTS:
                continue
            base = reviews[cid]
            if base.blocking and verdict == "simulate":
                verdict = "drop"  # the model cannot clear a blocking rule issue
            issues = base.issues + _issue_texts(item.get("issues"))
            expected = item.get("expected_failure")
            if isinstance(expected, str) and expected.strip() and expected != "none":
                issues.append(f"expected to fail: {expected.strip()}")
            revised = self.revision(by_id[cid], item.get("revised_expression"),
                                    item.get("neutralization"))
            reviews[cid] = Review(cid, verdict, issues, revised, "model", base.blocking)
        return reviews


class DebateJudge:
    def __init__(self, *, llm: LLMClient | None = None, max_output: int = 2048,
                 context: str = ""):
        self.llm = llm
        self.max_output = max_output
        self.context = context  # prompts.brain_context(config)

    def judge(self, contested: list[CandidateRecord], reviews: dict[str, Review],
              rationales: dict[str, str], out: ReviewOutcome) -> dict[str, Ruling]:
        rulings = {}
        for c in contested:
            r = reviews[c.candidate_id]
            if r.verdict == "revise" and r.revised is None:
                rulings[c.candidate_id] = Ruling(c.candidate_id, "simulate",
                                                 "revision suggested but none was valid")
            else:
                rulings[c.candidate_id] = Ruling(c.candidate_id, r.verdict, "critic verdict stands")
        if self.llm is None or not contested:
            return rulings
        items = []
        for c in contested:
            r = reviews[c.candidate_id]
            items.append((c, rationales.get(c.candidate_id), r.issues, r.verdict,
                          r.revised.expression if r.revised else None))
        try:
            reply, calls = ask_json(self.llm, judge_system(self.context), judge_user(items),
                                    max_output=self.max_output)
        except GenerationError as exc:
            out.llm_calls += 2
            out.notes.append(f"judge model reply unusable, critic verdicts used: {exc}")
            return rulings
        except ModelDown as exc:
            out.notes.append(f"judge model down, critic verdicts used: {exc}")
            return rulings
        out.llm_calls += calls
        ids = {c.candidate_id for c in contested}
        items = reply.get("rulings") if isinstance(reply, dict) else reply
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or item.get("candidate_id") not in ids:
                continue
            cid, ruling = item["candidate_id"], item.get("ruling")
            if ruling not in VERDICTS:
                continue
            if ruling == "revise" and reviews[cid].revised is None:
                continue
            if ruling == "simulate" and reviews[cid].blocking:
                continue  # a blocking rule issue cannot be overruled
            reason = item.get("reason") or item.get("decisive_point") or ""
            rulings[cid] = Ruling(cid, ruling, str(reason)[:300], "model")
        return rulings


def _issue_texts(items) -> list[str]:
    """Model issues as strings: ``{"category", "detail"}`` objects or plain strings."""
    out = []
    for i in items if isinstance(items, list) else []:
        if isinstance(i, dict):
            detail = str(i.get("detail") or "").strip()
            if detail:
                cat = str(i.get("category") or "").strip()
                out.append(f"{cat}: {detail}" if cat else detail)
        elif isinstance(i, str) and i.strip():
            out.append(i.strip())
    return out


def review_and_debate(candidates: list[CandidateRecord], rationales: dict[str, str],
                      critic: Critic, judge: DebateJudge) -> ReviewOutcome:
    """Critic, then the judge on contested candidates. Returns what to simulate. Repair
    children (any candidate with a lever) get the rule checks only: a model revision on top
    of a repair would change two things at once and blur what the lever did."""
    out = ReviewOutcome()
    fresh = [c for c in candidates if c.lever is None]
    out.reviews = critic.review(fresh, rationales, out)
    out.reviews.update(critic.review([c for c in candidates if c.lever is not None],
                                     rationales, out, use_model=False))
    contested = [c for c in candidates if out.reviews[c.candidate_id].verdict != "simulate"]
    out.rulings = judge.judge(contested, out.reviews, rationales, out)
    for c in candidates:
        ruling = out.rulings.get(c.candidate_id)
        decision = "simulate" if ruling is None else ruling.ruling
        if decision == "simulate":
            out.to_simulate.append(c)
        elif decision == "revise":
            out.to_simulate.append(out.reviews[c.candidate_id].revised)
        else:
            reason = ruling.reason if ruling else "dropped"
            issues = "; ".join(out.reviews[c.candidate_id].issues)
            out.dropped.append((c, f"{reason} ({issues})" if issues else reason))
    return out
