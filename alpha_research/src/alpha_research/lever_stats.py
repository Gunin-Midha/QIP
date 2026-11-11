"""Lever learning: which repair levers actually fixed which failures, learned from memory.

The playbook (config/repair_playbook.yaml) lists the levers worth trying for each failure
class. This module decides the order to try them in, from evidence. Nothing new is stored:
every repair already leaves a parent -> child pair in memory (``parent_id`` and ``lever``
on the child, a decision and result for each), so the evidence grows by itself with every
simulated repair, including hand-written rescue candidates that set ``parent_id`` and
``lever``. That is what makes the Repair Agent self-updating without editing config.

Method
1. A repair event is a child candidate with a lever whose parent and child both have a
   decision. Its target is the parent's failure class.
2. Reward, in [0, 1], per failure class = 0.5 x class part + 0.5 x overall part.
     class part    1.0 the child is accepted or the class no longer fails; 0.5 the class
                   still fails but its slack improved (e.g. Sharpe 1.14 -> 1.21 against
                   1.25); 0.0 otherwise
     overall part  1.0 if the child is accepted, else the share by which the child's
                   distance to accept (``distance``) fell below the parent's, in [0, 1]
   A child that did not complete (BRAIN rejected the rewrite) gets 0. The overall part is
   what stops the search from oscillating: fixing fitness while breaking sub-universe
   earns little unless the alpha as a whole moved closer to passing.
   A class "fails" when one of its gates failed in the decision or BRAIN's own check for
   it failed.
   Distance to accept = sum over the classes the alpha fails of the largest normalized
   shortfall among that class's gates ((threshold - value) / |threshold|, the other way
   round for limits where lower is better, the distance outside the range for turnover);
   a failing class with no measured value (BRAIN's CONCENTRATED_WEIGHT on a PASS-less
   report, other BRAIN checks) adds ``unmeasured_failure_penalty``. 0 = accepted.
3. One child gives several weighted pieces of evidence (``role``):
     target     (target class, lever), weight 1
     secondary  (each other class the parent failed, lever), weight ``secondary_weight``,
                so a lever aimed at sub-universe that also cleared concentration gets credit
     part       a combination ``combo:a+b+c`` also credits each part a, b, c with weight
                1/3 of the above, so the playbook's single levers learn from combinations
4. Per (failure class, lever), the weighted rewards update a Beta(prior_a, prior_b)
   posterior: a = prior_a + sum(w x reward), b = prior_b + sum(w x (1 - reward)).
   Per region (the child's), the same evidence also builds a region posterior whose prior
   is the all-region estimate m worth ``region_prior_strength`` (k) events:
   a = k x m + sum(w x reward), b = k x (1 - m) + sum(w x (1 - reward)). A region with no
   evidence ranks like all regions together; as its own repairs accumulate they take over,
   so a lever that works in USA is a starting guess in EUR, not a rule. Ranking and the
   past repairs shown to the model use the alpha's region.
5. Family budget (``family_budget``), so repairs never concentrate on the few alphas that
   came close: a lineage (an original candidate and all its repair descendants) may be
   repaired only while the next child's generation is at most ``max_depth``, fewer than
   ``max_trials_per_root`` descendants reached BRAIN, and the last ``patience``
   descendants did not all fail to beat the lineage's best distance so far. Every repair
   simulation also counts as a trial against the accepted alpha in the overfitting check
   (auditor.py), so the budget doubles as an overfitting guard.
6. Ranking (``rank_levers``). Candidates are the primary class's playbook levers plus the
   learned combinations: every ``combo:...`` lever with positive evidence for that class
   (``learned_combos``), so a combination that worked becomes a lever the agent can
   choose. A lever's score is the sum, over every class the alpha currently fails that
   lists the lever, of weight x estimate, with weight 1 for the primary class and
   ``secondary_weight`` for the others, so a lever that addresses two failures comes first.
   The estimate is the posterior mean
   (``order: learned``) or one posterior draw (``order: thompson``, explores more). Ties
   keep playbook order, so with no evidence the playbook order is used, and
   ``order: playbook`` turns learning off.
   Two levers are placed by rule, not score: ``flip_sign`` first (it applies only when
   Sharpe is negative, and then a near miss of -1.1 becomes +1.1, the most direct fix there
   is) and ``model_repair`` last (the catch-all after the mechanical levers; it costs a
   model call, its evidence mixes whatever the model wrote, and since every class lists it,
   the secondary bonus alone put it first on alphas failing several checks, as run
   rs20261104073248 showed).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from .config import ConfigError
from .evaluator import BRAIN_CHECK_TO_CLASS, check_class
from .repair import Lever, Playbook
from .research_loop import lineage_members, lineage_root, simulated
from .schemas import CandidateRecord, DecisionRecord, ResultRecord

ORDERS = ("learned", "thompson", "playbook")

# gates where a lower value is better
_LOWER_IS_BETTER = {"self_correlation", "weight_concentration", "drawdown_depth",
                    "pool_correlation"}


@dataclass(frozen=True)
class RepairLearningConfig:
    order: str = "learned"
    secondary_weight: float = 0.5
    prior: tuple[float, float] = (1.0, 1.0)
    examples_in_prompt: int = 3          # successful past repairs shown to the repair model
    resume_from_memory: bool = True      # re-repair parents whose child was never simulated
    max_resume: int = 3                  # at most this many, closest to accept first
    max_share_per_round: float = 0.5     # share of a round's slots queued repairs may take first
    max_depth: int = 2                   # repair generations below the original candidate
    max_trials_per_root: int = 4         # repair simulations per original candidate
    patience: int = 2                    # stop a lineage after this many non-improving children
    unmeasured_failure_penalty: float = 0.5   # distance added by a failure with no value
    region_prior_strength: float = 2.0   # events the all-region estimate counts for in a region

    @classmethod
    def from_dict(cls, sec: dict) -> "RepairLearningConfig":
        """The ``repair`` section of config/agents.yaml. Every key is optional."""
        d = cls()
        order = sec.get("order", d.order)
        if order not in ORDERS:
            raise ConfigError(f"agents config: repair.order must be one of {list(ORDERS)}")

        def num(key, lo, hi):
            v = sec.get(key, getattr(d, key))
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ConfigError(f"agents config: repair.{key} must be in [{lo}, {hi}]")
            return v

        prior = sec.get("prior", list(d.prior))
        if (not isinstance(prior, (list, tuple)) or len(prior) != 2
                or any(isinstance(p, bool) or not isinstance(p, (int, float)) or p <= 0
                       for p in prior)):
            raise ConfigError("agents config: repair.prior must be two positive numbers [a, b]")
        resume = sec.get("resume_from_memory", d.resume_from_memory)
        if not isinstance(resume, bool):
            raise ConfigError("agents config: repair.resume_from_memory must be true or false")
        examples = sec.get("examples_in_prompt", d.examples_in_prompt)
        if isinstance(examples, bool) or not isinstance(examples, int) or examples < 0:
            raise ConfigError("agents config: repair.examples_in_prompt must be an integer >= 0")
        def count(key, lo):
            v = sec.get(key, getattr(d, key))
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ConfigError(f"agents config: repair.{key} must be an integer >= {lo}")
            return v

        return cls(order=order, secondary_weight=float(num("secondary_weight", 0, 1)),
                   prior=(float(prior[0]), float(prior[1])), examples_in_prompt=examples,
                   resume_from_memory=resume, max_resume=count("max_resume", 0),
                   max_share_per_round=float(num("max_share_per_round", 0, 1)),
                   max_depth=count("max_depth", 1),
                   max_trials_per_root=count("max_trials_per_root", 1),
                   patience=count("patience", 1),
                   unmeasured_failure_penalty=float(num("unmeasured_failure_penalty", 0, 10)),
                   region_prior_strength=float(num("region_prior_strength", 0, 1000)))


# ---------------------------------------------------------------------------
# failing classes and slack
# ---------------------------------------------------------------------------

def failing_classes(decision: DecisionRecord | None, result: ResultRecord | None) -> set[str]:
    """Every failure class the alpha fails: failed gates and risk checks in the decision,
    plus BRAIN's own failed checks."""
    out: set[str] = set()
    if decision is not None:
        out |= {check_class(g.name) for g in decision.gate_results if g.passed is False}
    if result is not None and result.checks:
        out |= {BRAIN_CHECK_TO_CLASS[k] for k, ok in result.checks.items()
                if ok is False and k in BRAIN_CHECK_TO_CLASS}
    return out


def _gate_slack(name: str, value, threshold) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if name == "turnover_range":
        if not isinstance(threshold, (list, tuple)) or None in threshold:
            return None
        return min(value - threshold[0], threshold[1] - value)
    if name in _LOWER_IS_BETTER:
        return (threshold - value) if isinstance(threshold, (int, float)) else -value
    return (value - threshold) if isinstance(threshold, (int, float)) else None


def slack(failure_class: str, decision: DecisionRecord | None,
          result: ResultRecord | None) -> float | None:
    """Smallest margin to passing over the gates of a class (negative = failing). For
    concentration with no configured limit, BRAIN's reported value (lower is better)."""
    vals = []
    if decision is not None:
        for g in decision.gate_results:
            if check_class(g.name) == failure_class:
                s = _gate_slack(g.name, g.value, g.threshold)
                if s is not None:
                    vals.append(s)
    if not vals and failure_class == "concentration" and result is not None \
            and result.weight_concentration is not None:
        vals.append(-result.weight_concentration)
    return min(vals) if vals else None


def _shortfall(name: str, value, threshold) -> float | None:
    """Normalized amount by which a failed gate misses its threshold, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if name.startswith("brain:"):
        name = {"brain:LOW_SUB_UNIVERSE_SHARPE": "sub_universe"}.get(name, name)
    if name == "turnover_range":
        if not isinstance(threshold, (list, tuple)) or None in threshold:
            return None
        lo, hi = threshold
        if value < lo:
            return (lo - value) / lo if lo else None
        return (value - hi) / hi if value > hi and hi else None
    if not isinstance(threshold, (int, float)) or threshold == 0:
        return None
    gap = (value - threshold) if name in _LOWER_IS_BETTER else (threshold - value)
    return max(gap, 0.0) / abs(threshold)


def distance(decision: DecisionRecord | None, result: ResultRecord | None,
             unmeasured_penalty: float = 0.5) -> float | None:
    """Distance to accept (module doc, step 2). 0 for an accept; None when the result is
    missing or incomplete."""
    if decision is None or result is None or result.status != "complete":
        return None
    if decision.decision == "accept":
        return 0.0
    total = 0.0
    for cls in failing_classes(decision, result):
        gaps = [x for g in decision.gate_results
                if g.passed is False and check_class(g.name) == cls
                and (x := _shortfall(g.name, g.value, g.threshold)) is not None]
        total += max(gaps) if gaps else unmeasured_penalty
    return round(total, 6)


# ---------------------------------------------------------------------------
# events and statistics
# ---------------------------------------------------------------------------

COMBO_PREFIX = "combo:"


def combo_parts(lever: str) -> list[str]:
    """The levers of ``combo:a+b+c`` in order; [] for a single lever."""
    return lever[len(COMBO_PREFIX):].split("+") if lever.startswith(COMBO_PREFIX) else []


@dataclass(frozen=True)
class RepairEvent:
    parent_id: str
    child_id: str
    failure_class: str
    lever: str
    reward: float
    fixed: bool
    improved: bool
    collateral: tuple[str, ...]
    slack_before: float | None
    slack_after: float | None
    parent_expression: str
    child_expression: str
    settings_change: dict           # setting -> [parent value, child value], only changed ones
    role: str = "target"            # target | secondary | part
    weight: float = 1.0
    region: str | None = None       # the child's region


_SETTINGS = ("universe", "decay", "neutralization", "truncation")


def _outcome(cls, dp, rp, dc, rc, complete, fails_now, overall):
    before, after = slack(cls, dp, rp), slack(cls, dc, rc)
    fixed = complete and (dc.decision == "accept" or cls not in fails_now)
    improved = (complete and not fixed and before is not None and after is not None
                and after > before)
    part = 1.0 if fixed else 0.5 if improved else 0.0
    reward = 0.5 * part + 0.5 * overall if complete else 0.0
    return round(reward, 6), fixed, improved, before, after


def _overall_gain(dp, rp, dc, rc, penalty: float) -> float:
    if dc is not None and dc.decision == "accept":
        return 1.0
    before, after = distance(dp, rp, penalty), distance(dc, rc, penalty)
    if not before or after is None:
        return 0.0
    return min(1.0, max(0.0, (before - after) / before))


def repair_events(outcomes: Iterable[dict],
                  config: RepairLearningConfig | None = None) -> list[RepairEvent]:
    """Weighted repair evidence from ``MemoryStore.candidate_outcomes()`` rows, oldest
    first (see module doc, steps 2-3)."""
    config = config or RepairLearningConfig()
    rows = list(outcomes)
    by_id = {r["candidate"].candidate_id: r for r in rows}
    events = []
    for row in rows:
        c, dc, rc = row["candidate"], row["decision"], row["result"]
        if not c.parent_id or not c.lever or dc is None:
            continue
        parent = by_id.get(c.parent_id)
        if parent is None or parent["decision"] is None:
            continue
        p, dp, rp = parent["candidate"], parent["decision"], parent["result"]
        target = dp.failure_class
        if target in ("none", "incomplete"):
            continue
        complete = rc is not None and rc.status == "complete"
        fails_now = failing_classes(dc, rc)
        parent_fails = failing_classes(dp, rp)
        collateral = tuple(sorted(fails_now - parent_fails)) if complete else ()
        overall = _overall_gain(dp, rp, dc, rc, config.unmeasured_failure_penalty)
        change = {k: [getattr(p, k), getattr(c, k)] for k in _SETTINGS
                  if getattr(p, k) != getattr(c, k)}
        classes = [(target, "target", 1.0)] + [
            (cls, "secondary", config.secondary_weight) for cls in sorted(parent_fails - {target})]
        parts = combo_parts(c.lever)
        for cls, role, w in classes:
            if w <= 0:
                continue
            reward, fixed, improved, before, after = _outcome(
                cls, dp, rp, dc, rc, complete, fails_now, overall)
            base = dict(parent_id=p.candidate_id, child_id=c.candidate_id, failure_class=cls,
                        reward=reward, fixed=fixed, improved=improved, collateral=collateral,
                        slack_before=before, slack_after=after,
                        parent_expression=p.expression, child_expression=c.expression,
                        settings_change=change, region=c.region)
            events.append(RepairEvent(lever=c.lever, role=role, weight=w, **base))
            for part in parts:
                events.append(RepairEvent(lever=part, role="part", weight=w / len(parts),
                                          **base))
    return events


def family_budget(candidate: CandidateRecord, outcomes: list[dict],
                  config: RepairLearningConfig) -> tuple[int, str | None]:
    """(repair simulations still allowed for the candidate's lineage, reason when 0)."""
    if candidate.generation + 1 > config.max_depth:
        return 0, (f"repair depth limit: a child would be generation {candidate.generation + 1} "
                   f"> max_depth {config.max_depth}")
    cands = {r["candidate"].candidate_id: r["candidate"] for r in outcomes}
    root = lineage_root(candidate.candidate_id, cands)
    members = lineage_members(outcomes, root)
    tried = [r for r in members if r["candidate"].candidate_id != root and simulated(r)]
    if len(tried) >= config.max_trials_per_root:
        return 0, (f"lineage {root} used its repair budget: {len(tried)} of "
                   f"{config.max_trials_per_root} repair simulations")
    root_row = next((r for r in members if r["candidate"].candidate_id == root), None)
    best = distance(root_row["decision"], root_row["result"],
                    config.unmeasured_failure_penalty) if root_row else None
    stale = 0
    for r in tried:
        d = distance(r["decision"], r["result"], config.unmeasured_failure_penalty)
        if d is not None and (best is None or d < best):
            best, stale = d, 0
        else:
            stale += 1
    if stale >= config.patience:
        return 0, (f"lineage {root} stopped improving: the last {stale} repair(s) did not "
                   f"beat its best distance to accept ({best})")
    return config.max_trials_per_root - len(tried), None


@dataclass
class LeverRecord:
    n: float = 0.0          # weighted amount of evidence
    reward: float = 0.0     # weighted reward
    events: int = 0
    fixed: int = 0
    improved: int = 0


@dataclass
class LeverStats:
    prior: tuple[float, float] = (1.0, 1.0)
    records: dict = field(default_factory=dict)   # (failure class, lever) -> LeverRecord
    by_region: dict = field(default_factory=dict)  # (region, failure class, lever) -> LeverRecord
    region_strength: float = 2.0

    @classmethod
    def from_events(cls, events: Iterable[RepairEvent], prior: tuple[float, float] = (1.0, 1.0),
                    region_strength: float = 2.0) -> "LeverStats":
        stats = cls(prior, region_strength=region_strength)
        for e in events:
            keys = [(stats.records, (e.failure_class, e.lever))]
            if e.region:
                keys.append((stats.by_region, (e.region, e.failure_class, e.lever)))
            for table, key in keys:
                rec = table.setdefault(key, LeverRecord())
                rec.n += e.weight
                rec.reward += e.weight * e.reward
                rec.events += 1
                rec.fixed += e.fixed
                rec.improved += e.improved
        return stats

    def posterior(self, failure_class: str, lever: str,
                  region: str | None = None) -> tuple[float, float]:
        rec = self.records.get((failure_class, lever), LeverRecord())
        a, b = self.prior[0] + rec.reward, self.prior[1] + rec.n - rec.reward
        own = self.by_region.get((region, failure_class, lever)) if region else None
        if own is None:
            return a, b
        m, k = a / (a + b), self.region_strength
        # floor keeps a valid Beta when k = 0 and the region's evidence is all 0 or all 1
        return max(k * m + own.reward, 1e-6), max(k * (1 - m) + own.n - own.reward, 1e-6)

    def mean(self, failure_class: str, lever: str, region: str | None = None) -> float:
        a, b = self.posterior(failure_class, lever, region)
        return a / (a + b)

    def learned_combos(self, failure_class: str) -> list[Lever]:
        """Combinations with positive evidence for this class, best first."""
        found = [(self.mean(c, lv), lv) for (c, lv), r in self.records.items()
                 if c == failure_class and combo_parts(lv) and r.reward > 0]
        return [Lever(lv) for _, lv in sorted(found, key=lambda t: (-t[0], t[1]))]

    def table(self) -> list[dict]:
        """One row per (class, lever) with evidence, best first within a class."""
        rows = [{"failure_class": cls, "lever": lever, "events": r.events,
                 "evidence_weight": round(r.n, 3), "fixed": r.fixed, "improved": r.improved,
                 "mean_reward": round(r.reward / r.n, 3) if r.n else None,
                 "posterior_mean": round(self.mean(cls, lever), 3)}
                for (cls, lever), r in self.records.items()]
        return sorted(rows, key=lambda x: (x["failure_class"], -x["posterior_mean"]))

    def region_table(self) -> list[dict]:
        """Like ``table`` per (region, class, lever), with the region's shrunk estimate."""
        rows = [{"region": reg, "failure_class": cls, "lever": lever, "events": r.events,
                 "evidence_weight": round(r.n, 3), "fixed": r.fixed, "improved": r.improved,
                 "mean_reward": round(r.reward / r.n, 3) if r.n else None,
                 "posterior_mean": round(self.mean(cls, lever, reg), 3)}
                for (reg, cls, lever), r in self.by_region.items()]
        return sorted(rows, key=lambda x: (x["region"], x["failure_class"], -x["posterior_mean"]))


def rank_levers(playbook: Playbook, primary: str, failing: Iterable[str], stats: LeverStats,
                config: RepairLearningConfig, rng: random.Random | None = None,
                region: str | None = None) -> list[tuple[Lever, float]]:
    """The primary class's levers plus its learned combinations, best first, each with its
    score (see module doc), using the region's estimates when ``region`` is given."""
    if config.order == "playbook":
        return _pin([(lv, 0.0) for lv in playbook.entries.get(primary, [])])

    def options(cls: str) -> list[Lever]:
        listed = list(playbook.entries.get(cls, []))
        names = {lv.name for lv in listed}
        return listed + [lv for lv in stats.learned_combos(cls) if lv.name not in names]

    rng = rng or random.Random(0)
    weights = {primary: 1.0}
    for cls in failing:
        if cls != primary and cls in playbook.entries:
            weights[cls] = config.secondary_weight
    names_by_class = {cls: {lv.name for lv in options(cls)} for cls in weights}
    scored = []
    for pos, lv in enumerate(options(primary)):
        score = 0.0
        for cls, w in weights.items():
            if lv.name not in names_by_class[cls]:
                continue
            if config.order == "thompson":
                score += w * rng.betavariate(*stats.posterior(cls, lv.name, region))
            else:
                score += w * stats.mean(cls, lv.name, region)
        scored.append((-score, pos, lv))
    scored.sort(key=lambda t: (t[0], t[1]))
    return _pin([(lv, round(-s, 4)) for s, _, lv in scored])


PINNED_FIRST = ("flip_sign",)
PINNED_LAST = ("model_repair",)


def _pin(ranked: list[tuple[Lever, float]]) -> list[tuple[Lever, float]]:
    """Move the rule-placed levers to the front and back (module doc, point 6)."""
    first = [x for x in ranked if x[0].name in PINNED_FIRST]
    last = [x for x in ranked if x[0].name in PINNED_LAST]
    middle = [x for x in ranked if x[0].name not in PINNED_FIRST + PINNED_LAST]
    return first + middle + last


def successful_examples(events: Iterable[RepairEvent], failure_class: str,
                        limit: int, region: str | None = None) -> list[dict]:
    """The best past repairs of this class (reward >= 0.5) for the prompt: the region's
    own first, then by reward, newest first among equals."""
    good = [e for e in reversed(list(events))
            if e.failure_class == failure_class and e.reward >= 0.5 and e.role == "target"]
    good.sort(key=lambda e: (region is not None and e.region != region, -e.reward))  # stable
    return [{"lever": e.lever, "region": e.region, "before": e.parent_expression,
             "after": e.child_expression, "settings_change": e.settings_change,
             "result": "fixed" if e.fixed else "improved"} for e in good[:limit]]
