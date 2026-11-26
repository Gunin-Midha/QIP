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
     class part    1.0 the child is accepted or the class no longer fails; when the class
                   still fails but its slack improved, the share of the gap it closed
                   (Sharpe 1.20 -> 1.50 against 1.58 closes 0.30 of 0.38: 0.79), so a
                   large gain that still misses the bar counts almost as a fix and a tiny
                   one almost as nothing (owner 2026-11-23: the same change may carry an
                   alpha that needs less); 0.0 otherwise
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
   Gain match (owner 2026-11-23): each (class, lever) also keeps its past gains in slack
   (child minus parent, in the gate's own units, e.g. +0.35 Sharpe). When the alpha being
   repaired misses a class by ``need``, the lever's estimate for that class becomes
   (1 - gain_weight) x posterior mean + gain_weight x P(gain >= need), with
   P = (gains >= need + 1) / (gains + 2), once it has ``min_gain_evidence`` gains. A lever
   whose +0.35 never carried its alphas over the bar then ranks first for an alpha that
   needs +0.10.
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
from .evaluator import BRAIN_CHECK_TO_CLASS, SIGN_FIRST, check_class
from .repair import Lever, Playbook
from .research_loop import lineage_members, lineage_root, simulated
from .schemas import CandidateRecord, DecisionRecord, ResultRecord

ORDERS = ("learned", "thompson", "playbook")

# gates where a lower value is better
_LOWER_IS_BETTER = {"self_correlation", "weight_concentration", "drawdown_depth",
                    "pool_correlation"}


def _lower_is_better(name: str) -> bool:
    """Gates where a smaller value is better: ours above, and BRAIN's correlation checks
    (brain:PROD_CORRELATION, brain:SELF_CORRELATION, brain:POWER_POOL_CORRELATION). Until
    2026-11-26 those BRAIN checks were scored as higher-is-better, so a production
    correlation of 0.93 and one of 0.71 both counted as distance 0 to accept."""
    return name in _LOWER_IS_BETTER or (name.startswith("brain:")
                                        and name.endswith("CORRELATION"))


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
    gain_weight: float = 0.5             # weight of P(past gain >= needed gain) in a lever's estimate
    min_gain_evidence: int = 2           # gains a (class, lever) needs before the match is used
    fixes_per_wave: int = 4              # repair children written and simulated together
    waves_per_round: int = 2             # repair waves a research round may start
    fill_freed_slots: bool = True        # start ready waves in BRAIN slots freed mid-batch
    model_levers_per_wave: int = 1       # model-written fixes per wave (model calls are scarce)
    # wave size by how close and how real the alpha is (owner 2026-11-23: many fixes only
    # for alphas that are really close and have value). ((max distance or None, fixes), ...)
    wave_tiers: tuple = ()               # empty: fixes_per_wave for every alpha
    value_sharpe_share: float | None = None   # below this share of the Sharpe bar: low_value_fixes
    low_value_fixes: int = 1
    blocked_fixes: int = 1               # fails a BRAIN check no playbook lever addresses
    # leads (owner 2026-11-25): best Sharpe (raw or risk-neutralized, either sign) at least
    # lead_sharpe_share x the Sharpe bar gets at least lead_fixes mechanical fixes, however
    # far its other checks put it (run rs20261125130750: 43 simulations, leads at Sharpe
    # 1.05-1.39, no repair child at all). None = off.
    lead_sharpe_share: float | None = None
    lead_fixes: int = 0
    unmeasured_failure_penalty: float = 0.5   # distance added by a failure with no value
    # part tests (legs.py, owner 2026-11-26): a wave of a combined alpha also simulates each
    # of its parts alone (at most this many parts; 0 = off), once per lineage
    leg_tests: int = 0
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
                   gain_weight=float(num("gain_weight", 0, 1)),
                   min_gain_evidence=count("min_gain_evidence", 1),
                   fixes_per_wave=count("fixes_per_wave", 1),
                   waves_per_round=count("waves_per_round", 1),
                   fill_freed_slots=_flag(sec, "fill_freed_slots", d.fill_freed_slots),
                   model_levers_per_wave=count("model_levers_per_wave", 0),
                   wave_tiers=_wave_tiers(sec.get("wave_tiers")),
                   value_sharpe_share=(None if sec.get("value_sharpe_share") is None
                                       else float(num("value_sharpe_share", 0, 2))),
                   low_value_fixes=count("low_value_fixes", 0),
                   blocked_fixes=count("blocked_fixes", 0),
                   lead_sharpe_share=(None if sec.get("lead_sharpe_share") is None
                                      else float(num("lead_sharpe_share", 0, 2))),
                   lead_fixes=count("lead_fixes", 0),
                   leg_tests=count("leg_tests", 0),
                   unmeasured_failure_penalty=float(num("unmeasured_failure_penalty", 0, 10)),
                   region_prior_strength=float(num("region_prior_strength", 0, 1000)))


def _flag(sec: dict, key: str, default: bool) -> bool:
    v = sec.get(key, default)
    if not isinstance(v, bool):
        raise ConfigError(f"agents config: repair.{key} must be true or false")
    return v


def _wave_tiers(raw) -> tuple:
    if raw is None:
        return ()
    bad = ConfigError("agents config: repair.wave_tiers must be a list of {max_distance: "
                      "number or null, fixes: integer >= 0, mechanical_only: optional bool}, "
                      "distances increasing, the last null")
    if not isinstance(raw, list) or not raw:
        raise bad
    out, last = [], -1.0
    for t in raw:
        if not isinstance(t, dict) or not {"max_distance", "fixes"} <= set(t) \
                or set(t) - {"max_distance", "fixes", "mechanical_only"}:
            raise bad
        d, n, mech = t["max_distance"], t["fixes"], t.get("mechanical_only", False)
        if isinstance(n, bool) or not isinstance(n, int) or n < 0 or not isinstance(mech, bool):
            raise bad
        if d is not None and (isinstance(d, bool) or not isinstance(d, (int, float)) or d <= last):
            raise bad
        out.append((None if d is None else float(d), n, mech))
        last = float("inf") if d is None else float(d)
    if out[-1][0] is not None:
        raise bad
    return tuple(out)


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
    if _lower_is_better(name):
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
    gap = (value - threshold) if _lower_is_better(name) else (threshold - value)
    return max(gap, 0.0) / abs(threshold)


def distance(decision: DecisionRecord | None, result: ResultRecord | None,
             unmeasured_penalty: float = 0.5) -> float | None:
    """Distance to accept (module doc, step 2). 0 for an accept; None when the result is
    missing or incomplete."""
    if decision is None or result is None or result.status != "complete":
        return None
    if decision.decision == "accept":
        return 0.0
    flip = _sign_first(decision)
    total = 0.0
    for cls in failing_classes(decision, result):
        gaps = [x for g in decision.gate_results
                if g.passed is False and check_class(g.name) == cls
                and (x := _shortfall(g.name, *(_mirrored(g) if flip else
                                               (g.value, g.threshold)))) is not None]
        total += max(gaps) if gaps else unmeasured_penalty
    return round(total, 6)


def _sign_first(decision: DecisionRecord) -> bool:
    """The evaluator routed this alpha sign first: its Sharpe is negative and flipping it
    mirrors its performance (not in a short-restricted market such as CHN)."""
    return any(r.startswith(SIGN_FIRST) for r in decision.reasons)


def _mirrored(g) -> tuple:
    """(value, threshold) of a gate as the flipped alpha would show them. A sign-first
    alpha is measured as if flipped (run rs20261126060436: GBR Sharpe -1.54 / fitness -1.38
    counted as distance 6.6, so the resume order never reached it, though its flip is a
    near miss). Sharpe-, fitness- and returns-type gates change sign; a negative threshold
    is a share of the alpha's own Sharpe (sub-universe), so it changes sign too. Turnover
    and correlation gates are unchanged."""
    name = g.name.upper()
    if not (g.name in ("sharpe", "fitness", "sub_universe")
            or any(k in name for k in ("SHARPE", "FITNESS", "RETURNS"))):
        return g.value, g.threshold
    value = -g.value if isinstance(g.value, (int, float)) and not isinstance(g.value, bool) \
        else g.value
    t = g.threshold
    if isinstance(t, (int, float)) and not isinstance(t, bool) and t < 0:
        t = -t
    return value, t


def _sharpe_bar(decision: DecisionRecord | None) -> float | None:
    if decision is None:
        return None
    return next((g.threshold for g in decision.gate_results
                 if g.name in ("sharpe", "brain:LOW_SHARPE")
                 and isinstance(g.threshold, (int, float))), None)


def _size(decision: DecisionRecord | None, result: ResultRecord | None,
          config: RepairLearningConfig, playbook: Playbook | None) -> tuple[int, bool, str]:
    """(fixes, mechanical only, why); ``wave_size`` and ``mechanical_only`` share it. The
    blocked rule applies only with a ``playbook``."""
    dist = distance(decision, result, config.unmeasured_failure_penalty)
    n, mech, why = config.fixes_per_wave, False, "default size"
    for limit, fixes, m in config.wave_tiers:
        if limit is None or (dist is not None and dist <= limit):
            n, mech = fixes, m
            why = (f"distance to accept {dist}" if dist is not None else "distance unknown") \
                + (f" <= {limit}" if limit is not None else "") \
                + (", mechanical fixes only" if m and fixes else "")
            break
    if playbook is not None:
        blocked = sorted(c for c in failing_classes(decision, result)
                         if not playbook.entries.get(c))
        if blocked:
            if n > config.blocked_fixes:
                n, why = config.blocked_fixes, f"fails {', '.join(blocked)}, which no lever fixes"
            return n, mech, why
    bar, best = _sharpe_bar(decision), _best_sharpe(result)
    share = config.value_sharpe_share
    if share is not None and n > config.low_value_fixes and bar and best is not None \
            and best < share * bar:
        n, why = config.low_value_fixes, (f"best Sharpe {best:.2f} < {share:g} x bar "
                                          f"{bar:g}: little edge")
    lead = config.lead_sharpe_share
    if lead is not None and n < config.lead_fixes and bar and best is not None \
            and best >= lead * bar:
        n, mech = config.lead_fixes, True
        why += (f"; lead: best Sharpe {best:.2f} >= {lead:g} x bar {bar:g}, "
                f"{n} mechanical fixes")
    return n, mech, why


def wave_size(decision: DecisionRecord | None, result: ResultRecord | None,
              config: RepairLearningConfig, playbook: Playbook) -> tuple[int, str]:
    """(fixes for this alpha's next repair wave, why). The tier of its distance to accept
    (``wave_tiers``), then at most ``blocked_fixes`` when it fails a class no playbook
    lever addresses (no fix can make it pass, e.g. BRAIN's ASI checks), and at most
    ``low_value_fixes`` when its best Sharpe (raw or risk-neutralized, either sign) is
    below ``value_sharpe_share`` of the Sharpe bar (little real edge to save); finally a
    lead (best Sharpe >= ``lead_sharpe_share`` of the bar, not blocked) gets at least
    ``lead_fixes`` mechanical fixes."""
    n, _, why = _size(decision, result, config, playbook)
    return n, why


def mechanical_only(decision: DecisionRecord | None, result: ResultRecord | None,
                    config: RepairLearningConfig) -> bool:
    """Whether this alpha's wave allows mechanical fixes only (no model call): its tier
    says so (owner 2026-11-25, alphas 0.4-0.6 from accept), or it got fixes only as a lead."""
    return _size(decision, result, config, None)[1]


def _best_sharpe(result: ResultRecord | None) -> float | None:
    if result is None:
        return None
    vals = [abs(result.sharpe)] if isinstance(result.sharpe, (int, float)) else []
    rn = (result.alt_stats or {}).get("risk_neutralized") or {}
    if isinstance(rn.get("sharpe"), (int, float)):
        vals.append(abs(rn["sharpe"]))
    return max(vals) if vals else None


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
    part = 1.0 if fixed else gap_closed(before, after) if improved else 0.0
    reward = 0.5 * part + 0.5 * overall if complete else 0.0
    return round(reward, 6), fixed, improved, before, after


def gap_closed(before: float | None, after: float | None) -> float:
    """Share of a failing class's gap a change closed, in [0, 1]: slack -0.38 -> -0.08 is
    0.79. A class that was not failing (slack >= 0) or a slack without a value gives 0.5,
    the old flat credit for an improvement."""
    if before is None or after is None:
        return 0.0
    if before >= 0:
        return 0.5
    return round(min(1.0, max(0.0, (after - before) / -before)), 6)


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
    # a lineage with an accepted member is done: any further variant is a near copy of it
    # and fails SELF_CORRELATION (EUR reversal, 2026-11-15: two siblings at 0.01 from
    # passing were left once one blend was accepted)
    won = next((r["candidate"].candidate_id for r in members
                if r.get("decision") is not None and r["decision"].decision == "accept"), None)
    if won is not None:
        return 0, f"lineage {root} already has an accepted alpha ({won})"
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
    gains: list = field(default_factory=list)   # slack after - before, target/secondary only


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
                if e.role != "part" and e.slack_before is not None and e.slack_after is not None:
                    rec.gains.append(round(e.slack_after - e.slack_before, 6))
        return stats

    def gains(self, failure_class: str, lever: str, region: str | None = None) -> list[float]:
        """Past slack gains of a lever for a class: the region's own when it has
        evidence, else every region's."""
        own = self.by_region.get((region, failure_class, lever)) if region else None
        if own is not None and own.gains:
            return own.gains
        return self.records.get((failure_class, lever), LeverRecord()).gains

    def estimate(self, failure_class: str, lever: str, region: str | None = None,
                 need: float | None = None, gain_weight: float = 0.5,
                 min_gain_evidence: int = 2) -> float:
        """Posterior mean, blended with P(past gain >= need) when the alpha misses the
        class by ``need`` and the lever has enough past gains (module doc, point 6)."""
        mean = self.mean(failure_class, lever, region)
        gains = self.gains(failure_class, lever, region)
        if need is None or need <= 0 or len(gains) < min_gain_evidence or not gain_weight:
            return mean
        reach = (sum(g >= need for g in gains) + 1) / (len(gains) + 2)
        return (1 - gain_weight) * mean + gain_weight * reach

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
                 "posterior_mean": round(self.mean(cls, lever), 3),
                 "median_gain": _median(r.gains)}
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


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return round(s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2, 4)


def needs(decision: DecisionRecord | None, result: ResultRecord | None,
          classes: Iterable[str]) -> dict[str, float]:
    """{class: gain in slack still needed to pass} for the failing classes with a value."""
    out = {}
    for cls in classes:
        s = slack(cls, decision, result)
        if s is not None and s < 0:
            out[cls] = round(-s, 6)
    return out


def rank_levers(playbook: Playbook, primary: str, failing: Iterable[str], stats: LeverStats,
                config: RepairLearningConfig, rng: random.Random | None = None,
                region: str | None = None,
                need: dict[str, float] | None = None) -> list[tuple[Lever, float]]:
    """The primary class's levers plus its learned combinations, best first, each with its
    score (see module doc), using the region's estimates when ``region`` is given and the
    gain match when ``need`` ({class: gain needed}) is given."""
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
                score += w * stats.estimate(cls, lv.name, region, (need or {}).get(cls),
                                            config.gain_weight, config.min_gain_evidence)
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
             "result": "fixed" if e.fixed else _improved_text(e)} for e in good[:limit]]


def _improved_text(e: RepairEvent) -> str:
    if e.slack_before is None or e.slack_after is None:
        return "improved"
    gain = e.slack_after - e.slack_before
    return (f"improved by {gain:+.3g}, closing {gap_closed(e.slack_before, e.slack_after):.0%} "
            f"of the gap (still {-e.slack_after:.3g} short)")
