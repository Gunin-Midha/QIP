"""Evaluator (spec section 7). Rules only, no judgment.

``run_gates`` produces the gate results. ``decide`` turns gates plus the risk manager's
checks into a DecisionRecord.

BRAIN's own checks (owner's choice 2026-11-04: "use BRAIN's checks only"). With
``gates.use_brain_checks`` every PASS/FAIL check BRAIN reports (ResultRecord.checks:
LOW_SHARPE, CONCENTRATED_WEIGHT, LOW_SUB_UNIVERSE_SHARPE, ...) becomes a hard gate named
``brain:<CHECK>``, so whatever BRAIN fails is a definite failure of its class, even where
our own limit is null (concentration, OQ-4). We set no limits of our own for those checks.
A check BRAIN reports as WARNING or PENDING is not a PASS/FAIL and is not a gate, and
neither is a check listed in ``gates.brain_checks_ignored``. A check listed in
``gates.submission_checks`` (about submitting, not the alpha: daily quota, Power Pool,
theme matches) is a ``submit:<CHECK>`` gate decided last: only an alpha that passes every
other check is held, as pending, by a failed submission check.

Decision precedence (owner's choice: a definite failure beats "not evaluable"):
1. status != complete             -> reject / incomplete
2. any hard gate False            -> repair if the playbook has an entry for the failure
                                     class, else reject. For a class in
                                     ``repair.near_miss_share`` (sharpe, fitness) the failed
                                     value must also reach that share of its threshold by
                                     absolute value (Sharpe: the larger of raw and
                                     risk-neutralized); a far miss is rejected. A failed
                                     Sharpe that is negative, or far better risk-neutralized,
                                     goes to the Sharpe levers first (``sharpe_route``).
3. risk veto                      -> reject, class = first failed risk item
4. any hard gate None             -> pending
5. otherwise                      -> accept
"""

from __future__ import annotations

import re

import dataclasses
from typing import Iterable

from .config import NOT_CONFIGURED, Config
from .repair import Playbook
from .schemas import DecisionRecord, GateResult, ResultRecord

HARD_GATES = ("sharpe", "fitness", "turnover_range", "self_correlation", "sub_universe")
SOFT_GATES = ("weight_concentration", "drawdown")

FAILURE_PRIORITY = (
    "self_correlation",
    "prod_correlation",
    "turnover",
    "sharpe",
    "fitness",
    "recent_sharpe",
    "sub_universe",
    "concentration",
    "drawdown",
    "leg_imbalance",
    "weak_year",
    "other",
)

# BRAIN's check names (ResultRecord.checks) -> failure class; any other failed check is "other"
BRAIN_CHECK_TO_CLASS = {
    "LOW_SHARPE": "sharpe",
    "LOW_FITNESS": "fitness",
    "LOW_TURNOVER": "turnover",
    "HIGH_TURNOVER": "turnover",
    "CONCENTRATED_WEIGHT": "concentration",
    "LOW_SUB_UNIVERSE_SHARPE": "sub_universe",
    "SELF_CORRELATION": "self_correlation",
    # consultant level (live capture 2026-11-10)
    "LOW_2Y_SHARPE": "recent_sharpe",       # Sharpe of the last two in-sample years
    "IS_LADDER_SHARPE": "recent_sharpe",    # Sharpe over the last N in-sample years
    "PROD_CORRELATION": "prod_correlation",  # correlation with BRAIN's production alphas
    "POWER_POOL_CORRELATION": "prod_correlation",
    # the alpha on bigger, more tradable names (non-USA regions; run 2026-11-23: an IND
    # alpha at Sharpe 3.14 / fitness 2.20 was rejected for robust-universe Sharpe 0.95 of
    # 1.0 because these mapped to "other", which has no levers). Same question as the
    # sub-universe check, so the same levers (size and liquidity tilts, backfill, rank).
    "LOW_ROBUST_UNIVERSE_SHARPE": "sub_universe",
    "LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO": "sub_universe",
    "LOW_ROBUST_UNIVERSE_RETURNS": "sub_universe",
    "LOW_INVESTABILITY_CONSTRAINED_SHARPE": "sub_universe",
    # ASI: the alpha must also hold on the region's Japanese stocks (value = their Sharpe,
    # limit 1.0). A sub-universe question too; as class "other" it had no levers and blocked
    # every repair of the best ASI lead in run rs20261125111614 (owner 2026-11-25)
    "LOW_ASI_JPN_SHARPE": "sub_universe",
    # GLB: the alpha must also hold in each of the region's parts (value = that part's
    # Sharpe, limit 1.0). As "other" these blocked every GLB repair: 28-29 of our 29 GLB
    # alphas failed each of them by 2026-11-25 (owner 2026-11-25)
    "LOW_GLB_AMER_SHARPE": "sub_universe",
    "LOW_GLB_APAC_SHARPE": "sub_universe",
    "LOW_GLB_EMEA_SHARPE": "sub_universe",
}
BRAIN_GATE_PREFIX = "brain:"

# check name (evaluator gate, risk check, or diversity check) -> failure class
CHECK_TO_CLASS = {
    "sharpe": "sharpe",
    "fitness": "fitness",
    "turnover_range": "turnover",
    "self_correlation": "self_correlation",
    "sub_universe": "sub_universe",
    "weight_concentration": "concentration",
    "drawdown_depth": "drawdown",
    "leg_balance": "leg_imbalance",
    "weak_year": "weak_year",
    "top_pnl_removal": "other",
    "pool_correlation": "self_correlation",
    "deflated_sharpe": "other",
}


SUBMISSION_GATE_PREFIX = "submit:"   # gates.submission_checks: decided last (see decide)


def check_class(name: str) -> str:
    """Failure class of a gate or check name, including ``brain:<CHECK>`` gates."""
    if name.startswith(BRAIN_GATE_PREFIX):
        return BRAIN_CHECK_TO_CLASS.get(name[len(BRAIN_GATE_PREFIX):], "other")
    return CHECK_TO_CLASS.get(name, "other")


def is_hard(name: str) -> bool:
    return name in HARD_GATES or name.startswith(BRAIN_GATE_PREFIX)


def _brain_value(name: str, r: ResultRecord):
    """(value, limit) of a BRAIN check: BRAIN's own when it sent them, else our fields."""
    own = r.check_values.get(name) or {}
    if own.get("value") is not None or own.get("limit") is not None:
        return own.get("value"), own.get("limit")
    return {"LOW_SHARPE": (r.sharpe, None), "LOW_FITNESS": (r.fitness, None),
            "LOW_TURNOVER": (r.turnover, None), "HIGH_TURNOVER": (r.turnover, None),
            "CONCENTRATED_WEIGHT": (r.weight_concentration, None),
            "LOW_SUB_UNIVERSE_SHARPE": (r.sub_universe_sharpe, r.sub_universe_cutoff),
            "SELF_CORRELATION": (r.self_correlation, None)}.get(name, (None, None))


def _gt_gate(name: str, value, threshold) -> GateResult:
    if threshold is None:
        return GateResult(name, None, value, None, NOT_CONFIGURED)
    if value is None:
        return GateResult(name, None, None, threshold, "value missing")
    return GateResult(name, value > threshold, value, threshold, f"> {threshold}")


# BRAIN check whose limit replaces our gate key when gates.limits_from_brain is on
BRAIN_LIMIT_FOR = {"sharpe_min": "LOW_SHARPE", "fitness_min": "LOW_FITNESS",
                   "turnover_min": "LOW_TURNOVER", "turnover_max": "HIGH_TURNOVER",
                   "self_correlation_max": "SELF_CORRELATION"}


def gates_for(result: ResultRecord, config: Config):
    """The gate limits for this result: config (with its region overrides, applied by the
    caller), then BRAIN's reported limit per check when ``gates.limits_from_brain``."""
    g = config.gates
    if not g.limits_from_brain:
        return g
    limits = {key: (result.check_values.get(check) or {}).get("limit")
              for key, check in BRAIN_LIMIT_FOR.items()}
    return dataclasses.replace(g, **{k: v for k, v in limits.items() if v is not None})


# our gate -> BRAIN's check(s) for the same thing; with gates.brain_checks_only BRAIN's
# verdict decides the gate whenever BRAIN reported every check listed
OWN_GATE_CHECKS = {"sharpe": ("LOW_SHARPE",), "fitness": ("LOW_FITNESS",),
                   "turnover_range": ("LOW_TURNOVER", "HIGH_TURNOVER"),
                   "self_correlation": ("SELF_CORRELATION",),
                   "sub_universe": ("LOW_SUB_UNIVERSE_SHARPE",)}


def _brain_verdict(gate: GateResult, checks: dict) -> GateResult:
    names = OWN_GATE_CHECKS.get(gate.name)
    if not names or not all(n in checks for n in names):
        return gate
    ok = all(checks[n] for n in names)
    return GateResult(gate.name, ok, gate.value, gate.threshold,
                      f"BRAIN's {' and '.join(names)} {'PASS' if ok else 'FAIL'} decides "
                      f"(gates.brain_checks_only)")


# BRAIN check states that are not a verdict yet: PENDING while BRAIN computes the check
# (PROD_CORRELATION can take hours), ERROR when computing it failed for now (SELF_CORRELATION
# ERROR on CHN alphas 2026-11-24 turned PASS when fetched again). An alpha check in such a
# state is an unknown gate, so an alpha that passes everything else is pending, not accepted
# (2026-11-24: an alpha was accepted while its PROD_CORRELATION was ERROR).
UNFINISHED_CHECK_STATES = ("PENDING", "ERROR")
_NOT_COUNTED = re.compile(r"^brain_check_not_counted: ([A-Z0-9_.]+)='([A-Z_]+)'")


def unfinished_checks(result: ResultRecord | None) -> dict[str, str]:
    """{check: state} for BRAIN checks without a verdict yet (from the result's notes)."""
    out = {}
    for n in (result.notes if result is not None else []):
        m = _NOT_COUNTED.match(n)
        if m and m.group(2) in UNFINISHED_CHECK_STATES:
            out[m.group(1)] = m.group(2)
    return out


def run_gates(result: ResultRecord, config: Config) -> list[GateResult]:
    g = gates_for(result, config)
    gates = [
        _gt_gate("sharpe", result.sharpe, g.sharpe_min),
        _gt_gate("fitness", result.fitness, g.fitness_min),
    ]

    lo, hi, t = g.turnover_min, g.turnover_max, result.turnover
    if lo is None or hi is None:
        gates.append(GateResult("turnover_range", None, t, (lo, hi), NOT_CONFIGURED))
    elif t is None:
        gates.append(GateResult("turnover_range", None, None, (lo, hi), "value missing"))
    else:
        gates.append(GateResult("turnover_range", lo <= t <= hi, t, (lo, hi),
                                f"{lo} <= turnover <= {hi}"))

    sc, sc_max = result.self_correlation, g.self_correlation_max
    if sc_max is None:
        gates.append(GateResult("self_correlation", None, sc, None, NOT_CONFIGURED))
    elif sc is None:
        gates.append(GateResult("self_correlation", None, None, sc_max,
                                "not yet computed (computed at submission)"))
    else:
        gates.append(GateResult("self_correlation", sc < sc_max, sc, sc_max, f"< {sc_max}"))

    sub, cutoff = result.sub_universe_sharpe, result.sub_universe_cutoff
    if sub is None or cutoff is None:
        gates.append(GateResult("sub_universe", None, sub, cutoff,
                                "sub-universe sharpe or cutoff missing from result"))
    else:
        gates.append(GateResult("sub_universe", sub >= cutoff, sub, cutoff,
                                ">= cutoff from result"))
    if g.brain_checks_only:
        gates = [_brain_verdict(x, result.checks or {}) for x in gates]

    if g.use_brain_checks:
        ignored, submission = set(g.brain_checks_ignored), set(g.submission_checks)
        for name, ok in sorted((result.checks or {}).items()):
            if name in ignored:
                continue
            value, threshold = _brain_value(name, result)
            if name in submission:
                gates.append(GateResult(SUBMISSION_GATE_PREFIX + name, ok, value, threshold,
                                        "BRAIN submission check " + ("PASS" if ok else "FAIL")
                                        + " (decided only once every other check passes)"))
                continue
            gates.append(GateResult(BRAIN_GATE_PREFIX + name, ok, value, threshold,
                                    "BRAIN check " + ("PASS" if ok else "FAIL")))
        for name, state in sorted(unfinished_checks(result).items()):
            if name in ignored or name in submission or name in (result.checks or {}):
                continue
            value, threshold = _brain_value(name, result)
            gates.append(GateResult(BRAIN_GATE_PREFIX + name, None, value, threshold,
                                    f"BRAIN check {state}: no verdict yet (cli recheck)"))

    # Soft checks: recorded here, decided by the risk manager.
    gates.append(GateResult("weight_concentration", None, result.weight_concentration, None,
                            "soft: decided by risk manager"))
    gates.append(GateResult("drawdown", None, result.max_drawdown, None,
                            "soft: decided by risk manager"))
    return gates


def classify(checks: Iterable[GateResult]) -> str:
    failed = {check_class(c.name) for c in checks if c.passed is False}
    for cls in FAILURE_PRIORITY:
        if cls in failed:
            return cls
    return "none"


SIGN_FIRST = "sign first"
FACTOR_FIRST = "factor neutralization first"


def risk_neutralized_sharpe(result: ResultRecord | None) -> float | None:
    """BRAIN's Sharpe with its risk factors removed (``is.riskNeutralized``), or None."""
    if result is None:
        return None
    return ((result.alt_stats or {}).get("risk_neutralized") or {}).get("sharpe")


def _risk_neutralized(result: ResultRecord | None, key: str) -> float | None:
    if result is None:
        return None
    v = ((result.alt_stats or {}).get("risk_neutralized") or {}).get(key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


FLIP_DOES_NOT_MIRROR = "a flip does not mirror here"


def near_miss(failed: list[GateResult], failure_class: str,
              shares: dict | None, result: ResultRecord | None = None, *,
              sign_mirrors: bool = True,
              flipped: bool | None = None) -> tuple[bool, str | None]:
    """(repairable, reason). A class without a configured share is always repairable. For
    Sharpe and fitness the larger of |raw| and |risk-neutralized| counts: a signal hidden under BRAIN's
    risk factors is a near miss that a factor neutralization can uncover. Without
    ``sign_mirrors`` (``schemas.SHORT_RESTRICTED_REGIONS``: the opposite alpha does not flip
    the performance) values count with their sign, so a negative one is no near miss.
    ``flipped`` judges the value in the orientation the alpha will be repaired in, not by
    its size: True = as after a sign flip (-value; the alpha is sent sign first), False =
    as measured. A value with the opposite sign to the Sharpe (Sharpe -1.03 with 2-year
    Sharpe +0.48, run rs20261125130750) turns negative when flipped, which |value| hid; None
    = |value| (where the orientation is not known yet, e.g. the Sharpe itself)."""
    if flipped is None:
        size = abs if sign_mirrors else (lambda v: v)
    elif flipped:
        size = lambda v: -v  # noqa: E731
    else:
        size = lambda v: v  # noqa: E731
    share = (shares or {}).get(failure_class)
    if share is None:
        return True, None
    # fitness too (2026-11-25): the EUR accept's ancestor had raw fitness 0.41 but
    # risk-neutralized 0.96, and its neutralized child reached 1.30
    rn = (risk_neutralized_sharpe(result) if failure_class == "sharpe" else
          _risk_neutralized(result, "fitness") if failure_class == "fitness" else None)
    for g in failed:
        if check_class(g.name) != failure_class or g.value is None or g.threshold is None:
            continue
        need = share * abs(g.threshold)
        best = max(size(g.value), size(rn)) if rn is not None else size(g.value)
        if best < need:
            shown = f"{g.value}" + (f" (risk-neutralized {rn})" if rn is not None else "")
            how = ("value after the sign flip" if flipped else
                   "value" if flipped is False and sign_mirrors else
                   "|value|" if sign_mirrors else f"value ({FLIP_DOES_NOT_MIRROR})")
            return False, (f"{g.name} {shown} is too far from {g.threshold} to repair "
                           f"({how} < {share} x threshold = {need:.4g})")
    return True, None


def sharpe_route(failed: list[GateResult], result: ResultRecord, shares: dict | None,
                 gap: float | None, primary: str = "sharpe", *,
                 sign_mirrors: bool = True) -> str | None:
    """Send a failed Sharpe to the Sharpe levers ahead of a higher-priority class when the
    sign or BRAIN's risk factors are the bigger problem: a raw Sharpe that is negative and
    a near miss by size (``SIGN_FIRST``: flip it before tuning turnover), or a
    risk-neutralized Sharpe that is a near miss and at least ``gap`` above the raw one
    (``FACTOR_FIRST``). Run rs20261112054223: CHN retail flow at Sharpe -1.68 (risk-neutralized
    -2.62) was routed to turnover, smoothed, and rejected without the sign ever flipped.
    ``FACTOR_FIRST`` takes over from another ``primary`` class only when the raw Sharpe alone
    is too far to repair (the risk-neutralized one is what makes it a near miss); otherwise
    the usual priority stands (a self-correlation failure stays first)."""
    g = next((g for g in failed if check_class(g.name) == "sharpe" and g.value is not None
              and g.threshold is not None), None)
    if g is None or not near_miss([g], "sharpe", shares, result, sign_mirrors=sign_mirrors)[0]:
        return None
    rn = risk_neutralized_sharpe(result)
    if sign_mirrors and g.value < 0 and (rn is None or rn <= 0):
        # flipping mirrors the alpha, so a risk-neutralized Sharpe that is only less
        # negative is not a better alpha, it is a weaker mirror (run rs20261126082238: raw
        # -1.71, risk-neutralized -1.11 was routed factor first and scored 7 from accept,
        # though its flip was Sharpe 1.71, fitness 1.61); flip first
        return f"{SIGN_FIRST}: Sharpe {g.value} is negative"
    if gap is not None and rn is not None and rn - g.value >= gap:
        raw_near = near_miss([g], "sharpe", shares, sign_mirrors=sign_mirrors)[0]
        if primary == "sharpe" or not raw_near:
            return f"{FACTOR_FIRST}: risk-neutralized Sharpe {rn} vs raw {g.value}"
        return None
    if sign_mirrors and g.value < 0 and (rn is None or rn <= g.value + (gap or 0.0)):
        return f"{SIGN_FIRST}: Sharpe {g.value} is negative"
    return None


def decide(result: ResultRecord, gates: list[GateResult], playbook: Playbook,
           risk_checks: list[GateResult] | None = None, risk_veto: bool = False,
           risk_reasons: list[str] | None = None,
           near_miss_share: dict | None = None,
           risk_neutralized_gap: float | None = None, *,
           sign_mirrors: bool = True) -> DecisionRecord:
    risk_checks = list(risk_checks or [])
    risk_reasons = list(risk_reasons or [])
    all_checks = list(gates) + risk_checks

    def record(decision: str, failure_class: str, reasons: list[str]) -> DecisionRecord:
        return DecisionRecord(
            candidate_id=result.candidate_id, decision=decision, failure_class=failure_class,
            gate_results=all_checks, risk_veto=risk_veto, risk_reasons=risk_reasons,
            reasons=reasons,
        )

    if result.status != "complete":
        return record("reject", "incomplete", [f"result status is {result.status!r}"])

    hard = [g for g in gates if is_hard(g.name)]
    failed = [g for g in hard if g.passed is False]
    unknown = [g for g in hard if g.passed is None]

    if failed:
        failure_class = classify(failed)
        reasons = [f"{g.name} failed: value {g.value} vs {g.threshold}" for g in failed]
        route = sharpe_route(failed, result, near_miss_share, risk_neutralized_gap,
                             failure_class, sign_mirrors=sign_mirrors)
        if route and playbook.has_entry("sharpe"):
            if failure_class != "sharpe":
                reasons.append(f"sent to sharpe levers ahead of {failure_class}")
            failure_class = "sharpe"
            reasons.append(route)
        if not playbook.has_entry(failure_class):
            return record("reject", failure_class, reasons + [f"no playbook entry for {failure_class}"])
        # every failing class with a near-miss limit must be a near miss, not only the one
        # repaired first (owner 2026-11-25: rs20261125130750r01a1_003, Sharpe -0.59 and
        # fitness -0.12, went to repair / turnover because only turnover was checked)
        classes = [failure_class] + sorted({check_class(g.name) for g in failed} - {failure_class})
        if route and route.startswith(FACTOR_FIRST):
            # the factor neutralization changes every raw measure; recent-years Sharpe has no
            # risk-neutralized value to judge it by, so it does not reject here
            classes = [c for c in classes if c != "recent_sharpe"]
        # an alpha sent sign first is judged as it will be after the flip (owner 2026-11-27:
        # a value the flip changes must not reject it as measured); the Sharpe itself by size
        flip = True if route and route.startswith(SIGN_FIRST) else None
        for cls in classes:
            ok, why = near_miss(failed, cls, near_miss_share, result, sign_mirrors=sign_mirrors,
                                flipped=None if cls == "sharpe" else flip)
            if not ok:
                return record("reject", cls, reasons + [why])
        return record("repair", failure_class, reasons + [f"playbook has {failure_class} levers"])

    if risk_veto:
        return record("reject", classify(risk_checks), ["risk veto"] + risk_reasons)

    if unknown:
        return record("pending", "none",
                      [f"{g.name} not evaluable: {g.note}" for g in unknown])

    # submission checks last: the alpha passed every check about itself, but BRAIN would
    # not take it as it is (daily quota, Power Pool, ...): pending for the owner, not reject
    blocked = [g for g in gates if g.name.startswith(SUBMISSION_GATE_PREFIX) and g.passed is False]
    if blocked:
        return record("pending", "none",
                      ["every alpha check passed"] +
                      [f"{g.name} failed: value {g.value} vs {g.threshold} (submission check)"
                       for g in blocked])

    return record("accept", "none", ["all hard gates passed", "no risk veto"])


def evaluate(result: ResultRecord, config: Config, playbook: Playbook,
             risk_checks: list[GateResult] | None = None, risk_veto: bool = False,
             risk_reasons: list[str] | None = None) -> DecisionRecord:
    return decide(result, run_gates(result, config), playbook, risk_checks, risk_veto,
                  risk_reasons, config.repair.near_miss_share,
                  config.repair.risk_neutralized_gap)
