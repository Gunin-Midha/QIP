"""Evaluator (spec section 7). Rules only, no judgment.

``run_gates`` produces the gate results. ``decide`` turns gates plus the risk manager's
checks into a DecisionRecord.

Decision precedence (owner's choice: a definite failure beats "not evaluable"):
1. status != complete             -> reject / incomplete
2. any hard gate False            -> repair if the playbook has an entry for the failure
                                     class, else reject
3. risk veto                      -> reject, class = first failed risk item
4. any hard gate None             -> pending
5. otherwise                      -> accept
"""

from __future__ import annotations

from typing import Iterable

from .config import NOT_CONFIGURED, Config
from .repair import Playbook
from .schemas import DecisionRecord, GateResult, ResultRecord

HARD_GATES = ("sharpe", "fitness", "turnover_range", "self_correlation", "sub_universe")
SOFT_GATES = ("weight_concentration", "drawdown")

FAILURE_PRIORITY = (
    "self_correlation",
    "turnover",
    "sharpe",
    "fitness",
    "sub_universe",
    "concentration",
    "drawdown",
    "leg_imbalance",
    "weak_year",
    "other",
)

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
}


def _gt_gate(name: str, value, threshold) -> GateResult:
    if threshold is None:
        return GateResult(name, None, value, None, NOT_CONFIGURED)
    if value is None:
        return GateResult(name, None, None, threshold, "value missing")
    return GateResult(name, value > threshold, value, threshold, f"> {threshold}")


def run_gates(result: ResultRecord, config: Config) -> list[GateResult]:
    g = config.gates
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

    # Soft checks: recorded here, decided by the risk manager.
    gates.append(GateResult("weight_concentration", None, result.weight_concentration, None,
                            "soft: decided by risk manager"))
    gates.append(GateResult("drawdown", None, result.max_drawdown, None,
                            "soft: decided by risk manager"))
    return gates


def classify(checks: Iterable[GateResult]) -> str:
    failed = {CHECK_TO_CLASS.get(c.name, "other") for c in checks if c.passed is False}
    for cls in FAILURE_PRIORITY:
        if cls in failed:
            return cls
    return "none"


def decide(result: ResultRecord, gates: list[GateResult], playbook: Playbook,
           risk_checks: list[GateResult] | None = None, risk_veto: bool = False,
           risk_reasons: list[str] | None = None) -> DecisionRecord:
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

    hard = [g for g in gates if g.name in HARD_GATES]
    failed = [g for g in hard if g.passed is False]
    unknown = [g for g in hard if g.passed is None]

    if failed:
        failure_class = classify(failed)
        reasons = [f"{g.name} failed: value {g.value} vs {g.threshold}" for g in failed]
        if playbook.has_entry(failure_class):
            return record("repair", failure_class, reasons + [f"playbook has {failure_class} levers"])
        return record("reject", failure_class, reasons + [f"no playbook entry for {failure_class}"])

    if risk_veto:
        return record("reject", classify(risk_checks), ["risk veto"] + risk_reasons)

    if unknown:
        return record("pending", "none",
                      [f"{g.name} not evaluable: {g.note}" for g in unknown])

    return record("accept", "none", ["all hard gates passed", "no risk veto"])


def evaluate(result: ResultRecord, config: Config, playbook: Playbook,
             risk_checks: list[GateResult] | None = None, risk_veto: bool = False,
             risk_reasons: list[str] | None = None) -> DecisionRecord:
    return decide(result, run_gates(result, config), playbook, risk_checks, risk_veto,
                  risk_reasons)
