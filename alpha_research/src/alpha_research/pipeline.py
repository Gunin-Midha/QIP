"""Deterministic end-to-end pipeline (spec section 14.1).

Per candidate:
1. duplicate check (skip if a result already exists for the same experiment hash)
2. simulator wrapper (validation, submit/resume, poll, normalize, metric notes, persist)
3. variant registration for every candidate that reached BRAIN (section 10.1 counts every
   simulation, including failures, so this is not limited to accepts)
4. evaluator gates
5. risk manager (skipped when the result is not complete)
6. decision; if accept, diversity admission against the pool (fail -> reject /
   self_correlation; not evaluable -> pending)
7. persist the decision; admit accepts to the pool
8. print one summary line

Pending decisions are routed back onto the research loop queue but not re-run here.

``process_many`` runs step 2 for several candidates at once (up to
``simulator.max_concurrent`` on BRAIN together, OQ-3), then steps 3-8 one candidate at a
time in the given order, so decisions, pool admission and printed lines come in the same
order as with ``process`` (a duplicate is printed when it is found).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TextIO

from . import diversity, evaluator, risk
from .auditor import Auditor
from .brain_client import BrainClient
from .config import Config
from .memory import MemoryStore
from .pool import Pool
from .repair import Playbook
from .research_loop import ResearchLoop, experiment_id
from .schemas import CandidateRecord, DecisionRecord, GateResult, ResultRecord, SchemaError
from .simulator import Simulator

log = logging.getLogger(__name__)

META_KEYS = ("fixture_id", "expected_decision", "expected_failure_class")


@dataclass
class CandidateEntry:
    candidate: CandidateRecord
    meta: dict = field(default_factory=dict)


@dataclass
class Outcome:
    candidate_id: str
    decision: str                      # accept / repair / reject / pending / duplicate
    failure_class: str | None
    result: ResultRecord | None = None
    decision_record: DecisionRecord | None = None

    def line(self) -> str:
        r = self.result

        def fmt(v):
            return "None" if v is None else f"{v:.4f}"

        return " | ".join([
            self.candidate_id, self.decision, str(self.failure_class),
            fmt(r.sharpe if r else None), fmt(r.fitness if r else None),
            fmt(r.turnover if r else None),
        ])


def load_candidate_file(path: str | Path) -> list[CandidateEntry]:
    """Parse a candidate file. Structural problems raise SchemaError; domain problems
    (unknown region, bad parentheses) are left for the simulator to mark invalid."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read candidate file {path}: {exc}") from exc
    if isinstance(raw, dict) and "candidates" in raw:
        raw = raw["candidates"]
    if not isinstance(raw, list):
        raise SchemaError("candidate file must be a JSON list (or an object with 'candidates')")
    entries, seen = [], set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SchemaError(f"candidate #{i} is not an object")
        cand = CandidateRecord.from_dict({k: v for k, v in item.items() if k not in META_KEYS})
        if cand.candidate_id in seen:
            raise SchemaError(f"candidate_id {cand.candidate_id!r} appears twice")
        seen.add(cand.candidate_id)
        entries.append(CandidateEntry(cand, {k: item.get(k) for k in META_KEYS if k in item}))
    return entries


class Pipeline:
    def __init__(self, config: Config, playbook: Playbook, client: BrainClient,
                 memory: MemoryStore, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, out: TextIO | None = None):
        self.config = config
        self.playbook = playbook
        self.memory = memory
        self.simulator = Simulator(client, memory, config, sleep=sleep, clock=clock)
        self.loop = ResearchLoop(memory)
        self.pool = Pool(memory)
        self.auditor = Auditor(memory, config)
        self.out = out

    def _progress(self, r: ResultRecord, finished: int, total: int) -> None:
        """One line as each of several simulations finishes; decisions follow in order once
        all are done."""
        if r.status == "complete":
            stats = ", ".join(f"{k} {v:.2f}" for k, v in (("Sharpe", r.sharpe), ("fitness", r.fitness),
                                                           ("turnover", r.turnover)) if v is not None)
            what = f"complete: {stats}" if stats else "complete"
        else:
            what = r.status
        print(f"  BRAIN: {r.candidate_id} finished ({finished} of {total}), {what}", file=self.out)

    def _emit(self, outcome: Outcome) -> Outcome:
        if self.out is not None:
            print(outcome.line(), file=self.out)
        return outcome

    def process(self, candidate: CandidateRecord) -> Outcome:
        return self.process_many([candidate])[0]

    def process_many(self, candidates: list[CandidateRecord]) -> list[Outcome]:
        """Outcomes in the order of ``candidates``. Duplicates (of memory or of an earlier
        candidate in the list) are skipped; the rest are simulated together, then
        evaluated in order."""
        outcomes: list[Outcome | None] = [None] * len(candidates)
        to_run, seen = [], set()
        for i, candidate in enumerate(candidates):
            h = experiment_id(candidate)
            if h in seen or self.loop.is_duplicate(candidate):
                log.info("duplicate of an earlier experiment, skipped: %s", candidate.candidate_id)
                outcomes[i] = self._emit(Outcome(candidate.candidate_id, "duplicate", None))
                continue
            seen.add(h)
            to_run.append((i, candidate))
        results = self.simulator.run_many(
            [c for _, c in to_run],
            on_done=self._progress if self.out is not None and len(to_run) > 1 else None)
        for (i, candidate), result in zip(to_run, results):
            if result.status != "invalid":
                self.auditor.register_variant(candidate.family, candidate.candidate_id)
            outcomes[i] = self._evaluate(candidate, result)
        return outcomes  # type: ignore[return-value]

    def recover(self, candidate: CandidateRecord, sim_id: str) -> Outcome | None:
        """Decide an earlier submission that BRAIN finished after our wait ended (its
        variant was registered when it was first simulated). None while BRAIN is still
        running it or no longer knows it."""
        result = self.simulator.recover(candidate, sim_id)
        return None if result is None else self._evaluate(candidate, result)

    def _evaluate(self, candidate: CandidateRecord, result: ResultRecord) -> Outcome:
        config = self.config.for_settings(candidate.region, candidate.delay)
        gates = evaluator.run_gates(result, config)
        if result.status == "complete":
            veto, risk_checks, risk_reasons = risk.assess(result, config, gates)
        else:
            veto, risk_checks, risk_reasons = False, [], []
        brain_only = config.gates.brain_checks_only   # a pass is BRAIN's checks alone
        if brain_only and veto:
            veto = False
            risk_reasons = risk_reasons + ["risk veto recorded, not applied (gates.brain_checks_only)"]
        decision = evaluator.decide(result, gates, self.playbook, risk_checks, veto, risk_reasons,
                                    config.repair.near_miss_share,
                                    config.repair.risk_neutralized_gap)

        admitted_by_diversity = None
        if decision.decision == "accept":
            checked = self._overfitting(candidate, result, decision)
            decision = self._informational(decision, checked) if brain_only else checked
        if decision.decision == "accept":
            admitted_by_diversity, checked = self._diversity(result, decision)
            decision = self._informational(decision, checked) if brain_only else checked

        self.memory.put_decision(decision)
        if decision.decision == "accept":
            self.pool.admit(candidate, result, decision, admitted_by_diversity)
        self.loop.route(candidate, decision)
        return self._emit(Outcome(candidate.candidate_id, decision.decision,
                                  decision.failure_class, result, decision))

    @staticmethod
    def _informational(decision: DecisionRecord, checked: DecisionRecord) -> DecisionRecord:
        """Keep ``decision`` but record what our own check (``checked``) found: its gates
        and reasons, marked as not applied (gates.brain_checks_only)."""
        if checked.decision == decision.decision:
            return checked
        extra = checked.reasons[len(decision.reasons):]
        return dataclasses.replace(
            decision, gate_results=checked.gate_results,
            reasons=decision.reasons + [f"recorded, not applied (gates.brain_checks_only): {r}"
                                        for r in extra])

    def _overfitting(self, candidate: CandidateRecord, result: ResultRecord,
                     decision: DecisionRecord) -> DecisionRecord:
        """Overfitting auditor at accept time: deflated Sharpe over the repair lineage."""
        g = self.auditor.check_accept(candidate.candidate_id, result)
        gate_results = list(decision.gate_results) + [g]
        if g.passed is False:
            return dataclasses.replace(
                decision, decision="reject", failure_class="other", gate_results=gate_results,
                reasons=decision.reasons + [f"overfitting: deflated Sharpe {g.value} is not "
                                            f"above {g.threshold} ({g.note})"])
        return dataclasses.replace(decision, gate_results=gate_results,
                                   reasons=decision.reasons + [f"deflated_sharpe: {g.note}"])

    def _diversity(self, result: ResultRecord,
                   decision: DecisionRecord) -> tuple[bool | None, DecisionRecord]:
        try:
            admit, dgates = diversity.admission_check(result.daily_pnl, self.pool.matrix(),
                                                      self.config)
        except diversity.InsufficientHistoryError as exc:
            admit, dgates = None, [GateResult("pool_correlation", None, None, None, str(exc))]
        gate_results = list(decision.gate_results) + dgates
        if admit is False:
            failed = [f"{g.name} failed: value {g.value} vs {g.threshold}"
                      for g in dgates if g.passed is False]
            decision = dataclasses.replace(
                decision, decision="reject", failure_class="self_correlation",
                gate_results=gate_results, reasons=decision.reasons + ["diversity admission failed"] + failed)
        elif admit is None:
            notes = [f"{g.name} not evaluable: {g.note}" for g in dgates if g.passed is None]
            decision = dataclasses.replace(
                decision, decision="pending", failure_class="none", gate_results=gate_results,
                reasons=decision.reasons + ["diversity admission not evaluable"] + notes)
        else:
            decision = dataclasses.replace(decision, gate_results=gate_results,
                                           reasons=decision.reasons + ["diversity admission passed"])
        return admit, decision

    def run(self, candidates: list[CandidateRecord]) -> list[Outcome]:
        """Every candidate, ``simulator.max_concurrent`` at a time."""
        n = max(1, self.config.simulator.max_concurrent)
        return [o for i in range(0, len(candidates), n)
                for o in self.process_many(candidates[i:i + n])]
