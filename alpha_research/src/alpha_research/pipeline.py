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
import re
import time
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TextIO

from . import diversity, evaluator, power_pool, risk
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
        """``r01a0_001 | repair / sharpe | Sharpe 1.00 | fitness 0.50 | turnover 20%``:
        the id without its research run prefix, values that exist, labeled."""
        r = self.result
        what = self.decision
        if self.failure_class not in (None, "none"):
            what += f" / {self.failure_class}"
        parts = [_short(self.candidate_id), what]
        if r is not None:
            parts += [f"Sharpe {r.sharpe:.2f}" if r.sharpe is not None else None,
                      f"fitness {r.fitness:.2f}" if r.fitness is not None else None,
                      f"turnover {r.turnover * 100:.0f}%" if r.turnover is not None else None]
        return " | ".join(x for x in parts if x)


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


def _short(candidate_id: str) -> str:
    """A research-run candidate id without its run id prefix (rs + 14 digits)."""
    return re.sub(r"^rs\d{14}", "", candidate_id) or candidate_id


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def newly_repairable(decision: DecisionRecord | None, result: ResultRecord | None,
                     playbook: Playbook) -> bool:
    """A complete alpha rejected as "other" whose failed gates now map, at least in part,
    to a class the playbook has levers for (the check mapping grew since it was decided)."""
    if decision is None or decision.decision != "reject" or decision.failure_class != "other":
        return False
    if result is None or result.status != "complete":
        return False
    classes = {evaluator.check_class(g.name) for g in decision.gate_results if g.passed is False}
    return any(c != "other" and playbook.entries.get(c) for c in classes)


# unfinished BRAIN checks (PENDING / ERROR) live in the evaluator, which gates on them
UNFINISHED_CHECK_STATES = evaluator.UNFINISHED_CHECK_STATES
unfinished_checks = evaluator.unfinished_checks


def awaiting_brain(decision: DecisionRecord | None, result: ResultRecord | None) -> bool:
    """A complete alpha decided pending (or accepted, before 2026-11-24) while some BRAIN
    check had no verdict yet: fetching its result again may decide it."""
    # "accept" too: before 2026-11-24 an alpha could be accepted while a BRAIN check had no
    # verdict (now such a check is an unknown gate, so it is pending)
    return (decision is not None and decision.decision in ("pending", "accept")
            and result is not None and result.status == "complete"
            and bool(result.brain_sim_id) and bool(unfinished_checks(result)))


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

    # ---- live BRAIN lines while a batch runs (owner 2026-11-23: short and aligned) ----
    # one status line a minute for the whole batch, one line per finished alpha with the
    # fix it tried; in a terminal the "fetching" note is overwritten by the result

    def _begin_batch(self, candidates: list[CandidateRecord]) -> None:
        self._t0 = self.simulator.clock()
        self._levers = {c.candidate_id: c.lever for c in candidates}
        self._width = max((len(self._name(c.candidate_id)) for c in candidates), default=0)
        self._open: str | None = None   # candidate whose "fetching" line is not finished
        self._tty = bool(getattr(self.out, "isatty", lambda: False)())

    def _name(self, cid: str) -> str:
        lever = getattr(self, "_levers", {}).get(cid)
        return f"{_short(cid)} ({lever})" if lever else _short(cid)

    def _head(self, cid: str | None = None) -> str:
        t = _clock(self.simulator.clock() - self._t0)
        return f"  {t:>6}  " + (f"{self._name(cid):<{self._width}}  " if cid else "")

    def _close_line(self) -> None:
        if self._open is not None:
            print(file=self.out, flush=True)
            self._open = None

    def _progress(self, r: ResultRecord, finished: int, total: int) -> None:
        """A line as each of several simulations finishes (decisions follow, in order,
        once the whole batch is in)."""
        if r.status == "complete":
            parts = [f"Sharpe {r.sharpe:5.2f}" if r.sharpe is not None else None,
                     f"fitness {r.fitness:5.2f}" if r.fitness is not None else None,
                     f"turnover {r.turnover * 100:3.0f}%" if r.turnover is not None else None]
            what = "  ".join(x for x in parts if x) or "complete"
        else:
            what = r.status
        line = f"{self._head(r.candidate_id)}{what}   ({finished} of {total})"
        if self._open == r.candidate_id and self._tty:
            print("\r\033[K" + line, file=self.out, flush=True)   # replaces "fetching ..."
            self._open = None
        else:
            self._close_line()
            print(line, file=self.out, flush=True)

    def _status(self, snap: dict) -> None:
        """While simulations run: how many are done and the range of BRAIN's progress."""
        self._close_line()
        progs = [p * 100 if p <= 1 else p for _, _, p in snap["running"] if p is not None]
        rng = ""
        if progs:
            lo, hi = min(progs), max(progs)
            rng = f" (BRAIN {lo:.0f}%)" if round(lo) == round(hi) else f" (BRAIN {lo:.0f}-{hi:.0f}%)"
        tail = f", {snap['waiting']} waiting" if snap["waiting"] else ""
        print(f"{self._head()}{snap['finished']} of {snap['total']} done, "
              f"{len(snap['running'])} running{rng}{tail}", file=self.out, flush=True)

    def _fetching(self, candidate_id: str) -> None:
        self._close_line()
        text = f"{self._head(candidate_id)}simulated, fetching results and checks ..."
        if self._tty:
            print(text, end="", file=self.out, flush=True)
            self._open = candidate_id
        else:
            print(text, file=self.out, flush=True)

    def _emit(self, outcome: Outcome) -> Outcome:
        if self.out is not None:
            print(outcome.line(), file=self.out)
        return outcome

    def process(self, candidate: CandidateRecord) -> Outcome:
        return self.process_many([candidate])[0]

    def process_many(self, candidates: list[CandidateRecord], *,
                     on_outcome: Callable[[CandidateRecord, Outcome], None] | None = None,
                     more: Callable[[int], list[CandidateRecord]] | None = None
                     ) -> list[Outcome]:
        """Outcomes in the order of ``candidates``. Duplicates (of memory or of an earlier
        candidate in the list) are skipped; the rest are simulated together. Without
        ``on_outcome`` they are evaluated in order once all are in. With it (streaming,
        owner 2026-11-23: a slow alpha no longer holds up the others' decisions), each is
        evaluated the moment its result is in and passed to ``on_outcome``, and
        ``more(free slots)`` may add alphas to the running batch (checked for duplicates
        the same way; their outcomes reach only ``on_outcome``)."""
        outcomes: list[Outcome | None] = [None] * len(candidates)
        to_run, seen = [], set()
        for i, candidate in enumerate(candidates):
            if self._duplicate(candidate, seen):
                outcomes[i] = self._emit(Outcome(candidate.candidate_id, "duplicate", None))
                if on_outcome is not None:
                    on_outcome(candidate, outcomes[i])
                continue
            to_run.append((i, candidate))
        live = self.out is not None
        if live:
            self._begin_batch([c for _, c in to_run])
        streaming = on_outcome is not None
        index = {c.candidate_id: i for i, c in to_run}

        def decide(candidate: CandidateRecord, result: ResultRecord) -> None:
            if result.status != "invalid":
                self.auditor.register_variant(candidate.family, candidate.candidate_id)
            if live:
                if self._open == candidate.candidate_id and self._tty:
                    print("\r\033[K", end="", file=self.out)   # the decision replaces the note
                    self._open = None
                self._close_line()
            outcome = self._evaluate(candidate, result)
            if candidate.candidate_id in index:
                outcomes[index[candidate.candidate_id]] = outcome
            on_outcome(candidate, outcome)

        def extra(free: int) -> list[CandidateRecord]:
            out = []
            for c in more(free):
                if self._duplicate(c, seen):
                    dup = self._emit(Outcome(c.candidate_id, "duplicate", None))
                    on_outcome(c, dup)
                    continue
                out.append(c)
            if live and out:
                self._levers.update({c.candidate_id: c.lever for c in out})
                self._width = max([self._width] + [len(self._name(c.candidate_id)) for c in out])
            return out

        results = self.simulator.run_many(
            [c for _, c in to_run],
            on_done=self._progress if live and len(to_run) > 1 and not streaming else None,
            on_status=self._status if live else None,
            on_ready=self._fetching if live else None,
            on_result=decide if streaming else None,
            more=extra if streaming and more is not None else None)
        if live:
            self._close_line()
        if not streaming:
            for (i, candidate), result in zip(to_run, results):
                if result.status != "invalid":
                    self.auditor.register_variant(candidate.family, candidate.candidate_id)
                outcomes[i] = self._evaluate(candidate, result)
        return outcomes  # type: ignore[return-value]

    def _duplicate(self, candidate: CandidateRecord, seen: set) -> bool:
        h = experiment_id(candidate)
        if h in seen or self.loop.is_duplicate(candidate):
            log.info("duplicate of an earlier experiment, skipped: %s", candidate.candidate_id)
            return True
        seen.add(h)
        return False

    def reevaluate_rejected(self) -> list[Outcome]:
        """Decide again, from their stored results, the rejected alphas a playbook lever
        can now address (``newly_repairable``): a BRAIN check that mapped to "other" when
        they were decided and maps to a class with levers now (2026-11-23: robust-universe
        and investability checks -> sub_universe). Simulates nothing; the new decision is
        appended (memory is append-only), so the next repair step sees it."""
        out = []
        for row in self.memory.candidate_outcomes():
            if newly_repairable(row["decision"], row["result"], self.playbook):
                out.append(self._evaluate(row["candidate"], row["result"]))
        return out

    def recheck_pending(self, only: set[str] | None = None) -> tuple[list[Outcome], list[str]]:
        """Fetch BRAIN's result again (one poll of the finished simulation, which re-reads
        ``/check``) for alphas decided pending because a BRAIN check had no verdict yet
        (``awaiting_brain``), and decide them again. A new result and decision are stored
        only when BRAIN's unfinished checks changed (memory is append-only). Simulates
        nothing. Returns (outcomes, ids still waiting on BRAIN)."""
        outs, still = [], []
        for row in self.memory.candidate_outcomes():
            c, d, r = row["candidate"], row["decision"], row["result"]
            if (only is not None and c.candidate_id not in only) or not awaiting_brain(d, r):
                continue
            before = unfinished_checks(r)
            new = self.simulator.recover(
                c, r.brain_sim_id, keep=lambda res: res.status == "complete"
                and unfinished_checks(res) != before,
                note=f"rechecked: BRAIN checks fetched again ({', '.join(sorted(before))} "
                     f"had no verdict before)")
            if new is None:
                if d.decision == "accept":
                    # accepted while a check had no verdict (before that check became an
                    # unknown gate): decided again from the stored result -> pending
                    outs.append(self._evaluate(c, r))
                still.append(c.candidate_id)
                continue
            outs.append(self._evaluate(c, new))
        return outs, still

    def stuck_submissions(self, after_hours: float, now: datetime | None = None) -> list[dict]:
        """Unfinished submissions (``memory.unfinished_submissions``) submitted at least
        ``after_hours`` ago: ``{"candidate_id", "brain_sim_id", "hours"}``, oldest first."""
        now = now or datetime.now(timezone.utc)
        times = self.memory.submitted_at()
        out = []
        for u in self.memory.unfinished_submissions():
            t = times.get(u["candidate_id"])
            if t is None:
                continue
            hours = (now - datetime.fromisoformat(t)).total_seconds() / 3600
            if hours >= after_hours:
                out.append({**u, "hours": hours})
        return sorted(out, key=lambda r: -r["hours"])

    def cancel_submission(self, candidate: CandidateRecord, sim_id: str,
                          reason: str) -> Outcome | None:
        """Cancel an unfinished simulation on BRAIN and decide the candidate (a failed
        result: reject / incomplete). None when BRAIN refused the cancel."""
        result = self.simulator.cancel(candidate, sim_id, reason)
        return None if result is None else self._evaluate(candidate, result)

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

        submission = self.config.gates.submission_checks
        tag = power_pool.candidate_reason(result, decision, submission)
        if tag is not None:   # recorded only: the decision itself is unchanged
            decision = dataclasses.replace(decision, reasons=decision.reasons + [tag])
        self.memory.put_decision(decision)
        if config.diversity.pool_admits == "submitted":
            pass   # the owner adds submitted alphas by hand (cli pool --submitted)
        elif decision.decision == "accept":
            # under brain_checks_only our diversity check is recorded only, so an alpha
            # BRAIN passed joins the pool (2026-11-25: one failed our pool check against a
            # sibling accepted by mistake and was left out)
            self.pool.admit(candidate, result, decision,
                            True if brain_only else admitted_by_diversity)
        else:   # re-decided (recheck, re-evaluation): no longer an accepted alpha
            self.pool.retire(candidate.candidate_id,
                             f"re-decided {decision.decision} / {decision.failure_class}")
        self.loop.route(candidate, decision)
        outcome = self._emit(Outcome(candidate.candidate_id, decision.decision,
                                     decision.failure_class, result, decision))
        if tag is not None and self.out is not None:
            print("  " + power_pool.describe(candidate, result, submission), file=self.out,
                  flush=True)
        return outcome

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
        """Every candidate, at most ``simulator.max_concurrent`` on BRAIN at a time, each
        decided the moment its result is in and a freed slot refilled at once (owner
        2026-11-24: one stuck simulation held a whole ``cli run`` batch's decisions for 30
        minutes). Outcomes in the order of ``candidates``. A new batch starts only when
        none of the current one reached BRAIN (all duplicate or invalid: no result to
        refill on)."""
        n = max(1, self.config.simulator.max_concurrent)
        rest = deque(candidates)
        got: dict[int, Outcome] = {}

        def take(free: int) -> list[CandidateRecord]:
            return [rest.popleft() for _ in range(min(free, len(rest)))]

        def keep(candidate: CandidateRecord, outcome: Outcome) -> None:
            got[id(candidate)] = outcome

        while rest:
            self.process_many(take(n), on_outcome=keep, more=take)
        return [got[id(c)] for c in candidates]
