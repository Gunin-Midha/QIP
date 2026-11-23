"""Simulator wrapper (spec section 4.2).

Steps per candidate: validate locally, submit (or resume a pending submission), poll with
retries, normalize the raw dict into a ResultRecord, run the metric validator, persist,
return. A failure the client marks ``retryable`` (BRAIN's own error) is submitted once more.
``recover`` polls an earlier submission once, for results BRAIN finished after our wait.

``run_many`` keeps up to ``simulator.max_concurrent`` simulations running on BRAIN at once
(OQ-3): it submits until that many are running, polls each running one in turn, and submits
the next as soon as one finishes. One thread: BRAIN does the work in parallel, we only wait
for it, so memory (SQLite) and the virtual clock stay simple. ``run`` is ``run_many`` of one.

Time is injected (``sleep`` and ``clock``) so fixture runs can use ``VirtualClock`` and
never wait in real time.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import date
from typing import Callable

from . import metric_validator
from .brain_client import BrainClient, SubmissionRejected
from .config import Config
from .memory import MemoryStore
from .schemas import (
    CandidateRecord,
    ResultRecord,
    SchemaError,
    YearRow,
    series_from_dict,
)

log = logging.getLogger(__name__)

TRANSIENT_ERRORS = (TimeoutError, ConnectionError)
BACKOFF_BASE_SECONDS = 1.0

# raw key -> ResultRecord field, and whether the raw value is a percent. These are our raw
# keys (see brain_client.py); brain_api.translate maps BRAIN's responses onto them.
_AGGREGATE_MAP = {
    "sharpe": ("sharpe", False),
    "fitness": ("fitness", False),
    "turnover": ("turnover", True),
    "returns": ("returns", True),
    "drawdown": ("max_drawdown", True),
    "margin": ("margin_bps", False),
    "sub_universe_sharpe": ("sub_universe_sharpe", False),
    "sub_universe_cutoff": ("sub_universe_cutoff", False),
    # TODO(open-question) OQ-2: unit of weight_concentration in the real response is
    # unknown. Stored as given (fraction in fixtures).
    "weight_concentration": ("weight_concentration", False),
    "self_correlation": ("self_correlation", False),
    "long_pnl": ("long_pnl", False),
    "short_pnl": ("short_pnl", False),
}
_YEARLY_MAP = {
    "sharpe": ("sharpe", False),
    "turnover": ("turnover", True),
    "fitness": ("fitness", False),
    "returns": ("returns", True),
    "drawdown": ("max_drawdown", True),
    "margin": ("margin_bps", False),
}


class VirtualClock:
    """Deterministic clock for fixture runs: ``sleep`` advances time instantly."""

    def __init__(self, start: float = 0.0):
        self.now = start
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# local validation
# ---------------------------------------------------------------------------

def parentheses_balanced(expression: str) -> bool:
    depth = 0
    for ch in expression:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _overlaps(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    return a_start <= b_end and b_start <= a_end


def local_validation_errors(candidate: CandidateRecord, config: Config) -> list[str]:
    """Return reasons the candidate must not be submitted. Raises ConfigError if the
    holdout is not configured."""
    holdout = config.require_holdout()
    errors = []
    try:
        candidate.validate()
    except SchemaError as exc:
        errors.append(str(exc))
    if candidate.expression and not parentheses_balanced(candidate.expression):
        errors.append("expression has unbalanced parentheses")
    if candidate.start_date < candidate.end_date and _overlaps(
        candidate.start_date, candidate.end_date, holdout.start_date, holdout.end_date
    ):
        errors.append(
            f"date range {candidate.start_date}..{candidate.end_date} overlaps holdout "
            f"{holdout.start_date}..{holdout.end_date}"
        )
    return errors


def brain_period(notes: list[str]) -> tuple[date, date] | None:
    """The period BRAIN actually simulated, from the ``brain_period: start..end`` note the
    real client adds (candidate dates are not sent to BRAIN). None when absent."""
    for n in notes:
        if n.startswith("brain_period: "):
            start, _, end = n[len("brain_period: "):].partition("..")
            try:
                return date.fromisoformat(start.strip()), date.fromisoformat(end.strip())
            except ValueError:
                return None
    return None


def holdout_leak(result: ResultRecord, config: Config) -> str | None:
    """Reason the result must not be used because BRAIN's simulated period overlaps the
    holdout (OQ-12), else None."""
    period = brain_period(result.notes)
    holdout = config.holdout
    if period is None or not holdout.configured:
        return None
    if _overlaps(period[0], period[1], holdout.start_date, holdout.end_date):
        return (f"BRAIN simulated {period[0]}..{period[1]}, which overlaps holdout "
                f"{holdout.start_date}..{holdout.end_date}")
    return None


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def _to_float(value, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"raw field {key!r} is not numeric: {value!r}")
    return float(value)


def _number_maps(raw, key: str) -> dict[str, dict]:
    """{name: {field: number}} from the raw response (BRAIN units, never converted)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not all(isinstance(v, dict) for v in raw.values()):
        raise SchemaError(f"raw {key!r} must be an object of name to object")
    return {str(k): {str(f): _to_float(x, f"{key}.{k}.{f}") for f, x in v.items()}
            for k, v in raw.items()}


def _pct(value: float | None) -> float | None:
    return None if value is None else value / 100.0


def _to_int(value, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise SchemaError(f"raw field {key!r} is not an integer: {value!r}")
    return int(value)


def normalize(raw: dict, candidate_id: str, brain_sim_id: str | None) -> ResultRecord:
    """Map a raw response to a ResultRecord. Percents become fractions (unless the raw
    dict says ``units: "fraction"``). Missing keys become None, never zero. Keys outside
    section 3.2 are ignored, apart from ``notes``, which is copied onto the result."""
    status = raw.get("status")
    if status not in ("complete", "failed"):
        raise SchemaError(f"cannot normalize a response with status {status!r}")
    units = raw.get("units", "percent")
    if units not in ("percent", "fraction"):
        raise SchemaError(f"raw 'units' must be 'percent' or 'fraction', got {units!r}")
    convert = _pct if units == "percent" else (lambda v: v)
    values = {}
    for key, (name, is_pct) in _AGGREGATE_MAP.items():
        v = _to_float(raw.get(key), key)
        values[name] = convert(v) if is_pct else v

    yearly = []
    for row in raw.get("yearly") or []:
        y = {"year": _to_int(row.get("year"), "yearly.year")}
        for key, (name, is_pct) in _YEARLY_MAP.items():
            v = _to_float(row.get(key), f"yearly.{key}")
            y[name] = convert(v) if is_pct else v
        y["long_count"] = _to_int(row.get("long_count"), "yearly.long_count")
        y["short_count"] = _to_int(row.get("short_count"), "yearly.short_count")
        yearly.append(YearRow(**y))

    dd_date = raw.get("drawdown_date")
    checks = raw.get("checks") or {}
    if not isinstance(checks, dict):
        raise SchemaError("raw 'checks' must be an object of name to bool")

    result = ResultRecord(
        candidate_id=candidate_id,
        status=status,
        brain_sim_id=brain_sim_id,
        drawdown_date=date.fromisoformat(dd_date) if dd_date else None,
        yearly=yearly,
        daily_pnl=series_from_dict(raw.get("daily_pnl")),
        checks={str(k): bool(v) for k, v in checks.items()},
        notes=[str(n) for n in raw.get("notes") or []],
        check_values=_number_maps(raw.get("check_values"), "check_values"),
        alt_stats=_number_maps(raw.get("alt_stats"), "alt_stats"),
        **values,
    )
    result.validate()
    return result


# ---------------------------------------------------------------------------
# wrapper
# ---------------------------------------------------------------------------

@dataclass
class _Job:
    """One candidate on its way through BRAIN in ``run_many``."""
    index: int
    candidate: CandidateRecord
    sim_id: str | None = None        # None until submitted (or resubmitted)
    started: float | None = None
    retried_from: str | None = None  # first simulation id, once BRAIN's error was retried
    progress: float | None = None    # BRAIN's progress while running (its poll reply)


class Simulator:
    def __init__(self, client: BrainClient, memory: MemoryStore, config: Config,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.client = client
        self.memory = memory
        self.config = config
        self.sleep = sleep
        self.clock = clock

    def run(self, candidate: CandidateRecord) -> ResultRecord:
        return self.run_many([candidate], max_concurrent=1)[0]

    def run_many(self, candidates: list[CandidateRecord],
                 max_concurrent: int | None = None,
                 on_done: Callable[[ResultRecord, int, int], None] | None = None,
                 on_status: Callable[[dict], None] | None = None,
                 on_ready: Callable[[str], None] | None = None,
                 on_result: Callable[[CandidateRecord, ResultRecord], None] | None = None,
                 more: Callable[[int], list[CandidateRecord]] | None = None
                 ) -> list[ResultRecord]:
        """Results in the order of ``candidates``, each persisted as it finishes. At most
        ``max_concurrent`` (default ``simulator.max_concurrent``) run on BRAIN at a time; a
        resumed pending submission takes a slot too. Each simulation has its own
        ``max_poll_seconds`` from its own submission. ``on_done(result, finished, total)``
        is called as each submitted alpha finishes (progress; invalid ones are not
        submitted and not counted). ``on_status(snapshot)`` is called at most every
        ``simulator.progress_every_seconds`` while alphas run: {"running": [(candidate id,
        seconds since submission, BRAIN's progress or None)], "waiting", "finished",
        "total"}. ``on_ready(candidate id)`` is called when BRAIN has finished an alpha and
        its results are being fetched, for a client with an ``on_ready`` hook.
        ``on_result(candidate, result)`` is called with each persisted result as soon as it
        is in (invalid ones at once), so a caller can decide it without waiting for the
        batch. ``more(free slots)`` is asked after each result for alphas to add to the
        running batch (a freed slot is used at once); their results come through
        ``on_result`` and are appended to the returned list."""
        limit = max(1, max_concurrent or self.config.simulator.max_concurrent)
        results: list[ResultRecord | None] = [None] * len(candidates)
        waiting: deque[_Job] = deque()

        def add(i: int, candidate: CandidateRecord) -> bool:
            """Queue one alpha; False when it is invalid (its result is recorded now)."""
            errors = local_validation_errors(candidate, self.config)  # may raise ConfigError
            self.memory.put_candidate(candidate)
            if errors:
                results[i] = self._persist(ResultRecord(
                    candidate_id=candidate.candidate_id, status="invalid",
                    notes=[f"invalid: {e}" for e in errors]))
                if on_result is not None:
                    on_result(candidate, results[i])
                return False
            sim_id = self.memory.pending_sim_id(candidate.candidate_id)
            if sim_id is not None:
                log.info("resuming pending submission %s for %s", sim_id, candidate.candidate_id)
            waiting.append(_Job(i, candidate, sim_id))
            return True

        for i, candidate in enumerate(candidates):
            add(i, candidate)

        active: list[_Job] = []
        total = len(waiting)
        every = self.config.simulator.progress_every_seconds
        last_status = self.clock()
        hooked = on_ready is not None and hasattr(self.client, "on_ready")
        if hooked:
            def ready(sim_id: str) -> None:
                job = next((j for j in active if j.sim_id == sim_id), None)
                on_ready(job.candidate.candidate_id if job else sim_id)
            self.client.on_ready = ready
        def topup() -> int:
            """Alphas ``more`` adds to the batch; returns how many were queued."""
            free = limit - len(active) - len(waiting)
            if more is None or free <= 0:
                return 0
            added = 0
            for candidate in more(free):
                results.append(None)
                added += add(len(results) - 1, candidate)
            return added

        try:
            return self._loop(waiting, active, results, limit, total, on_done, on_status, every,
                              last_status, on_result, topup)
        finally:
            if hooked:
                self.client.on_ready = None

    def _loop(self, waiting, active, results, limit, total, on_done, on_status, every,
              last_status, on_result=None, topup=None) -> list[ResultRecord]:
        finished = 0

        def done(job: _Job, result: ResultRecord) -> None:
            nonlocal finished, total
            results[job.index] = self._done(job, result)
            finished += 1
            if on_done is not None:
                on_done(results[job.index], finished, total)
            if on_result is not None:
                on_result(job.candidate, results[job.index])
            if topup is not None:
                total += topup()

        while waiting or active:
            while waiting and len(active) < limit:
                job = waiting.popleft()
                if job.sim_id is None:
                    sim_id = self._submit(job.candidate)
                    if isinstance(sim_id, ResultRecord):  # submitting failed
                        done(job, sim_id)
                        continue
                    job.sim_id = sim_id
                    if job.retried_from is None:
                        self.memory.put_pending(job.candidate.candidate_id, sim_id)
                job.started = self.clock()
                active.append(job)
            waits = []
            for job in list(active):
                result, retryable, wait = self._poll_once(job)
                if result is None:
                    waits.append(wait)
                    continue
                active.remove(job)
                if retryable and job.retried_from is None:
                    # BRAIN's own error, not the alpha's: submit once more, first in line
                    # (its slot just freed). The new simulation id is on the result;
                    # pending_submissions keeps the first, since it is append-only.
                    log.info("BRAIN error on %s for %s, submitting once more", job.sim_id,
                             job.candidate.candidate_id)
                    job.retried_from, job.sim_id = job.sim_id, None
                    waiting.appendleft(job)
                else:
                    done(job, result)
            if on_status is not None and every and active \
                    and self.clock() - last_status >= every:
                last_status = self.clock()
                on_status({"running": [(j.candidate.candidate_id, self.clock() - j.started,
                                        j.progress) for j in active],
                           "waiting": len(waiting), "finished": finished, "total": total})
            if active and waits:
                self.sleep(min(waits))
        return results  # type: ignore[return-value]

    def _done(self, job: _Job, result: ResultRecord) -> ResultRecord:
        if job.retried_from is None and job.sim_id is None:
            return self._persist(result)   # first submission failed: nothing to check
        if job.retried_from is not None:
            result.notes.insert(0, f"retried once after BRAIN's error on simulation "
                                   f"{job.retried_from}")
        return self._persist(self._finish(result))

    def recover(self, candidate: CandidateRecord, sim_id: str) -> ResultRecord | None:
        """One poll of an earlier submission (one that timed out on our side, or a crash
        before its result): BRAIN keeps running a simulation after we stop waiting. Returns
        the persisted result when BRAIN has finished it, None while it is still running or
        when BRAIN no longer knows the simulation."""
        try:
            raw = self._with_retry(lambda: self.client.poll(sim_id))
        except TRANSIENT_ERRORS:
            return None
        if raw.get("status") not in ("complete", "failed"):
            return None
        result = normalize(raw, candidate.candidate_id, sim_id)
        if result.status == "failed" and any("simulation not found" in n for n in result.notes):
            return None
        result.notes.insert(0, f"recovered: result fetched after our wait ended ({sim_id})")
        return self._persist(self._finish(result))

    # --------------------------------------------------------------- internals

    def _submit(self, candidate: CandidateRecord) -> str | ResultRecord:
        """BRAIN's simulation id, or the failed/timeout result when submitting fails."""
        try:
            return self._with_retry(lambda: self.client.submit(candidate))
        except TRANSIENT_ERRORS as exc:
            return self._failure(candidate.candidate_id, None, exc, "submit")
        except SubmissionRejected as exc:
            return ResultRecord(candidate_id=candidate.candidate_id, status="failed",
                                notes=[f"submit_rejected: {exc}"])

    def _finish(self, result: ResultRecord) -> ResultRecord:
        leak = holdout_leak(result, self.config) if result.status == "complete" else None
        if leak:
            result.status = "invalid"  # never decide on numbers that saw the holdout
            result.notes.append(f"invalid: {leak}")
        if result.status in ("complete", "failed"):
            result.notes.extend(metric_validator.validate_result(result))
        return result

    def _persist(self, result: ResultRecord) -> ResultRecord:
        self.memory.put_result(result)
        return result

    def _failure(self, candidate_id: str, sim_id: str | None, exc: Exception,
                 stage: str) -> ResultRecord:
        # Exhausted retries: a TimeoutError becomes status "timeout", a ConnectionError "failed".
        status = "timeout" if isinstance(exc, TimeoutError) else "failed"
        return ResultRecord(candidate_id=candidate_id, status=status, brain_sim_id=sim_id,
                            notes=[f"{stage}_error: {type(exc).__name__} after retries: {exc}"])

    def _with_retry(self, fn):
        retries = self.config.simulator.max_retries
        for attempt in range(retries + 1):
            try:
                return fn()
            except TRANSIENT_ERRORS:
                if attempt == retries:
                    raise
                self.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))

    def _poll_once(self, job: _Job) -> tuple[ResultRecord | None, bool, float]:
        """One poll of a running simulation: (result or None while running, retryable,
        seconds to wait before the next poll). retryable when BRAIN failed with one of its
        own errors; a simulation running past ``max_poll_seconds`` becomes a timeout."""
        cfg = self.config.simulator
        cid, sim_id = job.candidate.candidate_id, job.sim_id
        try:
            raw = self._with_retry(lambda: self.client.poll(sim_id))
        except TRANSIENT_ERRORS as exc:
            return self._failure(cid, sim_id, exc, "poll"), False, 0.0
        if raw.get("status") in ("complete", "failed"):
            retryable = raw.get("status") == "failed" and bool(raw.get("retryable"))
            return normalize(raw, cid, sim_id), retryable, 0.0
        elapsed = self.clock() - job.started
        if elapsed >= cfg.max_poll_seconds:
            return ResultRecord(
                candidate_id=cid, status="timeout", brain_sim_id=sim_id,
                notes=[f"timeout: no completion after {elapsed:.0f}s "
                       f"(max_poll_seconds={cfg.max_poll_seconds:g})"],
            ), False, 0.0
        prog = raw.get("progress")
        job.progress = float(prog) if isinstance(prog, (int, float)) and not isinstance(prog, bool) \
            else job.progress
        wait = raw.get("retry_after")
        wait = float(wait) if isinstance(wait, (int, float)) and wait > 0 else cfg.poll_interval_seconds
        return None, False, wait
