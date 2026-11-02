"""Simulator wrapper (spec section 4.2).

Steps per candidate: validate locally, submit (or resume a pending submission), poll with
retries, normalize the raw dict into a ResultRecord, run the metric validator, persist,
return.

Time is injected (``sleep`` and ``clock``) so fixture runs can use ``VirtualClock`` and
never wait in real time. Single-threaded; ``simulator.max_concurrent`` is not used yet
(OQ-3).
"""

from __future__ import annotations

import logging
import time
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


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def _to_float(value, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"raw field {key!r} is not numeric: {value!r}")
    return float(value)


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
        **values,
    )
    result.validate()
    return result


# ---------------------------------------------------------------------------
# wrapper
# ---------------------------------------------------------------------------

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
        errors = local_validation_errors(candidate, self.config)  # may raise ConfigError
        if errors:
            result = ResultRecord(candidate_id=candidate.candidate_id, status="invalid",
                                  notes=[f"invalid: {e}" for e in errors])
            self.memory.put_candidate(candidate)
            self.memory.put_result(result)
            return result

        self.memory.put_candidate(candidate)
        sim_id = self.memory.pending_sim_id(candidate.candidate_id)
        if sim_id is not None:
            log.info("resuming pending submission %s for %s", sim_id, candidate.candidate_id)
        else:
            try:
                sim_id = self._with_retry(lambda: self.client.submit(candidate))
            except TRANSIENT_ERRORS as exc:
                return self._persist(self._failure(candidate.candidate_id, None, exc, "submit"))
            except SubmissionRejected as exc:
                return self._persist(ResultRecord(candidate_id=candidate.candidate_id,
                                                  status="failed", notes=[f"submit_rejected: {exc}"]))
            self.memory.put_pending(candidate.candidate_id, sim_id)

        result = self._poll_until_done(candidate.candidate_id, sim_id)
        if result.status in ("complete", "failed"):
            result.notes.extend(metric_validator.validate_result(result))
        return self._persist(result)

    # --------------------------------------------------------------- internals

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

    def _poll_until_done(self, candidate_id: str, sim_id: str) -> ResultRecord:
        cfg = self.config.simulator
        started = self.clock()
        while True:
            try:
                raw = self._with_retry(lambda: self.client.poll(sim_id))
            except TRANSIENT_ERRORS as exc:
                return self._failure(candidate_id, sim_id, exc, "poll")
            if raw.get("status") in ("complete", "failed"):
                return normalize(raw, candidate_id, sim_id)
            elapsed = self.clock() - started
            if elapsed >= cfg.max_poll_seconds:
                return ResultRecord(
                    candidate_id=candidate_id, status="timeout", brain_sim_id=sim_id,
                    notes=[f"timeout: no completion after {elapsed:.0f}s "
                           f"(max_poll_seconds={cfg.max_poll_seconds:g})"],
                )
            wait = raw.get("retry_after")
            wait = float(wait) if isinstance(wait, (int, float)) and wait > 0 else cfg.poll_interval_seconds
            self.sleep(wait)
