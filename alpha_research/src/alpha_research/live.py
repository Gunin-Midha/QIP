"""Live status of a running batch or research run, for the dashboard (owner 2026-11-28).

A run writes ``<data dir>/live/<run id>.json`` as things happen: console lines, the alphas on
BRAIN (elapsed time and BRAIN's progress), each result and decision, the models' state and the
budgets. The file is replaced in one step (written to a temporary file, then renamed), so a
reader never sees half of it. Nothing here talks to the network; the dashboard
(``dashboard.py``) only reads these files.

Method
1. ``LiveStatus`` keeps the last ``max_events`` console lines and ``max_decisions`` decisions,
   the alphas running now, and a ``state`` provider (a function returning the run's own
   numbers: round, regions, budgets, models). Every change writes the file, at most once per
   ``min_interval`` seconds unless forced (start, decisions, the end).
2. ``finish(state, reason)`` marks the run finished or stopped. The file also holds the
   process id, so the dashboard can tell a run that stopped without finishing (killed, crash)
   from one still going.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

LIVE_DIR = "live"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


class LiveStatus:
    """See the module doc. ``kind``: "research" or "batch" (``cli run``)."""

    def __init__(self, data_dir: str | Path, run_id: str, *, kind: str = "research",
                 state: Callable[[], dict] | None = None, clock: Callable[[], float] = time.time,
                 min_interval: float = 1.0, max_events: int = 300, max_decisions: int = 300):
        self.dir = Path(data_dir) / LIVE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.json"
        self.run_id, self.kind = run_id, kind
        self.state = state
        self.clock = clock
        self.min_interval = min_interval
        self.started_at = _now_iso()
        self.events: deque = deque(maxlen=max_events)
        self.decisions: deque = deque(maxlen=max_decisions)
        self.running: dict[str, dict] = {}
        self.waiting = 0
        self.status = "running"
        self.reason: str | None = None
        self.counts = {"simulated": 0, "accept": 0, "repair": 0, "reject": 0, "pending": 0,
                       "duplicate": 0}
        self._last_write = 0.0
        self.write(force=True)

    # ------------------------------------------------------------------ events

    def event(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            if line.strip():
                self.events.append({"t": _now_iso(), "text": line})
        self.write()

    def batch(self, candidates) -> None:
        """Alphas about to go to BRAIN together."""
        for c in candidates:
            self.running.setdefault(c.candidate_id, self._alpha(c))
        self.write(force=True)

    def status_snapshot(self, snap: dict) -> None:
        """The simulator's status (``Simulator.run_many(on_status=)``)."""
        for cid, elapsed, progress in snap.get("running", []):
            row = self.running.setdefault(cid, {"candidate_id": cid})
            row["elapsed"] = round(elapsed, 1) if _num(elapsed) is not None else None
            row["progress"] = _num(progress)
            if row.get("phase") in (None, "queued"):
                row["phase"] = "simulating"
        self.waiting = snap.get("waiting", 0)
        self.write()

    def fetching(self, candidate_id: str) -> None:
        self.running.setdefault(candidate_id, {"candidate_id": candidate_id})["phase"] = "fetching"
        self.write()

    def decided(self, candidate, outcome) -> None:
        """A decision (``pipeline.Outcome``): the alpha leaves the running list."""
        self.running.pop(candidate.candidate_id, None)
        r = outcome.result
        row = {"t": _now_iso(), "candidate_id": candidate.candidate_id,
               "region": candidate.region, "dataset": candidate.dataset,
               "lever": candidate.lever, "model": candidate.model,
               "decision": outcome.decision, "failure_class": outcome.failure_class,
               "status": r.status if r is not None else None}
        if r is not None:
            row.update(sharpe=_num(r.sharpe), fitness=_num(r.fitness),
                       turnover=_num(r.turnover))
        self.decisions.append(row)
        if outcome.decision in self.counts:
            self.counts[outcome.decision] += 1
        if r is not None and r.status not in ("invalid",) and outcome.decision != "duplicate":
            self.counts["simulated"] += 1
        self.write(force=True)

    def finish(self, status: str = "finished", reason: str | None = None) -> None:
        self.status, self.reason = status, reason
        self.running.clear()
        self.waiting = 0
        self.write(force=True)

    # ------------------------------------------------------------------ file

    @staticmethod
    def _alpha(c) -> dict:
        return {"candidate_id": c.candidate_id, "region": c.region, "dataset": c.dataset,
                "lever": c.lever, "model": c.model, "phase": "queued", "elapsed": None,
                "progress": None}

    def snapshot(self) -> dict:
        state = {}
        if self.state is not None:
            try:
                state = self.state() or {}
            except Exception as exc:   # a status provider must never break the run
                state = {"state_error": str(exc)}
        return {"run_id": self.run_id, "kind": self.kind, "pid": os.getpid(),
                "status": self.status, "reason": self.reason,
                "started_at": self.started_at, "updated_at": _now_iso(),
                "counts": dict(self.counts), "running": list(self.running.values()),
                "waiting": self.waiting, "state": state,
                "decisions": list(self.decisions), "events": list(self.events)}

    def write(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._last_write < self.min_interval:
            return
        self._last_write = now
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(self.snapshot(), default=str))
            os.replace(tmp, self.path)
        except OSError:   # a full disk or a removed folder must not stop the run
            pass
