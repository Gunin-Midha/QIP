"""Pool of admitted alphas (spec section 11.2).

Shares the memory store's SQLite file. Daily PnL is one CSV per alpha under
``<data_dir>/pnl/``; the pool table stores the path.

An alpha leaves the pool when its decision is no longer accept (2026-11-25: an alpha
accepted while a BRAIN check had no verdict stayed in the pool after it was re-decided).
Leaving is an append-only ``pool_events`` row ("retire", later "restore" if it is accepted
again; "submitted" marks an alpha the owner added by hand); the pool rows themselves are never updated or deleted, and every reader counts
only active alphas (latest event not "retire").
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .diversity import pnl_matrix
from .memory import MemoryStore
from .schemas import CandidateRecord, DecisionRecord, ResultRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pool (
    alpha_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    family TEXT,
    admitted_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    pnl_path TEXT
);
CREATE INDEX IF NOT EXISTS ix_pool_family ON pool(family);
CREATE TABLE IF NOT EXISTS pool_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alpha_id TEXT NOT NULL,
    event TEXT NOT NULL,
    at TEXT NOT NULL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS ix_pool_events_alpha ON pool_events(alpha_id);
"""

# an alpha is active unless its latest pool event is "retire"
_ACTIVE = ("NOT EXISTS (SELECT 1 FROM pool_events e WHERE e.alpha_id = pool.alpha_id AND "
           "e.id = (SELECT MAX(id) FROM pool_events WHERE alpha_id = pool.alpha_id) AND "
           "e.event = 'retire')")

_METRICS = ("sharpe", "fitness", "turnover", "returns", "margin_bps", "max_drawdown",
            "sub_universe_sharpe", "weight_concentration", "self_correlation")


class Pool:
    def __init__(self, memory: MemoryStore):
        self.memory = memory
        self.conn = memory.conn
        self.pnl_dir = memory.data_dir / "pnl"
        self.pnl_dir.mkdir(parents=True, exist_ok=True)
        with self.conn:
            self.conn.executescript(_SCHEMA)

    def admit(self, candidate: CandidateRecord, result: ResultRecord, decision: DecisionRecord,
              diversity_passed: bool | None) -> bool:
        """Write only if the decision is accept and the diversity admission passed. An alpha
        already in the pool is not written again; a retired one is restored."""
        if decision.decision != "accept" or diversity_passed is not True:
            return False
        if decision.candidate_id != candidate.candidate_id or result.candidate_id != candidate.candidate_id:
            raise ValueError("candidate, result and decision refer to different candidates")
        alpha_id = candidate.candidate_id
        if self._known(alpha_id):
            if self.is_active(alpha_id):
                return False
            self._event(alpha_id, "restore", "accepted again")
            return True
        pnl_path = None
        if result.daily_pnl is not None:
            path = self.pnl_dir / f"{alpha_id}.csv"
            pd.DataFrame({"date": result.daily_pnl.index.strftime("%Y-%m-%d"),
                          "pnl": result.daily_pnl.values}).to_csv(path, index=False)
            pnl_path = str(path)
        metrics = {m: getattr(result, m) for m in _METRICS}
        with self.conn:
            self.conn.execute(
                "INSERT INTO pool (alpha_id, candidate_id, family, admitted_at, metrics_json, "
                "pnl_path) VALUES (?, ?, ?, ?, ?, ?)",
                (alpha_id, candidate.candidate_id, candidate.family,
                 datetime.now(timezone.utc).isoformat(), json.dumps(metrics), pnl_path),
            )
        return True

    def _known(self, alpha_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM pool WHERE alpha_id = ?",
                                 (alpha_id,)).fetchone() is not None

    def is_active(self, alpha_id: str) -> bool:
        return self.conn.execute(f"SELECT 1 FROM pool WHERE alpha_id = ? AND {_ACTIVE}",
                                 (alpha_id,)).fetchone() is not None

    def _event(self, alpha_id: str, event: str, reason: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO pool_events (alpha_id, event, at, reason) VALUES (?, ?, ?, ?)",
                (alpha_id, event, datetime.now(timezone.utc).isoformat(), reason))

    def retire(self, alpha_id: str, reason: str) -> bool:
        """Take an active alpha out of the pool (an append-only event). False if it is not
        an active pool alpha."""
        if not self.is_active(alpha_id):
            return False
        self._event(alpha_id, "retire", reason)
        return True

    def add_submitted(self, candidate_id: str) -> str:
        """Put an alpha the owner submitted on BRAIN in the pool, from its latest result in
        memory (whatever our decision was: BRAIN took it). Returns "added", "restored" or
        "already in the pool"; raises ValueError when memory has no complete result."""
        c = self.memory.get_candidate(candidate_id)
        results = self.memory.results_for(candidate_id) if c is not None else []
        r = results[-1] if results else None
        if c is None or r is None or r.status != "complete":
            raise ValueError(f"{candidate_id}: no complete result in memory")
        if self.is_active(candidate_id):
            return "already in the pool"
        if self._known(candidate_id):
            self._event(candidate_id, "restore", "submitted on BRAIN (owner)")
            return "restored"
        self.admit(c, r, DecisionRecord(candidate_id, "accept", "none"), True)
        self._event(candidate_id, "submitted", "submitted on BRAIN (owner)")
        return "added"

    def sync(self, admit_accepted: bool) -> dict[str, list[str]]:
        """Make the pool match the latest decisions in memory: retire active alphas whose
        latest decision is not accept; with ``admit_accepted`` (``gates.brain_checks_only``,
        where our diversity check is recorded only) admit complete accepted alphas that
        are missing. Returns {"admitted": [...], "retired": [...]}."""
        admitted, retired = [], []
        for row in self.memory.candidate_outcomes():
            c, d, r = row["candidate"], row["decision"], row["result"]
            cid = c.candidate_id
            if d is None:
                continue
            if d.decision != "accept":
                if self.retire(cid, f"latest decision {d.decision} / {d.failure_class}"):
                    retired.append(cid)
            elif (admit_accepted and r is not None and r.status == "complete"
                  and not self.is_active(cid)
                  and self.admit(c, r, d, diversity_passed=True)):
                admitted.append(cid)
        return {"admitted": admitted, "retired": retired}

    def _load_pnl(self, path: str) -> pd.Series:
        df = pd.read_csv(path)
        return pd.Series(df["pnl"].astype(float).values,
                         index=pd.DatetimeIndex(pd.to_datetime(df["date"])))

    def matrix(self) -> pd.DataFrame:
        rows = self.conn.execute(
            f"SELECT alpha_id, pnl_path FROM pool WHERE pnl_path IS NOT NULL AND {_ACTIVE} "
            "ORDER BY admitted_at"
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pnl_matrix({r["alpha_id"]: self._load_pnl(r["pnl_path"]) for r in rows})

    def count(self) -> int:
        return int(self.conn.execute(f"SELECT COUNT(*) AS n FROM pool WHERE {_ACTIVE}").fetchone()["n"])

    def families(self) -> dict[str | None, int]:
        rows = self.conn.execute(
            f"SELECT family, COUNT(*) AS n FROM pool WHERE {_ACTIVE} GROUP BY family ORDER BY family"
        ).fetchall()
        return {r["family"]: int(r["n"]) for r in rows}

    def alpha_ids(self) -> list[str]:
        return [r["alpha_id"] for r in
                self.conn.execute(f"SELECT alpha_id FROM pool WHERE {_ACTIVE} "
                                  "ORDER BY admitted_at").fetchall()]
