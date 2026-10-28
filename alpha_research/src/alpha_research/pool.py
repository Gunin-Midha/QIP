"""Pool of admitted alphas (spec section 11.2).

Shares the memory store's SQLite file. Daily PnL is one CSV per alpha under
``<data_dir>/pnl/``; the pool table stores the path.
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
"""

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
        """Write only if the decision is accept and the diversity admission passed."""
        if decision.decision != "accept" or diversity_passed is not True:
            return False
        if decision.candidate_id != candidate.candidate_id or result.candidate_id != candidate.candidate_id:
            raise ValueError("candidate, result and decision refer to different candidates")
        alpha_id = candidate.candidate_id
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

    def _load_pnl(self, path: str) -> pd.Series:
        df = pd.read_csv(path)
        return pd.Series(df["pnl"].astype(float).values,
                         index=pd.DatetimeIndex(pd.to_datetime(df["date"])))

    def matrix(self) -> pd.DataFrame:
        rows = self.conn.execute(
            "SELECT alpha_id, pnl_path FROM pool WHERE pnl_path IS NOT NULL ORDER BY admitted_at"
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pnl_matrix({r["alpha_id"]: self._load_pnl(r["pnl_path"]) for r in rows})

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM pool").fetchone()["n"])

    def families(self) -> dict[str | None, int]:
        rows = self.conn.execute(
            "SELECT family, COUNT(*) AS n FROM pool GROUP BY family ORDER BY family"
        ).fetchall()
        return {r["family"]: int(r["n"]) for r in rows}

    def alpha_ids(self) -> list[str]:
        return [r["alpha_id"] for r in
                self.conn.execute("SELECT alpha_id FROM pool ORDER BY admitted_at").fetchall()]
