"""Append-only SQLite memory store (spec section 11.1).

Rules:
* candidates, results, decisions, variants and critic_reviews are append-only. This class
  exposes no method that updates or deletes those rows.
* ``os_is`` may be upserted (explicitly allowed by the spec) via ``record_os_is``.
* Every write runs inside a transaction.

The database lives at ``$ALPHA_RESEARCH_DATA/memory.sqlite`` (default ``./data``).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .schemas import CandidateRecord, DecisionRecord, ResultRecord, experiment_hash

DATA_ENV = "ALPHA_RESEARCH_DATA"
DEFAULT_DATA_DIR = "./data"
DB_NAME = "memory.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    expression_hash TEXT NOT NULL,
    family TEXT,
    parent_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    failure_class TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS variants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS os_is (
    alpha_id TEXT PRIMARY KEY,
    family TEXT,
    is_sharpe REAL,
    os_sharpe REAL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_submissions (
    candidate_id TEXT PRIMARY KEY,
    brain_sim_id TEXT NOT NULL,
    submitted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS critic_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    candidate_id TEXT NOT NULL,
    critic_model TEXT,
    role TEXT NOT NULL,
    verdict TEXT NOT NULL,
    source TEXT NOT NULL,
    generator_model TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_critic_reviews_candidate ON critic_reviews(candidate_id);
CREATE INDEX IF NOT EXISTS ix_candidates_hash ON candidates(expression_hash);
CREATE INDEX IF NOT EXISTS ix_candidates_family ON candidates(family);
CREATE INDEX IF NOT EXISTS ix_results_candidate ON results(candidate_id);
CREATE INDEX IF NOT EXISTS ix_decisions_candidate ON decisions(candidate_id);
CREATE INDEX IF NOT EXISTS ix_decisions_failure_class ON decisions(failure_class);
CREATE INDEX IF NOT EXISTS ix_decisions_decision ON decisions(decision);
CREATE INDEX IF NOT EXISTS ix_variants_family ON variants(family);
CREATE INDEX IF NOT EXISTS ix_variants_candidate ON variants(candidate_id);
CREATE INDEX IF NOT EXISTS ix_os_is_family ON os_is(family);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_data_dir() -> Path:
    return Path(os.environ.get(DATA_ENV, DEFAULT_DATA_DIR))


class MemoryStore:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / DB_NAME
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ writes

    def put_candidate(self, candidate: CandidateRecord) -> None:
        """Insert a candidate. Re-inserting an identical payload is a no-op; a different
        payload under an existing id raises, since rows are never overwritten."""
        payload = json.dumps(candidate.to_dict(), sort_keys=True)
        with self.conn:
            row = self.conn.execute(
                "SELECT payload_json FROM candidates WHERE candidate_id = ?",
                (candidate.candidate_id,),
            ).fetchone()
            if row is not None:
                if row["payload_json"] != payload:
                    raise ValueError(
                        f"candidate {candidate.candidate_id!r} already stored with a different "
                        "payload; memory is append-only"
                    )
                return
            self.conn.execute(
                "INSERT INTO candidates (candidate_id, expression_hash, family, parent_id, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (candidate.candidate_id, experiment_hash(candidate), candidate.family,
                 candidate.parent_id, payload, _now()),
            )

    def put_pending(self, candidate_id: str, brain_sim_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO pending_submissions (candidate_id, brain_sim_id, submitted_at) "
                "VALUES (?, ?, ?)",
                (candidate_id, brain_sim_id, _now()),
            )

    def put_result(self, result: ResultRecord) -> int:
        result.validate()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO results (candidate_id, status, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (result.candidate_id, result.status, json.dumps(result.to_dict()), _now()),
            )
        return cur.lastrowid

    def put_decision(self, decision: DecisionRecord) -> int:
        decision.validate()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO decisions (candidate_id, decision, failure_class, payload_json, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (decision.candidate_id, decision.decision, decision.failure_class,
                 json.dumps(decision.to_dict()), decision.created_at.isoformat()),
            )
        return cur.lastrowid

    def register_variant(self, family: str, candidate_id: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO variants (family, candidate_id, created_at) VALUES (?, ?, ?)",
                (family, candidate_id, _now()),
            )
        return cur.lastrowid

    def put_critic_review(self, *, run_id: str | None, candidate_id: str,
                          critic_model: str | None, role: str, verdict: str, source: str,
                          generator_model: str | None, payload: dict | None = None) -> int:
        """One critic verdict. ``role`` is "primary" (it decided) or "shadow" (recorded only,
        for comparing critic models). ``critic_model`` is None for a rules-only critic."""
        if role not in ("primary", "shadow"):
            raise ValueError(f"critic review role must be primary or shadow, got {role!r}")
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO critic_reviews (run_id, candidate_id, critic_model, role, verdict, "
                "source, generator_model, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, candidate_id, critic_model, role, verdict, source, generator_model,
                 json.dumps(payload or {}), _now()),
            )
        return cur.lastrowid

    def record_os_is(self, alpha_id: str, family: str | None, is_sharpe: float | None,
                     os_sharpe: float | None) -> None:
        """Upsert, explicitly allowed for this table only."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO os_is (alpha_id, family, is_sharpe, os_sharpe, recorded_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(alpha_id) DO UPDATE SET "
                "family = excluded.family, is_sharpe = excluded.is_sharpe, "
                "os_sharpe = excluded.os_sharpe, recorded_at = excluded.recorded_at",
                (alpha_id, family, is_sharpe, os_sharpe, _now()),
            )

    # ------------------------------------------------------------------ reads

    def get_candidate(self, candidate_id: str) -> CandidateRecord | None:
        row = self.conn.execute(
            "SELECT payload_json FROM candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        return None if row is None else CandidateRecord.from_dict(json.loads(row["payload_json"]))

    def results_for(self, candidate_id: str) -> list[ResultRecord]:
        rows = self.conn.execute(
            "SELECT payload_json FROM results WHERE candidate_id = ? ORDER BY id",
            (candidate_id,),
        ).fetchall()
        return [ResultRecord.from_dict(json.loads(r["payload_json"])) for r in rows]

    def decisions_for(self, candidate_id: str) -> list[DecisionRecord]:
        rows = self.conn.execute(
            "SELECT payload_json FROM decisions WHERE candidate_id = ? ORDER BY id",
            (candidate_id,),
        ).fetchall()
        return [DecisionRecord.from_dict(json.loads(r["payload_json"])) for r in rows]

    def query_by_failure_class(self, failure_class: str) -> list[DecisionRecord]:
        rows = self.conn.execute(
            "SELECT payload_json FROM decisions WHERE failure_class = ? ORDER BY id",
            (failure_class,),
        ).fetchall()
        return [DecisionRecord.from_dict(json.loads(r["payload_json"])) for r in rows]

    def query_by_family(self, family: str) -> list[CandidateRecord]:
        rows = self.conn.execute(
            "SELECT payload_json FROM candidates WHERE family = ? ORDER BY rowid", (family,)
        ).fetchall()
        return [CandidateRecord.from_dict(json.loads(r["payload_json"])) for r in rows]

    def history_for_expression_hash(self, h: str) -> list[dict]:
        """Every candidate with this experiment hash, with its results and decisions."""
        rows = self.conn.execute(
            "SELECT candidate_id FROM candidates WHERE expression_hash = ? ORDER BY rowid", (h,)
        ).fetchall()
        return [
            {
                "candidate": self.get_candidate(r["candidate_id"]),
                "results": self.results_for(r["candidate_id"]),
                "decisions": self.decisions_for(r["candidate_id"]),
            }
            for r in rows
        ]

    def has_result_for_hash(self, h: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM results r JOIN candidates c ON c.candidate_id = r.candidate_id "
            "WHERE c.expression_hash = ? LIMIT 1",
            (h,),
        ).fetchone()
        return row is not None

    def pending(self) -> list[dict]:
        """Submissions that have a brain_sim_id but no result yet (crash recovery)."""
        rows = self.conn.execute(
            "SELECT p.candidate_id, p.brain_sim_id, p.submitted_at FROM pending_submissions p "
            "WHERE NOT EXISTS (SELECT 1 FROM results r WHERE r.candidate_id = p.candidate_id) "
            "ORDER BY p.submitted_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def unfinished_submissions(self) -> list[dict]:
        """Submissions BRAIN may have finished after we stopped waiting: candidates whose
        latest result is a ``timeout`` with a simulation id, plus pending submissions with no
        result at all (crash). ``{"candidate_id", "brain_sim_id"}``, oldest first. Read-only."""
        out = {p["candidate_id"]: p["brain_sim_id"] for p in self.pending()}
        rows = self.conn.execute(
            "SELECT r.candidate_id, r.payload_json FROM results r WHERE r.id = "
            "(SELECT MAX(r2.id) FROM results r2 WHERE r2.candidate_id = r.candidate_id) "
            "AND r.status = 'timeout' ORDER BY r.id").fetchall()
        for r in rows:
            sim_id = json.loads(r["payload_json"]).get("brain_sim_id")
            if sim_id:
                out[r["candidate_id"]] = sim_id
        return [{"candidate_id": c, "brain_sim_id": s} for c, s in out.items()]

    def pending_sim_id(self, candidate_id: str) -> str | None:
        for p in self.pending():
            if p["candidate_id"] == candidate_id:
                return p["brain_sim_id"]
        return None

    def variant_count(self, family: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM variants WHERE family = ?", (family,)
        ).fetchone()
        return int(row["n"])

    def os_is_rows(self, family: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT alpha_id, family, is_sharpe, os_sharpe, recorded_at FROM os_is "
            "WHERE family = ? ORDER BY alpha_id",
            (family,),
        ).fetchall()
        return [dict(r) for r in rows]

    def candidate_outcomes(self) -> list[dict]:
        """Every candidate with its latest decision and latest result, oldest first. Used
        by the Research Director (warm start) and Strategy Evolution. Read-only."""
        rows = self.conn.execute(
            "SELECT c.candidate_id, c.payload_json AS cand, "
            "(SELECT payload_json FROM decisions d WHERE d.candidate_id = c.candidate_id "
            " ORDER BY d.id DESC LIMIT 1) AS dec, "
            "(SELECT payload_json FROM results r WHERE r.candidate_id = c.candidate_id "
            " ORDER BY r.id DESC LIMIT 1) AS res "
            "FROM candidates c ORDER BY c.rowid"
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "candidate": CandidateRecord.from_dict(json.loads(r["cand"])),
                "decision": DecisionRecord.from_dict(json.loads(r["dec"])) if r["dec"] else None,
                "result": ResultRecord.from_dict(json.loads(r["res"])) if r["res"] else None,
            })
        return out

    def simulations_by_region(self, since: str | None = None) -> dict[str, dict]:
        """{region: {"simulations": n, "last": time}} for candidates that reached BRAIN
        (any result but ``invalid``), counting results recorded on or after ``since`` (ISO
        date or time). Used to rotate regions across research runs. Read-only."""
        rows = self.conn.execute(
            "SELECT c.candidate_id, c.payload_json AS cand, MAX(r.created_at) AS last "
            "FROM results r JOIN candidates c ON c.candidate_id = r.candidate_id "
            "WHERE r.status != 'invalid' AND r.created_at >= ? GROUP BY c.candidate_id",
            (since or "",)).fetchall()
        out: dict[str, dict] = {}
        for r in rows:
            region = json.loads(r["cand"]).get("region")
            info = out.setdefault(region, {"simulations": 0, "last": None})
            info["simulations"] += 1
            info["last"] = max(info["last"] or "", r["last"])
        return out

    def critic_reviews(self) -> list[dict]:
        """Every critic verdict, oldest first. Read-only."""
        rows = self.conn.execute(
            "SELECT run_id, candidate_id, critic_model, role, verdict, source, generator_model, "
            "payload_json FROM critic_reviews ORDER BY id").fetchall()
        return [{**{k: r[k] for k in r.keys() if k != "payload_json"},
                 "payload": json.loads(r["payload_json"])} for r in rows]

    def lineage_levers(self, candidate_id: str) -> list[str]:
        """Lever names applied along the parent chain, oldest first, including this one."""
        levers: list[str] = []
        seen: set[str] = set()
        current = self.get_candidate(candidate_id)
        while current is not None and current.candidate_id not in seen:
            seen.add(current.candidate_id)
            if current.lever:
                levers.append(current.lever)
            current = self.get_candidate(current.parent_id) if current.parent_id else None
        return list(reversed(levers))
