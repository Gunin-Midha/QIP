"""Archive the research memory and rebuild it from the rows recorded since a date.

Owner, 2026-11-14: the simulations from before the consultant upgrade (2026-11-10) used
BRAIN's old period (2019-2023) and limits (Sharpe 1.25), so they should no longer inform
the agents. Memory is append-only (MemoryStore has no delete), so nothing is deleted: the
whole data directory is moved to the archive, and a new memory is built with the rows on
or after --since, in their original order. PnL files go along for the pool alphas kept,
and run reports (rs<date>...json) from --since on are copied back.

    PYTHONPATH=src python3 scripts/split_memory.py --since 2026-11-10 \\
        --archive data_archive/pre_consultant_20261114

Restore: move the archive back to data/.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

# table -> column holding its time (rows on or after --since are kept)
TABLES = {"candidates": "created_at", "results": "created_at", "decisions": "created_at",
          "variants": "created_at", "critic_reviews": "created_at",
          "pending_submissions": "submitted_at", "pool": "admitted_at", "os_is": "recorded_at"}


def counts(db: Path, since: str | None = None) -> dict[str, int]:
    con = sqlite3.connect(db)
    try:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}" + (f" WHERE {c} >= ?" if since else ""),
                               (since,) if since else ()).fetchone()[0]
                for t, c in TABLES.items()}
    finally:
        con.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--since", required=True, help="keep rows recorded on or after this date")
    ap.add_argument("--archive", required=True, help="where the current data directory goes")
    args = ap.parse_args(argv)
    data, archive = Path(args.data_dir), Path(args.archive)
    if not (data / "memory.sqlite").exists():
        print(f"no memory at {data / 'memory.sqlite'}", file=sys.stderr)
        return 2
    if archive.exists():
        print(f"archive {archive} already exists; choose another path", file=sys.stderr)
        return 2
    keep = counts(data / "memory.sqlite", args.since)
    total = counts(data / "memory.sqlite")

    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(data), str(archive))
    print(f"archived {data}/ -> {archive}/")

    data.mkdir()
    con = sqlite3.connect(data / "memory.sqlite")
    try:
        con.execute("ATTACH DATABASE ? AS old", (str(archive / "memory.sqlite"),))
        # the old database's exact schema (MemoryStore's tables plus the pool's), tables first
        schema = con.execute("SELECT sql FROM old.sqlite_master WHERE sql IS NOT NULL AND "
                             "name NOT LIKE 'sqlite_%' ORDER BY type = 'index'").fetchall()
        for (sql,) in schema:
            con.execute(sql)
        for table, col in TABLES.items():
            con.execute(f"INSERT INTO main.{table} SELECT * FROM old.{table} "
                        f"WHERE {col} >= ? ORDER BY rowid", (args.since,))
        con.commit()
        pnl = [p for (p,) in con.execute("SELECT pnl_path FROM main.pool WHERE pnl_path IS NOT NULL")]
    finally:
        con.close()
    for p in pnl:                                  # stored as "data/pnl/<id>.csv"
        src = archive / Path(p).relative_to(data)
        dst = data / Path(p).relative_to(data)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    day = args.since.replace("-", "")
    reports = [f for f in archive.glob("rs*.json") if f.stem[2:10] >= day]   # rsYYYYMMDD...
    for f in reports:
        shutil.copy2(f, data / f.name)

    got = counts(data / "memory.sqlite")
    if got != keep:
        print(f"row counts differ from what was expected: {got} vs {keep}", file=sys.stderr)
        return 1
    for t in TABLES:
        print(f"  {t:20s} kept {got[t]:4d} of {total[t]}")
    print(f"pool PnL files copied: {len(pnl)}; run reports copied: {len(reports)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
