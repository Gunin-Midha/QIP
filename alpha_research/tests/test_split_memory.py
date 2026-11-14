"""scripts/split_memory.py: archive the memory, rebuild it from the rows since a date."""

import importlib.util
import json
import sqlite3
from pathlib import Path

from alpha_research.memory import MemoryStore
from alpha_research.pool import Pool
from alpha_research.schemas import DecisionRecord, ResultRecord

_spec = importlib.util.spec_from_file_location(
    "split_memory", Path(__file__).parents[1] / "scripts" / "split_memory.py")
split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split)


def test_split_keeps_rows_since_the_date_and_archives_everything(tmp_path, candidate, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = Path("data")
    m = MemoryStore(data)
    Pool(m)                                                    # the pool's own table
    for cid in ("old", "new"):
        m.put_candidate(candidate(candidate_id=cid))
        m.put_result(ResultRecord(cid, "complete", sharpe=1.0))
        m.put_decision(DecisionRecord(cid, "accept", "none"))
    for cid, when in (("old", "2026-11-03T10:00:00"), ("new", "2026-11-12T10:00:00")):
        (data / "pnl").mkdir(exist_ok=True)
        (data / "pnl" / f"{cid}.csv").write_text("date,pnl\n2020-01-02,1.0\n")
        m.conn.execute("INSERT INTO pool VALUES (?, ?, 'f', ?, '{}', ?)",
                       (cid, cid, when, f"data/pnl/{cid}.csv"))
        for t in ("candidates", "results", "decisions"):        # test only: date the rows
            m.conn.execute(f"UPDATE {t} SET created_at = ? WHERE candidate_id = ?", (when, cid))
    m.conn.commit()
    m.close()
    for run in ("rs20261103171304", "rs20261114135300"):
        (data / f"{run}.json").write_text(json.dumps({"run_id": run}))

    assert split.main(["--since", "2026-11-10", "--archive", "arch/pre"]) == 0

    con = sqlite3.connect(data / "memory.sqlite")
    for t in ("candidates", "results", "decisions", "pool"):
        assert [r[0] for r in con.execute(f"SELECT candidate_id FROM {t}")] == ["new"]
    con.close()
    assert sorted(p.name for p in (data / "pnl").iterdir()) == ["new.csv"]
    assert sorted(p.name for p in data.glob("rs*.json")) == ["rs20261114135300.json"]
    # the archive is the untouched original, and the new memory opens normally
    assert split.counts(Path("arch/pre/memory.sqlite"))["candidates"] == 2
    assert [o["candidate"].candidate_id for o in MemoryStore(data).candidate_outcomes()] == ["new"]
    # refuses to overwrite an archive
    assert split.main(["--since", "2026-11-10", "--archive", "arch/pre"]) == 2
