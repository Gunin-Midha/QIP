import sqlite3

import pytest

from alpha_research.memory import MemoryStore
from alpha_research.schemas import DecisionRecord, ResultRecord, experiment_hash

FAMILIES = ["price_volume", "fundamentals", "analyst"]
CLASSES = ["sharpe", "turnover", "sub_universe", "self_correlation"]


def test_query_by_failure_class_exact_rows(memory, candidate):
    expected = {c: [] for c in CLASSES}
    for i in range(50):
        cid = f"c{i:02d}"
        fam, fc = FAMILIES[i % 3], CLASSES[i % 4]
        memory.put_candidate(candidate(candidate_id=cid, family=fam, expression=f"rank(x{i})"))
        memory.put_result(ResultRecord(cid, "complete", sharpe=1.0))
        memory.put_decision(DecisionRecord(cid, "reject", fc))
        expected[fc].append(cid)
    for fc in CLASSES:
        got = [d.candidate_id for d in memory.query_by_failure_class(fc)]
        assert got == expected[fc]
    assert len(memory.query_by_family("price_volume")) == 17
    assert memory.query_by_failure_class("drawdown") == []


def test_history_for_expression_hash(memory, candidate):
    c = candidate()
    memory.put_candidate(c)
    memory.put_result(ResultRecord("c1", "complete", sharpe=1.4))
    h = experiment_hash(c)
    hist = memory.history_for_expression_hash(h)
    assert len(hist) == 1 and hist[0]["results"][0].sharpe == 1.4
    assert memory.has_result_for_hash(h)


def test_pending_survives_crash(tmp_path, candidate):
    data = tmp_path / "data"
    s = MemoryStore(data)
    s.put_candidate(candidate())
    s.put_pending("c1", "sim-123")
    s.close()                               # simulated crash
    s = MemoryStore(data)
    assert s.pending() == [{"candidate_id": "c1", "brain_sim_id": "sim-123",
                            "submitted_at": s.pending()[0]["submitted_at"]}]
    s.put_result(ResultRecord("c1", "complete"))
    assert s.pending() == []
    s.close()


def test_no_update_or_delete_methods():
    names = [n for n in dir(MemoryStore) if not n.startswith("__")]
    assert not [n for n in names if "update" in n.lower() or "delete" in n.lower()
                or "remove" in n.lower()]


def test_results_and_decisions_are_append_only(memory, candidate):
    memory.put_candidate(candidate())
    memory.put_result(ResultRecord("c1", "timeout"))
    memory.put_result(ResultRecord("c1", "complete", sharpe=1.5))
    assert [r.status for r in memory.results_for("c1")] == ["timeout", "complete"]


def test_candidate_overwrite_refused(memory, candidate):
    memory.put_candidate(candidate())
    memory.put_candidate(candidate())             # identical: no-op
    with pytest.raises(ValueError):
        memory.put_candidate(candidate(decay=9))


def test_indexes_exist(memory):
    idx = {r[0] for r in memory.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    for name in ("ix_decisions_failure_class", "ix_decisions_decision", "ix_variants_family",
                 "ix_results_candidate"):
        assert name in idx


def test_data_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA_RESEARCH_DATA", str(tmp_path / "envdata"))
    s = MemoryStore()
    assert s.path == tmp_path / "envdata" / "memory.sqlite"
    s.close()


def test_lineage_levers(memory, candidate):
    memory.put_candidate(candidate(candidate_id="p"))
    memory.put_candidate(candidate(candidate_id="k1", parent_id="p", generation=1,
                                   lever="increase_decay", decay=8))
    memory.put_candidate(candidate(candidate_id="k2", parent_id="k1", generation=2,
                                   lever="lengthen_ts_window", decay=8,
                                   expression="rank(-ts_delta(close, 10))"))
    assert memory.lineage_levers("k2") == ["increase_decay", "lengthen_ts_window"]


def test_candidate_outcomes_latest_decision_and_result(memory, candidate):
    from alpha_research.schemas import DecisionRecord, ResultRecord
    memory.put_candidate(candidate(candidate_id="a"))
    memory.put_candidate(candidate(candidate_id="b", expression="rank(volume)"))
    memory.put_result(ResultRecord(candidate_id="a", status="timeout"))
    memory.put_result(ResultRecord(candidate_id="a", status="complete", sharpe=1.5))
    memory.put_decision(DecisionRecord("a", "pending", "none"))
    memory.put_decision(DecisionRecord("a", "accept", "none"))
    rows = memory.candidate_outcomes()
    assert [r["candidate"].candidate_id for r in rows] == ["a", "b"]
    assert rows[0]["decision"].decision == "accept" and rows[0]["result"].sharpe == 1.5
    assert rows[1]["decision"] is None and rows[1]["result"] is None
