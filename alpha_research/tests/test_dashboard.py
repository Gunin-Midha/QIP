"""Dashboard (owner 2026-11-28): live status file, read-only data layer and the local server's
handler. No sockets: the handler is driven with in-memory streams."""

import io
import json
import os
from types import SimpleNamespace as N

import pandas as pd
import pytest

from alpha_research import dashboard
from alpha_research.dashboard_data import DashboardData, origin_of, run_time
from alpha_research.live import LiveStatus
from alpha_research.pipeline import Outcome
from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord, YearRow

from test_research import make_run


def _cand(cid, **kw):
    return N(candidate_id=cid, region=kw.get("region", "USA"), dataset="pv1",
             lever=kw.get("lever"), model="m")


# ------------------------------------------------------------------ live status

def test_live_status_writes_whole_files_with_events_running_and_decisions(tmp_path):
    t = [0.0]
    lv = LiveStatus(tmp_path, "rs20261128000000", state=lambda: {"round": 2}, clock=lambda: t[0])
    path = tmp_path / "live" / "rs20261128000000.json"
    assert json.loads(path.read_text())["status"] == "running"
    lv.batch([_cand("a1"), _cand("a2", lever="hump")])
    lv.status_snapshot({"running": [("a1", 61.0, 0.4)], "waiting": 1, "finished": 0, "total": 2})
    lv.fetching("a2")
    t[0] = 5.0
    lv.event("--- round 2: 4 new")
    snap = json.loads(path.read_text())
    run = {r["candidate_id"]: r for r in snap["running"]}
    assert run["a1"]["phase"] == "simulating" and run["a1"]["progress"] == 0.4
    assert run["a2"]["phase"] == "fetching" and snap["waiting"] == 1
    assert snap["state"] == {"round": 2} and snap["pid"] == os.getpid()
    assert snap["events"][-1]["text"] == "--- round 2: 4 new"
    lv.decided(_cand("a1"), Outcome("a1", "accept", "none",
                                    ResultRecord("a1", "complete", sharpe=2.0, fitness=1.2)))
    snap = json.loads(path.read_text())
    assert [r["candidate_id"] for r in snap["running"]] == ["a2"]
    assert snap["counts"]["accept"] == 1 and snap["counts"]["simulated"] == 1
    assert snap["decisions"][-1]["sharpe"] == 2.0
    lv.finish("stopped", "stopped by hand")
    snap = json.loads(path.read_text())
    assert snap["status"] == "stopped" and snap["running"] == []
    assert not list((tmp_path / "live").glob("*.tmp"))


def test_a_failing_state_provider_never_breaks_the_run(tmp_path):
    def broken():
        raise ValueError("boom")
    lv = LiveStatus(tmp_path, "r1", state=broken)
    assert "boom" in json.loads(lv.path.read_text())["state"]["state_error"]


def test_a_research_run_reports_to_its_live_file(memory, config, playbook, tmp_path):
    run = make_run(memory, config, playbook)
    run.attach_live(LiveStatus(tmp_path, run.run_id))
    report = run.run(1)
    snap = json.loads((tmp_path / "live" / f"{run.run_id}.json").read_text())
    assert snap["status"] == "finished" and snap["state"]["round"] == 1
    assert snap["counts"]["simulated"] == report["simulations_used"] > 0
    assert len(snap["decisions"]) >= report["simulations_used"]
    assert any("round 1" in e["text"] for e in snap["events"])
    assert set(snap["state"]["models"]) and snap["state"]["simulations"]["max"] == 8


# ------------------------------------------------------------------ data layer

def _store(memory, candidate, cid, parent=None, lever=None, sharpe=1.0, decision="reject",
           cls="sharpe", checks=None, notes=None, region="USA"):
    memory.put_candidate(candidate(candidate_id=cid, parent_id=parent, lever=lever,
                                   region=region, expression=f"rank(ts_mean(close, {len(cid)}))"))
    pnl = pd.Series([1.0, -0.5, 2.0], index=pd.to_datetime(["2020-01-02", "2020-01-03",
                                                             "2020-01-06"]))
    memory.put_result(ResultRecord(cid, "complete", brain_sim_id="sim" + cid, sharpe=sharpe,
                                   fitness=sharpe / 2, turnover=0.2, returns=0.05,
                                   checks=checks or {"LOW_SHARPE": sharpe >= 1.58},
                                   check_values={"LOW_SHARPE": {"value": sharpe, "limit": 1.58}},
                                   notes=notes or [], daily_pnl=pnl,
                                   yearly=[YearRow(2020, sharpe, 0.2, 1.0, 0.05, 0.1, 5.0, 10, 10)]))
    memory.put_decision(DecisionRecord(cid, decision, cls, gate_results=[
        GateResult("sharpe", sharpe >= 1.58, sharpe, 1.58, "")]))


def test_alphas_runs_detail_and_lineage_from_memory(memory, candidate):
    rid = "rs20261127153052"
    _store(memory, candidate, f"{rid}r01a1_001", sharpe=1.2, decision="repair")
    _store(memory, candidate, f"{rid}r01a1_001_r1a", parent=f"{rid}r01a1_001",
           lever="hump", sharpe=1.9, decision="accept", cls="none")
    _store(memory, candidate, "hand_20261128_x", sharpe=-0.4, region="IND",
           notes=["brain_check_not_counted: PROD_CORRELATION='PENDING'"], decision="pending",
           cls="none")
    data = DashboardData(memory.data_dir)
    rows = {r["candidate_id"]: r for r in data.alphas()["alphas"]}
    assert len(rows) == 3
    child = rows[f"{rid}r01a1_001_r1a"]
    assert child["origin"] == rid and child["decision"] == "accept" and child["sharpe"] == 1.9
    assert rows[f"{rid}r01a1_001"]["failed_checks"][0]["name"] == "LOW_SHARPE"
    assert rows["hand_20261128_x"]["pending_checks"] == ["PROD_CORRELATION"]
    assert rows["hand_20261128_x"]["origin"] == "hand 2026-11-28"

    one = data.alpha(f"{rid}r01a1_001_r1a")
    assert one["pnl"]["values"] == [1.0, 0.5, 2.5]              # cumulative
    assert one["lineage"]["ancestors"] == [f"{rid}r01a1_001", f"{rid}r01a1_001_r1a"]
    assert one["lineage"]["tree"]["children"][0]["lever"] == "hump"
    assert one["checks"][0] == {"name": "LOW_SHARPE", "passed": True, "value": 1.9,
                                "limit": 1.58}
    assert data.alpha("nope") is None

    (memory.data_dir / f"{rid}.json").write_text(json.dumps({
        "run_id": rid, "stop_reason": "simulation budget used up", "simulations_used": 2,
        "model_calls": 5, "models_down": {"gemini-x": "quota"}, "regions_this_run": ["USA"]}))
    runs = {r["run_id"]: r for r in data.runs()["runs"]}
    assert runs[rid]["accepted"] == 1 and runs[rid]["alphas"] == 2
    assert runs[rid]["models_down"] == ["gemini-x"] and runs[rid]["best_sharpe"] == 1.9
    assert runs["hand 2026-11-28"]["kind"] == "hand" and runs["hand 2026-11-28"]["alphas"] == 1

    ov = data.overview()
    assert ov["totals"]["simulated"] == 3 and ov["totals"]["accepted"] == 1
    assert ov["totals"]["awaiting_checks"] == 1
    assert data.pool()["awaiting_checks"][0]["candidate_id"] == "hand_20261128_x"
    lev = {(r["lever"], r["parent_class"]): r for r in data.levers()["levers"]}
    assert lev[("hump", "all")]["accepted"] == 1 and lev[("hump", "sharpe")]["median_d_sharpe"] == 0.7


def test_new_rows_are_read_incrementally(memory, candidate):
    data = DashboardData(memory.data_dir)
    _store(memory, candidate, "hand_20261128_a")
    assert data.alphas()["count"] == 1
    _store(memory, candidate, "hand_20261128_b")
    data.refreshed_at = 0           # the 2-second cache is over
    assert data.alphas()["count"] == 2 and data.last["results"] == 2


def test_a_running_live_file_whose_process_is_gone_shows_as_lost(tmp_path):
    (tmp_path / "live").mkdir()
    (tmp_path / "live" / "rs1.json").write_text(json.dumps(
        {"run_id": "rs1", "status": "running", "pid": 999999999, "started_at": "x"}))
    (tmp_path / "live" / "rs2.json").write_text(json.dumps(
        {"run_id": "rs2", "status": "running", "pid": os.getpid(), "started_at": "y"}))
    got = {x["run_id"]: x["status"] for x in DashboardData(tmp_path).live()}
    assert got == {"rs1": "lost", "rs2": "running"}


def test_origin_and_run_time():
    assert origin_of("rs20261127153052r01a1_001_r1a") == "rs20261127153052"
    assert origin_of("seed_774d71e6_r1") == "seeds" and origin_of("x") == "other"
    assert run_time("rs20261127153052") == "2026-11-27T15:30:52+00:00"


# ------------------------------------------------------------------ server handler (no sockets)

def _get(data, path, host="127.0.0.1:8050"):
    handler = dashboard.make_handler(data)
    h = handler.__new__(handler)
    h.rfile = io.BytesIO()
    h.wfile = io.BytesIO()
    h.path = path
    h.headers = {"Host": host}
    h.request_version = "HTTP/1.1"
    h.requestline = f"GET {path} HTTP/1.1"
    h.command = "GET"
    h.client_address = ("127.0.0.1", 1)
    h.close_connection = True
    h.do_GET()
    head, _, body = h.wfile.getvalue().partition(b"\r\n\r\n")
    return int(head.split()[1]), body


def test_the_server_answers_locally_and_refuses_other_hosts(memory, candidate):
    _store(memory, candidate, "hand_20261128_a")
    data = DashboardData(memory.data_dir)
    status, body = _get(data, "/api/alphas")
    assert status == 200 and json.loads(body)["count"] == 1
    status, body = _get(data, "/")
    assert status == 200 and b"QIP Alpha Research" in body
    assert _get(data, "/api/alpha?id=nope")[0] == 404
    assert _get(data, "/api/nothing")[0] == 404
    assert _get(data, "/api/alphas", host="evil.example:8050")[0] == 403
    assert _get(data, "/api/alphas", host="localhost:8050")[0] == 200
    assert dashboard.HOST == "127.0.0.1"


def test_memory_is_opened_read_only(memory, candidate):
    _store(memory, candidate, "hand_20261128_a")
    con = DashboardData(memory.data_dir)._connect()
    with pytest.raises(Exception, match="readonly"):
        con.execute("CREATE TABLE x (a)")
    con.close()
