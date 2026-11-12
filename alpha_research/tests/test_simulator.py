import json
from datetime import date

import pytest

from alpha_research.brain_client import FixtureBrainClient
from alpha_research.config import ConfigError
from alpha_research.schemas import SchemaError
from alpha_research.memory import MemoryStore
from alpha_research.simulator import Simulator, VirtualClock, normalize, parentheses_balanced
from conftest import BRAIN_DIR


def make_sim(client, memory, config):
    vc = VirtualClock()
    return Simulator(client, memory, config, sleep=vc.sleep, clock=vc.clock), vc


def write_fixture(tmp_path, name, data):
    d = tmp_path / "brain_results"
    d.mkdir(exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(data))
    return d


def test_percent_to_fraction_on_ingest():
    raw = json.loads((BRAIN_DIR / "screenshot_aggregate.json").read_text())
    r = normalize(raw, "shot", "screenshot_aggregate")
    assert r.turnover == pytest.approx(0.2578)
    assert r.returns == pytest.approx(0.1119)
    assert r.max_drawdown == pytest.approx(0.0704)
    assert r.margin_bps == pytest.approx(8.68)       # bps, not converted
    y2022 = next(y for y in r.yearly if y.year == 2022)
    assert y2022.turnover == pytest.approx(0.2564)
    assert y2022.returns == pytest.approx(0.2851)
    assert y2022.long_count == 1654


def test_missing_keys_become_none_not_zero():
    r = normalize({"status": "complete", "sharpe": 1.5}, "c", "s")
    assert r.turnover is None and r.self_correlation is None and r.daily_pnl is None
    assert r.yearly == [] and r.checks == {}


def test_unknown_fields_ignored():
    r = normalize({"status": "complete", "sharpe": 1.5, "mystery_metric": 9}, "c", "s")
    assert not hasattr(r, "mystery_metric")


def test_parentheses():
    assert parentheses_balanced("rank(ts_delta(close, 5))")
    assert not parentheses_balanced("rank(ts_delta(close, 5)")
    assert not parentheses_balanced(")rank(")


@pytest.mark.parametrize("kw", [
    dict(expression="rank(close"), dict(region="XYZ"), dict(neutralization="BAD"),
    dict(universe="TOP7"), dict(delay=3),
    dict(start_date="2023-01-01", end_date="2024-06-30"),     # overlaps test holdout
])
def test_local_validation_returns_invalid_without_client_call(tmp_path, memory, config, candidate, kw):
    client = FixtureBrainClient(BRAIN_DIR, {"c1": "screenshot_aggregate"})
    sim, _ = make_sim(client, memory, config)
    r = sim.run(candidate(**kw))
    assert r.status == "invalid"
    assert client.submit_calls == []
    assert memory.results_for("c1")[0].status == "invalid"


def test_holdout_not_configured_raises(memory, make_config, candidate):
    cfg = make_config(holdout__start_date=None, holdout__end_date=None)
    sim, _ = make_sim(FixtureBrainClient(BRAIN_DIR, {}), memory, cfg)
    with pytest.raises(ConfigError):
        sim.run(candidate())


def test_happy_path_persists_result_and_notes(memory, config, candidate):
    client = FixtureBrainClient(BRAIN_DIR, {"c1": "screenshot_aggregate"})
    sim, _ = make_sim(client, memory, config)
    r = sim.run(candidate())
    assert r.status == "complete" and r.brain_sim_id == "screenshot_aggregate"
    assert any(n.startswith("fitness_ok") for n in r.notes)
    assert any(n.startswith("margin_mismatch") for n in r.notes)
    stored = memory.results_for("c1")
    assert len(stored) == 1 and stored[0].sharpe == 1.68


def test_polls_until_complete_and_honours_retry_after(tmp_path, memory, config, candidate):
    d = write_fixture(tmp_path, "seq", {"responses": [
        {"status": "running", "retry_after": 7}, {"status": "running"},
        {"status": "complete", "sharpe": 1.4}]})
    sim, vc = make_sim(FixtureBrainClient(d, {"c1": "seq"}), memory, config)
    r = sim.run(candidate())
    assert r.status == "complete"
    assert vc.sleeps == [7.0, 5.0]


@pytest.mark.parametrize("period, status", [
    ("2019-01-01..2023-12-31", "complete"),   # BRAIN's default on our account; before the holdout
    ("2020-01-01..2024-06-30", "invalid"),    # BRAIN moved its period into the test holdout
])
def test_brain_period_overlapping_holdout_makes_result_invalid(tmp_path, memory, config,
                                                               candidate, period, status):
    d = write_fixture(tmp_path, "p", {"status": "complete", "sharpe": 1.4,
                                      "notes": [f"brain_period: {period}"]})
    sim, _ = make_sim(FixtureBrainClient(d, {"c1": "p"}), memory, config)
    r = sim.run(candidate())
    assert r.status == status
    assert any("overlaps holdout" in n for n in r.notes) == (status == "invalid")


def test_timeout_after_max_poll_seconds(tmp_path, memory, config, candidate):
    d = write_fixture(tmp_path, "slow", {"responses": [{"status": "running"}]})
    sim, vc = make_sim(FixtureBrainClient(d, {"c1": "slow"}), memory, config)
    r = sim.run(candidate())
    assert r.status == "timeout"
    assert vc.now >= config.simulator.max_poll_seconds


def test_transient_poll_errors_are_retried(tmp_path, memory, config, candidate):
    d = write_fixture(tmp_path, "flaky", {"responses": [
        {"raise": "ConnectionError"}, {"raise": "TimeoutError"}, {"status": "complete", "sharpe": 1.3}]})
    sim, vc = make_sim(FixtureBrainClient(d, {"c1": "flaky"}), memory, config)
    r = sim.run(candidate())
    assert r.status == "complete"
    assert vc.sleeps == [1.0, 2.0]   # exponential backoff


def test_retries_exhausted(tmp_path, memory, config, candidate):
    d = write_fixture(tmp_path, "dead", {"responses": [{"raise": "TimeoutError"}]})
    client = FixtureBrainClient(d, {"c1": "dead"})
    sim, _ = make_sim(client, memory, config)
    r = sim.run(candidate())
    assert r.status == "timeout"
    assert len(client.poll_calls) == config.simulator.max_retries + 1


def test_submit_retry(memory, config, candidate):
    client = FixtureBrainClient(BRAIN_DIR, {"c1": "screenshot_aggregate"},
                                submit_errors={"c1": ["ConnectionError", "ConnectionError"]})
    sim, _ = make_sim(client, memory, config)
    assert sim.run(candidate()).status == "complete"
    assert len(client.submit_calls) == 3


def test_crash_after_submit_resumes_without_resubmitting(tmp_path, config, candidate):
    data = tmp_path / "data"
    store = MemoryStore(data)
    store.put_candidate(candidate())
    store.put_pending("c1", "screenshot_aggregate")    # process died before polling finished
    store.close()

    store = MemoryStore(data)
    assert [p["candidate_id"] for p in store.pending()] == ["c1"]
    client = FixtureBrainClient(BRAIN_DIR, {})        # submit would fail: no mapping
    sim, _ = make_sim(client, store, config)
    r = sim.run(candidate())
    assert r.status == "complete" and r.brain_sim_id == "screenshot_aggregate"
    assert client.submit_calls == []
    assert store.pending() == []
    store.close()


def test_retryable_failure_is_submitted_once_more(tmp_path, memory, config, candidate):
    d = write_fixture(tmp_path, "b", {"responses": [
        {"status": "failed", "retryable": True, "notes": ["brain_simulation_failed: ERROR"]},
        {"status": "complete", "sharpe": 1.4}]})
    client = FixtureBrainClient(d, {"c1": "b"})
    sim, _ = make_sim(client, memory, config)
    r = sim.run(candidate())
    assert r.status == "complete" and len(client.submit_calls) == 2
    assert "retried once" in r.notes[0]
    assert [x.status for x in memory.results_for("c1")] == ["complete"]


def test_plain_failure_is_not_resubmitted(tmp_path, memory, config, candidate):
    d = write_fixture(tmp_path, "f", {"status": "failed", "notes": ["bad field"]})
    client = FixtureBrainClient(d, {"c1": "f"})
    sim, _ = make_sim(client, memory, config)
    assert sim.run(candidate()).status == "failed"
    assert len(client.submit_calls) == 1


def test_recover_fetches_a_result_brain_finished_after_our_timeout(tmp_path, memory, config,
                                                                   candidate):
    # BRAIN keeps running after we stop waiting: 'slow' is running past max_poll_seconds,
    # then complete when the next run asks
    d = write_fixture(tmp_path, "slow", {"responses": [{"status": "running"}] * 300
                                         + [{"status": "complete", "sharpe": 1.4}]})
    client = FixtureBrainClient(d, {"c1": "slow"})
    sim, _ = make_sim(client, memory, config)
    assert sim.run(candidate()).status == "timeout"
    assert memory.unfinished_submissions() == [{"candidate_id": "c1", "brain_sim_id": "slow"}]
    client._cursor["slow"] = 300
    r = sim.recover(candidate(), "slow")
    assert r.status == "complete" and r.sharpe == 1.4 and r.notes[0].startswith("recovered")
    assert memory.results_for("c1")[-1].status == "complete"
    assert memory.unfinished_submissions() == []


def test_recover_leaves_running_and_unknown_simulations(tmp_path, memory, config, candidate):
    d = write_fixture(tmp_path, "run", {"status": "running"})
    d = write_fixture(tmp_path, "gone", {"status": "failed",
                                         "notes": ["brain_simulation_failed: simulation not found"]})
    sim, _ = make_sim(FixtureBrainClient(d, {}), memory, config)
    memory.put_candidate(candidate())
    assert sim.recover(candidate(), "run") is None
    assert sim.recover(candidate(), "gone") is None
    assert memory.results_for("c1") == []


def test_unfinished_submissions_lists_timeouts_and_crashes(memory, candidate):
    from alpha_research.schemas import ResultRecord
    for cid in ("t", "crash", "done", "t_no_id"):
        memory.put_candidate(candidate(candidate_id=cid))
    memory.put_result(ResultRecord("t", "timeout", brain_sim_id="s_t"))
    memory.put_pending("crash", "s_crash")
    memory.put_result(ResultRecord("done", "timeout", brain_sim_id="s_done"))
    memory.put_result(ResultRecord("done", "complete", brain_sim_id="s_done", sharpe=1.0))
    memory.put_result(ResultRecord("t_no_id", "timeout"))
    assert memory.unfinished_submissions() == [
        {"candidate_id": "crash", "brain_sim_id": "s_crash"},
        {"candidate_id": "t", "brain_sim_id": "s_t"}]


def test_fraction_units_skip_percent_conversion_and_notes_are_kept():
    raw = {"status": "complete", "units": "fraction", "turnover": 0.2578, "returns": 0.1119,
           "drawdown": 0.0704, "notes": ["from client"],
           "yearly": [{"year": 2019, "turnover": 0.2305}]}
    r = normalize(raw, "c1", "s1")
    assert (r.turnover, r.returns, r.max_drawdown) == (0.2578, 0.1119, 0.0704)
    assert r.yearly[0].turnover == 0.2305 and r.notes == ["from client"]
    assert normalize({"status": "complete", "turnover": 25.78}, "c1", "s1").turnover == pytest.approx(0.2578)


def test_unknown_units_is_a_schema_error():
    with pytest.raises(SchemaError, match="units"):
        normalize({"status": "complete", "units": "bps"}, "c1", "s1")
