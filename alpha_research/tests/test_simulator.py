import json
from datetime import date

import pytest

from alpha_research.brain_client import FixtureBrainClient, RealBrainClient
from alpha_research.config import ConfigError
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


def test_real_client_is_a_stub(candidate):
    with pytest.raises(NotImplementedError, match="OQ-2"):
        RealBrainClient().submit(candidate())
    with pytest.raises(NotImplementedError):
        RealBrainClient().poll("x")
