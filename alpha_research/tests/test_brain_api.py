"""brain_api: translation, session behaviour and RealBrainClient, all over a fake transport.

The BRAIN bodies below are illustrative, in the publicly known response shape (OQ-2), with
the screenshot's numbers. They are not a capture from the account.
"""

import base64
import json
import urllib.error
import urllib.parse

import pytest

from alpha_research import brain_api, cli
from alpha_research.brain_api import (BrainApiError, BrainAuthError, BrainSession,
                                      RealBrainClient, Response, brain_config_from_dict,
                                      list_datafields, list_operators, load_brain_config,
                                      recordset_rows, simulation_payload, translate)
from alpha_research.config import ConfigError
from alpha_research.pipeline import Pipeline
from alpha_research.schemas import DECISIONS
from alpha_research.simulator import Simulator, VirtualClock, normalize

from conftest import FIXTURES, PLAYBOOK, ROOT

BRAIN_CONFIG = ROOT / "config" / "brain.yaml"
SECRET = "s3cret-value"

ALPHA = {"id": "A1", "is": {
    "pnl": 4290000, "bookSize": 20000000, "longCount": 1437, "shortCount": 1273,
    "turnover": 0.2578, "returns": 0.1119, "drawdown": 0.0704, "margin": 0.000868,
    "sharpe": 1.68, "fitness": 1.11, "startDate": "2019-01-01",
    "checks": [
        {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25, "value": 1.68},
        {"name": "HIGH_TURNOVER", "result": "PASS", "limit": 0.7, "value": 0.2578},
        {"name": "CONCENTRATED_WEIGHT", "result": "FAIL", "value": 0.12},
        {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS", "limit": 0.73, "value": 1.07},
        {"name": "SELF_CORRELATION", "result": "PENDING"},
    ]}}
YEARLY_ROWS = [  # year, sharpe, turnover, fitness, returns, drawdown, margin, long, short
    (2019, 1.40, 0.2305, 0.73, 0.0621, 0.0333, 0.000539, 1437, 1273),
    (2020, 0.11, 0.2699, 0.02, 0.0080, 0.0704, 0.000059, 1387, 1329),
    (2021, 1.83, 0.2772, 1.28, 0.1361, 0.0428, 0.000982, 1491, 1365),
    (2022, 3.59, 0.2564, 3.79, 0.2851, 0.0511, 0.002224, 1654, 1318),
    (2023, 1.01, 0.2559, 0.49, 0.0595, 0.0478, 0.000465, 1576, 1393),
]
YEARLY = {
    "schema": {"name": "yearly-stats", "properties": [
        {"name": n} for n in ("year", "sharpe", "turnover", "fitness", "returns", "drawdown",
                              "margin", "longCount", "shortCount", "stage")]},
    "records": [[str(r[0]), *r[1:], "IS"] for r in YEARLY_ROWS],
}


def pnl_body(n=300):
    import pandas as pd
    dates = pd.bdate_range("2019-01-01", periods=n)
    cum, records = 0.0, []
    for i, d in enumerate(dates):
        cum += 1000.0 if i % 3 else -400.0
        records.append([d.date().isoformat(), cum])
    return {"schema": {"properties": [{"name": "date"}, {"name": "pnl"}]}, "records": records}


DONE = {"id": "sim1", "status": "COMPLETE", "alpha": "A1"}


def ok(body=None, status=200, **headers):
    return Response(status, {k.replace("_", "-").lower(): str(v) for k, v in headers.items()},
                    b"" if body is None else json.dumps(body).encode())


class FakeTransport:
    """Routes (method, path) to a list of Responses (the last repeats) or exceptions."""

    def __init__(self, routes=None):
        self.routes = {("POST", "/authentication"): [ok(status=201)], **(routes or {})}
        self.calls = []
        self._cursor = {}

    def request(self, method, url, body, headers):
        parts = urllib.parse.urlsplit(url)
        self.calls.append({"method": method, "path": parts.path, "query": parts.query,
                           "headers": dict(headers), "body": json.loads(body) if body else None})
        key = (method, parts.path)
        if key not in self.routes:
            return ok({"detail": "Not found."}, status=404)
        queue = self.routes[key]
        i = self._cursor.get(key, 0)
        self._cursor[key] = i + 1
        r = queue[min(i, len(queue) - 1)]
        if isinstance(r, Exception):
            raise r
        return r

    def paths(self, method=None):
        return [c["path"] for c in self.calls if method is None or c["method"] == method]


def brain_routes(overrides=None):
    routes = {
        ("POST", "/simulations"): [ok(status=201, location=f"{brain_api.API_ROOT}/simulations/sim1")],
        ("GET", "/simulations/sim1"): [ok({"progress": 0.2}, retry_after=2.5),
                                       ok({"progress": 0.8}, retry_after=2.5),
                                       ok({"id": "sim1", "status": "COMPLETE", "alpha": "A1"})],
        ("GET", "/alphas/A1"): [ok(ALPHA)],
        ("GET", "/alphas/A1/recordsets/yearly-stats"): [ok(YEARLY)],
        ("GET", "/alphas/A1/recordsets/pnl"): [ok(None, retry_after=1), ok(pnl_body())],
        ("GET", "/alphas/A1/correlations/self"): [ok({"max": 0.41, "min": -0.2, "records": []})],
    }
    routes.update(overrides or {})
    return routes


@pytest.fixture
def bcfg():
    return load_brain_config(BRAIN_CONFIG)


def make_client(routes, bcfg, vc=None, capture_dir=None):
    vc = vc or VirtualClock()
    transport = FakeTransport(routes)
    session = BrainSession(transport, "me@example.com", SECRET, sleep=vc.sleep,
                           rate_limit_max_wait=bcfg.rate_limit_max_wait_seconds)
    return RealBrainClient(session, bcfg, capture_dir=capture_dir), transport, vc


# ---------------------------------------------------------------- config


def test_main_brain_config_loads(bcfg):
    assert bcfg.settings["language"] == "FASTEXPR"
    assert bcfg.settings["pasteurization"] == "ON" and bcfg.settings["nanHandling"] == "OFF"
    assert bcfg.fetch_self_correlation is True


@pytest.mark.parametrize("raw, match", [
    ({"settings": {}}, "non-empty"),
    ({"settings": {"language": "FASTEXPR", "region": "USA"}}, "come from each candidate"),
    ({"settings": {"language": "FASTEXPR"}, "request_timeout_seconds": 0}, "positive"),
    ({"settings": {"language": "FASTEXPR"}, "fetch_self_correlation": "yes"}, "true or false"),
    ({"settings": {"language": "FASTEXPR"}, "base_url": "x"}, "unknown keys"),
])
def test_brain_config_rejects_bad_values(raw, match):
    with pytest.raises(ConfigError, match=match):
        brain_config_from_dict(raw)


def test_simulation_payload_uses_candidate_settings(bcfg, candidate):
    p = simulation_payload(candidate(decay=6, truncation=0.05), bcfg)
    assert p["type"] == "REGULAR" and p["regular"] == "rank(-ts_delta(close, 5))"
    s = p["settings"]
    assert (s["region"], s["universe"], s["delay"], s["decay"], s["neutralization"],
            s["truncation"]) == ("USA", "TOP3000", 1, 6, "SUBINDUSTRY", 0.05)
    assert s["instrumentType"] == "EQUITY" and "start_date" not in s


# ---------------------------------------------------------------- translation


def test_translate_maps_known_fields_and_keeps_fractions():
    raw = translate({"status": "COMPLETE", "alpha": "A1"}, ALPHA, YEARLY, pnl_body(5),
                    {"max": 0.41})
    assert raw["status"] == "complete" and raw["units"] == "fraction"
    assert (raw["sharpe"], raw["fitness"], raw["turnover"], raw["returns"], raw["drawdown"]) == \
        (1.68, 1.11, 0.2578, 0.1119, 0.0704)
    assert raw["margin"] == pytest.approx(8.68)
    assert (raw["sub_universe_sharpe"], raw["sub_universe_cutoff"]) == (1.07, 0.73)
    assert raw["weight_concentration"] == 0.12 and raw["self_correlation"] == 0.41
    assert raw["long_pnl"] is None and raw["short_pnl"] is None and raw["drawdown_date"] is None
    assert raw["checks"] == {"LOW_SHARPE": True, "HIGH_TURNOVER": True,
                             "CONCENTRATED_WEIGHT": False, "LOW_SUB_UNIVERSE_SHARPE": True}
    assert any("SELF_CORRELATION='PENDING'" in n for n in raw["notes"])
    assert brain_api.UNVERIFIED_NOTE in raw["notes"] and brain_api.DATES_NOTE in raw["notes"]


def test_translate_then_normalize_gives_screenshot_values():
    raw = translate(DONE, ALPHA, YEARLY, pnl_body(5), None)
    r = normalize(raw, "c1", "sim1")
    assert r.turnover == 0.2578 and r.returns == 0.1119 and r.max_drawdown == 0.0704
    assert r.margin_bps == pytest.approx(8.68)
    assert r.self_correlation is None  # PENDING check has no value; no correlation fetched
    y2022 = next(y for y in r.yearly if y.year == 2022)
    assert (y2022.sharpe, y2022.turnover, y2022.fitness, y2022.returns, y2022.max_drawdown) == \
        (3.59, 0.2564, 3.79, 0.2851, 0.0511)
    assert y2022.margin_bps == pytest.approx(22.24)
    assert (y2022.long_count, y2022.short_count) == (1654, 1318)
    assert brain_api.UNVERIFIED_NOTE in r.notes


def test_cumulative_pnl_is_differenced():
    body = {"schema": {"properties": [{"name": "date"}, {"name": "pnl"}]},
            "records": [["2020-01-02", 100.0], ["2020-01-03", 150.0], ["2020-01-06", None],
                        ["2020-01-07", 120.0]]}
    raw = translate(DONE, ALPHA, pnl=body)
    assert raw["daily_pnl"] == {"dates": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
                                "values": [100.0, 50.0, None, -30.0]}
    assert "brain_pnl_period: 2020-01-02..2020-01-07 from pnl (differenced)" in raw["notes"]
    raw = translate(DONE, ALPHA, pnl=None, daily_pnl=body)
    assert raw["daily_pnl"]["values"] == [100.0, 150.0, None, 120.0]
    assert "brain_pnl_period: 2020-01-02..2020-01-07 from daily-pnl" in raw["notes"]


def test_self_correlation_falls_back_to_check_value():
    alpha = json.loads(json.dumps(ALPHA))
    alpha["is"]["checks"][-1] = {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.33,
                                 "limit": 0.7}
    assert translate(DONE, alpha)["self_correlation"] == 0.33


def test_missing_and_non_numeric_fields_become_none_with_note():
    alpha = {"is": {"sharpe": "1.2", "checks": []}}
    raw = translate(DONE, alpha)
    assert raw["sharpe"] is None and raw["fitness"] is None and raw["yearly"] == []
    assert raw["daily_pnl"] is None
    assert any("brain_field_not_numeric: is.sharpe" in n for n in raw["notes"])


@pytest.mark.parametrize("sim, text", [
    ({"status": "ERROR", "message": "Attempted to use unknown variable \"clse\""}, "clse"),
    ({"status": "FAIL", "message": "boom"}, "FAIL: boom"),
    ({"status": "TIMEOUT"}, "TIMEOUT"),
    ({"details": "Simulation was cancelled"}, "cancelled"),
    ({"detail": "Not found."}, "Not found."),
    ({"children": ["a", "b"], "status": "COMPLETE"}, "multi-simulation"),
    ({"status": "COMPLETE"}, "without an alpha id"),
])
def test_failed_simulation_translates_to_failed(sim, text):
    raw = translate(sim, {"is": {}})
    assert raw["status"] == "failed"
    assert any(text in n for n in raw["notes"]), raw["notes"]
    assert normalize(raw, "c1", "sim1").status == "failed"


def test_recordset_rows_ignores_malformed_records():
    body = {"schema": {"properties": [{"name": "a"}, {"name": "b"}]},
            "records": [[1, 2], [3], "x"]}
    assert recordset_rows(body) == [{"a": 1, "b": 2}]
    assert recordset_rows(None) == []


# ---------------------------------------------------------------- session


def test_authenticate_uses_basic_auth_once(bcfg):
    client, transport, _ = make_client(brain_routes(), bcfg)
    client.session.request("GET", "/alphas/A1")
    client.session.request("GET", "/alphas/A1")
    auth_calls = [c for c in transport.calls if c["path"] == "/authentication"]
    assert len(auth_calls) == 1
    expected = base64.b64encode(f"me@example.com:{SECRET}".encode()).decode()
    assert auth_calls[0]["headers"]["Authorization"] == f"Basic {expected}"
    assert all("Authorization" not in c["headers"] for c in transport.calls[1:])


def test_expired_session_reauthenticates_once(bcfg):
    routes = brain_routes({("GET", "/alphas/A1"): [ok(status=401), ok(ALPHA)]})
    client, transport, _ = make_client(routes, bcfg)
    assert client.session.request("GET", "/alphas/A1").json()["id"] == "A1"
    assert transport.paths().count("/authentication") == 2


def test_wrong_credentials_raise_without_echoing_secret(bcfg):
    routes = {("POST", "/authentication"): [ok({"detail": "Invalid credentials."}, status=401)]}
    client, _, _ = make_client(routes, bcfg)
    with pytest.raises(BrainAuthError, match="401") as exc:
        client.session.authenticate()
    assert SECRET not in str(exc.value)


def test_biometric_login_is_reported(bcfg):
    routes = {("POST", "/authentication"): [
        ok(status=401, www_authenticate="persona", location="/authentication/persona?inquiry=x")]}
    client, _, _ = make_client(routes, bcfg)
    with pytest.raises(BrainAuthError, match="biometric.*persona"):
        client.session.authenticate()


def test_rate_limit_waits_then_succeeds(bcfg):
    routes = brain_routes({("GET", "/alphas/A1"): [ok(status=429, retry_after=30), ok(ALPHA)]})
    client, _, vc = make_client(routes, bcfg)
    assert client.session.request("GET", "/alphas/A1").status == 200
    assert vc.sleeps == [30.0]


def test_rate_limit_gives_up_as_connection_error(bcfg):
    routes = brain_routes({("GET", "/alphas/A1"): [ok(status=429, retry_after=400)]})
    client, _, vc = make_client(routes, bcfg)
    with pytest.raises(ConnectionError, match="rate limited"):
        client.session.request("GET", "/alphas/A1")
    assert sum(vc.sleeps) <= bcfg.rate_limit_max_wait_seconds


def test_server_error_is_transient(bcfg):
    routes = brain_routes({("GET", "/alphas/A1"): [ok(status=503)]})
    client, _, _ = make_client(routes, bcfg)
    with pytest.raises(ConnectionError, match="503"):
        client.session.request("GET", "/alphas/A1")


def test_credentials_come_from_env(monkeypatch):
    monkeypatch.delenv(brain_api.EMAIL_ENV_VAR, raising=False)
    monkeypatch.setenv(brain_api.SECRET_ENV_VAR, SECRET)
    with pytest.raises(BrainAuthError, match=brain_api.EMAIL_ENV_VAR):
        brain_api.credentials_from_env()
    monkeypatch.setenv(brain_api.EMAIL_ENV_VAR, "me@example.com")
    assert brain_api.credentials_from_env() == ("me@example.com", SECRET)


def test_urllib_transport_maps_errors(monkeypatch):
    t = brain_api.UrllibTransport(timeout=1)

    def raise_(exc):
        def _open(req, timeout):
            raise exc
        return _open

    monkeypatch.setattr(t._opener, "open", raise_(urllib.error.URLError("no route")))
    with pytest.raises(ConnectionError, match="no route"):
        t.request("GET", "https://example.invalid/x", None, {})
    monkeypatch.setattr(t._opener, "open", raise_(urllib.error.URLError(TimeoutError("slow"))))
    with pytest.raises(TimeoutError):
        t.request("GET", "https://example.invalid/x", None, {})
    import io
    err = urllib.error.HTTPError("https://example.invalid/x", 404, "nf", {"Retry-After": "3"},
                                 io.BytesIO(b'{"detail": "Not found."}'))
    monkeypatch.setattr(t._opener, "open", raise_(err))
    r = t.request("GET", "https://example.invalid/x", None, {})
    assert r.status == 404 and r.retry_after() == 3.0 and r.detail() == "Not found."


# ---------------------------------------------------------------- RealBrainClient + Simulator


def test_simulator_runs_end_to_end_on_real_client(bcfg, config, memory, candidate):
    client, transport, vc = make_client(brain_routes(), bcfg)
    sim = Simulator(client, memory, config, sleep=vc.sleep, clock=vc.clock)
    r = sim.run(candidate())
    assert r.status == "complete" and r.brain_sim_id == "sim1"
    assert (r.sharpe, r.fitness, r.turnover) == (1.68, 1.11, 0.2578)
    assert r.self_correlation == 0.41 and len(r.yearly) == 5
    assert r.daily_pnl is not None and len(r.daily_pnl) == 300
    assert "brain_alpha_id: A1" in r.notes
    assert vc.sleeps == [2.5, 2.5, 1.0]          # two sim polls, one recordset wait
    body = transport.calls[1]["body"]
    assert transport.calls[1]["path"] == "/simulations" and body["regular"] == candidate().expression
    assert memory.pending() == []


def test_rejected_submission_becomes_failed_result(bcfg, config, memory, candidate):
    routes = brain_routes({("POST", "/simulations"): [
        ok({"settings": {"universe": ["Invalid choice"]}}, status=400)]})
    client, transport, vc = make_client(routes, bcfg)
    r = Simulator(client, memory, config, sleep=vc.sleep, clock=vc.clock).run(candidate())
    assert r.status == "failed" and r.brain_sim_id is None
    assert "submit_rejected" in r.notes[0] and "Invalid choice" in r.notes[0]
    assert "/simulations/sim1" not in transport.paths()


def test_error_simulation_is_failed(bcfg, config, memory, candidate):
    routes = brain_routes({("GET", "/simulations/sim1"): [
        ok({"id": "sim1", "status": "ERROR", "message": "unknown variable"})]})
    client, transport, vc = make_client(routes, bcfg)
    r = Simulator(client, memory, config, sleep=vc.sleep, clock=vc.clock).run(candidate())
    assert r.status == "failed" and any("unknown variable" in n for n in r.notes)
    assert not any(p.startswith("/alphas") for p in transport.paths())


def test_missing_recordset_leaves_field_none(bcfg, config, memory, candidate):
    routes = brain_routes()
    del routes[("GET", "/alphas/A1/recordsets/pnl")]
    client, _, vc = make_client(routes, bcfg)
    r = Simulator(client, memory, config, sleep=vc.sleep, clock=vc.clock).run(candidate())
    assert r.status == "complete" and r.daily_pnl is None
    assert "brain_fetch_failed: pnl returned 404" in r.notes


def test_recordset_wait_is_bounded(bcfg, config, memory, candidate):
    routes = brain_routes({("GET", "/alphas/A1/recordsets/pnl"): [ok(None, retry_after=200)]})
    client, _, vc = make_client(routes, bcfg)
    r = Simulator(client, memory, config, sleep=vc.sleep, clock=vc.clock).run(candidate())
    assert r.status == "complete" and r.daily_pnl is None
    assert any(n.startswith("brain_fetch_timeout: pnl") for n in r.notes)


def test_self_correlation_fetch_can_be_disabled(config, memory, candidate):
    raw = json.loads(json.dumps(yaml_raw()))
    raw["fetch_self_correlation"] = False
    client, transport, vc = make_client(brain_routes(), brain_config_from_dict(raw))
    Simulator(client, memory, config, sleep=vc.sleep, clock=vc.clock).run(candidate())
    assert "/alphas/A1/correlations/self" not in transport.paths()


def yaml_raw():
    import yaml
    return yaml.safe_load(BRAIN_CONFIG.read_text())


def test_capture_dir_saves_raw_bodies_without_secrets(bcfg, config, memory, candidate, tmp_path):
    client, _, vc = make_client(brain_routes(), bcfg, capture_dir=tmp_path / "cap")
    Simulator(client, memory, config, sleep=vc.sleep, clock=vc.clock).run(candidate())
    saved = sorted(p.name for p in (tmp_path / "cap" / "sim1").iterdir())
    assert saved == ["alpha.json", "pnl.json", "self_correlation.json", "simulation.json",
                     "translated.json", "yearly_stats.json"]
    assert json.loads((tmp_path / "cap" / "sim1" / "alpha.json").read_text()) == ALPHA
    for p in (tmp_path / "cap").rglob("*.json"):
        assert SECRET not in p.read_text() and "me@example.com" not in p.read_text()


def test_pipeline_decides_on_real_client_results(bcfg, config, playbook, memory, candidate):
    client, _, vc = make_client(brain_routes(), bcfg)
    p = Pipeline(config, playbook, client, memory, sleep=vc.sleep, clock=vc.clock)
    out = p.process(candidate())
    assert out.result.status == "complete" and out.decision in DECISIONS
    # weight concentration 0.12 is above the test config's 0.10 limit: a risk veto
    assert (out.decision, out.failure_class) == ("reject", "concentration")


# ---------------------------------------------------------------- catalogs


def test_list_datafields_paginates(bcfg):
    page1 = {"count": 52, "results": [{"id": f"f{i}"} for i in range(50)]}
    page2 = {"count": 52, "results": [{"id": "f50"}, {"id": "f51"}]}
    client, transport, _ = make_client({("GET", "/data-fields"): [ok(page1), ok(page2)]}, bcfg)
    fields = list_datafields(client.session, region="USA", universe="TOP3000", delay=1,
                             dataset="fundamental6")
    assert [f["id"] for f in fields][-2:] == ["f50", "f51"] and len(fields) == 52
    queries = [urllib.parse.parse_qs(c["query"]) for c in transport.calls if c["path"] == "/data-fields"]
    assert [q["offset"] for q in queries] == [["0"], ["50"]]
    assert queries[0]["dataset.id"] == ["fundamental6"] and queries[0]["region"] == ["USA"]


def test_list_operators_and_errors(bcfg):
    ops = [{"name": "rank", "definition": "rank(x, rate=2)"}, {"category": "no name"}]
    client, _, _ = make_client({("GET", "/operators"): [ok(ops)]}, bcfg)
    assert list_operators(client.session) == [ops[0]]
    client, _, _ = make_client({("GET", "/operators"): [ok({"detail": "nope"}, status=403)]}, bcfg)
    with pytest.raises(BrainApiError, match="403"):
        list_operators(client.session)


# ---------------------------------------------------------------- CLI


@pytest.fixture
def fake_cli_session(monkeypatch, bcfg):
    """Point the CLI's make_session at a FakeTransport; time.sleep never really waits."""
    holder = {}

    def _install(routes):
        transport = FakeTransport(routes)
        holder["transport"] = transport
        monkeypatch.setattr(cli, "make_session", lambda cfg: BrainSession(
            transport, "me@example.com", SECRET, sleep=lambda s: None))
        monkeypatch.setattr(cli.time, "sleep", lambda s: None)
        return transport

    return _install


def test_cli_brain_auth_without_credentials_exits_7(monkeypatch, capsys):
    monkeypatch.delenv(brain_api.EMAIL_ENV_VAR, raising=False)
    monkeypatch.delenv(brain_api.SECRET_ENV_VAR, raising=False)
    assert cli.main(["brain", "auth", "--brain-config", str(BRAIN_CONFIG)]) == cli.EXIT_BRAIN
    assert brain_api.EMAIL_ENV_VAR in capsys.readouterr().err


def test_cli_brain_auth_ok(fake_cli_session, capsys):
    fake_cli_session({})
    assert cli.main(["brain", "auth", "--brain-config", str(BRAIN_CONFIG)]) == cli.EXIT_OK
    assert "login ok" in capsys.readouterr().out


def test_cli_brain_capture_writes_result(fake_cli_session, tmp_path, candidate, capsys):
    fake_cli_session(brain_routes())
    cands = tmp_path / "c.json"
    cands.write_text(json.dumps([candidate().to_dict()]))
    rc = cli.main(["brain", "capture", "--candidates", str(cands), "--out-dir",
                   str(tmp_path / "cap"), "--brain-config", str(BRAIN_CONFIG)])
    assert rc == cli.EXIT_OK
    result = json.loads((tmp_path / "cap" / "sim1" / "result.json").read_text())
    assert result["status"] == "complete" and result["sharpe"] == 1.68
    assert (tmp_path / "cap" / "sim1" / "simulation.json").exists()
    assert "c1 | sim1 | complete" in capsys.readouterr().out


def test_cli_brain_operators_diffs_seed_list(fake_cli_session, tmp_path, capsys):
    ops = [{"name": "rank", "definition": "rank(x, rate=2)"},
           {"name": "brand_new_op", "definition": "brand_new_op(x)"}]
    fake_cli_session({("GET", "/operators"): [ok(ops)]})
    out = tmp_path / "ops.json"
    rc = cli.main(["brain", "operators", "--out", str(out), "--brain-config", str(BRAIN_CONFIG),
                   "--fastexpr", str(ROOT / "config" / "fastexpr.yaml")])
    assert rc == cli.EXIT_OK and json.loads(out.read_text()) == ops
    text = capsys.readouterr().out
    assert "on BRAIN, not in seed list: brand_new_op" in text
    assert "in seed list, not on BRAIN: ts_mean" in text
    assert "check arity: rank  BRAIN: rank(x, rate=2)" in text


def test_cli_brain_datafields_feeds_generator(fake_cli_session, tmp_path):
    from alpha_research.generator import load_datafields
    fake_cli_session({("GET", "/data-fields"): [ok({"count": 2, "results": [
        {"id": "close", "dataset": {"id": "pv1"}}, {"id": "volume"}]})]})
    out = tmp_path / "fields.json"
    rc = cli.main(["brain", "datafields", "--region", "USA", "--universe", "TOP3000",
                   "--delay", "1", "--out", str(out), "--brain-config", str(BRAIN_CONFIG)])
    assert rc == cli.EXIT_OK and load_datafields(out) == ["close", "volume"]


def test_cli_brain_api_error_exits_7(fake_cli_session, tmp_path, capsys):
    fake_cli_session({("GET", "/operators"): [ok({"detail": "Forbidden"}, status=403)]})
    rc = cli.main(["brain", "operators", "--out", str(tmp_path / "o.json"),
                   "--brain-config", str(BRAIN_CONFIG)])
    assert rc == cli.EXIT_BRAIN and "Forbidden" in capsys.readouterr().err


# ---------------------------------------------------------------- BRAIN's own samples

SAMPLES = FIXTURES / "brain_api_samples"


def sample(name):
    return json.loads((SAMPLES / f"{name}.json").read_text())


def test_web_app_samples_translate_cleanly():
    alpha = sample("alpha")
    raw = translate(sample("simulation"), alpha, sample("yearly_stats"), sample("pnl"))
    r = normalize(raw, "c1", "sample-simulation-id-walkthrough")
    assert r.status == "complete"
    assert (r.sharpe, r.fitness, r.turnover, r.returns, r.max_drawdown) == \
        (2.83, 1.33, 0.6427, 0.1411, 0.0219)
    assert r.margin_bps == pytest.approx(43.9)
    assert (r.sub_universe_sharpe, r.sub_universe_cutoff) == (2.07, 1.23)
    assert r.weight_concentration == 0.2 and r.self_correlation is None
    assert r.checks["LOW_SHARPE"] is True and r.checks["MATCHES_COMPETITION"] is True
    assert "SELF_CORRELATION" not in r.checks
    for name in ("SELF_CORRELATION", "PROD_CORRELATION", "REGULAR_SUBMISSION", "IS_LADDER_SHARPE"):
        assert f"brain_check_not_counted: {name}='PENDING'" in r.notes
    assert len(r.yearly) == 11
    y2011 = r.yearly[0]
    assert (y2011.year, y2011.sharpe, y2011.turnover, y2011.long_count) == (2011, 4.16, 0.6485, 1504)
    assert y2011.margin_bps == pytest.approx(6.45)
    assert "brain_test_period: P1Y0M0D" in r.notes and "brain_is_start: 2011-04-07" in r.notes
    assert any(n.startswith("brain_test_stats: start 2020-02-12 sharpe 0.78") for n in r.notes)


def test_sample_pnl_is_cumulative_and_differences_back_to_is_pnl():
    raw = translate(sample("simulation"), sample("alpha"), pnl=sample("pnl"))
    values = raw["daily_pnl"]["values"]
    assert raw["daily_pnl"]["dates"][0] == "2011-04-07" and len(values) == 2458
    # the last cumulative value is within 0.02% of is.pnl in BRAIN's sample
    assert sum(v for v in values if v is not None) == pytest.approx(sample("alpha")["is"]["pnl"], rel=2e-4)


def test_sample_recordset_list_names_daily_pnl():
    names = [r["name"] for r in sample("recordsets")["results"]]
    assert {"pnl", "daily-pnl", "yearly-stats"} <= set(names)


def test_non_is_yearly_rows_are_excluded():
    body = sample("yearly_stats")
    body["records"].append(["2022", 1.0, 2e7, 1, 1, 0.5, 1.0, 0.1, 0.01, 0.0002, 0.5, "TEST"])
    raw = translate(sample("simulation"), sample("alpha"), yearly=body)
    assert len(raw["yearly"]) == 11
    assert "brain_yearly_rows_excluded: stages ['TEST']" in raw["notes"]


def test_requests_send_versioned_accept_header(bcfg):
    client, transport, _ = make_client(brain_routes(), bcfg)
    client.session.request("GET", "/alphas/A1")
    assert transport.calls[-1]["headers"]["Accept"] == "application/json;version=2.0"


def test_progress_without_retry_after_still_means_running(bcfg):
    routes = brain_routes({("GET", "/simulations/sim1"): [ok({"progress": 0.4})]})
    client, _, _ = make_client(routes, bcfg)
    assert client.poll("sim1") == {"status": "running", "retry_after": None, "progress": 0.4}


def test_missing_simulation_is_failed(bcfg):
    routes = brain_routes({("GET", "/simulations/sim1"): [ok({"detail": "Not found."}, status=404)]})
    client, _, _ = make_client(routes, bcfg)
    assert client.poll("sim1")["status"] == "failed"


def test_daily_pnl_recordset_is_preferred(bcfg, config, memory, candidate):
    daily = {"schema": {"properties": [{"name": "date"}, {"name": "pnl"}]},
             "records": [["2019-01-02", 5.0], ["2019-01-03", -2.0]]}
    routes = brain_routes({("GET", "/alphas/A1/recordsets/daily-pnl"): [ok(daily)]})
    client, transport, vc = make_client(routes, bcfg)
    r = Simulator(client, memory, config, sleep=vc.sleep, clock=vc.clock).run(candidate())
    assert list(r.daily_pnl.values) == [5.0, -2.0]
    assert "/alphas/A1/recordsets/pnl" not in transport.paths()


def test_settings_errors_are_flattened_into_the_rejection(bcfg, config, memory, candidate):
    routes = brain_routes({("POST", "/simulations"): [
        ok({"settings": {"universe": ["\"TOP9\" is not a valid choice."]}}, status=400)]})
    client, _, vc = make_client(routes, bcfg)
    r = Simulator(client, memory, config, sleep=vc.sleep, clock=vc.clock).run(candidate())
    assert r.status == "failed" and 'universe: "TOP9" is not a valid choice.' in r.notes[0]


OPTIONS_BODY = {"actions": {"POST": {
    "type": {"choices": [{"value": "REGULAR", "label": "Regular",
                          "settings": {"children": {"region": {"choices": [{"value": "USA"}]}}}},
                         {"value": "SUPER", "label": "Super"}]},
    "settings": {"children": {
        "neutralization": {"type": "choice", "choices": [{"value": "NONE"}, {"value": "MARKET"}]},
        "universe": {"type": "choice", "choices": {"instrumentType": {"EQUITY": {}}}},
        "truncation": {"type": "float", "min_value": 0, "max_value": 1}}}}}}


def test_simulation_options_reduction(bcfg):
    client, transport, _ = make_client({("OPTIONS", "/simulations"): [ok(OPTIONS_BODY)]}, bcfg)
    opts = brain_api.simulation_options(client.session)
    assert set(opts["common"]) == {"neutralization", "universe", "truncation"}
    assert list(opts["by_type"]) == ["REGULAR"]
    assert transport.calls[-1]["method"] == "OPTIONS"


def test_cli_brain_options(fake_cli_session, tmp_path, capsys):
    fake_cli_session({("OPTIONS", "/simulations"): [ok(OPTIONS_BODY)]})
    out = tmp_path / "opts.json"
    rc = cli.main(["brain", "options", "--out", str(out), "--brain-config", str(BRAIN_CONFIG)])
    assert rc == cli.EXIT_OK and json.loads(out.read_text())["by_type"]["REGULAR"]
    text = capsys.readouterr().out
    assert "neutralization: ['NONE', 'MARKET']" in text
    assert "universe: depends on instrumentType" in text
    assert "types: ['REGULAR']" in text
