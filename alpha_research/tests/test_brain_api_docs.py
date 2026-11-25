"""Gaps found against BRAIN's API docs and its simulation-limit docs (2026-11-25): the daily
simulation limit (X-Ratelimit-* headers on POST /simulations), the statuses WAITING,
SIMULATING and CANCELLED, the code location of an ERROR, and BRAIN's 4-hour login."""

import dataclasses

import pytest

from alpha_research import brain_api
from alpha_research.brain_api import brain_config_from_dict, simulation_quota
from alpha_research.brain_client import SimulationQuotaExhausted
from alpha_research.config import ConfigError

from test_brain_api import FakeTransport, brain_routes, make_client, ok


@pytest.fixture
def bcfg():
    return dataclasses.replace(brain_api.load_brain_config(
        brain_api.REPO_ROOT / "alpha_research" / "config" / "brain.yaml"), quota_reserve=50)


def test_quota_comes_from_the_unsuffixed_headers_only():
    assert simulation_quota({"x-ratelimit-limit": "5000", "x-ratelimit-remaining": "4321",
                             "x-ratelimit-reset": "3600"}) == \
        {"limit": 5000, "remaining": 4321, "reset_seconds": 3600}
    # per-hour / per-minute request limits of other endpoints are not the simulation limit
    assert simulation_quota({"x-ratelimit-remaining-hour": "1995",
                             "x-ratelimit-limit-minute": "60"}) == {}


def test_submit_records_the_quota_and_the_run_stops_near_the_reserve(bcfg, candidate):
    post = ok(status=201, location=f"{brain_api.API_ROOT}/simulations/sim1",
              x_ratelimit_limit=5000, x_ratelimit_remaining=120, x_ratelimit_reset=7200)
    client, _, _ = make_client(brain_routes({("POST", "/simulations"): [post]}), bcfg)
    assert client.quota_stop() is None                       # nothing known yet
    client.submit(candidate())
    assert client.quota == {"limit": 5000, "remaining": 120, "reset_seconds": 7200}
    assert client.quota_stop() is None                       # 120 > reserve 50
    client.quota["remaining"] = 50
    assert "nearly used" in client.quota_stop() and "50 of 5000 left" in client.quota_stop()
    client.config = dataclasses.replace(bcfg, quota_reserve=None)
    assert client.quota_stop() is None


def test_a_429_with_no_simulations_left_stops_without_waiting(bcfg, candidate):
    post = ok({"detail": "limit"}, status=429, retry_after=30, x_ratelimit_limit=5000,
              x_ratelimit_remaining=0, x_ratelimit_reset=36000)
    client, _, vc = make_client(brain_routes({("POST", "/simulations"): [post]}), bcfg)
    with pytest.raises(SimulationQuotaExhausted, match="used up.*0 of 5000.*10.0 h"):
        client.submit(candidate())
    assert vc.sleeps == []
    # a 429 with simulations left (too many at once) is still waited out
    busy = ok(status=429, retry_after=30, x_ratelimit_remaining=900)
    done = ok(status=201, location=f"{brain_api.API_ROOT}/simulations/sim1")
    client, _, vc = make_client(brain_routes({("POST", "/simulations"): [busy, done]}), bcfg)
    assert client.submit(candidate()) == "sim1" and vc.sleeps == [30.0]


@pytest.mark.parametrize("status", ["WAITING", "SIMULATING"])
def test_waiting_and_simulating_are_running_even_without_retry_after(bcfg, status):
    routes = brain_routes({("GET", "/simulations/sim1"): [ok({"id": "sim1", "status": status})]})
    client, _, _ = make_client(routes, bcfg)
    assert client.poll("sim1")["status"] == "running"


def test_cancelled_is_a_failure_and_an_error_names_its_code_location(bcfg):
    routes = brain_routes({("GET", "/simulations/sim1"): [ok({"id": "sim1", "status": "CANCELLED"})]})
    client, _, _ = make_client(routes, bcfg)
    raw = client.poll("sim1")
    assert raw["status"] == "failed" and any("CANCELLED" in n for n in raw["notes"])
    err = {"id": "sim1", "status": "ERROR", "message": "Invalid data field",
           "location": {"property": "regular", "line": 2, "start": 5, "end": 17}}
    client, _, _ = make_client(brain_routes({("GET", "/simulations/sim1"): [ok(err)]}), bcfg)
    assert any("Invalid data field (at line 2, columns 5-17)" in n
               for n in client.poll("sim1")["notes"])


def test_a_login_ending_soon_is_renewed_at_the_start(bcfg):
    auth = [ok({"token": {"expiry": 1800}}), ok({"token": {"expiry": 14400}})]
    client, transport, _ = make_client(brain_routes({("GET", "/authentication"): auth}), bcfg)
    client.session.authenticated = True                      # a saved session
    assert client.session.refresh_if_expiring(10800) == 14400
    assert transport.paths("POST") == ["/authentication"]   # logged in again
    # enough time left: no new login
    client, transport, _ = make_client(
        brain_routes({("GET", "/authentication"): [ok({"token": {"expiry": 12000}})]}), bcfg)
    client.session.authenticated = True
    assert client.session.refresh_if_expiring(10800) == 12000
    assert transport.paths("POST") == []
    # BRAIN answers 204 when the saved session is no longer logged in: log in
    client, transport, _ = make_client(
        brain_routes({("GET", "/authentication"): [ok(status=204), ok({"token": {"expiry": 14400}})]}),
        bcfg)
    client.session.authenticated = True
    assert client.session.refresh_if_expiring(10800) == 14400
    assert transport.paths("POST") == ["/authentication"]


def test_config_keys():
    base = {"settings": {"language": "FASTEXPR"}}
    c = brain_config_from_dict({**base, "quota_reserve": 0, "relogin_within_seconds": None})
    assert c.quota_reserve == 0 and c.relogin_within_seconds is None
    assert brain_config_from_dict({**base, "relogin_within_seconds": 3600}).relogin_within_seconds == 3600
    with pytest.raises(ConfigError, match="quota_reserve"):
        brain_config_from_dict({**base, "quota_reserve": -1})
    main = brain_api.load_brain_config(brain_api.REPO_ROOT / "alpha_research" / "config" / "brain.yaml")
    assert main.quota_reserve == 50 and main.relogin_within_seconds == 10800


def test_a_research_run_stops_before_a_round_when_the_quota_is_low(memory, config, playbook):
    from test_research import AGENTS, GEN_CONFIG, SPEC, RouterLLM, StubBrain, plans
    from alpha_research.research import Budget, ResearchRun
    from alpha_research.simulator import VirtualClock
    brain = StubBrain()
    brain.quota_stop = lambda: "BRAIN's daily simulation limit is nearly used (reserve 50)"
    vc = VirtualClock()
    run = ResearchRun(plans(), AGENTS, Budget(max_simulations=8, per_round=4, max_llm_calls=50),
                      llm_factory=lambda m, r=RouterLLM(): r, client=brain, config=config,
                      playbook=playbook, spec=SPEC, generator_config=GEN_CONFIG, memory=memory,
                      seed=0, sleep=vc.sleep, clock=vc.clock, run_id="rsq")
    report = run.run(3)
    assert "nearly used" in report["stop_reason"] and report["simulations_used"] == 0


def test_vec_operators_take_event_fields_only():
    # BRAIN's docs, troubleshooting: "matrix data field in a vector operator" is an invalid
    # data field; a constant there "must be an event data". Replay over memory (2026-11-25):
    # one hit, rs20261115045859r01a17_001_c1, which BRAIN failed.
    from alpha_research.fastexpr import FastExprSpec, analyze, catalog
    from conftest import ROOT
    spec = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
    cat = catalog(["evt", "mat"], {"evt": {"type": "VECTOR"}, "mat": {"type": "MATRIX"}})
    assert analyze("rank(vec_avg(evt))", spec, datafields=cat).ok
    for expr in ("rank(vec_avg(mat))", "rank(vec_avg(close))"):
        assert any("matrix field" in e for e in analyze(expr, spec, datafields=cat).errors), expr
    assert any("not a constant" in e
               for e in analyze("rank(close) + vec_avg(5)", spec, datafields=cat).errors)
    # untyped catalog: types unknown, not checked
    assert analyze("rank(vec_avg(mat))", spec, datafields=["evt", "mat"]).ok


DIVERSITY = {"count": 3, "alphas": [
    {"region": "CHN", "delay": 1, "alphaCount": 1, "dataCategory": {"id": "analyst"},
     "dataDiversity": {"check": "PASS", "limit": 0.3}},
    {"region": "CHN", "delay": 1, "alphaCount": 2, "dataCategory": {"id": "other"},
     "dataDiversity": {"check": "PASS", "limit": 0.3}},
    {"region": "CHN", "delay": 1, "dataCategory": {"name": None, "id": None}, "alphaCount": 2},
    {"region": "CHN", "delay": None, "dataCategory": {"name": None, "id": None}, "alphaCount": 2},
    {"region": "USA", "delay": None, "dataCategory": {"name": None, "id": None}, "alphaCount": 1}]}


def test_submission_diversity_reads_regions_and_categories(bcfg):
    routes = brain_routes({("GET", "/users/self/activities/diversity"): [ok(DIVERSITY)]})
    client, transport, _ = make_client(routes, bcfg)
    d = client.submission_diversity()
    assert d["total"] == 3 and d["regions"] == {"CHN": 2, "USA": 1}
    assert d["categories"][1] == {"region": "CHN", "delay": 1, "category": "other",
                                  "alphas": 2, "check": "PASS", "limit": 0.3}
    assert transport.calls[-1]["query"] == "grouping=region%2Cdelay%2CdataCategory"
    # BRAIN not answering is no error: it is context only
    client, _, _ = make_client(brain_routes({("GET", "/users/self/activities/diversity"):
                                             [ok({"detail": "x"}, status=403)]}), bcfg)
    assert client.submission_diversity() is None


def test_strategy_sees_our_submissions_by_region_and_category():
    from alpha_research.strategy import summarize
    from alpha_research.prompts import strategy_system
    d = {"total": 3, "regions": {"CHN": 2}, "categories": [
        {"region": "CHN", "delay": 1, "category": "analyst", "alphas": 1}]}
    s = summarize([], known_datasets=["a"], submissions=d)
    assert s["our_brain_submissions"] == {"total": 3, "by_region": {"CHN": 2},
                                          "categories_by_region": {"CHN": {"analyst": 1}}}
    assert "our_brain_submissions" not in summarize([], known_datasets=["a"])
    assert "our_brain_submissions" in strategy_system("")


def test_factor_first_sharpe_repairs_try_slow_and_fast_first(memory, candidate):
    from alpha_research.evaluator import FACTOR_FIRST
    from alpha_research.repair import Playbook
    from alpha_research.repair_agent import RepairAgent
    from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord
    from conftest import PLAYBOOK
    from test_rescue_lessons import SPEC
    agent = RepairAgent(Playbook.load(PLAYBOOK), SPEC, memory, llm=None)
    d = DecisionRecord("c1", "repair", "sharpe",
                       gate_results=[GateResult("brain:LOW_SHARPE", False, 1.0, 1.58, "")],
                       reasons=[f"{FACTOR_FIRST}: risk-neutralized Sharpe 1.9 vs raw 1.0"])
    r = ResultRecord("c1", "complete", sharpe=1.0,
                     alt_stats={"risk_neutralized": {"sharpe": 1.9}})
    order = [lv.name for lv, _ in agent.ranked(candidate(region="EUR", universe="TOP2500",
                                                          neutralization="COUNTRY"), d, r,
                                               agent.evidence()[1])]
    assert order[0] == "neutralize_slow_and_fast" and "flip_sign" not in order


def test_generator_lessons_carry_brains_search_space_advice():
    from alpha_research.generator import lessons_for, load_generator_config
    from conftest import ROOT
    text = " ".join(lessons_for(load_generator_config(ROOT / "config" / "generator.yaml"), "pv"))
    assert "5, 10 or 21 days" in text and "63, 121 or 252" in text
