"""Research plan v2: region blocks with inherited settings, a catalog per region and delay,
and arms keyed by delay as well as region."""

import json
from collections import deque

import pytest
import yaml

from alpha_research import cli
from alpha_research.config import ConfigError
from alpha_research.critic import Critic
from alpha_research.generator import GenerationSettings, Source
from alpha_research.prefilter import PreFilter, PrefilterConfig
from alpha_research.research import ArmPlan, Budget, ResearchRun, arm_label, load_plan
from alpha_research.simulator import VirtualClock

from conftest import ROOT
from test_research import AGENTS, GEN_CONFIG, SPEC, RouterLLM, StubBrain, make_run

V2 = ROOT / "config" / "research_plan.v2.yaml"
DEFAULTS = {"truncation": 0.08, "start_date": "2014-01-01", "end_date": "2023-12-31"}


def write_v2(tmp_path, regions, defaults=DEFAULTS):
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({"defaults": defaults, "regions": regions}))
    return p


def usa(**block):
    base = {"settings": {"universe": "TOP3000", "neutralization": "SUBINDUSTRY", "decay": 4},
            "arms": [{"dataset": "pv1", "delay": 1, "hypotheses": ["reversal"]}]}
    base.update(block)
    return base


def test_the_v2_plan_loads_every_region_and_delay():
    plans = load_plan(V2)
    markets = {(p.settings.region, p.settings.delay) for p in plans}
    assert markets == {("USA", 0), ("USA", 1), ("ALL", 1), ("GLB", 1), ("EUR", 0), ("EUR", 1),
                       ("ASI", 1), ("CHN", 0), ("CHN", 1), ("JPN", 0), ("JPN", 1), ("AMR", 0),
                       ("AMR", 1), ("IND", 1), ("DEU", 0), ("DEU", 1), ("GBR", 0), ("GBR", 1)}
    assert len(plans) == 37
    # the same dataset in one region at both delays is two arms
    assert {p.settings.delay for p in plans if p.dataset == "other561"} == {0, 1}
    opt = next(p for p in plans if p.dataset == "option8")
    assert (opt.settings.universe, opt.settings.neutralization, opt.settings.decay,
            opt.settings.truncation) == ("TOP3000", "CROWDING", 4, 0.08)
    assert opt.catalog.name == "datafields_usa_top3000_d1.json"
    assert all(p.catalog is not None and f"_d{p.settings.delay}." in p.catalog.name for p in plans)


def test_settings_are_inherited_and_arm_settings_win(tmp_path):
    regions = {"USA": usa(arms=[{"dataset": "pv1", "delay": 0, "settings": {"decay": 2},
                                 "hypotheses": ["h"]},
                                {"dataset": "pv1", "delay": 1, "hypotheses": ["h"]}],
                          catalogs={0: "d0.json", 1: "/abs/d1.json"})}
    a, b = load_plan(write_v2(tmp_path, regions))
    assert (a.settings.delay, a.settings.decay, b.settings.delay, b.settings.decay) == (0, 2, 1, 4)
    assert a.settings.start_date.isoformat() == "2014-01-01" and a.settings.region == "USA"
    assert a.catalog == tmp_path / "d0.json" and str(b.catalog) == "/abs/d1.json"


def test_a_disabled_region_is_skipped(tmp_path):
    regions = {"USA": usa(), "CHN": {"enabled": False, "settings": {}, "arms": "not checked"}}
    assert [p.settings.region for p in load_plan(write_v2(tmp_path, regions))] == ["USA"]


@pytest.mark.parametrize("regions, match", [
    ({}, "non-empty mapping"),
    ({"MARS": usa()}, "region 'MARS'"),
    ({"USA": usa(enabled="yes")}, "'enabled' must be true or false"),
    ({"USA": usa(enabled=False)}, "no arms in an enabled region"),
    ({"USA": usa(arms=[])}, "non-empty 'arms'"),
    ({"USA": usa(arms=[{"dataset": "pv1", "hypotheses": ["h"]}])}, "delay None"),
    ({"GLB": usa(arms=[{"dataset": "pv1", "delay": 0, "hypotheses": ["h"]}])}, "delay 0"),
    ({"CHN": usa()}, "universe 'TOP3000' is not allowed in CHN"),
    ({"AMR": usa(settings={"universe": "TOP600", "neutralization": "CROWDING", "decay": 4})},
     "neutralization 'CROWDING' is not allowed in AMR"),
    ({"USA": usa(arms=[{"dataset": "pv1", "delay": 1, "hypotheses": ["h"]}] * 2)},
     "pv1.*USA delay 1 twice"),
    ({"USA": usa(arms=[{"dataset": "pv1", "delay": 1}])}, "at least one hypothesis"),
    ({"USA": usa(settings={"universe": "TOP3000"})}, "missing generation settings"),
])
def test_v2_plan_errors(tmp_path, regions, match):
    with pytest.raises(ConfigError, match=match):
        load_plan(write_v2(tmp_path, regions))


# ---------------------------------------------------------------- one catalog per market

MARKETS = {("USA", 1): {"implied_volatility_call_60": "MATRIX"},
           ("CHN", 0): {"oth561_basic_stat_net_buy_user_num": "MATRIX"}}


def test_prefilter_checks_each_candidate_against_its_own_market(memory, candidate):
    pf = PreFilter(PrefilterConfig(), SPEC, memory, markets=MARKETS)
    usa_ok = candidate(candidate_id="u1", expression="rank(implied_volatility_call_60)")
    chn_bad = candidate(candidate_id="c1", expression="rank(implied_volatility_call_60)",
                        region="CHN", universe="TOP2000U", delay=0, neutralization="INDUSTRY")
    chn_ok = candidate(candidate_id="c2", expression="rank(oth561_basic_stat_net_buy_user_num)",
                       region="CHN", universe="TOP2000U", delay=0, neutralization="INDUSTRY")
    res = pf.run([usa_ok, chn_bad, chn_ok])
    assert [c.candidate_id for c in res.kept] == ["u1", "c2"]
    assert "implied_volatility_call_60" in res.dropped[0][1]


def test_critic_revision_uses_the_candidates_market(candidate):
    critic = Critic(SPEC, markets=MARKETS)
    chn = candidate(candidate_id="c1", expression="rank(oth561_basic_stat_net_buy_user_num)",
                    region="CHN", universe="TOP2000U", delay=0, neutralization="INDUSTRY")
    assert critic.revision(chn, "rank(implied_volatility_call_60)") is None
    assert critic.revision(chn, "-rank(oth561_basic_stat_net_buy_user_num)") is not None


def two_market_plans():
    def arm(region, universe, delay, neut):
        s = GenerationSettings.from_mapping({"region": region, "universe": universe,
                                             "delay": delay, "decay": 4, "neutralization": neut,
                                             **DEFAULTS, "dataset": "pv1"})
        return ArmPlan("pv1", s, deque([Source("hypothesis", "short-term reversal")]))
    return [arm("USA", "TOP3000", 1, "SUBINDUSTRY"), arm("USA", "TOP3000", 0, "SUBINDUSTRY")]


def test_each_delay_is_its_own_arm_with_its_own_catalog(memory, config, playbook):
    markets = {("USA", 1): (["close", "vwap"], {}), ("USA", 0): (["close", "volume"], {})}
    vc = VirtualClock()
    run = ResearchRun(two_market_plans(), AGENTS,
                      Budget(max_simulations=4, per_round=4, max_llm_calls=50),
                      llm_factory=lambda m, r=RouterLLM(): r, client=StubBrain(), config=config,
                      playbook=playbook, spec=SPEC, generator_config=GEN_CONFIG, memory=memory,
                      markets=markets, seed=0, sleep=vc.sleep, clock=vc.clock, run_id="rs1")
    d1, d0 = ("pv1", "USA", "gen-v3", "claude-test", 1), ("pv1", "USA", "gen-v3", "claude-test", 0)
    assert set(run.arms) == {d1, d0}
    assert run._generator(d1).datafields == ["close", "vwap"]
    assert run._generator(d0).datafields == ["close", "volume"]
    assert arm_label(d0) == "pv1/USA/D0/gen-v3/claude-test"
    report = run.run(1)
    assert set(report["arms"]) == {"pv1/USA/D1/gen-v3/claude-test", "pv1/USA/D0/gen-v3/claude-test"}


class NewDatasetStrategy(RouterLLM):
    def complete(self, system, user, *, max_output=4096):
        if "research strategist" in system:
            self.calls.append("strategy")
            return json.dumps({"guidance": "g", "hypotheses": [
                {"hypothesis": "retail flow idea", "dataset": "other561"}]})
        return super().complete(system, user, max_output=max_output)


def test_a_new_dataset_gets_arms_only_where_its_catalog_has_it(memory, config, playbook):
    markets = {("USA", 1): (["close"], {"close": {"dataset": "pv1"}}),
               ("USA", 0): (["close", "flow"], {"flow": {"dataset": "other561"}})}
    vc = VirtualClock()
    run = ResearchRun(two_market_plans(), AGENTS,
                      Budget(max_simulations=8, per_round=4, max_llm_calls=100),
                      llm_factory=lambda m, r=NewDatasetStrategy(): r, client=StubBrain(),
                      config=config, playbook=playbook, spec=SPEC, generator_config=GEN_CONFIG,
                      memory=memory, markets=markets, seed=0, sleep=vc.sleep, clock=vc.clock,
                      run_id="rs1")
    run.run(2)
    new = [k for k in run.arms if k[0] == "other561"]
    assert new == [("other561", "USA", "gen-v3+evo1", "claude-test", 0)]
    assert run.arms[new[0]].settings.delay == 0


def test_single_catalog_runs_keep_working(memory, config, playbook):
    run = make_run(memory, config, playbook)
    assert list(run.arms) == [("pv1", "USA", "gen-v3", "claude-test", 1)]


# ---------------------------------------------------------------- CLI


def test_cli_reads_each_plan_catalog_once(tmp_path, capsys):
    (tmp_path / "usa1.json").write_text(json.dumps([
        {"id": "implied_volatility_call_60", "type": "MATRIX", "dataset": {"id": "option8"}}]))
    regions = {"USA": usa(catalogs={1: "usa1.json"},
                          arms=[{"dataset": "option8", "delay": 1, "hypotheses": ["a"]},
                                {"dataset": "pv1", "delay": 1, "hypotheses": ["b"]}])}
    markets = cli._plan_catalogs(load_plan(write_v2(tmp_path, regions)))
    fields, meta = markets[("USA", 1)]
    assert fields == ["implied_volatility_call_60"] and meta["implied_volatility_call_60"]["dataset"] == "option8"
    assert capsys.readouterr().out.count("catalog USA D1: 1 fields") == 1


def test_cli_missing_plan_catalog_is_a_config_error(tmp_path):
    plans = load_plan(write_v2(tmp_path, {"USA": usa(catalogs={1: "missing.json"})}))
    with pytest.raises(ConfigError, match="brain_discovery"):
        cli._plan_catalogs(plans)
