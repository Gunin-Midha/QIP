"""Region rotation: each research run works on a few regions, the least simulated ones,
so that across runs every region gets the same number of simulations."""

import dataclasses
from collections import deque

import pytest
import yaml

from alpha_research.config import ConfigError
from alpha_research.generator import GenerationSettings, Source
from alpha_research.orchestrator import balance_slots
from alpha_research.research import (ArmPlan, Budget, ResearchRun, agents_config_from_dict,
                                     arm_label)
from alpha_research.schemas import ResultRecord
from alpha_research.simulator import VirtualClock

from conftest import ROOT
from test_research import AGENTS, GEN_CONFIG, MAIN_AGENTS, SPEC, RouterLLM, StubBrain

MARKETS = {"USA": ("TOP3000", "SUBINDUSTRY"), "EUR": ("TOP2500", "INDUSTRY"),
           "JPN": ("TOP1600", "INDUSTRY"), "CHN": ("TOP2000U", "INDUSTRY")}


def region_plans(regions=tuple(MARKETS)):
    out = []
    for r in regions:
        universe, neut = MARKETS[r]
        s = GenerationSettings.from_mapping({
            "region": r, "universe": universe, "delay": 1, "decay": 4, "neutralization": neut,
            "truncation": 0.08, "start_date": "2014-01-01", "end_date": "2023-12-31",
            "dataset": "pv1"})
        out.append(ArmPlan("pv1", s, deque([Source("hypothesis", "short-term reversal")])))
    return out


def rotating_run(memory, config, playbook, *, k=2, since=None, sims=6, run_id="rs1",
                 brain=None):
    agents = dataclasses.replace(AGENTS, regions_per_run=k, region_balance_since=since)
    vc = VirtualClock()
    return ResearchRun(region_plans(), agents,
                       Budget(max_simulations=sims, per_round=sims, max_llm_calls=100),
                       llm_factory=lambda m, r=RouterLLM(): r, client=brain or StubBrain(), config=config,
                       playbook=playbook, spec=SPEC, generator_config=GEN_CONFIG, memory=memory,
                       seed=0, sleep=vc.sleep, clock=vc.clock, run_id=run_id)


def test_balance_slots_fills_the_least_simulated_region_first():
    assert balance_slots(6, {"USA": 0, "EUR": 0, "JPN": 0}) == {"USA": 2, "EUR": 2, "JPN": 2}
    assert balance_slots(5, {"USA": 4, "EUR": 0, "JPN": 1}) == {"USA": 0, "EUR": 3, "JPN": 2}
    assert balance_slots(0, {"USA": 0}) == {"USA": 0}
    with pytest.raises(ValueError):
        balance_slots(-1, {"USA": 0})


def test_memory_counts_simulations_per_region(memory, candidate):
    memory.put_candidate(candidate(candidate_id="u1"))
    memory.put_result(ResultRecord("u1", "complete"))
    memory.put_candidate(candidate(candidate_id="u2"))
    memory.put_result(ResultRecord("u2", "invalid"))          # never reached BRAIN
    memory.put_candidate(candidate(candidate_id="e1", region="EUR", universe="TOP2500"))
    memory.put_result(ResultRecord("e1", "failed"))
    memory.put_result(ResultRecord("e1", "complete"))        # one candidate, counted once
    counts = memory.simulations_by_region()
    assert {r: v["simulations"] for r, v in counts.items()} == {"USA": 1, "EUR": 1}
    assert counts["EUR"]["last"] >= counts["USA"]["last"]
    assert memory.simulations_by_region("2999-01-01") == {}


def test_the_main_config_rotates_three_regions_a_run():
    assert MAIN_AGENTS.regions_per_run == 3 and MAIN_AGENTS.region_balance_since == "2026-11-11"


@pytest.mark.parametrize("patch, match", [
    ({"regions_per_run": 0}, "regions_per_run"),
    ({"regions_per_run": "3"}, "regions_per_run"),
    ({"region_balance_since": "last week"}, "ISO date"),
])
def test_rotation_settings_are_validated(patch, match):
    raw = yaml.safe_load((ROOT / "config" / "agents.yaml").read_text())
    raw["director"].update(patch)
    with pytest.raises(ConfigError, match=match):
        agents_config_from_dict(raw)


def test_null_rotation_and_dates_load():
    raw = yaml.safe_load((ROOT / "config" / "agents.yaml").read_text())
    import datetime
    raw["director"].update(regions_per_run=None, region_balance_since=datetime.date(2026, 1, 2))
    a = agents_config_from_dict(raw)
    assert a.regions_per_run is None and a.region_balance_since == "2026-01-02"


def test_a_run_works_only_on_its_regions_split_evenly(memory, config, playbook):
    brain = StubBrain()
    run = rotating_run(memory, config, playbook, brain=brain)
    assert run.run_regions == ["USA", "EUR"]   # all at 0: plan order
    report = run.run(1)
    per_region = {}
    for label, n in run.log[0]["allocation"].items():
        per_region[label.split("/")[1]] = per_region.get(label.split("/")[1], 0) + n
    assert per_region == {"USA": 3, "EUR": 3, "JPN": 0, "CHN": 0}
    assert {c.region for c in brain.submitted} <= {"USA", "EUR"}
    assert report["regions_this_run"] == ["USA", "EUR"]
    assert set(report["simulations_by_region"]) == {"USA", "EUR", "JPN", "CHN"}


def test_the_next_run_takes_the_regions_the_last_one_skipped(memory, config, playbook):
    first = rotating_run(memory, config, playbook)
    first.run(1)
    assert first.sims_used > 0
    second = rotating_run(memory, config, playbook, run_id="rs2")
    assert second.run_regions == ["JPN", "CHN"]
    second.run(1)
    third = rotating_run(memory, config, playbook, run_id="rs3")
    counts = third.region_sims
    # every region has been worked on, and the third run goes back to the least simulated
    assert all(counts[r] > 0 for r in MARKETS)
    left_out = [r for r in MARKETS if r not in third.run_regions]
    assert max(counts[r] for r in third.run_regions) <= min(counts[r] for r in left_out)


def test_simulations_before_the_balance_date_do_not_count(memory, config, playbook, candidate):
    for i in range(3):
        memory.put_candidate(candidate(candidate_id=f"old{i}"))
        memory.put_result(ResultRecord(f"old{i}", "complete"))
    assert rotating_run(memory, config, playbook).run_regions == ["EUR", "JPN"]
    assert rotating_run(memory, config, playbook, since="2999-01-01").run_regions == ["USA", "EUR"]


def test_queued_repairs_of_other_regions_wait_for_their_run(memory, config, playbook, candidate):
    run = rotating_run(memory, config, playbook)
    chn = next(k for k in run.arms if k[1] == "CHN")
    usa = next(k for k in run.arms if k[1] == "USA")
    run.repair_queue[chn].append(candidate(candidate_id="chn_r1", region="CHN",
                                           universe="TOP2000U", neutralization="INDUSTRY"))
    run.repair_queue[usa].append(candidate(candidate_id="usa_r1", expression="rank(-ts_delta(close, 21))"))
    taken = run._take_repairs(5)
    assert list(taken) == [usa] and len(run.repair_queue[chn]) == 1
    assert arm_label(chn).startswith("pv1/CHN/D1/")


def test_without_rotation_every_region_is_in_the_run(memory, config, playbook):
    run = rotating_run(memory, config, playbook, k=None)
    assert run.run_regions is None
    assert rotating_run(memory, config, playbook, k=4).run_regions is None
