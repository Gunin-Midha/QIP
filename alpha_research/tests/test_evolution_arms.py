"""Strategy Evolution without arm explosion (2026-11-14: one evolution added 140 arms to a
20-simulation run). Only the run's regions evolve, each evolved arm replaces its parent and
starts from its record, and arms for new datasets are capped."""

import dataclasses

from alpha_research.research import Budget, ResearchRun
from alpha_research.simulator import VirtualClock
from alpha_research.strategy import Evolution

from test_research import AGENTS, GEN_CONFIG, SPEC, TWO_MODELS, RouterLLM, StubBrain
from test_rotation import region_plans

META = {"close": {"dataset": "pv1"}, "flow": {"dataset": "newds"}}
MARKETS = {(r, 1): (["close", "flow"], META) for r in ("USA", "EUR", "JPN")}   # CHN lacks newds


def evolving_run(memory, config, playbook, max_new_arms=3):
    agents = dataclasses.replace(
        AGENTS, models=TWO_MODELS, regions_per_run=2, region_balance_since=None,
        strategy=dataclasses.replace(AGENTS.strategy, max_new_arms=max_new_arms))
    vc = VirtualClock()
    run = ResearchRun(region_plans(), agents, Budget(max_simulations=8, per_round=4,
                                                     max_llm_calls=50),
                      llm_factory=lambda m, r=RouterLLM(): r, client=StubBrain(), config=config,
                      playbook=playbook, spec=SPEC, generator_config=GEN_CONFIG, memory=memory,
                      markets=MARKETS, seed=0, sleep=vc.sleep, clock=vc.clock, run_id="rs1")
    run.log.append({"round": 1, "arms": []})
    return run


def evolve(run, version):
    run.strategist.evolve = lambda *a, **k: Evolution(
        version, "guidance", [("newds", "flow idea"), (None, "general idea")])
    run._evolve(1)


def test_only_the_runs_regions_evolve_and_each_evolved_arm_replaces_its_parent(
        memory, config, playbook):
    run = evolving_run(memory, config, playbook)
    assert run.run_regions == ["USA", "EUR"]
    parent = ("pv1", "USA", "gen-v7", "claude-a", 1)
    run.director.update(parent, False)
    run.director.update(parent, True)
    before = len(run.arms)
    evolve(run, "gen-v7+evo1")

    evolved = [k for k in run.arms if k[2] == "gen-v7+evo1" and k[0] == "pv1"]
    assert {k[1] for k in evolved} == {"USA", "EUR"} and len(evolved) == 4   # 2 regions x 2 models
    child = ("pv1", "USA", "gen-v7+evo1", "claude-a", 1)
    assert run.successor[parent] == child
    live = run._live_arms()
    assert parent not in live and child in live
    assert sum(1 for k in live if k[0] == "pv1") == 4                        # one arm per idea
    # the evolved arm starts from what its parent showed, not from scratch
    s = run.director.state[child]
    assert (s.successes, s.failures) == (1, 1)
    # new-dataset arms: capped, spread over markets of this run, then models
    new = sorted(k for k in run.arms if k[0] == "newds")
    assert new == [("newds", "EUR", "gen-v7+evo1", "gemini-b", 1),
                   ("newds", "USA", "gen-v7+evo1", "claude-a", 1),
                   ("newds", "USA", "gen-v7+evo1", "gemini-b", 1)]
    assert len(run.arms) - before == 4 + 3
    assert run.log[-1]["evolution"]["replaced"]


def test_a_second_evolution_evolves_the_successors_and_credit_follows_them(
        memory, config, playbook):
    run = evolving_run(memory, config, playbook, max_new_arms=0)
    parent = ("pv1", "EUR", "gen-v7", "gemini-b", 1)
    evolve(run, "gen-v7+evo1")
    evolve(run, "gen-v7+evo2")
    latest = ("pv1", "EUR", "gen-v7+evo2", "gemini-b", 1)
    assert run._current(parent) == latest
    assert not [k for k in run.arms if k[0] == "newds"]                     # cap 0
    assert sum(1 for k in run._live_arms() if k[0] == "pv1") == 4
    # an outcome of a repair child still on the original arm credits the latest arm
    n = run.director.state[latest].failures
    run.director.update(run._current(parent), False)
    assert run.director.state[latest].failures == n + 1


def test_max_new_arms_is_read_from_config():
    from alpha_research.research import load_agents_config
    from conftest import ROOT
    assert load_agents_config(ROOT / "config" / "agents.yaml").strategy.max_new_arms == 8
