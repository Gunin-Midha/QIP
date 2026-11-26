"""Runs go on while nobody is at the terminal (owner 2026-11-26), so they must end by
themselves: a loop guard (no alpha to or from BRAIN for a while, or model calls with no
alpha sent), a stop before a round when the BRAIN login is about to end, and a clean stop
(report, no prompt) when BRAIN ends the login mid-run."""

import dataclasses
import io

from alpha_research.brain_api import BrainApiConfig, RealBrainClient
from alpha_research.brain_client import BrainLoginRequired
from alpha_research.research import Budget

from test_research import AGENTS, RouterLLM, StubBrain, make_run


class NeverValid(RouterLLM):
    """The generator only ever writes text that is not JSON; other agents answer."""

    OTHERS = ("Critic in a quant", "judge in a debate", "research strategist",
              "You write the lessons", "repair WorldQuant")

    def complete(self, system, user, *, max_output=4096):
        if not any(k in system for k in self.OTHERS):
            self.calls.append(("generator", user))
            return "no alphas today"
        return super().complete(system, user, max_output=max_output)


def test_loop_guard_stops_on_model_calls_with_nothing_sent(memory, config, playbook):
    # many refill passes = a loop that sends nothing to BRAIN
    agents = dataclasses.replace(AGENTS, refill_passes=50, stall_model_calls=6,
                                 stall_minutes=None, stop_after_idle_rounds=None)
    llm, brain = NeverValid(), StubBrain()
    run = make_run(memory, config, playbook, llm=llm, brain=brain, agents=agents,
                   budget=Budget(max_simulations=8, per_round=4, max_llm_calls=None))
    run.out = io.StringIO()
    report = run.run(5)
    assert report["stop_reason"].startswith("loop guard: ")
    assert "model calls" in report["stop_reason"]
    assert 6 <= report["model_calls"] <= 9 and brain.submitted == []
    assert "loop guard" in run.out.getvalue()


def test_loop_guard_stops_on_time_with_nothing_sent(memory, config, playbook):
    agents = dataclasses.replace(AGENTS, refill_passes=50, stall_model_calls=None,
                                 stall_minutes=60, stop_after_idle_rounds=None)
    run = make_run(memory, config, playbook, llm=NeverValid(), agents=agents,
                   budget=Budget(max_simulations=8, per_round=4, max_llm_calls=None))
    now = [0.0]

    def clock():                     # every look at the clock is 10 minutes later
        now[0] += 600
        return now[0]
    run._clock = clock
    report = run.run(5)
    assert report["stop_reason"].startswith("loop guard: ") and "min" in report["stop_reason"]


def test_progress_keeps_the_guard_quiet(memory, config, playbook):
    agents = dataclasses.replace(AGENTS, stall_model_calls=6, stall_minutes=60)
    report = make_run(memory, config, playbook, agents=agents).run(2)
    assert not report["stop_reason"].startswith("loop guard")
    assert report["simulations_used"] > 0


class LoginEnds(StubBrain):
    def login_stop(self):
        return "BRAIN login ends in 20 min"


def test_run_stops_before_a_round_when_the_login_ends_soon(memory, config, playbook):
    brain = LoginEnds()
    report = make_run(memory, config, playbook, brain=brain).run(2)
    assert report["stop_reason"] == "BRAIN login ends in 20 min"
    assert brain.submitted == []


class LoginLost(StubBrain):
    def submit(self, c):
        raise BrainLoginRequired("BRAIN asks for biometric verification")


def test_login_lost_mid_run_stops_cleanly_with_a_report(memory, config, playbook):
    report = make_run(memory, config, playbook, brain=LoginLost()).run(2)
    assert report["stop_reason"].startswith("BRAIN login lost: ")
    assert "rounds" in report


class Session:
    def __init__(self, left):
        self.left = left

    def expires_in(self):
        return self.left


def test_client_login_stop_uses_the_reserve():
    cfg = BrainApiConfig(settings={"instrumentType": "EQUITY"}, login_reserve_seconds=2700)
    assert RealBrainClient(Session(3000), cfg).login_stop() is None
    assert "login ends in 30 min" in RealBrainClient(Session(1800), cfg).login_stop()
    assert RealBrainClient(Session(None), cfg).login_stop() is None
    off = BrainApiConfig(settings={"instrumentType": "EQUITY"})
    assert RealBrainClient(Session(10), off).login_stop() is None


def test_research_never_prompts_after_its_start_login():
    from alpha_research.cli import _unattended

    class S:
        on_biometric = staticmethod(lambda url: None)
        ask_credentials = staticmethod(lambda: ("a", "b"))
    s = S()
    _unattended(s)
    assert s.on_biometric is None and s.ask_credentials is None
