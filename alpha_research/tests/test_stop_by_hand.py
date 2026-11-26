"""Ctrl-C during a research run (owner 2026-11-26): the first press finishes what is on
BRAIN and writes lessons + report; the second cancels BRAIN's simulations and still writes
rule lessons + report; the third quits at once."""

import os
import signal

import pytest

from test_research import RouterLLM, StubBrain, make_run


class InterruptingBrain(StubBrain):
    """Presses Ctrl-C ``presses`` times in a row on the first poll (a second press raises
    KeyboardInterrupt there, as the real handler does)."""

    def __init__(self, presses=1, real_signal=False):
        super().__init__()
        self.run = None
        self.presses, self.real_signal = presses, real_signal
        self.cancelled = []
        self.polls_at_press = None

    def poll(self, sim_id):
        if self.polls_at_press is None and self.presses:
            self.polls_at_press = len(self.submitted)
            for _ in range(self.presses):
                if self.real_signal:
                    os.kill(os.getpid(), signal.SIGINT)
                else:
                    self.run._on_interrupt(signal.SIGINT, None)
        return super().poll(sim_id)

    def cancel(self, sim_id):
        self.cancelled.append(sim_id)
        return "cancelled"


def _run(memory, config, playbook, brain, llm=None):
    run = make_run(memory, config, playbook, llm=llm or RouterLLM(), brain=brain)
    brain.run = run
    return run


def test_first_ctrl_c_finishes_brain_work_then_writes_lessons(memory, config, playbook):
    llm, brain = RouterLLM(), InterruptingBrain(presses=1)
    run = _run(memory, config, playbook, brain, llm)
    report = run.run(3, catch_interrupt=True)

    assert report["stop_reason"] == "stopped by hand (Ctrl-C)"
    # nothing new went to BRAIN after the press, and what was there was decided
    assert len(brain.submitted) == brain.polls_at_press
    assert not brain.cancelled
    for c in brain.submitted:
        assert memory.decisions_for(c.candidate_id)
    assert run.wave_log == []   # no repair wave started after the press
    # after the press the only model call is the Lesson Writer's
    at_press = llm.calls.index("lessons") if "lessons" in llm.calls else len(llm.calls)
    assert all(c != "lessons" for c in llm.calls[:at_press])
    assert report["lessons"] is not None
    assert memory.latest_lessons() is not None or report["lessons"]["written"] == 0
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler   # restored


def test_second_ctrl_c_cancels_brain_and_still_writes_rule_lessons(memory, config, playbook):
    llm, brain = RouterLLM(), InterruptingBrain(presses=2)
    run = _run(memory, config, playbook, brain, llm)
    report = run.run(3, catch_interrupt=True)

    assert "Ctrl-C twice" in report["stop_reason"]
    assert brain.cancelled                       # the running batch was cancelled on BRAIN
    assert report["lessons"] is not None
    assert report["lessons"]["source"] != "model"
    assert "lessons" not in llm.calls             # rules only: no model call on a hard stop


def test_third_ctrl_c_quits_at_once(memory, config, playbook):
    brain = InterruptingBrain(presses=2)
    run = _run(memory, config, playbook, brain)
    orig = brain.cancel

    def cancel(sim_id):   # the third press lands while cancelling
        orig(sim_id)
        run._on_interrupt(signal.SIGINT, None)
    brain.cancel = cancel
    with pytest.raises(KeyboardInterrupt):
        run.run(3, catch_interrupt=True)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_a_real_sigint_takes_the_smooth_path(memory, config, playbook):
    brain = InterruptingBrain(presses=1, real_signal=True)
    run = _run(memory, config, playbook, brain)
    report = run.run(3, catch_interrupt=True)
    assert report["stop_reason"] == "stopped by hand (Ctrl-C)"
    assert report["lessons"] is not None


def test_without_catch_interrupt_ctrl_c_is_plain_keyboardinterrupt(memory, config, playbook):
    class Raises(StubBrain):
        def poll(self, sim_id):
            raise KeyboardInterrupt

        def cancel(self, sim_id):
            return "cancelled"
    run = make_run(memory, config, playbook, brain=Raises())
    with pytest.raises(KeyboardInterrupt):
        run.run(3)
