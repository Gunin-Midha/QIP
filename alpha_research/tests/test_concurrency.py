"""Several simulations on BRAIN at once (OQ-3, owner 2026-11-14: 3 at a time): the
simulator keeps up to ``simulator.max_concurrent`` running, the pipeline decides them in
order, and the research loop queues screened alphas and sends them together."""

import pytest

from alpha_research.pipeline import Pipeline
from alpha_research.simulator import Simulator, VirtualClock

from test_research import StubBrain, make_run


class SlowBrain:
    """Each simulation runs for ``polls`` polls (retry_after 10 s), then completes. Records
    how many were running at once. ``retryable``: candidate ids whose first simulation
    fails with BRAIN's own error."""

    def __init__(self, polls=3, retryable=(), sharpe=None, polls_by=None):
        self.polls, self.retryable = polls, set(retryable)
        self.polls_by = polls_by or {}
        self.sharpe = sharpe or {}
        self.left, self.owner, self.submits = {}, {}, []
        self.active = self.max_active = 0

    def submit(self, c):
        sim = f"sim{len(self.submits)}"
        self.submits.append(c.candidate_id)
        self.left[sim] = self.polls_by.get(c.candidate_id, self.polls)
        self.owner[sim] = c.candidate_id
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        return sim

    def poll(self, sim):
        self.left[sim] -= 1
        if self.left[sim] > 0:
            return {"status": "running", "retry_after": 10}
        self.active -= 1
        cid = self.owner[sim]
        if cid in self.retryable:
            self.retryable.discard(cid)
            return {"status": "failed", "retryable": True, "notes": ["BRAIN error"]}
        return {"status": "complete", "units": "fraction",
                "sharpe": self.sharpe.get(cid, 1.0), "fitness": 0.5, "turnover": 0.2}


def sim(brain, memory, config):
    vc = VirtualClock()
    return Simulator(brain, memory, config, sleep=vc.sleep, clock=vc.clock), vc


def cands(candidate, n):
    return [candidate(candidate_id=f"c{i}", expression=f"rank(-ts_delta(close, {i + 2}))")
            for i in range(n)]


def test_up_to_the_limit_run_at_once_and_results_keep_their_order(memory, make_config,
                                                                   candidate):
    brain = SlowBrain(polls=3, sharpe={f"c{i}": float(i) for i in range(5)})
    s, vc = sim(brain, memory, make_config(simulator__max_concurrent=3))
    results = s.run_many(cands(candidate, 5))
    assert [r.candidate_id for r in results] == [f"c{i}" for i in range(5)]
    assert [r.sharpe for r in results] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert brain.max_active == 3 and all(r.status == "complete" for r in results)
    # two waves of 3 polls (two 10 s waits each) instead of five
    assert vc.now == pytest.approx(40)
    assert all(memory.results_for(f"c{i}") for i in range(5))


def test_one_at_a_time_is_the_old_behaviour(memory, make_config, candidate):
    brain = SlowBrain(polls=3)
    s, vc = sim(brain, memory, make_config(simulator__max_concurrent=1))
    s.run_many(cands(candidate, 3))
    assert brain.max_active == 1 and vc.now == pytest.approx(60)


def test_brain_error_is_resubmitted_once_inside_a_batch(memory, make_config, candidate):
    brain = SlowBrain(polls=1, retryable={"c1"})
    s, _ = sim(brain, memory, make_config(simulator__max_concurrent=3))
    results = s.run_many(cands(candidate, 3))
    assert brain.submits == ["c0", "c1", "c2", "c1"]
    assert results[1].status == "complete" and results[1].brain_sim_id == "sim3"
    assert results[1].notes[0] == "retried once after BRAIN's error on simulation sim1"


def test_each_simulation_times_out_on_its_own_clock(memory, make_config, candidate):
    brain = SlowBrain(polls=10**6)
    config = make_config(simulator__max_concurrent=2)
    s, _ = sim(brain, memory, config)
    results = s.run_many(cands(candidate, 3))
    assert [r.status for r in results] == ["timeout"] * 3 and len(brain.submits) == 3
    # the third started when the first two timed out, and got its full wait too
    limit = config.simulator.max_poll_seconds
    assert all(f"after {limit:.0f}s" in r.notes[0] for r in results)


def test_invalid_candidates_take_no_slot(memory, make_config, candidate):
    brain = SlowBrain(polls=1)
    s, _ = sim(brain, memory, make_config(simulator__max_concurrent=3))
    bad = candidate(candidate_id="bad", expression="rank(close")
    results = s.run_many([bad, *cands(candidate, 2)])
    assert [r.status for r in results] == ["invalid", "complete", "complete"]
    assert brain.submits == ["c0", "c1"]


def test_pipeline_decides_in_order_and_skips_duplicates_in_the_batch(memory, make_config,
                                                                     playbook, candidate):
    brain = SlowBrain(polls=2)
    vc = VirtualClock()
    pipe = Pipeline(make_config(simulator__max_concurrent=3), playbook, brain, memory,
                    sleep=vc.sleep, clock=vc.clock)
    a, b = cands(candidate, 2)
    twin = candidate(candidate_id="twin", expression=a.expression)   # same experiment as a
    outs = pipe.process_many([a, twin, b])
    assert [o.candidate_id for o in outs] == ["c0", "twin", "c1"]
    assert outs[1].decision == "duplicate" and brain.submits == ["c0", "c1"]
    assert all(memory.decisions_for(c) for c in ("c0", "c1"))


class CountingBrain(StubBrain):
    """The research tests' fixture BRAIN, counting simulations running at once."""

    def __init__(self):
        super().__init__()
        self.active = self.max_active = 0

    def submit(self, c):
        sim_id = super().submit(c)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        return sim_id

    def poll(self, sim_id):
        raw = super().poll(sim_id)
        if raw.get("status") in ("complete", "failed"):
            self.active -= 1
        return raw


@pytest.mark.parametrize("limit", [1, 3])
def test_research_run_sends_screened_alphas_together(memory, make_config, playbook, limit):
    brain = CountingBrain()
    run = make_run(memory, make_config(simulator__max_concurrent=limit), playbook,
                   brain=brain)
    report = run.run(3)
    assert brain.max_active == limit
    assert report["simulations_used"] == len(brain.submitted) <= 8
    # every alpha sent to BRAIN was decided and logged, none is left in the queue
    logged = [o for r in run.log for a in r["arms"] for o in a.get("outcomes", [])]
    assert {o["candidate_id"] for o in logged} >= set(c.candidate_id for c in brain.submitted)
    assert run.sim_queue == []


def test_cli_run_takes_max_concurrent(tmp_path, capsys):
    from alpha_research.cli import main
    args = ["run", "--candidates", "tests/fixtures/candidates/hand_written.json",
            "--config", "tests/fixtures/thresholds_test.yaml", "--data-dir", str(tmp_path / "d")]
    assert main([*args, "--max-concurrent", "8"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if " | " in ln]
    assert len(lines) == 31            # every hand-written candidate decided, 8 at a time
    assert main([*args, "--max-concurrent", "0"]) == 2


def test_progress_lines_come_as_each_finishes_then_decisions_in_order(memory, make_config,
                                                                      playbook, candidate):
    import io
    brain = SlowBrain(polls_by={"c0": 4, "c1": 1, "c2": 2})     # c0 finishes last
    vc, out = VirtualClock(), io.StringIO()
    pipe = Pipeline(make_config(simulator__max_concurrent=3), playbook, brain, memory,
                    sleep=vc.sleep, clock=vc.clock, out=out)
    pipe.process_many(cands(candidate, 3))
    lines = out.getvalue().splitlines()
    progress = [ln for ln in lines if ln.startswith("  BRAIN: ")]
    assert [ln.split()[1] for ln in progress] == ["c1", "c2", "c0"]
    assert progress[0] == "  BRAIN: c1 finished (1 of 3), complete: Sharpe 1.00, fitness 0.50, turnover 0.20"
    decided = [ln.split(" | ")[0] for ln in lines if " | " in ln]
    assert decided == ["c0", "c1", "c2"] and lines.index(progress[-1]) < lines.index(
        next(ln for ln in lines if " | " in ln))
    # one alpha on its own: no progress line, as before
    out.truncate(0), out.seek(0)
    pipe.process_many([candidate(candidate_id="solo", expression="rank(-ts_delta(close, 9))")])
    assert not [ln for ln in out.getvalue().splitlines() if ln.startswith("  BRAIN: ")]
