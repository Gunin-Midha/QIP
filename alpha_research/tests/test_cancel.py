"""Stuck BRAIN simulations and unfinished checks (2026-11-24): BRAIN keeps running what we
stop waiting for (7 of our simulations sat at 90% for up to 10 days), so stuck submissions
are cancelled (DELETE /simulations/<id>, checked on our account) after
simulator.cancel_stuck_after_hours, and Ctrl-C cancels the batch's running simulations.
An alpha whose BRAIN check has no verdict yet (PENDING / ERROR) is pending, never accepted."""
import io
from datetime import datetime, timedelta, timezone

import pytest

from alpha_research import evaluator
from alpha_research.brain_api import BrainApiError
from alpha_research.pipeline import Pipeline
from alpha_research.schemas import ResultRecord
from alpha_research.simulator import Simulator, VirtualClock
from test_brain_api import FakeTransport, bcfg, make_client, ok  # noqa: F401  (fixture)

ALL_PASS = {"LOW_SHARPE": True, "LOW_FITNESS": True, "SELF_CORRELATION": True}


def complete(checks, notes=()):
    return ResultRecord("c1", "complete", sharpe=3.65, fitness=3.88, turnover=0.16,
                        sub_universe_sharpe=2.2, sub_universe_cutoff=1.5, self_correlation=0.2,
                        weight_concentration=0.02, checks=dict(checks), notes=list(notes))


# ------------------------------------------------------------------ BRAIN client


def test_real_client_cancels_with_delete(bcfg):
    client, transport, _ = make_client({("DELETE", "/simulations/s1"): [ok(status=200)]}, bcfg)
    assert client.cancel("s1") == "cancelled"
    assert transport.paths("DELETE") == ["/simulations/s1"]
    assert client.cancel("gone_one") == "gone"          # FakeTransport: unknown path -> 404


def test_real_client_cancel_refused_raises(bcfg):
    client, _, _ = make_client({("DELETE", "/simulations/s1"): [ok({"detail": "no"}, status=403)]},
                               bcfg)
    with pytest.raises(BrainApiError, match="403"):
        client.cancel("s1")


# ------------------------------------------------------------------ unfinished checks


def test_unfinished_prod_correlation_makes_a_passing_alpha_pending(config, playbook):
    # 2026-11-24: breadth_arm passed every check with a verdict, PROD_CORRELATION was ERROR,
    # and it was accepted
    r = complete(ALL_PASS, ["brain_check_not_counted: PROD_CORRELATION='ERROR' limit 0.7"])
    gates = evaluator.run_gates(r, config)
    g = next(x for x in gates if x.name == "brain:PROD_CORRELATION")
    assert g.passed is None and "ERROR" in g.note
    d = evaluator.decide(r, gates, playbook, [], False, [])
    assert d.decision == "pending"


def test_a_failed_check_still_beats_an_unfinished_one(config, playbook):
    r = complete({**ALL_PASS, "LOW_SHARPE": False},
                 ["brain_check_not_counted: PROD_CORRELATION='PENDING' limit 0.7"])
    d = evaluator.decide(r, evaluator.run_gates(r, config), playbook, [], False, [])
    assert d.decision in ("repair", "reject") and d.failure_class == "sharpe"


def test_warnings_and_submission_checks_are_not_held_pending(make_config, playbook):
    cfg = make_config(gates__submission_checks=["MATCHES_THEMES"])
    r = complete(ALL_PASS, ["brain_check_not_counted: MATCHES_THEMES='PENDING'",
                            "brain_check_not_counted: OSMOSIS_ALLOCATION='WARNING'"])
    names = {g.name for g in evaluator.run_gates(r, cfg)}
    assert "brain:MATCHES_THEMES" not in names and "brain:OSMOSIS_ALLOCATION" not in names


# ------------------------------------------------------------------ simulator


class Brain:
    """Submits s<n>; ``poll`` answers running, or raises KeyboardInterrupt after ``stop_after``
    polls (a Ctrl-C while the batch runs)."""

    def __init__(self, stop_after=None):
        self.n, self.polls, self.stop_after = 0, 0, stop_after
        self.cancelled = []

    def submit(self, candidate):
        self.n += 1
        return f"s{self.n}"

    def poll(self, sim_id):
        self.polls += 1
        if self.stop_after is not None and self.polls > self.stop_after:
            raise KeyboardInterrupt
        return {"status": "running", "retry_after": 5}

    def cancel(self, sim_id):
        self.cancelled.append(sim_id)
        return "cancelled"


def test_ctrl_c_cancels_the_running_simulations(memory, make_config, candidate):
    cfg = make_config(simulator__max_concurrent=2)
    brain = Brain(stop_after=3)
    vc = VirtualClock()
    sim = Simulator(brain, memory, cfg, sleep=vc.sleep, clock=vc.clock)
    cands = [candidate(candidate_id=f"c{i}", decay=i) for i in range(3)]
    with pytest.raises(KeyboardInterrupt):
        sim.run_many(cands)
    assert brain.cancelled == ["s1", "s2"]           # the third was never submitted
    for cid in ("c0", "c1"):
        r = memory.results_for(cid)[-1]
        assert r.status == "failed" and r.notes[0].startswith("cancelled on BRAIN (cancelled)")
    assert memory.unfinished_submissions() == []


def test_a_refused_cancel_keeps_the_submission_for_later(memory, config, candidate):
    class Refuses(Brain):
        def cancel(self, sim_id):
            raise BrainApiError("cancel s1 returned 403")
    memory.put_candidate(candidate())
    memory.put_pending("c1", "s1")
    sim = Simulator(Refuses(), memory, config)
    assert sim.cancel(candidate(), "s1", "test") is None
    assert memory.unfinished_submissions() == [{"candidate_id": "c1", "brain_sim_id": "s1"}]


# ------------------------------------------------------------------ stuck submissions


def test_stuck_submissions_are_the_old_unfinished_ones(memory, config, playbook, candidate):
    for cid in ("old", "new"):
        memory.put_candidate(candidate(candidate_id=cid))
        memory.put_pending(cid, f"s_{cid}")
    later = datetime.now(timezone.utc) + timedelta(hours=7)
    pipe = Pipeline(config, playbook, Brain(), memory, out=io.StringIO())
    assert [r["candidate_id"] for r in pipe.stuck_submissions(6, now=later)] == ["old", "new"]
    assert pipe.stuck_submissions(8, now=later) == []
    out = pipe.cancel_submission(candidate(candidate_id="old"), "s_old", "stuck")
    assert (out.decision, out.failure_class) == ("reject", "incomplete")
    assert [u["candidate_id"] for u in memory.unfinished_submissions()] == ["new"]
