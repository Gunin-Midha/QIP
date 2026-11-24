"""Recheck of alphas decided pending while a BRAIN check had no verdict yet (2026-11-24: two
CHN blends, Sharpe 3.8-4.0, stayed pending because PROD_CORRELATION was PENDING and
SELF_CORRELATION ERROR when their results were fetched, and nothing asked BRAIN again)."""
import io

from alpha_research.pipeline import Pipeline, awaiting_brain, unfinished_checks
from alpha_research.schemas import DecisionRecord, ResultRecord
from alpha_research.simulator import VirtualClock

PENDING_NOTES = ["brain_check_not_counted: SELF_CORRELATION='ERROR' limit 0.7",
                 "brain_check_not_counted: PROD_CORRELATION='PENDING' limit 0.7",
                 "brain_check_not_counted: MATCHES_THEMES='WARNING'"]


def raw(self_corr, checks, notes, check_values=None):
    return {"status": "complete", "units": "fraction", "sharpe": 3.83, "fitness": 4.19,
            "turnover": 0.2, "returns": 0.1, "drawdown": 0.05, "margin": 10.0,
            "sub_universe_sharpe": 1.9, "sub_universe_cutoff": 1.5,
            "weight_concentration": 0.03, "long_pnl": 1.0, "short_pnl": 1.0,
            "self_correlation": self_corr, "checks": checks,
            "check_values": check_values or {}, "notes": list(notes)}


class StubBrain:
    """poll answers in turn (the last repeats); no network."""

    def __init__(self, answers):
        self.answers, self.polls = list(answers), 0

    def submit(self, candidate):
        return "sim_chn"

    def poll(self, sim_id):
        a = self.answers[min(self.polls, len(self.answers) - 1)]
        self.polls += 1
        return a


def make(config, playbook, memory, client):
    vc = VirtualClock()
    return Pipeline(config, playbook, client, memory, sleep=vc.sleep, clock=vc.clock,
                    out=io.StringIO())


def test_unfinished_checks_reads_pending_and_error_only():
    r = ResultRecord("c1", "complete", notes=PENDING_NOTES)
    assert unfinished_checks(r) == {"SELF_CORRELATION": "ERROR", "PROD_CORRELATION": "PENDING"}
    assert unfinished_checks(None) == {}


def test_pending_alpha_is_decided_once_brain_finishes_the_check(config, playbook, memory,
                                                                candidate):
    first = raw(None, {"LOW_SHARPE": True}, PENDING_NOTES)
    later = raw(0.31, {"LOW_SHARPE": True, "PROD_CORRELATION": False},
                PENDING_NOTES[2:], {"PROD_CORRELATION": {"value": 0.83, "limit": 0.7}})
    client = StubBrain([first])
    c = candidate()
    out = make(config, playbook, memory, client).run([c])
    assert out[0].decision == "pending"
    r, d = memory.results_for("c1")[-1], memory.decisions_for("c1")[-1]
    assert awaiting_brain(d, r) and r.brain_sim_id == "sim_chn"

    # BRAIN still computing: nothing new is stored, the alpha is reported as still waiting
    outs, still = make(config, playbook, memory, client).recheck_pending()
    assert outs == [] and still == ["c1"]
    assert len(memory.results_for("c1")) == 1 and len(memory.decisions_for("c1")) == 1

    # BRAIN done: the new result is stored and the alpha decided on PROD_CORRELATION
    client.answers.append(later)
    client.polls = len(client.answers) - 1
    outs, still = make(config, playbook, memory, client).recheck_pending()
    assert still == [] and [(o.candidate_id, o.decision, o.failure_class) for o in outs] == \
        [("c1", "repair", "prod_correlation")]
    new = memory.results_for("c1")[-1]
    assert new.notes[0].startswith("rechecked:") and new.self_correlation == 0.31
    assert memory.decisions_for("c1")[-1].failure_class == "prod_correlation"
    # decided: a further recheck has nothing to do and never asks BRAIN
    polls = client.polls
    assert make(config, playbook, memory, client).recheck_pending() == ([], [])
    assert client.polls == polls


def test_recheck_skips_pending_without_unfinished_checks_and_other_ids(config, playbook, memory,
                                                                     candidate):
    memory.put_candidate(candidate(candidate_id="p1"))
    memory.put_result(ResultRecord("p1", "complete", brain_sim_id="s1", sharpe=2.0))
    memory.put_decision(DecisionRecord("p1", "pending", "none"))
    memory.put_candidate(candidate(candidate_id="p2"))
    memory.put_result(ResultRecord("p2", "complete", brain_sim_id="s2", notes=PENDING_NOTES))
    memory.put_decision(DecisionRecord("p2", "pending", "none"))
    client = StubBrain([raw(None, {}, PENDING_NOTES)])
    assert make(config, playbook, memory, client).recheck_pending(only={"p1"}) == ([], [])
    assert client.polls == 0
    assert make(config, playbook, memory, client).recheck_pending() == ([], ["p2"])


def test_alpha_accepted_with_an_unfinished_check_becomes_pending(config, playbook, memory,
                                                                 candidate):
    # 2026-11-24: breadth_arm was accepted while PROD_CORRELATION was ERROR
    r = ResultRecord("c1", "complete", brain_sim_id="s1", sharpe=3.65, fitness=3.88,
                     turnover=0.16, sub_universe_sharpe=2.2, sub_universe_cutoff=1.5,
                     self_correlation=0.2, checks={"LOW_SHARPE": True},
                     notes=["brain_check_not_counted: PROD_CORRELATION='ERROR' limit 0.7"])
    memory.put_candidate(candidate())
    memory.put_result(r)
    memory.put_decision(DecisionRecord("c1", "accept", "none"))
    from alpha_research.pool import Pool
    Pool(memory).admit(candidate(), r, DecisionRecord("c1", "accept", "none"), True)
    client = StubBrain([raw(0.2, {"LOW_SHARPE": True}, r.notes)])   # BRAIN: still no verdict
    outs, still = make(config, playbook, memory, client).recheck_pending()
    assert [o.decision for o in outs] == ["pending"] and still == ["c1"]
    assert Pool(memory).alpha_ids() == []               # no longer accepted: retired (2026-11-25)
    assert len(memory.results_for("c1")) == 1           # nothing new from BRAIN stored
    # now pending: a later recheck only asks BRAIN, it does not re-decide again
    outs, still = make(config, playbook, memory, client).recheck_pending()
    assert outs == [] and still == ["c1"]
