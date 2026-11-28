"""Run rs20261128045753 (2026-11-28) sat an hour after its last result with one alpha on BRAIN
and ended on "BRAIN login lost". Now the simulator never sleeps longer than
``simulator.max_poll_wait_seconds`` between polls (nor past an alpha's own timeout), and it
checks the login before each poll: with less than ``login_margin_seconds`` left it stops at
once, leaving the running alphas on BRAIN for the next start's recovery."""

import pytest

from alpha_research.brain_api import RealBrainClient, brain_config_from_dict
from alpha_research.brain_client import BrainLoginRequired

from test_concurrency import SlowBrain, cands, sim


class LongWaitBrain(SlowBrain):
    def poll(self, s):
        out = super().poll(s)
        if out.get("status") == "running":
            out["retry_after"] = 3600         # BRAIN asks for an hour
        return out


def test_poll_waits_are_capped(memory, make_config, candidate):
    brain = LongWaitBrain(polls=3)
    s, vc = sim(brain, memory, make_config(simulator__max_poll_wait_seconds=60))
    sleeps = []
    s.sleep = lambda x: (sleeps.append(x), vc.sleep(x))
    res = s.run_many(cands(candidate, 1))
    assert res[0].status == "complete"
    assert sleeps and max(sleeps) <= 60


def test_a_wait_never_runs_past_the_alphas_own_timeout(memory, make_config, candidate):
    brain = LongWaitBrain(polls=100)
    s, vc = sim(brain, memory, make_config(simulator__max_poll_wait_seconds=1000,
                                           simulator__max_poll_seconds=100))
    res = s.run_many(cands(candidate, 1))
    assert res[0].status == "timeout" and vc.clock() < 200


class ExpiringBrain(SlowBrain):
    """check_login raises once ``after`` polls were made."""

    def __init__(self, after, **kw):
        super().__init__(**kw)
        self.after, self.n = after, 0

    def poll(self, s):
        self.n += 1
        return super().poll(s)

    def check_login(self):
        if self.n >= self.after:
            raise BrainLoginRequired("BRAIN login ends in 2 min")


def test_a_login_ending_soon_stops_polling_and_leaves_alphas_for_recovery(memory, make_config,
                                                                         candidate):
    brain = ExpiringBrain(after=2, polls=10)
    s, vc = sim(brain, memory, make_config(simulator__max_concurrent=2))
    with pytest.raises(BrainLoginRequired):
        s.run_many(cands(candidate, 2))
    assert brain.n == 2                                 # no polling past the check
    # both were submitted and have no result: the next run start recovers them
    assert {u["candidate_id"] for u in memory.unfinished_submissions()} == {"c0", "c1"}


class FakeSession:
    def __init__(self, lefts):
        self.lefts, self.calls = list(lefts), 0

    def expires_in(self):
        self.calls += 1
        return self.lefts.pop(0) if self.lefts else None


def _client(session, margin=300, every=600):
    cfg = brain_config_from_dict({"settings": {"instrumentType": "EQUITY"},
                                  "login_margin_seconds": margin,
                                  "login_check_every_seconds": every})
    c = RealBrainClient.__new__(RealBrainClient)
    c.session, c.config = session, cfg
    return c


def test_check_login_asks_brain_at_most_every_interval(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr("alpha_research.brain_api.time.monotonic", lambda: t[0])
    sess = FakeSession([3600.0])
    c = _client(sess, every=10000)                     # no second question to BRAIN
    c.check_login()
    t[0] += 120
    c.check_login()                                    # counted down locally
    assert sess.calls == 1 and c.login_left() == pytest.approx(3480)
    t[0] += 3200                                       # 280 s left: below the 300 s margin
    with pytest.raises(BrainLoginRequired, match="login_margin_seconds"):
        c.check_login()


def test_a_login_ended_elsewhere_counts_as_lost(monkeypatch):
    t = [0.0]
    monkeypatch.setattr("alpha_research.brain_api.time.monotonic", lambda: t[0])
    sess = FakeSession([3600.0])                       # then BRAIN reports no login
    c = _client(sess, every=60)
    c.check_login()
    t[0] += 61
    with pytest.raises(BrainLoginRequired):
        c.check_login()


def test_margin_off_never_checks():
    sess = FakeSession([10.0])
    c = _client(sess, margin=None)
    c.check_login()
    assert sess.calls == 0
