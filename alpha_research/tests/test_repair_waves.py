"""Repair waves in the research loop (owner 2026-11-23): a near miss gets several different
fixes simulated together; the next wave starts from the best child if it got closer to
accept, else from the same parent with its next levers; the Director is credited once per
wave."""

import dataclasses

from alpha_research.lever_stats import RepairLearningConfig

from alpha_research.research import Budget
from test_research import AGENTS, ARM, StubBrain, _orphan_repair, make_run


def agents(**repair):
    return dataclasses.replace(AGENTS, repair=dataclasses.replace(
        RepairLearningConfig(), max_depth=3, max_trials_per_root=12, patience=8,
        max_share_per_round=1.0, **repair))


class NeverFixed(StubBrain):
    """Volume alphas fail turnover whatever their decay: no fix helps."""

    def submit(self, c):
        self.submitted.append(c)
        fid = "sim_tvr_01" if "volume" in c.expression else "sim_acc_01"
        self.fx.register(c.candidate_id, fid)
        return self.fx.submit(c)


def test_wave_runs_together_and_an_accept_ends_the_lineage(memory, make_config, playbook,
                                                         candidate):
    _orphan_repair(memory, candidate)
    brain = StubBrain()
    run = make_run(memory, make_config(simulator__max_concurrent=8), playbook, brain=brain,
                   agents=agents(fixes_per_wave=4))
    before = run.director.state[ARM]
    credits_before = before.successes + before.failures   # warm start from memory
    report = run.run(1)
    wave = report["repair_waves"][0]
    # four different turnover fixes, all sent to BRAIN in the first batch
    assert wave["parent"] == "old_001" and len(wave["fixes"]) == 4
    assert [c.candidate_id for c in brain.submitted[:4]] == list(wave["fixes"])
    assert wave["fixes"]["old_001_r1a"] == "increase_decay"
    assert report["rounds"][0]["arms"][0]["slots"] == 4
    # the decay fix passed, so no further wave for this lineage
    assert wave["results"]["old_001_r1a"]["decision"] == "accept / none"
    assert wave["next"].startswith("accepted")
    assert not [w for w in report["repair_waves"] if w["parent"].startswith("old_001_r1")]
    # one Director credit for the whole wave (an accept), not one per child; the rest of
    # the round's credits come from fresh alphas
    s = run.director.state[ARM]
    fresh = [o for r in report["rounds"] for a in r["arms"] if not a.get("repairs_first")
             for o in a["outcomes"] if o["decision"] in ("accept", "reject", "repair")]
    assert s.successes + s.failures == credits_before + 1 + len(fresh)


def test_no_child_closer_starts_the_next_wave_on_the_same_parent(memory, make_config,
                                                                 playbook, candidate):
    _orphan_repair(memory, candidate)
    brain = NeverFixed()
    run = make_run(memory, make_config(simulator__max_concurrent=8), playbook, brain=brain,
                   agents=agents(fixes_per_wave=2, waves_per_round=1))
    report = run.run(2)
    first, second = [w for w in report["repair_waves"] if w["parent"] == "old_001"][:2]
    assert "no child got closer" in first["next"]
    # the second wave uses levers the first did not
    assert not set(first["fixes"].values()) & set(second["fixes"].values())
    assert list(second["fixes"]) == ["old_001_r1c", "old_001_r1d"]


def test_next_start_prefers_the_closest_child_and_its_siblings_levers(memory, make_config,
                                                                      playbook, candidate):
    from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord
    run = make_run(memory, make_config(), playbook, agents=agents())

    def node(cid, lever, tvr):
        c = candidate(candidate_id=cid, lever=lever, parent_id="p" if lever else None)
        d = DecisionRecord(cid, "repair", "turnover", gate_results=[
            GateResult("turnover_range", False, tvr, (0.01, 0.7), "")])
        return c, d, ResultRecord(cid, "complete", sharpe=1.7, turnover=tvr)

    parent = node("p", None, 0.90)
    kids = [node("p_r1a", "increase_decay", 0.80), node("p_r1b", "hump", 0.75),
            node("p_r1c", "wrap_in_smoothing", 0.95)]
    c, d, r, priority, why = run._next_start(parent, kids)
    assert c.candidate_id == "p_r1b" and priority == ["increase_decay"] and "p_r1b" in why
    c, *_ = run._next_start(parent, [node("p_r1c", "wrap_in_smoothing", 0.95)])
    assert c.candidate_id == "p"                      # nobody closer: the parent again


def test_overfitting_is_a_flag_not_a_rejection(memory, make_config, playbook, candidate):
    # an impossible deflated-Sharpe floor: every accept misses it, and stays an accept
    config = make_config(auditor__min_deflated_sharpe=99.0, gates__brain_checks_only=True)
    report = make_run(memory, config, playbook).run(1)
    accepted = [o for r in report["rounds"] for a in r["arms"] for o in a["outcomes"]
                if o["decision"] == "accept"]
    assert accepted and all(o.get("overfitting_flag") for o in accepted)
    assert len(report["overfitting_flags"]) == len(accepted)


class AllTimeout(StubBrain):
    """Every alpha times out on our side (BRAIN still running it)."""

    def submit(self, c):
        self.submitted.append(c)
        return "sim_slow_" + c.candidate_id

    def poll(self, sim_id):
        return {"status": "running", "retry_after": 5}


def test_a_wave_with_no_completed_child_waits_for_recovery(memory, make_config, playbook,
                                                          candidate):
    # run 2026-11-23: four ASI fixes timed out and a second wave of four was started at once
    _orphan_repair(memory, candidate)
    run = make_run(memory, make_config(simulator__max_concurrent=8,
                                       simulator__max_poll_seconds=30), playbook,
                   brain=AllTimeout(), agents=agents(fixes_per_wave=2, waves_per_round=1))
    report = run.run(1)
    wave = next(w for w in report["repair_waves"] if w["parent"] == "old_001")
    assert all(r["decision"] == "reject / incomplete" for r in wave["results"].values())
    assert wave["next"].startswith("none: no child completed")
    assert len([w for w in report["repair_waves"] if w["parent"] == "old_001"]) == 1


def test_a_wave_joins_the_running_batch_within_the_round(memory, make_config, playbook):
    # a fresh alpha fails turnover; its wave starts in a slot freed in the same batch, and
    # the simulation budget still caps everything
    brain = StubBrain()
    run = make_run(memory, make_config(simulator__max_concurrent=8), playbook, brain=brain,
                   agents=agents(fixes_per_wave=2, waves_per_round=1),
                   budget=Budget(max_simulations=6, per_round=4, max_llm_calls=100))
    report = run.run(1)
    mid = [a for a in report["rounds"][0]["arms"] if a.get("repairs_mid_batch")]
    assert mid and all(o["candidate_id"].split("_")[-1].startswith("r1")
                       for a in mid for o in a["outcomes"])
    assert report["simulations_used"] == len(brain.submitted) <= 6
    # one wave a round: no second top-up wave in round 1
    assert len([w for w in report["repair_waves"]]) >= 1 and len(mid) == 1


def test_fill_freed_slots_can_be_switched_off(memory, make_config, playbook):
    brain = StubBrain()
    run = make_run(memory, make_config(simulator__max_concurrent=8), playbook, brain=brain,
                   agents=agents(fixes_per_wave=2, fill_freed_slots=False))
    report = run.run(1)
    assert not [a for r in report["rounds"] for a in r["arms"] if a.get("repairs_mid_batch")]
