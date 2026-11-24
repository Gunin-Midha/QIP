"""Pool follows the latest decisions (2026-11-25): accepted alphas join under
gates.brain_checks_only even when our informational pool-diversity check failed; an alpha
re-decided away from accept is retired (append-only pool_events). Also the CHN rescue
lessons in the blend lever: partner weight by the parent's production correlation and
less-used partner fields first."""

from alpha_research.memory import MemoryStore
from alpha_research.pool import Pool
from alpha_research.repair import Playbook
from alpha_research.repair_agent import blend_weight, partner_fields
from alpha_research.schemas import DecisionRecord, ResultRecord

from conftest import PLAYBOOK, gaussian_series
from test_rescue_lessons import META, SPEC


def _store(memory, candidate, cid, decision, seed=1):
    c = candidate(candidate_id=cid, expression=f"rank(x{seed})")
    r = ResultRecord(cid, "complete", sharpe=2.0, daily_pnl=gaussian_series(300, seed=seed))
    memory.put_candidate(c)
    memory.put_result(r)
    memory.put_decision(DecisionRecord(cid, decision, "none" if decision == "accept"
                                       else "prod_correlation"))
    return c, r


def test_retire_and_restore_are_events_and_readers_skip_retired(memory, candidate):
    pool = Pool(memory)
    c, r = _store(memory, candidate, "a", "accept")
    assert pool.admit(c, r, DecisionRecord("a", "accept", "none"), True)
    assert pool.admit(c, r, DecisionRecord("a", "accept", "none"), True) is False  # once
    assert pool.retire("a", "re-decided") and pool.retire("a", "again") is False
    assert pool.count() == 0 and pool.alpha_ids() == [] and pool.matrix().empty
    assert pool.admit(c, r, DecisionRecord("a", "accept", "none"), True)            # restore
    assert pool.alpha_ids() == ["a"] and pool.count() == 1
    events = [row["event"] for row in memory.conn.execute(
        "SELECT event FROM pool_events ORDER BY id")]
    assert events == ["retire", "restore"]


def test_sync_admits_accepted_and_retires_redecided(memory, candidate):
    pool = Pool(memory)
    c, r = _store(memory, candidate, "sib", "accept", seed=1)
    pool.admit(c, r, DecisionRecord("sib", "accept", "none"), True)
    memory.put_decision(DecisionRecord("sib", "repair", "prod_correlation"))   # re-decided
    _store(memory, candidate, "win", "accept", seed=2)                          # never admitted
    _store(memory, candidate, "lost", "repair", seed=3)
    assert Pool(memory).sync(admit_accepted=False) == {"admitted": [], "retired": ["sib"]}
    assert pool.sync(admit_accepted=True) == {"admitted": ["win"], "retired": []}
    assert pool.alpha_ids() == ["win"]
    assert pool.sync(admit_accepted=True) == {"admitted": [], "retired": []}


def test_pool_events_table_survives_reopen(tmp_path, candidate):
    m = MemoryStore(tmp_path)
    c, r = _store(m, candidate, "a", "accept")
    Pool(m).admit(c, r, DecisionRecord("a", "accept", "none"), True)
    Pool(m).retire("a", "x")
    m.close()
    m = MemoryStore(tmp_path)
    assert Pool(m).alpha_ids() == []
    m.close()


def test_blend_weight_follows_the_parents_production_correlation():
    step = {"weight": 1.0, "weight_by_prod": [[0.78, 1.0], [1.0, 2.0]]}
    def res(v):
        return ResultRecord("p", "complete",
                            check_values={"PROD_CORRELATION": {"value": v, "limit": 0.7}})
    assert blend_weight(step, res(0.73)) == 1.0       # EUR / IND parents passed at 1:1
    assert blend_weight(step, res(0.93)) == 2.0       # CHN retail part needed 2:1
    assert blend_weight(step, res(None)) == 1.0
    assert blend_weight(step, None) == 1.0
    assert blend_weight({"weight": 0.5}, res(0.93)) == 0.5


def test_partner_fields_less_used_first_and_playbook_step():
    plain = [f for f, _ in partner_fields(META, {"pv1"}, SPEC, min_coverage=0.5)]
    low = [f for f, _ in partner_fields(META, {"pv1"}, SPEC, min_coverage=0.5,
                                        prefer_low_usage=True)]
    assert set(plain) == set(low)
    counts = {f: META[f]["alpha_count"] for f in low}
    median = sorted(counts.values())[(len(counts) - 1) // 2]
    first_high = next(i for i, f in enumerate(low) if counts[f] > median)
    assert all(counts[f] <= median for f in low[:first_high])
    step = Playbook.load(PLAYBOOK).entries["prod_correlation"][0].step
    assert step["min_coverage"] == 0.3 and step["prefer_low_usage"] is True
    assert step["weight_by_prod"] == [[0.78, 1.0], [1.0, 2.0]]


# ------------------------------------------- pool = submitted alphas only (owner 2026-11-25)

def test_submitted_mode_pipeline_never_touches_the_pool(make_config, playbook, memory, candidate):
    import io
    from alpha_research.brain_client import FixtureBrainClient
    from alpha_research.pipeline import Pipeline
    from alpha_research.simulator import VirtualClock
    from conftest import FIXTURES
    import pytest
    from alpha_research.config import ConfigError
    with pytest.raises(ConfigError, match="pool_admits"):
        make_config(diversity__pool_admits="sometimes")
    config = make_config(diversity__pool_admits="submitted")
    assert config.diversity.pool_admits == "submitted"
    assert make_config().diversity.pool_admits == "accept"     # older configs
    pool = Pool(memory)
    c, r = _store(memory, candidate, "held", "accept", seed=5)
    assert pool.add_submitted("held") == "added"
    assert pool.add_submitted("held") == "already in the pool"
    # re-decided away from accept: a submitted alpha stays (it is on BRAIN)
    vc = VirtualClock()
    pipe = Pipeline(config, playbook, FixtureBrainClient(FIXTURES / "brain_results"), memory,
                    sleep=vc.sleep, clock=vc.clock, out=io.StringIO())
    pipe._evaluate(c, ResultRecord("held", "failed"))
    assert memory.decisions_for("held")[-1].decision == "reject"
    assert pool.alpha_ids() == ["held"]
    assert pool.retire("held", "owner") and pool.add_submitted("held") == "restored"
    events = [row["event"] for row in memory.conn.execute("SELECT event FROM pool_events ORDER BY id")]
    assert events == ["submitted", "retire", "restore"]


def test_add_submitted_needs_a_complete_result(memory, candidate):
    import pytest
    memory.put_candidate(candidate(candidate_id="x"))
    memory.put_result(ResultRecord("x", "failed"))
    with pytest.raises(ValueError, match="no complete result"):
        Pool(memory).add_submitted("x")
    with pytest.raises(ValueError):
        Pool(memory).add_submitted("missing")


def test_cli_pool_by_brain_id(tmp_path, candidate, capsys):
    from alpha_research.cli import main
    from conftest import ROOT
    m = MemoryStore(tmp_path)
    c = candidate(candidate_id="cand1", expression="rank(x1)")
    m.put_candidate(c)
    m.put_result(ResultRecord("cand1", "complete", sharpe=2.0, notes=["brain_alpha_id: AbCd123"],
                              daily_pnl=gaussian_series(300, seed=1)))
    m.put_decision(DecisionRecord("cand1", "accept", "none"))
    m.close()
    cfg = str(ROOT / "config" / "thresholds.yaml")
    assert main(["pool", "--data-dir", str(tmp_path), "--config", cfg, "--submitted", "abcd123"]) == 0
    assert "added: cand1" in capsys.readouterr().out
    assert main(["pool", "--data-dir", str(tmp_path), "--config", cfg, "--submitted", "nope"]) == 2
    assert main(["pool", "--data-dir", str(tmp_path), "--config", cfg, "--remove", "AbCd123"]) == 0
    out = capsys.readouterr().out
    assert "removed: cand1" in out and "0 alpha(s) in the pool" in out
