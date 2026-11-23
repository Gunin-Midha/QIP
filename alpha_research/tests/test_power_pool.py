"""Power Pool candidates (power_pool.py): tagged in the decision and printed, never acted on.
The case is BRAIN alpha XgaP3Rpl (CHN, 2026-11-15): regular misses only WARNING, every
ruled check passing, accepted into the Power Pool while our pipeline rejected it."""

import io

from alpha_research import cli
from alpha_research.pipeline import Pipeline
from alpha_research.power_pool import PREFIX, candidate_reason, warned_checks
from alpha_research.schemas import DecisionRecord, ResultRecord
from alpha_research.simulator import VirtualClock

from conftest import ROOT

SUBMISSION = ("REGULAR_SUBMISSION", "POWER_POOL_DESCRIPTION_LENGTH", "MATCHES_THEMES",
              "POWER_POOL_CORRELATION")
NOTES = ["brain_check_not_counted: LOW_SHARPE='WARNING' value 1.23 limit 2.07",
         "brain_check_not_counted: LOW_FITNESS='WARNING' value 0.92 limit 1.0",
         "brain_check_not_counted: MATCHES_THEMES='WARNING'",
         "brain_alpha_id: XgaP3Rpl"]


def xga(**kw):
    base = dict(candidate_id="c1", status="complete", sharpe=1.23, fitness=0.92, turnover=0.05,
                checks={"LOW_SUB_UNIVERSE_SHARPE": True, "POWER_POOL_CORRELATION": True,
                        "REGULAR_SUBMISSION": True},
                check_values={"POWER_POOL_CORRELATION": {"value": None, "limit": 0.5}},
                notes=list(NOTES))
    base.update(kw)
    return ResultRecord(**base)


def reject():
    return DecisionRecord("c1", "reject", "sharpe")


def test_the_xga_case_is_a_candidate():
    why = candidate_reason(xga(), reject(), SUBMISSION)
    assert why.startswith(PREFIX) and "(LOW_SHARPE, LOW_FITNESS)" in why and "passes" in why
    assert warned_checks(xga(), SUBMISSION) == ["LOW_SHARPE", "LOW_FITNESS"]


def test_what_is_not_a_candidate():
    assert candidate_reason(xga(), DecisionRecord("c1", "accept", "none"), SUBMISSION) is None
    assert candidate_reason(xga(checks={"POWER_POOL_CORRELATION": False}), reject(),
                            SUBMISSION) is None
    assert candidate_reason(xga(check_values={}), reject(), SUBMISSION) is None   # not evaluated
    failed = xga(checks={"POWER_POOL_CORRELATION": True, "LOW_SUB_UNIVERSE_SHARPE": False})
    assert candidate_reason(failed, reject(), SUBMISSION) is None
    assert candidate_reason(xga(notes=["brain_check_not_counted: MATCHES_THEMES='WARNING'"]),
                            reject(), SUBMISSION) is None      # nothing regular warned
    assert candidate_reason(xga(status="failed"), reject(), SUBMISSION) is None
    # a failed submission-only check (daily quota) does not matter
    quota = xga(checks={"POWER_POOL_CORRELATION": True, "REGULAR_SUBMISSION": False})
    assert candidate_reason(quota, reject(), SUBMISSION) is not None


class OneBrain:
    def submit(self, c):
        return "sim1"

    def poll(self, sim):
        r = xga()
        return {"status": "complete", "units": "fraction", "sharpe": r.sharpe,
                "fitness": r.fitness, "turnover": r.turnover, "checks": r.checks,
                "check_values": r.check_values, "notes": r.notes}


def test_pipeline_tags_the_decision_and_prints_a_line(memory, make_config, playbook, candidate):
    config = make_config(gates__submission_checks=list(SUBMISSION))
    vc, out = VirtualClock(), io.StringIO()
    p = Pipeline(config, playbook, OneBrain(), memory, sleep=vc.sleep, clock=vc.clock, out=out)
    o = p.process(candidate(candidate_id="c1"))
    assert o.decision in ("reject", "repair")                 # the decision is unchanged
    stored = memory.decisions_for("c1")[-1]
    assert stored.reasons[-1].startswith(PREFIX)
    assert "  power pool candidate: c1 (BRAIN alpha XgaP3Rpl), USA TOP3000 delay 1: Sharpe 1.23" \
        in out.getvalue()


def test_cli_lists_candidates_from_memory(memory, candidate, capsys, tmp_path):
    c = candidate(candidate_id="c1", hypothesis="x" * 120)
    memory.put_candidate(c)
    memory.put_result(xga())
    memory.put_decision(reject())
    assert cli.main(["power-pool", "--data-dir", str(memory.data_dir),
                     "--config", str(ROOT / "config/thresholds.yaml")]) == 0
    out = capsys.readouterr().out
    assert "power pool candidate: c1 (BRAIN alpha XgaP3Rpl)" in out
    assert "description (120 characters, ok)" in out and "1 candidate(s)" in out
    assert cli.main(["power-pool", "--data-dir", str(tmp_path / "empty"),
                     "--config", str(ROOT / "config/thresholds.yaml")]) == 0
    assert "no Power Pool candidates" in capsys.readouterr().out
