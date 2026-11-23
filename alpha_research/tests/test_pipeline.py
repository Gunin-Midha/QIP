import io
import json

import pytest

from alpha_research import brain_api, cli
from alpha_research.brain_client import FixtureBrainClient
from alpha_research.memory import MemoryStore
from alpha_research.pipeline import Pipeline, load_candidate_file
from alpha_research.schemas import SchemaError
from alpha_research.simulator import VirtualClock
from conftest import BRAIN_DIR, CANDIDATES_FILE, MAIN_CONFIG, PLAYBOOK, ROOT, TEST_CONFIG


def run_file(config, playbook, data_dir, path=CANDIDATES_FILE):
    entries = load_candidate_file(path)
    client = FixtureBrainClient(BRAIN_DIR, {e.candidate.candidate_id: e.meta["fixture_id"]
                                            for e in entries if e.meta.get("fixture_id")})
    vc = VirtualClock()
    out = io.StringIO()
    memory = MemoryStore(data_dir)
    pipe = Pipeline(config, playbook, client, memory, sleep=vc.sleep, clock=vc.clock, out=out)
    outcomes = pipe.run([e.candidate for e in entries])
    return entries, outcomes, out.getvalue(), memory, pipe


def test_candidate_file_coverage():
    entries = load_candidate_file(CANDIDATES_FILE)
    expected = [(e.meta["expected_decision"], e.meta["expected_failure_class"]) for e in entries]
    count = lambda d, f=None: sum(1 for x in expected if x[0] == d and (f is None or x[1] == f))
    assert len(entries) == 31
    assert count("accept") == 5
    assert count("reject", "sharpe") == 2 and count("repair", "sharpe") == 3   # near misses
    assert count("repair", "turnover") == 5
    assert count("reject", "concentration") + count("reject", "drawdown") == 3
    assert count("repair", "sub_universe") == 3
    assert count("repair", "self_correlation") == 3
    assert count("duplicate") == 2
    assert count("reject", "incomplete") == 4     # 2 invalid + 2 timeouts


def test_pipeline_acceptance(tmp_path, config, playbook):
    entries, outcomes, out, memory, pipe = run_file(config, playbook, tmp_path / "data")
    mismatches = [
        f"{e.candidate.candidate_id}: expected {e.meta['expected_decision']}/"
        f"{e.meta['expected_failure_class']}, got {o.decision}/{o.failure_class}"
        for e, o in zip(entries, outcomes)
        if (o.decision, o.failure_class) != (e.meta["expected_decision"], e.meta["expected_failure_class"])
    ]
    assert not mismatches, "\n" + "\n".join(mismatches)

    lines = out.strip().splitlines()
    assert len(lines) == 31
    assert lines[0].startswith("acc_01 | accept | Sharpe 1.85 | ")

    assert pipe.pool.count() == 5
    assert pipe.pool.matrix().shape[1] == 5
    # every candidate that reached BRAIN registers a variant (not invalids or duplicates)
    total_variants = memory.conn.execute("SELECT COUNT(*) FROM variants").fetchone()[0]
    assert total_variants == 31 - 2 - 2
    assert memory.pending() == []
    memory.close()


def test_rerun_skips_everything_as_duplicate(tmp_path, config, playbook):
    _, _, _, memory, _ = run_file(config, playbook, tmp_path / "data")
    memory.close()
    _, outcomes, _, memory, pipe = run_file(config, playbook, tmp_path / "data")
    assert {o.decision for o in outcomes} == {"duplicate"}
    assert pipe.pool.count() == 5
    memory.close()


def test_structural_error_in_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"candidate_id": "x", "expression": "rank(close)"}]))
    with pytest.raises(SchemaError):
        load_candidate_file(bad)


def test_cli_success(tmp_path, capsys):
    rc = cli.main(["run", "--candidates", str(CANDIDATES_FILE), "--config", str(TEST_CONFIG),
                   "--playbook", str(PLAYBOOK), "--fixtures-dir", str(BRAIN_DIR),
                   "--data-dir", str(tmp_path / "d")])
    assert rc == 0
    assert "div_01 | reject / self_correlation" in capsys.readouterr().out


def test_cli_config_error_when_holdout_unset(tmp_path, no_holdout_config):
    rc = cli.main(["run", "--candidates", str(CANDIDATES_FILE), "--config", str(no_holdout_config),
                   "--playbook", str(PLAYBOOK), "--fixtures-dir", str(BRAIN_DIR),
                   "--data-dir", str(tmp_path / "d")])
    assert rc == cli.EXIT_CONFIG


def test_cli_schema_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc = cli.main(["run", "--candidates", str(bad), "--config", str(TEST_CONFIG),
                   "--playbook", str(PLAYBOOK), "--data-dir", str(tmp_path / "d")])
    assert rc == cli.EXIT_SCHEMA


def test_cli_real_client_without_credentials_exits_7(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(brain_api.EMAIL_ENV_VAR, raising=False)
    monkeypatch.delenv(brain_api.SECRET_ENV_VAR, raising=False)
    rc = cli.main(["run", "--candidates", str(CANDIDATES_FILE), "--config", str(TEST_CONFIG),
                   "--playbook", str(PLAYBOOK), "--client", "real",
                   "--brain-config", str(ROOT / "config" / "brain.yaml"),
                   "--data-dir", str(tmp_path / "d")])
    assert rc == cli.EXIT_BRAIN
    assert "BRAIN_EMAIL" in capsys.readouterr().err


def test_cli_real_client_checks_holdout_before_login(tmp_path, monkeypatch, no_holdout_config):
    monkeypatch.setenv(brain_api.EMAIL_ENV_VAR, "someone@example.com")
    monkeypatch.setenv(brain_api.SECRET_ENV_VAR, "not-used")
    rc = cli.main(["run", "--candidates", str(CANDIDATES_FILE), "--config", str(no_holdout_config),
                   "--playbook", str(PLAYBOOK), "--client", "real",
                   "--brain-config", str(ROOT / "config" / "brain.yaml"),
                   "--data-dir", str(tmp_path / "d")])
    assert rc == cli.EXIT_CONFIG


def test_bandit_driver_floor_holds(capsys):
    rc = cli.main(["bandit", "--candidates", str(CANDIDATES_FILE), "--rounds", "10"])
    out = capsys.readouterr().out
    assert rc == 0
    rows = [l for l in out.splitlines() if l.strip()[:1].isdigit()]
    assert len(rows) == 10 and all(r.endswith("True") for r in rows)
    floors = {int(r.split("|")[-2]) for r in rows}
    assert floors == {2}                    # floor(0.2 * 50 / 4 arms)
    assert "XYZ" not in out


def test_robust_universe_checks_map_to_sub_universe():
    from alpha_research.evaluator import check_class
    for name in ("LOW_ROBUST_UNIVERSE_SHARPE", "LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO",
                 "LOW_ROBUST_UNIVERSE_RETURNS", "LOW_INVESTABILITY_CONSTRAINED_SHARPE"):
        assert check_class("brain:" + name) == "sub_universe"
    assert check_class("brain:LOW_ASI_JPN_SHARPE") == "other"    # no lever fits it yet


def test_a_rejected_alpha_a_lever_can_now_fix_is_re_decided(memory, config, playbook, candidate):
    # 2026-11-23: an IND alpha (Sharpe 3.14, fitness 2.20) was rejected as "other" for a
    # robust-universe Sharpe of 0.95 / 1.0; with the mapping it becomes repair / sub_universe
    from alpha_research.pipeline import Pipeline, newly_repairable
    from alpha_research.schemas import DecisionRecord, GateResult, ResultRecord
    c = candidate(candidate_id="ind_1")
    r = ResultRecord("ind_1", "complete", sharpe=3.14, fitness=2.2, turnover=0.35,
                     sub_universe_sharpe=1.25, sub_universe_cutoff=0.93,
                     checks={"LOW_ROBUST_UNIVERSE_SHARPE": False, "LOW_SHARPE": True},
                     check_values={"LOW_ROBUST_UNIVERSE_SHARPE": {"value": 0.95, "limit": 1.0}})
    old = DecisionRecord("ind_1", "reject", "other", gate_results=[
        GateResult("brain:LOW_ROBUST_UNIVERSE_SHARPE", False, 0.95, 1.0, "BRAIN check FAIL")],
        reasons=["no playbook entry for other"])
    memory.put_candidate(c)
    memory.put_result(r)
    memory.put_decision(old)
    assert newly_repairable(old, r, playbook)
    outs = Pipeline(config, playbook, None, memory).reevaluate_rejected()
    assert [(o.candidate_id, o.decision, o.failure_class) for o in outs] == \
        [("ind_1", "repair", "sub_universe")]
    assert memory.decisions_for("ind_1")[-1].decision == "repair"
    # decided once: a second pass finds nothing (its latest decision is no longer a reject)
    assert Pipeline(config, playbook, None, memory).reevaluate_rejected() == []
