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
    assert count("reject", "sharpe") == 5
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
    assert lines[0].startswith("acc_01 | accept | none | 1.8500 | ")

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
    assert "div_01 | reject | self_correlation" in capsys.readouterr().out


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
