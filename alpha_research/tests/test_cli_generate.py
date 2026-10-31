import json

import pytest

from alpha_research.cli import main
from alpha_research.llm import KEY_ENV_VAR
from alpha_research.pipeline import load_candidate_file

from conftest import ROOT

SETTINGS = ROOT / "config" / "generation_settings.example.yaml"
HYP = "Stocks with rising analyst estimates keep outperforming."


def alphas(*exprs):
    return json.dumps({"alphas": [{"expression": e, "family": "analyst"} for e in exprs]})


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Run `cli generate` from the project root with a scripted model."""
    monkeypatch.chdir(ROOT)

    def _run(replies, *extra, source=("--hypothesis", HYP), settings=True, out=None):
        script = tmp_path / "script.json"
        script.write_text(json.dumps(replies))
        out = out or tmp_path / "out" / "gen.json"
        argv = ["generate", *source, "--out", str(out),
                "--llm", "scripted", "--llm-script", str(script), *extra]
        if settings:
            argv += ["--settings", str(SETTINGS)]
        return main(argv), out

    return _run


def test_hypothesis_end_to_end_writes_loadable_candidates_and_report(run, capsys):
    code, out = run([alphas("rank(-ts_delta(close, 5))", "rank(ts_zscore(volume, 60))")],
                    "--n", "2")
    assert code == 0
    entries = load_candidate_file(out)
    assert [e.candidate.expression for e in entries] == [
        "rank(-ts_delta(close, 5))", "rank(ts_zscore(volume, 60))"]
    c = entries[0].candidate
    c.validate()
    assert (c.region, c.universe, c.neutralization) == ("USA", "TOP3000", "SUBINDUSTRY")
    assert c.model == "scripted" and c.family == "analyst"
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert len(report["candidates"]) == 2 and report["llm_calls"] == 1
    stdout = capsys.readouterr().out
    assert "rank(-ts_delta(close, 5))" in stdout and "2 candidates" in stdout


def test_flags_override_the_settings_file(run):
    code, out = run([alphas("rank(close)")], "--n", "1", "--region", "EUR", "--decay", "9",
                    "--dataset", "pv9")
    assert code == 0
    c = load_candidate_file(out)[0].candidate
    assert (c.region, c.decay, c.dataset) == ("EUR", 9, "pv9")


def test_settings_can_come_from_flags_alone(run):
    flags = ["--region", "USA", "--universe", "TOP3000", "--delay", "1", "--decay", "2",
             "--neutralization", "MARKET", "--truncation", "0.1",
             "--start-date", "2019-01-01", "--end-date", "2023-12-31"]
    code, out = run([alphas("rank(close)")], "--n", "1", *flags, settings=False)
    assert code == 0 and load_candidate_file(out)[0].candidate.neutralization == "MARKET"


def test_hypothesis_file_and_paper_inputs(run, tmp_path):
    hyp_file = tmp_path / "h.txt"
    hyp_file.write_text(HYP)
    code, _ = run([alphas("rank(close)")], "--n", "1", source=("--hypothesis-file", str(hyp_file)))
    assert code == 0

    paper = tmp_path / "paper.md"
    paper.write_text("# Estimate revisions\nAnalysts revise slowly. " * 10)
    replies = [json.dumps({"hypotheses": [{"hypothesis": "Revisions drift."}]}),
               alphas("rank(ts_mean(returns, 20))")]
    code, out = run(replies, "--n", "1", "--paper-title", "Revisions",
                    source=("--paper", str(paper)), out=tmp_path / "p.json")
    assert code == 0
    assert load_candidate_file(out)[0].candidate.hypothesis == "Revisions drift."


def test_datafield_catalog_filters_unknown_fields(run, tmp_path):
    catalog = tmp_path / "fields.txt"
    catalog.write_text("book_value\n")
    code, out = run([alphas("rank(book_value)", "rank(imaginary_field)"), alphas("rank(nope)")],
                    "--n", "2", "--datafields", str(catalog))
    assert code == 0
    assert [e.candidate.expression for e in load_candidate_file(out)] == ["rank(book_value)"]
    rejected = json.loads(out.with_suffix(".report.json").read_text())["rejected"]
    # the first attempt was sent back for repair, so the report lists the final failure
    assert [r["expression"] for r in rejected] == ["rank(nope)"]
    assert "unknown datafield 'nope'" in rejected[0]["errors"][0]


def test_no_valid_candidate_exits_6_but_still_writes_files(run, capsys):
    code, out = run([alphas("rank(nope(close))"), alphas("rank(nope(close))")], "--n", "1")
    assert code == 6
    assert json.loads(out.read_text()) == []
    assert "rejected |" in capsys.readouterr().out


@pytest.mark.parametrize("make_args, use_settings_file, fragment", [
    (lambda tmp: ["--settings", str(tmp / "missing.yaml")], False, "cannot read settings file"),
    (lambda tmp: ["--datafields", str(tmp / "missing.json")], True, "cannot read datafield catalog"),
])
def test_config_problems_exit_2(run, tmp_path, capsys, make_args, use_settings_file, fragment):
    code, _ = run([alphas("rank(close)")], *make_args(tmp_path), settings=use_settings_file)
    assert code == 2 and fragment in capsys.readouterr().err


def test_missing_settings_name_the_keys(run, capsys):
    code, _ = run([alphas("rank(close)")], settings=False)
    assert code == 2
    err = capsys.readouterr().err
    assert "missing generation settings" in err and "region" in err


def test_pdf_papers_are_refused_with_a_hint(run, tmp_path, capsys):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    code, _ = run([], source=("--paper", str(pdf)))
    assert code == 2 and "pdftotext" in capsys.readouterr().err


def test_empty_hypothesis_exits_2(run, capsys):
    code, _ = run([], source=("--hypothesis", "   "))
    assert code == 2 and "empty" in capsys.readouterr().err


def test_a_source_flag_is_required():
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--out", "x.json"])
    assert exc.value.code == 2


def test_scripted_model_needs_a_valid_script(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    base = ["generate", "--hypothesis", HYP, "--out", str(tmp_path / "o.json"),
            "--settings", str(SETTINGS), "--llm", "scripted"]
    assert main(base) == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a list"}')
    assert main(base + ["--llm-script", str(bad)]) == 2
    bad.write_text("not json")
    assert main(base + ["--llm-script", str(bad)]) == 2


def test_real_model_without_a_key_exits_5(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv(KEY_ENV_VAR, raising=False)
    code = main(["generate", "--hypothesis", HYP, "--out", str(tmp_path / "o.json"),
                 "--settings", str(SETTINGS)])
    assert code == 5 and "model error" in capsys.readouterr().err


def test_unparseable_model_replies_exit_5(run, capsys):
    code, _ = run(["not json", "still not json"], "--n", "1")
    assert code == 5 and "valid JSON" in capsys.readouterr().err


def test_generated_candidates_cannot_run_against_fixtures_and_say_why(run, tmp_path, capsys):
    code, out = run([alphas("rank(close)")], "--n", "1")
    assert code == 0
    capsys.readouterr()
    code = main(["run", "--candidates", str(out), "--config", "tests/fixtures/thresholds_test.yaml",
                 "--data-dir", str(tmp_path / "data")])
    assert code == 4
    assert "real client" in capsys.readouterr().err
