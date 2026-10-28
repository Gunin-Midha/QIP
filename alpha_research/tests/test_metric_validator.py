import pytest

from alpha_research.metric_validator import (
    check_yearly_consistency, expected_fitness, margin_check, naive_margin_bps, validate_result,
)
from alpha_research.schemas import ResultRecord, YearRow


def test_screenshot_aggregate_fitness():
    assert expected_fitness(1.68, 0.1119, 0.2578) == pytest.approx(1.11, abs=0.01)


def test_screenshot_2022_fitness():
    assert expected_fitness(3.59, 0.2851, 0.2564) == pytest.approx(3.79, abs=0.01)


@pytest.mark.parametrize("year,s,r,t,f", [
    (2019, 1.40, 0.0621, 0.2305, 0.73), (2020, 0.11, 0.0080, 0.2699, 0.02),
    (2021, 1.83, 0.1361, 0.2772, 1.28), (2023, 1.01, 0.0595, 0.2559, 0.49),
])
def test_screenshot_other_years(year, s, r, t, f):
    assert expected_fitness(s, r, t) == pytest.approx(f, abs=0.01)


def test_turnover_floor_applies():
    assert expected_fitness(1.0, 0.125, 0.01) == pytest.approx(1.0)


def test_none_inputs_skip():
    assert expected_fitness(None, 0.1, 0.2) is None
    r = ResultRecord("c", "complete", sharpe=1.0, returns=0.1, turnover=0.2, fitness=None)
    assert not any(n.startswith("fitness") for n in validate_result(r))


def test_malformed_input_raises():
    with pytest.raises(TypeError):
        expected_fitness("1.0", 0.1, 0.2)


def test_fitness_mismatch_is_a_note_not_an_error():
    r = ResultRecord("c", "complete", sharpe=1.68, returns=0.1119, turnover=0.2578, fitness=2.0)
    assert any(n.startswith("fitness_mismatch") for n in validate_result(r))


def test_yearly_consistency():
    rows = [YearRow(2020, sharpe=0.5), YearRow(2021, sharpe=2.0)]
    assert check_yearly_consistency(1.0, rows)[0].startswith("yearly_consistent")
    assert check_yearly_consistency(3.0, rows)[0].startswith("yearly_inconsistent")
    assert check_yearly_consistency(1.0, []) == []


def test_margin_mismatch_note_on_screenshot_values():
    r = ResultRecord("c", "complete", returns=0.1119, turnover=0.2578, margin_bps=8.68)
    naive = naive_margin_bps(0.1119, 0.2578)
    assert naive == pytest.approx(17.22, abs=0.01)
    notes = margin_check(r)
    assert any("margin_recorded" in n and "8.68" in n for n in notes)
    assert any(n.startswith("margin_mismatch") for n in notes)


def test_margin_within_band_no_mismatch():
    naive = naive_margin_bps(0.1, 0.2)
    r = ResultRecord("c", "complete", returns=0.1, turnover=0.2, margin_bps=naive)
    assert not any(n.startswith("margin_mismatch") for n in margin_check(r))
