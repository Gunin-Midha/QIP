import pytest

from alpha_research.config import ConfigError, config_from_dict, load_config
from conftest import MAIN_CONFIG, TEST_CONFIG


def test_main_config_loads_with_nulls():
    cfg = load_config(MAIN_CONFIG)
    assert cfg.gates.sharpe_min == 1.58  # consultant LOW_SHARPE limit (2026-11-10)
    assert cfg.gates.turnover_max == 0.70
    assert cfg.risk.max_drawdown is None
    assert cfg.diversity.min_marginal_pool_sharpe_gain is None
    assert cfg.simulator.max_concurrent == 8   # owner 2026-11-14


def test_main_config_holdout_follows_brain_default():
    # BRAIN's in-sample on our account ends 2023-12-31; the holdout starts the day after
    h = load_config(MAIN_CONFIG).require_holdout()
    assert str(h.start_date) == "2024-01-01" and h.end_date > h.start_date


def test_config_without_holdout_refuses_simulation(no_holdout_config):
    with pytest.raises(ConfigError):
        load_config(no_holdout_config).require_holdout()


def test_test_config_has_holdout():
    assert load_config(TEST_CONFIG).holdout.configured


@pytest.mark.parametrize("section", ["gates", "risk", "diversity", "auditor", "holdout", "simulator"])
def test_missing_section(raw_config, section):
    del raw_config[section]
    with pytest.raises(ConfigError):
        config_from_dict(raw_config)


def test_missing_key(raw_config):
    del raw_config["gates"]["sharpe_min"]
    with pytest.raises(ConfigError):
        config_from_dict(raw_config)


@pytest.mark.parametrize("section,key,value", [
    ("gates", "sharpe_min", "high"),
    ("gates", "turnover_max", 1.5),
    ("risk", "leg_balance_min_ratio", 2.0),
    ("simulator", "max_retries", -1),
    ("simulator", "max_poll_seconds", None),
    ("auditor", "deflation_method", "something_else"),
    ("holdout", "end_date", None),         # only one side set
])
def test_bad_values(raw_config, section, key, value):
    raw_config[section][key] = value
    with pytest.raises(ConfigError):
        config_from_dict(raw_config)


def test_turnover_bounds_ordered(raw_config):
    raw_config["gates"]["turnover_min"] = 0.8
    with pytest.raises(ConfigError):
        config_from_dict(raw_config)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")
