import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from alpha_research.config import config_from_dict
from alpha_research.memory import MemoryStore
from alpha_research.repair import Playbook
from alpha_research.schemas import CandidateRecord

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
BRAIN_DIR = FIXTURES / "brain_results"
CANDIDATES_FILE = FIXTURES / "candidates" / "hand_written.json"
TEST_CONFIG = FIXTURES / "thresholds_test.yaml"
MAIN_CONFIG = ROOT / "config" / "thresholds.yaml"
PLAYBOOK = ROOT / "config" / "repair_playbook.yaml"


@pytest.fixture
def no_holdout_config(tmp_path):
    """The main config with the holdout cleared, for the 'holdout must be set' paths."""
    raw = yaml.safe_load(MAIN_CONFIG.read_text())
    raw["holdout"] = {"start_date": None, "end_date": None}
    path = tmp_path / "thresholds_no_holdout.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def raw_config():
    return copy.deepcopy(yaml.safe_load(TEST_CONFIG.read_text()))


@pytest.fixture
def config(raw_config):
    return config_from_dict(raw_config)


@pytest.fixture
def make_config(raw_config):
    """Build a config from the test config with dotted-key overrides."""
    def _make(**overrides):
        raw = copy.deepcopy(raw_config)
        for dotted, value in overrides.items():
            section, key = dotted.split("__")
            raw[section][key] = value
        return config_from_dict(raw)
    return _make


@pytest.fixture
def playbook():
    return Playbook.load(PLAYBOOK)


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore(tmp_path / "data")
    yield store
    store.close()


@pytest.fixture
def candidate():
    def _make(**kw):
        base = dict(candidate_id="c1", expression="rank(-ts_delta(close, 5))", region="USA",
                    universe="TOP3000", delay=1, decay=4, neutralization="SUBINDUSTRY",
                    truncation=0.08, start_date="2019-01-01", end_date="2023-12-31",
                    family="price_volume", dataset="pv1")
        base.update(kw)
        return CandidateRecord.from_dict(base)
    return _make


def gaussian_series(n=500, mean=0.0, std=1.0, seed=0, start="2019-01-01"):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n), index=pd.bdate_range(start, periods=n))
