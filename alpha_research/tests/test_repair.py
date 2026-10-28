import pytest

from alpha_research.config import ConfigError
from alpha_research.repair import Playbook, next_lever


def test_turnover_first_lever(playbook):
    assert next_lever(playbook, "turnover", []).name == "increase_decay"


def test_turnover_second_lever(playbook):
    lever = next_lever(playbook, "turnover", ["increase_decay"])
    assert lever.name == "lengthen_ts_window" and lever.step == {"factor": 2}


def test_exhausted(playbook):
    tried = ["increase_decay", "lengthen_ts_window", "wrap_in_smoothing", "hump"]
    assert next_lever(playbook, "turnover", tried) is None


def test_unknown_class(playbook):
    assert next_lever(playbook, "sharpe", []) is None
    assert next_lever(playbook, "made_up", []) is None


def test_returns_single_lever(playbook):
    from alpha_research.repair import Lever
    assert isinstance(next_lever(playbook, "concentration", []), Lever)


def test_bad_playbook():
    with pytest.raises(ConfigError):
        Playbook.from_dict({"turnover": [{"step": {}}]})
    with pytest.raises(ConfigError):
        Playbook.from_dict({"turnover": []})
