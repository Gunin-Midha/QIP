import math

import numpy as np
import pytest

from alpha_research.auditor import (
    Auditor, deflated_sharpe, expected_max_normal, os_is_ratio,
)


def test_n1_deflated_equals_raw():
    assert expected_max_normal(1) == 0.0
    assert deflated_sharpe(1.7, 1, 5.0) == 1.7


def test_expected_max_grows_with_n():
    assert 0 < expected_max_normal(10) < expected_max_normal(200) < expected_max_normal(10_000)


def test_best_of_200_zero_edge_is_deflated():
    rng = np.random.default_rng(7)
    years, days = 5, 252 * 5
    sharpes = []
    for _ in range(200):
        r = rng.normal(0, 0.01, days)
        sharpes.append(r.mean() / r.std() * math.sqrt(252))
    best = max(sharpes)
    deflated = deflated_sharpe(best, 200, years)
    assert best - deflated >= 0.5
    assert deflated < 0.5     # no real edge survives


def test_variant_counting(memory, config):
    a = Auditor(memory, config)
    for i in range(3):
        a.register_variant("pv", f"c{i}")
    a.register_variant("fund", "f0")
    assert a.variant_count("pv") == 3 and a.variant_count("fund") == 1
    assert a.deflate(1.5, "pv", 5) < 1.5


def test_os_is_ratio():
    assert os_is_ratio(2.0, 1.0) == 0.5
    assert os_is_ratio(0.0, 1.0) is None
    assert os_is_ratio(-1.0, 1.0) is None
    assert os_is_ratio(1.0, None) is None


def test_family_flags(memory, make_config):
    a = Auditor(memory, make_config(auditor__os_is_ratio_floor=0.7))
    a.record_os_is("g1", "good", 2.0, 1.8)    # 0.9
    a.record_os_is("g2", "good", 1.0, 0.8)    # 0.8
    a.record_os_is("b1", "bad", 2.0, 0.4)     # 0.2
    a.record_os_is("b2", "bad", 1.0, 0.3)     # 0.3
    a.record_os_is("b3", "bad", -1.0, 0.3)    # None, ignored
    assert a.family_ratio("good") == pytest.approx(0.85)
    assert a.family_ratio("bad") == pytest.approx(0.25)
    assert a.flag_family("good") is False
    assert a.flag_family("bad") is True


def test_os_is_upsert(memory, make_config):
    a = Auditor(memory, make_config(auditor__os_is_ratio_floor=0.7))
    a.record_os_is("x", "f", 2.0, 0.2)
    a.record_os_is("x", "f", 2.0, 1.8)
    assert a.family_ratio("f") == pytest.approx(0.9)


def test_flag_not_configured(memory, config):
    a = Auditor(memory, config)
    a.record_os_is("x", "f", 2.0, 0.2)
    assert a.flag_family("f") is None
