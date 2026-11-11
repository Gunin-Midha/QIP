"""pick_datasets in scripts/brain_discovery.py: which datasets get their fields downloaded."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "brain_discovery", Path(__file__).parents[1] / "scripts" / "brain_discovery.py")
discovery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(discovery)


def ds(id_, cat, users, alphas, coverage=0.9, pyramid=1.0):
    return {"id": id_, "category": {"id": cat}, "userCount": users, "alphaCount": alphas,
            "coverage": coverage, "pyramidMultiplier": pyramid}


DATASETS = [
    ds("pv1", "pv", 1000, 9000), ds("pv13", "pv", 500, 4000),
    ds("fundamental6", "fundamental", 900, 8000), ds("fundamental2", "fundamental", 400, 3000),
    ds("analyst4", "analyst", 800, 7000), ds("news12", "news", 300, 2000),
    ds("rare_a", "other", 2, 5, pyramid=1.5), ds("rare_b", "model", 3, 5, pyramid=2.0),
    ds("thin", "other", 1, 1, coverage=0.1),
]


def ids(chosen):
    return [d["id"] for d in chosen]


def test_zero_or_large_n_keeps_every_dataset():
    assert ids(discovery.pick_datasets(DATASETS, 0)) == ids(DATASETS)
    assert ids(discovery.pick_datasets(DATASETS, 50)) == ids(DATASETS)


def test_top_dataset_per_category_then_least_explored():
    chosen = ids(discovery.pick_datasets(DATASETS, 6, explore_share=0.34))
    assert len(chosen) == 6
    assert chosen[:4] == ["pv1", "fundamental6", "analyst4", "news12"]
    # explore slots: lowest alphaCount with enough coverage, higher pyramid first on ties
    assert chosen[4:] == ["rare_b", "rare_a"]
    assert "thin" not in chosen


def test_fills_by_use_when_too_few_well_covered():
    few = [d for d in DATASETS if d["coverage"] < 0.5] + DATASETS[:3]
    chosen = ids(discovery.pick_datasets(few, 3, explore_share=0.5))
    assert len(chosen) == 3 and len(set(chosen)) == 3
