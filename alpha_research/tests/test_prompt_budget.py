"""Prompt budget, operator spotlight and lesson fitting (owner 2026-11-26): the instructions
go with every model call, so they stay small whatever the catalog size, and show operators
our alphas never used."""

import dataclasses

from alpha_research.generator import (MIN_PROMPT_FIELDS, AlphaGenerator, GenerationSettings,
                                      Source, fit_lessons, lessons_for,
                                      load_generator_config, spotlight_ops)
from alpha_research.fastexpr import FastExprSpec
from alpha_research.llm import ScriptedLLMClient

from conftest import ROOT

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")
GEN = load_generator_config(ROOT / "config" / "generator.yaml")
SETTINGS = GenerationSettings.from_mapping({
    "region": "USA", "universe": "TOP3000", "delay": 1, "decay": 4,
    "neutralization": "INDUSTRY", "truncation": 0.08, "start_date": "2014-01-01",
    "end_date": "2023-12-31", "dataset": "ds0"})


def big_catalog(n=400):
    meta = {}
    for i in range(n):
        meta[f"ds{i % 8}_field_{i}"] = {"dataset": f"ds{i % 8}", "category": "model",
                                        "coverage": 0.9, "alpha_count": i % 37,
                                        "type": "MATRIX",
                                        "description": f"model score number {i}"}
    return meta


class Recorder(ScriptedLLMClient):
    def __init__(self):
        super().__init__(['{"alphas": []}'] * 10)
        self.systems = []

    def complete(self, system, user, *, max_output=4096):
        self.systems.append((system, user))
        return super().complete(system, user, max_output=max_output)


def gen(config, meta):
    llm = Recorder()
    g = AlphaGenerator(llm, SPEC, SETTINGS, config, datafields=list(meta), field_meta=meta)
    return g, llm


def test_prompt_fits_the_budget_by_listing_fewer_fields():
    meta = big_catalog()
    g, llm = gen(dataclasses.replace(GEN, max_prompt_chars=9000, min_signal_sources=None), meta)
    try:
        g.generate(Source("hypothesis", "model scores that rise"), 3, batch_id="b")
    except Exception:
        pass
    system, user = llm.systems[0]
    assert len(system) + len(user) <= 9000
    # the cut never goes below the floor
    g2, llm2 = gen(dataclasses.replace(GEN, max_prompt_chars=500, min_signal_sources=None), meta)
    try:
        g2.generate(Source("hypothesis", "model scores that rise"), 3, batch_id="b")
    except Exception:
        pass
    listed = sum(1 for f in meta if f in llm2.systems[0][0])
    assert listed >= MIN_PROMPT_FIELDS


def test_catalog_size_does_not_grow_the_prompt():
    sizes = []
    for n in (200, 2000):
        g, llm = gen(dataclasses.replace(GEN, min_signal_sources=None), big_catalog(n))
        try:
            g.generate(Source("hypothesis", "model scores that rise"), 3, batch_id="b")
        except Exception:
            pass
        sizes.append(len(llm.systems[0][0]))
    assert abs(sizes[1] - sizes[0]) < 0.15 * sizes[0]


def test_spotlight_follows_failures_then_rotates_unused_operators():
    spot = spotlight_ops(GEN, SPEC, ["turnover"], ["ts_regression", "bucket", "sqrt"], 0)
    names = [op for op, _ in spot]
    assert names[0] == "trade_when" and len(names) <= GEN.spotlight_count
    assert any(op in names for op in ("ts_regression", "bucket", "sqrt"))
    assert "turnover" in dict(spot)["trade_when"].lower()        # BRAIN's description
    later = [op for op, _ in spotlight_ops(GEN, SPEC, [], ["ts_regression", "bucket", "sqrt"], 1)]
    assert later[0] == "bucket"                                   # rotated
    # an operator the account lacks is never shown
    cfg = dataclasses.replace(GEN, spotlight_rules=(("turnover", ("no_such_op",)),))
    assert "no_such_op" not in [op for op, _ in spotlight_ops(cfg, SPEC, ["turnover"], [], 0)]


def test_spotlight_reaches_the_prompt():
    meta = big_catalog(50)
    g, llm = gen(dataclasses.replace(GEN, min_signal_sources=None), meta)
    g.failures, g.unused_ops = ["turnover"], ["bucket"]
    try:
        g.generate(Source("hypothesis", "model scores that rise"), 3, batch_id="b")
    except Exception:
        pass
    system = llm.systems[0][0]
    assert "Operators worth considering" in system and "trade_when(x, y, z):" in system


def test_lessons_keep_the_specific_ones_within_the_budget():
    cfg = dataclasses.replace(GEN, category_lessons=(("all", "A" * 900), ("model", "M" * 500)),
                              region_lessons=(("USA", "U" * 500),))
    order = lessons_for(cfg, "model", "USA", specific_first=True)
    assert [t[0] for t in order] == ["M", "U", "A"]
    assert [t[0] for t in fit_lessons(order, 1100)] == ["M", "U"]
    assert fit_lessons(["X" * 5000], 100) == ["X" * 5000]        # at least the first
