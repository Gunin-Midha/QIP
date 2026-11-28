"""Investability as its own failure class, and the empty-alpha guard (owner, 2026-11-27).

Investability: BRAIN's LOW_INVESTABILITY_CONSTRAINED_SHARPE (ASI, JPN) had mapped to
sub_universe, whose levers (smaller universe, factor neutralizations, backfill) lowered or
kept the constrained / full Sharpe ratio; in memory the ratio falls with turnover, so the
class gets slowing levers, and its slack is measured on the ratio.

Empty alpha: a result with turnover 0, Sharpe 0 and no position in any year is an
expression bug (an unfilled vec_* part made it NaN everywhere). It is stored as failed (not
evidence about the data) and tried once more with the vector parts filled."""

import json
import re

from alpha_research.critic import nan_guarded
from alpha_research.evaluator import check_class, evaluate
from alpha_research.lever_stats import slack
from alpha_research.research import FILL_LEVER, Budget
from alpha_research.schemas import FAILURE_CLASSES, ResultRecord, YearRow
from alpha_research.simulator import EMPTY_ALPHA, empty_alpha

from test_brain_checks_only import brain_only
from test_pipeline_guards import passing
from test_research import RouterLLM, StubBrain, make_run

INV = "LOW_INVESTABILITY_CONSTRAINED_SHARPE"


# --------------------------------------------------------------------------- investability

def test_investability_check_has_its_own_class_and_slowing_levers(playbook):
    assert "investability" in FAILURE_CLASSES
    assert check_class(f"brain:{INV}") == "investability"
    levers = [lv.name for lv in playbook.entries["investability"]]
    assert levers[0] == "wrap_in_smoothing" and "increase_decay" in levers
    # the levers that lowered the ratio in memory are not offered for it
    assert not any(n == "change_universe" or n.startswith("neutralize_") for n in levers)


def test_failing_investability_alone_is_repaired_as_investability(make_config, playbook):
    r = passing(sharpe=2.57, checks={INV: False},
                check_values={INV: {"value": 1.75, "limit": 1.80}})
    d = evaluate(r, brain_only(make_config), playbook)
    assert (d.decision, d.failure_class) == ("repair", "investability")


def test_investability_slack_is_measured_on_the_ratio(make_config, playbook):
    r = passing(sharpe=3.5, checks={INV: False},
                check_values={INV: {"value": 1.96, "limit": 2.45}})
    d = evaluate(r, brain_only(make_config), playbook)
    assert slack("investability", d, r) == round((1.96 - 2.45) / 3.5, 6)   # -0.14
    # a child that only halves its Sharpe (limit ~0.7 x Sharpe) gains no slack
    child = passing(sharpe=1.75, checks={INV: False},
                    check_values={INV: {"value": 0.98, "limit": 1.225}})
    dc = evaluate(child, brain_only(make_config), playbook)
    assert slack("investability", dc, child) == slack("investability", d, r)


# --------------------------------------------------------------------------- empty alpha

def _empty(**kw):
    base = dict(candidate_id="c1", status="complete", sharpe=0.0, fitness=None, turnover=0.0,
                yearly=[YearRow(year=y, sharpe=0.0, long_count=0, short_count=0)
                        for y in (2014, 2015)])
    base.update(kw)
    return ResultRecord(**base)


def test_empty_alpha_needs_no_positions_in_any_year():
    assert empty_alpha(_empty()).startswith(EMPTY_ALPHA)
    assert empty_alpha(_empty(sharpe=None)) is not None
    held = _empty(yearly=[YearRow(year=2014, sharpe=0.0, long_count=3, short_count=0)])
    assert empty_alpha(held) is None
    assert empty_alpha(_empty(turnover=0.001)) is None
    assert empty_alpha(_empty(turnover=None)) is None     # missing is not zero
    assert empty_alpha(_empty(yearly=[])) is None          # no evidence of no positions


def test_nan_guard_fills_the_vector_part_with_its_middle():
    # the four empty alphas in memory, in shape
    e, filled, _ = nan_guarded("rank(group_zscore(ts_mean(close, 5), sector) "
                               "+ 0.25 * zscore(vec_avg(vfield)))")
    assert filled == ["vfield"]
    assert "0.25 * if_else(is_nan(zscore(vec_avg(vfield))), 0, zscore(vec_avg(vfield)))" in e
    e, filled, _ = nan_guarded("main = rank(close); part = zscore(vec_avg(v)); "
                               "rank(main + 0.3 * part)")
    assert filled == ["v"] and "if_else(is_nan(part), 0, part)" in e
    e, filled, _ = nan_guarded("rank(ts_delta(close, 10)) * rank(ts_backfill(vec_avg(v), 121))")
    assert filled == ["v"] and ", 0.5, rank(ts_backfill(vec_avg(v), 121)))" in e
    e, filled, _ = nan_guarded("dom = vec_avg(a); f = add(vec_avg(b), vec_avg(c)); "
                               "shift = ts_delta(divide(f, add(dom, 1)), 90); "
                               "rank(add(rank(shift), multiply(0.4, rank(returns))))")
    assert sorted(filled) == ["a", "b", "c"]
    assert "add(if_else(is_nan(rank(shift)), 0.5, rank(shift)), multiply(0.4, rank(returns)))" in e
    assert "add(dom, 1)" in e     # vector-only arithmetic is left alone


def test_nan_guard_leaves_what_it_cannot_fill_safely():
    whole = "rank(vec_avg(v))"                       # nothing else for a NaN to blank
    assert nan_guarded(whole) == (whole, [], [])
    done = "rank(close) + if_else(is_nan(rank(vec_avg(v))), 0.5, rank(vec_avg(v)))"
    assert nan_guarded(done)[0] == done               # the author already filled it
    unknown = "rank(close) + vec_avg(v)"              # scale unknown: no neutral value
    assert nan_guarded(unknown) == (unknown, [], ["v"])
    product = "rank(close) * zscore(vec_avg(v))"      # a zero would blank the product too
    assert nan_guarded(product) == (product, [], ["v"])


class VectorLLM(RouterLLM):
    """The generator writes alphas with an unfilled vector part."""

    def complete(self, system, user, *, max_output=4096):
        reply = super().complete(system, user, max_output=max_output)
        if self.calls[-1] != "generator":
            return reply
        alphas = json.loads(reply)["alphas"]
        for a in alphas:
            a["expression"] = f"rank(-ts_delta(close, {self.k})) + 0.3 * zscore(vec_avg(est_vec))"
            self.k += 1
        return json.dumps({"alphas": alphas})


class EmptyVectorBrain(StubBrain):
    """BRAIN holds no positions while a vector part is unfilled."""

    def __init__(self):
        super().__init__()
        self.empty: set[str] = set()

    def submit(self, c):
        sim = super().submit(c)
        if "vec_avg" in c.expression and "is_nan" not in c.expression:
            self.empty.add(sim)
        return sim

    def poll(self, sim_id):
        if sim_id in self.empty:
            return {"status": "complete", "units": "fraction", "sharpe": 0.0, "turnover": 0.0,
                    "yearly": [{"year": 2019, "sharpe": 0.0, "turnover": 0.0,
                                "long_count": 0, "short_count": 0}]}
        return super().poll(sim_id)


def test_research_run_tries_an_empty_alpha_once_with_the_vector_filled(memory, config,
                                                                        playbook):
    brain = EmptyVectorBrain()
    run = make_run(memory, config, playbook, llm=VectorLLM(), brain=brain,
                   budget=Budget(max_simulations=4, per_round=2, max_llm_calls=40))
    report = run.run(1)
    retries = report["empty_retries"]
    assert retries and all(r["status"] == "queued" and r["filled"] == ["est_vec"]
                           for r in retries)
    children = [c for c in brain.submitted if c.lever == FILL_LEVER]
    assert {c.candidate_id for c in children} == {r["child"] for r in retries}
    for c in children:
        assert c.candidate_id.endswith("_n1") and "is_nan" in c.expression
        parent = memory.get_candidate(c.parent_id)
        stored = memory.results_for(parent.candidate_id)[-1]
        assert stored.status == "failed" and any(n.startswith(EMPTY_ALPHA) for n in stored.notes)
        assert memory.results_for(c.candidate_id)[-1].status == "complete"
    # budget: every BRAIN simulation counted, retries included
    assert report["simulations_used"] == len(brain.submitted) <= 4
    # no retry of a retry
    assert not any(re.search(r"_n1_n1$", c.candidate_id) for c in brain.submitted)
