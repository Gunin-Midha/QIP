import json

from alpha_research.fastexpr import FastExprSpec
from alpha_research.llm import ScriptedLLMClient
from alpha_research.prefilter import PreFilter, PrefilterConfig, jaccard, terms
from alpha_research.schemas import ResultRecord

from conftest import ROOT

SPEC = FastExprSpec.load(ROOT / "config" / "fastexpr.yaml")


def pf(memory, **kw):
    llm = kw.pop("llm", None)
    return PreFilter(PrefilterConfig(**kw), SPEC, memory, llm=llm)


def test_terms_and_jaccard():
    a = terms("rank(-ts_delta(close, 5))")
    assert a == {"rank(", "u-", "ts_delta(", "close", "5"}
    assert jaccard(a, terms("rank( - ts_delta(close,5) )")) == 1.0
    assert jaccard(a, terms("rank(-ts_delta(close, 10))")) == 4 / 6
    assert terms("rank(") == frozenset()


def test_invalid_and_duplicate_are_dropped(memory, candidate):
    done = candidate(candidate_id="old")
    memory.put_candidate(done)
    memory.put_result(ResultRecord(candidate_id="old", status="complete"))
    res = pf(memory).run([candidate(candidate_id="bad", expression="rank(nosuch(close))"),
                          candidate(candidate_id="dup"), candidate(candidate_id="ok", expression="rank(volume)")])
    assert [c.candidate_id for c in res.kept] == ["ok"]
    reasons = dict((c.candidate_id, why) for c, why in res.dropped)
    assert reasons["bad"].startswith("invalid expression") and "already simulated" in reasons["dup"]
    assert "near-duplicate check not configured" in res.notes


def test_near_duplicates_against_memory_and_batch(memory, candidate):
    memory.put_candidate(candidate(candidate_id="m1", expression="rank(-ts_delta(close, 5)) * 1"))
    assert jaccard(terms("rank(-ts_delta(close, 5))"), terms("rank(-ts_delta(close, 5)) * 1")) == 5 / 7
    assert jaccard(terms("rank(ts_zscore(volume, 20))"), terms("rank(ts_zscore(volume, 40))")) == 3 / 5
    res = pf(memory, near_duplicate_similarity=0.6).run([
        candidate(candidate_id="a", expression="rank(-ts_delta(close, 5))"),        # 5/7 vs m1
        candidate(candidate_id="b", expression="rank(ts_zscore(volume, 20))"),
        candidate(candidate_id="c", expression="rank(ts_zscore(volume, 40))"),      # 3/5 vs b
        candidate(candidate_id="d", expression="rank(-ts_delta(close, 5))", universe="TOP1000"),
    ])
    assert [c.candidate_id for c in res.kept] == ["b", "d"]
    reasons = dict((c.candidate_id, why) for c, why in res.dropped)
    assert "near duplicate of m1" in reasons["a"] and "near duplicate of b" in reasons["c"]


def test_repairs_are_exempt_from_near_duplicate(memory, candidate):
    memory.put_candidate(candidate(candidate_id="p1", decay=4))
    child = candidate(candidate_id="p1_r1", decay=8, parent_id="p1", lever="increase_decay", generation=1)
    assert pf(memory, near_duplicate_similarity=0.9).run([child]).kept == [child]


def test_model_scores_filter_only_with_a_floor(memory, candidate):
    reply = json.dumps({"scores": [{"candidate_id": "a", "score": 2, "reason": "weak"},
                                   {"candidate_id": "b", "score": 8}]})
    cands = [candidate(candidate_id="a"), candidate(candidate_id="b", expression="rank(volume)")]
    res = pf(memory, use_model=True, llm=ScriptedLLMClient([reply])).run(cands)
    assert len(res.kept) == 2 and res.scores == {"a": 2.0, "b": 8.0} and res.llm_calls == 1
    res = pf(memory, use_model=True, min_model_score=5, llm=ScriptedLLMClient([reply])).run(cands)
    assert [c.candidate_id for c in res.kept] == ["b"] and "below 5" in res.dropped[0][1]


def test_unusable_model_reply_keeps_candidates(memory, candidate):
    res = pf(memory, use_model=True, min_model_score=5,
             llm=ScriptedLLMClient(["nope", "still nope"])).run([candidate()])
    assert len(res.kept) == 1 and "model scoring skipped" in res.notes[-1]
