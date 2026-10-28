from alpha_research.research_loop import (
    Action, ResearchLoop, assign_lineage, decide_next, experiment_id,
)
from alpha_research.schemas import DecisionRecord, ResultRecord


def test_whitespace_and_operator_case_do_not_change_hash(candidate):
    a = candidate(expression="rank(-ts_delta(close, 5))")
    b = candidate(candidate_id="c2", expression="  RANK( -TS_Delta( close ,5 ) ) ")
    assert experiment_id(a) == experiment_id(b)


def test_datafield_case_and_settings_change_hash(candidate):
    a = candidate()
    assert experiment_id(a) != experiment_id(candidate(expression="rank(-ts_delta(CLOSE, 5))"))
    assert experiment_id(a) != experiment_id(candidate(neutralization="MARKET"))


def test_duplicate_detected_after_result(memory, candidate):
    loop = ResearchLoop(memory)
    a = candidate()
    assert not loop.is_duplicate(a)
    memory.put_candidate(a)
    assert not loop.is_duplicate(a)          # candidate alone is not enough
    memory.put_result(ResultRecord("c1", "complete"))
    assert loop.is_duplicate(candidate(candidate_id="c9", expression=" rank(-ts_delta(close,5))"))


def test_lineage_increments_generation(candidate):
    parent = candidate(candidate_id="p")
    child = assign_lineage(candidate(candidate_id="k", decay=8), parent, lever="increase_decay")
    grandchild = assign_lineage(candidate(candidate_id="g", decay=9), child)
    assert (child.parent_id, child.generation, child.lever) == ("p", 1, "increase_decay")
    assert (grandchild.parent_id, grandchild.generation) == ("k", 2)


def test_decide_next_routes():
    assert decide_next(DecisionRecord("c", "accept", "none")) is Action.POOL
    assert decide_next(DecisionRecord("c", "repair", "turnover")) is Action.REPAIR
    assert decide_next(DecisionRecord("c", "reject", "sharpe")) is Action.MEMORY
    assert decide_next(DecisionRecord("c", "pending", "none")) is Action.REQUEUE


def test_pending_goes_back_to_queue_and_fifo_drain(memory, candidate):
    loop = ResearchLoop(memory)
    for i in range(3):
        loop.enqueue(candidate(candidate_id=f"q{i}"))
    loop.route(candidate(candidate_id="pend"), DecisionRecord("pend", "pending", "none"))
    loop.route(candidate(candidate_id="rej"), DecisionRecord("rej", "reject", "sharpe"))
    assert [c.candidate_id for c in loop.drain(2)] == ["q0", "q1"]
    assert [c.candidate_id for c in loop.drain(10)] == ["q2", "pend"]
    assert loop.drain(5) == []
