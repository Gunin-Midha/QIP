import pytest

from alpha_research.orchestrator import Orchestrator, run_rounds

ARMS = [("pv1", "USA", "pv0"), ("fundamental6", "USA", "pv0"), ("analyst4", "USA", "pv0"),
        ("news12", "EUR", "pv0"), ("model77", "USA", "pv0")]


def trained(seed=0):
    o = Orchestrator(ARMS, seed=seed)
    for _ in range(40):
        o.update(ARMS[0], True)
    for arm in ARMS[1:]:
        for _ in range(40):
            o.update(arm, False)
        o.update(arm, True)
    return o


def test_best_arm_gets_most_slots():
    alloc = trained().allocate(100, floor_share=0.2)
    assert max(alloc, key=alloc.get) == ARMS[0]
    assert alloc[ARMS[0]] > 70


def test_floor_holds_when_one_arm_dominates():
    alloc = trained().allocate(50, floor_share=0.2)
    floor = 50 * 0.2 // 5
    assert floor == 2
    assert all(n >= floor for n in alloc.values())


@pytest.mark.parametrize("budget", [0, 1, 7, 23, 100])
def test_allocation_sums_to_budget(budget):
    assert sum(trained().allocate(budget).values()) == budget


def test_reproducible_with_seed():
    assert trained(seed=3).allocate(40) == trained(seed=3).allocate(40)


def test_update_unknown_arm():
    with pytest.raises(KeyError):
        Orchestrator(ARMS).update(("x", "y", "z"), True)


def test_run_rounds_floor_every_round():
    outcomes = {ARMS[0]: [True, True, False], ARMS[1]: [False], ARMS[2]: [False, True],
                ARMS[3]: [False, False, False], ARMS[4]: []}
    hist = run_rounds(outcomes, rounds=10, budget=25, floor_share=0.2, seed=0)
    assert len(hist) == 10
    assert all(h["floor_held"] and h["total"] == 25 for h in hist)
    assert hist[-1]["allocation"][ARMS[0]] >= hist[-1]["allocation"][ARMS[3]]


def test_add_arm_starts_from_the_prior():
    o = Orchestrator(["a"], seed=0)
    o.update("a", True)
    assert o.add_arm("b") is True and o.add_arm("b") is False
    assert o.arms == ["a", "b"] and o.pass_rate("b") == 0.5
    assert sum(o.allocate(10).values()) == 10


def test_allocate_only_to_live_arms():
    o = Orchestrator(["a", "b", "c"], seed=0)
    alloc = o.allocate(6, floor_share=0.5, only=["a", "c"])
    assert alloc["b"] == 0 and alloc["a"] + alloc["c"] == 6 and set(alloc) == {"a", "b", "c"}
