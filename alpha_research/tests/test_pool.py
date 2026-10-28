import pytest

from alpha_research.pool import Pool
from alpha_research.schemas import DecisionRecord, ResultRecord
from conftest import gaussian_series


def admit(pool, candidate, cid, seed, decision="accept", diversity=True, family="pv"):
    c = candidate(candidate_id=cid, family=family, expression=f"rank(x{seed})")
    r = ResultRecord(cid, "complete", sharpe=1.5, daily_pnl=gaussian_series(300, seed=seed))
    return pool.admit(c, r, DecisionRecord(cid, decision, "none"), diversity)


def test_admit_only_accepts_with_diversity_pass(memory, candidate):
    pool = Pool(memory)
    assert admit(pool, candidate, "a", 1) is True
    assert admit(pool, candidate, "b", 2, decision="reject") is False
    assert admit(pool, candidate, "c", 3, diversity=False) is False
    assert admit(pool, candidate, "d", 4, diversity=None) is False
    assert pool.count() == 1


def test_matrix_and_families(memory, candidate):
    pool = Pool(memory)
    admit(pool, candidate, "a", 1, family="pv")
    admit(pool, candidate, "b", 2, family="pv")
    admit(pool, candidate, "c", 3, family="fund")
    m = pool.matrix()
    assert list(m.columns) == ["a", "b", "c"] and len(m) == 300
    assert pool.families() == {"fund": 1, "pv": 2}
    assert (memory.data_dir / "pnl" / "a.csv").exists()


def test_empty_pool_matrix(memory):
    assert Pool(memory).matrix().empty
