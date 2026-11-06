"""Research loop (spec section 13.2): experiment ids, duplicates, lineage, queue, routing.

Expression normalization before hashing: remove all whitespace and lowercase operator
names only (an identifier directly followed by ``(``). Datafield names and argument order
are kept as written; nothing is sorted. See ``schemas.normalize_expression``.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from enum import Enum

from .memory import MemoryStore
from .schemas import CandidateRecord, DecisionRecord, experiment_hash


class Action(str, Enum):
    POOL = "pool"            # accept
    REPAIR = "repair"        # hand to repair.py
    MEMORY = "memory"        # reject: memory only
    REQUEUE = "requeue"      # pending: back to the queue


_ROUTES = {
    "accept": Action.POOL,
    "repair": Action.REPAIR,
    "reject": Action.MEMORY,
    "pending": Action.REQUEUE,
}


def experiment_id(candidate: CandidateRecord) -> str:
    return experiment_hash(candidate)


def assign_lineage(candidate: CandidateRecord, parent: CandidateRecord,
                   lever: str | None = None) -> CandidateRecord:
    """Return a copy of ``candidate`` with parent_id set and generation = parent + 1."""
    return dataclasses.replace(candidate, parent_id=parent.candidate_id,
                               generation=parent.generation + 1,
                               lever=lever if lever is not None else candidate.lever)


def decide_next(decision: DecisionRecord) -> Action:
    try:
        return _ROUTES[decision.decision]
    except KeyError:
        raise ValueError(f"unknown decision {decision.decision!r}") from None


class ResearchLoop:
    def __init__(self, memory: MemoryStore):
        self.memory = memory
        self.queue: deque[CandidateRecord] = deque()

    def is_duplicate(self, candidate: CandidateRecord) -> bool:
        return self.memory.has_result_for_hash(experiment_id(candidate))

    def enqueue(self, candidate: CandidateRecord) -> None:
        self.queue.append(candidate)

    def drain(self, max_items: int) -> list[CandidateRecord]:
        out = []
        while self.queue and len(out) < max_items:
            out.append(self.queue.popleft())
        return out

    def route(self, candidate: CandidateRecord, decision: DecisionRecord) -> Action:
        action = decide_next(decision)
        if action is Action.REQUEUE:
            self.enqueue(candidate)
        return action


def lineage_root(candidate_id: str, cands: dict) -> str:
    """The original candidate of a repair lineage (follow ``parent_id`` up). ``cands`` maps
    candidate id -> CandidateRecord."""
    seen, cur = set(), candidate_id
    while cur in cands and cands[cur].parent_id and cur not in seen:
        seen.add(cur)
        cur = cands[cur].parent_id
    return cur


def lineage_members(outcomes: list[dict], root: str) -> list[dict]:
    """Every outcome row in the lineage of ``root`` (root included), oldest first."""
    cands = {r["candidate"].candidate_id: r["candidate"] for r in outcomes}
    return [r for r in outcomes if lineage_root(r["candidate"].candidate_id, cands) == root]


def simulated(row: dict) -> bool:
    """The row reached BRAIN (a result that is not ``invalid``)."""
    return row["result"] is not None and row["result"].status != "invalid"

