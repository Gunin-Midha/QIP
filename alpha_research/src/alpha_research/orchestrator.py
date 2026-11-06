"""Research Director: Thompson sampling with floor quotas (spec section 13.1).

Arms are hashable tuples: (dataset, region, prompt_version) in the Phase 2 driver, and
(dataset, region, prompt_version, generator model) in the research loop. Each arm has a Beta(1, 1) prior on its
pass rate. Novelty weighting is Phase 3 and not implemented.

Allocation method:
1. Floor: every arm gets ``floor(floor_share * budget / n_arms)`` slots. Rounding down
   guarantees the floors never sum to more than the budget.
2. Remaining slots are handed out one at a time. For each slot, sample every arm's
   posterior Beta(successes + 1, failures + 1) afresh and give the slot to the arm with
   the highest sample (ties go to the earlier arm). Re-sampling per slot spreads slots in
   proportion to each arm's probability of being best.
3. Cap (``max_share``, optional): no arm gets more than ceil(max_share * budget) slots
   while another arm is below the cap, so a few early accepts cannot pull a whole round
   into one dataset or model (owner 2026-11-04: keep the search diverse).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np

Arm = tuple  # (dataset, region, prompt_version[, generator model])


@dataclass
class ArmState:
    successes: int = 0
    failures: int = 0


class Orchestrator:
    def __init__(self, arms: Iterable[Hashable], seed: int | None = None):
        self.arms: list = list(dict.fromkeys(arms))
        if not self.arms:
            raise ValueError("orchestrator needs at least one arm")
        self.state = {a: ArmState() for a in self.arms}
        self.rng = np.random.default_rng(seed)

    def floor_slots(self, budget: int, floor_share: float, n_arms: int | None = None) -> int:
        return math.floor(floor_share * budget / (n_arms or len(self.arms)))

    def allocate(self, budget: int, floor_share: float = 0.2,
                 max_share: float | None = None, only=None) -> dict:
        """Slots per arm. ``only`` limits the allocation to those arms (e.g. arms whose
        model is down get none); the others are listed with 0."""
        if budget < 0:
            raise ValueError("budget must be >= 0")
        if not 0 <= floor_share <= 1:
            raise ValueError("floor_share must be between 0 and 1")
        arms = [a for a in self.arms if only is None or a in set(only)]
        if not arms:
            raise ValueError("no arm to allocate to")
        floor = self.floor_slots(budget, floor_share, len(arms))
        slots = Counter({a: floor for a in arms})
        remaining = budget - floor * len(arms)
        alphas = np.array([self.state[a].successes + 1 for a in arms], dtype=float)
        betas = np.array([self.state[a].failures + 1 for a in arms], dtype=float)
        cap = math.ceil(max_share * budget) if max_share is not None else None
        for _ in range(remaining):
            samples = self.rng.beta(alphas, betas)
            if cap is not None:
                full = np.array([slots[a] >= cap for a in arms])
                if not full.all():
                    samples = np.where(full, -1.0, samples)
            slots[arms[int(np.argmax(samples))]] += 1
        return {a: slots[a] for a in self.arms}

    def add_arm(self, arm: Hashable) -> bool:
        """Add a new arm with a fresh Beta(1, 1) prior (Strategy Evolution adds one per new
        prompt version). Returns False if the arm already exists."""
        if arm in self.state:
            return False
        self.arms.append(arm)
        self.state[arm] = ArmState()
        return True

    def update(self, arm: Hashable, passed: bool) -> None:
        if arm not in self.state:
            raise KeyError(f"unknown arm {arm!r}")
        if passed:
            self.state[arm].successes += 1
        else:
            self.state[arm].failures += 1

    def pass_rate(self, arm: Hashable) -> float:
        s = self.state[arm]
        return (s.successes + 1) / (s.successes + s.failures + 2)


def run_rounds(arm_outcomes: dict, rounds: int, budget: int, floor_share: float = 0.2,
               seed: int = 0) -> list[dict]:
    """Phase 2 driver. ``arm_outcomes`` maps each arm to a list of fixture outcomes (True =
    accepted). Each slot consumes the arm's next outcome, cycling through the list.
    Returns one record per round with the allocation and the floor check."""
    orch = Orchestrator(arm_outcomes.keys(), seed=seed)
    cursor = Counter()
    history = []
    for r in range(1, rounds + 1):
        alloc = orch.allocate(budget, floor_share)
        floor = orch.floor_slots(budget, floor_share)
        for arm, n in alloc.items():
            outcomes = arm_outcomes[arm]
            for _ in range(n):
                if outcomes:
                    orch.update(arm, bool(outcomes[cursor[arm] % len(outcomes)]))
                    cursor[arm] += 1
        history.append({
            "round": r,
            "allocation": alloc,
            "floor": floor,
            "floor_held": all(n >= floor for n in alloc.values()),
            "total": sum(alloc.values()),
        })
    return history
