"""Research Director: Thompson sampling with floor quotas (spec section 13.1).

Arms are hashable tuples: (dataset, region, prompt_version) in the Phase 2 driver, and
(dataset, region, prompt_version, generator model, delay) in the research loop. Each arm has a Beta(1, 1) prior on its
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
4. Regions (optional; arms carry their region at index 1). ``region_floor_share``: after
   the arm floors, each region is topped up to ``floor(region_floor_share * budget /
   n_regions)`` slots, each slot going to the region's arm with the highest posterior draw,
   so a region with no accepts yet keeps being explored. ``max_region_share``: no region gets
   more than ceil(max_region_share * budget) slots while another region is below it. What
   works in one region is learned per arm, so it never decides another region's slots.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np

Arm = tuple  # (dataset, region, prompt_version[, generator model, delay])


@dataclass
class ArmState:
    successes: int = 0
    failures: int = 0


def _region(arm) -> object:
    """The region of an arm tuple (index 1), or the arm itself for other arm types."""
    return arm[1] if isinstance(arm, tuple) and len(arm) > 1 else arm


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
                 max_share: float | None = None, only=None, *,
                 region_floor_share: float | None = None,
                 max_region_share: float | None = None) -> dict:
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
        regions = [_region(a) for a in arms]
        region_cap = math.ceil(max_region_share * budget) if max_region_share is not None else None

        def give(allowed: np.ndarray) -> None:
            samples = np.where(allowed, self.rng.beta(alphas, betas), -1.0)
            if cap is not None:
                full = np.array([slots[a] >= cap for a in arms])
                if not (full | ~allowed).all():
                    samples = np.where(full, -1.0, samples)
            if region_cap is not None:
                used = Counter()
                for a, r in zip(arms, regions):
                    used[r] += slots[a]
                full = np.array([used[r] >= region_cap for r in regions])
                if not (full | ~allowed).all():
                    samples = np.where(full, -1.0, samples)
            slots[arms[int(np.argmax(samples))]] += 1

        if region_floor_share:
            names = list(dict.fromkeys(regions))
            need = math.floor(region_floor_share * budget / len(names))
            for name in names:
                mask = np.array([r == name for r in regions])
                while remaining > 0 and sum(slots[a] for a, r in zip(arms, regions)
                                            if r == name) < need:
                    give(mask)
                    remaining -= 1
        everyone = np.ones(len(arms), dtype=bool)
        for _ in range(remaining):
            give(everyone)
        return {a: slots[a] for a in self.arms}

    def add_arm(self, arm: Hashable, like: Hashable | None = None) -> bool:
        """Add a new arm: a fresh Beta(1, 1) prior, or a copy of arm ``like``'s posterior
        (Strategy Evolution's successor of an arm starts from what that arm showed, not
        from scratch). Returns False if the arm already exists."""
        if arm in self.state:
            return False
        self.arms.append(arm)
        old = self.state.get(like) if like is not None else None
        self.state[arm] = ArmState(old.successes, old.failures) if old else ArmState()
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


def balance_slots(slots: int, counts: dict) -> dict:
    """Region rotation: each slot goes to the region with the fewest simulations so far
    (``counts`` plus the slots already given here), ties in ``counts`` order."""
    if slots < 0:
        raise ValueError("slots must be >= 0")
    order = {r: i for i, r in enumerate(counts)}
    given = Counter({r: 0 for r in counts})
    for _ in range(slots):
        given[min(counts, key=lambda r: (counts[r] + given[r], order[r]))] += 1
    return dict(given)


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
