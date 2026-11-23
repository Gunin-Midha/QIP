"""Power Pool candidates: recorded, never acted on (owner, 2026-11-15).

BRAIN accepted ``rs20261115045859r01a60_001`` (CHN, BRAIN alpha XgaP3Rpl, Sharpe 1.23,
fitness 0.92) into its Power Pool although our pipeline rejected it: BRAIN marked the
regular-alpha checks it missed (LOW_SHARPE against 2.07, LOW_FITNESS, LOW_2Y_SHARPE,
LOW_RETURNS) WARNING, not FAIL, and every check it ruled on passed, POWER_POOL_CORRELATION
included. Its rules are not otherwise known to us, so this module only tags such alphas
for the owner to submit by hand; decisions, repairs and the Director's rewards are
unchanged.

A result is a Power Pool candidate when
1. it is complete and was not accepted as a regular alpha;
2. BRAIN evaluated POWER_POOL_CORRELATION for it and did not fail it;
3. no other BRAIN check failed, apart from the submission-only ones
   (``gates.submission_checks``: daily quota, theme matches, the Power Pool description,
   which we write at submission);
4. at least one regular-alpha check was WARNING (the note ``brain_api.translate`` writes:
   ``brain_check_not_counted: <CHECK>='WARNING'``): the alpha misses the regular bar only
   softly.
"""

from __future__ import annotations

import re
from typing import Collection

from .schemas import CandidateRecord, DecisionRecord, ResultRecord

PREFIX = "power_pool_candidate"
CHECK = "POWER_POOL_CORRELATION"
_WARNING = re.compile(r"brain_check_not_counted: ([A-Z0-9_.]+)='WARNING'")
DESCRIPTION_MIN = 100   # BRAIN's POWER_POOL_DESCRIPTION_LENGTH limit seen on our account


def warned_checks(result: ResultRecord, ignore: Collection[str] = ()) -> list[str]:
    """BRAIN checks reported as WARNING, in note order, ``ignore`` left out."""
    found = [m.group(1) for n in result.notes for m in [_WARNING.search(n)] if m]
    return [c for c in dict.fromkeys(found) if c not in set(ignore)]


def candidate_reason(result: ResultRecord | None, decision: DecisionRecord | None,
                     submission_checks: Collection[str] = ()) -> str | None:
    """The tag text when the result is a Power Pool candidate (see module), else None."""
    if result is None or result.status != "complete":
        return None
    if decision is not None and decision.decision == "accept":
        return None
    if CHECK not in result.check_values or result.checks.get(CHECK) is False:
        return None
    submission = set(submission_checks) - {CHECK}
    if any(ok is False and name not in submission for name, ok in result.checks.items()):
        return None
    warned = warned_checks(result, submission)
    if not warned:
        return None
    pool = "passes" if result.checks.get(CHECK) is True else "not computed yet"
    return (f"{PREFIX}: every BRAIN check passes; regular-alpha checks only warned "
            f"({', '.join(warned)}); {CHECK} {pool}. Recorded only: submit by hand if wanted")


def brain_alpha_id(result: ResultRecord) -> str | None:
    for n in result.notes:
        if n.startswith("brain_alpha_id: "):
            return n.split(": ", 1)[1].strip()
    return None


def describe(c: CandidateRecord, r: ResultRecord, ignore: Collection[str] = ()) -> str:
    """One console line for a candidate (``ignore``: the submission-only checks)."""
    stats = ", ".join(f"{k} {v:.2f}" for k, v in (("Sharpe", r.sharpe), ("fitness", r.fitness))
                      if v is not None)
    alpha = brain_alpha_id(r)
    return (f"power pool candidate: {c.candidate_id}"
            + (f" (BRAIN alpha {alpha})" if alpha else "")
            + f", {c.region} {c.universe} delay {c.delay}: {stats}; warned: "
            + ", ".join(warned_checks(r, set(ignore) - {CHECK})))
