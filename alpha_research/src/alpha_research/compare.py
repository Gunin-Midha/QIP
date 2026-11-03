"""Model comparison: how each generator model and each critic model has done, from memory.

Generators
Every candidate is credited to the generator model of the arm it came from
(``arm_models``): the ``generator_model`` the Critic recorded for it or for an ancestor
(repair children and critic revisions inherit their parent's arm), else the root
candidate's ``model``. Per model: candidates simulated, latest decisions, and
accept rate = accept / (accept + reject + repair). Pending candidates are counted but left
out of the rate.

Critics
Every verdict from ``critic_reviews``, primary and shadow. Per critic model:
- verdict counts (simulate / revise / drop), and how many came from the model versus the
  rule fallback (an unusable reply);
- agreement with the primary critic on the same candidate (shadow rows only);
- against BRAIN, for candidates whose own expression was simulated and decided: how often
  a "simulate" verdict was accepted, and how often a "revise"/"drop" flag was right (not
  accepted). Candidates the primary critic dropped were never simulated, so a critic's
  false drops cannot be scored; shadows on simulated candidates are the evidence.
Rates are None when there is nothing to divide by.
"""

from __future__ import annotations

from collections import defaultdict

DECIDED = ("accept", "reject", "repair")


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def arm_models(outcomes: list[dict], reviews: list[dict]) -> dict[str, str | None]:
    """candidate_id -> generator model of its arm, for every candidate in memory."""
    gen_of = {r["candidate_id"]: r["generator_model"] for r in reviews
              if r.get("generator_model")}
    cands = {row["candidate"].candidate_id: row["candidate"] for row in outcomes}

    def find(cid: str) -> str | None:
        seen, cur, root_model = set(), cid, None
        while cur is not None and cur not in seen:
            seen.add(cur)
            if cur in gen_of:
                return gen_of[cur]
            c = cands.get(cur)
            if c is None:
                break
            root_model, cur = c.model, c.parent_id
        return root_model

    return {cid: find(cid) for cid in cands}


def generator_comparison(outcomes: list[dict], reviews: list[dict]) -> dict:
    models = arm_models(outcomes, reviews)
    table: dict = defaultdict(lambda: {"candidates": 0, "accept": 0, "reject": 0, "repair": 0,
                                       "pending": 0, "no_decision": 0})
    for row in outcomes:
        cid = row["candidate"].candidate_id
        t = table[models.get(cid) or "unknown"]
        t["candidates"] += 1
        d = row["decision"]
        if d is None:
            t["no_decision"] += 1
        elif d.decision in DECIDED or d.decision == "pending":
            t[d.decision] += 1
    for t in table.values():
        t["accept_rate"] = _rate(t["accept"], t["accept"] + t["reject"] + t["repair"])
    return dict(sorted(table.items()))


def critic_comparison(reviews: list[dict], outcomes: list[dict]) -> dict:
    decision = {row["candidate"].candidate_id: row["decision"].decision
                for row in outcomes if row["decision"] is not None}
    primary = {r["candidate_id"]: r["verdict"] for r in reviews if r["role"] == "primary"}
    table: dict = defaultdict(lambda: {
        "reviews": 0, "as_primary": 0, "as_shadow": 0, "from_model": 0, "rule_fallback": 0,
        "simulate": 0, "revise": 0, "drop": 0, "shadow_agreements": 0,
        "cleared_accepted": 0, "cleared_not_accepted": 0,
        "flagged_accepted": 0, "flagged_not_accepted": 0})
    for r in reviews:
        t = table[r["critic_model"] or "rules only"]
        t["reviews"] += 1
        t["as_" + r["role"]] += 1
        t["from_model" if r["source"] == "model" else "rule_fallback"] += 1
        if r["verdict"] in ("simulate", "revise", "drop"):
            t[r["verdict"]] += 1
        if r["role"] == "shadow" and primary.get(r["candidate_id"]) == r["verdict"]:
            t["shadow_agreements"] += 1
        d = decision.get(r["candidate_id"])
        if d in DECIDED:
            kind = "cleared" if r["verdict"] == "simulate" else "flagged"
            t[f"{kind}_{'accepted' if d == 'accept' else 'not_accepted'}"] += 1
    for t in table.values():
        t["shadow_agreement_rate"] = _rate(t["shadow_agreements"], t["as_shadow"])
        t["cleared_accept_rate"] = _rate(t["cleared_accepted"],
                                         t["cleared_accepted"] + t["cleared_not_accepted"])
        t["flag_precision"] = _rate(t["flagged_not_accepted"],
                                    t["flagged_accepted"] + t["flagged_not_accepted"])
    return dict(sorted(table.items()))


def model_comparison(outcomes: list[dict], reviews: list[dict]) -> dict:
    return {"generators": generator_comparison(outcomes, reviews),
            "critics": critic_comparison(reviews, outcomes)}
