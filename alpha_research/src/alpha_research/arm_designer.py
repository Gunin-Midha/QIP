"""Arm Designer: the pipeline writes its own research arms (owner 2026-11-26).

Until now every starting arm (dataset, settings, hypotheses) came from a hand-written research
plan. The designer picks them by rules, per market (region, delay), at the start of a run, and
only when the market's previous batch of arms is done (owner: "a new batch of arms should only
be written when previous arms are done"). No model call is made.

Method
1. Datasets. Candidates are the datasets of the market's downloaded field catalog with at
   least ``min_signal_fields`` signal fields (not id or time fields: ``generator.field_role``),
   not designed before in this market (once every candidate was, the oldest come back). Each
   gets a score, the weighted sum of
     value        BRAIN's valueScore / the market's highest (scores run 1-10; BRAIN rates a
                  dataset higher where few alphas exist)
     uncrowded    1 - percentile of BRAIN's alpha count among the candidates
     fields       min(1, signal fields / 20)
     evidence     our best |Sharpe| (raw or BRAIN's risk-neutralized) on the dataset in this
                  market / the market's Sharpe bar, capped at 1
     novelty      1 when we never simulated the dataset in this market
     prod         minus the share of our alphas on it that failed PROD_CORRELATION
   A batch takes ``arms_per_batch`` datasets: ``exploit_arms`` where our best |Sharpe| in
   the market is highest (at least ``exploit_min_share`` of the bar; designed before or
   not: all five accepts came from near misses, so new ideas keep coming where signal
   showed), then ``explore_arms`` never tried, then one per data category in score order,
   then the best of the rest.
2. Settings. Region, universe, truncation and dates are the market's (its plan arms'
   settings). Decay and neutralization come from our near misses in the market and the
   dataset's category (accepts, and repairs within ``near_distance`` of accept): the most
   common values once ``settings_evidence`` of them exist (group neutralizations only: a
   new alpha never starts from a factor neutralization or NONE); else ``decay_by_category``
   and the market's neutralization.
3. Hypotheses (no model, step 3 of the proposal is not built): two briefs from BRAIN's own
   dataset name and description plus what three of its signal fields measure (their BRAIN
   descriptions: a dataset's own description can be wrong, e.g. GBR other250 describes an
   ID mapping while its fields are product ratings, prices and bestseller ranks), one on
   levels and one on changes. Words that are field ids of the market's catalog are left out,
   since a named field would go to the top of the Generator's field listing whatever its
   crowding (gen-v6).
4. Done. A design is done once ``done_after_alphas`` new alphas (no repair lever) on its
   dataset in that market reached BRAIN since its batch was written, counting every model.
   The market gets a new batch when every design of its latest batch is done.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

from .config import ConfigError
from .lever_stats import distance
from .schemas import GROUP_NEUTRALIZATIONS

CHECK_PROD = "PROD_CORRELATION"


@dataclass(frozen=True)
class DesignerConfig:
    enabled: bool = False
    arms_per_batch: int = 4
    done_after_alphas: int = 6
    explore_arms: int = 1
    exploit_arms: int = 1
    exploit_min_share: float = 0.6
    min_signal_fields: int = 5
    weights: Mapping[str, float] = field(default_factory=lambda: {
        "value": 1.0, "uncrowded": 1.0, "fields": 0.5, "evidence": 1.0, "novelty": 0.5,
        "prod": 1.0})
    decay_by_category: Mapping[str, int] = field(default_factory=lambda: {"default": 6})
    settings_evidence: int = 3
    near_distance: float = 0.6

    @classmethod
    def from_dict(cls, raw) -> "DesignerConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ConfigError("agents config: designer must be a mapping")
        d = cls()
        known = {"enabled", "arms_per_batch", "done_after_alphas", "explore_arms",
                 "exploit_arms", "exploit_min_share",
                 "min_signal_fields", "weights", "decay_by_category", "settings_evidence",
                 "near_distance"}
        if set(raw) - known:
            raise ConfigError(f"agents config: designer: unknown keys {sorted(set(raw) - known)}")

        def pos_int(key, lo=1):
            v = raw.get(key, getattr(d, key))
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ConfigError(f"agents config: designer.{key} must be an integer >= {lo}")
            return v

        enabled = raw.get("enabled", d.enabled)
        if not isinstance(enabled, bool):
            raise ConfigError("agents config: designer.enabled must be true or false")
        weights = {**d.weights, **(raw.get("weights") or {})}
        if set(weights) - set(d.weights) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
                for v in weights.values()):
            raise ConfigError(f"agents config: designer.weights: keys {sorted(d.weights)}, "
                              "non-negative numbers")
        decays = raw.get("decay_by_category") or dict(d.decay_by_category)
        if "default" not in decays or not all(
                isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 512
                for v in decays.values()):
            raise ConfigError("agents config: designer.decay_by_category needs 'default' and "
                              "integer decays 0-512")
        share = raw.get("exploit_min_share", d.exploit_min_share)
        if isinstance(share, bool) or not isinstance(share, (int, float)) or share < 0:
            raise ConfigError("agents config: designer.exploit_min_share must be >= 0")
        near = raw.get("near_distance", d.near_distance)
        if isinstance(near, bool) or not isinstance(near, (int, float)) or near < 0:
            raise ConfigError("agents config: designer.near_distance must be >= 0")
        return cls(enabled=enabled, arms_per_batch=pos_int("arms_per_batch"),
                   done_after_alphas=pos_int("done_after_alphas"),
                   explore_arms=pos_int("explore_arms", 0),
                   exploit_arms=pos_int("exploit_arms", 0), exploit_min_share=float(share),
                   min_signal_fields=pos_int("min_signal_fields", 0),
                   weights=weights, decay_by_category=dict(decays),
                   settings_evidence=pos_int("settings_evidence"), near_distance=float(near))


# ---------------------------------------------------------------------------- step 1


def _market(outcomes: list[dict], region: str, delay: int) -> list[dict]:
    return [o for o in outcomes if o["candidate"].region == region
            and o["candidate"].delay == delay]


def _best_sharpe(result) -> float | None:
    vals = [abs(result.sharpe)] if result is not None and result.sharpe is not None else []
    rn = ((result.alt_stats or {}).get("risk_neutralized") or {}).get("sharpe") \
        if result is not None else None
    if isinstance(rn, (int, float)) and not isinstance(rn, bool):
        vals.append(abs(rn))
    return max(vals) if vals else None


def _prod_failed(decision) -> bool:
    return decision is not None and any(
        g.passed is False and g.name.endswith(CHECK_PROD) for g in decision.gate_results)


def score_datasets(candidates: Mapping[str, dict], outcomes: list[dict], region: str,
                   delay: int, bar: float, config: DesignerConfig) -> list[dict]:
    """Every candidate dataset with its score parts, best first. ``candidates``: {dataset:
    {category, value_score, alpha_count, signal_fields}}."""
    ours: dict[str, list[dict]] = {}
    for o in _market(outcomes, region, delay):
        r = o["result"]
        if r is not None and r.status == "complete" and o["candidate"].dataset:
            ours.setdefault(o["candidate"].dataset, []).append(o)
    counts = sorted(c["alpha_count"] for c in candidates.values()
                    if c.get("alpha_count") is not None)
    top_value = max((c["value_score"] for c in candidates.values()
                     if c.get("value_score") is not None), default=None)
    w = config.weights
    rows = []
    for ds, c in candidates.items():
        n = c.get("alpha_count")
        crowd = (sum(1 for x in counts if x < n) / len(counts)) if n is not None and counts \
            else 0.5
        tried = ours.get(ds, [])
        best = max((b for o in tried if (b := _best_sharpe(o["result"])) is not None),
                   default=None)
        parts = {
            "value": (c["value_score"] / top_value) if c.get("value_score") is not None
                     and top_value else 0.5,
            "uncrowded": 1 - crowd,
            "fields": min(1.0, c.get("signal_fields", 0) / 20),
            "evidence": min(1.0, best / bar) if best is not None and bar else 0.0,
            "novelty": 0.0 if tried else 1.0,
            "prod": -(sum(_prod_failed(o["decision"]) for o in tried) / len(tried))
                    if tried else 0.0,
        }
        rows.append({"dataset": ds, "category": c.get("category"),
                     "score": round(sum(w[k] * v for k, v in parts.items()), 4),
                     "parts": {k: round(v, 3) for k, v in parts.items()},
                     "value_score": c.get("value_score"), "brain_alphas": n,
                     "signal_fields": c.get("signal_fields", 0), "our_alphas": len(tried),
                     "our_best_sharpe": None if best is None else round(best, 2)})
    return sorted(rows, key=lambda x: (-x["score"], x["dataset"]))


def exploit_rows(scored: list[dict], bar: float, config: DesignerConfig) -> list[dict]:
    """Datasets with our strongest results in the market, best |Sharpe| first, at least
    ``exploit_min_share`` x the bar; at most ``exploit_arms``."""
    rows = [r for r in scored if r["our_best_sharpe"] is not None
            and r["our_best_sharpe"] >= config.exploit_min_share * bar]
    rows.sort(key=lambda r: (-r["our_best_sharpe"], r["dataset"]))
    return rows[:config.exploit_arms]


def choose_datasets(scored: list[dict], config: DesignerConfig,
                    exploit: list[dict] = ()) -> list[dict]:
    """``arms_per_batch`` datasets: ``exploit`` first, then never-tried ones
    (``explore_arms``), then one per data category in score order, then the best of the
    rest. Each gets a ``why``."""
    k = config.arms_per_batch
    picked: list[dict] = []

    def take(row, why):
        if all(p["dataset"] != row["dataset"] for p in picked) and len(picked) < k:
            picked.append({**row, "why": why})

    for row in exploit:
        take(row, f"exploit: our best |Sharpe| here {row['our_best_sharpe']}")
    for row in [r for r in scored if r["our_alphas"] == 0][:config.explore_arms]:
        take(row, "exploration: never simulated in this market")
    seen = {r["category"] for r in picked}
    for row in scored:
        if row["category"] not in seen:
            seen.add(row["category"])
            take(row, f"best score in category {row['category']}")
    for row in scored:
        take(row, "next best score")
    return picked


# ---------------------------------------------------------------------------- step 2


def design_settings(category: str | None, outcomes: list[dict], region: str, delay: int,
                    base_neutralization: str, dataset_category: Mapping[str, str | None],
                    config: DesignerConfig) -> tuple[int, str, str]:
    """(decay, neutralization, why) for a new arm on data of ``category`` in a market."""
    near = []
    for o in _market(outcomes, region, delay):
        c, d = o["candidate"], o["decision"]
        if d is None or dataset_category.get(c.dataset) != category:
            continue
        dist = distance(d, o["result"])
        if d.decision == "accept" or (d.decision == "repair" and dist is not None
                                      and dist <= config.near_distance):
            near.append(c)
    default_decay = config.decay_by_category.get(category or "",
                                                 config.decay_by_category["default"])
    if len(near) < config.settings_evidence:
        return (default_decay, base_neutralization,
                f"{len(near)} near misses on {category} data here (< "
                f"{config.settings_evidence}): category decay, market neutralization")
    decay = Counter(c.decay for c in near).most_common(1)[0][0]
    groups = Counter(c.neutralization for c in near if c.neutralization in GROUP_NEUTRALIZATIONS)
    neut = groups.most_common(1)[0][0] if groups else base_neutralization
    return decay, neut, (f"most common among {len(near)} near misses on {category} data "
                         f"here")


# ---------------------------------------------------------------------------- step 3


def _first_sentences(text: str, limit: int = 420) -> str:
    out = ""
    for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        if len(out) + len(s) > limit:
            break
        out = f"{out} {s}".strip()
    return out or (text or "")[:limit]


def _strip_ids(text: str, ids: set[str]) -> str:
    return " ".join(w for w in text.split() if re.sub(r"[^a-z0-9_]", "", w.lower()) not in ids)


def briefs(name: str | None, description: str | None, category: str | None, dataset: str,
           field_ids: set[str], field_notes: list[str] = ()) -> list[str]:
    """Two hypotheses from BRAIN's dataset text and what some of its fields measure, naming
    no field of the market's catalog."""
    title = name or dataset
    ids = {f.lower() for f in field_ids}
    about = _strip_ids(_first_sentences(description or "", 300), ids)
    about = f" BRAIN describes it as: {about}" if about else ""
    notes = [_strip_ids(_first_sentences(n, 110), ids) for n in field_notes if n]
    if notes:
        about += " Its fields measure, for example: " + "; ".join(n.rstrip(".") for n in notes) \
            + "."
    kind = f"{category} data" if category else "data"
    return [
        f"{title} ({kind}).{about} Find a cross-sectional signal in this dataset's "
        f"less crowded fields, from the mechanism the data measures; blend with another "
        f"dataset of this market only if the partner adds an independent idea.",
        f"{title} ({kind}): changes, not levels. Use how this dataset's values change "
        f"over weeks or months (revisions, surprises, shifts against the stock's own "
        f"history), ranked within the arm's group, with smoothing suited to how often the "
        f"data updates.",
    ]


# ---------------------------------------------------------------------------- step 4


def batch_done(designs: list[dict], fresh: Mapping[str, int],
               config: DesignerConfig) -> tuple[bool, list[str]]:
    """(every design of the batch is done, lines saying how far each one is)."""
    lines, done = [], True
    for d in designs:
        n = fresh.get(d["dataset"], 0)
        if n < config.done_after_alphas:
            done = False
        lines.append(f"{d['dataset']} {min(n, config.done_after_alphas)}/"
                     f"{config.done_after_alphas}")
    return done, lines


def latest_batch(designs: list[dict]) -> list[dict]:
    if not designs:
        return []
    last = designs[-1]["batch_id"]
    return [d for d in designs if d["batch_id"] == last]


def design_batch(*, region: str, delay: int, candidates: Mapping[str, dict],
                 dataset_text: Mapping[str, dict], field_ids: set[str],
                 outcomes: list[dict], bar: float, base_settings: Mapping,
                 earlier: set[str], config: DesignerConfig) -> list[dict]:
    """A new batch of designs for one market (steps 1-3). ``earlier``: datasets designed
    before in this market (left out while others remain)."""
    pool = {ds: c for ds, c in candidates.items()
            if c.get("signal_fields", 0) >= config.min_signal_fields}
    fresh = {ds: c for ds, c in pool.items() if ds not in earlier}
    scored = score_datasets(fresh or pool, outcomes, region, delay, bar, config)
    exploit = exploit_rows(score_datasets(pool, outcomes, region, delay, bar, config), bar,
                           config)
    chosen = choose_datasets(scored, config, exploit)
    cats = {ds: c.get("category") for ds, c in candidates.items()}
    out = []
    for row in chosen:
        decay, neut, why_settings = design_settings(
            row["category"], outcomes, region, delay, base_settings["neutralization"], cats,
            config)
        text = dataset_text.get(row["dataset"]) or {}
        out.append({
            "region": region, "delay": delay, "dataset": row["dataset"],
            "category": row["category"], "score": row["score"], "parts": row["parts"],
            "why": row["why"], "facts": {k: row[k] for k in (
                "value_score", "brain_alphas", "signal_fields", "our_alphas",
                "our_best_sharpe")},
            "settings": {**dict(base_settings), "decay": decay, "neutralization": neut,
                         "dataset": row["dataset"]},
            "settings_why": why_settings,
            "hypotheses": briefs(text.get("name"), text.get("description"), row["category"],
                                 row["dataset"], field_ids,
                                 candidates[row["dataset"]].get("field_notes", [])),
        })
    return out


def market_candidates(meta: Mapping[str, dict], role) -> dict[str, dict]:
    """{dataset: {category, value_score, alpha_count, signal_fields, field_notes}} from a
    field catalog's meta; ``role(field)`` is ``generator.field_role`` for the market;
    field_notes = BRAIN's descriptions of up to three of its least used signal fields."""
    out: dict[str, dict] = {}
    notes: dict[str, list[tuple]] = {}
    for f, m in (meta or {}).items():
        ds = (m or {}).get("dataset")
        if not ds:
            continue
        d = out.setdefault(ds, {"category": m.get("category"), "value_score": None,
                                "alpha_count": None, "signal_fields": 0})
        if role(f) == "signal" and m.get("description"):
            notes.setdefault(ds, []).append((m.get("alpha_count") or 0, f, m["description"]))
        if m.get("dataset_value_score") is not None:
            d["value_score"] = m["dataset_value_score"]
        if m.get("dataset_alpha_count") is not None:
            d["alpha_count"] = m["dataset_alpha_count"]
        if role(f) == "signal":
            d["signal_fields"] += 1
    for ds, rows in notes.items():
        seen, picked = set(), []
        for _, _, desc in sorted(rows):
            key = desc[:40].lower()
            if key not in seen:
                seen.add(key)
                picked.append(desc)
        out[ds]["field_notes"] = picked[:3]
    return out

