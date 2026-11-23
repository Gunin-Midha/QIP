"""Lesson Writer: after every batch, what memory shows becomes advice for the agents.

The hand rescues of 2026-11-14/15 showed what a person takes from a batch: which change
moved which BRAIN check, and which kinds of data clear production correlation. This module
does that after every research run and every ``cli run`` batch. It always reads all of
memory, not only the batch, so a lesson that later evidence contradicts is replaced at the
next write. Old lessons stay in memory (append-only); only the latest write is used.

Method
1. Evidence, no model (``evidence``):
   a. Change effects: every simulated child of a simulated parent (repair levers, critic
      revisions) gives the change in Sharpe, fitness, sub-universe ratio (sub-universe
      Sharpe / Sharpe, only at Sharpe >= ``RATIO_MIN_SHARPE``, where the ratio is stable),
      recent-years Sharpe (BRAIN's LOW_2Y_SHARPE or IS_LADDER_SHARPE value), production
      correlation (PROD_CORRELATION value) and turnover, and whether the parent's failure
      class is gone (for production and self correlation, which BRAIN computes only once
      every other check passes, only an accept or a measured passing value counts). Grouped by (lever, parent's failure class), over all regions and per
      region; the median change per metric (robust to one odd pair).
      Children without a lever (hand-written variants) say what changed in their
      hypothesis: the latest ``MAX_DESCRIBED`` such pairs go to the model as "described
      changes" with the same numbers (the rules cannot read them).
   b. Data and production correlation: alphas with a PROD_CORRELATION value, grouped by
      the dataset categories of the fields they read (catalog of their market; base fields
      count as pv): mean, and how many are above BRAIN's limit.
   c. Failure profile: per (region, category of the alpha's dataset) the simulated count,
      accepts and the most common failure classes.
   Groups with fewer than ``min_evidence`` alphas are left out.
2. Rule lessons (``rule_lessons``): one sentence per group, numbers included. Audience
   "repair" (change effects) or "generator" (data, failure profile); scope "all", a
   region, or "REGION/category".
3. With a model (``lessons.use_model``): it condenses the evidence into at most
   ``max_lessons`` lessons (prompts.lessons_system). A lesson is kept only with a known
   audience, a scope that appears in the evidence, a number in its text and at most
   ``MAX_CHARS`` characters. With none kept, no model, or a model failure, the rule lessons
   are used.
4. Stored append-only (``MemoryStore.put_lessons``, one batch id per write). Agents read the
   latest write at the start of a run (``for_agent``): the Generator next to the configured
   category lessons, the Repair Agent in its request. None of this changes a decision.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .config import ConfigError
from .fastexpr import FastExprError, FastExprSpec
from .generator import GenerationError, ask_json, dataset_category
from .lever_stats import failing_classes
from .llm import LLMClient, LLMUnavailable
from .prompts import lessons_system, lessons_user
from .repair_agent import expression_fields

AUDIENCES = ("generator", "repair")
MAX_CHARS = 400
RATIO_MIN_SHARPE = 0.5
COMPUTED_LAST = {"prod_correlation", "self_correlation"}
MAX_DESCRIBED = 40
METRICS = (("sharpe", "Sharpe", 2), ("fitness", "fitness", 2),
           ("sub_ratio", "sub-universe ratio", 2), ("recent_sharpe", "recent-years Sharpe", 2),
           ("prod_correlation", "production correlation", 3), ("turnover", "turnover", 3))


@dataclass(frozen=True)
class LessonsConfig:
    after_every_batch: bool = True
    use_model: bool = True
    min_evidence: int = 3
    max_lessons: int = 12
    # evidence sent to the model is trimmed to this many characters (Groq's free plan allows
    # 8,000 tokens a minute; a 2026-11-15 write asked for 8,569 and was refused)
    max_prompt_chars: int = 12000
    # research plan whose datafield catalogs give the dataset categories when no research
    # run supplies them (cli run, cli lessons); None = categories unknown
    catalog_plan: str | None = None

    @classmethod
    def from_dict(cls, raw) -> "LessonsConfig":
        if raw is None:
            return cls(after_every_batch=False)
        if not isinstance(raw, dict):
            raise ConfigError("agents config: 'lessons' must be a mapping")
        out = {}
        for key in ("after_every_batch", "use_model"):
            v = raw.get(key, getattr(cls, key))
            if not isinstance(v, bool):
                raise ConfigError(f"agents config: lessons.{key} must be true or false")
            out[key] = v
        for key in ("min_evidence", "max_lessons", "max_prompt_chars"):
            v = raw.get(key, getattr(cls, key))
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError(f"agents config: lessons.{key} must be an integer >= 1")
            out[key] = v
        plan = raw.get("catalog_plan")
        if plan is not None and not isinstance(plan, str):
            raise ConfigError("agents config: lessons.catalog_plan must be a path or null")
        return cls(**out, catalog_plan=plan)


@dataclass
class LessonsOutcome:
    lessons: list[dict] = field(default_factory=list)   # {audience, scope, text}
    source: str = "rules"                               # "rules" or "model"
    evidence: dict = field(default_factory=dict)
    llm_calls: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _check_value(result, *names) -> float | None:
    for n in names:
        v = _num(((result.check_values or {}).get(n) or {}).get("value"))
        if v is not None:
            return v
    return None


def metrics(result) -> dict | None:
    """The BRAIN numbers a change can move, or None for a result that is not complete."""
    if result is None or result.status != "complete":
        return None
    s, sub = _num(result.sharpe), _num(result.sub_universe_sharpe)
    return {"sharpe": s, "fitness": _num(result.fitness),
            "sub_ratio": (sub / s if sub is not None and s is not None and s >= RATIO_MIN_SHARPE
                          else None),
            "recent_sharpe": _check_value(result, "LOW_2Y_SHARPE", "IS_LADDER_SHARPE"),
            "prod_correlation": _check_value(result, "PROD_CORRELATION"),
            "turnover": _num(result.turnover)}


def _categories(c, spec: FastExprSpec, meta: Mapping) -> str:
    """Dataset categories of the fields an alpha reads, e.g. "analyst+pv"."""
    try:
        fields = expression_fields(c.expression, spec)
    except FastExprError:
        return "unknown"
    cats = {"pv" if f in spec.base_fields else (meta.get(f) or {}).get("category") or "unknown"
            for f in fields}
    return "+".join(sorted(cats)) or "pv"


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def evidence(outcomes: list[dict], spec: FastExprSpec,
             meta_for: Callable[[str, int], Mapping], min_evidence: int = 3) -> dict:
    """Change effects, production correlation by data read, failure profile (see module)."""
    by_id = {r["candidate"].candidate_id: r for r in outcomes}
    changes: dict[tuple, dict] = {}
    described = []
    for r in outcomes:
        c = r["candidate"]
        p = by_id.get(c.parent_id) if c.parent_id else None
        if p is None or p["decision"] is None:
            continue
        mc, mp = metrics(r["result"]), metrics(p["result"])
        if mc is None or mp is None:
            continue
        cls = p["decision"].failure_class
        accepted = r["decision"] is not None and r["decision"].decision == "accept"
        fixed = accepted or cls not in failing_classes(r["decision"], r["result"]) | (
            {r["decision"].failure_class} if r["decision"] is not None else set())
        if cls in COMPUTED_LAST and not accepted:
            # BRAIN computes these only once every other check passes: not failing them is
            # not evidence of a fix unless the value was measured
            fixed = cls == "prod_correlation" and mc["prod_correlation"] is not None and fixed
        if not c.lever:
            if c.hypothesis:
                # measured numbers first; the author's note says what was intended and often
                # predicts an effect that did not happen (2026-11-15: a model lesson took
                # "should take 0.739 to ~0.66" as a result when BRAIN measured 0.720)
                d_rec = r["decision"]
                described.append({
                    "region": c.region, "parent_failure": cls, "fixed": bool(fixed),
                    "child_outcome": (f"{d_rec.decision} / {d_rec.failure_class}"
                                      if d_rec else None),
                    **{f"measured_change_{k}": round(mc[k] - mp[k], d) for k, _, d in METRICS
                       if mc[k] is not None and mp[k] is not None},
                    **{f"child_{k}": round(mc[k], d) for k, _, d in METRICS
                       if k in ("prod_correlation", "recent_sharpe") and mc[k] is not None},
                    "author_note_unverified": " ".join(c.hypothesis.split())[:240]})
            continue
        for scope in ("all", c.region):
            g = changes.setdefault((scope, c.lever, cls), {"n": 0, "fixed": 0,
                                                           **{k: [] for k, _, _ in METRICS}})
            g["n"] += 1
            g["fixed"] += bool(fixed)
            for k, _, _ in METRICS:
                if mc[k] is not None and mp[k] is not None:
                    g[k].append(mc[k] - mp[k])
    change_rows = []
    for (scope, lever, cls), g in sorted(changes.items()):
        if g["n"] < min_evidence:
            continue
        row = {"scope": scope, "lever": lever, "failure_class": cls, "n": g["n"],
               "fixed": g["fixed"]}
        for k, _, digits in METRICS:
            if len(g[k]) >= min_evidence:
                row[f"median_change_{k}"] = round(_median(g[k]), digits)
        change_rows.append(row)

    prod: dict[tuple, list[tuple[float, float | None]]] = {}
    profile: dict[tuple, dict] = {}
    for r in outcomes:
        c, res, d = r["candidate"], r["result"], r["decision"]
        m = metrics(res)
        if m is None:
            continue
        meta = meta_for(c.region, c.delay) or {}
        if m["prod_correlation"] is not None:
            limit = _num(((res.check_values or {}).get("PROD_CORRELATION") or {}).get("limit"))
            label = _categories(c, spec, meta)
            for scope in ("all", c.region):
                prod.setdefault((scope, label), []).append((m["prod_correlation"], limit))
        cat = dataset_category(c.dataset, meta) or c.dataset or "unknown"
        g = profile.setdefault((c.region, cat), {"n": 0, "accepted": 0, "failures": Counter()})
        g["n"] += 1
        if d is not None and d.decision == "accept":
            g["accepted"] += 1
        elif d is not None and d.failure_class not in (None, "none"):
            g["failures"][d.failure_class] += 1
    prod_rows = []
    for (scope, label), vals in sorted(prod.items()):
        if len(vals) < min_evidence:
            continue
        over = sum(1 for v, lim in vals if lim is not None and v > lim)
        prod_rows.append({"scope": scope, "data": label, "n": len(vals),
                          "mean": round(_mean([v for v, _ in vals]), 3), "above_limit": over})
    profile_rows = [{"scope": f"{region}/{cat}", "n": g["n"], "accepted": g["accepted"],
                     "top_failures": dict(g["failures"].most_common(3))}
                    for (region, cat), g in sorted(profile.items()) if g["n"] >= min_evidence]
    return {"change_effects": change_rows, "production_correlation_by_data": prod_rows,
            "failure_profile": profile_rows, "described_changes": described[-MAX_DESCRIBED:]}


# ---------------------------------------------------------------------------
# lessons
# ---------------------------------------------------------------------------

def _where(scope: str) -> str:
    return "all regions" if scope == "all" else scope


def rule_lessons(ev: dict) -> list[dict]:
    out = []
    for row in ev.get("change_effects", []):
        moves = [f"{label} {row[f'median_change_{k}']:+g}" for k, label, _ in METRICS
                 if f"median_change_{k}" in row]
        text = (f"{row['lever']} on {row['failure_class']} failures ({_where(row['scope'])}, "
                f"{row['n']} repairs): fixed the class in {row['fixed']}"
                + (f"; median change {', '.join(moves)}" if moves else "") + ".")
        out.append({"audience": "repair", "scope": row["scope"], "text": text})
    by_scope: dict[str, list[dict]] = {}
    for row in ev.get("production_correlation_by_data", []):
        by_scope.setdefault(row["scope"], []).append(row)
    for scope, rows in by_scope.items():
        parts = [f"{r['data']} {r['mean']:g} mean over {r['n']} ({r['above_limit']} above "
                 "the limit)" for r in sorted(rows, key=lambda r: r["mean"])]
        out.append({"audience": "generator", "scope": scope,
                    "text": f"Production correlation by the data an alpha reads "
                            f"({_where(scope)}): {'; '.join(parts)}."})
    for row in ev.get("failure_profile", []):
        fails = ", ".join(f"{k} {v}" for k, v in row["top_failures"].items()) or "none"
        out.append({"audience": "generator", "scope": row["scope"],
                    "text": f"{row['scope']}: {row['n']} simulated, {row['accepted']} accepted; "
                            f"most common failures: {fails}."})
    return out


def trimmed(ev: dict, max_chars: int) -> dict:
    """``ev`` cut to about ``max_chars`` of compact JSON for the model: the oldest
    described changes go first, then the smallest failure-profile groups, then per-region
    change effects (those over all regions stay)."""
    ev = {k: list(v) for k, v in ev.items()}

    def size() -> int:
        return len(json.dumps(ev, separators=(",", ":"), default=str))

    ev.setdefault("described_changes", [])
    while size() > max_chars and ev["described_changes"]:
        ev["described_changes"].pop(0)
    prof = sorted(ev.get("failure_profile", []), key=lambda r: -r["n"])
    while size() > max_chars and prof:
        prof.pop()
        ev["failure_profile"] = prof
    ce = ev.get("change_effects", [])
    while size() > max_chars and any(r["scope"] != "all" for r in ce):
        last = max(i for i, r in enumerate(ce) if r["scope"] != "all")
        ce.pop(last)
    return ev


def _scopes(ev: dict) -> set[str]:
    out = {"all"}
    for rows in ev.values():
        for r in rows:
            scope = r.get("scope") or r.get("region")   # described changes carry a region
            if not scope:
                continue
            out.add(scope)
            if "/" in scope:
                out |= set(scope.split("/"))
    return out


def _valid(item, scopes: set[str]) -> dict | None:
    if not isinstance(item, dict):
        return None
    aud, scope, text = item.get("audience"), item.get("scope"), item.get("text")
    if aud not in AUDIENCES or scope not in scopes or not isinstance(text, str):
        return None
    text = " ".join(text.split())
    if not text or len(text) > MAX_CHARS or not any(ch.isdigit() for ch in text):
        return None
    return {"audience": aud, "scope": scope, "text": text}


def learn(outcomes: list[dict], spec: FastExprSpec, meta_for: Callable[[str, int], Mapping],
          config: LessonsConfig, *, llm: LLMClient | None = None, context: str = "",
          max_output: int = 2048) -> LessonsOutcome:
    """Evidence from all of memory, then lessons (model when given and usable, else rules)."""
    ev = evidence(outcomes, spec, meta_for, config.min_evidence)
    out = LessonsOutcome(rule_lessons(ev)[:config.max_lessons * 3], "rules", ev)
    if not out.lessons:
        out.notes.append(f"no group has {config.min_evidence} or more alphas yet")
        return out
    if llm is None or not config.use_model:
        return out
    try:
        reply, calls = ask_json(llm, lessons_system(context),
                                lessons_user(trimmed(ev, config.max_prompt_chars),
                                             config.max_lessons),
                                max_output=max_output)
    except GenerationError as exc:
        out.llm_calls += 2
        out.notes.append(f"lessons model reply unusable, rule lessons kept: {exc}")
        return out
    except LLMUnavailable as exc:   # down, out of quota, request too large: rules stand
        out.notes.append(f"lessons model unavailable, rule lessons kept: {exc}")
        return out
    out.llm_calls += calls
    items = reply.get("lessons") if isinstance(reply, dict) else None
    scopes = _scopes(ev)
    kept = [x for x in (_valid(i, scopes) for i in (items or [])) if x][:config.max_lessons]
    if kept:
        out.lessons, out.source = kept, "model"
    else:
        out.notes.append("no lesson in the model's reply passed the checks; rule lessons kept")
    return out


def for_agent(rows: list[dict], audience: str, region: str | None = None,
              category: str | None = None) -> list[str]:
    """Texts of the stored lessons for one agent and market, general ones first."""
    wanted = ["all", region, category, f"{region}/{category}" if region and category else None]
    order = {s: i for i, s in enumerate(wanted) if s}
    picked = [r for r in rows if r.get("audience") == audience and r.get("scope") in order]
    return [r["text"] for r in sorted(picked, key=lambda r: order[r["scope"]])]


def summary(out: LessonsOutcome) -> str:
    return (f"lessons: {len(out.lessons)} written ({out.source})"
            + (f"; {'; '.join(out.notes)}" if out.notes else ""))
