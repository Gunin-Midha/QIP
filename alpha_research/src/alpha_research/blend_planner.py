"""Blend Planner: model-planned repair waves built on blends (owner 2026-11-26).

Three of our five accepts were blends of the near miss with a signal from another dataset
(EUR reversal + analyst69 revision breadth, IND momentum + analyst4 EPS trend, CHN retail
flow + analyst69 breadth), each designed by hand from evidence. This module does that for
leads within ``max_distance`` of accept that fail an information class (production or self
correlation, Sharpe, recent Sharpe, sub/robust universe).

Method
1. Partner candidates, by rules (no model). The other datasets of the alpha's market
   (``exclude_categories`` left out), each with up to ``fields_per_candidate`` signal
   fields (least used first; crowded and heavily used fields left out) and our evidence on
   it: the strongest of our simulated alphas mainly on that dataset in the market (signed
   Sharpe, its daily-PnL correlation with the alpha being fixed and its highest with our
   submitted pool alphas of the region, its mean Sharpe in the alpha's weak years), and any
   "partner alone" tests of it. Scored for the failure: fewer BRAIN alphas, lower PnL
   correlation with the alpha and the pool, own |Sharpe|, strength in the weak years
   (recent Sharpe), coverage (sub-universe). A dataset is *proven* when an alpha of ours on
   it reached |Sharpe| >= ``min_partner_sharpe``, a partner-alone test of it did, or a
   partner part of one of our accepted alphas (any market) reads it. Since 2026-11-27
   (analyst69, the partner of two accepts, ranked 11th-15th): usage is the median BRAIN
   alphas of the dataset's signal fields, with no bonus below ``min_field_alphas`` (near
   empty is unknown, not unused signal); revision families (breadth.py) and a role in an
   accept add to the score; the ``family_items`` least-used families' up/down/count fields
   are listed first, then fields with at least ``min_field_alphas`` alphas.
2. One planning call (the repair model): the alpha with BRAIN's numbers, the lineage's
   tried fixes, the candidates with their evidence, the pairings already tried in this
   market and their results, the lessons. It returns blend variants (partner signal, weight,
   mechanism) and partner-alone tests.
3. Rule checks before anything reaches BRAIN: the partner reads only fields of its
   candidate dataset (and base fields); the weight is one of ``weights``; the mechanism is
   at least ``min_mechanism_chars`` long; a blend's partner dataset is proven (an untested
   one may only be a partner-alone test, at most ``partner_alone_per_wave``); a partner
   expression that failed alone (Sharpe below ``min_partner_sharpe``, its stated direction
   fixed: never flipped) is refused; at most ``max_partners_per_lineage`` partner datasets
   per lineage; a blend is built on the lineage's last non-blend alpha, so a blend never
   has more than two parts.
4. One review call (a critic model from another maker than the planner): does the
   mechanism make economic sense for the two signals, is the partner a different idea, and
   does the direction follow from the mechanism. Variants it does not keep are dropped.
5. After BRAIN: a blend whose yearly Sharpe beats its base alpha's in fewer than
   ``stable_year_share`` of the years gets a flag (recorded, decisions unchanged: an alpha
   passes on BRAIN's checks alone).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from . import breadth
from . import feasibility
from .config import ConfigError
from .fastexpr import FastExprError, FastExprSpec, analyze
from .generator import GenerationError, ask_json, crowded_fields, field_role, heavily_used
from .lever_stats import distance
from .llm import LLMClient, ModelDown
from .schemas import CandidateRecord, ResultRecord

BLEND_PLAN = "blend_plan"         # lever of a planned blend child
PARTNER_ALONE = "partner_alone"   # lever of a partner-alone test child
TAG = re.compile(r"\[blend partner=(?P<ds>[\w.-]+) weight=(?P<w>[\d.]+)\]")
INFO_CLASSES = ("prod_correlation", "self_correlation", "sharpe", "recent_sharpe",
                "sub_universe")


@dataclass(frozen=True)
class BlendConfig:
    enabled: bool = False
    classes: tuple[str, ...] = INFO_CLASSES
    max_distance: float = 0.4
    max_candidates: int = 10
    fields_per_candidate: int = 8
    exclude_categories: tuple[str, ...] = ("pv", "risk")
    min_coverage: float = 0.3
    min_partner_sharpe: float = 0.5
    weights: tuple[float, ...] = (1.0, 2.0)
    max_partners_per_lineage: int = 4
    partner_alone_per_wave: int = 1
    min_overlap_days: int = 120
    min_mechanism_chars: int = 40
    stable_year_share: float = 0.5
    review: bool = True
    # fields (and datasets, by their fields' median) read by fewer BRAIN alphas than this
    # are listed after the others and get no usage bonus: near-empty means unknown, not
    # unused signal (analyst69's least-used fields are metadata such as market status)
    min_field_alphas: int = 5
    family_items: int = 3        # revision families (breadth.py) listed first, per dataset

    @classmethod
    def from_dict(cls, raw) -> "BlendConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ConfigError("agents config: blend_plan must be a mapping")
        d = cls()
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"agents config: blend_plan: unknown keys {sorted(unknown)}")
        kw = {}
        for k in ("enabled", "review"):
            if k in raw:
                if not isinstance(raw[k], bool):
                    raise ConfigError(f"agents config: blend_plan.{k} must be true or false")
                kw[k] = raw[k]
        for k in ("max_candidates", "fields_per_candidate", "max_partners_per_lineage",
                  "partner_alone_per_wave", "min_overlap_days", "min_mechanism_chars",
                  "min_field_alphas", "family_items"):
            if k in raw:
                v = raw[k]
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    raise ConfigError(f"agents config: blend_plan.{k} must be an integer >= 0")
                kw[k] = v
        for k in ("max_distance", "min_coverage", "min_partner_sharpe", "stable_year_share"):
            if k in raw:
                v = raw[k]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                    raise ConfigError(f"agents config: blend_plan.{k} must be a number >= 0")
                kw[k] = float(v)
        for k in ("classes", "exclude_categories"):
            if k in raw:
                v = raw[k]
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise ConfigError(f"agents config: blend_plan.{k} must be a list of names")
                kw[k] = tuple(v)
        if "weights" in raw:
            v = raw["weights"]
            if not isinstance(v, list) or not v or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in v):
                raise ConfigError("agents config: blend_plan.weights must be positive numbers")
            kw["weights"] = tuple(float(x) for x in v)
        return cls(**{**{k: getattr(d, k) for k in cls.__dataclass_fields__}, **kw})


@dataclass
class PartnerCandidate:
    dataset: str
    category: str | None
    fields: list[tuple[str, str]]          # (field id, short note)
    brain_alphas: float | None = None
    value_score: float | None = None
    coverage: float | None = None
    evidence: dict | None = None            # our strongest alpha on the dataset here
    alone: list[dict] = field(default_factory=list)   # partner-alone tests
    proven: bool = False
    score: float = 0.0
    field_alphas: float | None = None       # median BRAIN alphas of its signal fields
    families: int = 0                       # revision families (breadth.py)
    in_accept: bool = False                 # a partner of one of our accepted alphas


@dataclass
class Variant:
    dataset: str
    partner: str
    mechanism: str
    weight: float | None = None             # None = a partner-alone test
    reason: str = ""                         # the reviewer's


@dataclass
class Plan:
    variants: list[Variant] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    candidates: list[PartnerCandidate] = field(default_factory=list)
    llm_calls: int = 0
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers


def eligible(decision, result, failing: set[str], config: BlendConfig) -> tuple[bool, str]:
    """(a planned blend wave applies, why): an information class fails and the alpha is
    within ``max_distance`` of accept."""
    if not config.enabled:
        return False, "blend planning off"
    hit = sorted(failing & set(config.classes))
    if not hit:
        return False, "no information class fails"
    dist = distance(decision, result)
    if dist is None or dist > config.max_distance:
        return False, f"distance to accept {dist} > {config.max_distance}"
    return True, f"fails {', '.join(hit)}; distance {dist:.2f}"


def corr(a: pd.Series | None, b: pd.Series | None, min_days: int) -> float | None:
    if a is None or b is None:
        return None
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < min_days:
        return None
    c = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    return None if pd.isna(c) else round(float(c), 3)


def _yearly(result: ResultRecord | None) -> dict[int, float]:
    return {y.year: y.sharpe for y in (result.yearly if result else []) if y.sharpe is not None}


def weak_years(result: ResultRecord | None, recent: bool) -> list[int]:
    """The alpha's weak years: below its median yearly Sharpe; for a recent-Sharpe failure,
    its last two years."""
    ys = _yearly(result)
    if not ys:
        return []
    if recent:
        return sorted(ys)[-2:]
    med = statistics.median(ys.values())
    return sorted(y for y, s in ys.items() if s < med)


def blend_base(candidate: CandidateRecord, by_id: Mapping[str, dict]) -> CandidateRecord:
    """The lineage's last alpha that is not a blend: blends are built on it, so a blend never
    has more than two parts."""
    c = candidate
    while c.lever in (BLEND_PLAN, "blend_other_dataset", "breadth_blend", "recipe_blend") \
            and c.parent_id in by_id:
        c = by_id[c.parent_id]["candidate"]
    return c


def lineage_root(candidate: CandidateRecord, by_id: Mapping[str, dict]) -> str:
    c = candidate
    while c.parent_id in by_id:
        c = by_id[c.parent_id]["candidate"]
    return c.candidate_id


def tag(dataset: str, weight: float) -> str:
    return f"[blend partner={dataset} weight={weight:g}]"


def pairings(outcomes: list[dict], region: str, delay: int, meta: Mapping[str, dict],
             limit: int = 12) -> list[str]:
    """Planned blends tried in this market, newest first: data categories paired, weight,
    and what BRAIN said against the base alpha (Sharpe, production correlation, decision)."""
    by_id = {o["candidate"].candidate_id: o for o in outcomes}
    cats = {(m or {}).get("dataset"): (m or {}).get("category") for m in meta.values()}
    lines = []
    for o in reversed(outcomes):
        c, r, d = o["candidate"], o["result"], o["decision"]
        if c.lever != BLEND_PLAN or c.region != region or c.delay != delay or r is None:
            continue
        m = TAG.search(c.hypothesis or "")
        if not m:
            continue
        base = blend_base(c, by_id)
        b = by_id.get(base.candidate_id, {}).get("result")

        def prod(x):
            v = ((x.check_values if x else {}) or {}).get("PROD_CORRELATION") or {}
            return v.get("value")
        bits = [f"{cats.get(base.dataset) or base.dataset} + {cats.get(m['ds']) or m['ds']} "
                f"({m['ds']}) x{m['w']}"]
        if r.sharpe is not None:
            bits.append(f"Sharpe {b.sharpe if b else None} -> {r.sharpe}")
        if prod(r) is not None:
            bits.append(f"PROD {prod(b)} -> {prod(r)}")
        if d is not None:
            bits.append(f"{d.decision}/{d.failure_class}")
        lines.append("; ".join(bits))
        if len(lines) >= limit:
            break
    return lines


# --------------------------------------------------------------------------- step 1


def partner_candidates(parent: CandidateRecord, parent_result: ResultRecord | None,
                       failing: set[str], outcomes: list[dict], meta: Mapping[str, dict],
                       spec: FastExprSpec, generator_config, pool_pnl: Mapping[str, pd.Series],
                       config: BlendConfig, bar: float) -> list[PartnerCandidate]:
    """Partner datasets for this alpha and failure, scored by fit (module doc, step 1)."""
    if not meta:
        return []
    by_id = {o["candidate"].candidate_id: o for o in outcomes}
    root = lineage_root(parent, by_id)
    base = blend_base(parent, by_id)
    try:
        used = {f for f in analyze(parent.expression, spec).fields_used}
    except FastExprError:
        used = set()
    own = {base.dataset, parent.dataset} | {(meta.get(f) or {}).get("dataset") for f in used}
    crowded = crowded_fields(meta, generator_config.crowded_percentile,
                             generator_config.crowded_min_alphas, exclude=spec.base_fields) \
        if generator_config.crowded_percentile is not None else set()
    crowded |= heavily_used(meta, generator_config.heavy_alphas, spec.base_fields)
    per: dict[str, PartnerCandidate] = {}
    for f, m in meta.items():
        m = m or {}
        ds = m.get("dataset")
        if (not ds or ds in own or m.get("category") in config.exclude_categories
                or f in crowded or f in spec.base_fields or m.get("type") == "GROUP"
                or field_role(f, meta, generator_config) != "signal"):
            continue
        cov = m.get("coverage")
        if isinstance(cov, (int, float)) and cov < config.min_coverage:
            continue
        p = per.setdefault(ds, PartnerCandidate(ds, m.get("category"), [],
                                                m.get("dataset_alpha_count"),
                                                m.get("dataset_value_score")))
        p.fields.append((f, m))
    parent_pnl = parent_result.daily_pnl if parent_result is not None else None
    weak = weak_years(parent_result, "recent_sharpe" in failing)
    fams = breadth.families(meta, breadth.BreadthConfig(),
                            skip=set(spec.base_fields) | set(spec.groups))
    # datasets a partner part of one of our accepted alphas (any market: field ids are the
    # same in every market) reads: proven by that accept
    in_accept = set()
    for o in outcomes:
        c, d = o["candidate"], o["decision"]
        if d is None or d.decision != "accept":
            continue
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", c.expression or ""))
        in_accept |= {ds for f in words
                      if (ds := (meta.get(f) or {}).get("dataset")) and ds != c.dataset
                      and f not in spec.base_fields}
    out = []
    for ds, p in per.items():
        n_alphas = [fm[1].get("alpha_count") or 0 for fm in p.fields]
        p.field_alphas = statistics.median(n_alphas) if n_alphas else None
        listed = {f for f, _ in p.fields}
        usable = [x for x in fams.get(ds, []) if x.up in listed and x.down in listed]
        p.families = len(usable)
        fam_note: dict[str, str] = {}   # the least-used families' fields, listed first
        for fam in usable[:config.family_items]:
            for role, x in (("up", fam.up), ("down", fam.down), ("count", fam.count)):
                if x in listed and x not in fam_note:
                    fam_note[x] = f"revision family {fam.item} ({role}); "
        p.in_accept = ds in in_accept
        order = list(fam_note)
        p.fields.sort(key=lambda fm: (order.index(fm[0]) if fm[0] in fam_note else len(order),
                                      (fm[1].get("alpha_count") or 0) < config.min_field_alphas,
                                      fm[1].get("alpha_count") or 0, fm[0]))
        covs = [fm[1]["coverage"] for fm in p.fields
                if isinstance(fm[1].get("coverage"), (int, float))]
        p.coverage = round(statistics.median(covs), 2) if covs else None
        keep = len(fam_note) + (config.fields_per_candidate // 2 if fam_note
                                else config.fields_per_candidate)
        p.fields = [(f, fam_note.get(f, "") + _note(m)) for f, m in p.fields[:keep]]
        # our evidence: strongest complete alpha mainly on this dataset here (not this lineage)
        best = None
        for o in outcomes:
            c, r = o["candidate"], o["result"]
            if (c.dataset != ds or c.region != parent.region or c.delay != parent.delay
                    or r is None or r.status != "complete" or r.sharpe is None
                    or c.candidate_id.startswith(root)):
                continue
            if c.lever == PARTNER_ALONE:
                p.alone.append({"expression": c.expression, "sharpe": r.sharpe,
                                "candidate_id": c.candidate_id,
                                "sub_universe_ratio": _r2(feasibility.ratio(r))})
                continue
            if best is None or abs(r.sharpe) > abs(best["result"].sharpe):
                best = o
        if best is not None:
            r = best["result"]
            ys = _yearly(r)
            sign = 1 if r.sharpe >= 0 else -1
            p.evidence = {
                "candidate_id": best["candidate"].candidate_id,
                "expression": " ".join(best["candidate"].expression.split())[:220],
                "sharpe": r.sharpe, "turnover": r.turnover,
                "pnl_corr_with_alpha": corr(r.daily_pnl, parent_pnl, config.min_overlap_days),
                "max_pnl_corr_with_pool": max(
                    (abs(c) for s in pool_pnl.values()
                     if (c := corr(r.daily_pnl, s, config.min_overlap_days)) is not None),
                    default=None),
                "sub_universe_ratio": _r2(feasibility.ratio(r)),
                "sharpe_in_weak_years": round(statistics.mean(
                    [sign * ys[y] for y in weak if y in ys]), 2) if any(
                    y in ys for y in weak) else None,
            }
        p.proven = bool((best is not None and abs(best["result"].sharpe)
                         >= config.min_partner_sharpe)
                        or any(a["sharpe"] >= config.min_partner_sharpe for a in p.alone)
                        or p.in_accept)
        p.score = _score(p, failing, bar, config.min_field_alphas,
                         feasibility.limit_share(parent_result))
        out.append(p)
    proven = sorted([p for p in out if p.proven], key=lambda p: (-p.score, p.dataset))
    untested = sorted([p for p in out if not p.proven], key=lambda p: (-p.score, p.dataset))
    keep = proven[:config.max_candidates]
    return keep + untested[:max(0, min(3, config.max_candidates - len(keep)))]


def _note(m: Mapping) -> str:
    desc = " ".join((m.get("description") or "").split())[:80]
    bits = [f"{m.get('alpha_count')} BRAIN alphas"] if m.get("alpha_count") is not None else []
    if m.get("type") == "VECTOR":
        bits.append("vector: wrap in vec_avg/vec_sum")
    return f"{desc} ({', '.join(bits)})" if bits else desc


def _r2(x):
    return None if x is None else round(x, 2)


def _score(p: PartnerCandidate, failing: set[str], bar: float,
           min_field_alphas: int = 5, limit_share: float | None = None) -> float:
    """Fit of a partner dataset (module doc, step 1). Usage is its signal fields' median
    BRAIN alphas, not the dataset's total (analyst69: 188,000 alphas in all, its revision
    fields 0-80 each); a near-empty dataset (median below ``min_field_alphas``) gets no
    usage bonus. Revision families and a partner role in one of our accepts add to it."""
    e = p.evidence or {}
    s = 0.0
    if p.field_alphas is not None and p.field_alphas >= min_field_alphas:
        s += 1.0 / (1.0 + p.field_alphas / 500.0) * (2.0 if "prod_correlation" in failing
                                                       else 1.0)
    if p.families:
        s += 1.0
    if p.in_accept:
        s += 1.0
    if e.get("sharpe") is not None and bar:
        s += min(1.0, abs(e["sharpe"]) / bar)
    if e.get("pnl_corr_with_alpha") is not None:
        s += 1.0 - abs(e["pnl_corr_with_alpha"])
    if e.get("max_pnl_corr_with_pool") is not None:
        s += (1.0 - e["max_pnl_corr_with_pool"]) * (
            1.5 if failing & {"prod_correlation", "self_correlation"} else 0.5)
    if "recent_sharpe" in failing and e.get("sharpe_in_weak_years") is not None and bar:
        s += max(-1.0, min(1.0, e["sharpe_in_weak_years"] / bar))
    if "sub_universe" in failing and p.coverage is not None:
        s += p.coverage
    if "sub_universe" in failing and limit_share:
        # owner 2026-11-27: a partner fixes sub-universe only with large-cap Sharpe of its
        # own (its measured sub-universe / full Sharpe at or above BRAIN's limit share);
        # below it, it lowers the blend's ratio (EUR breadth onto a small-cap base)
        ratios = [x for x in [e.get("sub_universe_ratio")]
                  + [a.get("sub_universe_ratio") for a in p.alone] if x is not None]
        if ratios:
            s += min(1.0, max(ratios) / limit_share) if max(ratios) >= limit_share else -1.0
    return round(s, 3)


# --------------------------------------------------------------------------- steps 2-4


class BlendPlanner:
    def __init__(self, spec: FastExprSpec, config: BlendConfig, *, llm: LLMClient | None,
                 reviewer: LLMClient | None = None, context: str = "",
                 max_output: int = 4096, limits=None):
        self.spec, self.config = spec, config
        self.llm, self.reviewer = llm, reviewer
        self.context, self.max_output, self.limits = context, max_output, limits

    def plan(self, parent: CandidateRecord, decision, result: ResultRecord | None,
             failing: set[str], candidates: list[PartnerCandidate], *, catalog,
             outcomes: list[dict], meta: Mapping[str, dict], max_variants: int,
             metrics: dict, tried: list[str], lessons: Sequence[str] = ()) -> Plan:
        from .prompts import blend_plan_system, blend_plan_user, blend_review_system, \
            blend_review_user
        plan = Plan(candidates=candidates)
        if self.llm is None or not candidates or max_variants <= 0:
            plan.notes.append("blend plan: no model or no partner candidates")
            return plan
        by_id = {o["candidate"].candidate_id: o for o in outcomes}
        root = lineage_root(parent, by_id)
        lineage = {TAG.search(o["candidate"].hypothesis or "")["ds"]
                   for o in outcomes if o["candidate"].candidate_id.startswith(root)
                   and o["candidate"].lever == BLEND_PLAN
                   and TAG.search(o["candidate"].hypothesis or "")}
        failed_alone = {" ".join(a["expression"].split()) for p in candidates for a in p.alone
                        if a["sharpe"] < self.config.min_partner_sharpe}
        user = blend_plan_user(parent, metrics, sorted(failing), candidates, tried,
                               pairings(outcomes, parent.region, parent.delay, meta),
                               list(lessons), max_variants, self.config)
        try:
            reply, calls = ask_json(self.llm, blend_plan_system(self.context, self.config),
                                    user, max_output=self.max_output)
        except GenerationError as exc:
            plan.llm_calls += 2
            plan.notes.append(f"blend plan: model reply unusable: {exc}")
            return plan
        except ModelDown as exc:
            plan.notes.append(f"blend plan: no model available: {exc}")
            return plan
        plan.llm_calls += calls
        cands = {p.dataset: p for p in candidates}
        alone_left = self.config.partner_alone_per_wave
        new_partners = set()
        for kind, items in (("blend", reply.get("variants")),
                            ("alone", reply.get("partner_alone"))):
            for item in items if isinstance(items, list) else []:
                v, why = self._check(item, kind, cands, catalog, failed_alone, lineage,
                                     new_partners)
                if v is None:
                    plan.dropped.append(why)
                    continue
                if v.weight is None:
                    if alone_left <= 0:
                        plan.dropped.append(f"{v.dataset}: partner-alone tests per wave used")
                        continue
                    alone_left -= 1
                elif v.dataset not in lineage:
                    new_partners.add(v.dataset)
                plan.variants.append(v)
        if self.config.review and plan.variants:
            if self.reviewer is None:
                plan.notes.append("blend review: no reviewer model from another maker; "
                                  "variants kept on the rule checks")
            else:
                try:
                    rev, calls = ask_json(self.reviewer, blend_review_system(self.context),
                                          blend_review_user(parent, plan.variants),
                                          max_output=self.max_output)
                    plan.llm_calls += calls
                    plan.variants = self._apply_review(rev, plan)
                except (GenerationError, ModelDown) as exc:
                    plan.notes.append(f"blend review unavailable ({exc}); variants kept on "
                                      f"the rule checks")
        plan.variants = plan.variants[:max_variants]
        return plan

    def _check(self, item, kind, cands, catalog, failed_alone, lineage, new_partners):
        if not isinstance(item, dict):
            return None, "variant is not an object"
        ds, partner = item.get("dataset"), item.get("partner")
        mech = " ".join(str(item.get("mechanism") or "").split())
        if not isinstance(ds, str) or ds not in cands:
            return None, f"{ds!r}: not a listed partner candidate"
        if not isinstance(partner, str) or not partner.strip():
            return None, f"{ds}: no partner expression"
        partner = partner.strip()
        if len(mech) < self.config.min_mechanism_chars:
            return None, f"{ds}: mechanism missing or too short"
        a = analyze(partner, self.spec, datafields=catalog, limits=self.limits)
        if not a.ok:
            return None, f"{ds}: partner fails validation: {'; '.join(a.errors)[:160]}"
        listed = {f for f, _ in cands[ds].fields}
        fields = set(a.fields_used) - set(self.spec.base_fields)
        if not fields & listed or fields - listed:
            return None, f"{ds}: partner must read listed fields of {ds} only"
        if " ".join(partner.split()) in failed_alone:
            return None, f"{ds}: this partner failed alone (direction is not flipped)"
        if kind == "alone":
            return Variant(ds, partner, mech, None), ""
        if not cands[ds].proven:
            return None, f"{ds}: untested partner; only a partner-alone test is allowed"
        w = item.get("weight")
        if isinstance(w, bool) or not isinstance(w, (int, float)) or float(w) not in \
                self.config.weights:
            return None, f"{ds}: weight {w!r} not in {list(self.config.weights)}"
        if (ds not in lineage and ds not in new_partners
                and len(lineage | new_partners) >= self.config.max_partners_per_lineage):
            return None, f"{ds}: {self.config.max_partners_per_lineage} partner datasets " \
                         f"already tried in this lineage"
        return Variant(ds, partner, mech, float(w)), ""

    @staticmethod
    def _apply_review(rev, plan: Plan) -> list[Variant]:
        rows = rev.get("reviews") if isinstance(rev, dict) else None
        verdict = {}
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and isinstance(r.get("index"), int):
                verdict[r["index"]] = (r.get("keep") is True, str(r.get("reason") or ""))
        kept = []
        for i, v in enumerate(plan.variants):
            ok, why = verdict.get(i, (False, "no review given"))
            v.reason = why
            if ok:
                kept.append(v)
            else:
                plan.dropped.append(f"{v.dataset}: reviewer: {why or 'not kept'}")
        return kept


# --------------------------------------------------------------------------- step 5


def stability_flag(child: ResultRecord | None, base: ResultRecord | None,
                   config: BlendConfig) -> str | None:
    """A flag when the blend beats its base alpha's yearly Sharpe in fewer than
    ``stable_year_share`` of their common years (None when stable or unknown)."""
    a, b = _yearly(child), _yearly(base)
    years = sorted(set(a) & set(b))
    if len(years) < 3:
        return None
    better = sum(1 for y in years if a[y] > b[y])
    if better / len(years) >= config.stable_year_share:
        return None
    return (f"blend beats its base alpha in only {better} of {len(years)} years "
            f"(blend_plan.stable_year_share {config.stable_year_share:g})")
