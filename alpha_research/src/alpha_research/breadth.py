"""Revision-breadth partners: blends built by rule, no model (owner 2026-11-27).

Five of our seven accepts (EUR reversal, CHN retail flow, GBR value x3) were the near miss
blended with analyst *net revision breadth*: for each forecast item, upward minus downward
estimate revisions over the number of estimates, averaged over a few less-used items. The
pipeline never wrote one: the Blend Planner lists fields one by one (fewest BRAIN alphas
first, which in analyst69 are metadata such as ``anl69_analyst_market_status``), and its
dataset score favoured near-empty datasets, so analyst69 ranked 11th-15th. This module finds
the revision families in a market's catalog and builds the partner itself.

Method
1. Families, by field ids, in data categories ``categories`` (analyst: an upside/downside
   beta or an up-move time is not a revision); a pair whose description names an industry,
   group or sector aggregate (``aggregate_words``; analyst15 counts companies in the
   industry with raised estimates, one value per group) is not a stock's own revisions. A field whose id has an "up" part (``up``, ``raisednum``,
   ``raised``, ``upgrades``, ... ``breadth.pairs``) whose "down" twin (same id, part
   swapped) is in the same dataset of the catalog is a family's up/down pair; its count
   field is the id with the up part and up to ``max_drop_parts`` parts before it removed
   plus a ``count_suffixes`` suffix (``anl69_best_eps_4wk_up`` -> ``anl69_best_eps_numest``,
   ``est_12m_pre_raisednum_4wks`` -> ``est_12m_pre_num``). Usage = BRAIN alphas on up + down;
   items with a ``prefer_items`` part (earnings, profit, sales: every accepted blend used
   these) first and ``avoid_items`` (net debt, book value, dividends, NAV, FFO) last; within
   each, least used first, those under ``min_item_alphas`` last. Blends never use the
   ``avoid_items`` items; items with ``heavy_item_alphas`` or more BRAIN alphas come last.
2. Partner expression per item: ``(up - dn) / max(count, 1)``, or ``(up - dn) /
   (up + dn + 1)`` without a count field; VECTOR fields in ``vec_avg``, fields with coverage
   below ``backfill_below_coverage`` in ``ts_backfill(x, backfill_days)``. The partner is the
   mean of ``items_per_variant`` items.
3. Variants: datasets of the market other than the alpha's own (``exclude_categories`` left
   out) with at least ``min_items`` families, ordered by: a partner of one of our accepted
   alphas (any market) read that dataset's families; more families; lower median usage.
   Per dataset (one horizon per item: ``item_root``): the first items at each of ``weights``, then the next least-used items
   at the first weight (the GBR round: least-used items at 2:1 had production correlation
   0.537, crowded items at 1:1 0.669). Combinations already tried along the lineage
   (hypothesis tag) are skipped.
4. The blend is ``repair_agent.build_blend`` (rank(alpha) + weight x rank(partner), 0.5
   where the partner has no value) on the lineage's last non-blend alpha, so blends never
   stack.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping

from .config import ConfigError

BREADTH_LEVER = "breadth_blend"
TAG = re.compile(r"\[breadth partner=(?P<ds>[\w.-]+) items=(?P<items>[\w.,-]+) "
                 r"weight=(?P<w>[\d.]+)\]")
INFO_CLASSES = ("prod_correlation", "self_correlation", "sharpe", "recent_sharpe",
                "sub_universe")
DEFAULT_PAIRS = (("up", "dn"), ("up", "down"), ("raisednum", "lowerednum"),
                 ("raised", "lowered"), ("upgrades", "downgrades"), ("upgrade", "downgrade"),
                 ("upnum", "dnnum"), ("numup", "numdn"))
DEFAULT_COUNT_SUFFIXES = ("numest", "num", "count", "numests")


@dataclass(frozen=True)
class BreadthConfig:
    enabled: bool = False
    classes: tuple[str, ...] = INFO_CLASSES
    max_distance: float = 0.4
    items_per_variant: int = 3
    min_items: int = 2
    weights: tuple[float, ...] = (1.0, 2.0)
    variants_per_wave: int = 2
    max_per_lineage: int = 4
    backfill_below_coverage: float = 0.5
    backfill_days: int = 20
    exclude_categories: tuple[str, ...] = ("pv", "risk")
    categories: tuple[str, ...] = ("analyst",)   # data categories whose families count
    min_item_alphas: int = 5      # items read by fewer BRAIN alphas come after the others
    heavy_item_alphas: int = 500  # items read by this many BRAIN alphas come after the others
    # a family whose up field's description has one of these is an industry or group
    # aggregate (analyst15: "number of companies in the industry with raised EPS"): the same
    # value for every stock of the group, so it cannot rank stocks inside a sector
    # items whose id has one of these parts go first (every accepted blend used earnings,
    # profit or sales revisions); items with an ``avoid_items`` part go last (net debt, book
    # value, dividends, NAV, FFO: estimates that rarely say much about the next weeks)
    prefer_items: tuple[str, ...] = ("eps", "net", "netprofit", "sales", "sal", "ptp", "pre",
                                     "ebit", "ebt", "ebitda", "ebs", "opr", "opp", "cps", "pretaxprofit",
                                     "gps", "grm", "revenue", "rev")
    avoid_items: tuple[str, ...] = ("debt", "ndt", "ndebt", "bps", "dps", "nav", "tbv",
                                    "ffo", "ffops", "capex", "ent")
    aggregate_words: tuple[str, ...] = ("number of companies", "companies in", "industry",
                                        "group", "sector", "aggregated")
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS
    count_suffixes: tuple[str, ...] = DEFAULT_COUNT_SUFFIXES
    max_drop_parts: int = 2

    @classmethod
    def from_dict(cls, raw) -> "BreadthConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ConfigError("agents config: breadth_blend must be a mapping")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"agents config: breadth_blend: unknown keys {sorted(unknown)}")
        kw = {}
        if "enabled" in raw:
            if not isinstance(raw["enabled"], bool):
                raise ConfigError("agents config: breadth_blend.enabled must be true or false")
            kw["enabled"] = raw["enabled"]
        for k in ("items_per_variant", "min_items", "variants_per_wave", "max_per_lineage",
                  "backfill_days", "max_drop_parts", "min_item_alphas", "heavy_item_alphas"):
            if k in raw:
                v = raw[k]
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    raise ConfigError(f"agents config: breadth_blend.{k} must be an "
                                      f"integer >= 0")
                kw[k] = v
        for k in ("max_distance", "backfill_below_coverage"):
            if k in raw:
                v = raw[k]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                    raise ConfigError(f"agents config: breadth_blend.{k} must be a number >= 0")
                kw[k] = float(v)
        for k in ("classes", "exclude_categories", "count_suffixes", "categories",
                  "aggregate_words", "prefer_items", "avoid_items"):
            if k in raw:
                v = raw[k]
                if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
                    raise ConfigError(f"agents config: breadth_blend.{k} must be a list of names")
                kw[k] = tuple(v)
        if "weights" in raw:
            v = raw["weights"]
            if not isinstance(v, list) or not v or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in v):
                raise ConfigError("agents config: breadth_blend.weights must be positive numbers")
            kw["weights"] = tuple(float(x) for x in v)
        if "pairs" in raw:
            v = raw["pairs"]
            if not isinstance(v, list) or not v or not all(
                    isinstance(p, list) and len(p) == 2
                    and all(isinstance(x, str) and x for x in p) for p in v):
                raise ConfigError("agents config: breadth_blend.pairs must be a list of "
                                  "[up part, down part]")
            kw["pairs"] = tuple((p[0], p[1]) for p in v)
        if kw.get("min_items", cls.min_items) < 1 or kw.get("items_per_variant",
                                                             cls.items_per_variant) < 1:
            raise ConfigError("agents config: breadth_blend.min_items and items_per_variant "
                              "must be >= 1")
        return cls(**kw)


@dataclass(frozen=True)
class Family:
    dataset: str
    item: str                 # the up field's id without its up part, e.g. est_12m_pre_4wks
    up: str
    down: str
    count: str | None
    usage: float              # BRAIN alphas on up + down
    coverage: float | None    # the lower of up's and down's
    vector: bool
    category: str | None


def families(meta: Mapping[str, dict], config: BreadthConfig,
             skip: Iterable[str] = ()) -> dict[str, list[Family]]:
    """{dataset: revision families}, each list least used first (method step 1). Fields in
    ``skip`` (base fields, groups) are never family members."""
    skip = set(skip)
    out: dict[str, list[Family]] = {}
    seen = set()
    for f in sorted(meta):
        m = meta.get(f) or {}
        ds = m.get("dataset")
        if not ds or f in skip or m.get("category") not in config.categories:
            continue
        toks = f.split("_")
        for i, t in enumerate(toks):
            for up, dn in config.pairs:
                if t != up:
                    continue
                down = "_".join(toks[:i] + [dn] + toks[i + 1:])
                dm = meta.get(down) or {}
                if dm.get("dataset") != ds or down in skip or (f, down) in seen:
                    continue
                desc = (m.get("description") or "").lower()
                if any(w in desc for w in config.aggregate_words):
                    continue
                seen.add((f, down))
                count = _count_field(toks, i, ds, meta, config, skip)
                covs = [c for c in (m.get("coverage"), dm.get("coverage"))
                        if isinstance(c, (int, float))]
                out.setdefault(ds, []).append(Family(
                    dataset=ds, item="_".join(toks[:i] + toks[i + 1:]), up=f, down=down,
                    count=count,
                    usage=float((m.get("alpha_count") or 0) + (dm.get("alpha_count") or 0)),
                    coverage=min(covs) if covs else None,
                    vector="VECTOR" in (m.get("type"), dm.get("type")),
                    category=m.get("category")))
    for ds in out:   # heavily used items last (crowded: higher production correlation);
        # then earnings/profit/sales items first, balance-sheet and payout items last;
        # within each, least used first, items almost nobody reads after the rest
        out[ds].sort(key=lambda x: (x.usage >= config.heavy_item_alphas,
                                    _item_class(x.item, config),
                                    x.usage < config.min_item_alphas, x.usage, x.item))
    return out


def _item_class(item: str, config: BreadthConfig) -> int:
    """0 = an earnings/profit/sales item, 2 = a balance-sheet or payout item, 1 = other."""
    parts = set(item.lower().split("_"))
    if parts & set(config.avoid_items):
        return 2
    return 0 if parts & set(config.prefer_items) else 1


HORIZON = re.compile(r"^\d+(w|wk|wks|week|weeks|m|mth|mths|month|months|d|y)$")


def item_root(item: str) -> str:
    """The item without its horizon parts: est_12m_grm_4wks and est_12m_grm_1mth are one item."""
    return "_".join(p for p in item.lower().split("_") if not HORIZON.match(p))


def _count_field(toks, i, ds, meta, config, skip) -> str | None:
    for drop in range(0, config.max_drop_parts + 1):
        if i - drop < 1:
            break
        prefix = toks[:i - drop]
        for suf in config.count_suffixes:
            name = "_".join(prefix + [suf])
            if name not in skip and (meta.get(name) or {}).get("dataset") == ds:
                return name
    return None


def _term(field: str, fam: Family, config: BreadthConfig) -> str:
    x = f"vec_avg({field})" if fam.vector else field
    if fam.coverage is not None and fam.coverage < config.backfill_below_coverage:
        x = f"ts_backfill({x}, {config.backfill_days})"
    return x


def item_expression(fam: Family, config: BreadthConfig) -> str:
    """One item's net revision breadth (method step 2)."""
    up, dn = _term(fam.up, fam, config), _term(fam.down, fam, config)
    if fam.count is not None:
        return f"({up} - {dn}) / max({_term(fam.count, fam, config)}, 1)"
    return f"({up} - {dn}) / ({up} + {dn} + 1)"


def partner_expression(items: list[Family], config: BreadthConfig) -> str:
    body = " + ".join(item_expression(f, config) for f in items)
    return f"({body}) / {len(items)}" if len(items) > 1 else body


def tag(dataset: str, items: list[Family], weight: float) -> str:
    return (f"[breadth partner={dataset} items={','.join(f.item for f in items)} "
            f"weight={weight:g}]")


def accepted_datasets(fams: Mapping[str, list[Family]],
                      accepted_expressions: Iterable[str]) -> set[str]:
    """Datasets one of whose families (up and down) an accepted alpha reads: the field ids
    are the same in every market, so an accept anywhere counts."""
    words = set()
    for e in accepted_expressions:
        words |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", e or ""))
    return {ds for ds, fs in fams.items() if any(f.up in words and f.down in words for f in fs)}


def variants(fams: Mapping[str, list[Family]], own: set, config: BreadthConfig, *,
             proven: set[str] = frozenset(), tried: set[str] = frozenset(),
             limit: int | None = None) -> list[tuple[str, list[Family], float]]:
    """(dataset, items, weight) in try order (method step 3); ``tried`` holds tags already
    used along the lineage."""
    ok = {ds: fs for ds, fs in fams.items()
          if ds not in own and len(fs) >= config.min_items
          and (fs[0].category not in config.exclude_categories)}
    order = sorted(ok, key=lambda ds: (ds not in proven, -len(ok[ds]),
                                       statistics.median(f.usage for f in ok[ds]), ds))
    k = config.items_per_variant
    out = []
    for ds in order:
        fs, roots = [], set()
        for x in ok[ds]:   # one horizon per item: the first (least used) in order; never
            # an ``avoid_items`` item (net debt, book value, dividends, NAV, FFO)
            if item_root(x.item) not in roots and _item_class(x.item, config) < 2:
                roots.add(item_root(x.item))
                fs.append(x)
        if len(fs) < config.min_items:
            continue
        low = fs[:k]
        # a weight above the first only when no item is heavily used: IND 2026-11-28, weight 2
        # on items incl. pre-tax profit (1,595 BRAIN alphas) raised PROD_CORRELATION 0.71 -> 0.81
        heavy = any(f.usage >= config.heavy_item_alphas for f in low)
        plan = [(low, w) for i, w in enumerate(config.weights) if i == 0 or not heavy]
        nxt = fs[k:2 * k]
        if len(nxt) >= config.min_items:
            plan.append((nxt, config.weights[0]))
        for items, w in plan:
            if tag(ds, items, w) in tried:
                continue
            out.append((ds, items, w))
    return out[:limit] if limit is not None else out


def tried_tags(hypotheses: Iterable[str]) -> set[str]:
    return {m.group(0) for h in hypotheses for m in TAG.finditer(h or "")}


def eligible(decision, result, failing: set[str], config: BreadthConfig,
             distance_fn) -> tuple[bool, str]:
    if not config.enabled:
        return False, "breadth blends off"
    hit = sorted(failing & set(config.classes))
    if not hit:
        return False, "no information class fails"
    dist = distance_fn(decision, result)
    if dist is None or dist > config.max_distance:
        return False, f"distance to accept {dist} > {config.max_distance}"
    return True, f"fails {', '.join(hit)}; distance {dist:.2f}"
