"""Which repair levers can close an alpha's gap, from our own measurements (owner 2026-11-27).

Why: in run rs20261127085437, 29 repair simulations gave one accept. The misses had a pattern
that memory already showed:

- Sharpe gaps. Over every repair of a Sharpe failure in memory, mechanical levers (decay,
  smoothing, tilts, universe, shape) changed Sharpe by x0.97 in the median and never by more
  than x1.07 (20 children); neutralizations by x1.10 (x1.3 or more only where BRAIN's
  risk-neutralized Sharpe was already high); blends by x1.77. A JPN alpha at Sharpe 1.10
  (needs x1.44) got 8 mechanical and neutralization children and none came close.
- Sub-universe. BRAIN's limit is a share of the alpha's own Sharpe (EUR TOP2500: 0.52 x
  Sharpe). A size tilt never raised the sub-universe Sharpe (13 children in memory, at most
  +0.06); it only cut the full Sharpe, so it can pass only when the sub-universe Sharpe alone
  already supports a passing alpha (>= limit share x Sharpe bar). EUR blends at Sharpe
  2.6-2.9 with sub-universe 0.65-0.77 lost Sharpe to 1.08 under a size tilt and still failed.
  A blend adds the partner's large-cap Sharpe but cannot rescue an alpha whose own large-cap
  part is near zero (EUR price-to-cash-flow part: ratio 0.09).

Method
1. ``blocked(lever, ...)`` says why a lever cannot close the gap, or None:
   a. Sharpe fails (not routed sign first), raw Sharpe S > 0 and the bar needs more than
      ``max_mechanical_sharpe_gain`` over S: mechanical levers are out; neutralizations stay
      only when BRAIN's risk-neutralized Sharpe is at least ``rn_support_share`` x the bar.
      Information levers (blends, part levers, model rewrites) stay.
   b. Sub-universe fails: the ratio levers (``tilt_levers``: size/liquidity tilt, a smaller
      universe, and combinations with them) only when sub-universe Sharpe >= limit share x
      bar x ``tilt_margin``. Other reshaping levers (decay, smoothing, rank, backfill, ...)
      only when the sub-universe Sharpe is within ``max_mechanical_sub_gain`` of that floor:
      over every sub-universe repair in memory only blends (median +0.48) and
      neutralizations (5 of 13 at +0.14 or more) raised the sub-universe Sharpe itself;
      reshapes stayed near 0 (best +0.22, 3 of 30 above +0.1).
   c. Sub-universe fails, the sub-universe Sharpe is below limit share x bar, and the alpha's
      own ratio (sub-universe / full Sharpe) is below ``blend_min_parent_ratio_share`` x the
      limit share: blends are out (the base has no large-cap part to blend onto).
2. ``measured_alone(fields, market)``: BRAIN results of our alphas in the market that read
   exactly these signal fields (base fields and groups aside), e.g. a partner tested alone.
   ``partner_check`` then refuses a partner whose measured sub-universe ratio is below the
   limit share when the alpha fails sub-universe.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from typing import Collection, Mapping

from .config import ConfigError
from .fastexpr import FastExprError, FastExprSpec, analyze

BLEND_LEVERS = frozenset({"blend_other_dataset", "blend_plan", "recipe_blend", "breadth_blend"})
PART_LEVERS = frozenset({"reweight_legs", "drop_weak_legs"})
NEUTRALIZING = ("neutralize_", "set_neutralization")


@dataclass(frozen=True)
class FeasibilityConfig:
    enabled: bool = False
    max_mechanical_sharpe_gain: float = 0.10
    rn_support_share: float = 0.9
    tilt_levers: tuple[str, ...] = ("size_tilt", "liquidity_tilt", "change_universe")
    tilt_margin: float = 1.0
    max_mechanical_sub_gain: float = 0.10
    blend_min_parent_ratio_share: float = 0.5
    # drop_weak_legs: a part is weak for sub-universe below this share of the limit share;
    # for Sharpe, below weak_part_sharpe_share x the best part, when the best part alone
    # has at least strong_part_share x the whole's Sharpe
    weak_part_ratio_share: float = 0.6
    weak_part_sharpe_share: float = 0.5
    strong_part_share: float = 0.9

    @classmethod
    def from_dict(cls, raw) -> "FeasibilityConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ConfigError("agents config: repair.feasibility must be a mapping")
        known = {f.name for f in dc_fields(cls)}
        extra = set(raw) - known
        if extra:
            raise ConfigError(f"agents config: repair.feasibility: unknown keys {sorted(extra)}")
        d = cls()
        out = {}
        for f in dc_fields(cls):
            v = raw.get(f.name, getattr(d, f.name))
            if f.name == "enabled":
                if not isinstance(v, bool):
                    raise ConfigError("agents config: repair.feasibility.enabled must be "
                                      "true or false")
            elif f.name == "tilt_levers":
                if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) for x in v):
                    raise ConfigError("agents config: repair.feasibility.tilt_levers must be "
                                      "a list of lever names")
                v = tuple(v)
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 10:
                raise ConfigError(f"agents config: repair.feasibility.{f.name} must be a "
                                  f"number in [0, 10]")
            out[f.name] = v
        return cls(**out)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def sharpe_bar(decision=None, result=None) -> float | None:
    """BRAIN's LOW_SHARPE limit for the alpha, else our Sharpe gate's threshold."""
    cv = ((getattr(result, "check_values", None) or {}).get("LOW_SHARPE") or {})
    if _num(cv.get("limit")):
        return float(cv["limit"])
    if decision is not None:
        for g in decision.gate_results:
            if g.name in ("sharpe", "brain:LOW_SHARPE") and _num(g.threshold):
                return float(g.threshold)
    return None


def ratio(result) -> float | None:
    """Sub-universe Sharpe / full Sharpe (positive full Sharpe only)."""
    if result is None or not _num(result.sharpe) or result.sharpe <= 0 \
            or not _num(result.sub_universe_sharpe):
        return None
    return result.sub_universe_sharpe / result.sharpe


def limit_share(result) -> float | None:
    """BRAIN's sub-universe limit as a share of the alpha's own Sharpe (EUR TOP2500: 0.52)."""
    if result is None or not _num(result.sharpe) or result.sharpe <= 0 \
            or not _num(result.sub_universe_cutoff) or result.sub_universe_cutoff <= 0:
        return None
    return result.sub_universe_cutoff / result.sharpe


def sub_floor(decision, result) -> float | None:
    """The smallest sub-universe Sharpe a passing alpha can have: limit share x Sharpe bar."""
    share, bar = limit_share(result), sharpe_bar(decision, result)
    return share * bar if share is not None and bar else None


def sub_failing(result) -> bool:
    return result is not None and _num(result.sub_universe_sharpe) \
        and _num(result.sub_universe_cutoff) \
        and result.sub_universe_sharpe < result.sub_universe_cutoff


def _risk_neutralized(result) -> float | None:
    rn = ((getattr(result, "alt_stats", None) or {}).get("risk_neutralized") or {})
    return float(rn["sharpe"]) if _num(rn.get("sharpe")) else None


def _parts(name: str) -> list[str]:
    return name[len("combo:"):].split("+") if name.startswith("combo:") else [name]


def information(name: str, mechanical: bool) -> bool:
    """A lever that brings new information (blend, part rewrite, model rewrite), as opposed
    to reshaping the same signal."""
    return name in BLEND_LEVERS or name in PART_LEVERS or not mechanical


def blocked(name: str, *, mechanical: bool, decision, result, failing: Collection[str],
            sign_first: bool, config: FeasibilityConfig) -> str | None:
    """Why ``name`` cannot close this alpha's gap (method step 1), or None."""
    if not config.enabled or result is None:
        return None
    parts = _parts(name)
    bar = sharpe_bar(decision, result)
    s = result.sharpe if _num(result.sharpe) else None
    if "sharpe" in failing and not sign_first and s is not None and s > 0 and bar \
            and bar / s - 1 > config.max_mechanical_sharpe_gain and name != "flip_sign" \
            and not information(name, mechanical):
        need = bar / s
        if any(p.startswith(NEUTRALIZING) for p in parts):
            rn = _risk_neutralized(result)
            if rn is None or rn < config.rn_support_share * bar:
                return (f"Sharpe needs x{need:.2f}; a neutralization helps that much only "
                        f"when the risk-neutralized Sharpe (here {rn}) is >= "
                        f"{config.rn_support_share:g} x bar {bar:g}")
        else:
            return (f"Sharpe needs x{need:.2f}; mechanical levers never raised Sharpe more "
                    f"than x{1 + config.max_mechanical_sharpe_gain:.2f}")
    if "sub_universe" in failing:
        floor = sub_floor(decision, result)
        sub = result.sub_universe_sharpe if _num(result.sub_universe_sharpe) else None
        if floor is not None and sub is not None:
            if any(p in config.tilt_levers for p in parts) \
                    and sub < floor * config.tilt_margin:
                return (f"sub-universe Sharpe {sub:.2f} < {floor:.2f} (limit share x bar): a "
                        f"tilt or smaller universe only cuts the full Sharpe, it never adds "
                        f"large-cap Sharpe")
            if not information(name, mechanical) and sub < floor \
                    and not any(p.startswith(NEUTRALIZING) for p in parts) \
                    and floor - sub > config.max_mechanical_sub_gain:
                return (f"sub-universe Sharpe {sub:.2f} is {floor - sub:.2f} below {floor:.2f} "
                        f"(limit share x bar); reshaping levers never raised it by more than "
                        f"~{config.max_mechanical_sub_gain:g}")
            r, share = ratio(result), limit_share(result)
            if name in BLEND_LEVERS and sub < floor and r is not None and share \
                    and r < config.blend_min_parent_ratio_share * share:
                return (f"the alpha's own sub-universe ratio {r:.2f} is below "
                        f"{config.blend_min_parent_ratio_share:g} x the limit share "
                        f"{share:.2f}: no large-cap part to blend onto")
    return None


# ---------------------------------------------------------------------------- step 2

_FIELDS: dict[str, frozenset] = {}


def signal_fields(expression: str, spec: FastExprSpec) -> frozenset:
    """Fields an expression reads, base fields and groups aside (cached)."""
    got = _FIELDS.get(expression)
    if got is None:
        try:
            used = analyze(expression, spec).fields_used
        except FastExprError:
            used = ()
        got = frozenset(f for f in used if f not in spec.base_fields and f not in spec.groups)
        _FIELDS[expression] = got
    return got


def measured_alone(fields: frozenset, region: str, delay: int, outcomes: list[dict],
                   spec: FastExprSpec) -> list:
    """Complete BRAIN results of our alphas in this market that read exactly ``fields``."""
    if not fields:
        return []
    out = []
    for o in outcomes:
        c, r = o["candidate"], o["result"]
        if c.region != region or c.delay != delay or r is None or r.status != "complete":
            continue
        if signal_fields(c.expression, spec) == fields:
            out.append(r)
    return out


def partner_check(partner: str, region: str, delay: int, outcomes: list[dict],
                  spec: FastExprSpec, result) -> str | None:
    """Why a partner cannot fix this alpha's sub-universe failure, from its measured
    sub-universe ratio, or None (also when it was never measured)."""
    if not sub_failing(result):
        return None
    share = limit_share(result)
    if share is None:
        return None
    rows = measured_alone(signal_fields(partner, spec), region, delay, outcomes, spec)
    ratios = [x for r in rows if (x := ratio(r)) is not None]
    if ratios and max(ratios) < share:
        return (f"partner measured alone has sub-universe ratio {max(ratios):.2f} < limit "
                f"share {share:.2f}")
    return None


def best_alone(fields: frozenset, region: str, delay: int, outcomes: list[dict],
               spec: FastExprSpec) -> float | None:
    """Best |Sharpe| of our alphas in the market reading exactly ``fields``, or None."""
    vals = [abs(r.sharpe) for r in measured_alone(fields, region, delay, outcomes, spec)
            if _num(r.sharpe)]
    return max(vals) if vals else None


__all__ = ["FeasibilityConfig", "blocked", "information", "sharpe_bar", "ratio",
           "limit_share", "sub_floor", "sub_failing", "signal_fields", "measured_alone",
           "partner_check", "best_alone", "BLEND_LEVERS", "PART_LEVERS"]
