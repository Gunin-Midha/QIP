"""Parts of a combined alpha, tested alone, and accepted blends reused as recipes (owner
2026-11-26).

Why: in run rs20261126082238 the IND sales-dispersion alpha (70% analyst dispersion, 30%
sector-relative return reversal) failed BRAIN's robust-universe check; the fixes that help
(drop or replace the weak part) were never tried, because no lever splits an alpha into its
parts and the model only sees the whole alpha's numbers (it raised the reversal's weight).
And the GBR overnight-intraday reversal failed fitness while the same reversal had passed in
EUR blended with analyst revision breadth: an accepted recipe that no lever reuses.

Method
1. ``split_legs``: the final value, followed through its variables, unwrapped from one
   rank/group_rank, is a sum or difference of terms (constant weights dropped, signs kept) or
   a product of non-constant factors. Each term with at least one field is a part; a factor
   built only from base fields with a size/liquidity shape (x (0.25 + rank(cap))) is a tilt,
   not a part. 2..``max_legs`` parts, else none. Each part is rendered as a complete alpha:
   the statements it needs (unused ones dropped: BRAIN refuses unused variables), wrapped in
   the same rank/group_rank as the original.
2. ``leg_children``: the Repair Agent adds one child per part (lever ``leg_alone``) to a
   wave, on top of the wave's fixes, so BRAIN measures each part alone in the alpha's
   settings. A part that is closer to accept than the whole is a normal next-wave start
   (dropping the other parts).
3. ``part_results``: for each part of an alpha, the latest BRAIN result of a ``leg_alone``
   child with the same normalized expression (any ancestor's test counts). The model levers
   see them as ``<parts>``; ``reweight_legs`` (a model lever) applies only when at least two
   parts were measured.
4. ``recipes``: accepted alphas that combine parts from different datasets (or were built by
   a blend lever). Each yields its partner part (the part from a dataset other than the
   alpha's own), the market, the class its parent failed, and BRAIN's numbers. Same-market
   recipes are applied mechanically (``recipe_blend``: the partner blended into the failing
   alpha by ``build_blend``); every recipe is shown to the blend levers as ``<recipes>`` so the
   model can write a market's own version of a recipe that passed elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping

from .fastexpr import Assign, Binary, Call, FastExprError, Name, Num, Unary, parse, render
from .schemas import normalize_expression

LEG_LEVER = "leg_alone"
REWEIGHT_LEVER = "reweight_legs"
RECIPE_LEVER = "recipe_blend"
BLEND_LEVERS = ("blend_other_dataset", "blend_plan", RECIPE_LEVER)
WRAPPERS = ("rank", "group_rank")


def _names(node, out: set) -> set:
    if isinstance(node, Name):
        out.add(node.name)
    elif isinstance(node, Call):
        for a in node.args:
            _names(a, out)
        for _, v in node.kwargs:
            _names(v, out)
    elif isinstance(node, Unary):
        _names(node.operand, out)
    elif isinstance(node, Binary):
        _names(node.left, out)
        _names(node.right, out)
    elif isinstance(node, Assign):
        _names(node.value, out)
    return out


def _resolve(node, env: dict, depth: int = 0):
    """A variable's value (followed while it is a plain name)."""
    while isinstance(node, Name) and node.name in env and depth < 20:
        node, depth = env[node.name], depth + 1
    return node


def _strip(node, env: dict):
    """(sign, core) of a term: unary minus flips the sign, constant weights are dropped."""
    sign = 1
    for _ in range(20):
        node = _resolve(node, env)
        if isinstance(node, Unary) and node.op == "-":
            sign, node = -sign, node.operand
        elif isinstance(node, Binary) and node.op == "*" and isinstance(node.left, (Num, Unary)) \
                and _is_const(node.left):
            sign, node = sign * _const_sign(node.left), node.right
        elif isinstance(node, Binary) and node.op in ("*", "/") and _is_const(node.right):
            sign, node = sign * _const_sign(node.right), node.left
        else:
            break
    return sign, node


def _is_const(node) -> bool:
    if isinstance(node, Num):
        return True
    return isinstance(node, Unary) and node.op == "-" and isinstance(node.operand, Num)


def _const_sign(node) -> int:
    return -1 if isinstance(node, Unary) else 1


def _terms(node, env: dict, sign: int, out: list) -> None:
    """Flatten a sum/difference into (sign, term)."""
    node = _resolve(node, env)
    if isinstance(node, Binary) and node.op in ("+", "-"):
        _terms(node.left, env, sign, out)
        _terms(node.right, env, sign if node.op == "+" else -sign, out)
        return
    if isinstance(node, Unary) and node.op == "-":
        inner = _resolve(node.operand, env)
        if isinstance(inner, Binary) and inner.op in ("+", "-"):
            _terms(inner, env, -sign, out)
            return
    s, core = _strip(node, env)
    out.append((sign * s, core))


def _is_tilt(node, env: dict, base_fields) -> bool:
    """A factor built only from base fields (size, liquidity): a tilt, not a part."""
    names = _all_fields(node, env)
    return bool(names) and names <= set(base_fields)


def _all_fields(node, env: dict, seen: frozenset = frozenset()) -> set:
    out = set()
    for n in _names(node, set()):
        if n in env and n not in seen:
            out |= _all_fields(env[n], env, seen | {n})
        elif n not in env:
            out.add(n)
    return out


def _factors(node, env: dict, out: list) -> None:
    node = _resolve(node, env)
    if isinstance(node, Binary) and node.op == "*" and not _is_const(node.left) \
            and not _is_const(node.right):
        _factors(node.left, env, out)
        _factors(node.right, env, out)
        return
    out.append(node)


def _parts(final, env: dict, spec) -> tuple[list[tuple[int, object]], list]:
    """(signed parts, tilts) of the final value."""
    sign, node = _strip(final, env)
    factors: list = []
    _factors(node, env, factors)
    tilts = [f for f in factors if _is_tilt(f, env, spec.base_fields)]
    core = [f for f in factors if f not in tilts]
    if len(core) >= 2:   # a product of signals: each factor is a part
        return [(sign, f) for f in core], tilts
    if len(core) != 1:
        return [], tilts
    node = _resolve(core[0], env)
    wrap = None
    if isinstance(node, Call) and node.name in WRAPPERS and node.args:
        wrap, node = node, node.args[0]
    terms: list = []
    _terms(node, env, sign, terms)
    if len(terms) < 2 and wrap is None:
        return [], tilts
    if wrap is not None:
        terms = [(s, Call(wrap.name, (t,) + wrap.args[1:], wrap.kwargs)) for s, t in terms]
    return terms, tilts


def _prune(head: tuple, final) -> tuple:
    """Statements the final value needs, in order (a variable no later statement reads is
    an error on BRAIN)."""
    need = _names(final, set())
    kept = []
    for s in reversed(head):
        if isinstance(s, Assign) and s.name in need:
            kept.append(s)
            need |= _names(s.value, set())
    return tuple(reversed(kept))


def split_legs(expression: str, spec, max_legs: int = 3) -> list[str]:
    """The alpha's parts as complete expressions (method step 1); [] when it has fewer than
    two parts or more than ``max_legs``."""
    try:
        stmts = parse(expression)
    except FastExprError:
        return []
    head, final = stmts[:-1], stmts[-1]
    env = {s.name: s.value for s in head if isinstance(s, Assign)}
    parts, _ = _parts(final, env, spec)
    parts = [(s, p) for s, p in parts if _all_fields(p, env)]
    if not 2 <= len(parts) <= max_legs:
        return []
    out = []
    for s, p in parts:
        leg = p if s > 0 else Unary("-", p)
        text = render(_prune(head, leg) + (leg,))
        if text not in out:
            out.append(text)
    return out if len(out) >= 2 else []


def _norm(expression: str) -> str:
    try:
        return normalize_expression(render(parse(expression)))
    except FastExprError:
        return normalize_expression(expression)


def part_results(candidate, outcomes: list[dict], spec, max_legs: int = 3) -> list[dict]:
    """Each part of ``candidate`` with the latest BRAIN result of a ``leg_alone`` test of
    the same expression (method step 3): [{"part", "result" or None, "settings"}]."""
    legs = split_legs(candidate.expression, spec, max_legs)
    if not legs:
        return []
    tests: dict[str, dict] = {}
    for o in outcomes:
        c, r = o["candidate"], o["result"]
        if c.lever == LEG_LEVER and r is not None and r.status == "complete":
            tests[_norm(c.expression)] = o   # outcomes are in order: the latest wins
    rows = []
    for leg in legs:
        o = tests.get(_norm(leg))
        rows.append({"part": leg, "candidate_id": o["candidate"].candidate_id if o else None,
                     "result": o["result"] if o else None,
                     "settings": (f"{o['candidate'].neutralization}, decay {o['candidate'].decay}"
                                  if o else None)})
    return rows


def part_lines(rows: list[dict], decision_of=None) -> list[str]:
    """``<parts>`` lines for the repair model."""
    out = []
    for i, row in enumerate(rows, 1):
        r = row["result"]
        if r is None:
            out.append(f"part {i}: {row['part']} | alone: not measured yet")
            continue
        bits = [f"Sharpe {r.sharpe}", f"fitness {r.fitness}", f"turnover {r.turnover}"]
        if r.sub_universe_sharpe is not None:
            bits.append(f"sub-universe {r.sub_universe_sharpe}")
        for name, cv in (r.check_values or {}).items():
            if r.checks.get(name) is False and isinstance(cv, Mapping):
                bits.append(f"{name} {cv.get('value')} (limit {cv.get('limit')})")
        for name, stats in (r.alt_stats or {}).items():
            if stats.get("sharpe") is not None:
                bits.append(f"{name.replace('_', ' ')} Sharpe {stats['sharpe']}")
        if r.yearly:
            weak = [f"{y.year} {y.sharpe}" for y in r.yearly
                    if y.sharpe is not None and y.sharpe < 0.5]
            if weak:
                bits.append("weak years " + ", ".join(weak))
        out.append(f"part {i}: {row['part']} | alone ({row['settings']}): " + ", ".join(bits))
    return out


# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------

def _unfill(node, env: dict):
    """The partner signal under the blend's fill and rank: if_else(is_nan(b), 0.5, b) -> b,
    rank(x) / group_rank(x, g) -> x."""
    node = _resolve(node, env)
    if isinstance(node, Call) and node.name == "if_else" and len(node.args) == 3:
        node = _resolve(node.args[2], env)
    if isinstance(node, Call) and node.name in WRAPPERS and node.args:
        node = node.args[0]
    return node


def _datasets(fields: set, meta: Mapping, spec) -> set:
    return {(meta.get(f) or {}).get("dataset") for f in fields
            if f not in spec.base_fields and f not in spec.groups} - {None}


def recipe_from(candidate, result, parent_class, meta: Mapping, spec) -> dict | None:
    """The partner part of an accepted blend (method step 4), or None."""
    try:
        stmts = parse(candidate.expression)
    except FastExprError:
        return None
    head, final = stmts[:-1], stmts[-1]
    env = {s.name: s.value for s in head if isinstance(s, Assign)}
    parts, tilts = _parts(final, env, spec)
    own = candidate.dataset
    found = []
    for s, p in parts:
        fields = _all_fields(p, env)
        ds = _datasets(fields, meta, spec)
        if ds and own not in ds:
            found.append((s, p, ds))
    if not found or len(parts) < 2:
        return None
    s, p, ds = found[0]
    core = _unfill(p, env)
    partner = core if s > 0 else Unary("-", core)
    text = render(_prune(head, partner) + (partner,))
    dataset = sorted(ds)[0]
    return {"candidate_id": candidate.candidate_id, "region": candidate.region,
            "delay": candidate.delay, "base_dataset": own, "partner_dataset": dataset,
            "partner_category": (meta.get(next(iter(_all_fields(p, env) - set(spec.base_fields)),
                                               "")) or {}).get("category"),
            "partner": text, "tilted": bool(tilts), "fixed": parent_class,
            "sharpe": result.sharpe if result else None,
            "fitness": result.fitness if result else None,
            "prod_correlation": (((result.check_values or {}).get("PROD_CORRELATION") or {})
                                 .get("value") if result else None)}


def recipes(outcomes: list[dict], meta_for, spec) -> list[dict]:
    """Every accepted alpha's recipe (method step 4), best Sharpe first. ``meta_for(region,
    delay)`` gives that market's catalog meta."""
    by_id = {o["candidate"].candidate_id: o for o in outcomes}
    out = []
    for o in outcomes:
        c, r, d = o["candidate"], o["result"], o["decision"]
        if d is None or d.decision != "accept":
            continue
        parent = by_id.get(c.parent_id) if c.parent_id else None
        pclass = parent["decision"].failure_class if parent and parent["decision"] else None
        rec = recipe_from(c, r, pclass, meta_for(c.region, c.delay) or {}, spec)
        if rec is not None and rec not in out:
            out.append(rec)
    out.sort(key=lambda x: -(x["sharpe"] or 0))
    return out


def recipe_lines(rows: list[dict], limit: int = 6) -> list[str]:
    """``<recipes>`` lines for the blend levers."""
    out = []
    for x in rows[:limit]:
        prod = f", PROD {x['prod_correlation']}" if x["prod_correlation"] is not None else ""
        out.append(f"{x['region']} D{x['delay']}: {x['base_dataset']} alpha + partner from "
                   f"{x['partner_dataset']} ({x['partner_category']})"
                   f"{' x size/liquidity tilt' if x['tilted'] else ''}; the parent failed "
                   f"{x['fixed'] or 'unknown'}; accepted at Sharpe {x['sharpe']}, fitness "
                   f"{x['fitness']}{prod}. Partner: {x['partner']}")
    return out


def tried_partner_texts(candidate, by_id: Mapping[str, dict]) -> set[str]:
    """Normalized partner texts already blended along the candidate's lineage."""
    out, c = set(), candidate
    for _ in range(50):
        if c is None:
            break
        if c.lever == RECIPE_LEVER and c.hypothesis and "[recipe " in c.hypothesis:
            out.add(c.hypothesis.split("[recipe ", 1)[1].split("]", 1)[0])
        o = by_id.get(c.parent_id) if c.parent_id else None
        c = o["candidate"] if o else None
    return out


__all__ = ["LEG_LEVER", "REWEIGHT_LEVER", "RECIPE_LEVER", "split_legs", "part_results",
           "part_lines", "recipes", "recipe_lines", "recipe_from", "tried_partner_texts"]
