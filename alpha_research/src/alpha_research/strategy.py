"""Strategy Evolution: learns from memory what works and changes how alphas are generated.

Method
1. ``summarize`` reads every decided candidate in memory (``MemoryStore.candidate_outcomes``)
   and counts, per family and per (dataset, region, prompt_version) arm, how many were
   decided, accepted and why the rest failed. It keeps the best accepted alphas (by Sharpe)
   and a few recent failures with their key metrics. No model.
   It also lists, per dataset, how much has been tried (``datasets``), the known datasets
   with no or few trials (``least_tried``: the plan's and the catalog's, passed in as
   ``known_datasets``), and the repair levers that worked (``what_fixes_worked``, from
   lever_stats.py) as construction techniques.
2. ``evolve`` turns the summary into guidance for the Generator and, with the model, new
   hypotheses to test. Diversity (owner 2026-11-04): the prompt asks for at least half the
   hypotheses on least-tried datasets and at most one building on an accepted alpha, and
   guidance may teach construction techniques but never narrows the datasets or families. Without the model (or when its reply is unusable) the guidance is
   built from the counts alone and no hypotheses are proposed.
   Per region (consultant level, 2026-11-11): the summary splits the counts by region
   (``regions``: decided, accepted, top failures, and how each neutralization fared), and
   the model may give ``region_guidance`` per region, which only that region's new arms
   get, because what works in one region may not work in another. ``region_notes`` writes
   the same evidence for one region as plain lines without a model; the research loop
   gives them to that region's Generator every round.
3. Each evolution is a new prompt version ``<base>+evo<k>``. The Research Director adds it
   as new arms next to the old ones, so the bandit measures whether the guidance helps
   instead of assuming it does.

Memory text (hypotheses, expressions) may come from papers or from a model, so it is passed
to the model inside <records> tags marked as data. The prompts are in prompts.py.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .fastexpr import analyze
from .generator import GenerationError, ask_json, dataset_appeal
from .lever_stats import LeverStats, repair_events
from .llm import LLMClient, ModelDown
from .prompts import BAR_CHECKS, strategy_system, strategy_user

@dataclass(frozen=True)
class StrategyConfig:
    every_rounds: int = 2
    max_hypotheses: int = 3
    max_guidance_chars: int = 1500
    tag: str = "evo"
    # arms for datasets no arm covers, per evolution (2026-11-14: one evolution added 140
    # arms for a 20-simulation run, so the Director could not learn which were good)
    max_new_arms: int = 8
    # whole prompt (system + records) cut to about this many characters by trimmed_summary;
    # None = no cut. Run rs20261125130750: an 11,000-token Strategy request was refused by
    # both Groq models (8,000 and 7,000 tokens a minute) and took both down.
    max_prompt_chars: int | None = None


@dataclass
class Evolution:
    prompt_version: str
    guidance: str
    hypotheses: list[tuple[str | None, str]] = field(default_factory=list)  # (dataset, text)
    source: str = "rules"
    notes: list[str] = field(default_factory=list)
    llm_calls: int = 0
    region_guidance: dict[str, str] = field(default_factory=dict)

    def guidance_for(self, region: str) -> str:
        """The guidance plus this region's own part, if the model gave one."""
        extra = self.region_guidance.get(region)
        return f"{self.guidance}\nIn {region}:\n{extra}" if extra else self.guidance


FEW_TRIALS = 3   # a dataset with fewer decided alphas than this counts as least tried


def summarize(outcomes: list[dict], *, top_n: int = 5,
              known_datasets: list[str] | None = None,
              dataset_info: dict[str, dict] | None = None,
              submissions: dict | None = None) -> dict:
    """``dataset_info``: {dataset: {category, value_score, alpha_count}} from BRAIN's dataset
    lists (``generator.dataset_facts``). Least-tried datasets are then ordered by our trials,
    then least crowded on BRAIN (``generator.dataset_appeal``), and their facts are shown as
    ``least_tried_facts`` (BRAIN's tip for production correlation, 2026-11-25)."""
    info = dataset_info or {}
    by_family: dict = defaultdict(lambda: {"decided": 0, "accepted": 0, "failures": Counter()})
    by_dataset: Counter = Counter()
    by_arm: dict = defaultdict(lambda: {"decided": 0, "accepted": 0})
    failures = Counter()
    accepted, failed = [], []
    for row in outcomes:
        c, d, r = row["candidate"], row["decision"], row["result"]
        if d is None:
            continue
        fam = by_family[c.family or "unlabelled"]
        arm = by_arm[f"{c.dataset or 'unknown'}/{c.region}/{c.prompt_version or 'none'}"]
        fam["decided"] += 1
        arm["decided"] += 1
        by_dataset[c.dataset or "unknown"] += 1
        metrics = {} if r is None else {k: round(v, 4) for k in ("sharpe", "fitness", "turnover")
                                         if (v := getattr(r, k)) is not None}
        if d.decision == "accept":
            fam["accepted"] += 1
            arm["accepted"] += 1
            accepted.append({"expression": c.expression, "family": c.family, **metrics})
        else:
            fam["failures"][d.failure_class] += 1
            failures[d.failure_class] += 1
            failed.append({"expression": c.expression, "decision": d.decision,
                           "failure_class": d.failure_class, **metrics})
    accepted.sort(key=lambda a: a.get("sharpe") or 0, reverse=True)
    known = sorted(set(known_datasets or []) | set(by_dataset))
    least = sorted((d for d in known if by_dataset[d] < FEW_TRIALS),
                   key=lambda d: (by_dataset[d], dataset_appeal(info.get(d)), d))[:20]
    least_facts = {d: {"our_trials": by_dataset[d], "category": info[d].get("category"),
                       "value_score": info[d].get("value_score"),
                       "brain_alphas": info[d].get("alpha_count")}
                   for d in least if d in info}
    worked = [r for r in LeverStats.from_events(repair_events(outcomes)).table()
              if r["posterior_mean"] > 0.5]
    return {
        "regions": region_breakdown(outcomes),
        "decided": sum(f["decided"] for f in by_family.values()),
        "accepted": sum(f["accepted"] for f in by_family.values()),
        "failure_classes": dict(failures.most_common()),
        "families": {k: {"decided": v["decided"], "accepted": v["accepted"],
                         "failures": dict(v["failures"].most_common(3))}
                     for k, v in sorted(by_family.items())},
        "arms": dict(sorted(by_arm.items())),
        "best_accepted": accepted[:top_n],
        "recent_failures": failed[-top_n:],
        "datasets": dict(by_dataset.most_common()),
        "least_tried": least,
        **({"least_tried_facts": least_facts} if least_facts else {}),
        **({"our_brain_submissions": submissions_view(submissions)} if submissions else {}),
        "what_fixes_worked": [{k: r[k] for k in ("failure_class", "lever", "fixed", "posterior_mean")}
                              for r in worked[:8]],
    }


def trimmed_summary(summary: dict, max_chars: int) -> dict:
    """``summary`` cut until its compact JSON is at most ``max_chars`` characters, least
    useful parts first: older recent failures (2 kept), the arms and families with the
    fewest decided alphas (5 kept), each region's dataset list (top 5), the weaker best
    accepted (2 kept), lever records (3 kept), the dataset counts (top 10). Totals, failure
    classes, least-tried datasets and submissions are never cut. Returns a new dict; the
    result can stay larger than ``max_chars`` when nothing more can go."""
    s = json.loads(json.dumps(summary, default=str))

    def size() -> int:
        return len(json.dumps(s, separators=(",", ":"), default=str))

    def decided(v) -> float:
        return v.get("decided", 0) if isinstance(v, dict) else 0

    steps = [
        lambda: len(s.get("recent_failures", [])) > 2 and s["recent_failures"].pop(0),
        lambda: len(s.get("arms", {})) > 5 and s["arms"].pop(
            min(s["arms"], key=lambda k: (decided(s["arms"][k]), k))),
        lambda: len(s.get("families", {})) > 5 and s["families"].pop(
            min(s["families"], key=lambda k: (decided(s["families"][k]), k))),
        lambda: _cut_region_datasets(s, 5),
        lambda: len(s.get("best_accepted", [])) > 2 and s["best_accepted"].pop(),
        lambda: len(s.get("what_fixes_worked", [])) > 3 and s["what_fixes_worked"].pop(),
        lambda: len(s.get("datasets", {})) > 10 and s["datasets"].pop(list(s["datasets"])[-1]),
    ]
    for step in steps:
        while size() > max_chars and step():
            pass
    return s


def _cut_region_datasets(s: dict, keep: int) -> bool:
    """Cut each region's dataset counts to its ``keep`` largest; False when none had more."""
    cut = False
    for reg in (s.get("regions") or {}).values():
        ds = reg.get("datasets") if isinstance(reg, dict) else None
        if isinstance(ds, dict) and len(ds) > keep:
            reg["datasets"] = dict(sorted(ds.items(), key=lambda kv: -kv[1])[:keep])
            cut = True
    return cut


def submissions_view(submissions: dict) -> dict:
    """BRAIN's count of our submitted alphas (``brain_api.submission_diversity``) as the
    Strategy summary shows it: the total, per region, and per region the data categories
    with their counts (an alpha counts in each category it uses)."""
    cats: dict = {}
    for row in submissions.get("categories") or []:
        cats.setdefault(row.get("region"), {})[row.get("category")] = row.get("alphas")
    return {"total": submissions.get("total"), "by_region": dict(submissions.get("regions") or {}),
            "categories_by_region": cats}


def region_breakdown(outcomes: list[dict]) -> dict:
    """Per region: decided, accepted, top failure classes, datasets tried and how each
    neutralization fared ({neutralization: [decided, accepted]})."""
    regions: dict = defaultdict(lambda: {"decided": 0, "accepted": 0, "failures": Counter(),
                                         "datasets": Counter(), "neutralizations": {}})
    for row in outcomes:
        c, d = row["candidate"], row["decision"]
        if d is None:
            continue
        reg = regions[c.region]
        ok = d.decision == "accept"
        reg["decided"] += 1
        reg["accepted"] += ok
        reg["datasets"][c.dataset or "unknown"] += 1
        if not ok:
            reg["failures"][d.failure_class] += 1
        n = reg["neutralizations"].setdefault(c.neutralization, [0, 0])
        n[0] += 1
        n[1] += ok
    return {r: {"decided": v["decided"], "accepted": v["accepted"],
                "failures": dict(v["failures"].most_common(3)),
                "datasets": len(v["datasets"]),
                "neutralizations": dict(sorted(v["neutralizations"].items()))}
            for r, v in sorted(regions.items())}


def brain_limits(outcomes: list[dict]) -> dict[tuple, dict[str, float]]:
    """{(region, delay): {check: limit}}: BRAIN's own limits as reported with its checks on
    our results (the latest seen). They differ a lot by market: JPN/AMR delay 0 need Sharpe
    2.69 and fitness 1.5, CHN delay 0 Sharpe 3.49 (run rs20261112054223), USA delay 1 1.58."""
    out: dict[tuple, dict[str, float]] = {}
    for row in outcomes:
        c, r = row["candidate"], row["result"]
        for check, v in ((r.check_values or {}).items() if r is not None else ()):
            if check in BAR_CHECKS and isinstance(v, dict) and v.get("limit") is not None:
                out.setdefault((c.region, c.delay), {})[check] = v["limit"]
    return out


def region_notes(outcomes: list[dict], region: str, *, max_levers: int = 3) -> str | None:
    """Plain evidence lines for one region's Generator, from memory; None with no history."""
    info = region_breakdown(outcomes).get(region)
    if not info:
        return None
    lines = [f"- In {region} so far: {info['accepted']} of {info['decided']} decided alphas "
             f"accepted, over {info['datasets']} datasets."]
    if info["failures"]:
        lines.append("- Most common failures there: " + ", ".join(
            f"{k} ({v})" for k, v in info["failures"].items()) + ".")
    tried = [f"{n} {acc}/{dec}" for n, (dec, acc) in info["neutralizations"].items()]
    lines.append("- Neutralizations tried there (accepted/decided): " + ", ".join(tried) + ".")
    stats = LeverStats.from_events(repair_events(outcomes))
    rows = [r for r in stats.region_table() if r["region"] == region]
    good = [r for r in rows if r["fixed"] > 0]
    for r in good[:max_levers]:
        lines.append(f"- Repair that fixed {r['failure_class']} there: {r['lever']}.")
    neut = [r for r in rows if "neutraliz" in r["lever"]]
    if neut:  # what changing the neutralization did to failing alphas in this region
        lines.append("- Neutralization changes by the Repair Agent there (fixed/improved/tried, "
                     "for which failure): " + ", ".join(
                         f"{r['lever']} {r['fixed']}/{r['improved']}/{r['events']} "
                         f"({r['failure_class']})" for r in neut) + ".")
    return "\n".join(lines)


def prod_correlation_fields(outcomes: list[dict], region: str, spec,
                            datafields=None) -> dict[str, int]:
    """{datafield: our simulated alphas in ``region`` that use it and failed BRAIN's
    PROD_CORRELATION check} from memory (latest result per candidate). The Generator treats
    these fields as crowded: their ideas are already in production. Base fields (close,
    adv20, ...) are left out: they appear in most alphas, so they say nothing about the idea."""
    out: Counter = Counter()
    for row in outcomes:
        c, r = row["candidate"], row.get("result")
        if c.region != region or r is None or r.checks.get("PROD_CORRELATION") is not False:
            continue
        used = analyze(c.expression, spec, datafields=datafields).fields_used
        out.update(f for f in used if f not in spec.base_fields)
    return dict(out)


def _best_abs_sharpe(r) -> float | None:
    """Best of |Sharpe| and |risk-neutralized Sharpe| of a complete result, else None."""
    if r is None or r.status != "complete":
        return None
    vals = [abs(r.sharpe)] if isinstance(r.sharpe, (int, float)) else []
    rn = ((getattr(r, "alt_stats", None) or {}).get("risk_neutralized") or {}).get("sharpe")
    if isinstance(rn, (int, float)):
        vals.append(abs(rn))
    return max(vals) if vals else None


def signal_evidence(outcomes: list[dict], region: str, delay: int, spec,
                    datafields=None) -> dict[str, float]:
    """{datafield: best |Sharpe| (raw or risk-neutralized, either sign) of our complete
    alphas in this market that use it} from memory. A field in a multi-field alpha shares
    that alpha's credit, so this is a hint of signal, not a measurement of the field alone.
    The Generator offers the strong ones of other datasets as blend partners. Base fields
    are left out."""
    out: dict[str, float] = {}
    for row in outcomes:
        c = row["candidate"]
        if c.region != region or c.delay != delay:
            continue
        best = _best_abs_sharpe(row.get("result"))
        if best is None:
            continue
        try:
            used = analyze(c.expression, spec, datafields=datafields).fields_used
        except Exception:   # an old expression the current grammar no longer reads
            continue
        for f in used:
            if f not in spec.base_fields and best > out.get(f, -1.0):
                out[f] = best
    return out


def arm_history(outcomes: list[dict], dataset: str | None, region: str, delay: int,
                limit: int, max_chars: int = 160) -> list[str]:
    """The latest ``limit`` simulated alphas of an arm's dataset in this market, newest
    first, one line each: expression (cut to ``max_chars``), BRAIN's Sharpe (and
    risk-neutralized Sharpe), fitness, turnover, production correlation when measured, and
    our decision. For the Generator, so it stops repeating forms that failed."""
    if limit <= 0:
        return []
    lines = []
    for row in reversed(outcomes):
        c, r, d = row["candidate"], row.get("result"), row.get("decision")
        if (c.dataset != dataset or c.region != region or c.delay != delay
                or r is None or r.status != "complete"):
            continue
        expr = " ".join(c.expression.split())
        expr = expr if len(expr) <= max_chars else expr[:max_chars - 3] + "..."
        bits = []
        if isinstance(r.sharpe, (int, float)):
            bits.append(f"Sharpe {r.sharpe:.2f}")
        rn = ((r.alt_stats or {}).get("risk_neutralized") or {}).get("sharpe")
        if isinstance(rn, (int, float)):
            bits.append(f"risk-neutralized {rn:.2f}")
        if isinstance(r.fitness, (int, float)):
            bits.append(f"fitness {r.fitness:.2f}")
        if isinstance(r.turnover, (int, float)):
            bits.append(f"turnover {r.turnover * 100:.0f}%")
        prod = ((r.check_values or {}).get("PROD_CORRELATION") or {}).get("value")
        if isinstance(prod, (int, float)):
            bits.append(f"PROD {prod:.2f}")
        if d is not None:
            bits.append(f"{d.decision}" + (f" / {d.failure_class}"
                                           if d.failure_class not in (None, "none") else ""))
        lines.append(f"- {expr} | {', '.join(bits)}")
        if len(lines) == limit:
            break
    return lines


_FAILURE_ADVICE = {
    "turnover": "turnover is the most common failure: prefer longer windows, decay, or smoothing (ts_mean, ts_decay_linear)",
    "sharpe": "Sharpe is the most common failure: the signals are too weak; combine related fields or condition on liquidity",
    "fitness": "fitness is the most common failure: returns are small relative to turnover; slow the signal down",
    "self_correlation": "self-correlation is the most common failure: the pool already has these ideas; use different datasets or operators",
    "sub_universe": "sub-universe Sharpe is the most common failure: the signal only works in illiquid names; prefer liquid fields",
    "concentration": "weight concentration is the most common failure: rank or winsorize before combining",
    "drawdown": "drawdown is the most common failure: volatility-scale the signal",
    "recent_sharpe": "recent-years Sharpe is the most common failure: the signals have faded lately; prefer less crowded fields and test robustness to the recent regime",
    "prod_correlation": "correlation with BRAIN's production alphas is the most common failure: the ideas are too well known; use less common fields, group-relative forms or conditioning",
    "incomplete": "many simulations were incomplete: check expressions and datafields before simulating",
}


def rule_guidance(summary: dict) -> str:
    lines = [f"- {summary['accepted']} of {summary['decided']} decided alphas were accepted so far."]
    if summary["failure_classes"]:
        top = next(iter(summary["failure_classes"]))
        lines.append(f"- {_FAILURE_ADVICE.get(top, f'most common failure: {top}')}.")
    good = [f for f, v in summary["families"].items() if v["accepted"]]
    if good:
        lines.append(f"- Families with accepted alphas: {', '.join(good)}.")
    if summary["best_accepted"]:
        lines.append(f"- Best accepted so far: {summary['best_accepted'][0]['expression']}")
    if summary.get("least_tried"):
        lines.append(f"- Keep exploring: {len(summary['least_tried'])} datasets have fewer than "
                     f"{FEW_TRIALS} trials, e.g. {', '.join(summary['least_tried'][:5])}. "
                     "Few accepts so far is too little evidence to narrow the search.")
    for r in summary.get("what_fixes_worked", [])[:3]:
        lines.append(f"- Construction that fixed {r['failure_class']} before: {r['lever']}.")
    return "\n".join(lines)


class StrategyEvolution:
    def __init__(self, config: StrategyConfig, *, llm: LLMClient | None = None,
                 max_output: int = 2048, context: str = ""):
        self.config = config
        self.context = context  # prompts.brain_context(config)
        self.llm = llm
        self.max_output = max_output
        self.count = 0

    def due(self, round_number: int) -> bool:
        return self.config.every_rounds > 0 and round_number % self.config.every_rounds == 0

    def evolve(self, outcomes: list[dict], base_prompt_version: str,
               known_datasets: list[str] | None = None,
               dataset_info: dict[str, dict] | None = None,
               submissions: dict | None = None) -> Evolution | None:
        summary = summarize(outcomes, known_datasets=known_datasets, dataset_info=dataset_info,
                            submissions=submissions)
        if summary["decided"] == 0:
            return None
        self.count += 1
        evo = Evolution(f"{base_prompt_version}+{self.config.tag}{self.count}",
                        rule_guidance(summary))
        if self.llm is not None:
            self._ask(summary, evo)
        evo.guidance = evo.guidance[: self.config.max_guidance_chars]
        return evo

    def _ask(self, summary: dict, evo: Evolution) -> None:
        system = strategy_system(self.context)
        user = strategy_user(summary, self.config.max_hypotheses)
        cap = self.config.max_prompt_chars
        if cap is not None and len(system) + len(user) > cap:
            before = len(system) + len(user)
            user = strategy_user(trimmed_summary(summary, max(cap - len(system) - 200, 0)),
                                 self.config.max_hypotheses)
            evo.notes.append(f"records trimmed: prompt {before:,} -> "
                             f"{len(system) + len(user):,} characters (cap {cap:,})")
        try:
            reply, calls = ask_json(self.llm, system, user, max_output=self.max_output)
        except GenerationError as exc:
            evo.llm_calls += 2
            evo.notes.append(f"strategy model reply unusable, rule guidance used: {exc}")
            return
        except ModelDown as exc:
            evo.notes.append(f"strategy model down, rule guidance used: {exc}")
            return
        evo.llm_calls += calls
        if not isinstance(reply, dict):
            evo.notes.append("strategy reply was not an object, rule guidance used")
            return
        rg = reply.get("region_guidance")
        if isinstance(rg, dict):
            for region, text in rg.items():
                if isinstance(text, list):
                    text = "\n".join(f"- {x}" for x in text if isinstance(x, str))
                if isinstance(region, str) and isinstance(text, str) and text.strip():
                    evo.region_guidance[region] = text.strip()[: self.config.max_guidance_chars]
        g = reply.get("guidance")
        if isinstance(g, list):
            g = "\n".join(f"- {x}" for x in g if isinstance(x, str))
        if isinstance(g, str) and g.strip():
            evo.guidance = g.strip()
            evo.source = "model"
        for item in reply.get("hypotheses") or []:
            if isinstance(item, str):
                item = {"hypothesis": item}
            if not isinstance(item, dict) or not isinstance(item.get("hypothesis"), str):
                continue
            text = item["hypothesis"].strip()
            if text:
                ds = item.get("dataset")
                evo.hypotheses.append((ds if isinstance(ds, str) and ds else None, text))
            if len(evo.hypotheses) == self.config.max_hypotheses:
                break
