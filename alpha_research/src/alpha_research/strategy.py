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

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .generator import GenerationError, ask_json
from .lever_stats import LeverStats, repair_events
from .llm import LLMClient, ModelDown
from .prompts import strategy_system, strategy_user

@dataclass(frozen=True)
class StrategyConfig:
    every_rounds: int = 2
    max_hypotheses: int = 3
    max_guidance_chars: int = 1500
    tag: str = "evo"


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
              known_datasets: list[str] | None = None) -> dict:
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
    least = sorted((d for d in known if by_dataset[d] < FEW_TRIALS), key=lambda d: (by_dataset[d], d))
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
        "least_tried": least[:20],
        "what_fixes_worked": [{k: r[k] for k in ("failure_class", "lever", "fixed", "posterior_mean")}
                              for r in worked[:8]],
    }


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
    good = [r for r in stats.region_table() if r["region"] == region and r["fixed"] > 0]
    for r in good[:max_levers]:
        lines.append(f"- Repair that fixed {r['failure_class']} there: {r['lever']}.")
    return "\n".join(lines)


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
               known_datasets: list[str] | None = None) -> Evolution | None:
        summary = summarize(outcomes, known_datasets=known_datasets)
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
        try:
            reply, calls = ask_json(self.llm, strategy_system(self.context),
                                    strategy_user(summary, self.config.max_hypotheses),
                                    max_output=self.max_output)
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
