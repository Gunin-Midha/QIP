"""Second consultant discovery run: dataset lists and datafield catalogs for every region and
delay our account allows, in ONE BRAIN login (one biometric check). No simulations.

For each region and delay it uses the region's first (largest) universe in
schemas.REGION_SETTINGS and saves the full dataset list (cheap: it carries fieldCount,
coverage, userCount, alphaCount, pyramidMultiplier). Fields are downloaded one dataset at a
time for at most ``--max-datasets`` datasets (``pick_datasets``; 0 = every dataset), so
BRAIN's 10,000-offset cap never cuts a catalog short. Each dataset's fields are saved to
``parts/<tag>/<dataset>.json`` as soon as they arrive, so Ctrl+C or a BRAIN 502 loses
nothing, and a rerun (or a larger ``--max-datasets``) only fetches what is missing.
``datafields_<tag>.json`` is rebuilt from every saved part. A dataset that fails is retried,
then skipped and listed at the end.

Run from alpha_research/ (BRAIN_EMAIL and BRAIN_PASSWORD in the environment):
    PYTHONPATH=src python3 scripts/brain_discovery.py [--max-datasets 20] [--regions USA EUR]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from alpha_research.brain_api import (BrainApiError, list_datafields, list_datasets,
                                      load_brain_config, make_session)
from alpha_research.schemas import REGION_SETTINGS

RETRIES = 3
RETRY_WAIT = (10, 30, 60)
MIN_EXPLORE_COVERAGE = 0.5   # a little-explored dataset must cover at least half the universe


def say(msg: str) -> None:
    print(msg, flush=True)


def biometric(url: str) -> None:
    say(f"\nBRAIN asks for biometric verification. Open this link, verify, then press Enter:\n  {url}")
    input()


def with_retries(what: str, fn):
    for attempt in range(RETRIES + 1):
        try:
            return fn()
        except (ConnectionError, TimeoutError, BrainApiError) as exc:
            if attempt == RETRIES:
                say(f"    gave up on {what}: {exc}")
                return None
            say(f"    {what} failed ({exc}); trying again in {RETRY_WAIT[attempt]}s")
            time.sleep(RETRY_WAIT[attempt])


def pick_datasets(datasets: list[dict], n: int, explore_share: float = 0.3) -> list[dict]:
    """At most ``n`` datasets (all when n <= 0). Most of the slots go to the most-used dataset
    of each category (categories in order of their top userCount), then to the next most used.
    ``explore_share`` of the slots go to little-explored datasets: lowest alphaCount among
    those with coverage >= MIN_EXPLORE_COVERAGE, higher pyramidMultiplier first on ties."""
    if n <= 0 or n >= len(datasets):
        return list(datasets)
    users = lambda d: d.get("userCount") or 0
    n_explore = int(n * explore_share)
    by_use = sorted(datasets, key=lambda d: -users(d))
    chosen, cats = [], set()
    for d in by_use:                                   # top dataset per category
        cat = (d.get("category") or {}).get("id")
        if cat not in cats and len(chosen) < n - n_explore:
            cats.add(cat)
            chosen.append(d)
    for d in by_use:                                   # then the most used
        if len(chosen) >= n - n_explore:
            break
        if d not in chosen:
            chosen.append(d)
    rest = [d for d in datasets if d not in chosen and (d.get("coverage") or 0) >= MIN_EXPLORE_COVERAGE]
    rest.sort(key=lambda d: (d.get("alphaCount") or 0, -(d.get("pyramidMultiplier") or 0)))
    chosen += rest[:n - len(chosen)]
    for d in by_use:                                   # too few well-covered ones: fill by use
        if len(chosen) >= n:
            break
        if d not in chosen:
            chosen.append(d)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="captures/consultant_catalogs")
    ap.add_argument("--brain-config", default="config/brain.yaml")
    ap.add_argument("--regions", nargs="*", default=list(REGION_SETTINGS))
    ap.add_argument("--max-datasets", type=int, default=20,
                    help="datasets per region/delay to download fields for (0 = all)")
    ap.add_argument("--explore-share", type=float, default=0.3,
                    help="share of those slots for little-explored datasets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_brain_config(args.brain_config)
    instrument = cfg.settings.get("instrumentType", "EQUITY")
    try:
        session = make_session(cfg, on_biometric=biometric)
        session.authenticate()
    except BrainApiError as exc:
        say(f"Could not log in to BRAIN: {exc}")
        say("Set them in this terminal first:\n  export BRAIN_EMAIL=\"you@example.com\"\n"
            "  read -s BRAIN_PASSWORD && export BRAIN_PASSWORD")
        return 1
    say("Logged in to BRAIN.")

    combos = [(r, REGION_SETTINGS[r]["universes"][0], d)
              for r in args.regions for d in REGION_SETTINGS[r]["delays"]]
    summary, gaps = [], []
    for n, (region, universe, delay) in enumerate(combos, 1):
        tag = f"{region.lower()}_{universe.lower()}_d{delay}"
        fields_file = out / f"datafields_{tag}.json"
        datasets_file = out / f"datasets_{tag}.json"
        parts = out / "parts" / tag
        say(f"\n[{n}/{len(combos)}] {region} / {universe} / delay {delay}")
        if fields_file.exists() and not parts.exists():
            # written by the earlier, uncapped version of this script: already complete
            count = len(json.loads(fields_file.read_text()))
            say(f"  already done earlier (every dataset): {count:,} fields")
            summary.append((tag, None, count))
            continue
        if datasets_file.exists():
            datasets = json.loads(datasets_file.read_text())
        else:
            datasets = with_retries("the dataset list", lambda: list_datasets(
                session, region=region, universe=universe, delay=delay, instrument_type=instrument))
            if datasets is None:
                gaps.append(f"{tag}: dataset list")
                continue
            datasets_file.write_text(json.dumps(datasets, indent=2) + "\n")
        picked = pick_datasets(datasets, args.max_datasets, args.explore_share)
        parts.mkdir(parents=True, exist_ok=True)
        todo = [ds for ds in picked if not (parts / f"{ds['id']}.json").exists()]
        say(f"  {len(datasets)} datasets; fields for {len(picked)} of them "
            f"({len(picked) - len(todo)} saved earlier): {', '.join(ds['id'] for ds in picked)}")
        missing = []
        for i, ds in enumerate(todo, 1):
            part = with_retries(f"dataset {ds['id']}", lambda: list_datafields(
                session, region=region, universe=universe, delay=delay,
                instrument_type=instrument, dataset=ds["id"]))
            if part is None:
                missing.append(ds["id"])
                continue
            (parts / f"{ds['id']}.json").write_text(json.dumps(part, indent=2) + "\n")
            say(f"  {i} of {len(todo)}: {ds['id']} ({len(part):,} fields)")
        fields, seen = [], set()
        for f_part in sorted(parts.glob("*.json")):
            for f in json.loads(f_part.read_text()):
                if f["id"] not in seen:
                    seen.add(f["id"])
                    fields.append(f)
        fields_file.write_text(json.dumps(fields, indent=2) + "\n")
        n_parts = len(list(parts.glob("*.json")))
        say(f"  saved {len(fields):,} fields from {n_parts} datasets")
        if missing:
            gaps.append(f"{tag}: {len(missing)} datasets failed ({', '.join(missing[:8])}"
                        f"{'...' if len(missing) > 8 else ''})")
        summary.append((tag, n_parts, len(fields)))

    say("\nSummary")
    for tag, nds, nf in summary:
        say(f"  {tag}: {nf:,} fields" + (f" from {nds} datasets" if nds is not None else ""))
    if gaps:
        say("Not complete (run the script again to retry these):")
        for g in gaps:
            say(f"  {g}")
    else:
        say("Every chosen dataset is downloaded.")
    say(f"Files are in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
