"""Fetch the current results of alphas from BRAIN (read only: no simulation, no memory write).

Each argument is a BRAIN alpha id (e.g. 3q08g78Q), one of our candidate ids (looked up in
memory: its result's `brain_alpha_id` note, or, when no result came back, its pending
simulation id), or a simulation id with --sim. For each alpha it prints the in-sample
stats and BRAIN's submission checks (GET /alphas/<id>/check, which is where PROD_CORRELATION
is filled in once BRAIN has computed it), plus the self-correlation max.

    PYTHONPATH=src python3 scripts/brain_fetch.py 3q08g78Q leaoZq2N
    PYTHONPATH=src python3 scripts/brain_fetch.py hand_20261124_chn2_portfolio_imbalance
    PYTHONPATH=src python3 scripts/brain_fetch.py --sim 28pX2hfir4tvaH0j6TvMajx --save out/

Uses the saved BRAIN session (config/brain.yaml session_file); a new login asks for
credentials and biometric verification in the terminal.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

from alpha_research.brain_api import (BrainApiError, load_brain_config, make_session,
                                      prompt_credentials)

ALPHA_ID = re.compile(r"^[A-Za-z0-9]{6,10}$")   # BRAIN alpha ids are short (8 chars so far)


def say(msg: str = "") -> None:
    print(msg, flush=True)


def biometric(url: str) -> None:
    say(f"\nBRAIN asks for biometric verification. Open this link, verify, then press Enter:\n  {url}")
    input()


def from_memory(db: Path, candidate_id: str) -> tuple[str | None, str | None]:
    """(alpha id, simulation id) for one of our candidates, from memory."""
    if not db.exists():
        return None, None
    con = sqlite3.connect(db)
    try:
        row = con.execute("select payload_json from results where candidate_id=? "
                          "order by id desc limit 1", (candidate_id,)).fetchone()
        if row:
            for note in json.loads(row[0]).get("notes") or []:
                if str(note).startswith("brain_alpha_id:"):
                    return note.split(":", 1)[1].strip(), None
        sim = con.execute("select * from pending_submissions where candidate_id=? "
                          "order by rowid desc limit 1", (candidate_id,)).fetchone()
        return None, (sim[1] if sim else None)
    finally:
        con.close()


def alpha_of_simulation(session, sim_id: str) -> str | None:
    resp = session.request("GET", f"/simulations/{sim_id}")
    if resp.status == 404:
        say(f"  simulation {sim_id}: not found on BRAIN")
        return None
    sim = resp.json() or {}
    if sim.get("alpha"):
        return sim["alpha"]
    if "progress" in sim and not sim.get("status"):
        say(f"  simulation {sim_id}: still running ({float(sim['progress']):.0%})")
    else:
        msg = sim.get("message") or sim.get("detail") or sim.get("details") or sim.get("status")
        say(f"  simulation {sim_id}: no alpha ({msg})")
    return None


def fmt(v, pct=False) -> str:
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v:.2f}" if isinstance(v, float) else str(v)


def show(session, alpha_id: str, wait: float, save: Path | None) -> None:
    resp = session.request("GET", f"/alphas/{alpha_id}")
    if resp.status != 200:
        say(f"  alpha {alpha_id}: BRAIN answered {resp.status}: {resp.detail()}")
        return
    alpha = resp.json() or {}
    s = alpha.get("is") or {}
    st = alpha.get("settings") or {}
    say(f"\n{alpha_id}  {st.get('region')}/{st.get('universe')}/D{st.get('delay')}  "
        f"{st.get('neutralization')}  decay {st.get('decay')}  status {alpha.get('status')}")
    say(f"  Sharpe {fmt(s.get('sharpe'))}  fitness {fmt(s.get('fitness'))}  "
        f"turnover {fmt(s.get('turnover'), True)}  returns {fmt(s.get('returns'), True)}  "
        f"drawdown {fmt(s.get('drawdown'), True)}")

    check = None
    try:
        r = session.get_ready(f"/alphas/{alpha_id}/check", wait)
        check = r.json() if r.status == 200 else None
        if check is None:
            say(f"  checks: BRAIN answered {r.status}")
    except TimeoutError:
        say(f"  checks: not ready after {wait:.0f}s (showing the alpha's own checks)")
    checks = ((check or {}).get("is") or {}).get("checks") or s.get("checks") or []
    bad = [c for c in checks if c.get("result") not in ("PASS",)]
    say(f"  checks: {sum(c.get('result') == 'PASS' for c in checks)} pass"
        + ("" if bad else ", nothing else"))
    for c in bad:
        vl = ""
        if c.get("value") is not None or c.get("limit") is not None:
            vl = f"  {fmt(c.get('value'))} vs {fmt(c.get('limit'))}"
        say(f"    {c.get('result'):8} {c.get('name')}{vl}")
    prod = next((c for c in checks if c.get("name") == "PROD_CORRELATION"), None)
    if prod:
        say(f"  PROD_CORRELATION: {prod.get('result')}  {fmt(prod.get('value'))} vs {fmt(prod.get('limit'))}")

    corr = None
    try:
        r = session.get_ready(f"/alphas/{alpha_id}/correlations/self", wait)
        corr = r.json() if r.status == 200 else None
        say(f"  self-correlation max: {fmt((corr or {}).get('max'))}")
    except TimeoutError:
        say(f"  self-correlation: not ready after {wait:.0f}s")

    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        out = save / f"{alpha_id}.json"
        out.write_text(json.dumps({"alpha": alpha, "check": check, "self_correlation": corr},
                                  indent=1))
        say(f"  saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ids", nargs="+", help="BRAIN alpha ids or our candidate ids")
    ap.add_argument("--sim", action="store_true", help="the ids are BRAIN simulation ids")
    ap.add_argument("--wait", type=float, default=300.0,
                    help="seconds to wait for checks / correlations BRAIN is computing")
    ap.add_argument("--save", type=Path, help="folder for the raw JSON (keep it out of git)")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--brain-config", default="config/brain.yaml")
    args = ap.parse_args()

    try:
        session = make_session(load_brain_config(args.brain_config), on_biometric=biometric,
                               ask_credentials=prompt_credentials)
        session.ensure_login()
    except (BrainApiError, ConnectionError) as exc:
        say(f"BRAIN login failed: {exc}")
        return 7

    db = args.data_dir / "memory.sqlite"
    for raw in args.ids:
        try:
            alpha_id = None
            if args.sim:
                alpha_id = alpha_of_simulation(session, raw)
            else:
                alpha_id, sim_id = from_memory(db, raw)
                if alpha_id is None and sim_id is not None:
                    say(f"{raw}: no result in memory, asking BRAIN about simulation {sim_id}")
                    alpha_id = alpha_of_simulation(session, sim_id)
                elif alpha_id is None and ALPHA_ID.match(raw):
                    alpha_id = raw
                elif alpha_id is None:
                    say(f"{raw}: not a BRAIN alpha id and not in memory")
            if alpha_id:
                show(session, alpha_id, args.wait, args.save)
        except (BrainApiError, ConnectionError) as exc:
            say(f"{raw}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
