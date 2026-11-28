"""What the dashboard shows, computed from memory, run reports and live files (owner
2026-11-28). No network here (``dashboard.py`` serves it); nothing is ever written.

Method
1. Memory is opened read-only (``file:...?mode=ro``) with a lock timeout, so a running
   research run is never disturbed. Rows are append-only, so ``refresh`` reads only rows
   with a larger id than last time and keeps a compact summary per candidate (latest
   result and latest decision, without the daily PnL, ~55 KB per result).
2. An alpha's run: a research run id is the first 16 characters of its own id
   (``rsYYYYMMDDHHMMSS``, UTC); runs also own the alphas they simulated between their start
   and end (resumed repairs keep their root's id). ``hand_YYYYMMDD`` alphas are hand batches.
3. Run reports (``<data dir>/rs*.json``) are read once per file change. Live files
   (``<data dir>/live/*.json``, live.py) are read on every request; a running one whose
   process is gone is shown as "lost".
4. Detail of one alpha: every result and decision, BRAIN's checks with value and limit,
   yearly rows, cumulative PnL thinned to ``PNL_POINTS``, the lineage (ancestors and the
   tree of descendants) and the critic reviews.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .live import LIVE_DIR

PNL_POINTS = 400
RUN_ID_RE = re.compile(r"^(rs\d{14})")
HAND_RE = re.compile(r"^hand_(\d{8})")
LOST_AFTER_SECONDS = 600   # a running live file not updated for this long, process unknown


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _r(x, n=3):
    x = _num(x)
    return round(x, n) if x is not None else None


def run_time(run_id: str) -> str | None:
    """rs20261127153052 -> 2026-11-27T15:30:52+00:00."""
    m = RUN_ID_RE.match(run_id or "")
    if not m:
        return None
    t = datetime.strptime(m.group(1)[2:], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return t.isoformat()


def origin_of(candidate_id: str) -> str:
    """Where an alpha comes from: a research run id, ``hand YYYY-MM-DD``, seeds, generate."""
    m = RUN_ID_RE.match(candidate_id)
    if m:
        return m.group(1)
    m = HAND_RE.match(candidate_id)
    if m:
        d = m.group(1)
        return f"hand {d[:4]}-{d[4:6]}-{d[6:]}"
    if candidate_id.startswith("seed_"):
        return "seeds"
    if candidate_id.startswith("gen"):
        return "generate"
    return "other"


def _alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _result_summary(p: dict, created: str) -> dict:
    checks = p.get("checks") or {}
    cv = p.get("check_values") or {}
    alt = p.get("alt_stats") or {}
    rn = alt.get("risk_neutralized") or {}
    inv = alt.get("investability_constrained") or {}
    failed = [{"name": n, "value": _r((cv.get(n) or {}).get("value"), 4),
               "limit": _r((cv.get(n) or {}).get("limit"), 4)}
              for n, ok in sorted(checks.items()) if ok is False]
    pending = sorted({m.group(1) for n in p.get("notes") or []
                      for m in [re.search(r"([A-Z_.]+)='(PENDING|ERROR)'", n)] if m})
    prod = (cv.get("PROD_CORRELATION") or {}).get("value")
    years = p.get("yearly") or []
    return {"status": p.get("status"), "brain_sim_id": p.get("brain_sim_id"),
            "sharpe": _r(p.get("sharpe"), 2), "fitness": _r(p.get("fitness"), 2),
            "turnover": _r(p.get("turnover"), 4), "returns": _r(p.get("returns"), 4),
            "drawdown": _r(p.get("max_drawdown"), 4), "margin_bps": _r(p.get("margin_bps"), 2),
            "sub_sharpe": _r(p.get("sub_universe_sharpe"), 2),
            "sub_cutoff": _r(p.get("sub_universe_cutoff"), 2),
            "self_corr": _r(p.get("self_correlation"), 3), "prod_corr": _r(prod, 3),
            "rn_sharpe": _r(rn.get("sharpe"), 2), "rn_fitness": _r(rn.get("fitness"), 2),
            "inv_sharpe": _r(inv.get("sharpe"), 2),
            "min_year_sharpe": _r(min((y.get("sharpe") for y in years
                                       if _num(y.get("sharpe")) is not None), default=None), 2),
            "failed_checks": failed, "pending_checks": pending, "simulated_at": created}


def _decision_summary(p: dict, created: str) -> dict:
    gates = p.get("gate_results") or []
    return {"decision": p.get("decision"), "failure_class": p.get("failure_class"),
            "failed_gates": [g.get("name") for g in gates if g.get("passed") is False],
            "decided_at": created}


class DashboardData:
    """Cached, read-only view of ``data_dir`` (method above)."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.db = self.data_dir / "memory.sqlite"
        self.lock = threading.Lock()
        self.cands: dict[str, dict] = {}
        self.order: list[str] = []
        self.results: dict[str, dict] = {}
        self.decisions: dict[str, dict] = {}
        self.result_count: dict[str, int] = {}
        self.children: dict[str, list[str]] = {}
        self.last = {"candidates": 0, "results": 0, "decisions": 0}
        self.reports: dict[str, tuple[float, dict]] = {}
        self.refreshed_at = 0.0

    # ------------------------------------------------------------------ reading

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db}?mode=ro", uri=True, timeout=15)

    def refresh(self, min_age: float = 2.0) -> None:
        """Read rows added since the last call (at most every ``min_age`` seconds)."""
        with self.lock:
            if time.time() - self.refreshed_at < min_age or not self.db.exists():
                return
            con = self._connect()
            try:
                for rowid, payload, created in con.execute(
                        "SELECT rowid, payload_json, created_at FROM candidates "
                        "WHERE rowid > ? ORDER BY rowid", (self.last["candidates"],)):
                    self.last["candidates"] = rowid
                    c = json.loads(payload)
                    c["created_at"] = created
                    cid = c["candidate_id"]
                    if cid not in self.cands:
                        self.order.append(cid)
                    self.cands[cid] = c
                    if c.get("parent_id"):
                        self.children.setdefault(c["parent_id"], []).append(cid)
                for rid, cid, payload, created in con.execute(
                        "SELECT id, candidate_id, payload_json, created_at FROM results "
                        "WHERE id > ? ORDER BY id", (self.last["results"],)):
                    self.last["results"] = rid
                    self.results[cid] = _result_summary(json.loads(payload), created)
                    self.result_count[cid] = self.result_count.get(cid, 0) + 1
                for did, cid, payload, created in con.execute(
                        "SELECT id, candidate_id, payload_json, created_at FROM decisions "
                        "WHERE id > ? ORDER BY id", (self.last["decisions"],)):
                    self.last["decisions"] = did
                    self.decisions[cid] = _decision_summary(json.loads(payload), created)
            finally:
                con.close()
            self.refreshed_at = time.time()

    def _row(self, cid: str) -> dict:
        c = self.cands[cid]
        row = {"candidate_id": cid, "origin": origin_of(cid), "created_at": c.get("created_at"),
               "region": c.get("region"), "universe": c.get("universe"), "delay": c.get("delay"),
               "dataset": c.get("dataset"), "model": c.get("model"), "lever": c.get("lever"),
               "generation": c.get("generation"), "parent_id": c.get("parent_id"),
               "neutralization": c.get("neutralization"), "decay": c.get("decay"),
               "family": c.get("family")}
        row.update(self.results.get(cid) or {"status": None})
        row.update(self.decisions.get(cid) or {"decision": None, "failure_class": None})
        return row

    def _simulated(self) -> list[dict]:
        """Rows of alphas with a result that reached BRAIN (not invalid)."""
        return [self._row(cid) for cid in self.order
                if (self.results.get(cid) or {}).get("status") not in (None, "invalid")]

    # ------------------------------------------------------------------ pages

    def alphas(self) -> dict:
        self.refresh()
        rows = self._simulated()
        rows.sort(key=lambda r: r.get("simulated_at") or "", reverse=True)
        return {"alphas": rows, "count": len(rows)}

    def overview(self) -> dict:
        self.refresh()
        rows = self._simulated()
        complete = [r for r in rows if r["status"] == "complete"]
        by_day: dict[str, dict] = {}
        for r in rows:
            day = (r.get("simulated_at") or "")[:10]
            d = by_day.setdefault(day, {"day": day, "simulations": 0, "accept": 0,
                                        "repair": 0, "reject": 0, "other": 0})
            d["simulations"] += 1
            k = r.get("decision")
            d[k if k in ("accept", "repair", "reject") else "other"] += 1
        regions: dict[str, dict] = {}
        for r in rows:
            g = regions.setdefault(r["region"] or "?", {"region": r["region"], "simulations": 0,
                                                       "accept": 0, "repair": 0, "reject": 0,
                                                       "best_sharpe": None})
            g["simulations"] += 1
            if r.get("decision") in ("accept", "repair", "reject"):
                g[r["decision"]] += 1
            s = r.get("sharpe")
            if s is not None and (g["best_sharpe"] is None or s > g["best_sharpe"]):
                g["best_sharpe"] = s
        classes: dict[str, int] = {}
        for r in rows:
            if r.get("decision") in ("repair", "reject") and r.get("failure_class"):
                classes[r["failure_class"]] = classes.get(r["failure_class"], 0) + 1
        bins = [round(-3 + 0.25 * i, 2) for i in range(33)]   # -3 .. 5
        hist = [0] * len(bins)
        for r in complete:
            s = r.get("sharpe")
            if s is None:
                continue
            i = min(max(int((s + 3) / 0.25), 0), len(bins) - 1)
            hist[i] += 1
        pool = self.pool()
        accepts = [r for r in rows if r.get("decision") == "accept"]
        last_day = max(by_day) if by_day else None
        return {
            "totals": {"simulated": len(rows), "complete": len(complete),
                       "accepted": len(accepts), "submitted": len(pool["pool"]),
                       "repair_open": sum(1 for r in rows if r.get("decision") == "repair"),
                       "awaiting_checks": len(pool["awaiting_checks"]),
                       "best_sharpe": max((r["sharpe"] for r in complete
                                           if r.get("sharpe") is not None), default=None),
                       "today": by_day.get(last_day, {}).get("simulations", 0) if last_day else 0,
                       "last_day": last_day,
                       "candidates": len(self.cands)},
            "by_day": [by_day[k] for k in sorted(by_day) if k],
            "regions": sorted(regions.values(), key=lambda g: -g["simulations"]),
            "failure_classes": dict(sorted(classes.items(), key=lambda kv: -kv[1])),
            "sharpe_hist": {"bins": bins, "counts": hist},
            "recent_accepts": sorted(accepts, key=lambda r: r.get("decided_at") or "",
                                     reverse=True)[:10],
        }

    def alpha(self, cid: str) -> dict | None:
        self.refresh()
        if cid not in self.cands:
            return None
        con = self._connect()
        try:
            res = [(json.loads(p), t) for p, t in con.execute(
                "SELECT payload_json, created_at FROM results WHERE candidate_id = ? ORDER BY id",
                (cid,))]
            dec = [(json.loads(p), t) for p, t in con.execute(
                "SELECT payload_json, created_at FROM decisions WHERE candidate_id = ? "
                "ORDER BY id", (cid,))]
            reviews = [json.loads(p) | {"critic_model": m, "role": role, "verdict": v,
                                        "created_at": t}
                       for p, m, role, v, t in con.execute(
                           "SELECT payload_json, critic_model, role, verdict, created_at "
                           "FROM critic_reviews WHERE candidate_id = ? ORDER BY id", (cid,))]
        finally:
            con.close()
        latest = res[-1][0] if res else {}
        checks = []
        cv = latest.get("check_values") or {}
        for name in sorted(set(cv) | set(latest.get("checks") or {})):
            ok = (latest.get("checks") or {}).get(name)
            v = cv.get(name) or {}
            checks.append({"name": name, "passed": ok, "value": _r(v.get("value"), 4),
                           "limit": _r(v.get("limit"), 4)})
        pnl = latest.get("daily_pnl") or {}
        dates, values = pnl.get("dates") or [], pnl.get("values") or []
        cum, total = [], 0.0
        for v in values:
            total += v if _num(v) is not None else 0.0
            cum.append(total)
        step = max(1, len(cum) // PNL_POINTS)
        series = {"dates": dates[::step], "values": [round(x, 1) for x in cum[::step]]}
        if dates and series["dates"][-1] != dates[-1]:
            series["dates"].append(dates[-1])
            series["values"].append(round(cum[-1], 1))
        return {
            "alpha": self._row(cid), "candidate": self.cands[cid],
            "results": [{**_result_summary(p, t), "notes": p.get("notes") or []}
                        for p, t in res],
            "decisions": [{"decision": p.get("decision"),
                           "failure_class": p.get("failure_class"),
                           "reasons": p.get("reasons") or [], "gates": p.get("gate_results") or [],
                           "decided_at": t} for p, t in dec],
            "checks": checks, "yearly": latest.get("yearly") or [],
            "alt_stats": latest.get("alt_stats") or {}, "pnl": series,
            "lineage": self.lineage(cid), "reviews": reviews,
        }

    def lineage(self, cid: str) -> dict:
        """Ancestors (root first) and the tree of descendants of the root."""
        chain, seen, cur = [], set(), cid
        while cur in self.cands and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = self.cands[cur].get("parent_id")
        chain.reverse()
        root = chain[0] if chain else cid

        def node(x: str, depth: int = 0) -> dict:
            r = self._row(x) if x in self.cands else {"candidate_id": x}
            out = {k: r.get(k) for k in ("candidate_id", "lever", "status", "sharpe", "fitness",
                                         "turnover", "decision", "failure_class")}
            if depth < 12:
                out["children"] = [node(k, depth + 1) for k in self.children.get(x, [])]
            return out

        return {"root": root, "ancestors": chain, "tree": node(root)}

    # ------------------------------------------------------------------ runs

    def _report(self, path: Path) -> dict:
        mtime = path.stat().st_mtime
        hit = self.reports.get(path.name)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        keep = {k: r.get(k) for k in (
            "run_id", "stop_reason", "simulations_used", "model_calls", "model_calls_by_model",
            "regions_this_run", "too_large_calls", "empty_replies", "rate_limit_pauses",
            "brain_quota", "max_concurrent")}
        keep["models_down"] = sorted((r.get("models_down") or {}).keys())
        keep["models_down_reasons"] = {m: str(v)[:240]
                                       for m, v in (r.get("models_down") or {}).items()}
        keep["repair_waves"] = len(r.get("repair_waves") or [])
        keep["power_pool_candidates"] = len(r.get("power_pool_candidates") or [])
        keep["cross_region_leads"] = r.get("cross_region_leads") or []
        q = r.get("generation_quality")
        if isinstance(q, dict):
            keep["fresh_mean_abs_sharpe"] = q.get("mean_abs_sharpe")
            keep["fresh_alphas"] = q.get("fresh_alphas")
        keep["ended_at"] = datetime.fromtimestamp(mtime, timezone.utc).isoformat(
            timespec="seconds")
        lessons = r.get("lessons") or {}
        keep["lessons"] = {"written": lessons.get("written"), "source": lessons.get("source")} \
            if isinstance(lessons, dict) else None
        self.reports[path.name] = (mtime, keep)
        return keep

    def live(self) -> list[dict]:
        out = []
        d = self.data_dir / LIVE_DIR
        if not d.is_dir():
            return out
        for p in d.glob("*.json"):
            try:
                x = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if x.get("status") == "running":
                age = time.time() - p.stat().st_mtime
                if not _alive(x.get("pid")) or age > LOST_AFTER_SECONDS * 6:
                    x["status"] = "lost"
                    x["reason"] = x.get("reason") or "the process is gone (killed or crashed)"
                elif age > LOST_AFTER_SECONDS:
                    x["quiet_seconds"] = round(age)
            out.append(x)
        out.sort(key=lambda x: x.get("started_at") or "", reverse=True)
        return out

    def runs(self) -> dict:
        self.refresh()
        rows = self._simulated()
        runs: dict[str, dict] = {}
        for p in self.data_dir.glob("rs*.json"):
            rep = self._report(p)
            if rep.get("run_id"):
                runs[rep["run_id"]] = {**rep, "kind": "research",
                                       "started_at": run_time(rep["run_id"]),
                                       "status": "finished"}
        for x in self.live():
            rid = x.get("run_id")
            base = runs.get(rid, {"run_id": rid, "kind": x.get("kind")})
            st = x.get("state") or {}
            base.update(status=x.get("status") if base.get("status") != "finished"
                        or x.get("status") == "running" else "finished",
                        started_at=base.get("started_at") or x.get("started_at"),
                        live_updated_at=x.get("updated_at"))
            if x.get("status") != "finished" or not base.get("stop_reason"):
                base["stop_reason"] = base.get("stop_reason") or x.get("reason")
            if "ended_at" not in base and x.get("status") != "running":
                base["ended_at"] = x.get("updated_at")
            if st.get("regions") and not base.get("regions_this_run"):
                base["regions_this_run"] = st.get("regions")
            runs[rid] = base
        # research runs with alphas but no report and no live file: still going (started
        # before live status existed, or without it) or stopped without writing a report
        seen_origins: dict[str, list] = {}
        for r in rows:
            if RUN_ID_RE.match(r["origin"]):
                seen_origins.setdefault(r["origin"], []).append(r)
        for rid, xs in seen_origins.items():
            if rid in runs:
                continue
            last = max(x.get("simulated_at") or "" for x in xs)
            recent = last and (datetime.now(timezone.utc)
                               - datetime.fromisoformat(last)).total_seconds() < 3600
            runs[rid] = {"run_id": rid, "kind": "research", "started_at": run_time(rid),
                         "status": "running?" if recent else "no report",
                         "stop_reason": ("no report yet; last alpha " + last[11:16] + " UTC"
                                         if recent else "stopped without a report"),
                         **({} if recent else {"ended_at": last})}
        # hand batches (rescue files run with ``cli run``) without a live file: one per day
        batch_days = {(r.get("started_at") or "")[:10] for r in runs.values()
                      if r.get("kind") == "batch"}
        hand: dict[str, list] = {}
        for r in rows:
            if r["origin"].startswith("hand "):
                hand.setdefault(r["origin"], []).append(r)
        for origin, xs in hand.items():
            if origin[5:] in batch_days:
                continue
            times = sorted(x.get("simulated_at") or "" for x in xs)
            runs[origin] = {"run_id": origin, "kind": "hand", "status": "finished",
                            "started_at": times[0], "ended_at": times[-1],
                            "stop_reason": "hand batch (cli run)",
                            "simulations_used": len(xs)}
        ordered = sorted(runs.values(), key=lambda r: r.get("started_at") or "")
        # alphas by origin id, else by simulation time inside the run's window
        for i, run in enumerate(ordered):
            start = run.get("started_at") or ""
            end = run.get("ended_at") or (ordered[i + 1].get("started_at")
                                          if i + 1 < len(ordered) else None) or "9999"
            own = [r for r in rows if r["origin"] == run["run_id"]
                   or (run.get("kind") == "research" and not r["origin"].startswith("hand")
                       and start <= (r.get("simulated_at") or "") <= end)]
            if run.get("kind") == "hand":
                own = [r for r in rows if r["origin"] == run["run_id"]]
            elif run.get("kind") == "batch":
                own = [r for r in rows if start <= (r.get("simulated_at") or "") <= end
                       and not RUN_ID_RE.match(r["candidate_id"])]
            run["alphas"] = len(own)
            run["accepted"] = sum(1 for r in own if r.get("decision") == "accept")
            run["repair"] = sum(1 for r in own if r.get("decision") == "repair")
            run["rejected"] = sum(1 for r in own if r.get("decision") == "reject")
            best = max((r for r in own if r.get("sharpe") is not None),
                       key=lambda r: r["sharpe"], default=None)
            run["best_sharpe"] = best["sharpe"] if best else None
            run["best_alpha"] = best["candidate_id"] if best else None
            vals = [abs(r["sharpe"]) for r in own if r.get("sharpe") is not None]
            run["median_abs_sharpe"] = round(statistics.median(vals), 2) if vals else None
            run["regions_simulated"] = sorted({r["region"] for r in own if r.get("region")})
        ordered.reverse()
        return {"runs": ordered}

    # ------------------------------------------------------------------ pool, levers, models

    def pool(self) -> dict:
        self.refresh()
        out, awaiting, last = [], [], {}
        if self.db.exists():
            con = self._connect()
            try:
                try:   # the pool tables appear with the first pool alpha
                    rows = con.execute("SELECT alpha_id, candidate_id, admitted_at, "
                                       "metrics_json FROM pool").fetchall()
                except sqlite3.OperationalError:
                    rows = []
                try:
                    events = con.execute("SELECT alpha_id, event, at, reason FROM pool_events "
                                         "ORDER BY id").fetchall()
                except sqlite3.OperationalError:
                    events = []
            finally:
                con.close()
            for aid, ev, at, reason in events:
                last[aid] = (ev, at, reason)
            for aid, cid, admitted, metrics in rows:
                ev = last.get(aid)
                if ev and ev[0] == "retire":
                    continue
                r = self._row(cid) if cid in self.cands else {"candidate_id": cid}
                out.append({**r, "alpha_id": aid, "admitted_at": admitted,
                            "pool_event": ev[0] if ev else "admitted",
                            "pool_reason": ev[2] if ev else None,
                            "metrics": json.loads(metrics or "{}")})
        for r in self._simulated():
            if r.get("decision") == "pending" and r.get("pending_checks"):
                awaiting.append(r)
        in_pool = {p["candidate_id"] for p in out}
        roots = {self.lineage(cid)["root"]: cid for cid in in_pool if cid in self.cands}
        set_aside = {aid for aid, ev in last.items() if ev[0] == "retire"} if self.db.exists() else set()
        accepted_unsubmitted = []
        for r in self._simulated():
            if r.get("decision") != "accept" or r["candidate_id"] in in_pool \
                    or r["candidate_id"] in set_aside:
                continue
            sib = roots.get(self.lineage(r["candidate_id"])["root"])
            accepted_unsubmitted.append({**r, "sibling_in_pool": sib})
        return {"pool": out, "awaiting_checks": awaiting,
                "accepted_not_in_pool": accepted_unsubmitted}

    def levers(self) -> dict:
        self.refresh()
        stats: dict[tuple, list] = {}
        for cid in self.order:
            c = self.cands[cid]
            lever, pid = c.get("lever"), c.get("parent_id")
            if not lever or not pid or lever == "critic_revision":
                continue
            child, parent = self.results.get(cid), self.results.get(pid)
            if not child or not parent or child.get("status") != "complete" \
                    or parent.get("status") != "complete":
                continue
            cls = (self.decisions.get(pid) or {}).get("failure_class") or "?"
            d = (self.decisions.get(cid) or {}).get("decision")
            ds = (child.get("sharpe") - parent["sharpe"]) if None not in (
                child.get("sharpe"), parent.get("sharpe")) else None
            df = (child.get("fitness") - parent["fitness"]) if None not in (
                child.get("fitness"), parent.get("fitness")) else None
            for key in ((lever, "all"), (lever, cls)):
                stats.setdefault(key, []).append((d, ds, df))
        # submitted alphas credit every fix on the path from their lineage's original
        on_path: dict[str, set] = {}
        for sid in (p["candidate_id"] for p in self.pool()["pool"]):
            for cid in self.lineage(sid)["ancestors"]:
                lev = (self.cands.get(cid) or {}).get("lever")
                if lev and lev != "critic_revision":
                    on_path.setdefault(lev, set()).add(sid)
        rows = []
        for (lever, cls), xs in stats.items():
            dss = [x[1] for x in xs if x[1] is not None]
            dfs = [x[2] for x in xs if x[2] is not None]
            rows.append({"lever": lever, "parent_class": cls, "children": len(xs),
                         "accepted": sum(1 for x in xs if x[0] == "accept"),
                         "submitted": len(on_path.get(lever, ())) if cls == "all" else None,
                         "submitted_ids": sorted(on_path.get(lever, ())),
                         "repair": sum(1 for x in xs if x[0] == "repair"),
                         "median_d_sharpe": _r(statistics.median(dss), 2) if dss else None,
                         "median_d_fitness": _r(statistics.median(dfs), 2) if dfs else None,
                         "sharpe_up_share": _r(sum(1 for v in dss if v > 0) / len(dss), 2)
                         if dss else None})
        rows.sort(key=lambda r: (r["parent_class"] != "all", -r["children"]))
        return {"levers": rows}

    def _credit(self, root: str) -> str:
        """Who wrote a lineage's original alpha: its generator model; seeds are the owner's
        alphas (their model field is only the arm they ran under); hand-written roots "hand"."""
        if root.startswith("seed_"):
            return "seed (owner's alpha)"
        model = (self.cands.get(root) or {}).get("model")
        return model if model and model != "hand" else "hand"

    def models(self) -> dict:
        """Per generator model and per market/dataset: the quality of the fresh alphas (as
        first written), and the accepts and submissions anywhere in the lineages they started
        (credit to the original's model and dataset, owner 2026-11-28: no alpha has passed as
        first written; every accept came from repairs, most of them hand rescues)."""
        self.refresh()
        groups: dict[tuple, dict] = {}

        def group(kind: str, name: str) -> dict:
            return groups.setdefault((kind, name), {"fresh": [], "accepted": set(),
                                                    "lineages": set(), "submitted": set(),
                                                    "pipeline": set()})

        for r in self._simulated():
            # as first simulated: no repair; a Critic revision counts (its original was never
            # simulated or stored)
            fresh = (not r.get("lever") and not r.get("parent_id")) or (
                r.get("lever") == "critic_revision" and r.get("parent_id") not in self.cands)
            if r["status"] != "complete" or not fresh:
                continue
            if r.get("model") and r["model"] != "hand" and not r["candidate_id"].startswith("seed_"):
                group("model", r["model"])["fresh"].append(r)
            group("dataset", f"{r['region']} / {r['dataset']}")["fresh"].append(r)
        submitted = {p["candidate_id"] for p in self.pool()["pool"]}
        for cid, d in self.decisions.items():
            if d.get("decision") != "accept" or cid not in self.cands \
                    or (self.results.get(cid) or {}).get("status") != "complete":
                continue
            root = self.lineage(cid)["root"]
            rc = self.cands.get(root) or {}
            for g in (group("model", self._credit(root)),
                      group("dataset", f"{rc.get('region')} / {rc.get('dataset')}")):
                g["accepted"].add(cid)
                g["lineages"].add(root)
                if cid in submitted:
                    g["submitted"].add(cid)
                if not cid.startswith("hand_"):
                    g["pipeline"].add(cid)
        out = {"model": [], "dataset": []}
        for (kind, name), g in groups.items():
            xs = g["fresh"]
            s = [abs(x["sharpe"]) for x in xs if x.get("sharpe") is not None]
            out[kind].append({
                "name": name, "alphas": len(xs),
                "accepted_lineages": len(g["lineages"]), "accepted": len(g["accepted"]),
                "accepted_by_pipeline": len(g["pipeline"]), "submitted": len(g["submitted"]),
                "submitted_ids": sorted(g["submitted"]), "accepted_ids": sorted(g["accepted"]),
                "mean_abs_sharpe": _r(statistics.mean(s), 2) if s else None,
                "best_sharpe": max((x["sharpe"] for x in xs if x.get("sharpe") is not None),
                                   default=None),
                "share_abs_sharpe_1": _r(sum(1 for v in s if v >= 1) / len(s), 2) if s else None})
        for k in out:
            out[k].sort(key=lambda g: (-g["submitted"], -g["accepted_lineages"], -g["alphas"]))
        return out
