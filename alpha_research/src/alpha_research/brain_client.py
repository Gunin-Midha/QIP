"""BRAIN client interface and implementations (spec section 4.1).

``FixtureBrainClient`` replays JSON fixtures. ``RealBrainClient`` is a stub: no HTTP code
exists in this phase.

Fixture file format (``<brain_sim_id>.json``). This is our own test format, not BRAIN's
API. TODO(open-question) OQ-2: the real response shape is unknown until the owner
provides a raw sample. The key names below are the names the simulator normalizes from.

    {
      "responses": [                 # optional: a sequence returned by successive polls;
        {"status": "running", "retry_after": 3},     # the last item repeats forever
        {"raise": "TimeoutError"},   # makes that poll raise a transient error
        {"status": "complete", "sharpe": 1.68, ...}
      ]
    }

Without ``responses`` the whole file is the single response. Aggregate keys: status,
sharpe, fitness, turnover (%), returns (%), drawdown (%), drawdown_date, margin (bps),
sub_universe_sharpe, sub_universe_cutoff, weight_concentration, self_correlation,
long_pnl, short_pnl, checks, yearly, and either daily_pnl ({"dates", "values"}) or
daily_pnl_file (CSV path relative to the fixtures root, columns date,pnl).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import pandas as pd

from .schemas import CandidateRecord

_TRANSIENT = {"TimeoutError": TimeoutError, "ConnectionError": ConnectionError}


class BrainClient(Protocol):
    def submit(self, candidate: CandidateRecord) -> str: ...        # returns brain_sim_id

    def poll(self, brain_sim_id: str) -> dict: ...                  # raw response dict


class FixtureNotFoundError(LookupError):
    pass


class FixtureBrainClient:
    """Replays fixtures from ``fixture_dir``.

    ``sim_ids`` maps candidate_id to fixture id (the fixture's file stem). Each candidate in
    the hand-written file stores that id as ``fixture_id``.
    """

    def __init__(self, fixture_dir: str | Path, sim_ids: dict[str, str] | None = None,
                 submit_errors: dict[str, list[str]] | None = None):
        self.fixture_dir = Path(fixture_dir)
        self.sim_ids = dict(sim_ids or {})
        # candidate_id -> list of transient error names to raise on successive submits (tests)
        self.submit_errors = {k: list(v) for k, v in (submit_errors or {}).items()}
        self.submit_calls: list[str] = []
        self.poll_calls: list[str] = []
        self._cursor: dict[str, int] = {}

    def register(self, candidate_id: str, fixture_id: str) -> None:
        self.sim_ids[candidate_id] = fixture_id

    def submit(self, candidate: CandidateRecord) -> str:
        self.submit_calls.append(candidate.candidate_id)
        pending_errors = self.submit_errors.get(candidate.candidate_id)
        if pending_errors:
            raise _TRANSIENT[pending_errors.pop(0)](f"fixture submit error for {candidate.candidate_id}")
        sim_id = self.sim_ids.get(candidate.candidate_id)
        if sim_id is None:
            raise FixtureNotFoundError(f"no fixture registered for {candidate.candidate_id!r}")
        if not (self.fixture_dir / f"{sim_id}.json").exists():
            raise FixtureNotFoundError(f"fixture file {sim_id}.json not found in {self.fixture_dir}")
        return sim_id

    def poll(self, brain_sim_id: str) -> dict:
        self.poll_calls.append(brain_sim_id)
        data = json.loads((self.fixture_dir / f"{brain_sim_id}.json").read_text())
        responses = data["responses"] if "responses" in data else [data]
        i = self._cursor.get(brain_sim_id, 0)
        self._cursor[brain_sim_id] = i + 1
        response = dict(responses[min(i, len(responses) - 1)])
        if "raise" in response:
            raise _TRANSIENT[response["raise"]](f"fixture poll error for {brain_sim_id}")
        if "daily_pnl_file" in response:
            response["daily_pnl"] = self._load_pnl(response.pop("daily_pnl_file"))
        return response

    def _load_pnl(self, rel_path: str) -> dict:
        path = self.fixture_dir.parent / rel_path
        df = pd.read_csv(path)
        return {"dates": df["date"].astype(str).tolist(), "values": df["pnl"].astype(float).tolist()}


class RealBrainClient:
    """Stub. No HTTP code until a real response sample exists (OQ-2)."""

    _MSG = "BRAIN API not yet integrated. Needs a real response sample. See OQ-2."

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, candidate: CandidateRecord) -> str:
        raise NotImplementedError(self._MSG)

    def poll(self, brain_sim_id: str) -> dict:
        raise NotImplementedError(self._MSG)
