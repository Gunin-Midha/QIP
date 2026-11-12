"""Data contracts shared by every component (spec section 2).

Parsing is split in two steps:

* ``from_dict`` checks structure: required keys present, values of the right type,
  dates parseable. A failure here is a malformed file and raises ``SchemaError``.
* ``validate`` checks domain rules (allowed regions, delay in {0, 1}, ...). The
  simulator calls it per candidate and turns a failure into a ``status="invalid"``
  result instead of stopping the run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd


class SchemaError(ValueError):
    """Raised when a record does not match its contract."""


# Settings our account allows per region (`cli brain options`, consultant level,
# 2026-11-10; the response is tests/fixtures/brain_api_capture_consultant/
# simulation_options.json and test_schemas checks this table against it).
_NEUTS = ("NONE", "REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING", "FAST", "SLOW",
          "SLOW_AND_FAST", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY")
_NEUTS_COUNTRY = _NEUTS + ("COUNTRY",)
# neutralize within a group; the rest (STATISTICAL, CROWDING, ...) are BRAIN's factor neutralizations
GROUP_NEUTRALIZATIONS = ("MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "COUNTRY")
REGION_SETTINGS = {
    # region: {"universes", "delays", "neutralizations"}
    "ALL": {"universes": ("LARGE", "MEDIUM", "SMALL"), "delays": (1,), "neutralizations": _NEUTS},
    "USA": {"universes": ("TOP3000", "TOP2000", "TOP1000", "TOP500", "TOP200", "TOPSP500"),
            "delays": (0, 1), "neutralizations": _NEUTS},
    "GLB": {"universes": ("TOP3000", "MINVOL1M", "MINVOL10M", "TOPDIV3000"), "delays": (1,),
            "neutralizations": _NEUTS_COUNTRY},
    "EUR": {"universes": ("TOP2500", "TOP1200", "TOP800", "TOP400", "TOPCS1600"),
            "delays": (0, 1), "neutralizations": _NEUTS_COUNTRY},
    "ASI": {"universes": ("MINVOL1M", "MINVOL10M", "TOP500"), "delays": (1,),
            "neutralizations": _NEUTS_COUNTRY},
    "CHN": {"universes": ("TOP2000U",), "delays": (0, 1), "neutralizations": _NEUTS},
    "JPN": {"universes": ("TOP1600", "TOP1200"), "delays": (0, 1), "neutralizations": _NEUTS},
    "AMR": {"universes": ("TOP600",), "delays": (0, 1),
            "neutralizations": ("NONE", "STATISTICAL", "MARKET", "SECTOR", "INDUSTRY",
                                "SUBINDUSTRY", "COUNTRY")},
    "IND": {"universes": ("TOP500",), "delays": (1,), "neutralizations": _NEUTS},
    "DEU": {"universes": ("TOP500",), "delays": (0, 1), "neutralizations": _NEUTS},
    "GBR": {"universes": ("TOP700",), "delays": (0, 1), "neutralizations": _NEUTS},
}
REGIONS = set(REGION_SETTINGS)
# BRAIN lists ALL in OPTIONS /simulations but refuses regular alphas there (400 "Region ALL
# is not available for simulation type REGULAR", run rs20261111162604), so plans can't use it.
NO_REGULAR_SIMULATION = frozenset({"ALL"})
UNIVERSES = {u for s in REGION_SETTINGS.values() for u in s["universes"]}
NEUTRALIZATIONS = {n for s in REGION_SETTINGS.values() for n in s["neutralizations"]}


def neutralizations_for(region: str) -> tuple[str, ...]:
    """The neutralizations BRAIN allows in ``region`` (empty for an unknown region)."""
    return REGION_SETTINGS.get(region, {}).get("neutralizations", ())
RESULT_STATUSES = {"complete", "failed", "timeout", "invalid"}
DECISIONS = {"accept", "repair", "reject", "pending"}
FAILURE_CLASSES = (
    "none",
    "self_correlation",
    "turnover",
    "sharpe",
    "fitness",
    "sub_universe",
    "concentration",
    "drawdown",
    "leg_imbalance",
    "weak_year",
    "recent_sharpe",      # BRAIN's LOW_2Y_SHARPE / IS_LADDER_SHARPE (consultant level)
    "prod_correlation",   # BRAIN's PROD_CORRELATION: too close to production alphas
    "incomplete",
    "other",
)


# ---------------------------------------------------------------------------
# small parsing helpers
# ---------------------------------------------------------------------------

def _parse_date(value: Any, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise SchemaError(f"{name}: cannot parse date {value!r}") from exc
    raise SchemaError(f"{name}: expected ISO date string, got {type(value).__name__}")


def _opt_date(value: Any, name: str) -> date | None:
    return None if value is None else _parse_date(value, name)


def _req(d: dict, key: str) -> Any:
    if key not in d or d[key] is None:
        raise SchemaError(f"missing required field {key!r}")
    return d[key]


def _req_str(d: dict, key: str) -> str:
    v = _req(d, key)
    if not isinstance(v, str):
        raise SchemaError(f"{key}: expected str, got {type(v).__name__}")
    return v


def _opt_str(d: dict, key: str) -> str | None:
    v = d.get(key)
    if v is not None and not isinstance(v, str):
        raise SchemaError(f"{key}: expected str or null, got {type(v).__name__}")
    return v


def _req_int(d: dict, key: str) -> int:
    v = _req(d, key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise SchemaError(f"{key}: expected int, got {type(v).__name__}")
    return v


def _opt_float(value: Any, name: str) -> float | None:
    """None stays None. Never turns a missing value into zero."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{name}: expected number or null, got {type(value).__name__}")
    value = float(value)
    if math.isnan(value):
        return None
    return value


def _opt_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{name}: expected int or null, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# expression normalisation and hashing (used by memory and research_loop)
# ---------------------------------------------------------------------------

_OPERATOR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(")


def normalize_expression(expression: str) -> str:
    """Remove all whitespace and lowercase operator names only.

    An operator name is an identifier immediately followed by ``(``. Datafield names,
    constants and argument order are left exactly as written; nothing is sorted.
    """
    compact = re.sub(r"\s+", "", expression)
    return _OPERATOR_RE.sub(lambda m: m.group(1).lower() + "(", compact)


def experiment_hash(candidate: "CandidateRecord") -> str:
    """Deterministic hash of the normalised expression plus every simulation setting."""
    key = {
        "expression": normalize_expression(candidate.expression),
        "region": candidate.region,
        "universe": candidate.universe,
        "delay": candidate.delay,
        "decay": candidate.decay,
        "neutralization": candidate.neutralization,
        "truncation": candidate.truncation,
        "start_date": candidate.start_date.isoformat(),
        "end_date": candidate.end_date.isoformat(),
    }
    blob = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 2.1 CandidateRecord
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    expression: str
    region: str
    universe: str
    delay: int
    decay: int
    neutralization: str
    truncation: float
    start_date: date
    end_date: date
    hypothesis: str | None = None
    dataset: str | None = None
    family: str | None = None
    parent_id: str | None = None
    prompt_version: str | None = None
    model: str | None = None
    # Not in the spec table. Needed by research_loop (generation number) and repair
    # (levers tried along the parent chain). Both default to a fresh candidate.
    generation: int = 0
    lever: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateRecord":
        if not isinstance(d, dict):
            raise SchemaError(f"candidate must be an object, got {type(d).__name__}")
        truncation = _opt_float(_req(d, "truncation"), "truncation")
        return cls(
            candidate_id=_req_str(d, "candidate_id"),
            expression=_req_str(d, "expression").strip(),
            region=_req_str(d, "region"),
            universe=_req_str(d, "universe"),
            delay=_req_int(d, "delay"),
            decay=_req_int(d, "decay"),
            neutralization=_req_str(d, "neutralization"),
            truncation=truncation,
            start_date=_parse_date(_req(d, "start_date"), "start_date"),
            end_date=_parse_date(_req(d, "end_date"), "end_date"),
            hypothesis=_opt_str(d, "hypothesis"),
            dataset=_opt_str(d, "dataset"),
            family=_opt_str(d, "family"),
            parent_id=_opt_str(d, "parent_id"),
            prompt_version=_opt_str(d, "prompt_version"),
            model=_opt_str(d, "model"),
            generation=_opt_int(d.get("generation", 0), "generation") or 0,
            lever=_opt_str(d, "lever"),
        )

    def validate(self) -> None:
        if not self.candidate_id:
            raise SchemaError("candidate_id must be non-empty")
        if not self.expression or not self.expression.strip():
            raise SchemaError("expression must be non-empty")
        if self.region not in REGIONS:
            raise SchemaError(f"region {self.region!r} not in {sorted(REGIONS)}")
        allowed = REGION_SETTINGS[self.region]
        if self.universe not in allowed["universes"]:
            raise SchemaError(f"universe {self.universe!r} not in {self.region}'s "
                              f"{list(allowed['universes'])}")
        if self.delay not in allowed["delays"]:
            raise SchemaError(f"delay {self.delay} not allowed in {self.region} "
                              f"({list(allowed['delays'])})")
        if self.decay < 0:
            raise SchemaError(f"decay must be >= 0, got {self.decay}")
        if self.neutralization not in allowed["neutralizations"]:
            raise SchemaError(f"neutralization {self.neutralization!r} not in {self.region}'s "
                              f"{list(allowed['neutralizations'])}")
        if self.truncation is None or not (0 < self.truncation <= 1):
            raise SchemaError(f"truncation must be in (0, 1], got {self.truncation}")
        if self.end_date <= self.start_date:
            raise SchemaError("end_date must be after start_date")
        if self.generation < 0:
            raise SchemaError("generation must be >= 0")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat()
        d["end_date"] = self.end_date.isoformat()
        return d


# ---------------------------------------------------------------------------
# 2.4 YearRow
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YearRow:
    year: int
    sharpe: float | None = None
    turnover: float | None = None
    fitness: float | None = None
    returns: float | None = None
    max_drawdown: float | None = None
    margin_bps: float | None = None
    long_count: int | None = None
    short_count: int | None = None

    def validate(self) -> None:
        if isinstance(self.year, bool) or not isinstance(self.year, int):
            raise SchemaError(f"year must be int, got {self.year!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "YearRow":
        row = cls(
            year=_req_int(d, "year"),
            sharpe=_opt_float(d.get("sharpe"), "sharpe"),
            turnover=_opt_float(d.get("turnover"), "turnover"),
            fitness=_opt_float(d.get("fitness"), "fitness"),
            returns=_opt_float(d.get("returns"), "returns"),
            max_drawdown=_opt_float(d.get("max_drawdown"), "max_drawdown"),
            margin_bps=_opt_float(d.get("margin_bps"), "margin_bps"),
            long_count=_opt_int(d.get("long_count"), "long_count"),
            short_count=_opt_int(d.get("short_count"), "short_count"),
        )
        row.validate()
        return row


# ---------------------------------------------------------------------------
# 2.2 ResultRecord
# ---------------------------------------------------------------------------

_RESULT_FLOATS = (
    "sharpe",
    "fitness",
    "turnover",
    "returns",
    "margin_bps",
    "max_drawdown",
    "sub_universe_sharpe",
    "sub_universe_cutoff",
    "weight_concentration",
    "self_correlation",
    "long_pnl",
    "short_pnl",
)


@dataclass
class ResultRecord:
    candidate_id: str
    status: str
    brain_sim_id: str | None = None
    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    returns: float | None = None
    margin_bps: float | None = None
    max_drawdown: float | None = None
    drawdown_date: date | None = None
    sub_universe_sharpe: float | None = None
    sub_universe_cutoff: float | None = None
    weight_concentration: float | None = None
    self_correlation: float | None = None
    long_pnl: float | None = None
    short_pnl: float | None = None
    yearly: list[YearRow] = field(default_factory=list)
    daily_pnl: pd.Series | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    # Not in the spec table: metric-validator notes (section 4.2 step 6, "attach notes").
    notes: list[str] = field(default_factory=list)
    # Not in the spec table (consultant level, live capture 2026-11-10): BRAIN's value and
    # limit per check, {name: {"value": x, "limit": y}} (either may be None), whatever the
    # result; and the same in-sample stats under BRAIN's risk neutralization and investability
    # constraints, {"risk_neutralized": {"sharpe": ..., ...}, "investability_constrained": ...}.
    check_values: dict[str, dict] = field(default_factory=dict)
    alt_stats: dict[str, dict] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.candidate_id:
            raise SchemaError("candidate_id is required")
        if self.status not in RESULT_STATUSES:
            raise SchemaError(f"status {self.status!r} not in {sorted(RESULT_STATUSES)}")
        for name in _RESULT_FLOATS:
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise SchemaError(f"{name}: expected number or None, got {v!r}")
        for row in self.yearly:
            if not isinstance(row, YearRow):
                raise SchemaError("yearly must contain YearRow items")
            row.validate()
        if self.daily_pnl is not None and not isinstance(self.daily_pnl, pd.Series):
            raise SchemaError("daily_pnl must be a pandas Series or None")
        if not isinstance(self.checks, dict) or not all(
            isinstance(k, str) and isinstance(v, bool) for k, v in self.checks.items()
        ):
            raise SchemaError("checks must be a dict of str to bool")
        for name in ("check_values", "alt_stats"):
            _check_number_maps(getattr(self, name), name)

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["drawdown_date"] = self.drawdown_date.isoformat() if self.drawdown_date else None
        d["check_values"] = {k: dict(v) for k, v in self.check_values.items()}
        d["alt_stats"] = {k: dict(v) for k, v in self.alt_stats.items()}
        d["yearly"] = [r.to_dict() for r in self.yearly]
        d["daily_pnl"] = series_to_dict(self.daily_pnl)
        d["checks"] = dict(self.checks)
        d["notes"] = list(self.notes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ResultRecord":
        rec = cls(
            candidate_id=_req_str(d, "candidate_id"),
            status=_req_str(d, "status"),
            brain_sim_id=_opt_str(d, "brain_sim_id"),
            drawdown_date=_opt_date(d.get("drawdown_date"), "drawdown_date"),
            yearly=[YearRow.from_dict(r) for r in d.get("yearly") or []],
            daily_pnl=series_from_dict(d.get("daily_pnl")),
            checks=dict(d.get("checks") or {}),
            notes=list(d.get("notes") or []),
            check_values={k: dict(v) for k, v in (d.get("check_values") or {}).items()},
            alt_stats={k: dict(v) for k, v in (d.get("alt_stats") or {}).items()},
            **{name: _opt_float(d.get(name), name) for name in _RESULT_FLOATS},
        )
        rec.validate()
        return rec


def _check_number_maps(d, name: str) -> None:
    """{str: {str: number or None}}"""
    if not isinstance(d, dict) or not all(
        isinstance(k, str) and isinstance(v, dict) and all(
            isinstance(kk, str) and (vv is None or (isinstance(vv, (int, float))
                                                    and not isinstance(vv, bool)))
            for kk, vv in v.items())
        for k, v in d.items()
    ):
        raise SchemaError(f"{name} must be a dict of str to {{str: number or None}}")


def series_to_dict(s: pd.Series | None) -> dict | None:
    if s is None:
        return None
    return {
        "dates": [pd.Timestamp(i).date().isoformat() for i in s.index],
        "values": [None if pd.isna(v) else float(v) for v in s.values],
    }


def series_from_dict(d: dict | None) -> pd.Series | None:
    if d is None:
        return None
    if not isinstance(d, dict) or "dates" not in d or "values" not in d:
        raise SchemaError("daily_pnl must be an object with 'dates' and 'values'")
    if len(d["dates"]) != len(d["values"]):
        raise SchemaError("daily_pnl dates and values differ in length")
    values = [float("nan") if v is None else float(v) for v in d["values"]]
    return pd.Series(values, index=pd.DatetimeIndex(pd.to_datetime(d["dates"])), dtype=float)


# ---------------------------------------------------------------------------
# 2.3 DecisionRecord
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """``passed=None`` means "not evaluable", never a fail."""

    name: str
    passed: bool | None
    value: Any = None
    threshold: Any = None
    note: str = ""

    def to_dict(self) -> dict:
        def plain(x: Any) -> Any:
            if isinstance(x, (date, datetime)):
                return x.isoformat()
            if isinstance(x, float) and math.isnan(x):
                return None
            return x

        return {
            "name": self.name,
            "passed": self.passed,
            "value": plain(self.value),
            "threshold": plain(self.threshold),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GateResult":
        return cls(d["name"], d.get("passed"), d.get("value"), d.get("threshold"), d.get("note", ""))


@dataclass(frozen=True)
class DecisionRecord:
    candidate_id: str
    decision: str
    failure_class: str
    gate_results: list[GateResult] = field(default_factory=list)
    risk_veto: bool = False
    risk_reasons: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def validate(self) -> None:
        if not self.candidate_id:
            raise SchemaError("candidate_id is required")
        if self.decision not in DECISIONS:
            raise SchemaError(f"decision {self.decision!r} not in {sorted(DECISIONS)}")
        if self.failure_class not in FAILURE_CLASSES:
            raise SchemaError(f"failure_class {self.failure_class!r} not allowed")
        if not all(isinstance(g, GateResult) for g in self.gate_results):
            raise SchemaError("gate_results must contain GateResult items")
        if self.created_at.tzinfo is None:
            raise SchemaError("created_at must be timezone-aware UTC")

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "failure_class": self.failure_class,
            "gate_results": [g.to_dict() for g in self.gate_results],
            "risk_veto": self.risk_veto,
            "risk_reasons": list(self.risk_reasons),
            "reasons": list(self.reasons),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionRecord":
        rec = cls(
            candidate_id=d["candidate_id"],
            decision=d["decision"],
            failure_class=d["failure_class"],
            gate_results=[GateResult.from_dict(g) for g in d.get("gate_results", [])],
            risk_veto=bool(d.get("risk_veto", False)),
            risk_reasons=list(d.get("risk_reasons", [])),
            reasons=list(d.get("reasons", [])),
            created_at=datetime.fromisoformat(d["created_at"]),
        )
        rec.validate()
        return rec
