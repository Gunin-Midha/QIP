"""Load and validate config/thresholds.yaml (spec section 5).

Every policy threshold may be ``null``. A ``null`` threshold means "not configured":
the check that uses it returns ``passed=None`` with note ``"not configured"``.
Missing keys are an error; ``null`` values are not.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

NOT_CONFIGURED = "not configured"
TRIALS_SCOPES = ("lineage", "dataset", "all")


class ConfigError(ValueError):
    """Raised for missing keys, wrong types, out-of-range values, or an unset holdout."""


@dataclass(frozen=True)
class GatesConfig:
    sharpe_min: float | None
    fitness_min: float | None
    turnover_min: float | None
    turnover_max: float | None
    self_correlation_max: float | None
    use_brain_checks: bool = False   # BRAIN's PASS/FAIL checks as hard gates (evaluator.py)
    brain_checks_ignored: tuple[str, ...] = ()   # BRAIN checks that are never gates
    # our own gates use the limit BRAIN reports for that alpha's check when it has one
    # (LOW_SHARPE -> sharpe_min, ...): BRAIN's limits differ by region and delay
    limits_from_brain: bool = False
    # owner 2026-11-14: an alpha passes on BRAIN's checks alone. Our five gates take BRAIN's
    # verdict for the matching check when BRAIN reports it; the overfitting, pool-diversity
    # and risk checks are recorded but never change the decision (pipeline.py)
    brain_checks_only: bool = False


# gate keys a region override may set, with their allowed range
REGION_GATE_KEYS = {"sharpe_min": (None, None), "fitness_min": (None, None),
                    "turnover_min": (0.0, 1.0), "turnover_max": (0.0, 1.0),
                    "self_correlation_max": (-1.0, 1.0)}
_REGION_KEY = re.compile(r"^[A-Z]{3}(/D\d)?$")


@dataclass(frozen=True)
class RiskConfig:
    max_drawdown: float | None
    weight_concentration_max: float | None
    leg_balance_min_ratio: float | None
    top_n_pnl_days_removed: int | None
    min_sharpe_after_top_pnl_removal: float | None
    weak_year_sharpe_floor: float | None
    max_yearly_sharpe_spread: float | None
    stress_windows: tuple[tuple[date, date], ...] = ()


@dataclass(frozen=True)
class DiversityConfig:
    max_pool_correlation: float | None
    min_marginal_pool_sharpe_gain: float | None


@dataclass(frozen=True)
class AuditorConfig:
    os_is_ratio_floor: float | None
    deflation_method: str
    # an accept is rejected when its Sharpe, deflated for the number of simulations in its
    # repair lineage (auditor.py), is not above this; null = not checked
    min_deflated_sharpe: float | None = None
    # which simulations count as trials against an accept: "lineage" (its repair family),
    # "dataset" (every simulation on the same dataset) or "all"
    trials_scope: str = "dataset"


@dataclass(frozen=True)
class HoldoutConfig:
    start_date: date | None
    end_date: date | None

    @property
    def configured(self) -> bool:
        return self.start_date is not None and self.end_date is not None


@dataclass(frozen=True)
class SimulatorConfig:
    max_poll_seconds: float
    poll_interval_seconds: float
    max_retries: int
    # simulations kept running on BRAIN at once (Simulator.run_many; one thread that polls
    # each in turn). OQ-3: the account's real limit is unknown.
    max_concurrent: int = 1


# failure classes whose repair can be limited to near misses, and the gate each one reads
NEAR_MISS_CLASSES = ("sharpe", "fitness", "recent_sharpe")


@dataclass(frozen=True)
class RepairPolicy:
    # failure class -> share of the gate threshold the failed value must reach (by absolute
    # value, so a sign flip can rescue a negative Sharpe) for the candidate to go to repair
    # instead of reject. A class not listed (or a null share) has no such limit.
    near_miss_share: dict = field(default_factory=dict)
    # BRAIN's risk-neutralized Sharpe this much above the raw one sends a failed Sharpe to
    # the factor neutralization levers first (evaluator.sharpe_route). None = off.
    risk_neutralized_gap: float | None = None


@dataclass(frozen=True)
class Config:
    gates: GatesConfig
    risk: RiskConfig
    diversity: DiversityConfig
    auditor: AuditorConfig
    holdout: HoldoutConfig
    simulator: SimulatorConfig
    source: str = field(default="<dict>", compare=False)
    repair: RepairPolicy = field(default_factory=RepairPolicy)
    # "USA" or "USA/D0" -> gate overrides ({key: value}); the region/delay key wins
    regions: dict = field(default_factory=dict, compare=False)

    def for_settings(self, region: str, delay: int | None = None) -> "Config":
        """This config with the gate overrides for ``region`` (then ``region/D<delay>``)."""
        over = {**self.regions.get(region, {}), **self.regions.get(f"{region}/D{delay}", {})}
        if not over:
            return self
        return dataclasses.replace(self, gates=dataclasses.replace(self.gates, **over))

    def require_holdout(self) -> HoldoutConfig:
        if not self.holdout.configured:
            raise ConfigError(
                "holdout.start_date and holdout.end_date must be set before any simulation "
                "runs (OQ-9)"
            )
        return self.holdout


# ---------------------------------------------------------------------------

def _section(raw: dict, name: str) -> dict:
    if name not in raw:
        raise ConfigError(f"missing required section {name!r}")
    sec = raw[name]
    if not isinstance(sec, dict):
        raise ConfigError(f"section {name!r} must be a mapping")
    return sec


def _key(sec: dict, section: str, key: str) -> Any:
    if key not in sec:
        raise ConfigError(f"missing required key {section}.{key}")
    return sec[key]


def _num(sec: dict, section: str, key: str, *, lo: float | None = None,
         hi: float | None = None, nullable: bool = True) -> float | None:
    v = _key(sec, section, key)
    if v is None:
        if nullable:
            return None
        raise ConfigError(f"{section}.{key} must not be null")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{section}.{key} must be a number or null, got {v!r}")
    v = float(v)
    if lo is not None and v < lo:
        raise ConfigError(f"{section}.{key}={v} is below {lo}")
    if hi is not None and v > hi:
        raise ConfigError(f"{section}.{key}={v} is above {hi}")
    return v


def _int(sec: dict, section: str, key: str, *, lo: int, nullable: bool) -> int | None:
    v = _key(sec, section, key)
    if v is None:
        if nullable:
            return None
        raise ConfigError(f"{section}.{key} must not be null")
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{section}.{key} must be an integer, got {v!r}")
    if v < lo:
        raise ConfigError(f"{section}.{key}={v} is below {lo}")
    return v


def _opt_bool(sec: dict, section: str, key: str) -> bool:
    """Optional true/false key; missing means false (configs written before the key)."""
    v = sec.get(key, False)
    if not isinstance(v, bool):
        raise ConfigError(f"{section}.{key} must be true or false")
    return v


def _opt_names(sec: dict, section: str, key: str) -> tuple[str, ...]:
    """Optional list of names; missing means none."""
    v = sec.get(key) or []
    if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
        raise ConfigError(f"{section}.{key} must be a list of names")
    return tuple(v)


def _date(sec: dict, section: str, key: str) -> date | None:
    v = _key(sec, section, key)
    if v is None:
        return None
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return date.fromisoformat(v)
        except ValueError as exc:
            raise ConfigError(f"{section}.{key}: cannot parse date {v!r}") from exc
    raise ConfigError(f"{section}.{key} must be an ISO date or null, got {v!r}")


def config_from_dict(raw: Any, source: str = "<dict>") -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    g = _section(raw, "gates")
    gates = GatesConfig(
        sharpe_min=_num(g, "gates", "sharpe_min"),
        fitness_min=_num(g, "gates", "fitness_min"),
        turnover_min=_num(g, "gates", "turnover_min", lo=0.0, hi=1.0),
        turnover_max=_num(g, "gates", "turnover_max", lo=0.0, hi=1.0),
        self_correlation_max=_num(g, "gates", "self_correlation_max", lo=-1.0, hi=1.0),
        use_brain_checks=_opt_bool(g, "gates", "use_brain_checks"),
        brain_checks_ignored=_opt_names(g, "gates", "brain_checks_ignored"),
        limits_from_brain=_opt_bool(g, "gates", "limits_from_brain"),
        brain_checks_only=_opt_bool(g, "gates", "brain_checks_only"),
    )
    if (gates.turnover_min is not None and gates.turnover_max is not None
            and gates.turnover_min > gates.turnover_max):
        raise ConfigError("gates.turnover_min must be <= gates.turnover_max")

    r = _section(raw, "risk")
    windows = []
    for i, w in enumerate(r.get("stress_windows") or []):
        if not isinstance(w, (list, tuple)) or len(w) != 2:
            raise ConfigError(f"risk.stress_windows[{i}] must be [start_date, end_date]")
        pair = {"start": w[0], "end": w[1]}
        start = _date(pair, f"risk.stress_windows[{i}]", "start")
        end = _date(pair, f"risk.stress_windows[{i}]", "end")
        if start is None or end is None or end < start:
            raise ConfigError(f"risk.stress_windows[{i}] must be two ordered dates")
        windows.append((start, end))
    risk = RiskConfig(
        max_drawdown=_num(r, "risk", "max_drawdown", lo=0.0),
        weight_concentration_max=_num(r, "risk", "weight_concentration_max", lo=0.0),
        leg_balance_min_ratio=_num(r, "risk", "leg_balance_min_ratio", lo=0.0, hi=1.0),
        top_n_pnl_days_removed=_int(r, "risk", "top_n_pnl_days_removed", lo=0, nullable=True),
        min_sharpe_after_top_pnl_removal=_num(r, "risk", "min_sharpe_after_top_pnl_removal"),
        weak_year_sharpe_floor=_num(r, "risk", "weak_year_sharpe_floor"),
        max_yearly_sharpe_spread=_num(r, "risk", "max_yearly_sharpe_spread", lo=0.0),
        stress_windows=tuple(windows),
    )

    dv = _section(raw, "diversity")
    diversity = DiversityConfig(
        max_pool_correlation=_num(dv, "diversity", "max_pool_correlation", lo=-1.0, hi=1.0),
        min_marginal_pool_sharpe_gain=_num(dv, "diversity", "min_marginal_pool_sharpe_gain"),
    )

    a = _section(raw, "auditor")
    method = _key(a, "auditor", "deflation_method")
    if method != "expected_max_normal":
        raise ConfigError(
            f"auditor.deflation_method must be 'expected_max_normal', got {method!r}"
        )
    auditor = AuditorConfig(
        os_is_ratio_floor=_num(a, "auditor", "os_is_ratio_floor"),
        deflation_method=method,
        min_deflated_sharpe=_num(a, "auditor", "min_deflated_sharpe")
        if "min_deflated_sharpe" in a else None,
        trials_scope=a.get("trials_scope", "dataset"),
    )
    if auditor.trials_scope not in TRIALS_SCOPES:
        raise ConfigError(f"auditor.trials_scope must be one of {list(TRIALS_SCOPES)}")

    h = _section(raw, "holdout")
    holdout = HoldoutConfig(_date(h, "holdout", "start_date"), _date(h, "holdout", "end_date"))
    if (holdout.start_date is None) != (holdout.end_date is None):
        raise ConfigError("holdout.start_date and holdout.end_date must both be set or both null")
    if holdout.configured and holdout.end_date <= holdout.start_date:
        raise ConfigError("holdout.end_date must be after holdout.start_date")

    s = _section(raw, "simulator")
    simulator = SimulatorConfig(
        max_poll_seconds=_num(s, "simulator", "max_poll_seconds", lo=0.0, nullable=False),
        poll_interval_seconds=_num(s, "simulator", "poll_interval_seconds", lo=0.001,
                                   nullable=False),
        max_retries=_int(s, "simulator", "max_retries", lo=0, nullable=False),
        max_concurrent=_int(s, "simulator", "max_concurrent", lo=1, nullable=False)
        if "max_concurrent" in s else 1,
    )

    # optional section: configs written before it existed keep working (no near-miss limit)
    rp = raw.get("repair") or {}
    if not isinstance(rp, dict):
        raise ConfigError("section 'repair' must be a mapping")
    shares = rp.get("near_miss_share") or {}
    if not isinstance(shares, dict):
        raise ConfigError("repair.near_miss_share must be a mapping of failure class to share")
    near_miss = {}
    for cls in shares:
        if cls not in NEAR_MISS_CLASSES:
            raise ConfigError(f"repair.near_miss_share: {cls!r} not in {list(NEAR_MISS_CLASSES)}")
        v = _num(shares, "repair.near_miss_share", cls, lo=0.0, hi=1.0)
        if v is not None:
            near_miss[cls] = v
    rn_gap = _num(rp, "repair", "risk_neutralized_gap", lo=0.0) \
        if "risk_neutralized_gap" in rp else None

    # optional section: per-region (or region/delay) gate overrides
    rg = raw.get("regions") or {}
    if not isinstance(rg, dict):
        raise ConfigError("section 'regions' must be a mapping of REGION or REGION/D<delay>")
    regions = {}
    for key, over in rg.items():
        if not isinstance(key, str) or not _REGION_KEY.match(key):
            raise ConfigError(f"regions: key {key!r} must look like USA or USA/D0")
        over = over or {}
        if not isinstance(over, dict) or set(over) - set(REGION_GATE_KEYS):
            raise ConfigError(f"regions.{key} may set only {sorted(REGION_GATE_KEYS)}")
        regions[key] = {k: _num(over, f"regions.{key}", k, lo=lo, hi=hi)
                        for k, (lo, hi) in REGION_GATE_KEYS.items() if k in over}

    return Config(gates, risk, diversity, auditor, holdout, simulator, source=source,
                  repair=RepairPolicy(near_miss, rn_gap), regions=regions)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc
    return config_from_dict(raw, source=str(path))
