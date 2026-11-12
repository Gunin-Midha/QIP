"""BRAIN API client: session, RealBrainClient, response translation, catalogs.

This is the only module with network code. Everything else depends on the ``BrainClient``
protocol in brain_client.py, and the tests drive this module through ``FakeTransport``
style objects, so no test touches the network. Only the standard library is used
(urllib, http.cookiejar); there is no new dependency.

Source of the API shape: BRAIN's own web app (platform.worldquantbrain.com), inspected on
2026-11-02. The endpoints, headers and poll handling below are the ones its bundled
JavaScript uses, and tests/fixtures/brain_api_samples/ holds the sample responses its editor
walkthrough ships with. The mapping was then confirmed against a live capture from our
account on 2026-11-03 (OQ-2), saved in tests/fixtures/brain_api_capture/.

Endpoints used (relative to API_ROOT; every request sends Accept: application/json;version=2.0):

    POST    /authentication                   HTTP Basic -> session cookie; 401 with
                                              WWW-Authenticate: persona = biometric step
    OPTIONS /simulations                      allowed settings: actions.POST.settings.children
    POST    /simulations                      201, Location: .../simulations/<sim id>;
                                              400 {"settings": {field: [messages]}}; 429 limit
    GET     /simulations/<sim id>             {"progress": p} while running (plus Retry-After),
                                              then {"status": COMPLETE|WARNING, "alpha"} or
                                              ERROR|FAIL|TIMEOUT with "message", or "detail(s)"
    GET     /alphas/<alpha id>                {"settings": {.., "testPeriod"}, "is": {sharpe,
                                              fitness, turnover, returns, drawdown, margin,
                                              longCount, shortCount, startDate, checks: [{name,
                                              result, limit, value}]}, "test": {...} or null}
    GET     /alphas/<alpha id>/recordsets/<name>   {"schema": {"properties": [{name, type}]},
                                              "records": [[...]]}; names: pnl (cumulative),
                                              daily-pnl, sharpe, turnover, yearly-stats
    GET     /alphas/<alpha id>/correlations/self   {"max", "min", ...} (Retry-After while computing)
    GET     /alphas/<alpha id>/check          {"is": {"checks": [...]}}: the final submission
                                              checks (Retry-After while computing; consultant)
    GET     /operators                        [{name, category, definition, ...}]
    GET     /data-sets?...                    {"count", "results": [{id, fieldCount, ...}]}
    GET     /data-fields?...                  {"count", "results": [{id, ...}]}; BRAIN serves at
                                              most 10,000 per query (offset cap), so a larger
                                              catalog is fetched one dataset at a time

Units: turnover, returns and drawdown are fractions (schema type "percent") and margin is a
fraction (schema type "permyriad", shown as bps). ``translate`` keeps fractions (marking the
dict with ``"units": "fraction"`` so the simulator skips its percent conversion) and turns
margin into bps.

Credentials are read from the BRAIN_EMAIL and BRAIN_PASSWORD environment variables and
nowhere else. They are never stored, logged or written to a capture.

Saved session (``session_file`` in config/brain.yaml): after a login (with its biometric
check) the session cookie is saved to that file, outside the repository, readable only by
the user (0600), and the next command reuses it, so BRAIN asks for verification again only
when it no longer accepts the session (a 401 logs in again). ``cli brain logout`` deletes
it. The file holds the session cookie only, never the email or password.
"""

from __future__ import annotations

import base64
import http.cookiejar
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import yaml

from .brain_client import SubmissionRejected
from .config import ConfigError
from .schemas import CandidateRecord

API_ROOT = "https://api.worldquantbrain.com"
EMAIL_ENV_VAR = "BRAIN_EMAIL"
SECRET_ENV_VAR = "BRAIN_PASSWORD"
ACCEPT = "application/json;version=2.0"
DATES_NOTE = ("brain_dates: candidate start/end dates are not sent; BRAIN simulates its own "
              "period, which the simulator checks against the holdout (OQ-12)")
DATAFIELD_PAGE_SIZE = 50
DATAFIELD_OFFSET_CAP = 10_000      # BRAIN returns nothing past this offset (live, 2026-11-10)

_DONE_STATUSES = {"COMPLETE", "WARNING"}
_FAILED_STATUSES = {"ERROR", "FAIL", "TIMEOUT"}


class BrainApiError(RuntimeError):
    """A non-transient API error: unexpected status code or unusable body."""


class BrainAuthError(BrainApiError):
    """Missing credentials, wrong credentials, or a login step we cannot do here."""


# ---------------------------------------------------------------------------
# config (config/brain.yaml): connection settings, not policy thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BrainApiConfig:
    settings: dict                       # sent with every simulation besides the candidate's own
    request_timeout_seconds: float = 60.0
    recordset_max_wait_seconds: float = 300.0
    rate_limit_max_wait_seconds: float = 900.0
    fetch_self_correlation: bool = True
    fetch_submission_check: bool = True      # GET /alphas/<id>/check: BRAIN's final checks
    check_max_wait_seconds: float = 600.0
    session_file: Path | None = None         # saved session cookie (outside the repo); None = off
    # BRAIN failure messages that are BRAIN's own trouble, not the alpha's: the simulator
    # submits such a candidate once more (run rs20261111174623)
    retry_error_messages: tuple[str, ...] = ("There was an error while running the simulation",)


_CANDIDATE_SETTINGS = {"region", "universe", "delay", "decay", "neutralization", "truncation"}


def brain_config_from_dict(raw, source: str = "<dict>") -> BrainApiConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: brain config must be a mapping")
    settings = raw.get("settings")
    if not isinstance(settings, dict) or not settings:
        raise ConfigError(f"{source}: 'settings' must be a non-empty mapping")
    clash = _CANDIDATE_SETTINGS & set(settings)
    if clash:
        raise ConfigError(f"{source}: settings {sorted(clash)} come from each candidate, "
                          "remove them from the brain config")
    kw = {}
    for key in ("request_timeout_seconds", "recordset_max_wait_seconds",
                "rate_limit_max_wait_seconds", "check_max_wait_seconds"):
        if key in raw:
            v = raw[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                raise ConfigError(f"{source}: {key} must be a positive number, got {v!r}")
            kw[key] = float(v)
    for key in ("fetch_self_correlation", "fetch_submission_check"):
        if key in raw:
            if not isinstance(raw[key], bool):
                raise ConfigError(f"{source}: {key} must be true or false")
            kw[key] = raw[key]
    if raw.get("session_file") is not None:
        kw["session_file"] = session_path(raw["session_file"], source)
    if "retry_error_messages" in raw:
        msgs = raw["retry_error_messages"] or []
        if not isinstance(msgs, list) or not all(isinstance(m, str) and m for m in msgs):
            raise ConfigError(f"{source}: retry_error_messages must be a list of strings")
        kw["retry_error_messages"] = tuple(msgs)
    unknown = set(raw) - {"settings", "session_file", *kw}
    if unknown:
        raise ConfigError(f"{source}: unknown keys {sorted(unknown)}")
    return BrainApiConfig(settings=dict(settings), **kw)


REPO_ROOT = Path(__file__).resolve().parents[3]


def session_path(value, source: str = "<dict>") -> Path:
    """The saved-session file: ``~`` expanded, and never inside the repository (rule 4)."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{source}: session_file must be a path or null")
    path = Path(value).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ConfigError(f"{source}: session_file {path} is inside the repository; "
                          "put it outside, e.g. ~/.config/qip/brain_session.txt")
    return path


def load_brain_config(path: str | Path) -> BrainApiConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read brain config {path}: {exc}") from exc
    return brain_config_from_dict(raw, str(path))


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

@dataclass
class Response:
    status: int
    headers: dict = field(default_factory=dict)     # keys lowercased
    body: bytes = b""

    def json(self):
        if not self.body:
            return None
        try:
            return json.loads(self.body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BrainApiError(f"response is not JSON (status {self.status})") from exc

    def retry_after(self) -> float | None:
        try:
            v = float(self.headers.get("retry-after", ""))
        except ValueError:
            return None
        return v if v > 0 else None

    def detail(self) -> str:
        try:
            body = self.json()
        except BrainApiError:
            return self.body[:200].decode(errors="replace")
        if isinstance(body, dict):
            if isinstance(body.get("settings"), dict):  # 400 on POST /simulations
                return "; ".join(f"{k}: {' '.join(map(str, v)) if isinstance(v, list) else v}"
                                 for k, v in body["settings"].items())[:300]
            return str(body.get("detail") or body.get("details") or body.get("message") or body)[:300]
        return str(body)[:300]


class Transport(Protocol):
    def request(self, method: str, url: str, body: bytes | None,
                headers: dict) -> Response: ...


class UrllibTransport:
    """Keeps the session cookie in a CookieJar, loaded from and saved to ``session_file``
    when one is given. Network failures become ConnectionError or TimeoutError, which the
    simulator retries."""

    def __init__(self, timeout: float = 60.0, session_file: Path | None = None):
        self.timeout = timeout
        self.session_file = session_file
        self._jar = http.cookiejar.LWPCookieJar()
        self.restored = False   # a saved session was loaded
        if session_file is not None and session_file.exists():
            try:
                self._jar.load(str(session_file), ignore_discard=True)  # expired ones dropped
                self.restored = len(self._jar) > 0
            except (OSError, http.cookiejar.LoadError):
                self._jar.clear()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self._jar))

    def save_session(self) -> None:
        """Write the session cookie to ``session_file``, readable by the user only."""
        if self.session_file is None:
            return
        self.session_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.session_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.close(fd)   # created 0600 before anything is written
        self._jar.save(str(self.session_file), ignore_discard=True)
        os.chmod(self.session_file, 0o600)

    def forget_session(self) -> bool:
        """Delete the saved session. Returns whether there was one."""
        self._jar.clear()
        if self.session_file is not None and self.session_file.exists():
            self.session_file.unlink()
            return True
        return False

    def request(self, method: str, url: str, body: bytes | None, headers: dict) -> Response:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                return Response(r.status, _lower(r.headers), r.read())
        except urllib.error.HTTPError as exc:
            return Response(exc.code, _lower(exc.headers), exc.read())
        except TimeoutError as exc:
            raise TimeoutError(f"{method} {url} timed out") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError(f"{method} {url} timed out") from exc
            raise ConnectionError(f"{method} {url} failed: {exc.reason}") from exc


def _lower(headers) -> dict:
    return {k.lower(): v for k, v in (headers or {}).items()}


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------

def credentials_from_env() -> tuple[str, str]:
    email, secret = os.environ.get(EMAIL_ENV_VAR), os.environ.get(SECRET_ENV_VAR)
    if not email or not secret:
        raise BrainAuthError(f"{EMAIL_ENV_VAR} and {SECRET_ENV_VAR} must both be set")
    return email, secret


class BrainSession:
    """Authenticated requests. Logs in lazily, logs in again once on a 401 (sessions
    expire), waits out 429 rate limits up to ``rate_limit_max_wait`` and turns 5xx into
    ConnectionError so callers retry."""

    def __init__(self, transport: Transport, email: str, secret: str, *,
                 sleep: Callable[[float], None] = time.sleep,
                 rate_limit_max_wait: float = 900.0, api_root: str = API_ROOT,
                 on_biometric: Callable[[str], None] | None = None):
        self.transport = transport
        self.on_biometric = on_biometric
        self._email = email
        self._secret = secret
        self.sleep = sleep
        self.rate_limit_max_wait = rate_limit_max_wait
        self.api_root = api_root.rstrip("/")
        # a saved session counts as logged in; BRAIN's 401 on the first request logs in again
        self.authenticated = bool(getattr(transport, "restored", False))
        self.restored = self.authenticated

    def url(self, path: str, params: dict | None = None) -> str:
        url = path if path.startswith(self.api_root) else self.api_root + "/" + path.lstrip("/")
        return url + ("?" + urllib.parse.urlencode(params) if params else "")

    def _send(self, method: str, url: str, json_body=None, extra: dict | None = None) -> Response:
        headers = {"Accept": ACCEPT, **(extra or {})}
        body = None
        if json_body is not None:
            body = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        return self.transport.request(method, url, body, headers)

    def authenticate(self) -> None:
        if self._email is None or self._secret is None:
            raise BrainAuthError(f"BRAIN needs a new login (the saved session is no longer "
                                 f"accepted): set {EMAIL_ENV_VAR} and {SECRET_ENV_VAR}")
        basic = base64.b64encode(f"{self._email}:{self._secret}".encode()).decode()
        resp = self._send("POST", self.url("/authentication"), extra={"Authorization": f"Basic {basic}"})
        if resp.status in (200, 201):
            self._logged_in()
            return
        if resp.status == 401 and resp.headers.get("www-authenticate", "").lower() == "persona":
            where = urllib.parse.urljoin(self.url("/authentication"), resp.headers.get("location", ""))
            if self.on_biometric is None:
                raise BrainAuthError(f"BRAIN asks for biometric verification, which needs an "
                                     f"interactive terminal. Open {where} in a browser, or run "
                                     f"the command in a terminal to be prompted.")
            # The web app's flow: the person verifies at the link, then the same session
            # POSTs to it to finish the login (each new login asks again).
            self.on_biometric(where)
            done = self._send("POST", where)
            if done.status in (200, 201):
                self._logged_in()
                return
            raise BrainAuthError(f"biometric verification not completed ({done.status}): "
                                 f"{done.detail()}")
        if resp.status >= 500:
            raise ConnectionError(f"authentication returned {resp.status}")
        raise BrainAuthError(f"authentication failed ({resp.status}): {resp.detail()}")

    def _logged_in(self) -> None:
        self.authenticated = True
        save = getattr(self.transport, "save_session", None)
        if save is not None:
            save()

    def ensure_login(self) -> None:
        """Log in unless already logged in or a saved session was loaded."""
        if not self.authenticated:
            self.authenticate()

    def request(self, method: str, path: str, *, json_body=None,
                params: dict | None = None) -> Response:
        if not self.authenticated:
            self.authenticate()
        url = self.url(path, params)
        waited, reauthed, attempt = 0.0, False, 0
        while True:
            resp = self._send(method, url, json_body)
            if resp.status == 401 and not reauthed:
                reauthed = True
                self.authenticate()
                continue
            if resp.status == 429:
                wait = resp.retry_after() or min(60.0, 5.0 * 2 ** attempt)
                attempt += 1
                if waited + wait > self.rate_limit_max_wait:
                    raise ConnectionError(f"{method} {path} still rate limited after {waited:.0f}s")
                self.sleep(wait)
                waited += wait
                continue
            if resp.status >= 500:
                raise ConnectionError(f"{method} {path} returned {resp.status}")
            return resp

    def get_ready(self, path: str, max_wait: float, params: dict | None = None) -> Response:
        """GET a resource BRAIN computes on demand: wait while it answers 2xx with a
        Retry-After header. Raises TimeoutError after ``max_wait`` seconds."""
        waited = 0.0
        while True:
            resp = self.request("GET", path, params=params)
            wait = resp.retry_after()
            if wait is None or resp.status >= 300:
                return resp
            if waited + wait > max_wait:
                raise TimeoutError(f"GET {path} not ready after {waited:.0f}s")
            self.sleep(wait)
            waited += wait


def make_session(config: BrainApiConfig, *, sleep: Callable[[float], None] = time.sleep,
                 transport: Transport | None = None,
                 on_biometric: Callable[[str], None] | None = None) -> BrainSession:
    """The real session. Fails fast (before any request) when credentials are missing and
    there is no saved session to reuse.
    ``on_biometric(url)`` is called when BRAIN asks for biometric verification and must
    return once the person has verified at ``url``; without it that login fails."""
    transport = transport or UrllibTransport(config.request_timeout_seconds,
                                             session_file=config.session_file)
    try:
        email, secret = credentials_from_env()
    except BrainAuthError:
        if not getattr(transport, "restored", False):
            raise
        email = secret = None   # the saved session is used; a new login would need them
    return BrainSession(transport, email, secret, sleep=sleep, rate_limit_max_wait=config.rate_limit_max_wait_seconds,
                        on_biometric=on_biometric)


def _expect(resp: Response, what: str, *ok: int):
    if resp.status not in ok:
        raise BrainApiError(f"{what} returned {resp.status}: {resp.detail()}")
    return resp.json()


# ---------------------------------------------------------------------------
# translation: BRAIN JSON -> the simulator's raw dict (see brain_client.py docstring)
# ---------------------------------------------------------------------------

def simulation_payload(candidate: CandidateRecord, config: BrainApiConfig) -> dict:
    return {
        "type": "REGULAR",
        "settings": {
            **config.settings,
            "region": candidate.region,
            "universe": candidate.universe,
            "delay": candidate.delay,
            "decay": candidate.decay,
            "neutralization": candidate.neutralization,
            "truncation": candidate.truncation,
        },
        "regular": candidate.expression,
    }


def recordset_rows(body) -> list[dict]:
    """``{"schema": {"properties": [{"name": ...}]}, "records": [[...]]}`` -> list of dicts."""
    if not isinstance(body, dict):
        return []
    props = (body.get("schema") or {}).get("properties") or []
    names = [p.get("name") for p in props if isinstance(p, dict)]
    rows = []
    for rec in body.get("records") or []:
        if isinstance(rec, list) and len(rec) == len(names):
            rows.append(dict(zip(names, rec)))
    return rows


def _num(value, key: str, notes: list[str]) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        notes.append(f"brain_field_not_numeric: {key}={value!r}")
        return None
    return None if math.isnan(value) else float(value)


def _bps(value: float | None) -> float | None:
    return None if value is None else round(value * 10_000, 6)  # drop float noise (4.5600000000000005)


def _int(value) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _yearly(rows: list[dict], notes: list[str]) -> list[dict]:
    """IS rows only. With a test period BRAIN may add rows for other stages."""
    out, other_stages = [], set()
    for row in rows:
        stage = row.get("stage")
        if stage is not None and stage != "IS":
            other_stages.add(str(stage))
            continue
        try:
            year = int(str(row.get("year"))[:4])
        except ValueError:
            notes.append(f"brain_yearly_row_skipped: year {row.get('year')!r}")
            continue
        out.append({
            "year": year,
            "sharpe": _num(row.get("sharpe"), "yearly.sharpe", notes),
            "turnover": _num(row.get("turnover"), "yearly.turnover", notes),
            "fitness": _num(row.get("fitness"), "yearly.fitness", notes),
            "returns": _num(row.get("returns"), "yearly.returns", notes),
            "drawdown": _num(row.get("drawdown"), "yearly.drawdown", notes),
            "margin": _bps(_num(row.get("margin"), "yearly.margin", notes)),
            "long_count": _int(row.get("longCount")),
            "short_count": _int(row.get("shortCount")),
        })
    if other_stages:
        notes.append(f"brain_yearly_rows_excluded: stages {sorted(other_stages)}")
    return out


def _pnl_points(rows: list[dict]) -> tuple[list[str], list[float | None]]:
    points = [(str(r.get("date")), r.get("pnl")) for r in rows if r.get("date") is not None]
    return ([d for d, _ in points],
            [v if isinstance(v, (int, float)) and not isinstance(v, bool) else None for _, v in points])


def _daily_pnl(daily_rows: list[dict], cumulative_rows: list[dict],
               notes: list[str]) -> dict | None:
    """Prefer the ``daily-pnl`` recordset. Otherwise difference the ``pnl`` recordset, which
    is cumulative (in BRAIN's sample its last value is within 0.02% of ``is.pnl``)."""
    dates, values = _pnl_points(daily_rows)
    source = "daily-pnl"
    if not dates:
        dates, cumulative = _pnl_points(cumulative_rows)
        source = "pnl (differenced)"
        values, prev = [], 0.0
        for v in cumulative:
            values.append(None if v is None else v - prev)
            prev = prev if v is None else v
    if not dates:
        return None
    notes.append(f"brain_pnl_period: {dates[0]}..{dates[-1]} from {source}")
    return {"dates": dates, "values": values}


def _failure_message(simulation: dict) -> str | None:
    """The web app's order: details, then an ERROR/FAIL/TIMEOUT status, then detail."""
    if simulation.get("details"):
        return str(simulation["details"])
    status = str(simulation.get("status", "")).upper()
    if status in _FAILED_STATUSES:
        return f"{status}: {simulation.get('message') or 'no message'}"
    if simulation.get("detail"):
        return str(simulation["detail"])
    if simulation.get("children"):
        return "multi-simulation responses are not supported"
    if not simulation.get("alpha"):
        return f"status {status or 'missing'} without an alpha id"
    return None


def translate(simulation: dict, alpha: dict | None, yearly: dict | None = None,
              pnl: dict | None = None, self_corr: dict | None = None, *,
              daily_pnl: dict | None = None, check: dict | None = None) -> dict:
    """Turn BRAIN responses into the raw dict ``simulator.normalize`` reads. Pure.
    Anything missing stays None; BRAIN checks that are neither PASS nor FAIL are left out
    of ``checks`` (missing is not false) and listed in notes.

    ``check`` is ``GET /alphas/<id>/check`` (the web app's "Check submission"). At
    consultant level the checks inside the alpha are provisional (live capture 2026-11-10:
    LOW_SHARPE was WARNING there and FAIL in the check, SELF_CORRELATION PENDING there and
    FAIL 0.7003 in the check), so the check's list replaces the alpha's when it has one."""
    simulation = simulation or {}
    notes = [DATES_NOTE]
    failure = _failure_message(simulation)
    if failure is not None or not alpha:
        return {"status": "failed",
                "notes": notes + [f"brain_simulation_failed: {failure or 'alpha not retrieved'}"]}
    if str(simulation.get("status", "")).upper() == "WARNING":
        notes.append(f"brain_simulation_warning: {simulation.get('message')}")

    is_ = alpha.get("is") or {}
    final = ((check or {}).get("is") or {}).get("checks")
    if isinstance(final, list) and final:
        notes.append("brain_checks_from: submission check")
    else:
        final = is_.get("checks")
        notes.append("brain_checks_from: simulation (provisional)")
    checks, by_name, check_values = {}, {}, {}
    for c in final or []:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str):
            continue
        by_name[c["name"]] = c
        vl = {k: _num(c.get(k), f"checks.{c['name']}.{k}", notes) for k in ("value", "limit")}
        if any(v is not None for v in vl.values()):
            check_values[c["name"]] = vl
        result = c.get("result")
        if result == "PASS":
            checks[c["name"]] = True
        elif result == "FAIL":
            checks[c["name"]] = False
        else:
            extra = "".join(f" {k} {c[k]}" for k in ("value", "limit") if c.get(k) is not None)
            notes.append(f"brain_check_not_counted: {c['name']}={result!r}{extra}")

    def check_field(name: str, key: str) -> float | None:
        return _num((by_name.get(name) or {}).get(key), f"checks.{name}.{key}", notes)

    self_correlation = _num((self_corr or {}).get("max"), "correlations.self.max", notes)
    if self_correlation is None:
        self_correlation = check_field("SELF_CORRELATION", "value")

    settings = alpha.get("settings") or {}
    if settings.get("testPeriod"):
        notes.append(f"brain_test_period: {settings['testPeriod']}")
    test = alpha.get("test")
    if isinstance(test, dict):
        notes.append(f"brain_test_stats: start {test.get('startDate')} sharpe {test.get('sharpe')} "
                     f"fitness {test.get('fitness')} (not gated)")
    if is_.get("startDate"):
        notes.append(f"brain_is_start: {is_['startDate']}")
    # consultant level: the same in-sample stats under BRAIN's investability constraints and
    # with its risk factors neutralized. Recorded, not gated.
    alt_stats = {}
    for key, label in (("investabilityConstrained", "investability_constrained"),
                       ("riskNeutralized", "risk_neutralized")):
        if isinstance(is_.get(key), dict):
            v = is_[key]
            alt_stats[label] = {k: _num(v.get(k), f"is.{key}.{k}", notes)
                                for k in ("sharpe", "fitness", "turnover", "returns", "drawdown")}
            notes.append(f"brain_{label}: sharpe {v.get('sharpe')} fitness {v.get('fitness')} "
                         f"turnover {v.get('turnover')} returns {v.get('returns')} (not gated)")
    if settings.get("startDate") and settings.get("endDate"):
        # read by simulator.brain_period, which checks it against the holdout
        notes.append(f"brain_period: {settings['startDate']}..{settings['endDate']}")

    return {
        "status": "complete",
        "units": "fraction",
        "sharpe": _num(is_.get("sharpe"), "is.sharpe", notes),
        "fitness": _num(is_.get("fitness"), "is.fitness", notes),
        "turnover": _num(is_.get("turnover"), "is.turnover", notes),
        "returns": _num(is_.get("returns"), "is.returns", notes),
        "drawdown": _num(is_.get("drawdown"), "is.drawdown", notes),
        "drawdown_date": None,      # TODO(open-question) OQ-2: no field in BRAIN's sample
        "margin": _bps(_num(is_.get("margin"), "is.margin", notes)),
        "sub_universe_sharpe": check_field("LOW_SUB_UNIVERSE_SHARPE", "value"),
        "sub_universe_cutoff": check_field("LOW_SUB_UNIVERSE_SHARPE", "limit"),
        # TODO(open-question) OQ-2: unit of CONCENTRATED_WEIGHT is unclear (sample: value
        # 0.2, limit 1). Stored as given. Our live capture had a PASS with no value at all,
        # so on a pass this is None (risk check "not evaluated"); BRAIN's PASS/FAIL is in checks.
        "weight_concentration": check_field("CONCENTRATED_WEIGHT", "value"),
        "self_correlation": self_correlation,
        "long_pnl": None,           # TODO(open-question) OQ-2: no long/short PnL in BRAIN's sample
        "short_pnl": None,
        "checks": checks,
        "check_values": check_values,
        "alt_stats": alt_stats,
        "yearly": _yearly(recordset_rows(yearly), notes),
        "daily_pnl": _daily_pnl(recordset_rows(daily_pnl), recordset_rows(pnl), notes),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# RealBrainClient
# ---------------------------------------------------------------------------

class RealBrainClient:
    """``BrainClient`` over the real API. ``submit`` returns the simulation id; ``poll``
    returns ``{"status": "running", "retry_after": s}`` until the simulation is done, then
    the translated result. With ``capture_dir`` every raw response body is saved under
    ``capture_dir/<sim id>/`` (never credentials or cookies)."""

    def __init__(self, session: BrainSession, config: BrainApiConfig, *,
                 capture_dir: str | Path | None = None):
        self.session = session
        self.config = config
        self.capture_dir = Path(capture_dir) if capture_dir else None

    def _capture(self, sim_id: str, name: str, body) -> None:
        if self.capture_dir is None:
            return
        d = self.capture_dir / sim_id
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(body, indent=2) + "\n")

    def submit(self, candidate: CandidateRecord) -> str:
        resp = self.session.request("POST", "/simulations",
                                    json_body=simulation_payload(candidate, self.config))
        if 400 <= resp.status < 500:
            raise SubmissionRejected(f"BRAIN rejected the simulation ({resp.status}): {resp.detail()}")
        if resp.status not in (200, 201, 202) or not resp.headers.get("location"):
            raise BrainApiError(f"submit returned {resp.status} without a Location header")
        return resp.headers["location"].rstrip("/").rsplit("/", 1)[-1]

    def poll(self, brain_sim_id: str) -> dict:
        resp = self.session.request("GET", f"/simulations/{brain_sim_id}")
        if resp.status == 404:
            return translate({"detail": "simulation not found"}, None)
        sim = _expect(resp, f"GET simulation {brain_sim_id}", 200) or {}
        wait = resp.retry_after()
        running = "progress" in sim and not sim.get("status")
        if wait is not None or running:
            return {"status": "running", "retry_after": wait, "progress": sim.get("progress")}
        self._capture(brain_sim_id, "simulation", sim)
        failure = _failure_message(sim)
        if failure is not None:
            raw = translate(sim, None)
            if any(m.lower() in failure.lower() for m in self.config.retry_error_messages):
                raw["retryable"] = True   # BRAIN's own error: the simulator submits once more
            return raw

        alpha_id = sim["alpha"]
        notes: list[str] = []
        base = f"/alphas/{alpha_id}"
        alpha = self._fetch(brain_sim_id, "alpha", base, notes)
        yearly = self._fetch(brain_sim_id, "yearly_stats", f"{base}/recordsets/yearly-stats", notes)
        daily = self._fetch(brain_sim_id, "daily_pnl", f"{base}/recordsets/daily-pnl", notes)
        pnl = (None if recordset_rows(daily)
               else self._fetch(brain_sim_id, "pnl", f"{base}/recordsets/pnl", notes))
        corr = (self._fetch(brain_sim_id, "self_correlation", f"{base}/correlations/self", notes)
                if self.config.fetch_self_correlation else None)
        check = (self._fetch(brain_sim_id, "check", f"{base}/check", notes,
                             max_wait=self.config.check_max_wait_seconds)
                 if self.config.fetch_submission_check else None)
        raw = translate(sim, alpha, yearly, pnl, corr, daily_pnl=daily, check=check)
        raw["notes"] = raw["notes"] + [f"brain_alpha_id: {alpha_id}"] + notes
        self._capture(brain_sim_id, "translated", raw)
        return raw

    def _fetch(self, sim_id: str, name: str, path: str, notes: list[str],
               max_wait: float | None = None):
        """A missing piece becomes None plus a note; it never fails the whole result."""
        try:
            resp = self.session.get_ready(path, max_wait or self.config.recordset_max_wait_seconds)
        except TimeoutError as exc:
            notes.append(f"brain_fetch_timeout: {name}: {exc}")
            return None
        if resp.status != 200:
            notes.append(f"brain_fetch_failed: {name} returned {resp.status}")
            return None
        body = resp.json()
        self._capture(sim_id, name, body)
        return body


def simulation_options(session: BrainSession) -> dict:
    """``OPTIONS /simulations`` -> ``{"common": {setting: spec}, "by_type": {type: {...}}}``,
    the same reduction the web app makes. Each spec is BRAIN's own (choices, limits)."""
    body = _expect(session.request("OPTIONS", "/simulations"), "OPTIONS simulations", 200) or {}
    post = (body.get("actions") or {}).get("POST") or {}
    by_type = {}
    for choice in (post.get("type") or {}).get("choices") or []:
        children = (choice.get("settings") or {}).get("children") if isinstance(choice, dict) else None
        if choice.get("value") and children:
            by_type[choice["value"]] = children
    return {"common": (post.get("settings") or {}).get("children") or {}, "by_type": by_type}


# ---------------------------------------------------------------------------
# catalogs
# ---------------------------------------------------------------------------

def list_operators(session: BrainSession) -> list[dict]:
    body = _expect(session.request("GET", "/operators"), "GET operators", 200)
    if isinstance(body, dict):
        body = body.get("results", [])
    if not isinstance(body, list):
        raise BrainApiError("operators response is not a list")
    return [o for o in body if isinstance(o, dict) and isinstance(o.get("name"), str)]


def _paged(session: BrainSession, path: str, params: dict, what: str,
           progress: Callable[[str], None] | None = None) -> tuple[list[dict], int | None]:
    """All pages of a ``{"count", "results"}`` listing, up to BRAIN's offset cap."""
    items, offset, count = [], 0, None
    while offset < DATAFIELD_OFFSET_CAP:
        body = _expect(session.request("GET", path, params={**params, "limit": DATAFIELD_PAGE_SIZE,
                                                            "offset": offset}), what, 200) or {}
        page = [f for f in body.get("results") or [] if isinstance(f, dict) and f.get("id")]
        items.extend(page)
        offset += DATAFIELD_PAGE_SIZE
        count = body.get("count") if isinstance(body.get("count"), int) else count
        if progress and (offset // DATAFIELD_PAGE_SIZE) % 20 == 0:
            total = f"{count:,}" if count is not None else "?"
            progress(f"{what}: {len(items):,} of {total} so far")
        if not page or (count is not None and offset >= count):
            break
    return items, count


def list_datasets(session: BrainSession, *, region: str, universe: str, delay: int,
                  instrument_type: str = "EQUITY") -> list[dict]:
    params = {"instrumentType": instrument_type, "region": region, "delay": delay,
              "universe": universe}
    return _paged(session, "/data-sets", params, "datasets")[0]


def list_datafields(session: BrainSession, *, region: str, universe: str, delay: int,
                    instrument_type: str = "EQUITY", dataset: str | None = None,
                    progress: Callable[[str], None] | None = None) -> list[dict]:
    """Every datafield for these settings. When BRAIN counts more than one query can page
    through (``DATAFIELD_OFFSET_CAP``; consultant catalogs do), it fetches one dataset at a
    time. ``progress(message)`` gets a line now and then for long downloads."""
    params = {"instrumentType": instrument_type, "region": region, "delay": delay,
              "universe": universe}
    if dataset:
        params["dataset.id"] = dataset
    fields_, count = _paged(session, "/data-fields", params, "datafields", progress)
    if dataset or count is None or count <= len(fields_):
        return fields_
    datasets = list_datasets(session, region=region, universe=universe, delay=delay,
                             instrument_type=instrument_type)
    if progress:
        progress(f"BRAIN has {count:,} fields, more than one query returns "
                 f"({DATAFIELD_OFFSET_CAP:,}); fetching {len(datasets)} datasets one by one")
    out, seen = [], set()
    for i, ds in enumerate(datasets, 1):
        part, part_count = _paged(session, "/data-fields", {**params, "dataset.id": ds["id"]},
                                  f"datafields {ds['id']}")
        if part_count is not None and part_count > len(part) and progress:
            progress(f"dataset {ds['id']}: only {len(part):,} of {part_count:,} fields returned")
        for f in part:
            if f["id"] not in seen:
                seen.add(f["id"])
                out.append(f)
        if progress and (i % 10 == 0 or i == len(datasets)):
            progress(f"datasets {i} of {len(datasets)} done, {len(out):,} fields so far")
    return out
