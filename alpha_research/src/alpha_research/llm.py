"""LLM client abstraction (Phase 3).

This is the only module that may talk to a model provider. Everything else depends on the
small ``LLMClient`` protocol, so the generator can be tested with ``ScriptedLLMClient``
and never needs a network.

* ``ScriptedLLMClient`` replays canned replies. Used by the tests and by the offline
  demo (``cli generate --llm scripted --llm-script replies.json``).
* ``AnthropicLLMClient`` calls the Anthropic Messages API. The SDK is an optional
  dependency (``pip install -e ".[llm]"``) and is imported only when the client is built.
  The key is read from the KEY_ENV_VAR environment variable and is never stored,
  logged or written to disk.
* ``GeminiLLMClient`` calls Google's Gemini ``generateContent`` REST endpoint with the
  standard library (``urllib``), so it needs no extra package. The key is read from
  GEMINI_KEY_ENV_VAR and sent only in the ``x-goog-api-key`` header.
* ``ChatCompletionsLLMClient`` calls a hosted chat-completions endpoint with ``urllib``
  for hosted models (Qwen, gpt-oss, Llama, DeepSeek...). The host comes from the model id:
  ``<host>:<maker>/<model>`` goes to that host (``groq``, ``nvidia``, ``github``), and a
  plain ``<maker>/<model>`` goes to OpenRouter (free variants end in ``:free``). With a
  host prefix the maker may be left out when the name says it (``groq:gpt-oss-120b``,
  ``groq:llama-3.3-70b-versatile``): BARE_MODEL_MAKERS gives the provider and, where the
  host wants it, the namespace sent to the host. Each host's key is read from its own
  environment variable (HOSTS) and sent only in the ``Authorization`` header.
  ``list_models(host)`` asks a host which models it serves (``cli models``).

Rate limits: when Gemini or a hosted endpoint answers HTTP 429 and says how long to wait
("try again in 7.95s", or Gemini's ``retryDelay``), the client waits that long and retries,
up to RATE_LIMIT_RETRIES times, as long as the wait is at most RATE_LIMIT_MAX_WAIT_SECONDS.
A longer wait (a daily quota) or a 429 without a wait fails as before. A temporary server
error (HTTP 500, 502, 503 or 504, such as Gemini's 503 "high demand") is retried after each
wait in SERVER_ERROR_WAITS_SECONDS (10 s, 30 s, 60 s) before it fails. The retries happen
inside one ``complete`` call, so the research budget counts them as one model call.

The provider is inferred from the model id (``provider_of``): ``claude-*`` is Anthropic,
``gemini-*`` is Google, and a hosted id belongs to its maker (``groq:qwen/...`` is
``qwen``), not to the host, because the critic rule is about who made the model. Agents
compare providers by this name, never by client class.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Protocol, Sequence


log = logging.getLogger(__name__)

KEY_ENV_VAR = "ANTHROPIC_API_KEY"
GEMINI_KEY_ENV_VAR = "GEMINI_API_KEY"
GEMINI_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TIMEOUT_SECONDS = 300.0   # thinking models can take minutes on a long prompt
HOSTED_TIMEOUT_SECONDS = 300.0
RATE_LIMIT_MAX_WAIT_SECONDS = 60.0   # a longer requested wait is a daily quota: give up
RATE_LIMIT_RETRIES = 3
SERVER_ERROR_STATUSES = (500, 502, 503, 504)
SERVER_ERROR_WAITS_SECONDS = (10.0, 30.0, 60.0)   # one retry after each wait
# host -> (chat-completions URL, key environment variable). The default host serves
# plain "<maker>/<model>" ids; other hosts are named as a "<host>:" prefix on the id.
HOSTS = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1/chat/completions", "NVIDIA_API_KEY"),
    "github": ("https://models.github.ai/inference/chat/completions", "GITHUB_MODELS_TOKEN"),
}
DEFAULT_HOST = "openrouter"
# where each host lists its models (default: the chat URL with chat/completions -> models)
MODEL_LIST_URLS = {"github": "https://models.github.ai/catalog/models"}
# a hosted id without "<maker>/": name prefix -> (provider, namespace the host expects or
# None to send the name as is). gpt-oss and GPT ids are namespaced on every host we use.
BARE_MODEL_MAKERS = {
    "gpt-": ("openai", "openai"),
    "llama": ("meta", None),
    "deepseek": ("deepseek", None),
    "qwen": ("qwen", None),
}
# host namespaces that name the same maker differently
MAKER_ALIASES = {"meta-llama": "meta", "deepseek-ai": "deepseek"}
# Groq sits behind Cloudflare, which refuses urllib's default User-Agent (error code 1010)
USER_AGENT = "qip-alpha-research/0.1"
# reasoning models (e.g. Qwen on Groq) may put their thinking inline; it is not the reply
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)

PROVIDER_PREFIXES = {"claude-": "anthropic", "gemini-": "google"}


def provider_of(model: str) -> str | None:
    """Provider for a model id, from its prefix. None when the prefix is unknown."""
    if model == "scripted":
        return "scripted"
    if not isinstance(model, str):
        return None
    for prefix, provider in PROVIDER_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    hosted = split_hosted(model)
    if not hosted:
        return None
    maker, slash, _ = hosted[1].partition("/")
    if not slash:
        return _bare_maker(maker)[0]
    maker = maker.lower()
    return MAKER_ALIASES.get(maker, maker)


def _bare_maker(name: str) -> tuple[str, str | None] | tuple[None, None]:
    for prefix, made in BARE_MODEL_MAKERS.items():
        if name.lower().startswith(prefix):
            return made
    return None, None


def split_hosted(model: str) -> tuple[str, str] | None:
    """``(host, id sent to that host)`` for a hosted id, else None. ``groq:qwen/x`` ->
    ``("groq", "qwen/x")``; ``qwen/x:free`` -> ``("openrouter", "qwen/x:free")``;
    ``groq:gpt-oss-120b`` -> ``("groq", "<namespace>/gpt-oss-120b")`` (BARE_MODEL_MAKERS);
    a bare name needs a host prefix and a known maker."""
    host, colon, rest = model.partition(":")
    if colon and host in HOSTS:
        model = rest
    else:
        host = DEFAULT_HOST
    maker, slash, name = model.partition("/")
    if slash:
        return (host, model) if maker and name else None
    if (host == DEFAULT_HOST and not colon) or not model:
        return None
    provider, namespace = _bare_maker(model)
    if provider is None:
        return None
    return host, f"{namespace}/{model}" if namespace else model


class LLMUnavailable(RuntimeError):
    """No usable model: missing SDK, missing key, no scripted reply left, or a failed call."""


class ModelDown(LLMUnavailable):
    """A model failed after its retries (e.g. a daily quota) and is out for the rest of the
    run. Agents with a rules fallback catch this and carry on without the model; other
    models keep working."""

    def __init__(self, model: str, reason: str):
        super().__init__(f"{model} is unavailable for the rest of the run: {reason}")
        self.model = model
        self.reason = reason


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str: ...


class ScriptedLLMClient:
    """Returns the next canned reply on every call. Each reply is a string or a function
    ``(system, user) -> str``. Every call is recorded in ``calls``."""

    model = "scripted"
    provider = "scripted"

    def __init__(self, replies: Sequence[str | Callable[[str, str], str]]):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        self.calls.append({"system": system, "user": user, "max_output": max_output})
        if not self._replies:
            raise LLMUnavailable("scripted client has no more replies")
        reply = self._replies.pop(0)
        return reply(system, user) if callable(reply) else reply


class AnthropicLLMClient:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None = None):
        key = api_key or os.environ.get(KEY_ENV_VAR)
        if not key:
            raise LLMUnavailable(f"{KEY_ENV_VAR} is not set; export it or use --llm scripted")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable(
                "the 'anthropic' package is not installed; run: pip install -e \".[llm]\"") from exc
        self.model = model
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_output, system=system,
                messages=[{"role": "user", "content": user}])
        except Exception as exc:  # provider errors (rate limit, auth, network) -> clean exit
            raise LLMUnavailable(f"model call failed: {type(exc).__name__}: {exc}") from exc
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise LLMUnavailable("model returned an empty reply")
        return text


# transport(url, body, headers, timeout) -> (status, response body); tests pass a fake
Transport = Callable[[str, bytes, dict, float], tuple[int, bytes]]


def _urllib_transport(url: str, body: bytes | None, headers: dict,
                      timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


_WAIT_RE = re.compile(
    r"(?:try again|retry) in\s+((?:\d+h)?(?:\d+m(?!s))?(?:\d+(?:\.\d+)?(?:ms|s))?)", re.I)
_WAIT_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(h|ms|m|s)")
_WAIT_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def rate_limit_wait(data, raw: bytes) -> float | None:
    """Seconds a 429 reply asks us to wait, or None when it does not say. Reads Gemini's
    ``RetryInfo.retryDelay`` ("43s") and messages such as "Please try again in 1m2.5s"."""
    texts = []
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        for d in error.get("details") or []:
            if isinstance(d, dict) and isinstance(d.get("retryDelay"), str):
                texts.append(f"retry in {d['retryDelay']}")
        if isinstance(error.get("message"), str):
            texts.append(error["message"])
    elif raw:
        texts.append(raw[:500].decode("utf-8", "replace"))
    for t in texts:
        m = _WAIT_RE.search(t)
        if m and m.group(1):
            return sum(float(n) * _WAIT_UNITS[u] for n, u in _WAIT_PART_RE.findall(m.group(1)))
    return None


def _post_json(transport: Transport, sleep: Callable[[float], None], url: str, body: bytes,
               headers: dict, timeout: float, model: str) -> tuple[int, bytes, object]:
    """POST with waits on short rate limits and temporary server errors. Returns (status,
    raw body, parsed JSON or {})."""
    rate_limited = server_errors = 0
    while True:
        try:
            status, raw = transport(url, body, headers, timeout)
        except (OSError, TimeoutError) as exc:  # network failure -> clean exit
            raise LLMUnavailable(f"model call failed: {type(exc).__name__}: {exc}") from exc
        try:
            data = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {}
        if status == 429 and rate_limited < RATE_LIMIT_RETRIES:
            wait = rate_limit_wait(data, raw)
            if wait is not None and wait <= RATE_LIMIT_MAX_WAIT_SECONDS:
                rate_limited += 1
                log.warning("%s is rate-limited; waiting %.0fs before retry %d of %d",
                            model, wait + 1, rate_limited, RATE_LIMIT_RETRIES)
                sleep(wait + 1)  # one second of margin over what the host asked for
                continue
        if status in SERVER_ERROR_STATUSES and server_errors < len(SERVER_ERROR_WAITS_SECONDS):
            wait = SERVER_ERROR_WAITS_SECONDS[server_errors]
            server_errors += 1
            log.warning("%s answered HTTP %d; waiting %.0fs before retry %d of %d", model,
                        status, wait, server_errors, len(SERVER_ERROR_WAITS_SECONDS))
            sleep(wait)
            continue
        return status, raw, data


class GeminiLLMClient:
    provider = "google"

    def __init__(self, model: str, api_key: str | None = None, *,
                 transport: Transport | None = None, timeout: float = GEMINI_TIMEOUT_SECONDS,
                 sleep: Callable[[float], None] = time.sleep):
        key = api_key or os.environ.get(GEMINI_KEY_ENV_VAR)
        if not key:
            raise LLMUnavailable(
                f"{GEMINI_KEY_ENV_VAR} is not set; export it or use --llm scripted")
        self.model = model
        self._key = key
        self._transport = transport or _urllib_transport
        self._timeout = timeout
        self._sleep = sleep

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        url = f"{GEMINI_ROOT}/models/{urllib.parse.quote(self.model)}:generateContent"
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_output},
        }).encode()
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._key}
        status, raw, data = _post_json(self._transport, self._sleep, url, body, headers,
                                       self._timeout, self.model)
        if status != 200:
            msg = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
            raise LLMUnavailable(f"model call failed: HTTP {status}: {msg or 'no message'}")
        candidates = data.get("candidates") if isinstance(data, dict) else None
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMUnavailable(f"model returned no candidates (block reason: {block})")
        first = candidates[0]
        parts = (first.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts
                       if isinstance(p, dict) and not p.get("thought"))
        if not text.strip():
            finish = first.get("finishReason")
            hint = ("; thinking used the whole output cap, raise max_output in "
                    "config/agents.yaml" if finish == "MAX_TOKENS" else "")
            raise LLMUnavailable(f"model returned an empty reply (finishReason {finish}){hint}")
        return text


class ChatCompletionsLLMClient:
    """A hosted chat-completions endpoint (see HOSTS). ``provider`` is the model's maker,
    ``host`` who serves it."""

    def __init__(self, model: str, api_key: str | None = None, *,
                 transport: Transport | None = None, timeout: float = HOSTED_TIMEOUT_SECONDS,
                 sleep: Callable[[float], None] = time.sleep):
        hosted = split_hosted(model)
        if hosted is None:
            raise LLMUnavailable(f"not a hosted model id: {model!r} (expected maker/model)")
        self.host, self._wire_model = hosted
        self._url, env_var = HOSTS[self.host]
        key = api_key or os.environ.get(env_var)
        if not key:
            raise LLMUnavailable(f"{env_var} is not set; export it or use --llm scripted")
        self.model = model
        self.provider = provider_of(model)
        self._key = key
        self._transport = transport or _urllib_transport
        self._timeout = timeout
        self._sleep = sleep

    def complete(self, system: str, user: str, *, max_output: int = 4096) -> str:
        body = json.dumps({
            "model": self._wire_model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_output,
        }).encode()
        # an explicit JSON Accept: NVIDIA's gateway may otherwise hold the reply open
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "Authorization": f"Bearer {self._key}",
                   "User-Agent": USER_AGENT, "X-Title": "QIP alpha research"}
        status, raw, data = _post_json(self._transport, self._sleep, self._url, body, headers,
                                       self._timeout, self.model)
        error = data.get("error") if isinstance(data, dict) else None
        if status != 200 or error:  # a host can report an upstream error with HTTP 200
            msg = error.get("message") if isinstance(error, dict) else None
            if not msg and raw and not data:  # not JSON, e.g. a proxy's "error code: 1010"
                msg = raw[:200].decode("utf-8", "replace").strip()
            raise LLMUnavailable(f"model call failed: HTTP {status}: {msg or 'no message'}")
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise LLMUnavailable("model returned no choices")
        first = choices[0] if isinstance(choices[0], dict) else {}
        text = _THINK_RE.sub("", (first.get("message") or {}).get("content") or "")
        if not text.strip():
            finish = first.get("finish_reason")
            hint = ("; reasoning used the whole output cap, raise max_output in "
                    "config/agents.yaml" if finish == "length" else "")
            raise LLMUnavailable(f"model returned an empty reply (finish_reason {finish}){hint}")
        return text


def list_models(host: str, api_key: str | None = None, *,
                transport: Transport | None = None,
                timeout: float = HOSTED_TIMEOUT_SECONDS) -> list[str]:
    """Model ids a host serves now, sorted (``cli models``), from its model list endpoint.
    Ids come back as the host names them; prefix them with ``<host>:`` in the config."""
    if host not in HOSTS:
        raise LLMUnavailable(f"unknown host {host!r}; known: {sorted(HOSTS)}")
    chat_url, env_var = HOSTS[host]
    key = api_key or os.environ.get(env_var)
    if not key:
        raise LLMUnavailable(f"{env_var} is not set; export it first")
    url = MODEL_LIST_URLS.get(host) or chat_url.replace("chat/completions", "models")
    headers = {"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT,
               "Accept": "application/json"}
    try:
        status, raw = (transport or _urllib_transport)(url, None, headers, timeout)
    except (OSError, TimeoutError) as exc:
        raise LLMUnavailable(f"model list failed: {type(exc).__name__}: {exc}") from exc
    try:
        data = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = {}
    if status != 200:
        raise LLMUnavailable(f"model list failed: HTTP {status}: {raw[:200]!r}")
    items = data.get("data") if isinstance(data, dict) else data
    ids = [i.get("id") for i in items or [] if isinstance(i, dict)]
    return sorted(i for i in ids if isinstance(i, str))


def make_real_client(model: str) -> LLMClient:
    """The real model client for a model id. Kept here so no other module has to name a
    provider."""
    provider = provider_of(model)
    if split_hosted(model):  # before the others: google/... is a hosted id, not Gemini's API
        return ChatCompletionsLLMClient(model)
    if provider == "anthropic":
        return AnthropicLLMClient(model)
    if provider == "google":
        return GeminiLLMClient(model)
    raise LLMUnavailable(f"unknown provider for model {model!r}: expected a name starting "
                         f"with one of {sorted(PROVIDER_PREFIXES)} or a hosted id "
                         "such as groq:qwen/qwen3.8-27b")
