"""Session-token authentication (ADR-0005). Owned by: security.

Flow:
* On launch a :class:`SessionAuth` is created; its random token is delivered in the
  URL *fragment* (``#token=...``), never in a query string.
* The SPA posts it to ``POST /api/auth/session``; the backend calls
  :func:`exchange_token` which verifies it (rate-limited per client IP) and returns
  the cookie value: ``HMAC-SHA256(server_random_key, token)``. The cookie is set with
  :func:`cookie_params` (HttpOnly, SameSite=Strict, Secure on https, path=/).
* Scripts may send ``Authorization: Bearer <token>`` instead.
* WebSockets: same credentials plus :func:`check_ws_origin`.

All comparisons use :func:`hmac.compare_digest`. The token and cookie value are
registered with :mod:`.redaction` on construction so they are masked in logs.

Framework-light: only :func:`require_auth_dependency` / :func:`authenticate_connection`
touch Starlette objects (``HTTPConnection`` covers both ``Request`` and ``WebSocket``).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

from ..errors import RateLimited, StudioError, Unauthorized  # noqa: F401
from . import redaction

try:  # Starlette is a hard dependency of the API layer; keep this module importable without it.
    from starlette.requests import HTTPConnection
except ModuleNotFoundError:  # pragma: no cover
    HTTPConnection = Any  # type: ignore[misc,assignment]

__all__ = [
    "COOKIE_NAME",
    "FailedAuthLimiter",
    "RateLimited",
    "SessionAuth",
    "Unauthorized",
    "authenticate_connection",
    "authenticate_headers",
    "check_ws_origin",
    "cookie_params",
    "exchange_token",
    "extract_bearer",
    "is_same_origin_request",
    "require_auth_dependency",
]

COOKIE_NAME = "autotune_session"
_MAX_CREDENTIAL_LEN = 512


# Errors live in autotune_studio.errors; re-exported here for convenience.


# --------------------------------------------------------------------------- helpers
def _to_bytes(value: str) -> bytes:
    return value.encode("utf-8", "surrogatepass")


def _consteq(a: str, b: str) -> bool:
    """Constant-time string equality that never raises on non-ASCII input."""
    return hmac.compare_digest(_to_bytes(a), _to_bytes(b))


def _get_ci(mapping: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup (works for Starlette ``Headers`` and plain dicts)."""
    val = mapping.get(name)
    if val is None:
        val = mapping.get(name.lower())
    if val is None:
        lname = name.lower()
        for k, v in mapping.items():
            if isinstance(k, str) and k.lower() == lname:
                return v
    return val


# --------------------------------------------------------------------------- session auth
class SessionAuth:
    """Per-launch session secret.

    ``token`` is shown to the user once (URL fragment). The cookie never contains the
    token itself, only ``HMAC-SHA256(key, token)`` with a per-process random key, so a
    leaked cookie cannot be turned back into a bearer token.
    """

    def __init__(self, token: str | None = None) -> None:
        if token is None:
            token = secrets.token_urlsafe(32)
        if not isinstance(token, str) or len(token) < 16:
            raise ValueError("session token must be a string of at least 16 characters")
        self._token = token
        self._key = secrets.token_bytes(32)
        self._cookie = hmac.new(self._key, _to_bytes(token), hashlib.sha256).hexdigest()
        redaction.register_secret(self._token)
        redaction.register_secret(self._cookie)

    @property
    def token(self) -> str:
        return self._token

    def __repr__(self) -> str:  # never leak the token via repr()
        return "SessionAuth(token=***)"

    def verify_token(self, candidate: str | None) -> bool:
        if not isinstance(candidate, str) or not candidate or len(candidate) > _MAX_CREDENTIAL_LEN:
            return False
        return _consteq(candidate, self._token)

    def cookie_value(self) -> str:
        return self._cookie

    def verify_cookie(self, value: str | None) -> bool:
        if not isinstance(value, str) or not value or len(value) > _MAX_CREDENTIAL_LEN:
            return False
        return _consteq(value, self._cookie)


def cookie_params(secure: bool) -> dict[str, Any]:
    """Keyword arguments for ``Response.set_cookie(COOKIE_NAME, value, **cookie_params(...))``.

    No ``max_age``/``expires``: a session cookie that dies with the browser session; the
    token itself dies with the server process.
    """
    return {"httponly": True, "samesite": "strict", "secure": bool(secure), "path": "/"}


def extract_bearer(headers: Mapping[str, str]) -> str | None:
    raw = _get_ci(headers, "authorization")
    if not raw or not isinstance(raw, str):
        return None
    scheme, _, cred = raw.strip().partition(" ")
    if scheme.lower() != "bearer":
        return None
    cred = cred.strip()
    return cred or None


def authenticate_headers(auth: SessionAuth, headers: Mapping[str, str], cookies: Mapping[str, str]) -> bool:
    """True if ``Authorization: Bearer <token>`` or the session cookie is valid."""
    bearer = extract_bearer(headers)
    if bearer is not None and auth.verify_token(bearer):
        return True
    cookie = cookies.get(COOKIE_NAME) if cookies else None
    return cookie is not None and auth.verify_cookie(cookie)


# --------------------------------------------------------------------------- websocket origin
_DEFAULT_PORTS = {"http": 80, "ws": 80, "https": 443, "wss": 443}


def _split_hostport(netloc: str) -> tuple[str, int | None] | None:
    try:
        parts = urlsplit("//" + netloc.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if not host or parts.username is not None or parts.password is not None:
        return None
    return host.lower().rstrip("."), port


def _origin_tuple(origin: str) -> tuple[str, str, int | None] | None:
    try:
        parts = urlsplit(origin.strip())
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    return scheme, host.lower().rstrip("."), port if port is not None else _DEFAULT_PORTS[scheme]


def check_ws_origin(origin: str | None, host: str | None, extra_allowed: Iterable[str] = ()) -> bool:
    """Cross-site WebSocket hijacking guard.

    * Missing/empty ``Origin`` → allowed (non-browser client; it must still authenticate).
    * ``Origin: null`` or unparsable → rejected.
    * Otherwise the origin's host[:port] must equal the ``Host`` header (scheme-agnostic;
      a missing port on either side means the origin scheme's default port), or match an
      ``extra_allowed`` entry: a full origin (``https://x.trycloudflare.com``) or a bare
      ``host[:port]``. Needed behind proxies that rewrite ``Host`` (Colab proxy).
    """
    if origin is None or origin.strip() == "":
        return True
    o = _origin_tuple(origin)
    if o is None:
        return False
    scheme, ohost, oport = o
    default = _DEFAULT_PORTS[scheme]

    def host_matches(netloc: str) -> bool:
        hp = _split_hostport(netloc)
        if hp is None:
            return False
        h, p = hp
        return h == ohost and (p if p is not None else default) == oport

    if host and host_matches(host):
        return True
    for entry in extra_allowed:
        if not entry:
            continue
        if "://" in entry:
            e = _origin_tuple(entry)
            # Scheme-agnostic: compare host + effective port only.
            if e is not None and e[1] == ohost and e[2] == oport:
                return True
        elif host_matches(entry):
            return True
    return False


# --------------------------------------------------------------------------- rate limiting
class FailedAuthLimiter:
    """Sliding-window limit on *failed* attempts per key (client IP). Thread-safe, bounded memory."""

    def __init__(
        self,
        max_per_minute: int = 20,
        *,
        window_seconds: float = 60.0,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_per_minute < 1:
            raise ValueError("max_per_minute must be >= 1")
        self.max_failures = max_per_minute
        self.window = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        self._lock = threading.Lock()
        self._fails: dict[str, deque[float]] = {}

    def _prune(self, key: str, now: float) -> deque[float] | None:
        q = self._fails.get(key)
        if q is None:
            return None
        while q and now - q[0] >= self.window:
            q.popleft()
        if not q:
            del self._fails[key]
            return None
        return q

    def retry_after(self, key: str) -> float:
        """Seconds until ``key`` may try again; 0.0 when allowed."""
        now = self._clock()
        with self._lock:
            q = self._prune(key, now)
            if q is None or len(q) < self.max_failures:
                return 0.0
            return max(0.0, self.window - (now - q[0]))

    def allow(self, key: str) -> bool:
        return self.retry_after(key) == 0.0

    def record_failure(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            q = self._prune(key, now)
            if q is None:
                if len(self._fails) >= self.max_keys:
                    self._evict(now)
                q = self._fails.setdefault(key, deque())
            q.append(now)
            while len(q) > self.max_failures:
                q.popleft()

    def reset(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)

    def _evict(self, now: float) -> None:
        for k in list(self._fails):
            self._prune(k, now)
        while len(self._fails) >= self.max_keys:  # still full: drop oldest-activity key
            oldest = min(self._fails, key=lambda k: self._fails[k][-1])
            del self._fails[oldest]


def exchange_token(
    auth: SessionAuth, limiter: FailedAuthLimiter, client_ip: str | None, candidate: Any
) -> str:
    """Logic for ``POST /api/auth/session``: verify ``candidate`` and return the cookie value.

    Raises :class:`RateLimited` (429) when the IP exceeded the failure budget (checked
    *before* verifying, so a correct guess while blocked still fails) and
    :class:`Unauthorized` (401) on a wrong token. Successful exchange resets the counter.
    """
    key = client_ip or "unknown"
    wait = limiter.retry_after(key)
    if wait > 0:
        raise RateLimited(wait)
    if not isinstance(candidate, str) or not auth.verify_token(candidate):
        limiter.record_failure(key)
        raise Unauthorized("invalid session token")
    limiter.reset(key)
    return auth.cookie_value()


# --------------------------------------------------------------------------- starlette glue
def authenticate_connection(
    auth: SessionAuth,
    conn: HTTPConnection,
    *,
    check_origin: bool | None = None,
    extra_origins: Iterable[str] = (),
) -> bool:
    """Authenticate a Starlette ``Request`` or ``WebSocket``.

    ``check_origin`` defaults to True for WebSocket scopes (ADR-0005). For HTTP the full
    origin check is off by default (Bearer clients may be cross-origin, e.g. with CORS), but
    *cookie*-authenticated unsafe requests must pass :func:`is_same_origin_request` (SEC-03).
    """
    is_ws = conn.scope.get("type") == "websocket"
    if check_origin is None:
        check_origin = is_ws
    headers = conn.headers
    if check_origin and not check_ws_origin(headers.get("origin"), headers.get("host"), extra_origins):
        return False
    bearer = extract_bearer(headers)
    if bearer is not None and auth.verify_token(bearer):
        return True  # not an ambient credential: CSRF-immune
    cookie = conn.cookies.get(COOKIE_NAME) if conn.cookies else None
    if cookie is None or not auth.verify_cookie(cookie):
        return False
    # Cookie auth on unsafe HTTP methods additionally requires a same-origin sender (SEC-03).
    return is_ws or check_origin or is_same_origin_request(
        str(conn.scope.get("method", "GET")), headers, extra_origins
    )


#: Methods that never change state; the CSRF origin check is skipped for them.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def is_same_origin_request(method: str, headers: Mapping[str, str], extra_origins: Iterable[str] = ()) -> bool:
    """CSRF defence-in-depth for *cookie*-authenticated HTTP requests (SEC-03).

    ``SameSite=Strict`` already stops cross-site requests from carrying the cookie; this check
    also rejects same-*site* but cross-*origin* senders (sibling subdomains on a shared parent
    domain that is not on the Public Suffix List) and old browsers that ignore SameSite.

    * Safe methods (GET/HEAD/OPTIONS) -> allowed.
    * ``Origin`` present -> must match ``Host`` or an ``extra_origins`` entry (same rules as
      :func:`check_ws_origin`; ``Origin: null`` is rejected).
    * ``Origin`` absent -> rejected only if ``Sec-Fetch-Site`` says ``cross-site``/``same-site``
      (non-browser clients send neither header and stay allowed).
    """
    if method.upper() in SAFE_METHODS:
        return True
    origin = _get_ci(headers, "origin")
    if origin is not None and origin.strip():
        return check_ws_origin(origin, _get_ci(headers, "host"), extra_origins)
    site = (_get_ci(headers, "sec-fetch-site") or "").strip().lower()
    return site not in ("cross-site", "same-site")


def require_auth_dependency(
    auth: SessionAuth, *, extra_origins: Iterable[str] = ()
) -> Callable[[HTTPConnection], None]:
    """FastAPI dependency factory: ``Depends(require_auth_dependency(auth))``.

    Raises :class:`Unauthorized`; the API layer's ``StudioError`` handler maps it to 401.
    For WebSockets the backend should call :func:`authenticate_connection` before
    ``accept()`` and close with code 1008 on failure (an exception raised from a WS
    dependency is not turned into an HTTP 401 by Starlette).
    """
    origins = tuple(extra_origins)

    def _require_auth(conn: HTTPConnection) -> None:
        if not authenticate_connection(auth, conn, extra_origins=origins):
            raise Unauthorized()

    return _require_auth
