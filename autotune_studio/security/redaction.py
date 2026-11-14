"""Secret redaction for log output. Owned by: security.

Masks Hugging Face tokens (``hf_...``), ``Bearer <x>`` credentials, ``token=<x>``
query/fragment parameters, the session cookie and every value registered via
:func:`register_secret` (session token, HF token, derived cookie value).

Two layers are installed by :func:`install_redaction`:

1. :class:`RedactingFilter` on the root logger's handlers and on the ``uvicorn``,
   ``uvicorn.access`` and ``uvicorn.error`` loggers and their handlers.
2. A wrapping ``LogRecord`` factory, so records are redacted at creation even if a
   handler is added later (e.g. uvicorn's ``dictConfig`` replaces handlers) or the
   ``lastResort`` handler is used.

``record.args`` keep their tuple/dict shape (uvicorn's ``AccessFormatter`` unpacks
``record.args`` positionally), each string argument is redacted individually.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterable
from typing import Any

__all__ = [
    "MASK",
    "MIN_SECRET_LENGTH",
    "RedactingFilter",
    "install_redaction",
    "redact",
    "redact_data",
    "register_secret",
    "registered_secret_count",
    "uninstall_redaction",
    "unregister_secret",
]

MASK = "***"
#: Registered secrets shorter than this are ignored to avoid mangling ordinary log text.
MIN_SECRET_LENGTH = 8

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Hugging Face user/org access tokens.
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "hf_" + MASK),
    # Authorization: Bearer <credential>
    (re.compile(r"(?i)\b(bearer)(\s+)[A-Za-z0-9._~+/=\-]+"), r"\1\2" + MASK),
    # token=..., access_token=..., hf_token=..., session_token=..., api_key=... in URLs, fragments, kv logs
    (
        re.compile(r"(?i)\b((?:access_|hf_|session_|auth_|id_)?token|api_key|apikey)(=|%3D)[^&\s#'\";,]+"),
        r"\1\2" + MASK,
    ),
    # Session cookie in Cookie / Set-Cookie dumps.
    (re.compile(r"(?i)\b(autotune_session=)[^;\s'\",]+"), r"\1" + MASK),
]

_lock = threading.RLock()
_secrets: set[str] = set()
_secrets_sorted: tuple[str, ...] = ()


def register_secret(secret: str | None) -> None:
    """Register a literal value that must never appear in logs. ``None``/short values are ignored."""
    global _secrets_sorted
    if not secret or not isinstance(secret, str):
        return
    secret = secret.strip()
    if len(secret) < MIN_SECRET_LENGTH:
        return
    with _lock:
        if secret in _secrets:
            return
        _secrets.add(secret)
        _secrets_sorted = tuple(sorted(_secrets, key=len, reverse=True))


def unregister_secret(secret: str) -> None:
    """Remove a previously registered secret (tests / token rotation)."""
    global _secrets_sorted
    with _lock:
        _secrets.discard(secret.strip() if isinstance(secret, str) else secret)
        _secrets_sorted = tuple(sorted(_secrets, key=len, reverse=True))


def registered_secret_count() -> int:
    with _lock:
        return len(_secrets)


def redact(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Return ``text`` with all known secret shapes and registered/extra secrets masked."""
    if not text:
        return text
    if not isinstance(text, str):
        text = str(text)
    literals = list(_secrets_sorted)
    extras = [s for s in extra_secrets if isinstance(s, str) and s]
    if extras:
        literals = sorted(set(literals) | set(extras), key=len, reverse=True)
    for lit in literals:
        if lit in text:
            text = text.replace(lit, MASK)
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact_data(value: Any, _depth: int = 0) -> Any:
    """Recursively redact every string inside a JSON-like value (dict/list/tuple/str).

    Intended for event payloads (``EventBus.publish`` data -> WebSocket / replay buffer) and
    other API-visible text that does not go through logging (SEC-04). Keys are left untouched;
    non-string scalars are returned as-is; nesting deeper than 32 levels is returned unchanged.
    """
    if isinstance(value, str):
        return redact(value)
    if _depth > 32:
        return value
    if isinstance(value, dict):
        return {k: redact_data(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_data(v, _depth + 1) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_data(v, _depth + 1) for v in value)
    return value


_PLAIN_TYPES = (int, float, bool, type(None), bytes)


def _redact_arg(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, _PLAIN_TYPES):
        return value
    try:
        s = str(value)
    except Exception:  # noqa: BLE001  # pragma: no cover - broken __str__
        return value
    red = redact(s)
    return value if red == s else red


def _redact_record(record: logging.LogRecord) -> None:
    if getattr(record, "_autotune_redacted", False):
        return
    if isinstance(record.msg, str):
        record.msg = redact(record.msg)
    elif record.msg is not None and not isinstance(record.msg, _PLAIN_TYPES):
        red = _redact_arg(record.msg)
        if red is not record.msg:
            record.msg = red
    args = record.args
    if isinstance(args, tuple):
        record.args = tuple(_redact_arg(a) for a in args)
    elif isinstance(args, dict):
        record.args = {k: _redact_arg(v) for k, v in args.items()}
    if record.exc_info and not record.exc_text:
        try:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        except Exception:  # noqa: BLE001  # pragma: no cover
            record.exc_text = None
    if record.exc_text:
        record.exc_text = redact(record.exc_text)
    if record.stack_info:
        record.stack_info = redact(record.stack_info)
    record._autotune_redacted = True  # type: ignore[attr-defined]


class RedactingFilter(logging.Filter):
    """Logging filter that masks secrets in ``msg``, ``args``, tracebacks and stack info. Never drops records."""

    def __init__(self, extra_secrets: Iterable[str] = ()) -> None:
        super().__init__()
        for s in extra_secrets:
            register_secret(s)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            _redact_record(record)
        except Exception:  # noqa: BLE001, S110  # pragma: no cover - never break logging
            pass
        return True


_UVICORN_LOGGERS = ("uvicorn", "uvicorn.access", "uvicorn.error")
_installed_filter: RedactingFilter | None = None
_original_factory: Any = None


def _attach(target: logging.Filterer, flt: RedactingFilter) -> None:
    if not any(isinstance(f, RedactingFilter) for f in getattr(target, "filters", [])):
        target.addFilter(flt)


def install_redaction(secrets: Iterable[str] = (), *, record_factory: bool = True) -> RedactingFilter:
    """Register ``secrets`` and attach redaction to root handlers and uvicorn loggers. Idempotent.

    Call again after uvicorn has configured logging if its handlers were replaced
    (the record factory already covers that case when ``record_factory=True``).
    """
    global _installed_filter, _original_factory
    for s in secrets:
        register_secret(s)
    with _lock:
        flt = _installed_filter or RedactingFilter()
        _installed_filter = flt
        root = logging.getLogger()
        for h in root.handlers:
            _attach(h, flt)
        for name in _UVICORN_LOGGERS:
            lg = logging.getLogger(name)
            _attach(lg, flt)
            for h in lg.handlers:
                _attach(h, flt)
        if record_factory and _original_factory is None:
            base = logging.getLogRecordFactory()
            _original_factory = base

            def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
                rec = base(*args, **kwargs)
                try:
                    _redact_record(rec)
                except Exception:  # noqa: BLE001, S110  # pragma: no cover - never break logging
                    pass
                return rec

            factory._autotune_redaction = True  # type: ignore[attr-defined]
            logging.setLogRecordFactory(factory)
    return flt


def uninstall_redaction() -> None:
    """Detach the filter and restore the original record factory (registered secrets are kept)."""
    global _installed_filter, _original_factory
    with _lock:
        flt = _installed_filter
        if flt is not None:
            targets: list[logging.Filterer] = list(logging.getLogger().handlers)
            for name in _UVICORN_LOGGERS:
                lg = logging.getLogger(name)
                targets.append(lg)
                targets.extend(lg.handlers)
            for t in targets:
                t.removeFilter(flt)
        _installed_filter = None
        if _original_factory is not None:
            if getattr(logging.getLogRecordFactory(), "_autotune_redaction", False):
                logging.setLogRecordFactory(_original_factory)
            _original_factory = None
