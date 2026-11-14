"""Server settings loaded from ``AUTOTUNE_*`` environment variables. Owned by: backend-engineering.

``pydantic-settings`` is not a dependency, so :func:`load_settings` maps environment
variables onto the :class:`StudioSettings` model by hand:

==============================  =========================  ==========================
Variable                        Field                      Default
==============================  =========================  ==========================
``AUTOTUNE_HOST``               ``host``                   ``127.0.0.1``
``AUTOTUNE_PORT``               ``port``                   ``7860``
``AUTOTUNE_UPLOAD_MAX_BYTES``   ``upload_max_bytes``       ``2147483648`` (2 GiB)
``AUTOTUNE_LOG_LEVEL``          ``log_level``              ``info``
``AUTOTUNE_CORS_ENABLED``       ``cors_enabled``           ``false``
``AUTOTUNE_CORS_ORIGINS``       ``cors_origins``           empty (comma-separated)
``AUTOTUNE_WS_PING_INTERVAL``   ``ws_ping_interval``       ``20`` seconds
``AUTOTUNE_REPLAY_SIZE``        ``replay_size``            ``2000`` events
==============================  =========================  ==========================

Booleans accept ``1/0``, ``true/false``, ``yes/no``, ``on/off`` (case-insensitive).
Invalid values raise :class:`ValueError` naming the variable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

__all__ = ["ENV_PREFIX", "StudioSettings", "load_settings"]

ENV_PREFIX = "AUTOTUNE_"

LogLevel = Literal["debug", "info", "warning", "error", "critical"]


class StudioSettings(BaseModel):
    """Runtime settings for the HTTP/WebSocket server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field("127.0.0.1", min_length=1, description="Bind address; tunnels connect to localhost.")
    port: int = Field(7860, ge=0, le=65535, description="0 = OS-assigned port.")
    upload_max_bytes: int = Field(2 * 1024**3, ge=1, description="Maximum dataset upload size.")
    log_level: LogLevel = "info"
    cors_enabled: bool = Field(
        False, description="CORS is off by default: the UI is served same-origin by this server."
    )
    cors_origins: list[str] = Field(default_factory=list, description="Allowed origins when CORS is enabled.")
    ws_ping_interval: float = Field(20.0, gt=0, le=3600, description="Seconds between WebSocket keepalives.")
    replay_size: int = Field(2000, ge=1, le=1_000_000, description="EventBus replay buffer size.")

    @field_validator("log_level", mode="before")
    @classmethod
    def _lower(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [o.strip().rstrip("/") for o in v.split(",") if o.strip()]
        return v


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _parse_bool(name: str, raw: str) -> bool:
    low = raw.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f"{name}: expected a boolean (1/0, true/false, yes/no, on/off), got {raw!r}")


def load_settings(env: Mapping[str, str] | None = None, **overrides: Any) -> StudioSettings:
    """Build :class:`StudioSettings` from ``env`` (default ``os.environ``) plus keyword ``overrides``.

    :raises ValueError: an ``AUTOTUNE_*`` variable has an invalid value (message names the variable).
    """
    source = os.environ if env is None else env
    data: dict[str, Any] = {}
    for field_name in StudioSettings.model_fields:
        var = ENV_PREFIX + field_name.upper()
        raw = source.get(var)
        if raw is None:
            continue
        if field_name == "cors_enabled":
            data[field_name] = _parse_bool(var, raw)
        else:
            data[field_name] = raw.strip()
    data.update(overrides)
    try:
        return StudioSettings.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{ENV_PREFIX}{'.'.join(str(p) for p in e['loc']).upper()}: {e['msg']}" for e in exc.errors()
        )
        raise ValueError(f"invalid AutoTune settings: {problems}") from None
