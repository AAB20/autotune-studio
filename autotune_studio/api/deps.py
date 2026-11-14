"""FastAPI dependencies shared by the routers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from ..core.config import StudioSettings
from ..security import require_auth_dependency

if TYPE_CHECKING:  # pragma: no cover
    from ..studio import Studio

__all__ = ["get_settings", "get_studio", "require_auth"]


def get_studio(request: Request) -> Studio:
    return request.app.state.studio


def get_settings(request: Request) -> StudioSettings:
    return request.app.state.settings


def require_auth(request: Request) -> None:
    """Authenticate every protected ``/api`` request (Bearer token or session cookie).

    ``studio.extra_origins`` is read on every call because the tunnel URL is only known after
    ``Studio.launch()`` exposed the server (the security helper freezes its origin list).
    Raises :class:`~autotune_studio.errors.Unauthorized` -> 401 via the StudioError handler.
    """
    studio = request.app.state.studio
    require_auth_dependency(studio.auth, extra_origins=studio.extra_origins)(request)
