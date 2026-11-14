"""Public endpoints: health check and token -> session-cookie exchange."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ... import __version__
from ...security import COOKIE_NAME, cookie_params, exchange_token
from ..schemas import HealthResponse, OkResponse, SessionRequest

router = APIRouter(tags=["auth"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe used by the launcher through the tunnel. Unauthenticated; no sensitive data."""
    return HealthResponse(status="ok", version=__version__)


def _is_https(request: Request) -> bool:
    if request.url.scheme == "https":
        return True
    # Tunnels / the Colab proxy terminate TLS and forward plain HTTP to 127.0.0.1.
    proto = request.headers.get("x-forwarded-proto", "")
    return proto.split(",")[0].strip().lower() == "https"


@router.post("/auth/session", response_model=OkResponse)
async def create_session(body: SessionRequest, request: Request) -> JSONResponse:
    """Exchange the launch token for an HttpOnly session cookie.

    Failed attempts are rate-limited per client IP. ``X-Forwarded-For`` is deliberately ignored
    (spoofable); behind a tunnel every client shares 127.0.0.1, which makes the limit global —
    the conservative choice.
    """
    studio = request.app.state.studio
    limiter = request.app.state.auth_limiter
    client_ip = request.client.host if request.client else None
    cookie = exchange_token(studio.auth, limiter, client_ip, body.token)
    resp = JSONResponse(OkResponse().model_dump())
    resp.set_cookie(COOKIE_NAME, cookie, **cookie_params(_is_https(request)))
    return resp
