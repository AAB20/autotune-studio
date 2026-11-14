"""Error mapping for the API: every failure leaves the server as a JSON :class:`ApiError`.

* :class:`~autotune_studio.errors.StudioError` -> ``exc.http_status`` with ``code/message/detail``.
* Request validation (FastAPI) and pydantic ``ValidationError`` raised inside Studio calls ->
  422 ``validation_error`` with ``detail.errors = [{loc, msg, type}]`` (the UI maps ``loc`` to
  form fields). Raw ``input``/``ctx`` values are dropped: they may echo secrets or be unserialisable.
* Starlette ``HTTPException`` (unknown route, wrong method) -> same envelope.
* Anything else -> 500 ``internal_error``; the traceback is logged (through the redaction
  filter installed by ``Studio.launch``) and never sent to the client. This is done by
  :class:`InternalErrorMiddleware` *inside* the app's middleware stack so the 500 response
  still carries the security / ``no-store`` headers.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..contracts import ApiError, ErrorCode
from ..errors import RateLimited, StudioError, Unauthorized

log = logging.getLogger("autotune_studio.api")

__all__ = ["InternalErrorMiddleware", "error_response", "install_error_handlers", "validation_payload"]


def error_response(
    status: int, code: str, message: str, detail: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> JSONResponse:
    body = ApiError(code=code, message=message, detail=detail).model_dump(mode="json")
    return JSONResponse(body, status_code=status, headers=headers)


def _clean_errors(errors: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in errors:
        if not isinstance(e, dict):
            continue
        loc = [p if isinstance(p, (str, int)) else str(p) for p in e.get("loc", ())]
        out.append({"loc": loc, "msg": str(e.get("msg", "invalid value")), "type": str(e.get("type", "value_error"))})
    return out


def validation_payload(errors: Sequence[Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(message, detail)`` for a list of pydantic error dicts."""
    cleaned = _clean_errors(errors)
    if cleaned:
        first = cleaned[0]
        field = ".".join(str(p) for p in first["loc"] if p not in ("body", "query", "path"))
        message = f"{field}: {first['msg']}" if field else first["msg"]
        if len(cleaned) > 1:
            message += f" (+{len(cleaned) - 1} more)"
    else:
        message = "validation failed"
    return message, {"errors": cleaned}


async def _studio_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StudioError)
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimited):
        headers["Retry-After"] = str(max(1, int(exc.retry_after + 0.999)))
    elif isinstance(exc, Unauthorized):
        headers["WWW-Authenticate"] = "Bearer"
    return error_response(exc.http_status, exc.code, exc.message, exc.detail, headers or None)


async def _request_validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    message, detail = validation_payload(exc.errors())
    return error_response(422, ErrorCode.VALIDATION, message, detail)


async def _pydantic_validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ValidationError)
    message, detail = validation_payload(exc.errors())
    return error_response(422, ErrorCode.VALIDATION, message, detail)


_HTTP_CODES = {404: ErrorCode.NOT_FOUND, 401: ErrorCode.UNAUTHORIZED, 405: "method_not_allowed", 413: "payload_too_large"}


async def _http_exception(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _HTTP_CODES.get(exc.status_code, f"http_{exc.status_code}")
    message = exc.detail if isinstance(exc.detail, str) else "request failed"
    return error_response(exc.status_code, code, message, headers=getattr(exc, "headers", None))


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StudioError, _studio_error)
    app.add_exception_handler(RequestValidationError, _request_validation_error)
    app.add_exception_handler(ValidationError, _pydantic_validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_exception)


class InternalErrorMiddleware:
    """Turn unhandled exceptions into a 500 ``internal_error`` JSON body (no traceback text)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        started = False

        async def _send(message: Message) -> None:
            nonlocal started
            if message["type"] in ("http.response.start", "websocket.accept", "websocket.close"):
                started = True
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except Exception:
            error_id = uuid.uuid4().hex[:12]
            log.exception("unhandled error %s on %s %s", error_id, scope.get("method", "WS"), scope.get("path"))
            if scope["type"] != "http" or started:
                return  # response already under way / websocket: nothing safe to send
            resp = error_response(
                500, ErrorCode.INTERNAL, "internal server error", {"error_id": error_id}
            )
            await resp(scope, receive, send)
