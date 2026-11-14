"""Typed exceptions mapped to :class:`~autotune_studio.contracts.ApiError` codes. Owned by: Chief Architect."""

from __future__ import annotations

from typing import Any

from .contracts import ErrorCode


class StudioError(Exception):
    code: str = ErrorCode.INTERNAL
    http_status: int = 500

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ValidationFailed(StudioError):
    code, http_status = ErrorCode.VALIDATION, 422


class NotFound(StudioError):
    code, http_status = ErrorCode.NOT_FOUND, 404


class Conflict(StudioError):
    code, http_status = ErrorCode.CONFLICT, 409


class ForbiddenPath(StudioError):
    code, http_status = ErrorCode.FORBIDDEN_PATH, 403


class DependencyMissing(StudioError):
    code, http_status = ErrorCode.DEPENDENCY_MISSING, 424


class DatasetEmpty(StudioError):
    code, http_status = ErrorCode.DATASET_EMPTY, 422


class DatasetMalformed(StudioError):
    code, http_status = ErrorCode.DATASET_MALFORMED, 422


class StepsRequired(StudioError):
    code, http_status = ErrorCode.STEPS_REQUIRED, 422


class OutOfMemory(StudioError):
    code, http_status = ErrorCode.OOM, 507


class Unauthorized(StudioError):
    code, http_status = ErrorCode.UNAUTHORIZED, 401

    def __init__(self, message: str = "authentication required", detail: dict[str, Any] | None = None) -> None:
        super().__init__(message, detail)


class RateLimited(StudioError):
    code, http_status = ErrorCode.RATE_LIMITED, 429

    def __init__(self, retry_after: float) -> None:
        super().__init__(
            "too many failed authentication attempts", {"retry_after_seconds": round(retry_after, 1)}
        )
        self.retry_after = retry_after


class HubError(StudioError):
    """The Hugging Face Hub rejected or failed a request (bad token, no write access, network)."""

    code, http_status = ErrorCode.HUB_ERROR, 502
