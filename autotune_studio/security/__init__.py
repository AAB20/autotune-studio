"""Security module: session auth, path allowlist, upload validation, log redaction, HF token lookup.

Owned by: security. Stdlib-only except the optional Starlette glue in ``authentication``.
"""

from __future__ import annotations

from .authentication import (
    COOKIE_NAME,
    FailedAuthLimiter,
    RateLimited,
    SessionAuth,
    Unauthorized,
    authenticate_connection,
    authenticate_headers,
    check_ws_origin,
    cookie_params,
    exchange_token,
    is_same_origin_request,
    require_auth_dependency,
)
from .redaction import RedactingFilter, install_redaction, redact, redact_data, register_secret
from .secrets import get_hf_token
from .validation import (
    ALLOWED_EXTENSIONS,
    PathPolicy,
    UploadLimits,
    asave_upload_stream,
    sanitize_upload_filename,
    save_upload_stream,
    validate_hf_dataset_name,
    validate_model_name,
)

__all__ = [
    "ALLOWED_EXTENSIONS",
    "COOKIE_NAME",
    "FailedAuthLimiter",
    "PathPolicy",
    "RateLimited",
    "RedactingFilter",
    "SessionAuth",
    "Unauthorized",
    "UploadLimits",
    "asave_upload_stream",
    "authenticate_connection",
    "authenticate_headers",
    "check_ws_origin",
    "cookie_params",
    "exchange_token",
    "get_hf_token",
    "install_redaction",
    "is_same_origin_request",
    "redact",
    "redact_data",
    "register_secret",
    "require_auth_dependency",
    "sanitize_upload_filename",
    "save_upload_stream",
    "validate_hf_dataset_name",
    "validate_model_name",
]
