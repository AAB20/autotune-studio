"""Hugging Face token discovery. Owned by: security.

Sources, in order: env ``HF_TOKEN`` → env ``HUGGING_FACE_HUB_TOKEN`` → Colab
``google.colab.userdata`` → Kaggle ``kaggle_secrets``. Platform modules are imported
lazily and every failure is swallowed (missing module, secret not attached, access
denied). The value is registered with :mod:`.redaction` and never logged or returned
by the API — callers should only pass it to ``huggingface_hub`` / ``from_pretrained``.

Note: this module is named ``secrets`` inside the ``security`` package; other modules
here use absolute ``import secrets`` which resolves to the standard library.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from . import redaction

__all__ = ["ENV_VARS", "SECRET_NAME", "get_hf_token", "hf_token_source"]

log = logging.getLogger(__name__)

ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
SECRET_NAME = "HF_TOKEN"


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _from_colab() -> str | None:
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        return _clean(userdata.get(SECRET_NAME))
    except Exception:  # noqa: BLE001 - ModuleNotFoundError, SecretNotFoundError, NotebookAccessError, ...
        return None


def _from_kaggle() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        return _clean(UserSecretsClient().get_secret(SECRET_NAME))
    except Exception:  # noqa: BLE001 - not on Kaggle / secret not attached
        return None


def _discover() -> tuple[str | None, str | None]:
    for var in ENV_VARS:
        tok = _clean(os.environ.get(var))
        if tok:
            return tok, f"env:{var}"
    probes: tuple[tuple[str, Callable[[], str | None]], ...] = (
        ("colab", _from_colab),
        ("kaggle", _from_kaggle),
    )
    for label, probe in probes:
        tok = probe()
        if tok:
            return tok, label
    return None, None


def get_hf_token() -> str | None:
    """Return the HF token or ``None``. The token is registered for log redaction."""
    tok, source = _discover()
    if tok:
        redaction.register_secret(tok)
        log.debug("Hugging Face token found (source=%s)", source)  # source only, never the value
    return tok


def hf_token_source() -> str | None:
    """Where the token came from (``env:HF_TOKEN``, ``colab``, ``kaggle``) — safe to show in the UI."""
    tok, source = _discover()
    if tok:
        redaction.register_secret(tok)
    return source
