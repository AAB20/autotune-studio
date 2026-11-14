"""Compatibility-test isolation: scrub runtime env vars / notebook modules, restore redaction state."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

RUNTIME_ENV = [
    "COLAB_RELEASE_TAG",
    "KAGGLE_KERNEL_RUN_TYPE",
    "AUTOTUNE_ALLOWED_ROOTS",
    "AUTOTUNE_WORKSPACE",
    "AUTOTUNE_CLOUDFLARED",
]
NOTEBOOK_MODULES = ["google.colab", "google.colab.output", "IPython", "IPython.display", "ipykernel"]


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in RUNTIME_ENV:
        monkeypatch.delenv(var, raising=False)
    for mod in NOTEBOOK_MODULES:
        if mod in sys.modules:
            monkeypatch.delitem(sys.modules, mod)
    from autotune_studio.core import environment

    monkeypatch.setattr(environment, "CHECK_KAGGLE_DIR", False)


@pytest.fixture(autouse=True)
def restore_redaction() -> Iterator[None]:
    from autotune_studio.security import redaction

    before = set(redaction._secrets)
    yield
    redaction.uninstall_redaction()  # Studio.launch() installs a global logging filter
    for s in set(redaction._secrets) - before:
        redaction.unregister_secret(s)
