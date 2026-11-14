"""Fixtures for runtime-compatibility tests: isolate env vars and notebook modules."""

from __future__ import annotations

import sys

import pytest

_ENV = [
    "COLAB_RELEASE_TAG",
    "KAGGLE_KERNEL_RUN_TYPE",
    "AUTOTUNE_ALLOWED_ROOTS",
    "AUTOTUNE_WORKSPACE",
    "AUTOTUNE_CLOUDFLARED",
    "AUTOTUNE_CACHE_DIR",
    "CUDA_VISIBLE_DEVICES",
]
_MODULES = ["google.colab", "google.colab.output", "IPython", "IPython.display", "ipykernel", "pynvml"]


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    for mod in _MODULES:
        if mod in sys.modules:
            monkeypatch.delitem(sys.modules, mod)
    from autotune_studio.core import environment

    monkeypatch.setattr(environment, "CHECK_KAGGLE_DIR", False)
