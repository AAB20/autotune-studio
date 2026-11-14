"""Fixtures for security tests."""

from __future__ import annotations

import pytest

from autotune_studio.security import redaction


@pytest.fixture
def clean_redaction():
    """Snapshot/restore global redaction state so tests don't leak secrets or filters."""
    before = set(redaction._secrets)
    yield
    redaction.uninstall_redaction()
    for s in set(redaction._secrets) - before:
        redaction.unregister_secret(s)
