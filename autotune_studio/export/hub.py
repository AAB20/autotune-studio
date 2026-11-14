"""Hugging Face Hub login verification and folder upload (ADR-0007).

All network calls go through ``huggingface_hub.HfApi`` (lazily imported). Tokens are never logged or
returned; errors are re-raised as :class:`~autotune_studio.errors.HubError` with redacted messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import HFCredentialsStatus
from ..errors import DependencyMissing, HubError, ValidationFailed
from ..security.redaction import redact, register_secret

USERNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,94}[A-Za-z0-9])?$")
REPO_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
UPLOAD_IGNORE = ["autotune_state.json", "*.partial", "*.partial/**"]


@dataclass
class HFCredentials:
    """In-memory credentials. ``token`` must never be serialised."""

    username: str
    token: str
    role: str | None
    can_write: bool | None

    def __repr__(self) -> str:  # never show the token in tracebacks/debug output
        return f"HFCredentials(username={self.username!r}, role={self.role!r})"

    def status(self) -> HFCredentialsStatus:
        msg = "verified"
        if self.can_write is False:
            msg = "verified, but this token is read-only: pushing models needs a token with write access"
        return HFCredentialsStatus(
            configured=True, username=self.username, verified=True,
            can_write=self.can_write, token_role=self.role, message=msg,
        )


def _api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - installed with the [hf] extra
        raise DependencyMissing("huggingface_hub is required: pip install 'autotune-studio[hf]'") from exc
    return HfApi()


def _hub_error(action: str, exc: Exception) -> HubError:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    msg = redact(str(exc).splitlines()[0] if str(exc) else type(exc).__name__)[:300]
    if status in (401, 403):
        msg = f"the Hub rejected the token ({status}): {msg}"
    return HubError(f"{action} failed: {msg}", {"status": status} if status else None)


def verify_credentials(username: str, token: str) -> HFCredentials:
    """Check the token with ``whoami`` and that ``username`` is its user or one of its organisations."""
    username = (username or "").strip()
    token = (token or "").strip()
    if not USERNAME_RE.match(username):
        raise ValidationFailed("invalid Hugging Face username", {"field": "username"})
    if not token or any(c.isspace() for c in token):
        raise ValidationFailed("invalid Hugging Face token", {"field": "token"})
    register_secret(token)
    try:
        info = _api().whoami(token=token)
    except Exception as exc:  # noqa: BLE001 - network/auth errors become HubError
        raise _hub_error("verifying the Hugging Face token", exc) from None
    name = str(info.get("name", ""))
    orgs = [str(o.get("name", "")) for o in info.get("orgs", []) if isinstance(o, dict)]
    if username.lower() != name.lower() and username.lower() not in {o.lower() for o in orgs}:
        raise ValidationFailed(
            f"this token belongs to '{name}', not '{username}' (or one of its organisations)",
            {"field": "username"},
        )
    access = (info.get("auth") or {}).get("accessToken") or {}
    role = access.get("role")
    can_write: bool | None
    if role == "write":
        can_write = True
    elif role == "read":
        can_write = False
    else:  # fine-grained tokens: permissions are per-repo, cannot be decided here
        can_write = None
    return HFCredentials(username=username if username.lower() != name.lower() else name,
                         token=token, role=role, can_write=can_write)


def default_repo_name(checkpoint_id: str, fmt: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", checkpoint_id).strip("-.")[:80] or "model"
    return f"autotune-{base}-{'lora' if fmt == 'lora_adapter' else 'merged'}"[:96]


def push_folder(folder: str | Path, creds: HFCredentials, repo_name: str, private: bool) -> tuple[str, str]:
    """Create ``<username>/<repo_name>`` if needed and upload ``folder``. Returns ``(repo_id, url)``."""
    if not REPO_NAME_RE.match(repo_name) or ".." in repo_name or repo_name.endswith((".", "-")):
        raise ValidationFailed("invalid Hub repository name (letters, digits, '.', '_', '-')",
                               {"field": "hub_repo_name"})
    if creds.can_write is False:
        raise HubError("the saved Hugging Face token is read-only; create a token with write access")
    repo_id = f"{creds.username}/{repo_name}"
    api = _api()
    try:
        api.create_repo(repo_id, token=creds.token, private=private, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=repo_id, folder_path=str(folder), token=creds.token, repo_type="model",
            commit_message="Upload fine-tuned model from AutoTune Studio", ignore_patterns=UPLOAD_IGNORE,
        )
    except Exception as exc:  # noqa: BLE001
        raise _hub_error(f"pushing to {repo_id}", exc) from None
    return repo_id, f"https://huggingface.co/{repo_id}"
