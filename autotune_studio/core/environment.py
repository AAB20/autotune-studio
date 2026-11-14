"""Runtime environment detection (Colab / Kaggle / Jupyter / script).

Owned by: runtime-compatibility. Skill: ``notebook-runtime-detection``.

Everything here is *pure and robust*: :func:`detect_environment` never raises,
never creates directories and never touches the network unless asked to
(``probe_network``). Anything noteworthy is recorded in ``EnvironmentInfo.notes``.

Detection order (first match wins):

1. **Colab**  - ``google.colab`` importable / already imported, or ``COLAB_RELEASE_TAG`` set.
2. **Kaggle** - ``KAGGLE_KERNEL_RUN_TYPE`` set, or ``/kaggle/working`` exists.
3. **Jupyter** - an IPython kernel is active (``ZMQInteractiveShell``) or ``ipykernel`` imported.
4. **script** - anything else.

Environment variables honoured:

* ``AUTOTUNE_ALLOWED_ROOTS`` - extra dataset roots, ``os.pathsep`` separated.
* ``AUTOTUNE_WORKSPACE``     - override the default writable workspace directory.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform as _platform
import sys
from pathlib import Path

from ..contracts import EnvironmentInfo, RuntimeKind

__all__ = [
    "INTERNET_PROBE_URL",
    "KAGGLE_INTERNET_HINT",
    "detect_environment",
    "detect_runtime_kind",
    "ensure_workspace",
    "probe_internet",
]

log = logging.getLogger(__name__)

INTERNET_PROBE_URL = "https://huggingface.co"

KAGGLE_INTERNET_HINT = (
    "Kaggle: outbound internet appears to be disabled. Open the notebook's right-hand 'Settings' panel "
    "(or 'Session options'), switch 'Internet' ON (requires a phone-verified Kaggle account), then re-run. "
    "Without internet the Hugging Face Hub and the cloudflared tunnel cannot be used."
)

# Module-level so tests can point them at tmp dirs.
COLAB_ROOT = "/content"
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"
KAGGLE_TEMP = "/kaggle/temp"
# The /kaggle/working existence check only makes sense on POSIX (on Windows "/kaggle" is drive-relative).
CHECK_KAGGLE_DIR = os.name == "posix"


# --------------------------------------------------------------------------- detection helpers
def _module_available(name: str) -> bool:
    """True if ``name`` is already imported or importable. Never raises."""
    try:
        if name in sys.modules:
            return sys.modules[name] is not None
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        # find_spec raises if a parent package is missing or a module has no __spec__.
        return False


def _is_colab() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG")) or _module_available("google.colab")


def _is_kaggle() -> bool:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    try:
        return CHECK_KAGGLE_DIR and Path(KAGGLE_WORKING).is_dir()
    except OSError:
        return False


def _ipython_shell_name() -> str | None:
    """Class name of the active IPython shell, or None. Does not import IPython."""
    ipy = sys.modules.get("IPython")
    if ipy is None:
        return None
    try:
        get_ipython = getattr(ipy, "get_ipython", None)
        shell = get_ipython() if callable(get_ipython) else None
    except Exception:  # pragma: no cover - defensive
        return None
    return type(shell).__name__ if shell is not None else None


def _is_jupyter_kernel() -> bool:
    name = _ipython_shell_name()
    if name == "ZMQInteractiveShell":
        return True
    return sys.modules.get("ipykernel") is not None


def detect_runtime_kind() -> RuntimeKind:
    """Return the runtime kind using the documented precedence. Never raises."""
    try:
        if _is_colab():
            return RuntimeKind.COLAB
        if _is_kaggle():
            return RuntimeKind.KAGGLE
        if _is_jupyter_kernel():
            return RuntimeKind.JUPYTER
    except Exception:  # pragma: no cover - defensive
        log.debug("runtime detection failed", exc_info=True)
    return RuntimeKind.SCRIPT


# --------------------------------------------------------------------------- network
def probe_internet(timeout: float = 3.0, url: str = INTERNET_PROBE_URL) -> bool:
    """Return True if an HTTPS request to ``url`` gets *any* HTTP response within ``timeout``.

    Any exception (DNS, TLS, timeout, proxy refusal, missing httpx) yields False.
    """
    try:
        import httpx
    except Exception:
        httpx = None  # type: ignore[assignment]
    try:
        if httpx is not None:
            resp = httpx.head(url, timeout=timeout, follow_redirects=False)
            return resp.status_code > 0
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except urllib.error.HTTPError:
            return True  # server answered
    except Exception:
        return False


# --------------------------------------------------------------------------- paths
def _norm_local(p: str | os.PathLike[str]) -> str:
    try:
        return str(Path(p).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return str(Path(p).expanduser().absolute())


def _extra_roots(notes: list[str]) -> list[str]:
    raw = os.environ.get("AUTOTUNE_ALLOWED_ROOTS", "")
    roots: list[str] = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        if "\x00" in part:
            notes.append("AUTOTUNE_ALLOWED_ROOTS: ignored an entry containing a NUL byte.")
            continue
        resolved = _norm_local(part)
        try:
            exists = Path(resolved).is_dir()
        except OSError:
            exists = False
        if not exists:
            notes.append(f"AUTOTUNE_ALLOWED_ROOTS: '{resolved}' does not exist (kept as allowed root).")
        roots.append(resolved)
    return roots


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = os.path.normcase(it)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _layout(kind: RuntimeKind, cwd: str, notes: list[str]) -> tuple[str, list[str]]:
    """Return (workspace_dir, allowed_roots) for ``kind`` (before extra roots)."""
    override = os.environ.get("AUTOTUNE_WORKSPACE", "").strip()
    if kind == RuntimeKind.COLAB:
        workspace = override or f"{COLAB_ROOT}/autotune"
        roots = [COLAB_ROOT]
        try:
            if not Path(f"{COLAB_ROOT}/drive").is_dir():
                notes.append(
                    "Colab: Google Drive is not mounted; outputs in /content are lost when the runtime "
                    "is recycled. Mount Drive (from google.colab import drive; drive.mount('/content/drive')) "
                    "to persist checkpoints."
                )
        except OSError:
            pass
    elif kind == RuntimeKind.KAGGLE:
        workspace = override or f"{KAGGLE_WORKING}/autotune"
        roots = [KAGGLE_INPUT, KAGGLE_WORKING, KAGGLE_TEMP]
        notes.append(
            "Kaggle: /kaggle/input is read-only; outputs go to /kaggle/working (persisted with the "
            "notebook version, ~20 GB limit)."
        )
    else:
        workspace = override or os.path.join(cwd, ".autotune")
        roots = [cwd]
    if override:
        workspace = _norm_local(override) if kind in (RuntimeKind.JUPYTER, RuntimeKind.SCRIPT) else override
        notes.append(f"AUTOTUNE_WORKSPACE override in effect: {workspace}")
    roots.append(workspace)
    return workspace, roots


def ensure_workspace(env: EnvironmentInfo) -> Path:
    """Create ``env.workspace_dir`` if needed and return it. Raises OSError if not creatable."""
    p = Path(env.workspace_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- public API
def detect_environment(*, probe_network: bool | None = None, cwd: str | None = None) -> EnvironmentInfo:
    """Detect the runtime and return an :class:`EnvironmentInfo`. Never raises.

    ``probe_network``: ``True`` always probes outbound HTTPS, ``False`` never does,
    ``None`` (default) probes only on Kaggle where internet is commonly disabled.
    ``internet_access`` stays ``None`` when not probed. The workspace directory is
    *not* created (see :func:`ensure_workspace`).
    """
    notes: list[str] = []
    try:
        kind = detect_runtime_kind()
        try:
            cwd_s = _norm_local(cwd if cwd is not None else os.getcwd())
        except OSError:
            cwd_s = _norm_local(".")
            notes.append("Current working directory is not accessible; using '.'.")
        workspace, roots = _layout(kind, cwd_s, notes)
        roots = _dedupe(roots + _extra_roots(notes))

        is_notebook = kind in (RuntimeKind.COLAB, RuntimeKind.KAGGLE, RuntimeKind.JUPYTER)
        if kind == RuntimeKind.KAGGLE and os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch":
            notes.append("Kaggle batch run ('Save & Run All'): no interactive browser session is available.")

        internet: bool | None = None
        do_probe = probe_network if probe_network is not None else kind == RuntimeKind.KAGGLE
        if do_probe:
            internet = bool(probe_internet())
            if not internet:
                notes.append(
                    KAGGLE_INTERNET_HINT
                    if kind == RuntimeKind.KAGGLE
                    else "No outbound internet: Hugging Face Hub datasets/models and public tunnels are unavailable."
                )

        return EnvironmentInfo(
            kind=kind,
            python_version=_platform.python_version(),
            platform=f"{_platform.system()}-{_platform.machine()}".strip("-") or sys.platform,
            is_notebook=is_notebook,
            workspace_dir=workspace,
            allowed_roots=roots,
            internet_access=internet,
            notes=notes,
        )
    except Exception as exc:  # pragma: no cover - last-resort safety net
        log.debug("detect_environment failed", exc_info=True)
        here = _norm_local(".")
        return EnvironmentInfo(
            kind=RuntimeKind.SCRIPT,
            python_version=_platform.python_version(),
            platform=sys.platform,
            is_notebook=False,
            workspace_dir=os.path.join(here, ".autotune"),
            allowed_roots=[here],
            internet_access=None,
            notes=notes + [f"Environment detection failed ({type(exc).__name__}); assuming a local script."],
        )
