"""Choose and run the UI exposure adapter (ADR-0002): Colab proxy -> cloudflared -> local.

Owned by: runtime-compatibility.

Rules enforced here:

* ``publicly_verified`` is True **only** after ``GET {public_url}/api/health`` returned 200
  through the public URL. The Colab proxy cannot be probed from the kernel (it needs the
  user's browser cookies) so it is always reported unverified with an explanatory note.
* The session token goes only into the URL *fragment* (``#token=``), never the query string,
  and never into diagnostics or logs.
* If tunnelling fails the backend keeps running; diagnostics point at ``local_url``.

Default adapter per runtime (``prefer=None``):

* Colab  : colab_proxy, falling back to cloudflared, then local.
* Kaggle : cloudflared (needs notebook Internet ON), then local.
* Jupyter/script: local only. A public tunnel is opt-in (``prefer="cloudflared"``) so that a
  personal machine is never exposed to the internet implicitly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import quote, urlsplit, urlunsplit

from ..contracts import EnvironmentInfo, LaunchInfo, LaunchMethod, RuntimeKind
from ..core.environment import KAGGLE_INTERNET_HINT
from . import cloudflared as _cf
from .colab import colab_proxy_url

__all__ = ["HEALTH_PATH", "ExposureHandle", "build_open_url", "expose", "http_health_probe"]

log = logging.getLogger(__name__)

HEALTH_PATH = "/api/health"
Prefer = Literal["colab_proxy", "cloudflared", "local"]
_VALID_PREFER = {"colab_proxy", "cloudflared", "local"}


@dataclass
class ExposureHandle:
    """Owns whatever was started to expose the UI. ``stop()`` is idempotent; it never stops the backend."""

    method: LaunchMethod
    tunnel: _cf.CloudflaredTunnel | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def stop(self) -> None:
        if self.tunnel is not None:
            self.tunnel.stop()
            self.tunnel = None


def build_open_url(base_url: str, token: str) -> str:
    """``<base>/#token=<token>``; the token lives in the fragment only (not sent to servers/proxies)."""
    parts = urlsplit(base_url)
    path = parts.path or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, "token=" + quote(token, safe="")))


def _health_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + HEALTH_PATH, parts.query, ""))


def http_health_probe(url: str, timeout: float = 5.0) -> bool:
    """True if ``GET url`` returns HTTP 200. Never raises."""
    try:
        import httpx

        return httpx.get(url, timeout=timeout, follow_redirects=False).status_code == 200
    except Exception:
        return False


def _probe_with_retries(probe: Callable[[str], bool], url: str, deadline_s: float, interval: float) -> bool:
    """Quick-tunnel DNS/edge registration takes a few seconds; retry until ``deadline_s``."""
    end = time.monotonic() + deadline_s
    while True:
        try:
            if probe(url):
                return True
        except Exception:
            log.debug("health probe raised", exc_info=True)
        if time.monotonic() + interval > end:
            return False
        time.sleep(interval)


def _local_url(host: str, port: int) -> str:
    h = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    h = f"[{h}]" if ":" in h else h
    return f"http://{h}:{port}"


def expose(
    port: int,
    env: EnvironmentInfo,
    token: str,
    prefer: str | None = None,
    probe: Callable[[str], bool] | None = None,
    *,
    host: str = "127.0.0.1",
    tunnel_timeout: float = 30.0,
    probe_deadline: float = 30.0,
    probe_interval: float = 1.5,
    tunnel_starter: Callable[[int], _cf.CloudflaredTunnel] | None = None,
    colab_url_fn: Callable[[int], str | None] | None = None,
) -> tuple[LaunchInfo, ExposureHandle]:
    """Expose an already-running backend on ``host:port``. Never raises for adapter failures.

    ``prefer``: None (runtime default), "colab_proxy", "cloudflared" or "local".
    ``probe``: callable taking the full health URL and returning True on HTTP 200
    (default :func:`http_health_probe`). ``tunnel_starter`` / ``colab_url_fn`` are
    injection points for tests and the facade.
    """
    if prefer is not None and prefer not in _VALID_PREFER:
        raise ValueError(f"prefer must be one of {sorted(_VALID_PREFER)} or None, got {prefer!r}")
    probe = probe or http_health_probe
    colab_url_fn = colab_url_fn or colab_proxy_url
    tunnel_starter = tunnel_starter or (lambda p: _cf.start_tunnel(p, timeout=tunnel_timeout, host=host))
    local_url = _local_url(host, port)
    diags: list[str] = []

    kind = env.kind
    if prefer is None:
        order: list[str] = {
            RuntimeKind.COLAB: ["colab_proxy", "cloudflared"],
            RuntimeKind.KAGGLE: ["cloudflared"],
        }.get(kind, [])
    elif prefer == "colab_proxy":
        order = ["colab_proxy", "cloudflared"]
    elif prefer == "cloudflared":
        order = ["cloudflared"]
    else:
        order = []

    for method in order:
        if method == "colab_proxy":
            if kind != RuntimeKind.COLAB:
                diags.append("Colab proxy requested but this is not a Colab runtime; skipped.")
                continue
            try:
                url = colab_url_fn(port)
            except Exception as exc:  # defensive: injected fns
                url = None
                diags.append(f"Colab proxy failed ({type(exc).__name__}).")
            if url:
                diags.append(
                    "Colab proxy URL is tied to your Google session: open it in the same browser that runs "
                    "this notebook. It cannot be health-probed from the kernel, so it is reported as "
                    "not publicly verified."
                )
                info = LaunchInfo(local_url=local_url, public_url=url, method="colab_proxy",
                                  publicly_verified=False, open_url=build_open_url(url, token), diagnostics=diags)
                return info, ExposureHandle("colab_proxy")
            diags.append("Colab proxy (google.colab.kernel.proxyPort) unavailable; trying cloudflared.")

        elif method == "cloudflared":
            if env.internet_access is False:
                diags.append(KAGGLE_INTERNET_HINT if kind == RuntimeKind.KAGGLE else
                             "No outbound internet detected; cloudflared tunnel skipped.")
                continue
            try:
                tunnel = tunnel_starter(port)
            except _cf.CloudflaredError as exc:
                diags.append(f"cloudflared tunnel failed: {exc}")
                if kind == RuntimeKind.KAGGLE and env.internet_access is None:
                    diags.append(KAGGLE_INTERNET_HINT)
                continue
            except Exception as exc:
                diags.append(f"cloudflared tunnel failed unexpectedly ({type(exc).__name__}: {exc}).")
                continue
            public = tunnel.url
            if not public:
                tunnel.stop()
                diags.append("cloudflared started but reported no URL; tunnel stopped.")
                continue
            verified = _probe_with_retries(probe, _health_url(public), probe_deadline, probe_interval)
            diags.append("cloudflared quick tunnels are best-effort (no uptime guarantee); the UI is protected "
                         "by the one-time session token.")
            if not verified:
                diags.append(
                    f"Tunnel URL {public} did not answer {HEALTH_PATH} within {probe_deadline:.0f}s; it may "
                    f"still become reachable (DNS propagation) but public access is NOT verified. "
                    f"The backend keeps running locally at {local_url}."
                )
            info = LaunchInfo(local_url=local_url, public_url=public, method="cloudflared",
                              publicly_verified=verified, open_url=build_open_url(public, token), diagnostics=diags)
            return info, ExposureHandle("cloudflared", tunnel=tunnel)

    # ---------------------------------------------------------------- local fallback
    if order:
        diags.append(f"No public URL could be established. The backend is still running locally at {local_url}.")
    if kind in (RuntimeKind.COLAB, RuntimeKind.KAGGLE):
        diags.append(f"{local_url} is only reachable from inside the {kind.value} VM, not from your browser.")
    elif prefer != "local" and not order:
        diags.append("Local mode: the UI is reachable only from this machine. Use prefer='cloudflared' to opt in "
                     "to a public quick tunnel.")
    info = LaunchInfo(local_url=local_url, public_url=None, method="local", publicly_verified=False,
                      open_url=build_open_url(local_url, token), diagnostics=diags)
    return info, ExposureHandle("local")
