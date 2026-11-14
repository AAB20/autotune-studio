"""UI launch: in-process server thread and exposure adapters (ADR-0002). Owned by: runtime-compatibility."""

from __future__ import annotations

from .adapters import HEALTH_PATH, ExposureHandle, build_open_url, expose, http_health_probe
from .browser import open_in_new_tab, render_link_html
from .cloudflared import (
    PINNED_SHA256,
    PINNED_VERSION,
    CloudflaredError,
    CloudflaredTunnel,
    download_cloudflared,
    resolve_cloudflared,
    start_tunnel,
)
from .colab import colab_proxy_url
from .server import ServerThread, bind_socket, find_free_port

__all__ = [
    "HEALTH_PATH",
    "PINNED_SHA256",
    "PINNED_VERSION",
    "CloudflaredError",
    "CloudflaredTunnel",
    "ExposureHandle",
    "ServerThread",
    "bind_socket",
    "build_open_url",
    "colab_proxy_url",
    "download_cloudflared",
    "expose",
    "find_free_port",
    "http_health_probe",
    "open_in_new_tab",
    "render_link_html",
    "resolve_cloudflared",
    "start_tunnel",
]
