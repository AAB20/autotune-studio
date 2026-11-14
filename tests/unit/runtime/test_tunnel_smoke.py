"""Real cloudflared quick-tunnel smoke test (opens a public tunnel to a throwaway health-only app).

Opt-in: requires network, a cloudflared binary (PATH / AUTOTUNE_CLOUDFLARED / verified download)
and ``AUTOTUNE_TUNNEL_SMOKE=1`` so a plain test run never opens a public tunnel.
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI

from autotune_studio.contracts import EnvironmentInfo, RuntimeKind
from autotune_studio.launch import cloudflared as cf
from autotune_studio.launch.adapters import expose
from autotune_studio.launch.server import ServerThread

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("AUTOTUNE_TUNNEL_SMOKE") != "1", reason="set AUTOTUNE_TUNNEL_SMOKE=1"),
]


def test_real_quick_tunnel_roundtrip() -> None:
    try:
        binary = cf.resolve_cloudflared(allow_download=os.environ.get("AUTOTUNE_TUNNEL_DOWNLOAD") == "1")
    except cf.CloudflaredError as exc:
        pytest.skip(str(exc))

    app = FastAPI()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    env = EnvironmentInfo(kind=RuntimeKind.SCRIPT, python_version="", platform="", is_notebook=False,
                          workspace_dir=".", allowed_roots=["."], internet_access=True)
    with ServerThread(app, port=0) as srv:
        assert srv.wait_ready(15)
        info, handle = expose(srv.port, env, "smoke-token", prefer="cloudflared",
                              tunnel_starter=lambda p: cf.start_tunnel(p, timeout=60, binary=binary),
                              probe_deadline=60, probe_interval=2)
        try:
            print("\nLaunchInfo:", info.model_dump(exclude={"open_url"}))
            assert info.method == "cloudflared", info.diagnostics
            assert info.public_url and info.public_url.endswith(".trycloudflare.com")
            assert info.publicly_verified is True, info.diagnostics
        finally:
            handle.stop()
        assert handle.tunnel is None
