"""expose(): adapter order, fallbacks, probe-gated verification, fragment-only token."""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI

from autotune_studio.contracts import EnvironmentInfo, RuntimeKind
from autotune_studio.launch import adapters
from autotune_studio.launch import cloudflared as cf
from autotune_studio.launch.adapters import build_open_url, expose
from autotune_studio.launch.server import ServerThread

TOKEN = "tok_ABC-123_secret"


def _env(kind: RuntimeKind, internet: bool | None = None) -> EnvironmentInfo:
    return EnvironmentInfo(kind=kind, python_version="3.11", platform="test", is_notebook=kind != RuntimeKind.SCRIPT,
                           workspace_dir="/w", allowed_roots=["/w"], internet_access=internet)


class FakeTunnel:
    def __init__(self, url: str | None) -> None:
        self.url = url
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _no_tunnel(port: int):  # type: ignore[no-untyped-def]
    raise cf.CloudflaredError("cloudflared not found on PATH and downloading is disabled.")


def _assert_token_fragment_only(url: str | None) -> None:
    assert url is not None
    parts = urlsplit(url)
    assert parts.fragment == f"token={TOKEN}"
    assert TOKEN not in parts.query and TOKEN not in parts.path


def _assert_no_token_in_diags(info) -> None:  # type: ignore[no-untyped-def]
    assert all(TOKEN not in d for d in info.diagnostics)


def test_open_url_builder() -> None:
    assert build_open_url("http://127.0.0.1:7860", TOKEN) == f"http://127.0.0.1:7860/#token={TOKEN}"
    assert build_open_url("https://x.trycloudflare.com/", TOKEN) == f"https://x.trycloudflare.com/#token={TOKEN}"
    u = build_open_url("https://p.colab.dev/?authuser=0", "a b&c")
    assert urlsplit(u).query == "authuser=0" and urlsplit(u).fragment == "token=a%20b%26c"


def test_script_default_is_local_only() -> None:
    def must_not_run(port: int):  # type: ignore[no-untyped-def]
        raise AssertionError("tunnel must be opt-in for local scripts")

    info, handle = expose(7860, _env(RuntimeKind.SCRIPT), TOKEN, tunnel_starter=must_not_run)
    assert info.method == "local" and info.public_url is None and not info.publicly_verified
    assert info.local_url == "http://127.0.0.1:7860"
    _assert_token_fragment_only(info.open_url)
    assert any("prefer='cloudflared'" in d for d in info.diagnostics)
    handle.stop()


def test_colab_proxy_success_is_unverified() -> None:
    probes: list[str] = []
    info, handle = expose(
        7860, _env(RuntimeKind.COLAB), TOKEN,
        colab_url_fn=lambda p: f"https://{p}-abc.prod.colab.dev/",
        probe=lambda u: probes.append(u) or True, tunnel_starter=_no_tunnel,
    )
    assert info.method == "colab_proxy" and info.public_url == "https://7860-abc.prod.colab.dev/"
    assert info.publicly_verified is False and probes == []  # cannot be probed from the kernel
    assert any("same browser" in d for d in info.diagnostics)
    _assert_token_fragment_only(info.open_url)
    handle.stop()


def test_colab_proxy_unavailable_and_cloudflared_unavailable_falls_back_local() -> None:
    info, _ = expose(7860, _env(RuntimeKind.COLAB), TOKEN, colab_url_fn=lambda p: None, tunnel_starter=_no_tunnel)
    assert info.method == "local" and not info.publicly_verified and info.public_url is None
    text = " ".join(info.diagnostics)
    assert "Colab proxy" in text and "cloudflared tunnel failed" in text
    assert "still running locally at http://127.0.0.1:7860" in text
    _assert_token_fragment_only(info.open_url)
    _assert_no_token_in_diags(info)


def test_real_colab_proxy_fn_outside_colab_returns_none() -> None:
    from autotune_studio.launch.colab import colab_proxy_url

    assert colab_proxy_url(7860) is None


def test_colab_proxy_fn_with_fake_google_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    from autotune_studio.launch.colab import colab_proxy_url

    seen: list[str] = []
    output = types.SimpleNamespace(eval_js=lambda s, timeout_sec=None: seen.append(s) or "https://x-7860.colab.dev/")
    mod = types.ModuleType("google.colab")
    mod.output = output  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.colab", mod)
    assert colab_proxy_url(7860) == "https://x-7860.colab.dev/"
    assert seen == ["google.colab.kernel.proxyPort(7860)"]
    output.eval_js = lambda s, timeout_sec=None: "not a url"
    assert colab_proxy_url(7860) is None


def test_kaggle_internet_disabled_diagnostic() -> None:
    def must_not_run(port: int):  # type: ignore[no-untyped-def]
        raise AssertionError("no tunnel without internet")

    info, _ = expose(7860, _env(RuntimeKind.KAGGLE, internet=False), TOKEN, tunnel_starter=must_not_run)
    assert info.method == "local"
    text = " ".join(info.diagnostics)
    assert "Internet" in text and "Settings" in text and "only reachable from inside the kaggle VM" in text


def test_cloudflared_probe_success_verified() -> None:
    tunnel = FakeTunnel("https://quiet-river.trycloudflare.com")
    probes: list[str] = []
    info, handle = expose(7860, _env(RuntimeKind.KAGGLE, internet=True), TOKEN,
                          tunnel_starter=lambda p: tunnel, probe=lambda u: probes.append(u) or True)
    assert info.method == "cloudflared" and info.publicly_verified is True
    assert probes == ["https://quiet-river.trycloudflare.com/api/health"]
    _assert_token_fragment_only(info.open_url)
    assert all(TOKEN not in p for p in probes)
    handle.stop()
    assert tunnel.stopped


def test_cloudflared_probe_failure_not_verified() -> None:
    tunnel = FakeTunnel("https://slow.trycloudflare.com")
    info, handle = expose(7860, _env(RuntimeKind.KAGGLE, internet=True), TOKEN, tunnel_starter=lambda p: tunnel,
                          probe=lambda u: False, probe_deadline=0.3, probe_interval=0.1)
    assert info.method == "cloudflared" and info.publicly_verified is False
    assert any("NOT verified" in d and "http://127.0.0.1:7860" in d for d in info.diagnostics)
    handle.stop()


def test_probe_retries_until_success() -> None:
    results = iter([False, False, True])
    info, _ = expose(7860, _env(RuntimeKind.KAGGLE, internet=True), TOKEN,
                     tunnel_starter=lambda p: FakeTunnel("https://a.trycloudflare.com"),
                     probe=lambda u: next(results), probe_deadline=5, probe_interval=0.01)
    assert info.publicly_verified is True


def test_probe_exception_treated_as_failure() -> None:
    def boom(u: str) -> bool:
        raise RuntimeError("x")

    info, _ = expose(7860, _env(RuntimeKind.KAGGLE, internet=True), TOKEN,
                     tunnel_starter=lambda p: FakeTunnel("https://a.trycloudflare.com"),
                     probe=boom, probe_deadline=0.1, probe_interval=0.05)
    assert info.publicly_verified is False


def test_invalid_prefer() -> None:
    with pytest.raises(ValueError):
        expose(7860, _env(RuntimeKind.SCRIPT), TOKEN, prefer="ngrok")


def test_prefer_local_on_kaggle_skips_tunnel() -> None:
    def must_not_run(port: int):  # type: ignore[no-untyped-def]
        raise AssertionError

    info, _ = expose(7860, _env(RuntimeKind.KAGGLE, internet=True), TOKEN, prefer="local", tunnel_starter=must_not_run)
    assert info.method == "local"


# ----------------------------------------------------- cloudflared failure paths via the real starter
FAKE = textwrap.dedent(
    """
    import os, sys, time
    if os.environ.get("FAKE_CF_MODE") == "exit":
        print("ERR failed to request quick Tunnel", flush=True); sys.exit(1)
    time.sleep(60)
    """
)


@pytest.mark.parametrize("mode", ["missing", "checksum", "timeout", "exit"])
def test_cloudflared_failure_paths_fall_back_local(mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    monkeypatch.setattr(cf.shutil, "which", lambda name: None)  # never use a real installed binary here

    if mode == "missing":
        starter = lambda p: cf.start_tunnel(p, timeout=2, allow_download=False)
    elif mode == "checksum":
        def starter(p: int):  # type: ignore[no-untyped-def]
            binary = cf.download_cloudflared(tmp_path / "cache", asset="cloudflared-linux-amd64",
                                             downloader=lambda u, d: d.write_bytes(b"tampered"))
            return cf.start_tunnel(p, binary=str(binary))
    else:
        monkeypatch.setenv("FAKE_CF_MODE", mode)
        starter = lambda p: cf.start_tunnel(p, timeout=2, binary=[sys.executable, str(script)])

    info, handle = expose(7860, _env(RuntimeKind.SCRIPT), TOKEN, prefer="cloudflared", tunnel_starter=starter)
    assert info.method == "local" and info.publicly_verified is False and info.public_url is None
    assert info.diagnostics, "diagnostics must explain the failure"
    text = " ".join(info.diagnostics)
    assert "cloudflared tunnel failed" in text and "still running locally" in text
    expected = {"missing": "not found on PATH", "checksum": "Checksum mismatch", "timeout": "within 2s",
                "exit": "exited with code 1"}[mode]
    assert expected in text
    _assert_token_fragment_only(info.open_url)
    handle.stop()


# ----------------------------------------------------- end-to-end with a real server and default HTTP probe
def test_end_to_end_real_probe_against_local_server() -> None:
    app = FastAPI()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    with ServerThread(app, port=0) as srv:
        assert srv.wait_ready(15)
        # A "tunnel" whose public URL is the local server: exercises the real http_health_probe.
        info, handle = expose(srv.port, _env(RuntimeKind.KAGGLE, internet=True), TOKEN,
                              tunnel_starter=lambda p: FakeTunnel(srv.url), probe_deadline=5)
        assert info.publicly_verified is True and info.method == "cloudflared"
        handle.stop()
    # Server stopped: the real probe must now fail.
    assert adapters.http_health_probe(srv.url + "/api/health", timeout=1) is False
