"""cloudflared: asset selection, binary resolution, checksum verification, URL parsing, fake process."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest

from autotune_studio.launch import cloudflared as cf
from autotune_studio.launch.cloudflared import (
    CloudflaredError,
    CloudflaredTunnel,
    asset_name,
    download_cloudflared,
    parse_tunnel_url,
    resolve_cloudflared,
)

FAKE_CF = textwrap.dedent(
    """
    import os, sys, time
    mode = os.environ.get("FAKE_CF_MODE", "url")
    assert sys.argv[1:4] == ["tunnel", "--url", sys.argv[3]] and sys.argv[-1] == "--no-autoupdate", sys.argv
    print("2026-11-14T00:00:00Z INF Thank you for trying Cloudflare Tunnel.", flush=True)
    if mode == "url":
        print('ERR failed to request: Post "https://api.trycloudflare.com/tunnel": retrying', flush=True)
        print("INF |  https://quiet-river-1234.trycloudflare.com  |", flush=True)
        print("INF Registered tunnel connection", flush=True)
        while True:
            print("INF heartbeat", flush=True)
            time.sleep(0.2)
    elif mode == "exit":
        print("ERR failed to unmarshal quick Tunnel: connection refused", flush=True)
        sys.exit(3)
    else:  # silent: never prints a URL
        time.sleep(60)
    """
)


@pytest.fixture()
def fake_cf(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_cloudflared.py"
    script.write_text(FAKE_CF)
    return [sys.executable, str(script)]


# --------------------------------------------------------------------------- assets / pins
@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", "cloudflared-linux-amd64"),
        ("Linux", "aarch64", "cloudflared-linux-arm64"),
        ("Windows", "AMD64", "cloudflared-windows-amd64.exe"),
        ("Darwin", "arm64", "cloudflared-darwin-arm64.tgz"),
        ("Darwin", "x86_64", "cloudflared-darwin-amd64.tgz"),
        ("Linux", "riscv64", None),
        ("Windows", "ARM64", None),
    ],
)
def test_asset_name(system: str, machine: str, expected: str | None) -> None:
    assert asset_name(system, machine) == expected


def test_pinned_table_covers_supported_assets() -> None:
    for sysm in [("Linux", "x86_64"), ("Linux", "aarch64"), ("Windows", "AMD64"), ("Darwin", "arm64"),
                 ("Darwin", "x86_64")]:
        a = asset_name(*sysm)
        assert a in cf.PINNED_SHA256 and len(cf.PINNED_SHA256[a]) == 64


# --------------------------------------------------------------------------- resolution
def test_resolve_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    b = tmp_path / "cloudflared.exe"
    b.write_bytes(b"x")
    monkeypatch.setenv("AUTOTUNE_CLOUDFLARED", str(b))
    assert resolve_cloudflared(allow_download=False) == str(b)


def test_resolve_env_override_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOTUNE_CLOUDFLARED", str(tmp_path / "nope"))
    with pytest.raises(CloudflaredError, match="not a file"):
        resolve_cloudflared()


def test_resolve_binary_missing_no_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf.shutil, "which", lambda name: None)
    with pytest.raises(CloudflaredError, match="not found on PATH"):
        resolve_cloudflared(allow_download=False)


def test_resolve_prefers_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf.shutil, "which", lambda name: "/usr/local/bin/cloudflared")
    assert resolve_cloudflared(allow_download=False) == "/usr/local/bin/cloudflared"


# --------------------------------------------------------------------------- download + checksum
PAYLOAD = b"fake cloudflared binary"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _writer(data: bytes, calls: list[str]):  # type: ignore[no-untyped-def]
    def dl(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(data)

    return dl


def test_download_verifies_and_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(cf.PINNED_SHA256, "cloudflared-linux-amd64", PAYLOAD_SHA)
    calls: list[str] = []
    p = download_cloudflared(tmp_path, asset="cloudflared-linux-amd64", downloader=_writer(PAYLOAD, calls))
    assert p.read_bytes() == PAYLOAD
    assert calls == [cf.RELEASE_URL.format(version=cf.PINNED_VERSION, asset="cloudflared-linux-amd64")]
    # second call: served from cache, no download
    p2 = download_cloudflared(tmp_path, asset="cloudflared-linux-amd64", downloader=_writer(PAYLOAD, calls))
    assert p2 == p and len(calls) == 1
    # tampering with the cached binary forces a re-download
    p.write_bytes(b"tampered")
    p3 = download_cloudflared(tmp_path, asset="cloudflared-linux-amd64", downloader=_writer(PAYLOAD, calls))
    assert p3.read_bytes() == PAYLOAD and len(calls) == 2


def test_checksum_mismatch_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    with pytest.raises(CloudflaredError, match="Checksum mismatch"):
        download_cloudflared(tmp_path, asset="cloudflared-linux-amd64", downloader=_writer(b"evil", calls))
    vdir = tmp_path / cf.PINNED_VERSION
    assert not any(vdir.iterdir())  # nothing left behind


def test_unpinned_version_requires_runtime_checksum(tmp_path: Path) -> None:
    calls: list[str] = []
    with pytest.raises(CloudflaredError, match="No trusted SHA256"):
        download_cloudflared(tmp_path, version="2099.1.0", asset="cloudflared-linux-amd64",
                             downloader=_writer(PAYLOAD, calls), checksum_fetcher=lambda v, a: None)
    assert calls == []  # refused before downloading
    p = download_cloudflared(tmp_path, version="2099.1.0", asset="cloudflared-linux-amd64",
                             downloader=_writer(PAYLOAD, calls), checksum_fetcher=lambda v, a: PAYLOAD_SHA)
    assert p.read_bytes() == PAYLOAD


def test_darwin_tgz_extracts_only_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in (("../evil", b"no"), ("cloudflared", PAYLOAD)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    archive = buf.getvalue()
    monkeypatch.setitem(cf.PINNED_SHA256, "cloudflared-darwin-arm64.tgz", hashlib.sha256(archive).hexdigest())
    p = download_cloudflared(tmp_path / "c", asset="cloudflared-darwin-arm64.tgz", downloader=_writer(archive, []))
    assert p.read_bytes() == PAYLOAD
    assert not (tmp_path / "evil").exists()


def test_download_network_error_wrapped(tmp_path: Path) -> None:
    def fail(url: str, dest: Path) -> None:
        raise ConnectionError("no route to host")

    with pytest.raises(CloudflaredError, match="Downloading cloudflared failed"):
        download_cloudflared(tmp_path, asset="cloudflared-linux-amd64", downloader=fail)


def test_fetch_release_checksum_parses_digest_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    sha = "a" * 64

    class R:
        status_code = 200

        def __init__(self, data: dict) -> None:
            self._d = data

        def json(self) -> dict:
            return self._d

    monkeypatch.setattr(httpx, "get", lambda *a, **k: R({"assets": [{"name": "x", "digest": "sha256:" + sha}]}))
    assert cf.fetch_release_checksum("v", "x") == sha
    monkeypatch.setattr(httpx, "get", lambda *a, **k: R({"assets": [], "body": f"```\nx: {'b' * 64}\n```"}))
    assert cf.fetch_release_checksum("v", "x") == "b" * 64
    monkeypatch.setattr(httpx, "get", lambda *a, **k: R({"assets": [], "body": ""}))
    assert cf.fetch_release_checksum("v", "x") is None


# --------------------------------------------------------------------------- URL parsing + process
def test_parse_tunnel_url_ignores_api_host() -> None:
    assert parse_tunnel_url('Post "https://api.trycloudflare.com/tunnel": EOF') is None
    assert parse_tunnel_url("|  https://Quiet-River-1.trycloudflare.com  |") == "https://quiet-river-1.trycloudflare.com"
    assert parse_tunnel_url("https://evil.trycloudflare.com.attacker.net") is None
    assert parse_tunnel_url("nothing here") is None


def test_fake_tunnel_url_found_and_stop(fake_cf: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CF_MODE", "url")
    t = CloudflaredTunnel(fake_cf, 7860)
    assert t.command[-4:] == ["tunnel", "--url", "http://127.0.0.1:7860", "--no-autoupdate"]
    url = t.start(timeout=20)
    assert url == "https://quiet-river-1234.trycloudflare.com"
    assert t.is_running()
    t.stop()
    assert not t.is_running()


def test_fake_tunnel_exits_early(fake_cf: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CF_MODE", "exit")
    t = CloudflaredTunnel(fake_cf, 7860)
    with pytest.raises(CloudflaredError, match="exited with code 3"):
        t.start(timeout=20)
    assert not t.is_running()


def test_fake_tunnel_no_url_within_timeout(fake_cf: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CF_MODE", "silent")
    t = CloudflaredTunnel(fake_cf, 7860)
    with pytest.raises(CloudflaredError, match="within 2s"):
        t.start(timeout=2)
    assert not t.is_running()  # process killed on timeout


def test_binary_not_executable(tmp_path: Path) -> None:
    t = CloudflaredTunnel(str(tmp_path / ("missing.exe" if os.name == "nt" else "missing")), 7860)
    with pytest.raises(CloudflaredError, match="Could not start cloudflared"):
        t.start(timeout=2)
