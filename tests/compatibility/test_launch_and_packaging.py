"""Launch through the facade on a real local server, packaging metadata, CLI and Python version guard."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[2]
WEB_INDEX = REPO / "autotune_studio" / "web" / "index.html"


# --------------------------------------------------------------------------- local launch
def test_launch_local_serves_api_ui_and_shuts_down(tmp_path) -> None:
    from autotune_studio.studio import Studio

    s = Studio(workspace_dir=tmp_path / "ws", allowed_roots=[str(tmp_path)])
    try:
        info = s.launch(open_browser=False, prefer="local")
        assert info.method == "local" and info.public_url is None and info.publicly_verified is False
        assert info.local_url.startswith("http://127.0.0.1:")
        assert info.open_url and "#token=" in info.open_url and "?token=" not in info.open_url
        assert info.open_url.split("#", 1)[0].rstrip("/") == info.local_url
        assert s.auth.token not in " ".join(info.diagnostics)

        base = info.local_url
        r = httpx.get(f"{base}/api/health", timeout=5)
        assert r.status_code == 200 and r.json()["status"] == "ok"
        assert httpx.get(f"{base}/api/env", timeout=5).status_code == 401
        auth = {"Authorization": f"Bearer {s.auth.token}"}
        env = httpx.get(f"{base}/api/env", headers=auth, timeout=5)
        assert env.status_code == 200 and env.json()["kind"] in ("script", "jupyter")
        # /api/launch never leaks the tokenised open_url
        li = httpx.get(f"{base}/api/launch", headers=auth, timeout=5).json()
        assert li["open_url"] is None and li["method"] == "local"

        ui = httpx.get(f"{base}/", timeout=5)
        assert ui.status_code == 200 and "text/html" in ui.headers["content-type"]
        if WEB_INDEX.is_file():
            assert ui.text == WEB_INDEX.read_text(encoding="utf-8")
            deep = httpx.get(f"{base}/training/run", timeout=5)  # SPA history-route fallback
            assert deep.status_code == 200 and deep.text == ui.text
        # Calling launch again while running returns the same info (no second server).
        assert s.launch(open_browser=False, prefer="local") == info
    finally:
        s.shutdown()
    assert s.launch_info is None
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{base}/api/health", timeout=2)


@pytest.mark.skipif(not WEB_INDEX.is_file(), reason="frontend not built (autotune_studio/web/index.html)")
def test_built_ui_assets_referenced_by_index_are_served(tmp_path) -> None:
    import re

    from autotune_studio.studio import Studio

    html = WEB_INDEX.read_text(encoding="utf-8")
    assets = re.findall(r'(?:src|href)="(?:\./|/)?(assets/[^"]+)"', html)
    assert assets, "index.html references no built assets"
    s = Studio(workspace_dir=tmp_path / "ws")
    try:
        info = s.launch(open_browser=False, prefer="local")
        for a in assets:
            r = httpx.get(f"{info.local_url}/{a.lstrip('/')}", timeout=5)
            assert r.status_code == 200 and r.content, a
    finally:
        s.shutdown()


# --------------------------------------------------------------------------- packaging / CLI
def test_installed_distribution_has_console_entry_point() -> None:
    dist = metadata.distribution("autotune-studio")
    eps = {e.name: e for e in dist.entry_points if e.group == "console_scripts"}
    assert "autotune-studio" in eps
    assert eps["autotune-studio"].value == "autotune_studio.cli:main"
    assert callable(eps["autotune-studio"].load())
    import autotune_studio

    assert dist.version == autotune_studio.__version__


def test_python_m_env_prints_valid_json(tmp_path) -> None:
    env = {**os.environ, "PYTHONUTF8": "1", "AUTOTUNE_WORKSPACE": str(tmp_path / "ws")}
    for k in ("COLAB_RELEASE_TAG", "KAGGLE_KERNEL_RUN_TYPE"):
        env.pop(k, None)
    proc = subprocess.run([sys.executable, "-m", "autotune_studio", "env"], capture_output=True, text=True,
                          timeout=120, cwd=str(tmp_path), env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.loads(proc.stdout)
    assert set(data) == {"environment", "hardware"}
    assert data["environment"]["kind"] == "script"
    assert data["environment"]["python_version"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}")
    assert data["hardware"]["cpu_count"] >= 1 and data["hardware"]["ram_total_bytes"] > 0


def test_cli_help_lists_commands() -> None:
    proc = subprocess.run([sys.executable, "-m", "autotune_studio", "--help"], capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 0 and "launch" in proc.stdout and "env" in proc.stdout


# --------------------------------------------------------------------------- python version guard
def test_requires_python_covers_colab_kaggle_versions() -> None:
    from packaging.specifiers import SpecifierSet

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    spec = SpecifierSet(pyproject["project"]["requires-python"])
    for ok in ("3.10.12", "3.11.9", "3.12.3", "3.13.1"):  # Colab/Kaggle ship 3.10-3.12
        assert ok in spec, ok
    for bad in ("3.9.18", "3.14.0"):
        assert bad not in spec, bad
    assert f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}" in spec
    assert SpecifierSet(metadata.metadata("autotune-studio")["Requires-Python"]) == spec  # installed == source
    assert pyproject["tool"]["ruff"]["target-version"] == "py310"
