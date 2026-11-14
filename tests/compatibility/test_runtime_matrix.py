"""Runtime matrix: Colab / Kaggle / Jupyter / script detection -> Studio allowed roots -> launch
adapter ordering, diagnostics and the ``publicly_verified`` rule, with notebook runtimes simulated.

Unit tests in ``tests/unit/runtime`` cover each function in isolation; this module checks the
combination (detection feeding ``Studio`` and ``expose``) per runtime.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
import pytest

from autotune_studio.contracts import RuntimeKind
from autotune_studio.core import environment
from autotune_studio.core.environment import KAGGLE_INTERNET_HINT, detect_environment
from autotune_studio.launch.adapters import expose

TOKEN = "tok-" + "x" * 40


class FakeTunnel:
    def __init__(self, url: str | None) -> None:
        self.url = url
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class Recorder:
    """Fake colab proxy fn + tunnel starter recording call order."""

    def __init__(self, colab_url: str | None = None, tunnel_url: str | None = "https://abc.trycloudflare.com",
                 tunnel_error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.colab_url = colab_url
        self.tunnel_url = tunnel_url
        self.tunnel_error = tunnel_error
        self.tunnels: list[FakeTunnel] = []

    def colab(self, port: int) -> str | None:
        self.calls.append("colab_proxy")
        return self.colab_url

    def tunnel(self, port: int) -> FakeTunnel:
        self.calls.append("cloudflared")
        if self.tunnel_error is not None:
            raise self.tunnel_error
        t = FakeTunnel(self.tunnel_url)
        self.tunnels.append(t)
        return t


# --------------------------------------------------------------------------- runtime simulators
def _sim_colab(mp: pytest.MonkeyPatch, tmp: Path, via: str = "module") -> Path:
    root = tmp / "content"
    root.mkdir()
    mp.setattr(environment, "COLAB_ROOT", str(root))
    if via == "module":
        mod = types.ModuleType("google.colab")
        mod.output = types.SimpleNamespace(  # type: ignore[attr-defined]
            eval_js=lambda s, timeout_sec=None: "https://8000-m-s-abc.prod.colab.dev/")
        mp.setitem(sys.modules, "google.colab", mod)
    else:
        mp.setenv("COLAB_RELEASE_TAG", "release-colab-20261101")
    return root


def _sim_kaggle(mp: pytest.MonkeyPatch, tmp: Path, via: str = "env") -> dict[str, Path]:
    paths = {k: tmp / "kaggle" / k for k in ("input", "working", "temp")}
    for p in paths.values():
        p.mkdir(parents=True)
    mp.setattr(environment, "KAGGLE_INPUT", str(paths["input"]))
    mp.setattr(environment, "KAGGLE_WORKING", str(paths["working"]))
    mp.setattr(environment, "KAGGLE_TEMP", str(paths["temp"]))
    if via == "env":
        mp.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    else:  # directory-based detection (as on a real Kaggle VM)
        mp.setattr(environment, "CHECK_KAGGLE_DIR", True)
    return paths


def _sim_jupyter(mp: pytest.MonkeyPatch) -> None:
    shell_cls = type("ZMQInteractiveShell", (), {})
    ipy = types.ModuleType("IPython")
    ipy.get_ipython = lambda: shell_cls()  # type: ignore[attr-defined]
    mp.setitem(sys.modules, "IPython", ipy)


def _no_internet_probe(mp: pytest.MonkeyPatch, result: bool) -> None:
    mp.setattr(environment, "probe_internet", lambda *a, **k: result)


# --------------------------------------------------------------------------- detection matrix
@pytest.mark.parametrize("via", ["module", "env"])
def test_colab_detection_roots_and_studio(monkeypatch, tmp_path, via) -> None:
    root = _sim_colab(monkeypatch, tmp_path, via)
    env = detect_environment(probe_network=False)
    assert env.kind == RuntimeKind.COLAB and env.is_notebook
    assert env.workspace_dir == f"{root}/autotune"
    assert str(root) in env.allowed_roots
    assert any("Drive is not mounted" in n for n in env.notes)

    from autotune_studio.studio import Studio

    s = Studio(workspace_dir=tmp_path / "ws")
    try:
        assert s.detect_environment().kind == RuntimeKind.COLAB
        assert str(root) in s.detect_environment().allowed_roots
        assert str((tmp_path / "ws").resolve()) in s.detect_environment().allowed_roots
        # a file under /content is loadable, one outside is not
        f = root / "d.jsonl"
        f.write_text('{"text": "a"}\n{"text": "b"}\n', encoding="utf-8")
        assert s.load_dataset("jsonl", path=str(f)).detected_format == "text"
    finally:
        s.shutdown()


@pytest.mark.parametrize("via", ["env", "dir"])
def test_kaggle_detection_roots(monkeypatch, tmp_path, via) -> None:
    paths = _sim_kaggle(monkeypatch, tmp_path, via)
    _no_internet_probe(monkeypatch, True)
    env = detect_environment()
    assert env.kind == RuntimeKind.KAGGLE and env.is_notebook
    assert env.workspace_dir == f"{paths['working']}/autotune"
    for k in ("input", "working", "temp"):
        assert str(paths[k]) in env.allowed_roots
    assert env.internet_access is True  # Kaggle probes by default
    assert any("/kaggle/input is read-only" in n for n in env.notes)


def test_kaggle_internet_off_diagnostics(monkeypatch, tmp_path) -> None:
    _sim_kaggle(monkeypatch, tmp_path)
    _no_internet_probe(monkeypatch, False)
    env = detect_environment()
    assert env.kind == RuntimeKind.KAGGLE and env.internet_access is False
    assert KAGGLE_INTERNET_HINT in env.notes

    rec = Recorder()
    info, handle = expose(8000, env, TOKEN, colab_url_fn=rec.colab, tunnel_starter=rec.tunnel,
                          probe=lambda u: True, probe_deadline=0.05, probe_interval=0.01)
    assert rec.calls == []  # tunnel never attempted without internet
    assert info.method == "local" and info.public_url is None and not info.publicly_verified
    text = " ".join(info.diagnostics)
    assert "Internet" in text and "only reachable from inside the kaggle VM" in text
    handle.stop()


def test_kaggle_batch_run_note(monkeypatch, tmp_path) -> None:
    _sim_kaggle(monkeypatch, tmp_path)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    env = detect_environment(probe_network=False)
    assert any("batch run" in n.lower() for n in env.notes)


def test_jupyter_and_script_detection(monkeypatch, tmp_path) -> None:
    env = detect_environment(probe_network=False, cwd=str(tmp_path))
    assert env.kind == RuntimeKind.SCRIPT and not env.is_notebook
    assert env.allowed_roots[0] == str(tmp_path.resolve())
    assert env.workspace_dir == str(tmp_path.resolve() / ".autotune")
    _sim_jupyter(monkeypatch)
    env = detect_environment(probe_network=False, cwd=str(tmp_path))
    assert env.kind == RuntimeKind.JUPYTER and env.is_notebook
    assert str(tmp_path.resolve()) in env.allowed_roots


def test_colab_wins_over_kaggle_and_jupyter(monkeypatch, tmp_path) -> None:
    _sim_kaggle(monkeypatch, tmp_path)
    _sim_jupyter(monkeypatch)
    _sim_colab(monkeypatch, tmp_path, "env")
    assert detect_environment(probe_network=False).kind == RuntimeKind.COLAB


def test_extra_allowed_roots_env_on_every_runtime(monkeypatch, tmp_path) -> None:
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.setenv("AUTOTUNE_ALLOWED_ROOTS", str(extra))
    for setup in (lambda: None, lambda: _sim_jupyter(monkeypatch),
                  lambda: _sim_kaggle(monkeypatch, tmp_path), lambda: _sim_colab(monkeypatch, tmp_path, "env")):
        setup()
        env = detect_environment(probe_network=False)
        assert str(extra.resolve()) in env.allowed_roots, env.kind


# --------------------------------------------------------------------------- launch adapter ordering
def _env_for(kind: str, monkeypatch, tmp_path, internet: bool | None = None):
    if kind == "colab":
        _sim_colab(monkeypatch, tmp_path, "env")
    elif kind == "kaggle":
        _sim_kaggle(monkeypatch, tmp_path)
    elif kind == "jupyter":
        _sim_jupyter(monkeypatch)
    env = detect_environment(probe_network=False)
    return env.model_copy(update={"internet_access": internet})


@pytest.mark.parametrize("kind, colab_url, expected_calls, expected_method", [
    ("colab", "https://p.colab.dev/", ["colab_proxy"], "colab_proxy"),
    ("colab", None, ["colab_proxy", "cloudflared"], "cloudflared"),
    ("kaggle", None, ["cloudflared"], "cloudflared"),
    ("jupyter", None, [], "local"),
    ("script", None, [], "local"),
])
def test_default_adapter_order_per_runtime(monkeypatch, tmp_path, kind, colab_url, expected_calls,
                                           expected_method) -> None:
    env = _env_for(kind, monkeypatch, tmp_path)
    rec = Recorder(colab_url=colab_url)
    info, handle = expose(8000, env, TOKEN, colab_url_fn=rec.colab, tunnel_starter=rec.tunnel,
                          probe=lambda u: True, probe_deadline=0.05, probe_interval=0.01)
    assert rec.calls == expected_calls
    assert info.method == expected_method
    assert info.open_url and "#token=" in info.open_url and "?token=" not in info.open_url
    assert TOKEN not in " ".join(info.diagnostics)
    if expected_method == "colab_proxy":
        assert info.public_url == colab_url and info.publicly_verified is False  # cannot be probed
    elif expected_method == "cloudflared":
        assert info.publicly_verified is True
    else:
        assert info.public_url is None and info.publicly_verified is False
    handle.stop()


@pytest.mark.parametrize("probe_ok", [True, False])
def test_publicly_verified_only_with_passing_probe(monkeypatch, tmp_path, probe_ok) -> None:
    env = _env_for("kaggle", monkeypatch, tmp_path, internet=True)
    rec = Recorder()
    probed: list[str] = []

    def probe(url: str) -> bool:
        probed.append(url)
        return probe_ok

    info, handle = expose(8000, env, TOKEN, tunnel_starter=rec.tunnel, probe=probe,
                          probe_deadline=0.05, probe_interval=0.01)
    assert info.method == "cloudflared" and info.public_url == "https://abc.trycloudflare.com"
    assert probed and probed[0] == "https://abc.trycloudflare.com/api/health"
    assert info.publicly_verified is probe_ok
    if not probe_ok:
        assert any("NOT verified" in d for d in info.diagnostics)
    handle.stop()
    assert rec.tunnels[0].stopped


def test_tunnel_failure_falls_back_to_local_with_diagnostics(monkeypatch, tmp_path) -> None:
    from autotune_studio.launch.cloudflared import CloudflaredError

    env = _env_for("colab", monkeypatch, tmp_path)
    rec = Recorder(colab_url=None, tunnel_error=CloudflaredError("binary not found"))
    info, _ = expose(8000, env, TOKEN, colab_url_fn=rec.colab, tunnel_starter=rec.tunnel,
                     probe=lambda u: True, probe_deadline=0.05)
    assert rec.calls == ["colab_proxy", "cloudflared"]
    assert info.method == "local" and not info.publicly_verified
    text = " ".join(info.diagnostics)
    assert "binary not found" in text and "only reachable from inside the colab VM" in text


def test_studio_launch_on_simulated_colab_uses_proxy(monkeypatch, tmp_path) -> None:
    """Facade-level: Studio.launch on (fake) Colab picks the kernel proxy and trusts its origin for WS."""
    _sim_colab(monkeypatch, tmp_path, "module")
    from autotune_studio.studio import Studio

    s = Studio(workspace_dir=tmp_path / "ws")
    try:
        info = s.launch(open_browser=False)
        assert info.method == "colab_proxy" and info.publicly_verified is False
        assert info.public_url == "https://8000-m-s-abc.prod.colab.dev/"
        assert info.open_url.startswith(info.public_url) and "#token=" in info.open_url
        assert "?token=" not in info.open_url
        assert info.public_url in s.extra_origins
        from fastapi.testclient import TestClient

        with TestClient(s.create_app()) as c:
            hdr = {"Authorization": f"Bearer {s.auth.token}", "Origin": "https://8000-m-s-abc.prod.colab.dev"}
            with c.websocket_connect("/ws", headers=hdr) as ws:
                assert ws.receive_json()["type"] == "hello"
    finally:
        s.shutdown()
    assert s.launch_info is None
