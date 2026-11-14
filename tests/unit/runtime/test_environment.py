"""Runtime detection: Colab / Kaggle / Jupyter / script, allowed roots, internet probe."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from autotune_studio.contracts import RuntimeKind
from autotune_studio.core import environment
from autotune_studio.core.environment import detect_environment, ensure_workspace, probe_internet


def _no_probe(monkeypatch: pytest.MonkeyPatch, result: bool = True) -> list[int]:
    calls: list[int] = []

    def fake(timeout: float = 3.0, url: str = "") -> bool:
        calls.append(1)
        return result

    monkeypatch.setattr(environment, "probe_internet", fake)
    return calls


def test_script_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _no_probe(monkeypatch)
    env = detect_environment(cwd=str(tmp_path))
    assert env.kind == RuntimeKind.SCRIPT
    assert env.is_notebook is False
    cwd = str(tmp_path.resolve())
    assert env.workspace_dir == os.path.join(cwd, ".autotune")
    assert env.allowed_roots == [cwd, os.path.join(cwd, ".autotune")]
    assert env.internet_access is None and calls == []  # not probed outside Kaggle by default
    assert not Path(env.workspace_dir).exists()  # created lazily
    assert ensure_workspace(env).is_dir()


def test_colab_via_sys_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_probe(monkeypatch)
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    env = detect_environment(cwd=str(tmp_path))
    assert env.kind == RuntimeKind.COLAB
    assert env.is_notebook is True
    assert env.workspace_dir == "/content/autotune"
    assert env.allowed_roots == ["/content", "/content/autotune"]
    assert any("Drive" in n for n in env.notes)


def test_colab_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_probe(monkeypatch)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab_20261101")
    assert detect_environment().kind == RuntimeKind.COLAB


def test_colab_takes_precedence_over_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_probe(monkeypatch)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "x")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert detect_environment().kind == RuntimeKind.COLAB


def test_kaggle_env_probes_internet(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _no_probe(monkeypatch, result=True)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    env = detect_environment()
    assert env.kind == RuntimeKind.KAGGLE and env.is_notebook
    assert env.workspace_dir == "/kaggle/working/autotune"
    assert env.allowed_roots[:3] == ["/kaggle/input", "/kaggle/working", "/kaggle/temp"]
    assert env.internet_access is True and calls == [1]


def test_kaggle_internet_disabled_note(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_probe(monkeypatch, result=False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    env = detect_environment()
    assert env.internet_access is False
    assert any("Internet" in n and "Settings" in n for n in env.notes)


def test_kaggle_via_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_probe(monkeypatch)
    working = tmp_path / "kaggle" / "working"
    working.mkdir(parents=True)
    monkeypatch.setattr(environment, "CHECK_KAGGLE_DIR", True)
    monkeypatch.setattr(environment, "KAGGLE_WORKING", str(working))
    assert detect_environment(probe_network=False).kind == RuntimeKind.KAGGLE


def test_jupyter_via_ipython_shell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_probe(monkeypatch)
    shell_cls = type("ZMQInteractiveShell", (), {})
    fake_ipy = types.ModuleType("IPython")
    fake_ipy.get_ipython = lambda: shell_cls()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipy)
    env = detect_environment(cwd=str(tmp_path))
    assert env.kind == RuntimeKind.JUPYTER and env.is_notebook
    assert env.workspace_dir == os.path.join(str(tmp_path.resolve()), ".autotune")


def test_terminal_ipython_is_script(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_probe(monkeypatch)
    shell_cls = type("TerminalInteractiveShell", (), {})
    fake_ipy = types.ModuleType("IPython")
    fake_ipy.get_ipython = lambda: shell_cls()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipy)
    assert detect_environment().kind == RuntimeKind.SCRIPT


def test_jupyter_via_ipykernel_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_probe(monkeypatch)
    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))
    assert detect_environment().kind == RuntimeKind.JUPYTER


def test_extra_allowed_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_probe(monkeypatch)
    a = tmp_path / "a"
    a.mkdir()
    missing = tmp_path / "missing"
    monkeypatch.setenv("AUTOTUNE_ALLOWED_ROOTS", os.pathsep.join([str(a), "", str(missing), str(a)]))
    env = detect_environment(cwd=str(tmp_path / "cwd"))
    assert str(a.resolve()) in env.allowed_roots
    assert str(missing.resolve()) in env.allowed_roots
    assert env.allowed_roots.count(str(a.resolve())) == 1  # deduplicated
    assert any("does not exist" in n for n in env.notes)


def test_extra_roots_on_colab(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_probe(monkeypatch)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "x")
    monkeypatch.setenv("AUTOTUNE_ALLOWED_ROOTS", str(tmp_path))
    env = detect_environment()
    assert env.allowed_roots[:2] == ["/content", "/content/autotune"]
    assert str(tmp_path.resolve()) in env.allowed_roots


def test_workspace_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_probe(monkeypatch)
    ws = tmp_path / "ws"
    monkeypatch.setenv("AUTOTUNE_WORKSPACE", str(ws))
    env = detect_environment(cwd=str(tmp_path))
    assert env.workspace_dir == str(ws.resolve())
    assert env.workspace_dir in env.allowed_roots


def test_probe_flag_forces_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _no_probe(monkeypatch, result=False)
    env = detect_environment(probe_network=True)
    assert env.internet_access is False and calls == [1]
    assert any("No outbound internet" in n for n in env.notes)


def test_probe_internet_failure_returns_false() -> None:
    # Invalid host (.invalid TLD never resolves): must return False, never raise.
    assert probe_internet(timeout=0.5, url="https://host.invalid") is False


@pytest.mark.network
def test_probe_internet_real() -> None:
    assert probe_internet(timeout=5.0) is True
