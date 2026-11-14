"""open_in_new_tab: notebook HTML anchor + window.open, script webbrowser, no token in logs."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from autotune_studio.contracts import EnvironmentInfo, RuntimeKind
from autotune_studio.launch import browser
from autotune_studio.launch.browser import open_in_new_tab, render_link_html

URL = 'https://x.trycloudflare.com/#token=SECRET"><script>alert(1)</script>'


def _env(nb: bool) -> EnvironmentInfo:
    kind = RuntimeKind.JUPYTER if nb else RuntimeKind.SCRIPT
    return EnvironmentInfo(kind=kind, python_version="3.11", platform="t", is_notebook=nb, workspace_dir="/w",
                           allowed_roots=["/w"])


def test_anchor_is_escaped_and_opens_new_tab() -> None:
    h = render_link_html(URL)
    assert 'target="_blank"' in h and 'rel="noopener noreferrer"' in h
    assert "<script>" not in h and "&quot;" in h


def test_notebook_displays_link_and_js(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    shown: list[object] = []
    disp = types.ModuleType("IPython.display")

    class HTML:
        def __init__(self, data: str) -> None:
            self.data = data

    class Javascript(HTML):
        pass

    disp.HTML, disp.Javascript = HTML, Javascript  # type: ignore[attr-defined]
    disp.display = shown.append  # type: ignore[attr-defined]
    ipy = types.ModuleType("IPython")
    ipy.display = disp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", ipy)
    monkeypatch.setitem(sys.modules, "IPython.display", disp)
    caplog.set_level(logging.DEBUG)
    assert open_in_new_tab(URL, _env(True)) is True
    assert isinstance(shown[0], HTML) and 'target="_blank"' in shown[0].data  # type: ignore[attr-defined]
    js = shown[1].data  # type: ignore[attr-defined]
    assert "window.open(" in js and "</script>" not in js
    assert "SECRET" not in caplog.text


def test_notebook_without_ipython_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    assert open_in_new_tab(URL, _env(True)) is False


def test_script_uses_webbrowser(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    opened: list[str] = []
    monkeypatch.setattr(browser.webbrowser, "open_new_tab", lambda u: opened.append(u) or True)
    caplog.set_level(logging.DEBUG)
    assert open_in_new_tab(URL, _env(False)) is True
    assert opened == [URL]
    assert "SECRET" not in caplog.text
