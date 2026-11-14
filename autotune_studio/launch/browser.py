"""Open the studio UI in a *separate* browser tab.

Owned by: runtime-compatibility.

* Notebooks (Colab / Kaggle / Jupyter): display an HTML ``<a target="_blank" rel="noopener noreferrer">``
  button (always works because the user clicks it) and additionally attempt
  ``window.open`` via ``IPython.display.Javascript`` (may be blocked by popup blockers
  or by JupyterLab's untrusted-output policy; the button is the reliable path).
  ``google.colab.output.serve_kernel_port_as_window`` is deliberately *not* used: it builds its
  own URL from the port and we could not confirm that it preserves the ``#token=`` fragment.
* Scripts: :func:`webbrowser.open_new_tab`.

The URL (which carries the session token in its fragment) is written only into the
user's own notebook output / handed to the OS browser launcher. It is never logged.
"""

from __future__ import annotations

import html
import json
import logging
import webbrowser

from ..contracts import EnvironmentInfo

__all__ = ["open_in_new_tab", "render_link_html"]

log = logging.getLogger(__name__)


def render_link_html(url: str, label: str = "Open AutoTune Studio") -> str:
    href = html.escape(url, quote=True)
    return (
        f'<a href="{href}" target="_blank" rel="noopener noreferrer" '
        'style="display:inline-block;padding:8px 16px;border-radius:6px;background:#2563eb;color:#fff;'
        f'font-weight:600;text-decoration:none;font-family:sans-serif">{html.escape(label)} &#8599;</a>'
    )


def _window_open_js(url: str) -> str:
    # json.dumps gives a safely quoted JS string literal; "</" is escaped to avoid closing a <script>.
    lit = json.dumps(url).replace("</", "<\\/")
    return f"try {{ window.open({lit}, '_blank', 'noopener'); }} catch (e) {{}}"


def open_in_new_tab(url: str, env: EnvironmentInfo, *, try_popup: bool = True) -> bool:
    """Show/open ``url`` in a new tab. Returns True if a link was displayed or a browser launched."""
    if env.is_notebook:
        try:
            from IPython.display import HTML, Javascript, display  # type: ignore[import-not-found]
        except Exception:
            log.warning("IPython.display unavailable; cannot render the UI link in the notebook")
            return False
        try:
            display(HTML(render_link_html(url)))
        except Exception:
            log.warning("could not display the UI link")
            return False
        if try_popup:
            try:
                display(Javascript(_window_open_js(url)))
            except Exception:
                pass  # popup is best-effort
        return True
    try:
        ok = bool(webbrowser.open_new_tab(url))
    except Exception:
        ok = False
    if ok:
        log.info("opened the UI in a browser tab")
    else:
        log.info("no browser could be launched; open the printed UI URL manually")
    return ok
