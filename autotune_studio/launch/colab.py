"""Google Colab kernel proxy helpers.

Owned by: runtime-compatibility.

``google.colab.kernel.proxyPort(PORT)`` returns an https URL on
``*.googleusercontent.com`` (or ``*.colab.dev``) that forwards to ``localhost:PORT``
inside the VM. The URL is bound to the user's Google session cookies, so it can
be opened from the same browser but cannot be probed from the kernel.
"""

from __future__ import annotations

import logging

__all__ = ["colab_proxy_url"]

log = logging.getLogger(__name__)


def colab_proxy_url(port: int, timeout_sec: float = 15.0) -> str | None:
    """Return the Colab proxy URL for ``port``, or None outside Colab / on any failure."""
    try:
        from google.colab import output  # type: ignore[import-not-found]
    except Exception:
        return None
    script = "google.colab.kernel.proxyPort(%d)" % int(port)
    try:
        try:
            url = output.eval_js(script, timeout_sec=timeout_sec)
        except TypeError:  # older google.colab without timeout_sec
            url = output.eval_js(script)
    except Exception as exc:
        log.warning("Colab proxyPort(%d) failed: %s", port, type(exc).__name__)
        return None
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url.startswith("https://"):
        log.warning("Colab proxyPort returned an unexpected value; ignoring it")
        return None
    return url
