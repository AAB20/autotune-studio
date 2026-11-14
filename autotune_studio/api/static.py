"""SPA hosting and response security headers.

Framing policy (deliberate): we send **no** ``X-Frame-Options`` and **no** CSP
``frame-ancestors``. Colab / Jupyter front-ends may render the studio inside an output-cell
iframe (``google.colab.output.serve_kernel_port_as_iframe`` / ``IPython.display.IFrame``);
blocking framing would break that display path. Clickjacking risk is limited because every
state-changing action needs the session cookie, which is ``SameSite=Strict`` — a page on another
site framing the UI gets an unauthenticated UI (the iframe has no ``#token`` fragment and the
cookie is not sent in a cross-site context).
"""

from __future__ import annotations

from pathlib import Path

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..contracts import ErrorCode
from .errors import error_response

__all__ = ["DEFAULT_WEB_DIR", "SecurityHeadersMiddleware", "spa_handler"]

DEFAULT_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

#: Applied to HTML documents. ``connect-src`` lists ws:/wss: because some browsers do not yet
#: treat ``'self'`` as covering same-host WebSocket URLs. No ``frame-ancestors`` (see module doc).
CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        "connect-src 'self' ws: wss:",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ]
)

_NOT_BUILT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>AutoTune Studio</title>
<meta name="referrer" content="no-referrer"></head>
<body style="font-family:system-ui,sans-serif;max-width:40rem;margin:4rem auto;line-height:1.5">
<h1>AutoTune Studio</h1>
<p>The backend is running, but the web UI has not been built.</p>
<p>Build it with:</p>
<pre>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>The API is available under <code>/api</code> (see <code>/api/health</code>).</p>
</body></html>
"""


class SecurityHeadersMiddleware:
    """Pure-ASGI middleware (safe for streaming and WebSockets) adding security headers.

    * every HTTP response: ``X-Content-Type-Options: nosniff``, ``Referrer-Policy: no-referrer``,
      (no CORP/COOP: they can interfere with notebook iframes and the new-tab launch flow);
    * ``/api/*``: ``Cache-Control: no-store`` (responses may contain paths and run state).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        is_api = path == "/api" or path.startswith("/api/")

        async def _send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "no-referrer")
                if is_api:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, _send)


def _index_response(web_dir: Path) -> Response:
    index = web_dir / "index.html"
    if index.is_file():
        return FileResponse(
            index, media_type="text/html", headers={"Cache-Control": "no-cache", "Content-Security-Policy": CSP}
        )
    return HTMLResponse(_NOT_BUILT, headers={"Cache-Control": "no-cache", "Content-Security-Policy": CSP})


def spa_handler(web_dir: Path):
    """Build the catch-all GET handler: static file if it exists under ``web_dir``, else ``index.html``.

    ``/api/*`` and ``/ws`` never fall back to the SPA (unknown API routes return a JSON 404).
    The UI is detected per request, so running ``npm run build`` after launch works without restart.
    """
    root = web_dir.resolve()

    async def serve(request: Request) -> Response:
        rel: str = request.path_params.get("path", "") or ""
        first = rel.split("/", 1)[0]
        if first in ("api", "ws"):
            return error_response(404, ErrorCode.NOT_FOUND, "not found")
        if rel and "\x00" not in rel:
            try:
                candidate = (root / rel).resolve()
            except (OSError, ValueError):
                candidate = None
            if candidate is not None and candidate.is_relative_to(root) and candidate.is_file():
                headers = {}
                if candidate.parent.name == "assets":  # content-hashed by Vite
                    headers["Cache-Control"] = "public, max-age=31536000, immutable"
                elif candidate.suffix.lower() in (".html", ".htm"):
                    headers["Cache-Control"] = "no-cache"
                    headers["Content-Security-Policy"] = CSP
                return FileResponse(candidate, headers=headers)
            if first == "assets":  # missing hashed asset: a real 404, not the HTML shell
                return error_response(404, ErrorCode.NOT_FOUND, "not found")
        return _index_response(root)

    return serve
