"""In-process uvicorn server running in a daemon thread.

Owned by: runtime-compatibility.

Works from a Jupyter/Colab/Kaggle cell: the notebook's asyncio loop stays in the
main thread untouched; the server runs its *own* event loop in a background
thread. Signal handlers are never installed (they may only be installed from the
main thread and would hijack the kernel's Ctrl-C/interrupt handling).

The listening socket is bound *before* the thread starts so port conflicts are
reported synchronously and there is no find-port/bind race. Default host is
``127.0.0.1`` - tunnels and the Colab proxy connect to localhost, so the server
is never exposed on all interfaces unless the caller explicitly asks for it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
import time
from collections.abc import Iterable, Iterator
from typing import Any

__all__ = ["ServerThread", "bind_socket", "find_free_port"]

log = logging.getLogger(__name__)

_PORT_SCAN = 20


def _new_socket(host: str) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    if os.name == "nt":
        # Prevent another process from binding the same port with SO_REUSEADDR (port hijack).
        excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if excl is not None:
            sock.setsockopt(socket.SOL_SOCKET, excl, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allow quick rebinding after TIME_WAIT
    return sock


def bind_socket(host: str, ports: Iterable[int]) -> socket.socket:
    """Bind and listen on the first free port of ``ports`` (0 = OS-assigned). Raises OSError if none."""
    last: OSError | None = None
    for port in ports:
        sock = _new_socket(host)
        try:
            sock.bind((host, int(port)))
            sock.listen(2048)
            sock.setblocking(False)
            return sock
        except OSError as exc:
            last = exc
            sock.close()
    raise last or OSError("no port candidates given")


def _candidates(preferred: int | None) -> Iterator[int]:
    if preferred:
        yield from range(preferred, min(preferred + _PORT_SCAN, 65536))
    yield 0


def find_free_port(preferred: int = 7860, host: str = "127.0.0.1") -> int:
    """Return ``preferred`` if free, else the next free port in ``preferred..+19``, else an OS-assigned one.

    Note: the port is released before returning, so another process could grab it;
    :class:`ServerThread` avoids that race by binding the socket itself.
    """
    sock = bind_socket(host, _candidates(preferred))
    try:
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _make_server_class() -> type:
    import uvicorn

    class _ThreadServer(uvicorn.Server):
        """uvicorn.Server that never touches process signal handlers."""

        def install_signal_handlers(self) -> None:  # uvicorn < 0.29
            return None

        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:  # uvicorn >= 0.29
            yield

    return _ThreadServer


class ServerThread:
    """Run an ASGI ``app`` with uvicorn inside a daemon thread of the current process.

    >>> srv = ServerThread(app, port=None)   # picks 7860 or the next free port
    >>> srv.start(); srv.wait_ready(15)
    >>> srv.url  # 'http://127.0.0.1:7860'
    >>> srv.stop()
    """

    def __init__(
        self,
        app: Any,
        host: str = "127.0.0.1",
        port: int | None = None,
        *,
        preferred_port: int = 7860,
        log_level: str = "warning",
        access_log: bool = False,
        health_path: str | None = "/api/health",
        **config_kwargs: Any,
    ) -> None:
        self.app = app
        self.host = host
        self.requested_port = port
        self.preferred_port = preferred_port
        self.log_level = log_level
        self.access_log = access_log
        self.health_path = health_path
        self.config_kwargs = config_kwargs
        self.error: BaseException | None = None
        self._server: Any = None
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self.port: int | None = None

    # ------------------------------------------------------------------ properties
    @property
    def url(self) -> str:
        if self.port is None:
            raise RuntimeError("server not started")
        host = f"[{self.host}]" if ":" in self.host else self.host
        if host in ("0.0.0.0", "[::]"):
            host = "127.0.0.1"
        return f"http://{host}:{self.port}"

    @property
    def started(self) -> bool:
        return bool(self._server is not None and self._server.started)

    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> ServerThread:
        """Bind the socket and start the server thread. Raises OSError if the port cannot be bound."""
        if self.is_running():
            return self
        import uvicorn

        ports = [self.requested_port] if self.requested_port is not None else list(_candidates(self.preferred_port))
        self._sock = bind_socket(self.host, ports)
        self.port = int(self._sock.getsockname()[1])
        kwargs: dict[str, Any] = {
            "log_config": None,  # do not reconfigure the notebook's logging
            "lifespan": "auto",
            "loop": "asyncio",
            "timeout_graceful_shutdown": 5,
        }
        kwargs.update(self.config_kwargs)
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level=self.log_level,
            access_log=self.access_log, **kwargs,
        )
        self._server = _make_server_class()(config)
        self.error = None
        self._thread = threading.Thread(target=self._run, name=f"autotune-uvicorn-{self.port}", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            # Server.run() creates a fresh event loop in this thread (asyncio.run).
            self._server.run(sockets=[self._sock])
        except BaseException as exc:  # SystemExit on startup failure must not kill the kernel
            self.error = exc
            log.error("uvicorn server thread exited with %s", type(exc).__name__)
        finally:
            self._close_socket()

    def wait_ready(self, timeout: float = 15.0) -> bool:
        """Wait until the server reports started and (if ``health_path``) the health route returns 200."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error is not None or (self._thread is not None and not self._thread.is_alive()):
                return False
            if self.started:
                if self.health_path is None or self._health_ok(max(0.1, min(2.0, deadline - time.monotonic()))):
                    return True
            time.sleep(0.05)
        return False

    def _health_ok(self, timeout: float) -> bool:
        try:
            import httpx

            r = httpx.get(self.url + self.health_path, timeout=timeout)  # type: ignore[operator]
            return r.status_code == 200
        except Exception:
            return False

    def stop(self, timeout: float = 10.0) -> None:
        """Request graceful shutdown, then force it if it takes longer than ``timeout``/2."""
        srv, th = self._server, self._thread
        if srv is not None:
            srv.should_exit = True
        if th is not None and th.is_alive():
            th.join(timeout / 2)
            if th.is_alive() and srv is not None:
                srv.force_exit = True
                th.join(timeout / 2)
            if th.is_alive():
                log.warning("uvicorn thread did not stop within %.1fs", timeout)
        self._close_socket()

    def _close_socket(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()

    def __enter__(self) -> ServerThread:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
