"""ServerThread: start / health / stop, Jupyter-like running loop, port handling."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import socket
import threading
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from autotune_studio.launch.server import ServerThread, bind_socket, find_free_port


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def test_start_health_stop() -> None:
    sigint_before = signal.getsignal(signal.SIGINT)
    srv = ServerThread(_app(), port=0)
    srv.start()
    try:
        assert srv.wait_ready(15)
        assert srv.url.startswith("http://127.0.0.1:")
        r = httpx.get(srv.url + "/api/health", timeout=5)
        assert r.status_code == 200 and r.json() == {"status": "ok"}
        assert srv._thread is not None and srv._thread.daemon
    finally:
        srv.stop()
    assert not srv.is_running()
    assert signal.getsignal(signal.SIGINT) is sigint_before  # kernel's Ctrl-C handling untouched
    with pytest.raises(httpx.HTTPError):
        httpx.get(srv.url + "/api/health", timeout=1)


def test_start_from_running_event_loop() -> None:
    """Simulates a Jupyter cell: an asyncio loop is already running in the calling thread."""

    async def cell() -> int:
        srv = ServerThread(_app(), port=0)
        srv.start()
        try:
            assert srv.wait_ready(15)
            async with httpx.AsyncClient() as c:
                r = await c.get(srv.url + "/api/health", timeout=5)
            return r.status_code
        finally:
            srv.stop()

    assert asyncio.run(cell()) == 200


def test_context_manager_and_restart_on_same_port() -> None:
    port = find_free_port(0)
    with ServerThread(_app(), port=port) as srv:
        assert srv.wait_ready(15) and srv.port == port
    with ServerThread(_app(), port=port) as srv2:  # port released by stop()
        assert srv2.wait_ready(15)


def test_explicit_port_in_use_raises() -> None:
    blocker = bind_socket("127.0.0.1", [0])
    try:
        port = blocker.getsockname()[1]
        with pytest.raises(OSError):
            ServerThread(_app(), port=port).start()
    finally:
        blocker.close()


def test_preferred_port_busy_falls_forward() -> None:
    blocker = bind_socket("127.0.0.1", [0])
    try:
        busy = blocker.getsockname()[1]
        if busy >= 65535 - 20:
            pytest.skip("ephemeral port too high for scan test")
        srv = ServerThread(_app(), preferred_port=busy)
        srv.start()
        try:
            assert srv.port != busy and srv.wait_ready(15)
        finally:
            srv.stop()
        assert find_free_port(busy) != busy
    finally:
        blocker.close()


def test_find_free_port_returns_bindable_port() -> None:
    p = find_free_port(0)
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p))
    finally:
        s.close()


def test_wait_ready_false_without_health_route() -> None:
    srv = ServerThread(FastAPI(), port=0)
    srv.start()
    try:
        assert srv.wait_ready(1.0) is False  # 404 on /api/health
        assert srv.started
    finally:
        srv.stop()


def test_startup_failure_is_captured_not_raised() -> None:
    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        raise RuntimeError("startup failed")
        yield  # pragma: no cover

    app = FastAPI(lifespan=lifespan)
    srv = ServerThread(app, port=0, lifespan="on")
    srv.start()
    assert srv.wait_ready(5) is False
    srv.stop()
    assert srv.error is not None
    assert threading.main_thread().is_alive()
