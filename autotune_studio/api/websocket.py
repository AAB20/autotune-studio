"""``/ws`` live event stream.

Protocol
========
1. Client connects to ``/ws?since=<seq>`` (``since`` = highest ``seq`` it has processed, 0 on first
   connect). Credentials: the session cookie or ``Authorization: Bearer <token>``; the ``Origin``
   header must match ``Host`` or a launch-time public origin (``Studio.extra_origins``).
   Authentication happens **before** ``accept()``; failures are closed with code **1008**
   (policy violation), which browsers surface as a failed handshake.
2. Server sends ``{"seq":0,"type":"hello","ts":...,"data":{"last_seq":N,"gap":bool}}``.
   ``gap`` is true when events after ``since`` were already evicted from the replay buffer
   (the client should refetch REST state). If ``since > last_seq`` the server restarted: the
   whole buffer is replayed from 0 and the client sees ``last_seq`` lower than its own.
3. Server streams the replayed events and then live events, each one JSON ``Event``
   (``Event.model_dump(mode="json")``): ``{"seq", "type", "ts", "data"}``, ``seq`` strictly increasing.
4. Every ``ws_ping_interval`` seconds the server sends ``{"type":"ping","ts":...}`` (not an Event;
   the UI ignores it) so idle tunnels/proxies do not drop the connection.
5. Client messages (e.g. ``{"type":"ack"}``) are read and ignored.

Event payloads (``data``) by ``type``
=====================================
=============  ===============================================================================
``status``     :class:`~autotune_studio.contracts.TrainingStatus`
``metric``     :class:`~autotune_studio.contracts.MetricPoint`
``resource``   :class:`~autotune_studio.contracts.ResourceSnapshot`
``log``        :class:`~autotune_studio.contracts.LogEventData` (``level``, ``source``, ``message``, extra keys)
``dataset``    ``{"dataset_id": str, **StreamStats}`` (flat; the UI also accepts ``{"stats": ...}``)
``checkpoint`` :class:`~autotune_studio.contracts.CheckpointInfo`
``export``     :class:`~autotune_studio.contracts.ExportResult`
``error``      ``{"source": "training"|"dataset"|..., "message": str, ...context (job_id, dataset_id, code)}``
=============  ===============================================================================

Slow clients: each subscriber queue holds 1000 events and drops the oldest when full
(``EventBus``); such a client can miss events without a ``gap`` flag until it reconnects.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from ..contracts import Event
from ..security import authenticate_connection

log = logging.getLogger("autotune_studio.api.ws")

router = APIRouter()

POLICY_VIOLATION = 1008


def _parse_since(raw: str | None) -> int:
    try:
        return max(0, int(raw)) if raw is not None else 0
    except ValueError:
        return 0


@router.websocket("/ws")
async def events_ws(websocket: WebSocket) -> None:
    studio = websocket.app.state.studio
    settings = websocket.app.state.settings
    if not authenticate_connection(studio.auth, websocket, extra_origins=studio.extra_origins):
        await websocket.close(code=POLICY_VIOLATION)
        return
    await websocket.accept()

    bus = studio.events
    since = _parse_since(websocket.query_params.get("since"))
    last_seq = bus.last_seq
    if since > last_seq:  # server restarted (seq counter reset): replay everything we have
        since = 0
    _, gap = bus.replay(since)
    send_lock = asyncio.Lock()

    async def send(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_text(json.dumps(payload, separators=(",", ":")))

    hello = Event(seq=0, type="hello", data={"last_seq": last_seq, "gap": gap})
    try:
        await send(hello.model_dump(mode="json"))
    except Exception:  # noqa: BLE001 - client vanished during handshake
        return

    subscription = bus.subscribe(since_seq=since)

    async def pump_events() -> None:
        async for ev in subscription:
            await send(ev.model_dump(mode="json"))

    async def keepalive() -> None:
        while True:
            await asyncio.sleep(settings.ws_ping_interval)
            await send({"type": "ping", "ts": time.time()})

    async def read_client() -> None:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                return

    tasks = [
        asyncio.create_task(pump_events(), name="ws-events"),
        asyncio.create_task(keepalive(), name="ws-ping"),
        asyncio.create_task(read_client(), name="ws-recv"),
    ]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            exc = t.exception()
            if exc is not None and not isinstance(exc, (WebSocketDisconnect, RuntimeError, OSError)):
                log.warning("websocket task %s ended with %s", t.get_name(), type(exc).__name__)
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(BaseException):
                await t
        with contextlib.suppress(BaseException):
            await subscription.aclose()  # unregisters the bus subscriber
        if websocket.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await websocket.close()
