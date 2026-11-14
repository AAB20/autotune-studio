"""/ws: auth before accept, hello frame, replay with since, live events, origin check, keepalive, cleanup."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from autotune_studio.api.app import create_app
from autotune_studio.core.config import StudioSettings
from autotune_studio.core.events import EventBus

from .conftest import wait_for


def _auth(studio) -> dict[str, str]:
    return {"Authorization": f"Bearer {studio.auth.token}"}


def _recv_non_ping(ws) -> dict:
    while True:
        msg = ws.receive_json()
        if msg.get("type") not in ("ping", "resource"):
            return msg


def test_unauthenticated_ws_rejected(anon: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as ei, anon.websocket_connect("/ws"):
        pass
    assert ei.value.code == 1008


def test_ws_wrong_token_rejected(anon: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as ei:
        with anon.websocket_connect("/ws", headers={"Authorization": "Bearer wrong-wrong-wrong-wrong"}):
            pass
    assert ei.value.code == 1008


def test_cross_origin_rejected_even_with_token(anon: TestClient, studio) -> None:
    with pytest.raises(WebSocketDisconnect) as ei:
        with anon.websocket_connect("/ws", headers={**_auth(studio), "Origin": "https://evil.example"}):
            pass
    assert ei.value.code == 1008


def test_same_origin_and_dynamic_extra_origin_accepted(anon: TestClient, studio) -> None:
    with anon.websocket_connect("/ws", headers={**_auth(studio), "Origin": "http://testserver"}) as ws:
        assert ws.receive_json()["type"] == "hello"
    # Tunnel URL learned after the app was built (Studio.launch) is honoured.
    studio._extra_origins.add("https://abc.trycloudflare.com")
    with anon.websocket_connect("/ws", headers={**_auth(studio), "Origin": "https://abc.trycloudflare.com"}) as ws:
        assert ws.receive_json()["type"] == "hello"


def test_cookie_auth_ws(app, studio) -> None:
    with TestClient(app) as c:
        assert c.post("/api/auth/session", json={"token": studio.auth.token}).status_code == 200
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"


def test_hello_then_live_events(anon: TestClient, studio) -> None:
    base = studio.events.last_seq
    with anon.websocket_connect(f"/ws?since={base}", headers=_auth(studio)) as ws:
        hello = ws.receive_json()
        assert hello == {"seq": 0, "type": "hello", "ts": hello["ts"], "data": {"last_seq": base, "gap": False}}
        studio.events.publish("log", {"level": "info", "source": "test", "message": "live one"})
        ev = _recv_non_ping(ws)
        assert ev["type"] == "log" and ev["seq"] == base + 1 and ev["data"]["message"] == "live one"
        assert set(ev) == {"seq", "type", "ts", "data"}


def test_reconnect_since_replays_only_newer(anon: TestClient, studio) -> None:
    base = studio.events.last_seq
    for i in range(3):
        studio.events.publish("log", {"message": f"m{i}"})
    with anon.websocket_connect(f"/ws?since={base + 2}", headers=_auth(studio)) as ws:
        hello = ws.receive_json()
        assert hello["data"] == {"last_seq": base + 3, "gap": False}
        ev = _recv_non_ping(ws)
        assert ev["seq"] == base + 3 and ev["data"]["message"] == "m2"
        studio.events.publish("log", {"message": "m3"})
        assert _recv_non_ping(ws)["seq"] == base + 4


def test_since_beyond_last_seq_replays_everything(anon: TestClient, studio) -> None:
    base = studio.events.last_seq
    studio.events.publish("log", {"message": "a"})
    studio.events.publish("log", {"message": "b"})
    with anon.websocket_connect("/ws?since=999999", headers=_auth(studio)) as ws:
        assert ws.receive_json()["data"]["last_seq"] == base + 2
        seqs = []
        while len(seqs) < base + 2:
            seqs.append(_recv_non_ping(ws)["seq"])
        assert seqs == list(range(1, base + 3))


def test_gap_flag_when_events_evicted(anon: TestClient, studio) -> None:
    studio.events = EventBus(replay_size=2)
    for i in range(5):
        studio.events.publish("log", {"message": str(i)})
    with anon.websocket_connect("/ws?since=1", headers=_auth(studio)) as ws:
        assert ws.receive_json()["data"] == {"last_seq": 5, "gap": True}
        assert [_recv_non_ping(ws)["seq"] for _ in range(2)] == [4, 5]


def test_invalid_since_treated_as_zero(anon: TestClient, studio) -> None:
    with anon.websocket_connect("/ws?since=abc", headers=_auth(studio)) as ws:
        assert ws.receive_json()["type"] == "hello"


def test_keepalive_ping(studio, limiter, web_dir) -> None:
    app = create_app(studio, StudioSettings(ws_ping_interval=0.05), web_dir=web_dir, auth_limiter=limiter)
    with TestClient(app) as c, c.websocket_connect("/ws", headers=_auth(studio)) as ws:
        assert ws.receive_json()["type"] == "hello"
        msg = ws.receive_json()
        assert msg["type"] == "ping"
        ws.send_json({"type": "ack"})  # ignored by the server
        assert ws.receive_json()["type"] == "ping"


def test_disconnect_unsubscribes(anon: TestClient, studio) -> None:
    with anon.websocket_connect("/ws", headers=_auth(studio)) as ws:
        ws.receive_json()
        studio.events.publish("log", {"message": "x"})
        _recv_non_ping(ws)
        assert len(studio.events._subscribers) == 1
    assert wait_for(lambda: len(studio.events._subscribers) == 0)
