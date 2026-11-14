"""Session token, cookie, header auth, WS origin and rate limiting."""

from __future__ import annotations

import hmac

import pytest
from starlette.websockets import WebSocket, WebSocketDisconnect

from autotune_studio.contracts import ErrorCode
from autotune_studio.errors import StudioError
from autotune_studio.security import authentication as authmod
from autotune_studio.security.authentication import (
    COOKIE_NAME,
    FailedAuthLimiter,
    RateLimited,
    SessionAuth,
    Unauthorized,
    authenticate_headers,
    check_ws_origin,
    cookie_params,
    exchange_token,
)


# ----------------------------------------------------------------- token
def test_generated_token_is_random_and_long():
    a, b = SessionAuth(), SessionAuth()
    assert a.token != b.token
    assert len(a.token) >= 43  # token_urlsafe(32)
    assert "***" in repr(a) and a.token not in repr(a)


def test_verify_token_accepts_exact_and_rejects_others():
    auth = SessionAuth("x" * 20 + "SECRET")
    assert auth.verify_token("x" * 20 + "SECRET")
    for bad in [
        None,
        "",
        "x" * 20 + "SECRE",
        "x" * 20 + "SECRET ",
        "X" * 20 + "SECRET",
        123,
        b"bytes",
        "x" * 10_000,
        "ünïcødé-tøken-that-is-long",
        "\ud800lone-surrogate-xxxxxxxxx",
    ]:
        assert auth.verify_token(bad) is False


def test_short_explicit_token_rejected():
    with pytest.raises(ValueError):
        SessionAuth("short")


def test_verify_uses_constant_time_compare(monkeypatch):
    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(authmod.hmac, "compare_digest", spy)
    auth = SessionAuth()
    assert auth.verify_token(auth.token)
    assert not auth.verify_token("wrong-token-value-xxxxxxxx")
    assert auth.verify_cookie(auth.cookie_value())
    assert len(calls) == 3


# ----------------------------------------------------------------- cookie
def test_cookie_is_hmac_not_token_and_verifies():
    auth = SessionAuth()
    cv = auth.cookie_value()
    assert cv != auth.token and auth.token not in cv
    assert len(cv) == 64 and all(c in "0123456789abcdef" for c in cv)
    assert auth.verify_cookie(cv)
    assert not auth.verify_cookie(auth.token)
    assert not auth.verify_cookie(cv[:-1] + ("0" if cv[-1] != "0" else "1"))
    assert not auth.verify_cookie(None) and not auth.verify_cookie("")


def test_cookie_key_is_per_instance():
    a = SessionAuth("same-token-for-both-instances")
    b = SessionAuth("same-token-for-both-instances")
    assert a.cookie_value() != b.cookie_value()
    assert not a.verify_cookie(b.cookie_value())


def test_cookie_params():
    assert cookie_params(True) == {"httponly": True, "samesite": "strict", "secure": True, "path": "/"}
    assert cookie_params(False)["secure"] is False


# ----------------------------------------------------------------- headers
@pytest.mark.parametrize("hdr_name", ["Authorization", "authorization", "AUTHORIZATION"])
def test_bearer_header_case_insensitive(hdr_name):
    auth = SessionAuth()
    assert authenticate_headers(auth, {hdr_name: f"Bearer {auth.token}"}, {})
    assert authenticate_headers(auth, {hdr_name: f"bearer   {auth.token}  "}, {})


def test_header_and_cookie_auth_matrix():
    auth = SessionAuth()
    ok_cookie = {COOKIE_NAME: auth.cookie_value()}
    assert authenticate_headers(auth, {}, ok_cookie)
    assert not authenticate_headers(auth, {}, {})
    assert not authenticate_headers(auth, {"Authorization": "Bearer nope"}, {})
    assert not authenticate_headers(auth, {"Authorization": f"Basic {auth.token}"}, {})
    assert not authenticate_headers(auth, {"Authorization": auth.token}, {})
    assert not authenticate_headers(auth, {}, {COOKIE_NAME: auth.token})  # raw token is not a cookie
    assert not authenticate_headers(auth, {}, {"other": auth.cookie_value()})
    # wrong bearer but valid cookie -> still authenticated
    assert authenticate_headers(auth, {"Authorization": "Bearer nope"}, ok_cookie)


def test_starlette_dependency_401_and_ok():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    auth = SessionAuth()
    app = fastapi.FastAPI()
    dep = authmod.require_auth_dependency(auth)

    @app.exception_handler(StudioError)
    async def _h(_req, exc: StudioError):
        return JSONResponse({"code": exc.code, "message": exc.message}, status_code=exc.http_status)

    @app.get("/api/x", dependencies=[fastapi.Depends(dep)])
    def x():
        return {"ok": True}

    c = TestClient(app)
    r = c.get("/api/x")
    assert r.status_code == 401 and r.json()["code"] == ErrorCode.UNAUTHORIZED
    assert c.get("/api/x", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/x", headers={"Authorization": f"Bearer {auth.token}"}).status_code == 200
    c.cookies.set(COOKIE_NAME, auth.cookie_value())
    assert c.get("/api/x").status_code == 200


def test_unauthorized_error_shape():
    e = Unauthorized()
    assert isinstance(e, StudioError) and e.http_status == 401 and e.code == ErrorCode.UNAUTHORIZED


# ----------------------------------------------------------------- websocket origin
@pytest.mark.parametrize(
    "origin,host,extra,expected",
    [
        (None, "localhost:8000", (), True),  # non-browser client (must still authenticate)
        ("", "localhost:8000", (), True),
        ("http://localhost:8000", "localhost:8000", (), True),
        ("https://localhost:8000", "localhost:8000", (), True),  # scheme-agnostic
        ("http://LOCALHOST:8000", "localhost:8000", (), True),
        ("https://abc.trycloudflare.com", "abc.trycloudflare.com", (), True),  # default 443 both sides
        ("https://abc.trycloudflare.com", "abc.trycloudflare.com:443", (), True),
        ("http://[::1]:8000", "[::1]:8000", (), True),
        ("http://evil.com", "localhost:8000", (), False),
        ("http://localhost:9999", "localhost:8000", (), False),
        ("http://localhost.evil.com:8000", "localhost:8000", (), False),
        ("http://evil.com", "evil.com.localhost:8000", (), False),
        ("null", "localhost:8000", (), False),
        ("file://", "localhost:8000", (), False),
        ("javascript:alert(1)", "localhost:8000", (), False),
        ("http://localhost:8000", None, (), False),
        ("http://localhost:8000", "localhost:notaport", (), False),
        ("http://localhost:8000", "user@localhost:8000", (), False),
        (
            "https://x-colab.googleusercontent.com",
            "127.0.0.1:8000",
            ("https://x-colab.googleusercontent.com",),
            True,
        ),
        ("https://x-colab.googleusercontent.com", "127.0.0.1:8000", ("x-colab.googleusercontent.com",), True),
        (
            "https://evil.googleusercontent.com",
            "127.0.0.1:8000",
            ("https://x-colab.googleusercontent.com",),
            False,
        ),
    ],
)
def test_check_ws_origin(origin, host, extra, expected):
    assert check_ws_origin(origin, host, extra) is expected


def test_ws_connection_rejected_cross_origin_and_unauthenticated():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    auth = SessionAuth()
    app = fastapi.FastAPI()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        if not authmod.authenticate_connection(auth, websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    c = TestClient(app)
    good = {"Authorization": f"Bearer {auth.token}"}
    with c.websocket_connect("/ws", headers=good) as s:
        assert s.receive_text() == "hello"
    for headers in [{}, {**good, "Origin": "http://evil.com"}]:
        with pytest.raises(WebSocketDisconnect) as ei, c.websocket_connect("/ws", headers=headers) as s:
            s.receive_text()
        assert ei.value.code == 1008
    # Same origin as Host (TestClient Host = "testserver") + cookie
    c.cookies.set(COOKIE_NAME, auth.cookie_value())
    with c.websocket_connect("/ws", headers={"Origin": "http://testserver"}) as s:
        assert s.receive_text() == "hello"


# ----------------------------------------------------------------- rate limiting
class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_limiter_blocks_after_max_failures_and_recovers():
    clk = FakeClock()
    lim = FailedAuthLimiter(max_per_minute=3, clock=clk)
    for _ in range(3):
        assert lim.allow("1.2.3.4")
        lim.record_failure("1.2.3.4")
    assert not lim.allow("1.2.3.4")
    assert 0 < lim.retry_after("1.2.3.4") <= 60
    assert lim.allow("5.6.7.8")  # other IPs unaffected
    clk.t += 60.0
    assert lim.allow("1.2.3.4")


def test_limiter_default_is_20_per_minute():
    clk = FakeClock()
    lim = FailedAuthLimiter(clock=clk)
    for _ in range(19):
        lim.record_failure("ip")
    assert lim.allow("ip")
    lim.record_failure("ip")
    assert not lim.allow("ip")


def test_limiter_reset_and_memory_bound():
    clk = FakeClock()
    lim = FailedAuthLimiter(max_per_minute=2, max_keys=5, clock=clk)
    lim.record_failure("a")
    lim.record_failure("a")
    assert not lim.allow("a")
    lim.reset("a")
    assert lim.allow("a")
    for i in range(50):
        clk.t += 0.01
        lim.record_failure(f"ip{i}")
    assert len(lim._fails) <= 5


def test_exchange_token_flow():
    clk = FakeClock()
    auth = SessionAuth()
    lim = FailedAuthLimiter(max_per_minute=2, clock=clk)
    assert exchange_token(auth, lim, "1.1.1.1", auth.token) == auth.cookie_value()
    for _ in range(2):
        with pytest.raises(Unauthorized):
            exchange_token(auth, lim, "1.1.1.1", "wrong")
    with pytest.raises(RateLimited) as ei:
        exchange_token(auth, lim, "1.1.1.1", auth.token)  # blocked even with the right token
    assert ei.value.http_status == 429 and ei.value.retry_after > 0
    with pytest.raises(Unauthorized):
        exchange_token(auth, lim, "2.2.2.2", {"token": auth.token})  # non-str body
    clk.t += 61
    assert exchange_token(auth, lim, "1.1.1.1", auth.token) == auth.cookie_value()
