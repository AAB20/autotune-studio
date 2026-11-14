"""Authentication: every protected route rejects anonymous calls; token exchange; cookie + Bearer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autotune_studio.contracts import LaunchInfo
from autotune_studio.security import COOKIE_NAME

PROTECTED_ROUTES: list[tuple[str, str]] = [
    ("GET", "/api/env"),
    ("GET", "/api/hardware"),
    ("GET", "/api/resources"),
    ("GET", "/api/launch"),
    ("POST", "/api/datasets"),
    ("POST", "/api/datasets/upload"),
    ("GET", "/api/datasets"),
    ("GET", "/api/datasets/abc"),
    ("GET", "/api/datasets/abc/preview"),
    ("GET", "/api/datasets/abc/stats"),
    ("POST", "/api/datasets/abc/scan"),
    ("POST", "/api/datasets/abc/cancel"),
    ("GET", "/api/model"),
    ("PUT", "/api/model"),
    ("GET", "/api/models/suggested"),
    ("GET", "/api/training/config"),
    ("PUT", "/api/training/config"),
    ("POST", "/api/training/start"),
    ("POST", "/api/training/stop"),
    ("POST", "/api/training/pause"),
    ("POST", "/api/training/resume"),
    ("GET", "/api/training/status"),
    ("GET", "/api/training/metrics"),
    ("GET", "/api/training/validate"),
    ("GET", "/api/sessions"),
    ("GET", "/api/checkpoints"),
    ("POST", "/api/checkpoints"),
    ("POST", "/api/checkpoints/ck1/resume"),
    ("POST", "/api/export"),
]


@pytest.mark.parametrize(("method", "path"), PROTECTED_ROUTES)
def test_protected_routes_require_auth(anon: TestClient, method: str, path: str) -> None:
    # A syntactically valid body must not bypass auth (auth runs before body validation matters).
    r = anon.request(method, path, json={})
    assert r.status_code == 401, (method, path, r.text)
    body = r.json()
    assert body["code"] == "unauthorized"
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(("method", "path"), PROTECTED_ROUTES[:4])
def test_wrong_bearer_and_wrong_cookie_rejected(anon: TestClient, method: str, path: str) -> None:
    assert anon.request(method, path, headers={"Authorization": "Bearer nope-nope-nope-nope"}).status_code == 401
    anon.cookies.set(COOKIE_NAME, "f" * 64)
    assert anon.request(method, path).status_code == 401


def test_health_is_public_and_minimal(anon: TestClient) -> None:
    r = anon.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": r.json()["version"]}
    assert set(r.json()) == {"status", "version"}


def test_bearer_auth_works(client: TestClient) -> None:
    r = client.get("/api/env")
    assert r.status_code == 200
    assert "allowed_roots" in r.json()


def test_bad_token_401_then_rate_limited(anon: TestClient, studio) -> None:
    for _ in range(3):  # limiter fixture: 3 failures per minute
        r = anon.post("/api/auth/session", json={"token": "wrong-token-wrong-token"})
        assert r.status_code == 401
        assert r.json()["code"] == "unauthorized"
        assert COOKIE_NAME not in r.headers.get("set-cookie", "")
    r = anon.post("/api/auth/session", json={"token": "wrong-token-wrong-token"})
    assert r.status_code == 429
    assert r.json()["code"] == "rate_limited"
    assert int(r.headers["retry-after"]) >= 1
    # Even the right token is refused while blocked.
    r = anon.post("/api/auth/session", json={"token": studio.auth.token})
    assert r.status_code == 429


def test_session_body_validation(anon: TestClient) -> None:
    r = anon.post("/api/auth/session", json={})
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"


def test_good_token_sets_httponly_cookie_and_cookie_auth_works(app, studio) -> None:
    with TestClient(app) as c:
        assert c.get("/api/env").status_code == 401
        r = c.post("/api/auth/session", json={"token": studio.auth.token})
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        sc = r.headers["set-cookie"]
        assert sc.startswith(f"{COOKIE_NAME}=")
        low = sc.lower()
        assert "httponly" in low and "samesite=strict" in low and "path=/" in low
        assert "secure" not in low  # plain http test server
        assert studio.auth.token not in sc  # cookie holds an HMAC, never the token
        assert c.get("/api/env").status_code == 200  # cookie jar now authenticates


def test_cookie_secure_behind_https_proxy(anon: TestClient, studio) -> None:
    r = anon.post(
        "/api/auth/session", json={"token": studio.auth.token}, headers={"X-Forwarded-Proto": "https"}
    )
    assert r.status_code == 200
    assert "secure" in r.headers["set-cookie"].lower()


def test_launch_info_never_contains_token(client: TestClient, studio) -> None:
    assert client.get("/api/launch").json() is None
    token = studio.auth.token
    studio._launch_info = LaunchInfo(
        local_url="http://127.0.0.1:7860", public_url="https://x.trycloudflare.com", method="cloudflared",
        open_url=f"https://x.trycloudflare.com/#token={token}",
    )
    r = client.get("/api/launch")
    assert r.status_code == 200
    assert r.json()["open_url"] is None
    assert r.json()["public_url"] == "https://x.trycloudflare.com"
    assert token not in r.text
