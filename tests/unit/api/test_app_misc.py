"""Error mapping, security headers, SPA serving, system routes."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from autotune_studio.api.app import create_app
from autotune_studio.contracts import TrainingConfig
from autotune_studio.errors import Conflict


def test_unexpected_exception_is_500_without_traceback(client: TestClient, studio, monkeypatch) -> None:
    def boom():
        raise RuntimeError("secret-internal-detail /very/private/path")

    monkeypatch.setattr(studio, "training_status", boom)
    r = client.get("/api/training/status")
    assert r.status_code == 500
    body = r.json()
    assert body["code"] == "internal_error"
    assert set(body) == {"code", "message", "detail"}
    assert "secret-internal-detail" not in r.text and "Traceback" not in r.text and "RuntimeError" not in r.text
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_studio_error_mapping_with_detail(client: TestClient, studio, monkeypatch) -> None:
    def busy():
        raise Conflict("busy now", {"job_id": "j1"})

    monkeypatch.setattr(studio, "pause_training", busy)
    r = client.post("/api/training/pause")
    assert r.status_code == 409
    assert r.json() == {"code": "conflict", "message": "busy now", "detail": {"job_id": "j1"}}


def test_pydantic_error_inside_studio_maps_to_422(client: TestClient, studio, monkeypatch) -> None:
    def bad(cfg):
        return TrainingConfig.model_validate({"learning_rate": -5})

    monkeypatch.setattr(studio, "set_training_config", bad)
    r = client.put("/api/training/config", json={})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation_error"
    assert body["detail"]["errors"][0]["loc"] == ["learning_rate"]
    assert "input" not in body["detail"]["errors"][0]


def test_security_headers(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"
    assert "x-frame-options" not in r.headers  # framing allowed for notebook iframes (documented)
    r = client.get("/")
    assert "frame-ancestors" not in r.headers["content-security-policy"]
    assert "script-src 'self'" in r.headers["content-security-policy"]


def test_spa_index_assets_and_fallback(anon: TestClient) -> None:
    r = anon.get("/")
    assert r.status_code == 200 and "<title>SPA</title>" in r.text
    assert r.headers["cache-control"] == "no-cache"
    r = anon.get("/assets/app-123.js")
    assert r.status_code == 200 and r.text == "console.log(1)"
    assert "immutable" in r.headers["cache-control"]
    r = anon.get("/training/live")  # client-side route -> index.html
    assert r.status_code == 200 and "<title>SPA</title>" in r.text
    assert anon.get("/assets/missing.js").status_code == 404
    r = anon.get("/api/does-not-exist")
    assert r.status_code == 404 and r.json()["code"] == "not_found"
    assert anon.head("/").status_code == 200


def test_spa_traversal_blocked(anon: TestClient, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
    for p in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/..%2f..%2fsecret.txt", "/..%5csecret.txt"):
        r = anon.get(p)
        assert "TOPSECRET" not in r.text, p


def test_ui_not_built_page(studio, limiter, tmp_path: Path, settings) -> None:
    app = create_app(studio, settings, web_dir=tmp_path / "no-web", auth_limiter=limiter)
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert "npm run build" in r.text
        assert c.get("/some/route").status_code == 200


def test_default_web_dir_and_env_settings(studio, monkeypatch) -> None:
    monkeypatch.setenv("AUTOTUNE_WS_PING_INTERVAL", "7")
    app = studio.create_app()  # Studio facade path: create_app(self), settings from env
    assert app.state.settings.ws_ping_interval == 7.0
    with TestClient(app) as c:
        assert c.get("/").status_code == 200
        assert c.get("/api/health").status_code == 200


def test_system_routes(client: TestClient, studio) -> None:
    env = client.get("/api/env").json()
    assert env["workspace_dir"] == str(studio.workspace)
    hw = client.get("/api/hardware").json()
    assert hw["cpu_count"] == 4  # stubbed in fixture
    res = client.get("/api/resources").json()
    assert res["ram_total_bytes"] > 0


def test_hardware_refresh_real_detection(client: TestClient) -> None:
    r = client.get("/api/hardware", params={"refresh": "true"})
    assert r.status_code == 200
    assert r.json()["cpu_count"] >= 1


def test_no_openapi_docs_exposed(anon: TestClient) -> None:
    assert anon.get("/openapi.json").headers["content-type"].startswith("text/html")  # SPA, not schema
    assert "openapi" not in anon.get("/docs").text.lower()
