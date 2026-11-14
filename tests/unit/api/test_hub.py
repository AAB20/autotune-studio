"""Hugging Face credentials (memory-only, write-only token) and push-to-Hub export (ADR-0007). Hub faked."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

TOKEN = "hf_FAKEtestTOKENvalue1234567890abcdef"


class FakeHfApi:
    calls: list[tuple[str, dict[str, Any]]] = []
    role = "write"
    fail_whoami: Exception | None = None

    def whoami(self, token=None, **_):
        FakeHfApi.calls.append(("whoami", {"token": token}))
        if FakeHfApi.fail_whoami:
            raise FakeHfApi.fail_whoami
        return {"name": "alice", "orgs": [{"name": "acme"}], "auth": {"accessToken": {"role": FakeHfApi.role}}}

    def create_repo(self, repo_id, **kw):
        FakeHfApi.calls.append(("create_repo", {"repo_id": repo_id, **kw}))

    def upload_folder(self, **kw):
        FakeHfApi.calls.append(("upload_folder", kw))


@pytest.fixture(autouse=True)
def fake_hub(monkeypatch):
    import huggingface_hub

    FakeHfApi.calls = []
    FakeHfApi.role = "write"
    FakeHfApi.fail_whoami = None
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return FakeHfApi


def test_requires_auth(anon: TestClient) -> None:
    assert anon.get("/api/hf/credentials").status_code == 401
    assert anon.put("/api/hf/credentials", json={"username": "alice", "token": TOKEN}).status_code == 401


def test_set_status_clear_never_returns_token(client: TestClient, studio, caplog) -> None:
    assert client.get("/api/hf/credentials").json()["configured"] is False
    with caplog.at_level(logging.DEBUG):
        r = client.put("/api/hf/credentials", json={"username": "alice", "token": TOKEN})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["configured"] and body["verified"] and body["username"] == "alice" and body["can_write"] is True
    assert TOKEN not in r.text
    assert TOKEN not in client.get("/api/hf/credentials").text
    assert TOKEN not in caplog.text
    assert all(TOKEN not in str(e.data) for e in studio.events.replay(0)[0])
    assert TOKEN not in repr(studio._hf_credentials)
    assert client.delete("/api/hf/credentials").json()["configured"] is False


def test_org_username_accepted_and_wrong_user_rejected(client: TestClient) -> None:
    assert client.put("/api/hf/credentials", json={"username": "acme", "token": TOKEN}).json()["username"] == "acme"
    r = client.put("/api/hf/credentials", json={"username": "mallory", "token": TOKEN})
    assert r.status_code == 422 and r.json()["code"] == "validation_error"
    assert "alice" in r.json()["message"] and TOKEN not in r.text


def test_bad_token_maps_to_hub_error_redacted(client: TestClient, fake_hub) -> None:
    class Resp:
        status_code = 401

    exc = RuntimeError(f"401 Unauthorized for token {TOKEN}")
    exc.response = Resp()  # type: ignore[attr-defined]
    fake_hub.fail_whoami = exc
    r = client.put("/api/hf/credentials", json={"username": "alice", "token": TOKEN})
    assert r.status_code == 502 and r.json()["code"] == "hub_error"
    assert TOKEN not in r.text
    assert client.get("/api/hf/credentials").json()["configured"] is False


def test_validation_error_does_not_echo_token(client: TestClient) -> None:
    r = client.put("/api/hf/credentials", json={"username": "bad name!", "token": TOKEN})
    assert r.status_code == 422 and TOKEN not in r.text


def test_push_requires_credentials(client: TestClient, fake_checkpoint_modules) -> None:
    client.put("/api/model", json={"model_name": "sshleifer/tiny-gpt2"})
    r = client.post("/api/export", json={"push_to_hub": True})
    assert r.status_code == 422 and "Hugging Face" in r.json()["message"]


def test_export_push_uploads_to_user_repo(client: TestClient, fake_checkpoint_modules, fake_hub) -> None:
    client.put("/api/model", json={"model_name": "sshleifer/tiny-gpt2"})
    client.put("/api/hf/credentials", json={"username": "alice", "token": TOKEN})
    r = client.post("/api/export", json={"push_to_hub": True, "hub_repo_name": "my-adapter", "hub_private": True})
    assert r.status_code == 200, r.text
    assert r.json()["hub_repo_id"] == "alice/my-adapter"
    assert r.json()["hub_url"] == "https://huggingface.co/alice/my-adapter"
    assert TOKEN not in r.text
    names = [c[0] for c in fake_hub.calls]
    assert names[-2:] == ["create_repo", "upload_folder"]
    create, upload = fake_hub.calls[-2][1], fake_hub.calls[-1][1]
    assert create["repo_id"] == "alice/my-adapter" and create["private"] is True and create["exist_ok"] is True
    assert upload["repo_id"] == "alice/my-adapter" and "autotune_state.json" in upload["ignore_patterns"]


def test_default_repo_name_and_invalid_name(client: TestClient, fake_checkpoint_modules) -> None:
    client.put("/api/model", json={"model_name": "sshleifer/tiny-gpt2"})
    client.put("/api/hf/credentials", json={"username": "alice", "token": TOKEN})
    r = client.post("/api/export", json={"push_to_hub": True})
    assert r.status_code == 200 and r.json()["hub_repo_id"].startswith("alice/autotune-job-x-step2-lora")
    r = client.post("/api/export", json={"push_to_hub": True, "hub_repo_name": "../evil"})
    assert r.status_code == 422


def test_read_only_token_cannot_push(client: TestClient, fake_checkpoint_modules, fake_hub) -> None:
    fake_hub.role = "read"
    client.put("/api/model", json={"model_name": "sshleifer/tiny-gpt2"})
    st = client.put("/api/hf/credentials", json={"username": "alice", "token": TOKEN}).json()
    assert st["can_write"] is False and "read-only" in st["message"]
    r = client.post("/api/export", json={"push_to_hub": True})
    assert r.status_code == 502 and r.json()["code"] == "hub_error"
