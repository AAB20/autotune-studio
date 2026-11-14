"""Model + training config, lifecycle control with a fake engine, metrics, sessions, validation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autotune_studio.contracts import TrainingConfig

from .conftest import wait_for

MODEL = {"model_name": "sshleifer/tiny-gpt2", "max_seq_length": 128, "load_in_4bit": False}


def _state(client: TestClient) -> str:
    return client.get("/api/training/status").json()["state"]


def test_model_get_404_then_put_get(client: TestClient) -> None:
    r = client.get("/api/model")
    assert r.status_code == 404 and r.json()["code"] == "not_found"
    r = client.put("/api/model", json=MODEL)
    assert r.status_code == 200, r.text
    assert r.json()["model_name"] == "sshleifer/tiny-gpt2"
    assert r.json()["trust_remote_code"] is False
    assert client.get("/api/model").json()["max_seq_length"] == 128


def test_model_invalid_names_422(client: TestClient) -> None:
    r = client.put("/api/model", json={"model_name": "https://evil.example/model"})
    assert r.status_code == 422 and r.json()["code"] == "validation_error"
    assert r.json()["detail"]["errors"][0]["loc"][-1] == "model_name"
    r = client.put("/api/model", json={"model_name": "bad name with spaces"})
    assert r.status_code == 422
    assert r.json()["detail"] == {"field": "model_name"}
    r = client.put("/api/model", json={"model_name": "gpt2", "trust_remote_code": True})
    assert r.status_code == 422


def test_suggested_models(client: TestClient) -> None:
    r = client.get("/api/models/suggested")
    assert r.status_code == 200
    assert isinstance(r.json(), list) and all(isinstance(m, str) for m in r.json()) and r.json()


def test_training_config_roundtrip_and_validation(client: TestClient, studio) -> None:
    default = client.get("/api/training/config").json()
    assert default == studio.training_config.model_dump(mode="json")
    TrainingConfig.model_validate(default)
    r = client.put("/api/training/config", json={**default, "max_steps": 10, "method": "lora"})
    assert r.status_code == 200 and r.json()["max_steps"] == 10
    assert client.get("/api/training/config").json()["method"] == "lora"

    r = client.put("/api/training/config", json={**default, "learning_rate": -1, "lora_r": 0})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation_error"
    fields = {e["loc"][-1] for e in body["detail"]["errors"]}
    assert {"learning_rate", "lora_r"} <= fields
    assert all(set(e) == {"loc", "msg", "type"} for e in body["detail"]["errors"])
    r = client.put("/api/training/config", json={"unknown_field": 1})
    assert r.status_code == 422


def test_validate_requires_model(client: TestClient, dataset_id: str, fake_training) -> None:
    r = client.get("/api/training/validate", params={"dataset_id": dataset_id})
    assert r.status_code == 422 and r.json()["code"] == "validation_error"
    client.put("/api/model", json=MODEL)
    r = client.get("/api/training/validate", params={"dataset_id": dataset_id})
    assert r.status_code == 200
    assert r.json() == {"planned_steps": 5, "warnings": ["fake warning"]}


def test_start_runs_to_completion_with_metrics(client: TestClient, dataset_id: str, fake_training, studio) -> None:
    engine = fake_training(steps=3)
    client.put("/api/model", json=MODEL)
    r = client.post("/api/training/start", json={"dataset_id": dataset_id})
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["state"] in ("preparing", "running", "completed")
    assert st["job_id"]
    assert wait_for(lambda: _state(client) == "completed")
    assert engine.ran.is_set()
    assert client.get("/api/training/metrics").status_code == 200
    assert client.get("/api/training/metrics", params={"since_step": -1}).status_code == 422
    sessions = client.get("/api/sessions").json()
    assert sessions and sessions[0]["job_id"] == st["job_id"] and sessions[0]["state"] == "completed"


def test_metrics_endpoint_returns_engine_points(client: TestClient, dataset_id: str, fake_training) -> None:
    fake_training(steps=3)
    client.put("/api/model", json=MODEL)
    assert client.post("/api/training/start", json={"dataset_id": dataset_id}).status_code == 200
    assert wait_for(lambda: _state(client) == "completed")
    assert [m["step"] for m in client.get("/api/training/metrics").json()] == [1, 2, 3]
    assert [m["step"] for m in client.get("/api/training/metrics", params={"since_step": 2}).json()] == [3]


def test_start_without_body_uses_active_dataset(client: TestClient, dataset_id: str, fake_training) -> None:
    client.put("/api/model", json=MODEL)
    r = client.post("/api/training/start")
    assert r.status_code == 200, r.text
    assert wait_for(lambda: _state(client) == "completed")


def test_second_start_conflict_and_controls(client: TestClient, dataset_id: str, fake_training) -> None:
    fake_training(steps=100_000, delay=0.002)
    client.put("/api/model", json=MODEL)
    assert client.post("/api/training/start", json={"dataset_id": dataset_id}).status_code == 200
    assert wait_for(lambda: _state(client) == "running")

    r = client.post("/api/training/start", json={"dataset_id": dataset_id})
    assert r.status_code == 409 and r.json()["code"] == "conflict"

    r = client.post("/api/training/pause")
    assert r.status_code == 200 and r.json()["state"] == "paused"
    assert client.post("/api/training/pause").status_code == 409
    r = client.post("/api/training/resume")
    assert r.status_code == 200 and r.json()["state"] == "running"

    ck = client.post("/api/checkpoints")
    assert ck.status_code == 200, ck.text
    assert ck.json()["checkpoint_id"].startswith(r.json()["job_id"])

    r = client.post("/api/training/stop")
    assert r.status_code == 200
    assert r.json()["state"] in ("stopping", "stopped")
    assert wait_for(lambda: _state(client) == "stopped")
    assert client.post("/api/training/resume").status_code == 409
    assert client.get("/api/sessions").json()[0]["state"] == "stopped"


def test_save_checkpoint_without_job_conflict(client: TestClient) -> None:
    r = client.post("/api/checkpoints")
    assert r.status_code == 409 and r.json()["code"] == "conflict"


def test_start_without_dataset_404(client: TestClient, fake_training) -> None:
    client.put("/api/model", json=MODEL)
    r = client.post("/api/training/start", json={})
    assert r.status_code == 404 and r.json()["code"] == "not_found"
