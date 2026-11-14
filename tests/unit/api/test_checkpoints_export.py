"""Checkpoint listing / resume and export through the Studio facade (training modules faked)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from .conftest import wait_for


def test_list_checkpoints_empty_without_runs(client: TestClient) -> None:
    r = client.get("/api/checkpoints")
    assert r.status_code == 200 and r.json() == []


def test_list_and_resume_checkpoint(
    client: TestClient, dataset_id: str, fake_training, fake_checkpoint_modules
) -> None:
    engine = fake_training(steps=2)
    ckpts = client.get("/api/checkpoints").json()
    assert [c["checkpoint_id"] for c in ckpts] == ["job-x-step2"]

    r = client.post("/api/checkpoints/job-x-step2/resume", json={"dataset_id": dataset_id})
    assert r.status_code == 200, r.text
    assert wait_for(lambda: client.get("/api/training/status").json()["state"] == "completed")
    assert engine.resume_from is not None and engine.resume_from.checkpoint_id == "job-x-step2"
    # model config restored from the checkpoint state
    assert client.get("/api/model").json()["model_name"] == "sshleifer/tiny-gpt2"

    r = client.post("/api/checkpoints/nope/resume")
    assert r.status_code == 404
    r = client.post("/api/checkpoints/bad%2F..%2Fid/resume")
    assert r.status_code in (404, 405, 422)  # encoded slash never reaches the handler


def test_export_latest(client: TestClient, fake_checkpoint_modules, studio) -> None:
    client.put("/api/model", json={"model_name": "sshleifer/tiny-gpt2"})
    seq = studio.events.last_seq
    r = client.post("/api/export", json={"format": "lora_adapter"})
    assert r.status_code == 200, r.text
    assert r.json()["format"] == "lora_adapter" and r.json()["files"] == ["adapter_model.safetensors"]
    _req, ck, _model_cfg = fake_checkpoint_modules["export"]
    assert ck.checkpoint_id == "job-x-step2"
    events, _ = studio.events.replay(seq)
    assert any(e.type == "export" and e.data["path"] == r.json()["path"] for e in events)


def test_export_output_dir_outside_roots_forbidden(client: TestClient, fake_checkpoint_modules, tmp_path: Path) -> None:
    client.put("/api/model", json={"model_name": "sshleifer/tiny-gpt2"})
    outside = str(Path(tmp_path.anchor) / "definitely-not-allowed-export")
    r = client.post("/api/export", json={"output_dir": outside})
    assert r.status_code == 403 and r.json()["code"] == "forbidden_path"


def test_export_validation(client: TestClient) -> None:
    r = client.post("/api/export", json={"format": "gguf"})
    assert r.status_code == 422
    assert r.json()["detail"]["errors"][0]["loc"][-1] == "format"
