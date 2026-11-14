"""Failure recovery: bad inputs map to typed 4xx errors (never 500), training failures are reported as
``failed`` + ``error`` event, the server keeps working, and dataset files are never modified."""

from __future__ import annotations

from pathlib import Path

import pytest

from ..conftest import sha256_file
from .conftest import put_tiny_model, tiny_training_config, wait_terminal


def _corrupt_parquet(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Valid magic at both ends, garbage footer/metadata in between.
    path.write_bytes(b"PAR1" + b"\x00\x13garbage-not-a-thrift-footer" * 20 + b"\x10\x00\x00\x00PAR1")
    return path


# --------------------------------------------------------------------------- datasets
def test_corrupt_parquet_upload_is_422_dataset_malformed(auth_client, tmp_path) -> None:
    bad = _corrupt_parquet(tmp_path / "src" / "bad.parquet")
    r = auth_client.post("/api/datasets/upload", files={"file": ("bad.parquet", bad.read_bytes(), "application/octet-stream")})
    assert r.status_code == 422, r.text
    assert r.json()["code"] == "dataset_malformed"
    assert "Traceback" not in r.text
    # Rejected upload is cleaned up.
    uploads = tmp_path / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_corrupt_parquet_by_path_is_422_dataset_malformed(auth_client, tmp_path) -> None:
    bad = _corrupt_parquet(tmp_path / "data" / "bad.parquet")
    h = sha256_file(bad)
    r = auth_client.post("/api/datasets", json={"source": "parquet", "path": str(bad)})
    assert sha256_file(bad) == h
    assert r.status_code == 422, (r.status_code, r.text)
    assert r.json()["code"] == "dataset_malformed"


@pytest.mark.parametrize("fmt, content", [
    ("jsonl", b""),
    ("jsonl", b"\n\n  \n"),
    ("csv", b""),
    ("json", b"[]"),
])
def test_empty_file_is_dataset_empty(auth_client, tmp_path, fmt, content) -> None:
    p = tmp_path / "data" / f"empty.{fmt}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    r = auth_client.post("/api/datasets", json={"source": fmt, "path": str(p)})
    assert r.status_code == 422, (r.status_code, r.text)
    assert r.json()["code"] == "dataset_empty", r.json()
    assert p.read_bytes() == content


def test_malformed_jsonl_lines_are_counted_not_dropped(auth_client, tmp_path) -> None:
    p = tmp_path / "data" / "mixed.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    good = [f'{{"text": "row {i}"}}' for i in range(10)]
    lines = good[:3] + ["{not json"] + good[3:7] + ["[1, 2, 3]", '{"text": "unterminated'] + good[7:]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    h = sha256_file(p)

    r = auth_client.post("/api/datasets", json={"source": "jsonl", "path": str(p)})
    assert r.status_code == 200, r.text
    did = r.json()["dataset_id"]

    ds_stats = auth_client.app.state.studio.scan_dataset(did)  # synchronous facade scan
    assert ds_stats.state == "exhausted"
    assert ds_stats.rows_seen == 10
    assert ds_stats.malformed_rows == 3
    assert ds_stats.malformed_examples
    info = auth_client.get(f"/api/datasets/{did}").json()
    assert info["cardinality"] == {"rows": 10, "exact": True, "method": "scan"}
    assert auth_client.get(f"/api/datasets/{did}/stats").json()["malformed_rows"] == 3
    assert sha256_file(p) == h


def test_all_malformed_jsonl_is_rejected_not_silently_empty(auth_client, tmp_path) -> None:
    p = tmp_path / "data" / "allbad.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{bad\n{worse\nnope\n", encoding="utf-8")
    r = auth_client.post("/api/datasets", json={"source": "jsonl", "path": str(p)})
    assert r.status_code == 422, r.text
    assert r.json()["code"] in ("dataset_malformed", "dataset_empty")


def test_missing_and_forbidden_paths(auth_client, tmp_path, tmp_path_factory) -> None:
    r = auth_client.post("/api/datasets", json={"source": "csv", "path": str(tmp_path / "nope.csv")})
    assert r.status_code == 404 and r.json()["code"] == "not_found", r.text
    outside = tmp_path_factory.mktemp("outside") / "x.jsonl"  # sibling dir, not under an allowed root
    outside.write_text('{"text": "a"}\n', encoding="utf-8")
    r = auth_client.post("/api/datasets", json={"source": "jsonl", "path": str(outside)})
    assert r.status_code == 403 and r.json()["code"] == "forbidden_path", r.text
    r = auth_client.post("/api/datasets", json={"source": "jsonl", "path": str(Path(__file__).resolve())})
    assert r.status_code in (403, 422), r.text  # wrong extension and/or outside roots, never 500


# --------------------------------------------------------------------------- training failure
@pytest.mark.slow
@pytest.mark.parametrize("bad_model", ["repo", "empty_dir"])
def test_training_failure_reported_and_server_recovers(studio, auth_client, make_dataset, tmp_path,
                                                       tiny_model_id, bad_model) -> None:
    c = auth_client
    src = make_dataset("jsonl", "instruction", n=20)
    h = sha256_file(src)
    did = c.post("/api/datasets", json={"source": "jsonl", "path": str(src)}).json()["dataset_id"]
    if bad_model == "repo":
        name = "hf-internal-testing/this-model-does-not-exist-autotune"  # HF_HUB_OFFLINE=1 -> cannot load
    else:
        d = tmp_path / "models" / "empty-model"
        d.mkdir(parents=True)
        name = str(d)
    r = c.put("/api/model", json={"model_name": name, "max_seq_length": 64, "load_in_4bit": False})
    assert r.status_code == 200, r.text
    assert c.put("/api/training/config", json=tiny_training_config(max_steps=2, save_steps=2)).status_code == 200

    seq0 = studio.events.last_seq
    r = c.post("/api/training/start", json={"dataset_id": did})
    assert r.status_code == 200, r.text
    st = wait_terminal(c, timeout=120)
    assert st["state"] == "failed", st
    assert st["error"]
    evs = studio.events.replay(seq0)[0]
    errors = [e for e in evs if e.type == "error"]
    assert errors and errors[-1].data.get("source") == "training" and errors[-1].data.get("message")
    assert any(e.type == "status" and e.data.get("state") == "failed" for e in evs)

    # Server keeps working.
    assert c.get("/api/health").status_code == 200
    assert c.get(f"/api/datasets/{did}/preview", params={"n": 2}).status_code == 200
    sessions = c.get("/api/sessions").json()
    assert sessions[-1]["state"] == "failed"

    # A subsequent job with a valid model starts and completes.
    put_tiny_model(c, tiny_model_id)
    r = c.post("/api/training/start", json={"dataset_id": did})
    assert r.status_code == 200, r.text
    st2 = wait_terminal(c)
    assert st2["state"] == "completed" and st2["step"] == 2
    assert sha256_file(src) == h


def test_bad_ids_and_idle_controls_are_4xx(auth_client) -> None:
    c = auth_client
    r = c.post("/api/training/start", json={"dataset_id": "nope"})
    assert r.status_code == 422 and "model" in r.json()["message"]  # no model configured yet
    put_tiny_model(c, "hf-internal-testing/tiny-random-LlamaForCausalLM")  # validation only, never loaded
    assert c.post("/api/training/start", json={"dataset_id": "nope"}).status_code == 404
    assert c.post("/api/checkpoints/does-not-exist/resume", json={}).status_code == 404
    assert c.post("/api/export", json={"format": "lora_adapter"}).status_code == 404
    assert c.post("/api/checkpoints").status_code == 409  # nothing running
