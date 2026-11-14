"""End-to-end over HTTP + WebSocket with the REAL hf training engine and a cached tiny Llama.

auth -> upload JSONL (unknown size) -> preview/stats -> PUT model -> PUT training config -> start
-> follow /ws until completed -> metrics -> checkpoints -> export (adapter + merged) -> reload from disk.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ..conftest import sha256_file
from .conftest import put_tiny_model, recv_until, tiny_training_config, wait_terminal, wait_until

pytestmark = pytest.mark.slow


def _is_final_status(ev: dict) -> bool:
    return ev["type"] == "status" and ev["data"].get("state") in ("completed", "failed", "stopped")


def test_full_http_flow_train_export_reload(studio, make_dataset, tiny_model_id) -> None:
    src = make_dataset("jsonl", "instruction", n=40)
    src_hash = sha256_file(src)

    with TestClient(studio.create_app(), raise_server_exceptions=False) as c:
        # ---- auth: protected route refused, then token -> HttpOnly cookie
        assert c.get("/api/env").status_code == 401
        r = c.post("/api/auth/session", json={"token": studio.auth.token})
        assert r.status_code == 200 and "httponly" in r.headers["set-cookie"].lower()
        assert c.get("/api/env").status_code == 200

        # ---- upload a JSONL file of unknown size (no row count anywhere)
        r = c.post("/api/datasets/upload",
                   files={"file": ("train.jsonl", src.read_bytes(), "application/x-ndjson")})
        assert r.status_code == 200, r.text
        info = r.json()
        did = info["dataset_id"]
        assert info["cardinality"] == {"rows": None, "exact": False, "method": "unknown"}
        assert info["detected_format"] == "instruction"
        stored = Path(info["spec"]["path"])
        stored_hash = sha256_file(stored)
        assert stored_hash == src_hash  # upload stored byte-identical

        pv = c.get(f"/api/datasets/{did}/preview", params={"n": 5}).json()
        assert len(pv["rows"]) == 5 and {"instruction", "output"} <= set(pv["columns"])
        st = c.get(f"/api/datasets/{did}/stats").json()
        assert st["malformed_rows"] == 0 and st["state"] in ("idle", "exhausted", "streaming")

        # ---- configure
        put_tiny_model(c, tiny_model_id, 64)
        r = c.put("/api/training/config", json=tiny_training_config(max_steps=8, save_steps=4))
        assert r.status_code == 200, r.text
        val = c.get("/api/training/validate", params={"dataset_id": did})
        assert val.status_code == 200 and val.json()["planned_steps"] == 8

        # ---- start + follow the live event stream until completion
        with c.websocket_connect(f"/ws?since={studio.events.last_seq}") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            r = c.post("/api/training/start", json={"dataset_id": did})
            assert r.status_code == 200, r.text
            assert r.json()["state"] in ("preparing", "running")
            events = recv_until(ws, _is_final_status)

        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        assert all(b > a for a, b in zip(seqs, seqs[1:]))
        types = {e["type"] for e in events}
        assert {"status", "metric", "checkpoint"} <= types, types
        assert events[-1]["data"]["state"] == "completed", events[-1]
        metric_steps = [e["data"]["step"] for e in events if e["type"] == "metric"]
        assert metric_steps and max(metric_steps) == 8
        ck_steps = sorted(e["data"]["step"] for e in events if e["type"] == "checkpoint")
        assert ck_steps == [4, 8]

        status = c.get("/api/training/status").json()
        assert status["state"] == "completed" and status["step"] == 8 and status["backend"] == "hf"

        metrics = c.get("/api/training/metrics").json()
        assert metrics and all(math.isfinite(m["loss"]) for m in metrics if m["loss"] is not None)

        ckpts = c.get("/api/checkpoints").json()
        assert [k["step"] for k in ckpts] == [4, 8] and ckpts[-1]["is_final"]
        assert all(k["data_cursor"] for k in ckpts)
        sessions = c.get("/api/sessions").json()
        assert sessions and sessions[-1]["state"] == "completed"

        # ---- export LoRA adapter + merged 16-bit
        r = c.post("/api/export", json={"format": "lora_adapter"})
        assert r.status_code == 200, r.text
        adapter = r.json()
        assert "adapter_model.safetensors" in adapter["files"] and "adapter_config.json" in adapter["files"]
        r = c.post("/api/export", json={"checkpoint_id": ckpts[-1]["checkpoint_id"], "format": "merged_16bit"})
        assert r.status_code == 200, r.text
        merged = r.json()
        assert "config.json" in merged["files"]
        assert any(f.endswith(".safetensors") for f in merged["files"])
        for res in (adapter, merged):
            for f in res["files"]:
                assert (Path(res["path"]) / f).is_file()

    # ---- dataset files never modified by any operation
    assert sha256_file(src) == src_hash
    assert sha256_file(stored) == stored_hash

    # ---- exported artifacts reload from disk
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_model_id)
    peft_model = PeftModel.from_pretrained(base, adapter["path"])
    ids = torch.tensor([[1, 5, 9, 13]])
    with torch.no_grad():
        out = peft_model(input_ids=ids)
    assert torch.isfinite(out.logits).all()

    m2 = AutoModelForCausalLM.from_pretrained(merged["path"])
    with torch.no_grad():
        out2 = m2(input_ids=ids)
    assert torch.isfinite(out2.logits).all()
    assert not any("lora" in n for n, _ in m2.named_parameters())


def test_stop_mid_run_then_resume_from_checkpoint(studio, auth_client, make_dataset, tiny_model_id) -> None:
    c = auth_client
    src = make_dataset("jsonl", "instruction", n=60)
    src_hash = sha256_file(src)
    did = c.post("/api/datasets", json={"source": "jsonl", "path": str(src)}).json()["dataset_id"]
    put_tiny_model(c, tiny_model_id, 64)
    total = 40
    assert c.put("/api/training/config",
                 json=tiny_training_config(max_steps=total, save_steps=1000)).status_code == 200

    assert c.post("/api/training/start", json={"dataset_id": did}).status_code == 200
    wait_until(lambda: c.get("/api/training/status").json()["step"] >= 2, timeout=120, interval=0.02)
    # While running: export of "latest" is refused, an explicit save-now works.
    assert c.post("/api/export", json={"format": "lora_adapter"}).status_code == 409
    r = c.post("/api/checkpoints")
    assert r.status_code == 200, r.text
    assert r.json()["step"] >= 2 and not r.json()["is_final"]
    wait_until(lambda: c.get("/api/training/status").json()["step"] >= 3, timeout=120, interval=0.02)
    r = c.post("/api/training/stop")
    assert r.status_code == 200 and r.json()["state"] in ("stopping", "stopped")
    st = wait_terminal(c)
    assert st["state"] == "stopped", st
    stop_step = st["step"]
    assert 3 <= stop_step < total

    ckpts = c.get("/api/checkpoints").json()
    assert ckpts, "stop must save a checkpoint"
    ck = ckpts[-1]
    assert ck["step"] == stop_step and ck["data_cursor"]
    assert (Path(ck["path"]) / "adapter_model.safetensors").is_file()
    first_job = st["job_id"]

    # Resume continues from the checkpoint step (not from 0) and runs to the original max_steps.
    seq0 = studio.events.last_seq
    r = c.post(f"/api/checkpoints/{ck['checkpoint_id']}/resume", json={"dataset_id": did})
    assert r.status_code == 200, r.text
    st2 = wait_terminal(c)
    assert st2["state"] == "completed", st2
    assert st2["job_id"] != first_job and st2["step"] == total

    new_events = [e for e in studio.events.replay(seq0)[0] if e.data.get("job_id", st2["job_id"]) == st2["job_id"]]
    resumed_steps = [e.data["step"] for e in new_events if e.type == "metric"]
    assert resumed_steps, "resumed job produced no metrics"
    assert min(resumed_steps) == stop_step + 1, resumed_steps[:5]
    assert max(resumed_steps) == total

    final = [k for k in c.get("/api/checkpoints").json() if k["job_id"] == st2["job_id"]]
    assert final and final[-1]["is_final"] and final[-1]["step"] == total
    assert sha256_file(src) == src_hash


def test_ws_disconnect_reconnect_no_duplicates_no_gaps(studio, auth_client, make_dataset, tiny_model_id) -> None:
    c = auth_client
    src = make_dataset("jsonl", "text", n=40)
    did = c.post("/api/datasets", json={"source": "jsonl", "path": str(src)}).json()["dataset_id"]
    put_tiny_model(c, tiny_model_id, 64)
    assert c.put("/api/training/config", json=tiny_training_config(max_steps=20, save_steps=10)).status_code == 200

    since0 = studio.events.last_seq
    with c.websocket_connect(f"/ws?since={since0}") as ws:
        assert ws.receive_json()["type"] == "hello"
        assert c.post("/api/training/start", json={"dataset_id": did}).status_code == 200
        first = recv_until(ws, lambda e: e["type"] == "metric" and e["data"]["step"] >= 2)
    last_seen = first[-1]["seq"]

    # Disconnected while training keeps producing events.
    wait_until(lambda: studio.events.last_seq > last_seen + 3, timeout=60)

    with c.websocket_connect(f"/ws?since={last_seen}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["data"]["gap"] is False
        second = recv_until(ws, _is_final_status)

    seqs = [e["seq"] for e in first + second]
    assert len(seqs) == len(set(seqs)), "duplicate events after reconnect"
    assert seqs == list(range(since0 + 1, seqs[-1] + 1)), "missing events after reconnect"
    assert second[0]["seq"] == last_seen + 1
    assert second[-1]["data"]["state"] == "completed"
