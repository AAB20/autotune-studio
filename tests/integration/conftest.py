"""Integration-test helpers: offline HF hub, cached tiny model, HTTP/WS polling utilities."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import pytest

# The real training engine must never reach the network in this suite.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TERMINAL = {"completed", "stopped", "failed"}


@pytest.fixture(scope="session")
def tiny_model_id() -> str:
    """The cached tiny Llama repo id; skips (with a how-to) if it is not in the local HF cache."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(TINY_MODEL, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{TINY_MODEL} not in the HF cache ({type(exc).__name__}); "
                    f"run once online: python -c \"from huggingface_hub import snapshot_download;"
                    f"snapshot_download('{TINY_MODEL}')\"")
    return TINY_MODEL


def wait_until(pred: Callable[[], Any], timeout: float = 120.0, interval: float = 0.05) -> Any:
    """Poll ``pred`` until truthy; returns its value or raises AssertionError on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(interval)
    v = pred()
    assert v, f"condition not met within {timeout}s"
    return v


def wait_terminal(client: Any, timeout: float = 180.0) -> dict[str, Any]:
    """Poll ``GET /api/training/status`` until the job reaches a terminal state."""

    def done() -> dict[str, Any] | None:
        st = client.get("/api/training/status").json()
        return st if st["state"] in TERMINAL else None

    return wait_until(done, timeout=timeout, interval=0.1)


def tiny_training_config(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "method": "lora", "max_steps": 8, "save_steps": 4, "per_device_batch_size": 2,
        "gradient_accumulation_steps": 1, "warmup_steps": 0, "logging_steps": 1, "learning_rate": 1e-3,
    }
    cfg.update(overrides)
    return cfg


def put_tiny_model(client: Any, model_id: str, max_seq_length: int = 64) -> None:
    r = client.put("/api/model", json={"model_name": model_id, "max_seq_length": max_seq_length,
                                       "load_in_4bit": False})
    assert r.status_code == 200, r.text


def recv_until(ws: Any, pred: Callable[[dict[str, Any]], bool], limit: int = 100_000) -> list[dict[str, Any]]:
    """Read WS frames (skipping keepalive pings) until ``pred(event)``; returns the events read."""
    out: list[dict[str, Any]] = []
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == "ping":
            continue
        out.append(msg)
        if pred(msg):
            return out
    raise AssertionError("predicate never satisfied on the WebSocket stream")
