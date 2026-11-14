"""Fixtures for API tests: a real Studio in tmp_path, a TestClient, fake training engine/modules."""

from __future__ import annotations

import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import autotune_studio.training as training_pkg
from autotune_studio.api.app import create_app
from autotune_studio.contracts import CheckpointInfo, ExportResult, HardwareInfo
from autotune_studio.core.config import StudioSettings
from autotune_studio.interfaces import TrainingResult
from autotune_studio.security import FailedAuthLimiter, redaction
from autotune_studio.studio import Studio


# --------------------------------------------------------------------------- fake engine
class FakeEngine:
    """Deterministic engine honouring stop / pause / save_now; ``steps`` metric points."""

    backend = "hf"

    def __init__(self, steps: int = 3, delay: float = 0.0) -> None:
        self.steps = steps
        self.delay = delay
        self.resume_from: CheckpointInfo | None = None
        self.ran = threading.Event()

    def run(self, job_id, model_cfg, train_cfg, dataset, events, control, resume_from=None) -> TrainingResult:
        self.resume_from = resume_from
        self.ran.set()
        out = Path(train_cfg.output_dir or ".")
        events.publish("status", {"state": "running", "backend": self.backend, "message": "running"})
        ckpts: list[CheckpointInfo] = []
        step = 0
        stopped = False
        while step < self.steps:
            if control.stop.is_set():
                stopped = True
                break
            if control.save_now.is_set():
                control.save_now.clear()
                ck = CheckpointInfo(
                    checkpoint_id=f"{job_id}-step{step}", path=str(out / f"checkpoint-{step}"), step=step,
                    created_at=time.time(), size_bytes=1, job_id=job_id,
                )
                ckpts.append(ck)
                events.publish("checkpoint", ck.model_dump(mode="json"))
            if control.pause.is_set():
                time.sleep(0.005)
                continue
            step += 1
            events.publish("metric", {"step": step, "loss": 1.0 / step, "learning_rate": 1e-4, "timestamp": time.time()})
            if self.delay:
                time.sleep(self.delay)
        return TrainingResult(job_id=job_id, final_step=step, final_loss=1.0 / max(step, 1),
                              stopped_early=stopped, checkpoints=ckpts, output_dir=str(out))


FAKE_HW = HardwareInfo(
    cpu_count=4, ram_total_bytes=8 * 1024**3, ram_available_bytes=4 * 1024**3, disk_path=".",
    disk_total_bytes=100 * 1024**3, disk_free_bytes=50 * 1024**3, cuda_available=False,
)


# --------------------------------------------------------------------------- core fixtures
@pytest.fixture(autouse=True)
def _restore_redaction() -> Iterator[None]:
    before = set(redaction._secrets)
    yield
    for s in set(redaction._secrets) - before:
        redaction.unregister_secret(s)


@pytest.fixture
def studio(tmp_path: Path) -> Iterator[Studio]:
    ws = tmp_path / "ws"
    s = Studio(workspace_dir=ws, allowed_roots=[str(tmp_path)], resource_interval=0.05)
    s._hardware = FAKE_HW  # skip the ~3 s real detection; /api/hardware?refresh=true still runs it
    yield s
    try:
        if s.lifecycle.is_active:
            s.lifecycle.stop(wait=True, timeout=10)
    finally:
        s.sampler.stop()


@pytest.fixture
def settings() -> StudioSettings:
    return StudioSettings(ws_ping_interval=30.0, upload_max_bytes=1024 * 1024)


@pytest.fixture
def limiter() -> FailedAuthLimiter:
    return FailedAuthLimiter(max_per_minute=3)


@pytest.fixture
def web_dir(tmp_path: Path) -> Path:
    d = tmp_path / "web"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<!doctype html><title>SPA</title><div id=root></div>", encoding="utf-8")
    (d / "assets" / "app-123.js").write_text("console.log(1)", encoding="utf-8")
    return d


@pytest.fixture
def app(studio: Studio, settings: StudioSettings, limiter: FailedAuthLimiter, web_dir: Path):
    return create_app(studio, settings, web_dir=web_dir, auth_limiter=limiter)


@pytest.fixture
def anon(app) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def client(app, studio: Studio) -> Iterator[TestClient]:
    """Authenticated with ``Authorization: Bearer <token>``."""
    with TestClient(app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = f"Bearer {studio.auth.token}"
        yield c


# --------------------------------------------------------------------------- data
@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "data" / "train.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = ["instruction,output"] + [f"question {i},answer {i}" for i in range(50)]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def jsonl_file(tmp_path: Path) -> Path:
    p = tmp_path / "data" / "chat.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f'{{"text": "sample number {i}"}}\n' for i in range(30)), encoding="utf-8")
    return p


@pytest.fixture
def dataset_id(client: TestClient, csv_file: Path) -> str:
    r = client.post("/api/datasets", json={"source": "csv", "path": str(csv_file)})
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


# --------------------------------------------------------------------------- training fakes
@pytest.fixture
def fake_training(monkeypatch: pytest.MonkeyPatch, studio: Studio) -> Callable[..., FakeEngine]:
    """Patch the (parallel-built) training module functions and inject a FakeEngine factory.

    Returns ``make(steps=..., delay=...)`` which installs a new engine and returns it.
    """
    monkeypatch.setattr(training_pkg, "plan_steps", lambda cfg, info: 5, raising=False)
    monkeypatch.setattr(
        training_pkg, "validate_training_request", lambda m, t, info, hw: ["fake warning"], raising=False
    )

    def make(steps: int = 3, delay: float = 0.0) -> FakeEngine:
        engine = FakeEngine(steps=steps, delay=delay)
        studio.lifecycle.engine_factory = lambda m, t: engine
        return engine

    make()
    return make


@pytest.fixture
def fake_checkpoint_modules(monkeypatch: pytest.MonkeyPatch, studio: Studio, tmp_path: Path) -> dict[str, Any]:
    """Fake ``training.checkpoints`` / ``export.model_export`` modules and a fixed checkpoint list."""
    ck_dir = studio.runs_dir / "job-x" / "checkpoint-2"
    ck_dir.mkdir(parents=True)
    ckpt = CheckpointInfo(checkpoint_id="job-x-step2", path=str(ck_dir), step=2, created_at=time.time(),
                          size_bytes=10, job_id="job-x")
    calls: dict[str, Any] = {"ckpt": ckpt}

    ck_mod = types.ModuleType("autotune_studio.training.checkpoints")
    ck_mod.list_checkpoints = lambda runs_dir: [ckpt]  # type: ignore[attr-defined]
    ck_mod.load_checkpoint_state = lambda path: {  # type: ignore[attr-defined]
        "model_config": {"model_name": "sshleifer/tiny-gpt2"},
        "training_config": {"method": "lora", "max_steps": 4},
    }
    exp_mod = types.ModuleType("autotune_studio.export.model_export")

    def fake_export(req, ck, model_cfg, workspace):
        calls["export"] = (req, ck, model_cfg)
        out = Path(req.output_dir or (Path(workspace) / "exports" / ck.checkpoint_id))
        return ExportResult(path=str(out), format=req.format, files=["adapter_model.safetensors"], size_bytes=123)

    exp_mod.export_model = fake_export  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autotune_studio.training.checkpoints", ck_mod)
    monkeypatch.setitem(sys.modules, "autotune_studio.export.model_export", exp_mod)
    return calls


def wait_for(pred: Callable[[], bool], timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()
