"""TrainingLifecycle with a fake engine (no ML deps)."""

from __future__ import annotations

import threading
import time

import pytest

from autotune_studio.contracts import (
    Cardinality,
    CheckpointInfo,
    DatasetInfo,
    DatasetSource,
    DatasetSpec,
    MetricPoint,
    ModelConfig,
    TrainingConfig,
    TrainingState,
)
from autotune_studio.core.events import EventBus
from autotune_studio.core.lifecycle import TrainingLifecycle
from autotune_studio.errors import Conflict, ValidationFailed
from autotune_studio.interfaces import TrainingResult


class FakeDataset:
    def __init__(self) -> None:
        self.info = DatasetInfo(
            dataset_id="ds1",
            spec=DatasetSpec(source=DatasetSource.JSONL, path="x.jsonl"),
            columns=[],
            cardinality=Cardinality(),
        )


class FakeEngine:
    backend = "hf"

    def __init__(self, steps: int = 20, step_time: float = 0.01, fail: Exception | None = None) -> None:
        self.steps, self.step_time, self.fail = steps, step_time, fail
        self.started = threading.Event()

    def run(self, job_id, model_cfg, train_cfg, dataset, events, control, resume_from=None):
        events.publish("status", {"state": "running", "max_steps": self.steps, "backend": "hf"})
        self.started.set()
        ckpts = []
        step = 0
        for step in range(1, self.steps + 1):
            while control.pause.is_set() and not control.stop.is_set():
                if control.save_now.is_set():
                    break
                time.sleep(0.01)
            if self.fail and step == 3:
                raise self.fail
            time.sleep(self.step_time)
            events.publish("metric", MetricPoint(step=step, loss=1.0 / step, learning_rate=1e-4).model_dump())
            if control.save_now.is_set() or control.stop.is_set():
                c = CheckpointInfo(checkpoint_id=f"{job_id}-{step}", path=f"/tmp/{step}", step=step,
                                   created_at=time.time(), size_bytes=1, job_id=job_id)
                ckpts.append(c)
                events.publish("checkpoint", c.model_dump())
                control.save_now.clear()
            if control.stop.is_set():
                return TrainingResult(job_id, step, 1.0 / step, True, ckpts, "/tmp")
        return TrainingResult(job_id, step, 1.0 / step, False, ckpts, "/tmp")


def _mk(engine, tmp_path):
    bus = EventBus()
    lc = TrainingLifecycle(bus, lambda m, t: engine, history_path=tmp_path / "sessions.json")
    return bus, lc


MODEL = ModelConfig(model_name="tiny/model")
CFG = TrainingConfig(method="lora", max_steps=20)


def test_run_to_completion_tracks_metrics_and_history(tmp_path):
    eng = FakeEngine(steps=10)
    bus, lc = _mk(eng, tmp_path)
    lc.start(MODEL, CFG, FakeDataset(), max_steps=10)
    st = lc.wait(10)
    assert st.state == TrainingState.COMPLETED
    assert st.step == 10 and st.loss == pytest.approx(0.1)
    assert len(lc.metrics.list()) == 10
    types = [e.type for e in bus.replay(0)[0]]
    assert types[0] == "status" and "metric" in types and types[-1] == "status"
    # history persisted and reloaded
    lc2 = TrainingLifecycle(bus, lambda m, t: eng, history_path=tmp_path / "sessions.json")
    assert [s.state for s in lc2.sessions()] == [TrainingState.COMPLETED]


def test_second_start_conflicts(tmp_path):
    eng = FakeEngine(steps=200, step_time=0.01)
    _, lc = _mk(eng, tmp_path)
    lc.start(MODEL, CFG, FakeDataset())
    with pytest.raises(Conflict):
        lc.start(MODEL, CFG, FakeDataset())
    lc.stop(wait=True, timeout=10)


def test_pause_resume_stop_and_checkpoint(tmp_path):
    eng = FakeEngine(steps=500, step_time=0.005)
    _, lc = _mk(eng, tmp_path)
    lc.start(MODEL, CFG, FakeDataset())
    assert eng.started.wait(5)
    time.sleep(0.05)
    assert lc.pause().state == TrainingState.PAUSED
    step_paused = lc.status().step
    time.sleep(0.1)
    assert lc.status().step <= step_paused + 1
    ck = lc.request_checkpoint(timeout=5)
    assert ck.job_id == lc.status().job_id
    assert lc.resume().state == TrainingState.RUNNING
    st = lc.stop(wait=True, timeout=10)
    assert st.state == TrainingState.STOPPED


def test_failure_is_reported(tmp_path):
    eng = FakeEngine(steps=10, fail=ValidationFailed("bad config"))
    bus, lc = _mk(eng, tmp_path)
    lc.start(MODEL, CFG, FakeDataset())
    st = lc.wait(10)
    assert st.state == TrainingState.FAILED
    assert "bad config" in (st.error or "")
    assert any(e.type == "error" for e in bus.replay(0)[0])


def test_checkpoint_when_idle_conflicts(tmp_path):
    _, lc = _mk(FakeEngine(), tmp_path)
    with pytest.raises(Conflict):
        lc.request_checkpoint(timeout=0.1)
