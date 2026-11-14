"""The hf engine driven by the real ``core.lifecycle.TrainingLifecycle`` (contract smoke test)."""

from __future__ import annotations

import time

import pytest

from autotune_studio.contracts import ModelConfig, TrainingConfig, TrainingState
from autotune_studio.training import create_engine
from autotune_studio.training.checkpoints import list_checkpoints

from .conftest import FakeSink

pytestmark = pytest.mark.slow


def test_lifecycle_pause_checkpoint_resume_complete(tiny_model_dir, jsonl_dataset, tmp_path):
    from autotune_studio.core.lifecycle import TrainingLifecycle

    sink = FakeSink()
    lc = TrainingLifecycle(sink, create_engine)
    runs = tmp_path / "runs"
    m = ModelConfig(model_name=str(tiny_model_dir), max_seq_length=64, load_in_4bit=False)
    t = TrainingConfig(method="lora", max_steps=30, save_steps=100, per_device_batch_size=2,
                       gradient_accumulation_steps=1, warmup_steps=0, output_dir=str(runs / "job-l"))
    lc.start(m, t, jsonl_dataset, max_steps=30, job_id="job-l")
    deadline = time.monotonic() + 120
    while lc.status().step < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    lc.pause()
    ck = lc.request_checkpoint(timeout=60)  # saved while paused
    step_at_pause = lc.status().step
    time.sleep(0.5)
    assert lc.status().state == TrainingState.PAUSED and lc.status().step == step_at_pause
    lc.resume()
    st = lc.wait(180)
    assert st.state == TrainingState.COMPLETED, st
    assert st.step == 30 and st.backend == "hf"
    ids = [c.checkpoint_id for c in list_checkpoints(runs)]
    assert ck.checkpoint_id in ids and ids[-1] == "job-l-checkpoint-30"
    assert lc.last_result is not None and not lc.last_result.stopped_early
