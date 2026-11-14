"""End-to-end CPU LoRA training with the hf backend on a fully local tiny Llama (no network)."""

from __future__ import annotations

import math
import threading
from pathlib import Path

import pytest

from autotune_studio.contracts import CheckpointInfo, ExportRequest, ModelConfig, TrainingConfig
from autotune_studio.errors import Conflict, OutOfMemory, ValidationFailed
from autotune_studio.export.model_export import export_model
from autotune_studio.interfaces import TrainingControl
from autotune_studio.training.checkpoints import list_checkpoints, load_checkpoint_state
from autotune_studio.training.engine import HFTrainingEngine

from .conftest import FakeSink, make_rows, open_jsonl, write_jsonl

pytestmark = pytest.mark.slow


def cfgs(model_dir: Path, out: Path, **kw) -> tuple[ModelConfig, TrainingConfig]:
    base = {"method": "lora", "max_steps": 7, "save_steps": 3, "per_device_batch_size": 2, "gradient_accumulation_steps": 1,
                "learning_rate": 1e-3, "warmup_steps": 0, "logging_steps": 1, "lora_r": 4, "lora_alpha": 8,
                "output_dir": str(out), "val_fraction": 0.2}
    base.update(kw)
    return (ModelConfig(model_name=str(model_dir), max_seq_length=64, load_in_4bit=False, dtype="auto"),
            TrainingConfig(**base))


@pytest.fixture(scope="module")
def full_run(tiny_model_dir, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("full")
    ds = open_jsonl(write_jsonl(tmp / "train.jsonl", make_rows(40)))
    runs = tmp / "runs"
    m, t = cfgs(tiny_model_dir, runs / "job-1")
    sink = FakeSink()
    result = HFTrainingEngine().run("job-1", m, t, ds, sink, TrainingControl())
    return {"result": result, "sink": sink, "runs": runs, "model": m, "train": t, "ds": ds, "tmp": tmp}


def test_full_run_events_and_checkpoints(full_run):
    r, sink = full_run["result"], full_run["sink"]
    assert full_run["ds"].info.cardinality.rows is None  # unknown length -> max_steps used
    assert r.final_step == 7 and not r.stopped_early and r.final_loss is not None
    assert [c.step for c in r.checkpoints] == [3, 6, 7]
    assert [c.is_final for c in r.checkpoints] == [False, False, True]
    states = [s["state"] for s in sink.of("status") if "state" in s]
    assert states[0] == "preparing" and "running" in states
    metrics = [m for m in sink.of("metric") if m.get("loss") is not None]
    assert [m["step"] for m in metrics] == list(range(1, 8))
    assert all(math.isfinite(m["loss"]) and m["learning_rate"] is not None and m["grad_norm"] is not None
               for m in metrics)
    evals = [m for m in sink.of("metric") if m.get("eval_loss") is not None]
    assert [m["step"] for m in evals] == [3, 6, 7] and all(math.isfinite(m["eval_loss"]) for m in evals)
    assert [c["step"] for c in sink.of("checkpoint")] == [3, 6, 7]
    assert any(e["source"] == "training" and "checkpoint saved" in e["message"] for e in sink.of("log"))
    for c in r.checkpoints:
        p = Path(c.path)
        assert (p / "autotune_state.json").is_file() and (p / "adapter_model.safetensors").is_file()
        assert (p / "tokenizer.json").is_file() and c.size_bytes > 0
        st = load_checkpoint_state(p)
        assert st["step"] == c.step and st["job_id"] == "job-1" and st["data_cursor"]["kind"] == "jsonl"
        assert st["model_config"]["model_name"] == full_run["model"].model_name
    # data cursor: source rows consumed grows monotonically (2 train rows per step + skipped val rows)
    cur = [c.data_cursor["rows_consumed"] for c in r.checkpoints]
    assert cur == sorted(cur) and cur[0] >= 6
    listed = list_checkpoints(full_run["runs"])
    assert [c.checkpoint_id for c in listed] == [c.checkpoint_id for c in r.checkpoints]


def test_export_lora_adapter_reloads_with_peft(full_run, tmp_path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    final = full_run["result"].checkpoints[-1]
    res = export_model(ExportRequest(format="lora_adapter"), final, full_run["model"], tmp_path)
    assert Path(res.path) == (tmp_path / "exports" / f"{final.checkpoint_id}-lora").resolve()
    assert "adapter_model.safetensors" in res.files and "adapter_config.json" in res.files
    assert "tokenizer.json" in res.files and "optimizer.pt" not in res.files and res.size_bytes > 0
    base = AutoModelForCausalLM.from_pretrained(full_run["model"].model_name)
    model = PeftModel.from_pretrained(base, res.path)
    out = model(input_ids=torch.tensor([[1, 5, 6, 7]]))
    assert out.logits.shape[-1] == base.config.vocab_size
    with pytest.raises(Conflict):  # never overwrite a non-empty destination
        export_model(ExportRequest(format="lora_adapter"), final, full_run["model"], tmp_path)


def test_export_merged_16bit_loads_and_runs(full_run, tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    final = full_run["result"].checkpoints[-1]
    dest = tmp_path / "merged"
    res = export_model(ExportRequest(format="merged_16bit", output_dir=str(dest)), final, full_run["model"],
                       tmp_path / "unused")
    assert Path(res.path) == dest.resolve() and "config.json" in res.files
    assert any(f.endswith(".safetensors") for f in res.files) and "adapter_config.json" not in res.files
    model = AutoModelForCausalLM.from_pretrained(res.path)
    assert model.dtype in (torch.float16, torch.bfloat16)
    out = model.float()(input_ids=torch.tensor([[1, 5, 6, 7]]))
    assert torch.isfinite(out.logits).all()


def test_resume_from_checkpoint_continues_exactly(full_run, tiny_model_dir, tmp_path):
    import torch
    from safetensors.torch import load_file

    r = full_run["result"]
    ck3 = r.checkpoints[0]
    m, t = cfgs(tiny_model_dir, tmp_path / "job-2")
    sink = FakeSink()
    res = HFTrainingEngine().run("job-2", m, t, full_run["ds"], sink, TrainingControl(), resume_from=ck3)
    steps = [p["step"] for p in sink.of("metric") if p.get("loss") is not None]
    assert steps == [4, 5, 6, 7]  # continues after the checkpoint step
    assert [c.step for c in res.checkpoints] == [6, 7] and res.final_step == 7
    # same data position and (near-)identical weights as the uninterrupted run
    assert res.checkpoints[0].data_cursor == r.checkpoints[1].data_cursor
    a = load_file(str(Path(r.checkpoints[-1].path) / "adapter_model.safetensors"))
    b = load_file(str(Path(res.checkpoints[-1].path) / "adapter_model.safetensors"))
    assert a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=1e-5) for k in a)


def test_stop_mid_run_saves_checkpoint(tiny_model_dir, jsonl_dataset, tmp_path):
    control = TrainingControl()

    def hook(type_, data):
        if type_ == "metric" and data.get("step") == 2:
            control.stop.set()

    m, t = cfgs(tiny_model_dir, tmp_path / "job-s", max_steps=20, save_steps=50, val_fraction=0.0)
    res = HFTrainingEngine().run("job-s", m, t, jsonl_dataset, FakeSink(hook), control)
    # the metric for step 2 is logged after step 2's boundary check, so the stop is honoured after step 3
    assert res.stopped_early and res.final_step == 3
    assert [c.step for c in res.checkpoints] == [3] and not res.checkpoints[0].is_final
    assert (Path(res.checkpoints[0].path) / "adapter_model.safetensors").is_file()


def test_pause_save_now_and_resume(tiny_model_dir, jsonl_dataset, tmp_path):
    control = TrainingControl()

    def hook(type_, data):
        if type_ == "metric" and data.get("step") == 2:
            control.pause.set()

    m, t = cfgs(tiny_model_dir, tmp_path / "job-p", max_steps=5, save_steps=50, val_fraction=0.0)
    sink = FakeSink(hook)
    out: dict = {}
    th = threading.Thread(target=lambda: out.update(
        r=HFTrainingEngine().run("job-p", m, t, jsonl_dataset, sink, control)))
    th.start()
    assert sink.wait_for(lambda ty, d: ty == "status" and d.get("state") == "paused", timeout=120)
    paused_step = next(s["step"] for s in sink.of("status") if s.get("state") == "paused")
    assert paused_step == 3  # pause requested during step 2's log -> honoured at the next step boundary
    control.save_now.set()  # honoured while paused
    assert sink.wait_for(lambda ty, d: ty == "checkpoint", timeout=60)
    assert sink.of("checkpoint")[0]["step"] == paused_step
    assert max(p["step"] for p in sink.of("metric")) < paused_step  # no further step while paused
    assert not control.save_now.is_set()
    control.pause.clear()
    th.join(120)
    r = out["r"]
    assert not r.stopped_early and r.final_step == 5
    assert [c.step for c in r.checkpoints] == [paused_step, 5] and r.checkpoints[-1].is_final
    states = [s["state"] for s in sink.of("status") if "state" in s]
    assert "running" in states[states.index("paused"):]


def test_save_now_while_running(tiny_model_dir, jsonl_dataset, tmp_path):
    control = TrainingControl()

    def hook(type_, data):
        if type_ == "metric" and data.get("step") == 2:
            control.save_now.set()

    m, t = cfgs(tiny_model_dir, tmp_path / "job-n", max_steps=4, save_steps=50, val_fraction=0.0)
    res = HFTrainingEngine().run("job-n", m, t, jsonl_dataset, FakeSink(hook), control)
    assert [c.step for c in res.checkpoints] == [3, 4]


class _OOMOnce:
    """Trainer callback raising a CUDA OOM at ``step`` during the first attempt only."""

    def __init__(self, step: int) -> None:
        self.step, self.fired = step, False

    def __getattr__(self, name):  # TrainerCallback interface: every other hook is a no-op
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)

    def on_step_end(self, args, state, control, **kw):
        import torch

        if not self.fired and state.global_step == self.step:
            self.fired = True
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (simulated)")


def test_simulated_oom_recovers_from_last_checkpoint(tiny_model_dir, jsonl_dataset, tmp_path):
    m, t = cfgs(tiny_model_dir, tmp_path / "job-o", max_steps=6, save_steps=3, val_fraction=0.0)
    sink = FakeSink()
    eng = HFTrainingEngine(extra_callbacks=[_OOMOnce(4)])
    res = eng.run("job-o", m, t, jsonl_dataset, sink, TrainingControl())
    assert eng.attempts == 2 and res.final_step == 6 and not res.stopped_early
    assert [c.step for c in res.checkpoints] == [3, 6]
    warn = [e for e in sink.of("log") if e["level"] == "warning"]
    assert any(e.get("source") == "oom_policy" and e["new_micro_batch"] == 1 and e["new_grad_accum"] == 2
               for e in warn)
    assert any("resuming from checkpoint step 3" in e["message"] for e in warn)
    steps = [p["step"] for p in sink.of("metric") if p.get("loss") is not None]
    assert steps == [1, 2, 3, 4, 5, 6]  # step 4 raised before logging; retried after resume from step 3
    assert any(s.get("micro_batch_size") == 1 and s.get("effective_batch_size") == 2 for s in sink.of("status"))
    st = load_checkpoint_state(res.checkpoints[-1].path)
    assert st["micro_batch_size"] == 1 and st["gradient_accumulation_steps"] == 2


def test_oom_at_minimum_micro_batch_raises(tiny_model_dir, jsonl_dataset, tmp_path):
    m, t = cfgs(tiny_model_dir, tmp_path / "job-x", max_steps=3, per_device_batch_size=1, val_fraction=0.0)
    with pytest.raises(OutOfMemory):
        HFTrainingEngine(extra_callbacks=[_OOMOnce(1)]).run("job-x", m, t, jsonl_dataset, FakeSink(),
                                                            TrainingControl())


def test_output_dir_required(tiny_model_dir, jsonl_dataset):
    m, t = cfgs(tiny_model_dir, Path("unused"))
    with pytest.raises(ValidationFailed):
        HFTrainingEngine().run("j", m, t.model_copy(update={"output_dir": None}), jsonl_dataset, FakeSink(),
                               TrainingControl())


def test_resume_requires_trainer_state(tiny_model_dir, jsonl_dataset, tmp_path):
    m, t = cfgs(tiny_model_dir, tmp_path / "j")
    fake = CheckpointInfo(checkpoint_id="x", path=str(tmp_path), step=1, created_at=0, size_bytes=0)
    with pytest.raises(ValidationFailed):
        HFTrainingEngine().run("j", m, t, jsonl_dataset, FakeSink(), TrainingControl(), resume_from=fake)
