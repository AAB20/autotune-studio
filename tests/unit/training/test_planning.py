"""Backend selection, validation and step planning (fast, no model)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autotune_studio import training
from autotune_studio.contracts import DatasetSource, DatasetSpec, HardwareInfo, ModelConfig, TrainingConfig
from autotune_studio.data_engine import open_dataset
from autotune_studio.errors import DependencyMissing, StepsRequired, ValidationFailed
from autotune_studio.training import planning

from .conftest import make_rows, open_jsonl, write_jsonl

CPU = HardwareInfo(cpu_count=4, ram_total_bytes=8 << 30, ram_available_bytes=4 << 30, disk_path=".",
                   disk_total_bytes=1 << 40, disk_free_bytes=1 << 39, cuda_available=False)
GPU = CPU.model_copy(update={"cuda_available": True, "bf16_supported": True})
MODEL = ModelConfig(model_name="tiny", load_in_4bit=False)


def _parquet(tmp_path: Path, n: int):
    p = tmp_path / "d.parquet"
    pq.write_table(pa.Table.from_pylist(make_rows(n)), p)
    return open_dataset(DatasetSpec(source=DatasetSource.PARQUET, path=str(p)))


def test_known_cardinality_parquet_plans_from_epochs(tmp_path):
    ds = _parquet(tmp_path, 100)
    assert ds.info.cardinality.rows == 100
    cfg = TrainingConfig(method="lora", per_device_batch_size=2, gradient_accumulation_steps=4, num_epochs=2)
    assert training.plan_steps(cfg, ds.info) == 2 * 13  # ceil(100/8)=13 per epoch
    cfg = cfg.model_copy(update={"val_fraction": 0.2, "num_epochs": 1.5})
    assert training.plan_steps(cfg, ds.info) == 15  # ceil(ceil(80/8)*1.5)
    assert training.plan_steps(cfg, ds.info, world_size=2) == 8  # ceil(ceil(80/16)*1.5)
    assert training.plan_steps(cfg.model_copy(update={"max_steps": 7}), ds.info) == 7  # explicit wins


def test_unknown_cardinality_requires_max_steps(tmp_path):
    ds = open_jsonl(write_jsonl(tmp_path / "d.jsonl", make_rows(10)))
    assert ds.info.cardinality.rows is None
    with pytest.raises(StepsRequired):
        training.plan_steps(TrainingConfig(method="lora"), ds.info)
    with pytest.raises(StepsRequired):
        training.validate_training_request(MODEL, TrainingConfig(method="lora"), ds.info, CPU)
    assert training.plan_steps(TrainingConfig(method="lora", max_steps=5), ds.info) == 5


def test_qlora_on_cpu_is_rejected(tmp_path):
    ds = _parquet(tmp_path, 10)
    with pytest.raises(ValidationFailed) as ei:
        training.validate_training_request(MODEL, TrainingConfig(method="qlora"), ds.info, CPU)
    assert 'method="lora"' in ei.value.message
    assert ei.value.detail["suggestion"] == "lora"


@pytest.mark.parametrize("opt", ["adamw_8bit", "paged_adamw_8bit"])
def test_8bit_optimizer_on_cpu_is_rejected(tmp_path, opt):
    ds = _parquet(tmp_path, 10)
    with pytest.raises(ValidationFailed):
        training.validate_training_request(MODEL, TrainingConfig(method="lora", optimizer=opt), ds.info, CPU)


def test_gpu_without_bitsandbytes_reports_dependency(tmp_path, monkeypatch):
    ds = _parquet(tmp_path, 10)
    monkeypatch.setattr(planning, "module_available", lambda n: n not in ("bitsandbytes", "unsloth"))
    with pytest.raises(DependencyMissing):
        training.validate_training_request(MODEL, TrainingConfig(method="qlora"), ds.info, GPU)
    with pytest.raises(DependencyMissing):
        training.validate_training_request(MODEL, TrainingConfig(method="lora", optimizer="adamw_8bit"), ds.info, GPU)


def test_lora_on_cpu_warnings(tmp_path):
    ds = _parquet(tmp_path, 10)
    w = training.validate_training_request(ModelConfig(model_name="tiny", dtype="bfloat16"),
                                           TrainingConfig(method="lora", save_steps=1000), ds.info, CPU)
    text = " ".join(w)
    assert "load_in_4bit is ignored" in text and "float32" in text and "only the final checkpoint" in text


def test_select_backend(monkeypatch):
    cfg = TrainingConfig(method="lora")
    monkeypatch.delenv(planning.BACKEND_ENV, raising=False)
    monkeypatch.setattr(planning, "cuda_available", lambda: False)
    assert training.select_backend(MODEL, cfg) == "hf"
    monkeypatch.setattr(planning, "cuda_available", lambda: True)
    monkeypatch.setattr(planning, "module_available", lambda n: n == "unsloth")
    assert training.select_backend(MODEL, cfg) == "unsloth"
    monkeypatch.setattr(planning, "module_available", lambda n: False)
    assert training.select_backend(MODEL, cfg) == "hf"
    monkeypatch.setenv(planning.BACKEND_ENV, "hf")
    monkeypatch.setattr(planning, "module_available", lambda n: True)
    assert training.select_backend(MODEL, cfg) == "hf"


def test_create_engine_hf_on_cpu(monkeypatch):
    monkeypatch.delenv(planning.BACKEND_ENV, raising=False)
    eng = training.create_engine(MODEL, TrainingConfig(method="lora"))
    assert eng.backend == "hf"


def test_create_engine_missing_dependency(monkeypatch):
    monkeypatch.setattr(training, "module_available", lambda n: n != "peft")
    with pytest.raises(DependencyMissing) as ei:
        training.create_engine(MODEL, TrainingConfig(method="lora"))
    assert "peft" in ei.value.detail["missing"]


def test_create_engine_unsloth_with_mocked_module(monkeypatch):
    fake = types.ModuleType("unsloth")
    fake.FastLanguageModel = object()
    monkeypatch.setitem(sys.modules, "unsloth", fake)
    monkeypatch.setenv(planning.BACKEND_ENV, "unsloth")
    eng = training.create_engine(MODEL, TrainingConfig(method="qlora"))
    assert eng.backend == "unsloth"


def test_create_engine_unsloth_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", None)  # import -> ImportError
    monkeypatch.setenv(planning.BACKEND_ENV, "unsloth")
    with pytest.raises(DependencyMissing):
        training.create_engine(MODEL, TrainingConfig(method="qlora"))


def test_unsloth_kwargs_construction():
    from autotune_studio.training.unsloth_backend import unsloth_load_kwargs, unsloth_peft_kwargs

    m = ModelConfig(model_name="unsloth/llama-3-8b", max_seq_length=1024, revision="abc")
    t = TrainingConfig(method="qlora", lora_r=8, lora_alpha=32, lora_dropout=0.05, seed=7)
    assert unsloth_load_kwargs(m, t) == {"model_name": "unsloth/llama-3-8b", "max_seq_length": 1024, "dtype": None,
                                         "load_in_4bit": True, "trust_remote_code": False, "revision": "abc"}
    assert unsloth_load_kwargs(m.model_copy(update={"dtype": "float16"}),
                               t.model_copy(update={"method": "lora"}))["load_in_4bit"] is False
    pk = unsloth_peft_kwargs(m, t)
    assert pk["r"] == 8 and pk["lora_alpha"] == 32 and pk["lora_dropout"] == 0.05 and pk["random_state"] == 7
    assert pk["use_gradient_checkpointing"] == "unsloth" and "q_proj" in pk["target_modules"]
    assert unsloth_peft_kwargs(m, t.model_copy(update={"target_modules": ["q_proj"]}))["target_modules"] == ["q_proj"]


def test_training_package_import_is_light():
    import subprocess

    code = ("import sys, autotune_studio.training, autotune_studio.training.checkpoints, autotune_studio.export\n"
            "print([m for m in ('torch','transformers','peft','trl','datasets') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
