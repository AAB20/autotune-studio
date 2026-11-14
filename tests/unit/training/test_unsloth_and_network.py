"""Unsloth backend wiring with a mocked ``unsloth`` module (CPU), and a cached-Hub tiny-model run."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest

from autotune_studio.contracts import ModelConfig, TrainingConfig
from autotune_studio.interfaces import TrainingControl

from .conftest import FakeSink


def _fake_unsloth(calls: dict) -> types.ModuleType:
    """Stand-in for Unsloth's FastLanguageModel backed by Transformers + PEFT (records the kwargs)."""

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kw):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            calls["from_pretrained"] = kw
            return AutoModelForCausalLM.from_pretrained(kw["model_name"]), AutoTokenizer.from_pretrained(
                kw["model_name"])

        @staticmethod
        def get_peft_model(model, **kw):
            from peft import LoraConfig, get_peft_model

            calls["get_peft_model"] = kw
            targets = [t for t in kw["target_modules"] if t in ("q_proj", "k_proj", "v_proj", "o_proj")]
            return get_peft_model(model, LoraConfig(r=kw["r"], lora_alpha=kw["lora_alpha"],
                                                    lora_dropout=kw["lora_dropout"], target_modules=targets,
                                                    task_type="CAUSAL_LM"))

    mod = types.ModuleType("unsloth")
    mod.FastLanguageModel = FastLanguageModel
    return mod


@pytest.mark.slow
def test_unsloth_engine_with_mocked_module_runs_sfttrainer(tiny_model_dir, jsonl_dataset, tmp_path, monkeypatch):
    calls: dict = {}
    monkeypatch.setitem(sys.modules, "unsloth", _fake_unsloth(calls))
    from autotune_studio.training.unsloth_backend import UnslothTrainingEngine

    m = ModelConfig(model_name=str(tiny_model_dir), max_seq_length=64, load_in_4bit=False)
    t = TrainingConfig(method="lora", max_steps=3, save_steps=2, per_device_batch_size=2,
                       gradient_accumulation_steps=1, warmup_steps=0, lora_r=4, lora_alpha=8,
                       output_dir=str(tmp_path / "job-u"))
    eng = UnslothTrainingEngine()
    sink = FakeSink()
    res = eng.run("job-u", m, t, jsonl_dataset, sink, TrainingControl())
    assert calls["from_pretrained"]["load_in_4bit"] is False and calls["from_pretrained"]["dtype"] is None
    assert calls["get_peft_model"]["use_gradient_checkpointing"] == "unsloth"
    assert eng.trainer_class().__mro__[1].__name__ == "SFTTrainer"
    assert res.final_step == 3 and [c.step for c in res.checkpoints] == [2, 3]
    assert all(s.get("backend") in (None, "unsloth") for s in sink.of("status"))
    assert all(math.isfinite(p["loss"]) for p in sink.of("metric") if p.get("loss") is not None)
    from autotune_studio.training.checkpoints import load_checkpoint_state

    assert load_checkpoint_state(res.checkpoints[-1].path)["backend"] == "unsloth"


@pytest.mark.network
@pytest.mark.slow
def test_hub_tiny_random_llama_lora_run(tmp_path, monkeypatch):
    from autotune_studio.training.engine import HFTrainingEngine

    from .conftest import make_rows, open_jsonl, write_jsonl

    ds = open_jsonl(write_jsonl(tmp_path / "d.jsonl", make_rows(12)))
    m = ModelConfig(model_name="hf-internal-testing/tiny-random-LlamaForCausalLM", max_seq_length=32,
                    load_in_4bit=False)
    t = TrainingConfig(method="lora", max_steps=3, save_steps=3, per_device_batch_size=2,
                       gradient_accumulation_steps=1, warmup_steps=0, output_dir=str(tmp_path / "job-h"))
    res = HFTrainingEngine().run("job-h", m, t, ds, FakeSink(), TrainingControl())
    assert res.final_step == 3 and res.checkpoints[-1].is_final
    assert (Path(res.checkpoints[-1].path) / "adapter_model.safetensors").is_file()
    assert res.final_loss is not None and math.isfinite(res.final_loss)
