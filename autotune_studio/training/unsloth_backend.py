"""Unsloth backend (CUDA only): ``FastLanguageModel`` + TRL ``SFTTrainer``.

**UNVERIFIED on real hardware**: the dev box has no GPU and Unsloth is not installed. The
model-loading calls follow Unsloth's documented API; the trainer path (SFTTrainer over the
streaming ``IterableDataset`` with ``skip_prepare_dataset``) is exercised in unit tests with a
mocked ``unsloth`` module on CPU. Import ``unsloth`` before ``transformers`` (``create_engine``
does this before importing this module).

Owned by: llm-training.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ..contracts import ModelConfig, TrainingConfig
from ..errors import DependencyMissing
from .engine import LLAMA_STYLE_TARGETS, HFTrainingEngine, _cursor_tracking, resolve_dtype
from .planning import GPU_INSTALL_HINT


def import_unsloth() -> Any:
    """Import and return the ``unsloth`` module, or raise ``DependencyMissing``."""
    try:
        return importlib.import_module("unsloth")
    except ImportError as e:
        raise DependencyMissing(f"Unsloth is not installed. Install with: {GPU_INSTALL_HINT}",
                                {"missing": ["unsloth"]}) from e


def unsloth_load_kwargs(model_cfg: ModelConfig, train_cfg: TrainingConfig) -> dict[str, Any]:
    """Keyword arguments for ``FastLanguageModel.from_pretrained``."""
    kw: dict[str, Any] = {
        "model_name": model_cfg.model_name,
        "max_seq_length": model_cfg.max_seq_length,
        "dtype": None,  # Unsloth auto-detects (bf16 on Ampere+, else fp16)
        "load_in_4bit": train_cfg.method == "qlora",
        "trust_remote_code": False,
    }
    if model_cfg.dtype == "float16":
        kw["dtype"] = "float16"
    elif model_cfg.dtype == "bfloat16":
        kw["dtype"] = "bfloat16"
    if model_cfg.revision:
        kw["revision"] = model_cfg.revision
    return kw


def unsloth_peft_kwargs(model_cfg: ModelConfig, train_cfg: TrainingConfig) -> dict[str, Any]:
    """Keyword arguments for ``FastLanguageModel.get_peft_model`` (``model`` passed positionally)."""
    return {
        "r": train_cfg.lora_r,
        "target_modules": list(train_cfg.target_modules or LLAMA_STYLE_TARGETS),
        "lora_alpha": train_cfg.lora_alpha,
        "lora_dropout": train_cfg.lora_dropout,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": train_cfg.seed,
        "max_seq_length": model_cfg.max_seq_length,
    }


def _torch_dtype(name: Any) -> Any:
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(name, name)


class UnslothTrainingEngine(HFTrainingEngine):
    """Same control/event/checkpoint machinery as :class:`HFTrainingEngine`; Unsloth model + SFTTrainer."""

    backend = "unsloth"

    def load_model_and_tokenizer(self, model_cfg: ModelConfig, train_cfg: TrainingConfig) -> tuple[Any, Any]:
        unsloth = import_unsloth()
        flm = unsloth.FastLanguageModel
        kw = unsloth_load_kwargs(model_cfg, train_cfg)
        kw["dtype"] = _torch_dtype(kw["dtype"])
        model, tokenizer = flm.from_pretrained(**kw)
        model = flm.get_peft_model(model, **unsloth_peft_kwargs(model_cfg, train_cfg))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return model, tokenizer

    def trainer_class(self) -> type:
        from trl import SFTTrainer

        return _cursor_tracking(SFTTrainer)

    def training_arguments(self, train_cfg: TrainingConfig, output_dir: Path, max_steps: int, micro: int,
                           accum: int, model_cfg: ModelConfig) -> Any:
        from trl import SFTConfig

        cuda = self._cuda()
        dtype = resolve_dtype(model_cfg, cuda)
        kw = self._args_kwargs(train_cfg, output_dir, max_steps, micro, accum, cuda, dtype)
        kw.update(
            max_length=model_cfg.max_seq_length,
            packing=False,
            dataset_kwargs={"skip_prepare_dataset": True},  # rows are tokenised by our streaming pipeline
        )
        return SFTConfig(**kw)
