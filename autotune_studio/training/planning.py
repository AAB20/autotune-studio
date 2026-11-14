"""Backend selection, request validation and step planning (light: no heavy imports at module load).

Owned by: llm-training. See ADR-0003 (step planning) and ADR-0004 (backends).
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import sys
from typing import Literal

from ..contracts import DatasetInfo, HardwareInfo, ModelConfig, TrainingConfig
from ..errors import DatasetEmpty, DependencyMissing, StepsRequired, ValidationFailed

log = logging.getLogger(__name__)

Backend = Literal["unsloth", "hf"]

#: Environment override for backend selection (``hf`` or ``unsloth``); mainly for debugging.
BACKEND_ENV = "AUTOTUNE_TRAINING_BACKEND"

EIGHT_BIT_OPTIMIZERS = frozenset({"adamw_8bit", "paged_adamw_8bit"})

TRAIN_INSTALL_HINT = 'pip install "autotune-studio[train]"  (torch, transformers, peft, trl, accelerate)'
GPU_INSTALL_HINT = 'pip install "autotune-studio[gpu]"  (unsloth, bitsandbytes) on a CUDA runtime'


def module_available(name: str) -> bool:
    """True if ``name`` is importable, without importing it (``find_spec``)."""
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def cuda_available() -> bool:
    """``torch.cuda.is_available()``; False when torch is missing or the probe fails."""
    if not module_available("torch"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken CUDA install means "no CUDA"
        return False


def _bf16_supported() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        return False


def select_backend(model_cfg: ModelConfig, train_cfg: TrainingConfig) -> Backend:
    """``"unsloth"`` if CUDA is available and Unsloth is installed, else ``"hf"``.

    ``AUTOTUNE_TRAINING_BACKEND=hf`` forces the HF backend; ``=unsloth`` forces Unsloth
    (fails later with ``DependencyMissing`` if it is not installed).
    """
    forced = os.environ.get(BACKEND_ENV, "").strip().lower()
    if forced in ("hf", "unsloth"):
        return forced  # type: ignore[return-value]
    if cuda_available() and module_available("unsloth"):
        return "unsloth"
    return "hf"


def plan_steps(train_cfg: TrainingConfig, dataset_info: DatasetInfo, world_size: int = 1) -> int:
    """Number of optimizer steps for a run.

    * explicit ``max_steps`` always wins;
    * known cardinality: ``ceil(ceil(rows * (1 - val_fraction) / (micro * accum * world)) * num_epochs)``;
    * unknown cardinality without ``max_steps`` -> :class:`~autotune_studio.errors.StepsRequired`.

    Raises ``DatasetEmpty`` if the dataset is known to have zero rows.
    """
    if train_cfg.max_steps is not None:
        return int(train_cfg.max_steps)
    rows = dataset_info.cardinality.rows
    if rows is None:
        raise StepsRequired(
            "The dataset length is unknown (streamed source without row-count metadata). "
            "Set training max_steps, or run a full dataset scan first to count rows.",
            {"dataset_id": dataset_info.dataset_id},
        )
    if rows <= 0:
        raise DatasetEmpty(f"dataset '{dataset_info.dataset_id}' has no rows")
    if train_cfg.num_epochs is None:
        raise StepsRequired("Set either num_epochs or max_steps.")
    per_step = train_cfg.per_device_batch_size * train_cfg.gradient_accumulation_steps * max(1, int(world_size))
    train_rows = max(1, math.ceil(rows * (1.0 - train_cfg.val_fraction)))
    steps_per_epoch = math.ceil(train_rows / per_step)
    return max(1, math.ceil(steps_per_epoch * float(train_cfg.num_epochs)))


def validate_training_request(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    dataset_info: DatasetInfo,
    hardware: HardwareInfo | None,
) -> list[str]:
    """Check a training request before any model is loaded. Returns human-readable warnings.

    Raises:
        ValidationFailed: QLoRA or 8-bit optimizers without CUDA, unsupported bf16, ...
        DependencyMissing: QLoRA / 8-bit optimizers on CUDA without bitsandbytes/unsloth.
        StepsRequired / DatasetEmpty: from :func:`plan_steps`.
    """
    warnings: list[str] = []
    cuda = hardware.cuda_available if hardware is not None else cuda_available()
    bf16 = hardware.bf16_supported if hardware is not None else (cuda and _bf16_supported())
    has_bnb = module_available("bitsandbytes")
    has_unsloth = module_available("unsloth")

    if train_cfg.method == "qlora":
        if not cuda:
            raise ValidationFailed(
                "QLoRA (4-bit quantised base model) requires a CUDA GPU, and none is available. "
                'Use method="lora" to train a full-precision LoRA adapter on CPU (slow, small models only), '
                "or run on a GPU runtime (Colab/Kaggle).",
                {"field": "method", "value": "qlora", "suggestion": "lora"},
            )
        if not (has_bnb or has_unsloth):
            raise DependencyMissing(
                f"QLoRA needs bitsandbytes or unsloth. Install with: {GPU_INSTALL_HINT}",
                {"missing": ["bitsandbytes", "unsloth"]},
            )
        if not model_cfg.load_in_4bit:
            warnings.append('method="qlora" always loads the base model in 4-bit (load_in_4bit=False is ignored).')
    elif model_cfg.load_in_4bit:
        warnings.append('method="lora" trains on a 16/32-bit base model; load_in_4bit is ignored.')

    if train_cfg.optimizer in EIGHT_BIT_OPTIMIZERS:
        if not cuda:
            raise ValidationFailed(
                f'optimizer="{train_cfg.optimizer}" requires bitsandbytes on a CUDA GPU. '
                'Use optimizer="adamw_torch" on CPU.',
                {"field": "optimizer", "value": train_cfg.optimizer, "suggestion": "adamw_torch"},
            )
        if not has_bnb:
            raise DependencyMissing(
                f'optimizer="{train_cfg.optimizer}" needs bitsandbytes. Install with: {GPU_INSTALL_HINT}',
                {"missing": ["bitsandbytes"]},
            )

    if model_cfg.dtype == "bfloat16" and cuda and not bf16:
        raise ValidationFailed(
            'dtype="bfloat16" is not supported by this GPU; use dtype="float16" or "auto".',
            {"field": "dtype"},
        )
    if not cuda:
        if model_cfg.dtype in ("float16", "bfloat16"):
            warnings.append(f'No CUDA GPU: training runs in float32 on CPU (dtype="{model_cfg.dtype}" ignored).')
        warnings.append("No CUDA GPU detected: training on CPU is very slow and only practical for tiny models.")

    if not module_available("transformers") or not module_available("peft") or not module_available("torch"):
        warnings.append(f"Training dependencies are not installed. Install with: {TRAIN_INSTALL_HINT}")

    if dataset_info.detected_format == "unknown":
        warnings.append(
            "Dataset format not recognised (chat / instruction / text); rows are trained as raw JSON text."
        )

    steps = plan_steps(train_cfg, dataset_info)
    if train_cfg.warmup_steps >= steps:
        warnings.append(f"warmup_steps ({train_cfg.warmup_steps}) >= total steps ({steps}); the LR never peaks.")
    if train_cfg.save_steps > steps:
        warnings.append(
            f"save_steps ({train_cfg.save_steps}) > total steps ({steps}); only the final checkpoint will be saved."
        )
    if train_cfg.val_fraction > 0:
        warnings.append(
            "Validation loss is estimated on a bounded deterministic sample of the validation split "
            "at every checkpoint."
        )
    return warnings
