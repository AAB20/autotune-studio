"""Training: backend selection, validation, step planning and the training engines (ADR-0004).

Importing this package is light; torch / transformers / peft / trl / unsloth are imported only
by :func:`create_engine` (and the engines it returns). Owned by: llm-training.
"""

from __future__ import annotations

import importlib

from ..contracts import ModelConfig, TrainingConfig
from ..errors import DependencyMissing
from ..interfaces import TrainingEngineProtocol
from .planning import (
    TRAIN_INSTALL_HINT,
    module_available,
    plan_steps,
    select_backend,
    validate_training_request,
)

__all__ = ["create_engine", "plan_steps", "select_backend", "validate_training_request"]

_REQUIRED = ("torch", "transformers", "peft", "accelerate", "datasets")


def create_engine(model_cfg: ModelConfig, train_cfg: TrainingConfig) -> TrainingEngineProtocol:
    """Return the engine for :func:`select_backend` (``UnslothTrainingEngine`` or ``HFTrainingEngine``).

    Raises ``DependencyMissing`` (with an install hint) if the training stack is not installed.
    """
    missing = [m for m in _REQUIRED if not module_available(m)]
    backend = select_backend(model_cfg, train_cfg)
    if backend == "unsloth":
        missing += [m for m in ("trl",) if not module_available(m)]
    if missing:
        raise DependencyMissing(f"Missing training dependencies {missing}. Install with: {TRAIN_INSTALL_HINT}",
                                {"missing": missing})
    if backend == "unsloth":
        # Unsloth must be imported before transformers (imported by .engine) so its patches apply.
        try:
            importlib.import_module("unsloth")
        except ImportError as e:
            from .planning import GPU_INSTALL_HINT

            raise DependencyMissing(f"Unsloth is not installed. Install with: {GPU_INSTALL_HINT}",
                                    {"missing": ["unsloth"]}) from e
        from .unsloth_backend import UnslothTrainingEngine

        return UnslothTrainingEngine()
    from .engine import HFTrainingEngine

    return HFTrainingEngine()
