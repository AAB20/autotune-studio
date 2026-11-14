"""Model export from a training checkpoint: LoRA adapter copy or merged 16-bit model.

* ``lora_adapter``: copies adapter + tokenizer files (and ``autotune_state.json`` for
  provenance) from the checkpoint; optimizer / scheduler / RNG / trainer state are omitted.
* ``merged_16bit``: loads the base model in 16-bit, applies the adapter, ``merge_and_unload()``
  and saves model + tokenizer. With the Unsloth backend on CUDA, Unsloth's
  ``save_pretrained_merged(save_method="merged_16bit")`` is used instead (UNVERIFIED here).

The destination is ``request.output_dir`` or ``<default_output_root>/exports/<checkpoint_id>-<suffix>``.
A non-empty existing destination raises :class:`~autotune_studio.errors.Conflict` (never overwritten).
Files are written to a sibling ``.partial`` directory and renamed into place when complete.

Owned by: llm-training.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from ..contracts import CheckpointInfo, ExportRequest, ExportResult, ModelConfig
from ..errors import Conflict, DependencyMissing, NotFound, ValidationFailed
from ..training.checkpoints import STATE_FILE, load_checkpoint_state
from ..training.planning import TRAIN_INSTALL_HINT, cuda_available, module_available

log = logging.getLogger(__name__)

#: Trainer-internal files that are never part of an exported adapter.
_TRAINER_ONLY = {"optimizer.pt", "scheduler.pt", "scaler.pt", "trainer_state.json", "training_args.bin"}
_SUFFIX = {"lora_adapter": "lora", "merged_16bit": "merged"}


def _is_trainer_only(name: str) -> bool:
    return name in _TRAINER_ONLY or (name.startswith("rng_state") and name.endswith(".pth"))


def _destination(request: ExportRequest, checkpoint: CheckpointInfo, default_output_root: str | Path) -> Path:
    if request.output_dir:
        return Path(request.output_dir)
    return Path(default_output_root) / "exports" / f"{checkpoint.checkpoint_id}-{_SUFFIX[request.format]}"


def _ensure_free(dest: Path) -> None:
    if dest.exists():
        if not dest.is_dir():
            raise Conflict(f"export destination '{dest}' exists and is not a directory")
        if any(dest.iterdir()):
            raise Conflict(f"export destination '{dest}' already exists and is not empty",
                           {"path": str(dest)})


def _list_files(root: Path) -> tuple[list[str], int]:
    files: list[str] = []
    total = 0
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            p = Path(dirpath) / n
            files.append(p.relative_to(root).as_posix())
            total += p.stat().st_size
    return sorted(files), total


def _export_adapter(src: Path, tmp: Path) -> None:
    if not any((src / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValidationFailed(f"checkpoint '{src}' contains no LoRA adapter weights")
    for item in src.iterdir():
        if item.is_file() and not _is_trainer_only(item.name) and not item.name.endswith(".tmp"):
            shutil.copy2(item, tmp / item.name)


def _export_merged(src: Path, tmp: Path, model_cfg: ModelConfig, backend: str | None) -> None:
    if backend == "unsloth" and cuda_available() and module_available("unsloth"):  # pragma: no cover - GPU only
        _export_merged_unsloth(src, tmp, model_cfg)
        return
    for m in ("torch", "transformers", "peft"):
        if not module_available(m):
            raise DependencyMissing(f"merged export needs {m}. Install with: {TRAIN_INSTALL_HINT}",
                                    {"missing": [m]})
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    out_dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    # Merge in float32 on CPU (fp16 matmuls on CPU are slow/imprecise); cast to 16-bit for saving.
    load_dtype = out_dtype if cuda else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, revision=model_cfg.revision,
                                                dtype=load_dtype, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, str(src), is_trainable=False)
    merged = model.merge_and_unload()
    merged = merged.to(out_dtype)
    merged.save_pretrained(str(tmp))
    tok_src = src if (src / "tokenizer_config.json").is_file() else model_cfg.model_name
    AutoTokenizer.from_pretrained(str(tok_src), trust_remote_code=False).save_pretrained(str(tmp))


def _export_merged_unsloth(src: Path, tmp: Path, model_cfg: ModelConfig) -> None:  # pragma: no cover - GPU only
    """UNVERIFIED: Unsloth's documented merged export."""
    import unsloth  # noqa: F401 - must precede transformers patches
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(src), max_seq_length=model_cfg.max_seq_length,
                                                         dtype=None, load_in_4bit=False)
    model.save_pretrained_merged(str(tmp), tokenizer, save_method="merged_16bit")


def export_model(request: ExportRequest, checkpoint: CheckpointInfo, model_cfg: ModelConfig,
                 default_output_root: str | Path) -> ExportResult:
    """Export ``checkpoint`` as ``request.format``; returns relative file names and total size.

    Raises ``NotFound`` (checkpoint missing), ``Conflict`` (non-empty destination),
    ``ValidationFailed`` (no adapter in checkpoint), ``DependencyMissing``.
    """
    src = Path(checkpoint.path)
    if not src.is_dir():
        raise NotFound(f"checkpoint directory '{src}' does not exist")
    if request.format not in _SUFFIX:
        raise ValidationFailed(f"unsupported export format {request.format!r}")
    dest = _destination(request, checkpoint, default_output_root)
    _ensure_free(dest)
    try:
        backend: Any = load_checkpoint_state(src).get("backend")
    except Exception:  # noqa: BLE001 - provenance is optional for export
        backend = None
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        if request.format == "lora_adapter":
            _export_adapter(src, tmp)
        else:
            _export_merged(src, tmp, model_cfg, backend)
            if (src / STATE_FILE).is_file():
                shutil.copy2(src / STATE_FILE, tmp / STATE_FILE)
        _ensure_free(dest)
        if dest.exists():
            dest.rmdir()  # empty (checked above)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    files, size = _list_files(dest)
    log.info("exported %s (%s, %d files, %d bytes)", dest, request.format, len(files), size)
    return ExportResult(path=str(dest.resolve()), format=request.format, files=files, size_bytes=size)
