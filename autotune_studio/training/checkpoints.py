"""Checkpoint metadata: ``<output_dir>/checkpoint-<step>/autotune_state.json``.

Layout written by the training engine (one directory per optimizer step saved)::

    <runs_root>/<job_id>/checkpoint-<step>/
        adapter_config.json, adapter_model.safetensors     (PEFT save_pretrained)
        tokenizer files                                    (tokenizer.save_pretrained)
        optimizer.pt, scheduler.pt, rng_state.pth,
        trainer_state.json, training_args.bin               (HF Trainer, used for exact resume)
        autotune_state.json                                 (this module)

``autotune_state.json`` keys: ``checkpoint_id, job_id, step, created_at, data_cursor,
model_config, training_config, is_final, backend, micro_batch_size, gradient_accumulation_steps``.

Light module: no heavy imports. Owned by: llm-training.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from ..contracts import CheckpointInfo, ModelConfig, TrainingConfig
from ..errors import NotFound, ValidationFailed

log = logging.getLogger(__name__)

STATE_FILE = "autotune_state.json"
CHECKPOINT_PREFIX = "checkpoint-"
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_id_for(job_id: str | None, step: int) -> str:
    """Stable, filesystem-safe checkpoint id: ``<job_id>-checkpoint-<step>``."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", job_id or "job")
    return f"{safe}-{CHECKPOINT_PREFIX}{int(step)}"


def dir_size(path: str | Path) -> int:
    """Total size in bytes of all regular files below ``path``."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def write_checkpoint_state(
    ckpt_dir: str | Path,
    *,
    job_id: str,
    step: int,
    data_cursor: dict[str, Any] | None,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    is_final: bool,
    backend: str | None = None,
    extra: dict[str, Any] | None = None,
) -> CheckpointInfo:
    """Write ``autotune_state.json`` atomically into ``ckpt_dir`` and return its :class:`CheckpointInfo`."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "checkpoint_id": checkpoint_id_for(job_id, step),
        "job_id": job_id,
        "step": int(step),
        "created_at": time.time(),
        "data_cursor": data_cursor,
        "model_config": model_cfg.model_dump(mode="json"),
        "training_config": train_cfg.model_dump(mode="json"),
        "is_final": bool(is_final),
        "backend": backend,
    }
    if extra:
        state.update(extra)
    tmp = ckpt_dir / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, ckpt_dir / STATE_FILE)
    return load_checkpoint_info(ckpt_dir)


def load_checkpoint_state(path: str | Path) -> dict[str, Any]:
    """Parsed ``autotune_state.json`` of checkpoint directory ``path`` (or of the file itself).

    Raises ``NotFound`` if missing, ``ValidationFailed`` if unreadable.
    """
    p = Path(path)
    f = p if p.name == STATE_FILE else p / STATE_FILE
    if not f.is_file():
        raise NotFound(f"no {STATE_FILE} in checkpoint '{p}'")
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValidationFailed(f"unreadable checkpoint state {f}: {e}") from e
    if not isinstance(data, dict) or "step" not in data:
        raise ValidationFailed(f"invalid checkpoint state {f}")
    return data


def load_checkpoint_info(path: str | Path) -> CheckpointInfo:
    """:class:`CheckpointInfo` for checkpoint directory ``path``; ``size_bytes`` is measured on disk."""
    p = Path(path)
    if p.name == STATE_FILE:
        p = p.parent
    state = load_checkpoint_state(p)
    step = int(state["step"])
    return CheckpointInfo(
        checkpoint_id=str(state.get("checkpoint_id") or checkpoint_id_for(state.get("job_id"), step)),
        path=str(p.resolve()),
        step=step,
        created_at=float(state.get("created_at") or p.stat().st_mtime),
        size_bytes=dir_size(p),
        job_id=state.get("job_id"),
        is_final=bool(state.get("is_final", False)),
        data_cursor=state.get("data_cursor"),
    )


def _candidate_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        return out
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if _CKPT_RE.match(child.name):  # root is itself a job dir
            out.append(child)
            continue
        try:
            for sub in child.iterdir():  # root/<job_id>/checkpoint-*
                if sub.is_dir() and _CKPT_RE.match(sub.name):
                    out.append(sub)
        except OSError:
            continue
    return out


def list_checkpoints(output_root: str | Path) -> list[CheckpointInfo]:
    """All checkpoints under ``<output_root>/<job_id>/checkpoint-*/`` (and ``<output_root>/checkpoint-*/``)
    that contain ``autotune_state.json``, sorted by ``created_at`` (then step). Unreadable ones are skipped."""
    infos: list[CheckpointInfo] = []
    for d in _candidate_dirs(Path(output_root)):
        if not (d / STATE_FILE).is_file():
            continue
        try:
            infos.append(load_checkpoint_info(d))
        except Exception as e:  # noqa: BLE001 - one corrupt checkpoint must not hide the others
            log.warning("skipping unreadable checkpoint %s: %s", d, e)
    infos.sort(key=lambda c: (c.created_at, c.step))
    return infos


def latest_checkpoint(output_dir: str | Path, job_id: str | None = None) -> CheckpointInfo | None:
    """Checkpoint with the highest step directly under ``output_dir`` (optionally of ``job_id``)."""
    best: CheckpointInfo | None = None
    root = Path(output_dir)
    if not root.is_dir():
        return None
    for d in root.iterdir():
        if d.is_dir() and _CKPT_RE.match(d.name) and (d / STATE_FILE).is_file():
            try:
                info = load_checkpoint_info(d)
            except Exception as e:  # noqa: BLE001
                log.debug("skipping unreadable checkpoint %s: %s", d, e)
                continue
            if job_id is not None and info.job_id != job_id:
                continue
            if best is None or info.step > best.step:
                best = info
    return best
