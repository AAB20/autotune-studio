"""OOM detection, adaptive batch geometry and VRAM heuristics.

The policy here only changes *batch geometry* (micro-batch x gradient
accumulation). It never drops, skips or truncates samples: after an OOM the
trainer must resume from its last checkpoint and data cursor with the new
geometry, so every sample is still seen.

Owned by: gpu-memory-optimization.
"""

from __future__ import annotations

import gc
import logging
import math
import sys
from typing import Any

from ..errors import OutOfMemory
from ..interfaces import EventSink

log = logging.getLogger(__name__)

_OOM_MARKERS = ("out of memory", "cuda error: out of memory", "cuda_error_out_of_memory")


def _torch_oom_types() -> tuple[type[BaseException], ...]:
    """OOM exception classes from torch, only if torch is *already imported*.

    An exception raised by torch implies torch is loaded, so we never import it here.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return ()
    types: list[type[BaseException]] = []
    for owner, name in ((getattr(torch, "cuda", None), "OutOfMemoryError"), (torch, "OutOfMemoryError")):
        cls = getattr(owner, name, None) if owner is not None else None
        if isinstance(cls, type) and issubclass(cls, BaseException):
            types.append(cls)
    return tuple(types)


def is_oom_error(exc: BaseException | None) -> bool:
    """True if ``exc`` (or an exception in its ``__cause__``/``__context__`` chain) is an OOM.

    Matches ``torch.cuda.OutOfMemoryError`` / ``torch.OutOfMemoryError`` (when torch
    is loaded), any ``MemoryError`` (host RAM exhaustion), and ``RuntimeError``s whose
    message contains "out of memory" (e.g. "CUDA error: out of memory"), case-insensitive.
    """
    torch_types = _torch_oom_types()
    seen: set[int] = set()
    cur = exc
    while cur is not None and id(cur) not in seen and len(seen) < 16:
        seen.add(id(cur))
        if torch_types and isinstance(cur, torch_types):
            return True
        if isinstance(cur, MemoryError):
            return True
        if isinstance(cur, RuntimeError):
            msg = str(cur).lower()
            if any(m in msg for m in _OOM_MARKERS):
                return True
        cur = cur.__cause__ or cur.__context__
    return False


def release_cuda_memory() -> bool:
    """``gc.collect()`` then ``torch.cuda.empty_cache()`` if CUDA is in use.

    Safe without torch/CUDA. torch is not imported if it is not already loaded
    (no torch means no CUDA allocations to free). Returns True if the CUDA cache
    was emptied. Never raises.
    """
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return False
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception as exc:  # noqa: BLE001
            log.debug("ipc_collect failed: %s", exc)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("release_cuda_memory failed: %s", exc)
        return False


class OOMPolicy:
    """Halve the micro-batch on OOM while keeping the effective batch size.

    ``target = micro_batch * grad_accum`` is fixed at construction. Each
    :meth:`on_oom` sets ``micro = max(micro // 2, min_micro_batch)`` and
    ``accum = ceil(target / micro)``.

    Rounding: when ``target`` is not divisible by the new micro-batch (odd sizes),
    ``accum`` is rounded *up*, so the effective batch is the smallest multiple of
    ``micro`` that is >= the original (e.g. 5x3=15 -> 2x8=16 -> 1x15=15). It is
    always computed from the original target, so rounding never accumulates.

    When the micro-batch is already at ``min_micro_batch``, :meth:`on_oom` raises
    :class:`~autotune_studio.errors.OutOfMemory`. If ``events`` is given, every
    adaptation publishes a ``"log"`` event with level ``"warning"``.
    """

    def __init__(
        self,
        micro_batch: int,
        grad_accum: int,
        min_micro_batch: int = 1,
        events: EventSink | None = None,
    ) -> None:
        if micro_batch < 1 or grad_accum < 1 or min_micro_batch < 1:
            raise ValueError("micro_batch, grad_accum and min_micro_batch must be >= 1")
        self.micro_batch = int(micro_batch)
        self.grad_accum = int(grad_accum)
        self.min_micro_batch = int(min_micro_batch)
        self.target_effective_batch = self.micro_batch * self.grad_accum
        self.events = events
        self.history: list[tuple[int, int]] = [(self.micro_batch, self.grad_accum)]

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch * self.grad_accum

    @property
    def can_shrink(self) -> bool:
        return self.micro_batch > self.min_micro_batch

    def on_oom(self) -> tuple[int, int]:
        """Return the new ``(micro_batch, grad_accum)`` or raise ``OutOfMemory``."""
        old_micro, old_accum = self.micro_batch, self.grad_accum
        if not self.can_shrink:
            raise OutOfMemory(
                f"Out of memory at micro-batch {old_micro} (minimum {self.min_micro_batch}); "
                "reduce max_seq_length, enable 4-bit loading or use a smaller model.",
                detail={
                    "micro_batch": old_micro,
                    "grad_accum": old_accum,
                    "min_micro_batch": self.min_micro_batch,
                },
            )
        new_micro = max(old_micro // 2, self.min_micro_batch)
        new_accum = math.ceil(self.target_effective_batch / new_micro)
        self.micro_batch, self.grad_accum = new_micro, new_accum
        self.history.append((new_micro, new_accum))
        msg = (
            f"CUDA out of memory: micro-batch {old_micro}->{new_micro}, "
            f"grad accumulation {old_accum}->{new_accum} "
            f"(effective batch {old_micro * old_accum}->{new_micro * new_accum})"
        )
        log.warning(msg)
        if self.events is not None:
            data: dict[str, Any] = {
                "level": "warning",
                "source": "oom_policy",
                "message": msg,
                "old_micro_batch": old_micro,
                "old_grad_accum": old_accum,
                "new_micro_batch": new_micro,
                "new_grad_accum": new_accum,
                "effective_batch_size": new_micro * new_accum,
            }
            try:
                self.events.publish("log", data)
            except Exception as exc:  # noqa: BLE001 - monitoring must never break training
                log.debug("OOM log event publish failed: %s", exc)
        return new_micro, new_accum


# --------------------------------------------------------------------------- heuristic
CPU_DEFAULT_MICRO_BATCH = 1
MAX_SUGGESTED_MICRO_BATCH = 64
_RESERVE_BYTES = 512 * 1024**2  # CUDA context, fragmentation, LoRA/optimizer state
_USABLE_FRACTION = 0.85
_ACT_BYTES_COEFF = 8.0  # bytes per (token * sqrt(n_params)); calibrated loosely, see below


def suggest_micro_batch(
    free_vram_bytes: int | None,
    seq_len: int,
    n_params: int,
    bytes_per_param: float = 2.0,
    load_in_4bit: bool = False,
    weights_already_loaded: bool = False,
) -> int:
    """Rough starting micro-batch size. **This is an estimate, not a guarantee.**

    Model::

        weights     = n_params * (0.5 if load_in_4bit else bytes_per_param)   (skipped if already loaded)
        per_sample  = seq_len * sqrt(n_params) * 8 bytes                      (activations, grad-checkpointed)
        budget      = free_vram * 0.85 - weights - 512 MiB
        micro       = floor(budget / per_sample), rounded down to a power of two, clamped to [1, 64]

    ``sqrt(n_params)`` stands in for hidden-size x layer count; for a 7B model at
    2048 tokens it gives ~1.4 GB/sample, in the range seen with gradient
    checkpointing. Real usage depends on architecture, attention kernel and
    optimizer, so the OOM policy remains the safety net. Returns
    ``CPU_DEFAULT_MICRO_BATCH`` (1) when no VRAM figure is available (CPU).
    Always returns >= 1.
    """
    if not free_vram_bytes or free_vram_bytes <= 0:
        return CPU_DEFAULT_MICRO_BATCH
    seq_len = max(1, int(seq_len))
    n_params = max(1, int(n_params))
    weights = 0.0 if weights_already_loaded else n_params * (0.5 if load_in_4bit else float(bytes_per_param))
    per_sample = seq_len * math.sqrt(n_params) * _ACT_BYTES_COEFF
    budget = free_vram_bytes * _USABLE_FRACTION - weights - _RESERVE_BYTES
    if budget <= per_sample:
        return 1
    micro = int(budget // per_sample)
    micro = 1 << (micro.bit_length() - 1)  # round down to power of two
    return max(1, min(MAX_SUGGESTED_MICRO_BATCH, micro))
