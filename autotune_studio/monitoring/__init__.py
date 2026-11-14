"""Resource monitoring, metric history and OOM recovery policy.

Importing this package is light: torch / pynvml are imported only when a GPU
probe actually runs. Owned by: gpu-memory-optimization.
"""

from __future__ import annotations

import logging

from ..contracts import ResourceSnapshot
from . import gpu, memory

log = logging.getLogger(__name__)


class ResourceMonitor:
    """Merge host and GPU readings into a :class:`ResourceSnapshot`.

    Satisfies :class:`~autotune_studio.interfaces.ResourceMonitorProtocol`.
    :meth:`snapshot` never raises; GPU fields are ``None`` without CUDA.

    ``probe_gpu=False`` skips GPU probing entirely (avoids importing torch in a
    process that never trains).
    """

    def __init__(self, device: int = 0, probe_gpu: bool = True) -> None:
        self.device = device
        self.probe_gpu = probe_gpu

    def snapshot(self) -> ResourceSnapshot:
        try:
            used, total = memory.system_memory()
            cpu = memory.cpu_percent()
        except Exception:  # noqa: BLE001 - pragma: no cover - functions are already defensive
            used, total, cpu = 0, 0, 0.0
        vram_used: int | None = None
        vram_reserved: int | None = None
        vram_total: int | None = None
        util: float | None = None
        if self.probe_gpu:
            try:
                mem = gpu.gpu_memory(self.device)
                if mem is not None:
                    vram_used, vram_reserved, vram_total = mem["allocated"], mem["reserved"], mem["total"]
                    util = gpu.gpu_utilization(self.device)
            except Exception as exc:  # noqa: BLE001 - pragma: no cover - defensive
                log.debug("GPU probe failed: %s", exc)
        return ResourceSnapshot(
            cpu_pct=cpu,
            ram_used_bytes=used,
            ram_total_bytes=total,
            vram_used_bytes=vram_used,
            vram_reserved_bytes=vram_reserved,
            vram_total_bytes=vram_total,
            gpu_util_pct=util,
        )


from .metrics import EtaEstimator, MetricsStore, ResourceSampler
from .oom import OOMPolicy, is_oom_error, release_cuda_memory, suggest_micro_batch

__all__ = [
    "EtaEstimator",
    "MetricsStore",
    "OOMPolicy",
    "ResourceMonitor",
    "ResourceSampler",
    "is_oom_error",
    "release_cuda_memory",
    "suggest_micro_batch",
]
