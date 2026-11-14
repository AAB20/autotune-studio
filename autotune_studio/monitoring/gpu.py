"""GPU / VRAM probing. torch and pynvml are imported lazily and are optional.

Every public function returns ``None``/``False`` instead of raising when CUDA,
torch or NVML are missing or broken (e.g. a torch build whose DLLs fail to load).

Owned by: gpu-memory-optimization.
"""

from __future__ import annotations

import importlib
import logging
import threading
from types import ModuleType
from typing import Any

log = logging.getLogger(__name__)


def _torch() -> ModuleType | None:
    """Import torch on demand; ``None`` if absent or broken.

    Deliberately not cached: re-importing a loaded module is a dict lookup, and
    not caching lets tests simulate absence through ``sys.modules``.
    """
    try:
        return importlib.import_module("torch")
    except Exception:  # noqa: BLE001 - ImportError, OSError (DLL load), ...
        return None


def cuda_available() -> bool:
    """True if torch is importable and reports a usable CUDA device."""
    torch = _torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def gpu_memory(device: int = 0) -> dict[str, int] | None:
    """Return ``{"allocated", "reserved", "total"}`` in bytes for ``device``, or None without CUDA.

    ``allocated``/``reserved`` come from PyTorch's caching allocator and cover
    this process only; ``total`` is the device's physical memory.
    """
    if not cuda_available():
        return None
    torch = _torch()
    try:
        return {
            "allocated": int(torch.cuda.memory_allocated(device)),  # type: ignore[union-attr]
            "reserved": int(torch.cuda.memory_reserved(device)),  # type: ignore[union-attr]
            "total": int(torch.cuda.get_device_properties(device).total_memory),  # type: ignore[union-attr]
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("gpu_memory(%s) failed: %s", device, exc)
        return None


def free_vram_bytes(device: int = 0) -> int | None:
    """Free device memory in bytes as reported by the driver (all processes), or None without CUDA."""
    if not cuda_available():
        return None
    torch = _torch()
    try:
        free, _total = torch.cuda.mem_get_info(device)  # type: ignore[union-attr]
        return int(free)
    except Exception as exc:  # noqa: BLE001
        log.debug("free_vram_bytes(%s) failed: %s", device, exc)
        return None


# --------------------------------------------------------------------------- NVML
class _Nvml:
    """Initialise NVML at most once per process; remember failure and stop retrying."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempted = False
        self._mod: Any = None

    def module(self) -> Any:
        with self._lock:
            if not self._attempted:
                self._attempted = True
                try:
                    mod = importlib.import_module("pynvml")
                    mod.nvmlInit()
                    self._mod = mod
                except Exception as exc:  # noqa: BLE001
                    log.debug("NVML unavailable: %s", exc)
                    self._mod = None
            return self._mod

    def reset(self) -> None:
        """Forget the cached init result (for tests)."""
        with self._lock:
            self._attempted = False
            self._mod = None


_nvml = _Nvml()


def nvml_available() -> bool:
    """True if pynvml imported and ``nvmlInit()`` succeeded (attempted once per process)."""
    return _nvml.module() is not None


def gpu_utilization(device: int = 0) -> float | None:
    """GPU core utilisation percent from NVML, or None if NVML is unavailable.

    NVML indexes physical devices; under ``CUDA_VISIBLE_DEVICES`` remapping the
    index may differ from the torch device index.
    """
    mod = _nvml.module()
    if mod is None:
        return None
    try:
        handle = mod.nvmlDeviceGetHandleByIndex(device)
        return float(mod.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception as exc:  # noqa: BLE001
        log.debug("gpu_utilization(%s) failed: %s", device, exc)
        return None
