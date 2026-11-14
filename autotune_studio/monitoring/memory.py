"""Host RAM / CPU sampling via psutil. Never raises: failures degrade to zeros.

Owned by: gpu-memory-optimization.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:  # psutil is a core dependency, but monitoring must never crash training.
    import psutil as _psutil
except Exception:  # noqa: BLE001 - pragma: no cover - exercised only on broken installs
    _psutil = None  # type: ignore[assignment]


def system_memory() -> tuple[int, int]:
    """Return ``(used_bytes, total_bytes)`` of host RAM.

    ``used`` is ``total - available`` (memory unavailable to new allocations),
    which is more meaningful across platforms than psutil's ``used`` field.
    Returns ``(0, 0)`` if psutil is unavailable or fails.
    """
    if _psutil is None:
        return 0, 0
    try:
        vm = _psutil.virtual_memory()
        total = int(vm.total)
        return max(0, total - int(vm.available)), total
    except Exception as exc:  # noqa: BLE001 - pragma: no cover - defensive
        log.debug("system_memory failed: %s", exc)
        return 0, 0


def cpu_percent() -> float:
    """System-wide CPU utilisation in percent since the previous call (non-blocking).

    Uses ``psutil.cpu_percent(interval=None)``; the first call in a process
    returns 0.0 because psutil has no previous reference point. 0.0 on failure.
    """
    if _psutil is None:
        return 0.0
    try:
        return float(_psutil.cpu_percent(interval=None))
    except Exception as exc:  # noqa: BLE001 - pragma: no cover - defensive
        log.debug("cpu_percent failed: %s", exc)
        return 0.0
