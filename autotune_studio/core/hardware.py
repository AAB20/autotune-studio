"""Hardware detection (CPU, RAM, disk, CUDA GPUs, NVML utilisation).

Owned by: runtime-compatibility. Skill: ``notebook-runtime-detection``.

:func:`detect_hardware` never raises. ``torch`` and ``pynvml`` are imported lazily
and are optional: without torch ``cuda_available`` is False (GPUs may still be
listed from NVML), without pynvml ``nvml_available`` is False and utilisation is
``None``. Because :class:`~autotune_studio.contracts.HardwareInfo` has no notes
field, callers may pass a list as ``notes`` to receive human-readable notes about
probes that failed or were skipped.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from ..contracts import GPUInfo, HardwareInfo

__all__ = ["detect_hardware"]

log = logging.getLogger(__name__)


def _existing_ancestor(path: str) -> str:
    """Nearest existing directory for ``path`` (disk_usage needs an existing path)."""
    p = Path(path).expanduser()
    try:
        p = p.resolve(strict=False)
    except (OSError, RuntimeError):
        p = p.absolute()
    for cand in (p, *p.parents):
        try:
            if cand.exists():
                return str(cand)
        except OSError:
            continue
    return str(p.anchor or os.sep)


def _cpu_ram(notes: list[str]) -> tuple[int, int, int]:
    cpu = os.cpu_count() or 1
    try:
        import psutil

        cpu = psutil.cpu_count(logical=True) or cpu
        vm = psutil.virtual_memory()
        return int(cpu), int(vm.total), int(vm.available)
    except Exception as exc:
        notes.append(f"RAM detection unavailable ({type(exc).__name__}); install psutil.")
        return int(cpu), 0, 0


def _disk(path: str, notes: list[str]) -> tuple[str, int, int]:
    target = _existing_ancestor(path)
    try:
        du = shutil.disk_usage(target)
        return target, int(du.total), int(du.free)
    except Exception as exc:
        notes.append(f"Disk usage for '{target}' unavailable ({type(exc).__name__}).")
        return target, 0, 0


def _visible_device_map(n_torch: int) -> list[int | None]:
    """Map torch device index -> NVML physical index using CUDA_VISIBLE_DEVICES (numeric only)."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return list(range(n_torch))
    out: list[int | None] = []
    for part in raw.split(","):
        part = part.strip()
        out.append(int(part) if part.isdigit() else None)  # UUID/MIG ids: no mapping
    out += [None] * max(0, n_torch - len(out))
    return out[:n_torch]


class _Nvml:
    """Thin lazy wrapper around pynvml; all calls swallow errors."""

    def __init__(self, notes: list[str]) -> None:
        self.mod: Any = None
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self.mod = pynvml
        except ImportError:
            pass  # not installed: silent, nvml_available=False
        except Exception as exc:
            notes.append(f"NVML present but initialisation failed ({type(exc).__name__}); no GPU utilisation.")

    @property
    def ok(self) -> bool:
        return self.mod is not None

    def count(self) -> int:
        try:
            return int(self.mod.nvmlDeviceGetCount())
        except Exception:
            return 0

    def query(self, idx: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        m = self.mod
        try:
            h = m.nvmlDeviceGetHandleByIndex(idx)
        except Exception:
            return out
        for key, fn in (
            ("name", lambda: m.nvmlDeviceGetName(h)),
            ("mem", lambda: m.nvmlDeviceGetMemoryInfo(h)),
            ("util", lambda: m.nvmlDeviceGetUtilizationRates(h).gpu),
            ("cc", lambda: m.nvmlDeviceGetCudaComputeCapability(h)),
        ):
            try:
                out[key] = fn()
            except Exception:
                pass
        if isinstance(out.get("name"), bytes):
            out["name"] = out["name"].decode(errors="replace")
        return out

    def close(self) -> None:
        try:
            if self.mod is not None:
                self.mod.nvmlShutdown()
        except Exception:
            pass


def _torch_info(notes: list[str]) -> dict[str, Any]:
    info: dict[str, Any] = {"torch_version": None, "cuda_available": False, "cuda_version": None,
                            "bf16": False, "devices": []}
    try:
        import torch
    except Exception as exc:  # ImportError, or sys.modules['torch'] = None, or broken install
        notes.append(f"PyTorch not importable ({type(exc).__name__}); GPU detection via torch skipped.")
        return info
    info["torch_version"] = str(getattr(torch, "__version__", None) or "") or None
    try:
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        notes.append(f"torch.cuda.is_available() failed ({type(exc).__name__}).")
        return info
    if not info["cuda_available"]:
        if info["cuda_version"] is None:
            notes.append("PyTorch is a CPU-only build; training will run on CPU (very slow).")
        else:
            notes.append("PyTorch has CUDA support but no CUDA device is available.")
        return info
    try:
        info["bf16"] = bool(torch.cuda.is_bf16_supported())
    except Exception:
        info["bf16"] = False
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 0
    for i in range(n):
        d: dict[str, Any] = {"index": i}
        try:
            props = torch.cuda.get_device_properties(i)
            d["name"] = str(props.name)
            d["total"] = int(props.total_memory)
            d["cc"] = f"{props.major}.{props.minor}"
        except Exception as exc:
            notes.append(f"Could not read properties of cuda:{i} ({type(exc).__name__}).")
            d.setdefault("name", f"cuda:{i}")
            d.setdefault("total", 0)
        info["devices"].append(d)
    return info


def _torch_free_vram(index: int) -> int | None:
    """Free VRAM via torch (creates a CUDA context on that device; used only without NVML)."""
    try:
        import torch

        free, _total = torch.cuda.mem_get_info(index)
        return int(free)
    except Exception:
        return None


def detect_hardware(path: str | None = None, notes: list[str] | None = None) -> HardwareInfo:
    """Detect hardware. ``path`` selects the disk to measure (default: CWD). Never raises.

    If ``notes`` is given, human-readable notes about skipped/failed probes are appended to it.
    """
    notes = notes if notes is not None else []
    try:
        cpu, ram_total, ram_avail = _cpu_ram(notes)
        try:
            base = path if path else os.getcwd()
        except OSError:
            base = os.sep
        disk_path, disk_total, disk_free = _disk(base, notes)

        t = _torch_info(notes)
        nvml = _Nvml(notes)
        gpus: list[GPUInfo] = []
        try:
            if t["devices"]:
                phys = _visible_device_map(len(t["devices"]))
                for d, p in zip(t["devices"], phys):
                    q = nvml.query(p) if (nvml.ok and p is not None) else {}
                    mem = q.get("mem")
                    free = int(mem.free) if mem is not None else _torch_free_vram(d["index"])
                    util = q.get("util")
                    gpus.append(GPUInfo(
                        index=d["index"], name=d.get("name", f"cuda:{d['index']}"),
                        total_vram_bytes=int(d.get("total", 0)), free_vram_bytes=free,
                        utilization_pct=float(util) if util is not None else None,
                        compute_capability=d.get("cc"),
                    ))
            elif nvml.ok:
                # GPUs visible to the driver but not usable through torch (missing/CPU torch).
                for i in range(nvml.count()):
                    q = nvml.query(i)
                    mem = q.get("mem")
                    cc = q.get("cc")
                    gpus.append(GPUInfo(
                        index=i, name=str(q.get("name", f"gpu{i}")),
                        total_vram_bytes=int(mem.total) if mem is not None else 0,
                        free_vram_bytes=int(mem.free) if mem is not None else None,
                        utilization_pct=float(q["util"]) if "util" in q else None,
                        compute_capability=f"{cc[0]}.{cc[1]}" if cc else None,
                    ))
                if gpus:
                    notes.append("NVIDIA GPU(s) detected by NVML but not usable from PyTorch; "
                                 "install a CUDA-enabled torch build.")
        finally:
            nvml_ok = nvml.ok
            nvml.close()

        return HardwareInfo(
            cpu_count=cpu, ram_total_bytes=ram_total, ram_available_bytes=ram_avail,
            disk_path=disk_path, disk_total_bytes=disk_total, disk_free_bytes=disk_free,
            cuda_available=bool(t["cuda_available"]), cuda_version=t["cuda_version"],
            torch_version=t["torch_version"], bf16_supported=bool(t["bf16"]),
            gpus=gpus, nvml_available=nvml_ok, notes=list(notes),
        )
    except Exception as exc:  # pragma: no cover - last-resort safety net
        log.debug("detect_hardware failed", exc_info=True)
        notes.append(f"Hardware detection failed ({type(exc).__name__}); reporting minimal values.")
        return HardwareInfo(
            cpu_count=os.cpu_count() or 1, ram_total_bytes=0, ram_available_bytes=0,
            disk_path=str(path or "."), disk_total_bytes=0, disk_free_bytes=0, cuda_available=False,
            notes=list(notes),
        )
