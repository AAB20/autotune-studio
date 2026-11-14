"""Hardware detection with torch absent, fake CUDA torch + NVML, and on the real machine."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest

from autotune_studio.core.hardware import detect_hardware


def test_real_machine(tmp_path: Path) -> None:
    notes: list[str] = []
    hw = detect_hardware(str(tmp_path), notes=notes)
    assert hw.cpu_count >= 1
    assert hw.ram_total_bytes > 0 and 0 < hw.ram_available_bytes <= hw.ram_total_bytes
    assert hw.disk_total_bytes > 0 and hw.disk_free_bytes > 0
    assert Path(hw.disk_path) == tmp_path.resolve()
    # This dev box: CPU-only torch, no NVML.
    assert hw.torch_version is not None
    assert hw.cuda_available is False and hw.gpus == []
    assert hw.nvml_available is False


def test_torch_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    monkeypatch.setitem(sys.modules, "pynvml", None)
    notes: list[str] = []
    hw = detect_hardware(str(tmp_path), notes=notes)
    assert hw.torch_version is None and hw.cuda_available is False and hw.cuda_version is None
    assert hw.gpus == [] and hw.bf16_supported is False
    assert any("PyTorch not importable" in n for n in notes)


def test_torch_import_raises_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def bad_import(name, *a, **k):  # type: ignore[no-untyped-def]
        if name == "torch":
            raise OSError("broken CUDA libs")
        return real_import(name, *a, **k)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", bad_import)
    hw = detect_hardware()
    assert hw.cuda_available is False and hw.torch_version is None


def test_nonexistent_path_uses_existing_ancestor(tmp_path: Path) -> None:
    hw = detect_hardware(str(tmp_path / "does" / "not" / "exist"))
    assert Path(hw.disk_path) == tmp_path.resolve() and hw.disk_total_bytes > 0


def _fake_torch(n: int = 2) -> types.ModuleType:
    t = types.ModuleType("torch")
    t.__version__ = "9.9.9+cu999"  # type: ignore[attr-defined]
    t.version = types.SimpleNamespace(cuda="12.8")  # type: ignore[attr-defined]
    props = [types.SimpleNamespace(name=f"Fake GPU {i}", total_memory=(16 + i) * 2**30, major=8, minor=i)
             for i in range(n)]
    t.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        device_count=lambda: n,
        get_device_properties=lambda i: props[i],
        mem_get_info=lambda i: (10 * 2**30, props[i].total_memory),
    )
    return t


def _fake_pynvml() -> types.ModuleType:
    m = types.ModuleType("pynvml")
    m.nvmlInit = lambda: None  # type: ignore[attr-defined]
    m.nvmlShutdown = lambda: None  # type: ignore[attr-defined]
    m.nvmlDeviceGetCount = lambda: 2  # type: ignore[attr-defined]
    m.nvmlDeviceGetHandleByIndex = lambda i: i  # type: ignore[attr-defined]
    m.nvmlDeviceGetName = lambda h: b"NVML GPU"  # type: ignore[attr-defined]
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(  # type: ignore[attr-defined]
        total=24 * 2**30, free=(5 + h) * 2**30)
    m.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=40 + h)  # type: ignore[attr-defined]
    m.nvmlDeviceGetCudaComputeCapability = lambda h: (8, 6)  # type: ignore[attr-defined]
    return m


def test_fake_cuda_without_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(2))
    monkeypatch.setitem(sys.modules, "pynvml", None)
    hw = detect_hardware()
    assert hw.cuda_available and hw.cuda_version == "12.8" and hw.bf16_supported
    assert hw.torch_version == "9.9.9+cu999" and hw.nvml_available is False
    assert [g.name for g in hw.gpus] == ["Fake GPU 0", "Fake GPU 1"]
    assert hw.gpus[1].total_vram_bytes == 17 * 2**30
    assert hw.gpus[0].free_vram_bytes == 10 * 2**30  # torch.cuda.mem_get_info fallback
    assert hw.gpus[1].compute_capability == "8.1" and hw.gpus[0].utilization_pct is None


def test_fake_cuda_with_nvml_and_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(1))
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    hw = detect_hardware()
    assert hw.nvml_available is True and len(hw.gpus) == 1
    g = hw.gpus[0]
    assert g.index == 0 and g.free_vram_bytes == 6 * 2**30 and g.utilization_pct == 41.0  # physical GPU 1


def test_nvml_only_gpus_when_torch_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())
    notes: list[str] = []
    hw = detect_hardware(notes=notes)
    assert hw.cuda_available is False and hw.nvml_available is True
    assert len(hw.gpus) == 2 and hw.gpus[0].name == "NVML GPU" and hw.gpus[0].compute_capability == "8.6"
    assert any("not usable from PyTorch" in n for n in notes)


def test_nvml_init_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _fake_pynvml()

    def boom() -> None:
        raise RuntimeError("NVML_ERROR_DRIVER_NOT_LOADED")

    m.nvmlInit = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", m)
    notes: list[str] = []
    hw = detect_hardware(notes=notes)
    assert hw.nvml_available is False
    assert any("NVML present but initialisation failed" in n for n in notes)
