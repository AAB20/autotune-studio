"""Resource probing: CPU-only snapshot, torch absence, fake CUDA/NVML paths."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types

import pytest

from autotune_studio.contracts import ResourceSnapshot
from autotune_studio.monitoring import ResourceMonitor, gpu, memory


def test_system_memory_and_cpu():
    used, total = memory.system_memory()
    assert total > 0 and 0 <= used <= total
    assert memory.cpu_percent() >= 0.0


def test_snapshot_on_cpu_machine():
    snap = ResourceMonitor().snapshot()
    assert isinstance(snap, ResourceSnapshot)
    assert snap.ram_total_bytes > 0
    if not gpu.cuda_available():  # this dev box has no GPU
        assert snap.vram_used_bytes is None
        assert snap.vram_reserved_bytes is None
        assert snap.vram_total_bytes is None
        assert snap.gpu_util_pct is None
        assert gpu.gpu_memory() is None
        assert gpu.free_vram_bytes() is None
    assert snap.model_dump()["ram_total_bytes"] == snap.ram_total_bytes


def test_snapshot_without_gpu_probe():
    assert ResourceMonitor(probe_gpu=False).snapshot().vram_total_bytes is None


def test_torch_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` now raises ImportError
    mod = importlib.reload(gpu)
    try:
        assert mod._torch() is None
        assert mod.cuda_available() is False
        assert mod.gpu_memory() is None
        assert mod.free_vram_bytes() is None
        snap = ResourceMonitor().snapshot()
        assert snap.vram_used_bytes is None and snap.gpu_util_pct is None

        from autotune_studio.monitoring import oom

        assert oom.release_cuda_memory() is False
        assert oom.is_oom_error(RuntimeError("CUDA error: out of memory"))
        assert not oom.is_oom_error(ValueError("x"))
    finally:
        monkeypatch.undo()
        importlib.reload(gpu)


def test_package_import_is_light():
    code = (
        "import sys, autotune_studio.monitoring as m\n"
        "m.ResourceMonitor(probe_gpu=False).snapshot()\n"
        "print([x for x in ('torch', 'pynvml') if x in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


class _FakeCuda:
    def is_available(self):
        return True

    def memory_allocated(self, d):
        return 100

    def memory_reserved(self, d):
        return 200

    def get_device_properties(self, d):
        return types.SimpleNamespace(total_memory=1000)

    def mem_get_info(self, d):
        return (700, 1000)


def test_fake_cuda_path(monkeypatch):
    monkeypatch.setattr(gpu, "_torch", lambda: types.SimpleNamespace(cuda=_FakeCuda()))
    monkeypatch.setattr(gpu, "gpu_utilization", lambda device=0: 42.0)
    assert gpu.cuda_available() is True
    assert gpu.gpu_memory() == {"allocated": 100, "reserved": 200, "total": 1000}
    assert gpu.free_vram_bytes() == 700
    snap = ResourceMonitor().snapshot()
    assert (snap.vram_used_bytes, snap.vram_reserved_bytes, snap.vram_total_bytes) == (100, 200, 1000)
    assert snap.gpu_util_pct == 42.0


def test_fake_cuda_errors_degrade(monkeypatch):
    class Broken(_FakeCuda):
        def memory_allocated(self, d):
            raise RuntimeError("driver gone")

        def mem_get_info(self, d):
            raise RuntimeError("driver gone")

    monkeypatch.setattr(gpu, "_torch", lambda: types.SimpleNamespace(cuda=Broken()))
    assert gpu.gpu_memory() is None
    assert gpu.free_vram_bytes() is None
    assert ResourceMonitor().snapshot().vram_total_bytes is None


@pytest.fixture
def fresh_nvml():
    gpu._nvml.reset()
    yield
    gpu._nvml.reset()


def test_nvml_absent(fresh_nvml, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert gpu.nvml_available() is False
    assert gpu.gpu_utilization() is None


def test_nvml_fake_init_once(fresh_nvml, monkeypatch):
    calls = {"init": 0}
    fake = types.ModuleType("pynvml")

    def nvmlInit():
        calls["init"] += 1

    fake.nvmlInit = nvmlInit
    fake.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"
    fake.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=73, memory=10)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert gpu.gpu_utilization(0) == 73.0
    assert gpu.gpu_utilization(0) == 73.0
    assert calls["init"] == 1


def test_nvml_init_failure_not_retried(fresh_nvml, monkeypatch):
    calls = {"init": 0}
    fake = types.ModuleType("pynvml")

    def nvmlInit():
        calls["init"] += 1
        raise RuntimeError("NVML Shared Library Not Found")

    fake.nvmlInit = nvmlInit
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert gpu.gpu_utilization() is None
    assert gpu.gpu_utilization() is None
    assert calls["init"] == 1
