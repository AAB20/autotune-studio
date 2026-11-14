"""OOM detection, OOMPolicy geometry, release_cuda_memory, suggest_micro_batch."""

from __future__ import annotations

import pytest

from autotune_studio.errors import OutOfMemory
from autotune_studio.monitoring import OOMPolicy, is_oom_error, release_cuda_memory, suggest_micro_batch
from autotune_studio.monitoring.oom import CPU_DEFAULT_MICRO_BATCH, MAX_SUGGESTED_MICRO_BATCH

GiB = 1024**3


# --------------------------------------------------------------------------- is_oom_error
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA error: out of memory"),
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("[enforce fail] DefaultCPUAllocator: Out Of Memory"),
        MemoryError(),
        MemoryError("Unable to allocate 8 GiB"),
    ],
)
def test_is_oom_true(exc):
    assert is_oom_error(exc)


@pytest.mark.parametrize(
    "exc",
    [None, RuntimeError("shape mismatch"), ValueError("out of memory"), KeyError("x"), OSError("disk full")],
)
def test_is_oom_false(exc):
    assert not is_oom_error(exc)


def test_is_oom_chained():
    try:
        try:
            raise RuntimeError("CUDA error: out of memory")
        except RuntimeError as inner:
            raise ValueError("trainer step failed") from inner
    except ValueError as outer:
        assert is_oom_error(outer)


def test_is_oom_torch_type():
    torch = pytest.importorskip("torch")
    cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if cls is None:
        pytest.skip("torch build without OutOfMemoryError")
    assert is_oom_error(cls("boom"))  # matched by type even without marker text


def test_release_cuda_memory_safe_on_cpu():
    assert release_cuda_memory() in (True, False)


# --------------------------------------------------------------------------- OOMPolicy
def test_policy_sequence_8x4():
    p = OOMPolicy(8, 4)
    assert p.on_oom() == (4, 8)
    assert p.on_oom() == (2, 16)
    assert p.on_oom() == (1, 32)
    assert p.effective_batch_size == 32
    assert not p.can_shrink
    with pytest.raises(OutOfMemory) as ei:
        p.on_oom()
    assert ei.value.code == "out_of_memory"
    assert ei.value.detail["micro_batch"] == 1
    assert p.history == [(8, 4), (4, 8), (2, 16), (1, 32)]


@pytest.mark.parametrize(
    "micro,accum,expected",
    [
        (5, 3, [(2, 8), (1, 15)]),  # 15 -> 16 (rounded up) -> 15
        (3, 4, [(1, 12)]),
        (7, 3, [(3, 7), (1, 21)]),
        (5, 2, [(2, 5), (1, 10)]),
        (6, 1, [(3, 2), (1, 6)]),
    ],
)
def test_policy_odd_sizes(micro, accum, expected):
    p = OOMPolicy(micro, accum)
    target = micro * accum
    got = []
    while p.can_shrink:
        m, a = p.on_oom()
        got.append((m, a))
        assert m * a >= target  # never below the original effective batch
        assert m * a < target + m  # smallest multiple of m that is >= target
    assert got == expected
    with pytest.raises(OutOfMemory):
        p.on_oom()


def test_policy_min_micro_batch():
    p = OOMPolicy(8, 1, min_micro_batch=2)
    assert p.on_oom() == (4, 2)
    assert p.on_oom() == (2, 4)
    with pytest.raises(OutOfMemory):
        p.on_oom()


def test_policy_starting_at_one_raises():
    with pytest.raises(OutOfMemory):
        OOMPolicy(1, 8).on_oom()


def test_policy_rejects_invalid():
    with pytest.raises(ValueError):
        OOMPolicy(0, 1)


def test_policy_publishes_log_warning():
    class Sink:
        def __init__(self):
            self.events = []

        def publish(self, type, data):
            self.events.append((type, data))

    sink = Sink()
    OOMPolicy(4, 2, events=sink).on_oom()
    typ, data = sink.events[0]
    assert typ == "log" and data["level"] == "warning"
    assert (data["old_micro_batch"], data["new_micro_batch"]) == (4, 2)
    assert (data["old_grad_accum"], data["new_grad_accum"]) == (2, 4)


# --------------------------------------------------------------------------- suggest_micro_batch
def test_suggest_cpu_default():
    assert suggest_micro_batch(None, 2048, 7_000_000_000) == CPU_DEFAULT_MICRO_BATCH
    assert suggest_micro_batch(0, 2048, 7_000_000_000) == CPU_DEFAULT_MICRO_BATCH


def test_suggest_bounds_and_monotonic():
    assert suggest_micro_batch(1 * GiB, 4096, 70_000_000_000) == 1  # never 0
    assert suggest_micro_batch(80 * GiB, 128, 1_000_000) == MAX_SUGGESTED_MICRO_BATCH  # capped
    small = suggest_micro_batch(16 * GiB, 2048, 1_000_000_000)
    big = suggest_micro_batch(40 * GiB, 2048, 1_000_000_000)
    assert 1 <= small <= big <= MAX_SUGGESTED_MICRO_BATCH
    assert suggest_micro_batch(16 * GiB, 4096, 1_000_000_000) <= small  # longer seq never more
    assert suggest_micro_batch(16 * GiB, 1024, 7_000_000_000, load_in_4bit=True) >= suggest_micro_batch(
        16 * GiB, 1024, 7_000_000_000
    )
    v = suggest_micro_batch(24 * GiB, 1024, 1_000_000_000)
    assert v & (v - 1) == 0  # power of two


def test_suggest_weights_already_loaded():
    a = suggest_micro_batch(20 * GiB, 2048, 7_000_000_000, weights_already_loaded=True)
    b = suggest_micro_batch(20 * GiB, 2048, 7_000_000_000)
    assert a >= b >= 1
