"""Dynamic batching, bounded shuffle, memory budget and backpressure buffer."""

from __future__ import annotations

import threading
import time

import pytest

from autotune_studio.data_engine import (
    BoundedBuffer,
    BufferClosed,
    MemoryBudget,
    batch_by_bytes,
    batch_by_tokens,
    shuffle_buffer,
)
from autotune_studio.data_engine.memory_manager import MiB


def test_batch_by_bytes_never_drops_and_respects_budget() -> None:
    items = ["x" * n for n in (10, 20, 30, 500, 5, 5)]
    batches = list(batch_by_bytes(items, max_bytes=60, size_fn=len))
    assert [x for b in batches for x in b] == items
    assert all(sum(map(len, b)) <= 60 or len(b) == 1 for b in batches)
    assert ["x" * 500] in batches  # oversize item alone
    assert list(batch_by_bytes([], 10)) == []
    with pytest.raises(ValueError):
        list(batch_by_bytes(items, 0))


def test_batch_by_bytes_is_lazy_over_unbounded_input() -> None:
    def endless():
        i = 0
        while True:
            yield {"i": i}
            i += 1

    gen = batch_by_bytes(endless(), max_bytes=1000)
    first, second = next(gen), next(gen)
    assert first[0]["i"] == 0 and second[0]["i"] == len(first)


def test_batch_by_tokens_padded_cost() -> None:
    lengths = [10, 10, 10, 40, 5, 5, 200, 1]
    batches = list(batch_by_tokens(lengths, length_fn=lambda n: n, max_tokens=60))
    assert [x for b in batches for x in b] == lengths
    for b in batches:
        assert max(b) * len(b) <= 60 or len(b) == 1
    capped = list(batch_by_tokens([1] * 10, length_fn=lambda n: n, max_tokens=1000, max_batch=3))
    assert [len(b) for b in capped] == [3, 3, 3, 1]


def test_shuffle_buffer_seeded_permutation() -> None:
    items = list(range(5000))
    a = list(shuffle_buffer(iter(items), buffer_bytes=28 * 200, seed=1))
    b = list(shuffle_buffer(iter(items), buffer_bytes=28 * 200, seed=1))
    c = list(shuffle_buffer(iter(items), buffer_bytes=28 * 200, seed=2))
    assert a == b != c
    assert sorted(a) == items and a != items
    # bounded: an item can only move ahead by at most the buffer size
    assert all(pos >= item - 200 for pos, item in enumerate(a))


def test_memory_budget_from_system() -> None:
    b = MemoryBudget.from_system()
    assert 256 * MiB <= b.total_bytes <= 4096 * MiB
    assert 1 * MiB <= b.batch_bytes <= 16 * MiB
    assert b.sample_bytes == 4 * MiB
    small = MemoryBudget.from_system(available_bytes=100 * MiB)
    assert small.total_bytes == 256 * MiB and small.batch_bytes == 4 * MiB
    big = MemoryBudget.from_system(available_bytes=10**15)
    assert big.total_bytes == 4096 * MiB and big.batch_bytes == 16 * MiB and big.shuffle_buffer_bytes == 256 * MiB
    with pytest.raises(ValueError):
        MemoryBudget.from_system(fraction=0)


def test_bounded_buffer_backpressure_no_drops() -> None:
    buf: BoundedBuffer[int] = BoundedBuffer(capacity_bytes=10, size_fn=lambda _: 1)
    produced = 1000
    max_used: list[int] = []

    def producer() -> None:
        for i in range(produced):
            buf.put(i)
            max_used.append(buf.used_bytes)
        buf.close()

    t = threading.Thread(target=producer)
    t.start()
    out = []
    while True:
        try:
            out.append(buf.get(timeout=5))
        except BufferClosed:
            break
        if len(out) % 100 == 0:
            time.sleep(0.01)  # slow consumer -> producer must block, not drop
    t.join(5)
    assert out == list(range(produced))
    assert max(max_used) <= 10


def test_bounded_buffer_oversize_timeout_and_closed() -> None:
    buf: BoundedBuffer[str] = BoundedBuffer(capacity_bytes=4, size_fn=len)
    buf.put("toolarge")  # admitted alone when empty
    with pytest.raises(TimeoutError):
        buf.put("x", timeout=0.05)
    assert buf.get() == "toolarge"
    with pytest.raises(TimeoutError):
        buf.get(timeout=0.05)
    buf.put("ab")
    buf.close()
    with pytest.raises(BufferClosed):
        buf.put("c")
    assert buf.get() == "ab"
    with pytest.raises(BufferClosed):
        buf.get()
