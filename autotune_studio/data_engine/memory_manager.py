"""Memory budget derived from available RAM, plus a byte-bounded buffer with backpressure."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

MiB = 1024 * 1024
GiB = 1024 * MiB

T = TypeVar("T")


def _clamp(v: float, lo: int, hi: int) -> int:
    return int(min(max(v, lo), hi))


def approx_size(obj: Any) -> int:
    """Cheap recursive size estimate (bytes) of a python row / value."""
    if isinstance(obj, (str, bytes, bytearray)):
        return len(obj) + 49
    if isinstance(obj, dict):
        return 64 + sum(approx_size(k) + approx_size(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return 56 + sum(approx_size(v) for v in obj)
    return 28


@dataclass(frozen=True)
class MemoryBudget:
    """Byte budgets for the data engine. Batch sizes are always chosen in bytes, never rows."""

    total_bytes: int
    """Overall RAM the data engine may use."""
    batch_bytes: int
    """Target source bytes per connector batch."""
    sample_bytes: int = 4 * MiB
    """Bytes sampled for schema / format discovery."""
    shuffle_buffer_bytes: int = 64 * MiB
    """Capacity of the bounded shuffle buffer."""

    @classmethod
    def from_system(cls, fraction: float = 0.10, min_total: int = 256 * MiB, max_total: int = 4 * GiB,
                    available_bytes: int | None = None) -> MemoryBudget:
        """Derive budgets from ``psutil.virtual_memory().available``.

        ``total = clamp(available * fraction, min_total, max_total)``;
        ``batch_bytes = clamp(total / 64, 1 MiB, 16 MiB)``;
        ``shuffle_buffer_bytes = clamp(total / 8, 4 MiB, 256 MiB)``.
        """
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if available_bytes is None:
            import psutil

            available_bytes = int(psutil.virtual_memory().available)
        total = _clamp(available_bytes * fraction, min_total, max_total)
        return cls(
            total_bytes=total,
            batch_bytes=_clamp(total / 64, 1 * MiB, 16 * MiB),
            sample_bytes=4 * MiB,
            shuffle_buffer_bytes=_clamp(total / 8, 4 * MiB, 256 * MiB),
        )


class BufferClosed(Exception):
    """Raised by :meth:`BoundedBuffer.get` when the buffer is closed and drained."""


class BoundedBuffer(Generic[T]):
    """Thread-safe FIFO bounded by total item bytes, with blocking backpressure.

    * :meth:`put` blocks while adding the item would exceed ``capacity_bytes``
      (an item larger than the capacity is admitted alone when the buffer is
      empty, so nothing deadlocks and nothing is ever dropped).
    * :meth:`get` blocks until an item is available; after :meth:`close` it
      drains the remaining items and then raises :class:`BufferClosed`.
    * A ``put`` after ``close`` raises :class:`BufferClosed` (never silently lost).
    """

    def __init__(self, capacity_bytes: int, size_fn: Callable[[T], int] = approx_size) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be > 0")
        self.capacity_bytes = int(capacity_bytes)
        self._size_fn = size_fn
        self._items: deque[tuple[T, int]] = deque()
        self._used = 0
        self._closed = False
        self._cond = threading.Condition()

    @property
    def used_bytes(self) -> int:
        with self._cond:
            return self._used

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)

    def put(self, item: T, size: int | None = None, timeout: float | None = None) -> None:
        """Append ``item``; block while full. Raises ``TimeoutError`` if ``timeout`` elapses."""
        n = self._size_fn(item) if size is None else int(size)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._closed and self._items and self._used + n > self.capacity_bytes:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("BoundedBuffer.put timed out (buffer full)")
                self._cond.wait(remaining)
            if self._closed:
                raise BufferClosed("buffer is closed")
            self._items.append((item, n))
            self._used += n
            self._cond.notify_all()

    def get(self, timeout: float | None = None) -> T:
        """Pop the oldest item; block while empty. Raises ``TimeoutError`` / :class:`BufferClosed`."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._items:
                if self._closed:
                    raise BufferClosed("buffer is closed and empty")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("BoundedBuffer.get timed out (buffer empty)")
                self._cond.wait(remaining)
            item, n = self._items.popleft()
            self._used -= n
            self._cond.notify_all()
            return item

    def close(self) -> None:
        """Stop accepting items; consumers drain what is left."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
