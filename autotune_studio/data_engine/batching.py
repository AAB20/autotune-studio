"""Dynamic batching helpers. All are generators with no cap on the number of items.

Note on shuffling: a global shuffle of a stream is impossible without
materialising it. :func:`shuffle_buffer` gives a *local* seeded shuffle within a
bounded byte budget; items far apart in the source stay far apart.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

from .memory_manager import approx_size

T = TypeVar("T")


def batch_by_bytes(rows: Iterable[T], max_bytes: int, size_fn: Callable[[T], int] = approx_size) -> Iterator[list[T]]:
    """Group ``rows`` into lists whose summed ``size_fn`` stays <= ``max_bytes``.

    A single item larger than ``max_bytes`` is yielded alone (never dropped).
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    batch: list[T] = []
    used = 0
    for row in rows:
        n = size_fn(row)
        if batch and used + n > max_bytes:
            yield batch
            batch, used = [], 0
        batch.append(row)
        used += n
    if batch:
        yield batch


def batch_by_tokens(texts: Iterable[T], length_fn: Callable[[T], int], max_tokens: int,
                    max_batch: int | None = None) -> Iterator[list[T]]:
    """Group items so the *padded* token cost ``max(length) * len(batch)`` stays <= ``max_tokens``.

    ``max_batch`` optionally bounds items per batch (a micro-batch size, not a
    cap on the total). An item longer than ``max_tokens`` is yielded alone.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if max_batch is not None and max_batch <= 0:
        raise ValueError("max_batch must be > 0")
    batch: list[T] = []
    longest = 0
    for item in texts:
        n = max(0, int(length_fn(item)))
        new_longest = max(longest, n)
        if batch and (new_longest * (len(batch) + 1) > max_tokens or (max_batch is not None and len(batch) >= max_batch)):
            yield batch
            batch, new_longest = [], n
        batch.append(item)
        longest = new_longest
    if batch:
        yield batch


def shuffle_buffer(iterable: Iterable[T], buffer_bytes: int, seed: int,
                   size_fn: Callable[[T], int] = approx_size) -> Iterator[T]:
    """Seeded bounded shuffle: every input item is yielded exactly once.

    Fills a buffer up to ``buffer_bytes``; then each new item replaces a random
    buffered item which is yielded. The remainder is shuffled and flushed at the
    end. Same input + same seed -> same order.
    """
    if buffer_bytes <= 0:
        raise ValueError("buffer_bytes must be > 0")
    rng = random.Random(seed)
    buf: list[T] = []
    sizes: list[int] = []
    used = 0
    for item in iterable:
        n = size_fn(item)
        if buf and used + n > buffer_bytes:
            i = rng.randrange(len(buf))
            out, used = buf[i], used - sizes[i]
            buf[i], sizes[i] = item, n
            used += n
            yield out
            continue
        buf.append(item)
        sizes.append(n)
        used += n
    rng.shuffle(buf)
    yield from buf
