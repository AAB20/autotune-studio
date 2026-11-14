"""Zero row configuration data engine (ADR-0003).

Entry point: :func:`open_dataset` -> :class:`StreamingDataset`.
"""

from __future__ import annotations

from ..connectors import make_connector
from ..contracts import DatasetSpec
from ..interfaces import EventSink
from .batching import batch_by_bytes, batch_by_tokens, shuffle_buffer
from .discovery import detect_format, discover
from .memory_manager import BoundedBuffer, BufferClosed, MemoryBudget, approx_size
from .streaming import StreamingDataset, split_of
from .validation import RowValidator, canonical_json, rows_checksum, to_text

__all__ = [
    "BoundedBuffer",
    "BufferClosed",
    "MemoryBudget",
    "RowValidator",
    "StreamingDataset",
    "approx_size",
    "batch_by_bytes",
    "batch_by_tokens",
    "canonical_json",
    "detect_format",
    "discover",
    "open_dataset",
    "rows_checksum",
    "shuffle_buffer",
    "split_of",
    "to_text",
]


def open_dataset(spec: DatasetSpec, budget: MemoryBudget | None = None,
                 events: EventSink | None = None) -> StreamingDataset:
    """Open ``spec`` as a :class:`StreamingDataset` after byte-bounded discovery.

    The caller must have validated ``spec.path`` with the security module.
    Raises ``DatasetEmpty`` (no rows), ``DatasetMalformed`` (only malformed
    records in the sample) or ``ValidationFailed`` (unknown ``text_column``).
    """
    budget = budget or MemoryBudget.from_system()
    sample_connector = make_connector(spec, min(budget.batch_bytes, budget.sample_bytes))
    info = discover(sample_connector, spec, budget.sample_bytes)
    return StreamingDataset(info, make_connector(spec, budget.batch_bytes), budget=budget, events=events)
