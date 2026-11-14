"""Module boundary interfaces (Protocols). Owned by: Chief Architect.

Implementations live in their owning packages; consumers depend only on these
protocols so modules can be built and tested independently.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa

from .contracts import (
    Cardinality,
    CheckpointInfo,
    DatasetInfo,
    DatasetPreview,
    ExportRequest,
    ExportResult,
    ModelConfig,
    ResourceSnapshot,
    StreamStats,
    TrainingConfig,
)


@dataclass
class StreamCursor:
    """Resumable position inside a stream. Must be JSON-serialisable via ``to_dict``.

    ``kind`` identifies the connector; ``state`` is connector-specific
    (e.g. file byte offset + rows consumed, parquet row-group/batch index,
    HF ``IterableDataset.state_dict()``).
    """

    kind: str
    rows_consumed: int = 0
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "rows_consumed": self.rows_consumed, "state": self.state}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamCursor":
        return cls(kind=d["kind"], rows_consumed=int(d.get("rows_consumed", 0)), state=dict(d.get("state", {})))


@runtime_checkable
class Connector(Protocol):
    """Reads a source as a sequence of Arrow record batches. Never materialises the whole source."""

    kind: str

    def iter_batches(
        self,
        cursor: StreamCursor | None = None,
        on_malformed: Callable[[Any], None] | None = None,
    ) -> Iterator[tuple[pa.RecordBatch, StreamCursor]]:
        """Yield (batch, cursor_after_batch). Raises on I/O / parse failure (never returns silently).

        Unparseable records are reported through ``on_malformed`` (ADR-0006), never dropped silently.
        """
        ...

    def cardinality(self) -> Cardinality:
        """Row count if cheaply knowable (metadata), else rows=None."""
        ...

    def total_bytes(self) -> int | None:
        """Total source size in bytes for progress estimation, or None."""
        ...


@runtime_checkable
class StreamingDatasetProtocol(Protocol):
    """Consumer-facing dataset handle produced by ``data_engine``."""

    info: DatasetInfo

    @property
    def stats(self) -> StreamStats: ...

    def iter_rows(
        self,
        cursor: StreamCursor | None = None,
        split: str | None = None,
        val_fraction: float = 0.0,
        seed: int = 0,
    ) -> Iterator[tuple[dict[str, Any], StreamCursor]]:
        """Yield rows one by one with the cursor after each row.

        ``split`` in {None, "train", "validation"} applies the deterministic hash split.
        """
        ...

    def to_text(self, row: dict[str, Any]) -> str:
        """Format a row into training text according to ``info.detected_format``."""
        ...

    def preview(self, n: int = 20) -> DatasetPreview: ...

    def cancel(self) -> None: ...


class ResourceMonitorProtocol(Protocol):
    def snapshot(self) -> ResourceSnapshot: ...


class EventSink(Protocol):
    def publish(self, type: str, data: dict[str, Any]) -> None: ...


@dataclass
class TrainingControl:
    """Cooperative control flags shared between API and training thread."""

    stop: threading.Event = field(default_factory=threading.Event)
    pause: threading.Event = field(default_factory=threading.Event)
    save_now: threading.Event = field(default_factory=threading.Event)


@dataclass
class TrainingResult:
    job_id: str
    final_step: int
    final_loss: float | None
    stopped_early: bool
    checkpoints: list[CheckpointInfo]
    output_dir: str


class TrainingEngineProtocol(Protocol):
    backend: str

    def run(
        self,
        job_id: str,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        dataset: StreamingDatasetProtocol,
        events: EventSink,
        control: TrainingControl,
        resume_from: CheckpointInfo | None = None,
    ) -> TrainingResult: ...


ExportFn = Callable[[ExportRequest, CheckpointInfo], ExportResult]
