"""Hugging Face connector: ``datasets.load_dataset(..., streaming=True)``.

``datasets`` is imported lazily (inside :meth:`HuggingFaceConnector._load`) so that
importing ``autotune_studio.connectors`` stays light.

Security: ``trust_remote_code`` is never passed. ``datasets`` 5.x removed
dataset loading scripts entirely (the kwarg is popped and a truthy value is
rejected with an error log), so not passing it is equivalent to ``False``.
The token is read from the ``HF_TOKEN`` environment variable and never logged.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from typing import Any

import pyarrow as pa

from ..contracts import Cardinality
from ..interfaces import StreamCursor
from .base import ConnectorBase, MalformedSink, rows_to_batch


def _approx_size(v: Any) -> int:
    if isinstance(v, (str, bytes)):
        return len(v) + 16
    if isinstance(v, dict):
        return 32 + sum(len(str(k)) + _approx_size(x) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return 16 + sum(_approx_size(x) for x in v)
    return 8


class HuggingFaceConnector(ConnectorBase):
    """Stream a Hugging Face dataset split as Arrow batches of ~``batch_bytes``.

    Cursor ``state``: ``{"hf_state": IterableDataset.state_dict(), "base_rows": n,
    "bytes": approx}`` when the state dict is JSON-serialisable; otherwise only
    ``rows_consumed`` is kept and resume uses ``IterableDataset.skip``.
    Cardinality: ``info.splits[split].num_examples`` when the builder knows it
    (``method="metadata"``), else unknown.

    ``data_files`` / ``data_dir`` are passed through for packaged builders
    (``"json"``, ``"csv"``, ``"parquet"``), which also makes the connector
    testable offline.
    """

    kind = "huggingface"

    def __init__(self, name: str, batch_bytes: int, *, split: str = "train", config_name: str | None = None,
                 revision: str | None = None, data_files: str | Sequence[str] | None = None,
                 data_dir: str | None = None) -> None:
        super().__init__()
        self.name = name
        self.batch_bytes = max(64 * 1024, int(batch_bytes))
        self.split = split
        self.config_name = config_name
        self.revision = revision
        self.data_files = data_files
        self.data_dir = data_dir
        self._info_ds: Any = None

    def _load(self) -> Any:
        """Create a fresh streaming ``IterableDataset`` (cheap: nothing is downloaded yet)."""
        try:
            import datasets  # heavy, lazy on purpose
        except ImportError as e:  # pragma: no cover - datasets is a core dependency
            from ..errors import DependencyMissing
            raise DependencyMissing("the 'datasets' package is required for Hugging Face sources") from e
        kwargs: dict[str, Any] = {"split": self.split, "streaming": True}
        if self.revision:
            kwargs["revision"] = self.revision
        token = os.environ.get("HF_TOKEN")
        if token:
            kwargs["token"] = token
        if self.data_files is not None:
            kwargs["data_files"] = self.data_files
        if self.data_dir is not None:
            kwargs["data_dir"] = self.data_dir
        return datasets.load_dataset(self.name, self.config_name, **kwargs)

    def _split_info(self) -> Any:
        if self._info_ds is None:
            self._info_ds = self._load()
        splits = getattr(self._info_ds.info, "splits", None) or {}
        try:
            return splits.get(self.split) if hasattr(splits, "get") else splits[self.split]
        except (KeyError, TypeError):
            return None

    def cardinality(self) -> Cardinality:
        si = self._split_info()
        n = getattr(si, "num_examples", None) if si is not None else None
        if isinstance(n, int) and n > 0:
            return Cardinality(rows=n, exact=True, method="metadata")
        return Cardinality(rows=None, exact=False, method="unknown")

    def total_bytes(self) -> int | None:
        si = self._split_info()
        n = getattr(si, "num_bytes", None) if si is not None else None
        return n if isinstance(n, int) and n > 0 else None

    def iter_batches(self, cursor: StreamCursor | None = None,
                     on_malformed: MalformedSink | None = None) -> Iterator[tuple[pa.RecordBatch, StreamCursor]]:
        """Yield ``(batch, cursor_after_batch)``; errors from ``datasets`` propagate."""
        cur = self._check_cursor(cursor)
        ds = self._load()
        hf_state = cur.state.get("hf_state")
        base_rows = int(cur.state.get("base_rows", 0)) if hf_state is not None else 0
        if hf_state is not None:
            ds.load_state_dict(hf_state)
        skip = cur.rows_consumed - base_rows
        if skip < 0:
            raise ValueError("cursor.rows_consumed is smaller than state.base_rows")
        if skip and hf_state is None:
            ds = ds.skip(skip)  # streams and discards; nothing materialised
            skip = 0
        rows_total = cur.rows_consumed  # includes rows skipped below
        nbytes = int(cur.state.get("bytes", 0))
        can_state = hasattr(ds, "state_dict")
        rows: list[dict[str, Any]] = []
        acc = 0
        for row in ds:
            if skip > 0:
                skip -= 1
                continue
            rows.append(row)
            acc += _approx_size(row)
            if acc >= self.batch_bytes:
                rows_total += len(rows)
                nbytes += acc
                yield self._make(rows, rows_total, nbytes, ds if can_state else None)
                rows, acc = [], 0
        if rows:
            rows_total += len(rows)
            nbytes += acc
            yield self._make(rows, rows_total, nbytes, ds if can_state else None)

    def _make(self, rows: list[dict[str, Any]], rows_total: int, nbytes: int, ds: Any) -> tuple[pa.RecordBatch, StreamCursor]:
        batch, _ = rows_to_batch(rows)
        state: dict[str, Any] = {"bytes": nbytes}
        if ds is not None:
            try:
                st = ds.state_dict()
                json.dumps(st)
                state.update(hf_state=st, base_rows=rows_total)
            except (TypeError, ValueError, NotImplementedError):
                pass
        return batch, self._cursor(rows_total, **state)
