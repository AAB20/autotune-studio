"""CSV connector: ``pyarrow.csv.open_csv`` streaming reader, one block at a time."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.csv as pacsv

from ..contracts import Cardinality
from ..interfaces import StreamCursor
from .base import ConnectorBase, MalformedSink, slice_skip


class CSVConnector(ConnectorBase):
    """Stream a CSV file as Arrow batches of roughly ``batch_bytes`` source bytes.

    By default every column is read as (nullable) string: type inference from the
    first block can fail on a later block (``"N/A"`` in an int column) which would
    abort a long training run mid-stream. Pass ``infer_types=True`` to let Arrow
    infer column types instead.

    Cursor: ``rows_consumed``; resume re-opens the file and discards rows batch by
    batch (never more than one batch in memory). ``state["bytes"]`` approximates
    the file position for progress.
    """

    kind = "csv"

    def __init__(self, path: str | os.PathLike[str], batch_bytes: int, *, infer_types: bool = False,
                 delimiter: str = ",") -> None:
        super().__init__()
        self.path = os.fspath(path)
        self.batch_bytes = max(64 * 1024, int(batch_bytes))
        self.infer_types = infer_types
        self.delimiter = delimiter

    def total_bytes(self) -> int | None:
        return os.path.getsize(self.path)

    def cardinality(self) -> Cardinality:
        return Cardinality(rows=None, exact=False, method="unknown")

    def _open(self, f: object) -> pacsv.CSVStreamingReader:
        read_opts = pacsv.ReadOptions(block_size=self.batch_bytes)
        parse_opts = pacsv.ParseOptions(delimiter=self.delimiter, newlines_in_values=True)
        if self.infer_types:
            return pacsv.open_csv(f, read_options=read_opts, parse_options=parse_opts)
        # Discover column names (reads one block), then re-open with string types.
        with open(self.path, "rb") as probe:
            names = pacsv.open_csv(probe, read_options=read_opts, parse_options=parse_opts).schema.names
        conv = pacsv.ConvertOptions(column_types={n: pa.string() for n in names}, strings_can_be_null=True)
        return pacsv.open_csv(f, read_options=read_opts, parse_options=parse_opts, convert_options=conv)

    def iter_batches(self, cursor: StreamCursor | None = None,
                     on_malformed: MalformedSink | None = None) -> Iterator[tuple[pa.RecordBatch, StreamCursor]]:
        """Yield ``(batch, cursor_after_batch)``; parse errors raise ``pyarrow.ArrowInvalid``."""
        cur = self._check_cursor(cursor)
        if os.path.getsize(self.path) == 0:
            return
        yield from slice_skip(self._iter(), cur.rows_consumed)

    def _iter(self) -> Iterator[tuple[pa.RecordBatch, StreamCursor]]:
        size = os.path.getsize(self.path)
        with open(self.path, "rb") as f:
            reader = self._open(f)
            rows = 0
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    return
                if batch.num_rows == 0:
                    continue
                rows += batch.num_rows
                yield batch, self._cursor(rows, bytes=min(f.tell(), size))
