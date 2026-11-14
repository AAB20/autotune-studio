"""Parquet connector: ``ParquetFile.iter_batches`` over one file or a sorted directory of files."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts import Cardinality
from ..errors import DatasetEmpty
from ..interfaces import StreamCursor
from .base import ConnectorBase, MalformedSink


class ParquetConnector(ConnectorBase):
    """Stream Parquet data with exact cardinality from file footers.

    ``path`` may be a file or a directory; a directory is searched recursively
    for ``*.parquet`` files, sorted by relative POSIX path (deterministic order).

    Cursor: ``rows_consumed`` = global row offset. Resume skips whole files and
    whole row groups using footer metadata (no data read), then slices the first
    batch. Batch size in rows is derived from the byte budget and the average
    uncompressed row size -- it bounds memory per batch, not the total.
    """

    kind = "parquet"

    def __init__(self, path: str | os.PathLike[str], batch_bytes: int) -> None:
        super().__init__()
        self.path = os.fspath(path)
        self.batch_bytes = max(64 * 1024, int(batch_bytes))
        p = Path(self.path)
        if p.is_dir():
            files = sorted((f for f in p.rglob("*.parquet") if f.is_file()),
                           key=lambda f: f.relative_to(p).as_posix())
            if not files:
                raise DatasetEmpty(f"no .parquet files found in directory {self.path}", {"path": self.path})
            self.files = [str(f) for f in files]
        else:
            self.files = [self.path]
        self._rows = [_num_rows(f) for f in self.files]

    def total_bytes(self) -> int | None:
        return sum(os.path.getsize(f) for f in self.files)

    def cardinality(self) -> Cardinality:
        return Cardinality(rows=sum(self._rows), exact=True, method="metadata")

    def iter_batches(self, cursor: StreamCursor | None = None,
                     on_malformed: MalformedSink | None = None) -> Iterator[tuple[pa.RecordBatch, StreamCursor]]:
        """Yield ``(batch, cursor_after_batch)``. Parquet has no per-record parse errors."""
        cur = self._check_cursor(cursor)
        skip = cur.rows_consumed
        total_rows = sum(self._rows) or 1
        total_bytes = self.total_bytes() or 0
        done = 0
        for path, nrows in zip(self.files, self._rows):
            if skip >= nrows:
                skip -= nrows
                done += nrows
                continue
            pf = pq.ParquetFile(path)
            try:
                md = pf.metadata
                groups: list[int] = []
                for g in range(md.num_row_groups):
                    g_rows = md.row_group(g).num_rows
                    if skip >= g_rows:
                        skip -= g_rows
                        done += g_rows
                    else:
                        groups.append(g)
                if not groups:
                    continue
                uncompressed = sum(md.row_group(g).total_byte_size for g in groups)
                g_rows_total = sum(md.row_group(g).num_rows for g in groups) or 1
                row_bytes = max(1, uncompressed // g_rows_total)
                batch_size = max(1, self.batch_bytes // row_bytes)
                for batch in pf.iter_batches(batch_size=batch_size, row_groups=groups):
                    if skip:
                        if batch.num_rows <= skip:
                            skip -= batch.num_rows
                            done += batch.num_rows
                            continue
                        done += skip
                        batch = batch.slice(skip)
                        skip = 0
                    if batch.num_rows == 0:
                        continue
                    done += batch.num_rows
                    yield batch, self._cursor(done, bytes=int(total_bytes * done / total_rows))
            finally:
                pf.close()


def _num_rows(path: str) -> int:
    """Row count from the footer; the file handle is closed immediately (Windows keeps files locked)."""
    pf = pq.ParquetFile(path)
    try:
        return int(pf.metadata.num_rows)
    finally:
        pf.close()
