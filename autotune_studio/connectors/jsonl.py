"""JSON Lines connector: buffered binary line reader, cursor = byte offset of the next unread line."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from ..contracts import Cardinality
from ..interfaces import StreamCursor
from .base import ConnectorBase, MalformedRecord, MalformedSink, make_snippet, rows_to_batch

_BOM = b"\xef\xbb\xbf"


class JSONLConnector(ConnectorBase):
    """Stream a JSON Lines file in batches of ~``batch_bytes`` source bytes (line by line).

    * Each non-blank line must be a JSON object; other lines (syntax errors,
      invalid UTF-8, arrays/scalars) are reported to the malformed sink with
      their byte offset and are not emitted as rows. Blank lines carry no
      record and are ignored.
    * Cursor ``state = {"offset": byte offset of the first line of the next
      batch, "base_rows": rows before that offset, "bytes": offset}``. Resume
      seeks to ``offset`` and skips ``rows_consumed - base_rows`` rows.
    * ``coerced_batches`` counts batches in which a mixed-type column was
      converted to strings (no row is lost).
    """

    kind = "jsonl"

    def __init__(self, path: str | os.PathLike[str], batch_bytes: int) -> None:
        super().__init__()
        self.path = os.fspath(path)
        self.batch_bytes = max(64 * 1024, int(batch_bytes))
        self.coerced_batches = 0

    def total_bytes(self) -> int | None:
        return os.path.getsize(self.path)

    def cardinality(self) -> Cardinality:
        return Cardinality(rows=None, exact=False, method="unknown")

    def iter_batches(self, cursor: StreamCursor | None = None,
                     on_malformed: MalformedSink | None = None) -> Iterator[tuple[pa.RecordBatch, StreamCursor]]:
        """Yield ``(batch, cursor_after_batch)``; malformed lines go to ``on_malformed``."""
        cur = self._check_cursor(cursor)
        sink = self._sink(on_malformed)
        offset = int(cur.state.get("offset", 0))
        base_rows = int(cur.state.get("base_rows", 0)) if "offset" in cur.state else 0
        rows_total = cur.rows_consumed  # rows skipped below are consumed too
        skip = cur.rows_consumed - base_rows
        if skip < 0:
            raise ValueError("cursor.rows_consumed is smaller than state.base_rows")

        with open(self.path, "rb", buffering=1024 * 1024) as f:
            f.seek(offset)
            pos = offset  # file offset of the next line
            rows: list[dict[str, Any]] = []
            acc = 0
            for line in iter(f.readline, b""):  # one line in memory at a time (C-level buffering)
                here = pos
                pos += len(line)
                if here == 0 and line.startswith(_BOM):
                    line = line[len(_BOM):]
                if line.strip():
                    row, reason = _parse_line(line)
                    if row is None:
                        if skip <= 0:
                            sink(MalformedRecord(f"byte {here}", reason, make_snippet(line)))
                    elif skip > 0:
                        skip -= 1
                    else:
                        rows.append(row)
                        acc += len(line)
                if acc >= self.batch_bytes:
                    yield self._emit(rows, rows_total := rows_total + len(rows), pos)
                    rows, acc = [], 0
            if rows:
                yield self._emit(rows, rows_total + len(rows), pos)

    def _emit(self, rows: list[dict[str, Any]], rows_total: int, pos: int) -> tuple[pa.RecordBatch, StreamCursor]:
        batch, coerced = rows_to_batch(rows)
        self.coerced_batches += coerced
        return batch, self._cursor(rows_total, offset=pos, base_rows=rows_total, bytes=pos)


def _parse_line(line: bytes) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(line)
    except UnicodeDecodeError:
        return None, "invalid UTF-8"
    except ValueError as e:  # json.JSONDecodeError subclasses ValueError
        return None, f"invalid JSON ({e.msg if hasattr(e, 'msg') else e})"
    if not isinstance(value, dict):
        return None, f"expected a JSON object, got {type(value).__name__}"
    return value, ""
