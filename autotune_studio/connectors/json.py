"""JSON connector: a top-level array (or records array inside an object) streamed with ijson.

Layout detection (from the first bytes, never by loading the file):

* ``[ ... ]`` -- array of records, streamed with ``ijson.items(f, "item")``.
* first line is a complete JSON object and more non-blank lines follow -- the
  file is JSON Lines; everything is delegated to :class:`JSONLConnector`
  (``kind`` becomes ``"jsonl"``).
* a single object -- if one of its top-level keys (other than chat keys such as
  ``messages``) holds an array of objects, that array is streamed as the
  records (``"<key>.item"``) and a note is recorded in ``layout_note``;
  otherwise the whole object is one row.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import ijson
import pyarrow as pa

from ..contracts import Cardinality
from ..errors import DatasetMalformed
from ..interfaces import StreamCursor
from .base import ConnectorBase, MalformedRecord, MalformedSink, make_snippet, rows_to_batch
from .jsonl import JSONLConnector

_FIRST_LINE_LIMIT = 8 * 1024 * 1024
_CHAT_KEYS = {"messages", "conversations"}


def _approx_bytes(row: dict[str, Any]) -> int:
    n = 2
    for k, v in row.items():
        n += len(k) + 4 + (len(v) if isinstance(v, str) else len(json.dumps(v, default=str)) if isinstance(v, (list, dict)) else 8)
    return n


class JSONConnector(ConnectorBase):
    """Stream a ``.json`` file; see module docstring for the layouts supported.

    Cursor: ``rows_consumed`` (rows emitted); resume re-parses from the start and
    discards rows without keeping them (O(rows) time, O(1) memory).
    ``state["items"]`` = array items consumed, ``state["bytes"]`` approx position.
    """

    kind = "json"

    def __init__(self, path: str | os.PathLike[str], batch_bytes: int) -> None:
        super().__init__()
        self.path = os.fspath(path)
        self.batch_bytes = max(64 * 1024, int(batch_bytes))
        self.coerced_batches = 0
        self.layout_note: str | None = None
        self._delegate: JSONLConnector | None = None
        self._prefix = "item"
        self._detect()

    # ------------------------------------------------------------------ detection
    def _detect(self) -> None:
        with open(self.path, "rb") as f:
            head = f.read(4096).lstrip(b"\xef\xbb\xbf \t\r\n")
            if not head or head[:1] == b"[":
                return
            if head[:1] != b"{":
                raise DatasetMalformed(
                    f"JSON file must start with '[' or '{{', found {head[:1]!r}", {"path": self.path})
            f.seek(0)
            first = f.readline(_FIRST_LINE_LIMIT)
            first_ok = False
            try:
                first_ok = isinstance(json.loads(first), dict)
            except ValueError:
                pass
            if first_ok:
                rest_has_content = any(line.strip() for line in iter(lambda: f.readline(65536), b""))
                if rest_has_content:
                    self._delegate = JSONLConnector(self.path, self.batch_bytes)
                    self.kind = self._delegate.kind
                    self.layout_note = "file contains JSON Lines; read as JSONL"
                    return
        self._prefix = self._find_records_prefix()

    def _find_records_prefix(self) -> str:
        """For a single top-level object, return ``"<key>.item"`` for a records array, else ``""``."""
        with open(self.path, "rb") as f:
            events = ijson.parse(f, use_float=True)
            prev: tuple[str, str] | None = None
            try:
                for prefix, event, _ in events:
                    if (prev is not None and prev[1] == "start_array" and "." not in prev[0] and prev[0]
                            and event == "start_map" and prefix == prev[0] + ".item"
                            and prev[0] not in _CHAT_KEYS):
                        self.layout_note = f"records read from top-level key {prev[0]!r}"
                        return prefix
                    prev = (prefix, event)
            except ijson.JSONError as e:
                raise DatasetMalformed(f"invalid JSON: {e}", {"path": self.path}) from e
        return ""

    # ------------------------------------------------------------------ protocol
    def total_bytes(self) -> int | None:
        return os.path.getsize(self.path)

    def cardinality(self) -> Cardinality:
        if self._delegate is not None:
            return self._delegate.cardinality()
        return Cardinality(rows=None, exact=False, method="unknown")

    def iter_batches(self, cursor: StreamCursor | None = None,
                     on_malformed: MalformedSink | None = None) -> Iterator[tuple[pa.RecordBatch, StreamCursor]]:
        """Yield ``(batch, cursor_after_batch)``; non-object items go to ``on_malformed``.

        A JSON syntax error cannot be skipped inside an array and raises
        :class:`DatasetMalformed` (after the rows before it were yielded).
        """
        if self._delegate is not None:
            yield from self._delegate.iter_batches(cursor, on_malformed)
            return
        cur = self._check_cursor(cursor)
        sink = self._sink(on_malformed)
        skip = cur.rows_consumed
        size = os.path.getsize(self.path)
        if size == 0:
            return
        with open(self.path, "rb") as f:
            rows: list[dict[str, Any]] = []
            acc = 0
            emitted = 0
            items = 0
            try:
                for item in ijson.items(f, self._prefix, use_float=True):
                    items += 1
                    if not isinstance(item, dict):
                        if skip <= 0:
                            sink(MalformedRecord(f"item {items - 1}", f"expected a JSON object, got {type(item).__name__}",
                                                 make_snippet(json.dumps(item, default=str))))
                        continue
                    if skip > 0:
                        skip -= 1
                        emitted += 1
                        continue
                    rows.append(item)
                    acc += _approx_bytes(item)
                    if acc >= self.batch_bytes:
                        emitted += len(rows)
                        yield self._emit(rows, emitted, items, min(f.tell(), size))
                        rows, acc = [], 0
            except ijson.JSONError as e:
                if rows:
                    emitted += len(rows)
                    yield self._emit(rows, emitted, items, min(f.tell(), size))
                raise DatasetMalformed(f"invalid JSON after item {items}: {e}",
                                       {"path": self.path, "items": items}) from e
            if rows:
                emitted += len(rows)
                yield self._emit(rows, emitted, items, size)

    def _emit(self, rows: list[dict[str, Any]], emitted: int, items: int, pos: int) -> tuple[pa.RecordBatch, StreamCursor]:
        batch, coerced = rows_to_batch(rows)
        self.coerced_batches += coerced
        return batch, self._cursor(emitted, items=items, bytes=pos)
