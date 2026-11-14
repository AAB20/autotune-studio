"""Shared building blocks for connectors.

Every connector yields ``(pyarrow.RecordBatch, StreamCursor)`` pairs and never
materialises its whole source. This module holds what they have in common:

* :class:`MalformedRecord` / :data:`MalformedSink` -- the side channel through
  which connectors surface records they could not parse (bad JSON lines,
  non-object items, undecodable bytes). Nothing is dropped silently: each such
  record is either passed to the sink given to ``iter_batches`` or, when no
  sink is given, counted on the connector (``malformed_total``) and logged.
* :func:`rows_to_batch` -- robust ``list[dict] -> RecordBatch`` conversion that
  never loses a row (columns with mixed Python types are coerced to strings).
* :func:`slice_skip` -- drop the first ``n`` rows of a batch stream (resume).

Cursor convention (all connectors)
----------------------------------
``StreamCursor.rows_consumed`` is the number of rows (records that were emitted
in batches) before the resume point. ``state`` optionally holds a fast seek
position (``offset`` / ``hf_state``) that corresponds to ``state["base_rows"]``
rows; on resume the connector seeks there and then skips
``rows_consumed - base_rows`` rows. A cursor with empty ``state`` means "skip
``rows_consumed`` rows from the start" and is always valid. ``state["bytes"]``
is an approximate count of source bytes consumed (used for progress only).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from ..interfaces import StreamCursor

log = logging.getLogger(__name__)

SNIPPET_CHARS = 200


@dataclass(frozen=True)
class MalformedRecord:
    """A source record that could not be turned into a row."""

    location: str
    """Human-readable position, e.g. ``"byte 1234"`` or ``"item 17"``."""
    reason: str
    snippet: str
    """The beginning of the raw record (at most :data:`SNIPPET_CHARS` chars)."""

    def describe(self) -> str:
        """One-line description used in ``StreamStats.malformed_examples``."""
        return f"{self.location}: {self.reason}: {self.snippet}"


MalformedSink = Callable[[MalformedRecord], None]


def make_snippet(raw: bytes | str) -> str:
    """Return a short printable excerpt of a raw record."""
    if isinstance(raw, bytes):
        raw = raw[: SNIPPET_CHARS * 4].decode("utf-8", errors="replace")
    raw = raw.strip()
    return raw if len(raw) <= SNIPPET_CHARS else raw[:SNIPPET_CHARS] + "..."


class ConnectorBase:
    """Common helpers: cursor validation and the default malformed sink."""

    kind: str = "base"

    def __init__(self) -> None:
        self.malformed_total = 0
        """Malformed records seen while no explicit sink was supplied."""

    def _sink(self, on_malformed: MalformedSink | None) -> MalformedSink:
        if on_malformed is not None:
            return on_malformed

        def _default(rec: MalformedRecord) -> None:
            self.malformed_total += 1
            log.warning("malformed record in %s source (%s): %s", self.kind, rec.location, rec.reason)

        return _default

    def _check_cursor(self, cursor: StreamCursor | None) -> StreamCursor:
        if cursor is None:
            return StreamCursor(kind=self.kind)
        if cursor.kind != self.kind:
            raise ValueError(f"cursor kind {cursor.kind!r} does not match connector kind {self.kind!r}")
        if cursor.rows_consumed < 0:
            raise ValueError("cursor.rows_consumed must be >= 0")
        return cursor

    def _cursor(self, rows: int, **state: Any) -> StreamCursor:
        return StreamCursor(kind=self.kind, rows_consumed=rows, state=dict(state))


def _jsonable(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)


def rows_to_batch(rows: list[dict[str, Any]]) -> tuple[pa.RecordBatch, bool]:
    """Convert python rows to a RecordBatch without losing any row.

    Returns ``(batch, coerced)``. If Arrow cannot infer a single type for a
    column (e.g. ``1`` in one row and ``"x"`` in another) every column that
    fails is converted to strings (JSON for non-strings) and ``coerced`` is
    True. Missing keys become nulls.
    """
    try:
        return pa.RecordBatch.from_pylist(rows), False
    except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError):
        pass
    names: dict[str, None] = {}
    for r in rows:
        for k in r:
            names.setdefault(k, None)
    arrays: list[pa.Array] = []
    for name in names:
        values = [r.get(name) for r in rows]
        try:
            arrays.append(pa.array(values))
        except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError):
            arrays.append(pa.array([None if v is None else _jsonable(v) for v in values], type=pa.string()))
    return pa.RecordBatch.from_arrays(arrays, names=list(names)), True


def slice_skip(
    batches: Iterator[tuple[pa.RecordBatch, StreamCursor]], skip: int
) -> Iterator[tuple[pa.RecordBatch, StreamCursor]]:
    """Drop the first ``skip`` rows of a batch stream, one batch in memory at a time."""
    for batch, cur in batches:
        if skip > 0:
            if batch.num_rows <= skip:
                skip -= batch.num_rows
                continue
            batch = batch.slice(skip)
            skip = 0
        yield batch, cur


def iter_with_sink(connector: Any, cursor: StreamCursor | None,
                   sink: MalformedSink) -> Iterator[tuple[pa.RecordBatch, StreamCursor]]:
    """Call ``connector.iter_batches`` passing ``on_malformed`` when the connector supports it."""
    try:
        return connector.iter_batches(cursor, on_malformed=sink)  # type: ignore[no-any-return]
    except TypeError:  # a plain interfaces.Connector without the side channel
        return connector.iter_batches(cursor)  # type: ignore[no-any-return]
