"""``StreamingDataset``: the consumer-facing handle implementing ``StreamingDatasetProtocol``."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from ..connectors.base import MalformedRecord, iter_with_sink
from ..contracts import Cardinality, DatasetInfo, DatasetPreview, StreamStats
from ..interfaces import Connector, EventSink, StreamCursor
from .memory_manager import MemoryBudget
from .validation import RowValidator, canonical_json, ensure_usable, to_text

log = logging.getLogger(__name__)

SPLITS = (None, "train", "validation")
PREVIEW_MAX_ROWS = 1000
PREVIEW_VALUE_CHARS = 500
_SPLIT_BUCKETS = 10_000


def split_of(row: dict[str, Any], val_fraction: float, seed: int = 0) -> str:
    """Deterministic split: ``blake2b(canonical_json(row), key=seed) % 10_000 < val_fraction * 10_000``.

    The same row always lands in the same split for a given seed, in every pass.
    """
    if val_fraction <= 0:
        return "train"
    h = hashlib.blake2b(canonical_json(row).encode("utf-8"), digest_size=8, key=str(seed).encode("ascii"))
    bucket = int.from_bytes(h.digest(), "big") % _SPLIT_BUCKETS
    return "validation" if bucket < val_fraction * _SPLIT_BUCKETS else "train"


def _truncate(v: Any) -> tuple[Any, bool]:
    if isinstance(v, str):
        if len(v) > PREVIEW_VALUE_CHARS:
            return v[:PREVIEW_VALUE_CHARS] + "...", True
        return v, False
    if isinstance(v, list):
        out, cut = [], False
        for x in v:
            y, c = _truncate(x)
            out.append(y)
            cut |= c
        return out, cut
    if isinstance(v, dict):
        d, cut = {}, False
        for k, x in v.items():
            y, c = _truncate(x)
            d[k] = y
            cut |= c
        return d, cut
    return v, False


class StreamingDataset:
    """Streams validated rows from a connector; never materialises the dataset.

    * ``stats`` can be read from any thread while another thread iterates
      (it returns a copy taken under a lock). Stats describe the most recent pass.
    * ``cancel()`` sets a ``threading.Event`` that the active pass checks between
      rows/batches (state ``"cancelled"``). Each new pass clears it.
    * Exhaustion -> state ``"exhausted"``; an exception from the source -> state
      ``"failed"`` with ``error`` set, and the exception is re-raised.
    * Malformed records (parse errors from the connector, or rows unusable for the
      detected format) are skipped, counted in ``stats.malformed_rows`` and up to
      5 examples are kept; they are never dropped silently.
    * With ``events``, ``"dataset"`` events are published at most every
      ``publish_interval`` seconds plus on every state change.
    """

    def __init__(self, info: DatasetInfo, connector: Connector, budget: MemoryBudget | None = None,
                 events: EventSink | None = None, publish_interval: float = 0.5) -> None:
        self.info = info
        self.connector = connector
        self.budget = budget
        self.events = events
        self.publish_interval = publish_interval
        self._lock = threading.Lock()
        self._stats = StreamStats()
        self._cancel = threading.Event()
        self._last_publish = 0.0

    # ------------------------------------------------------------------ protocol
    @property
    def stats(self) -> StreamStats:
        """Snapshot of the current/last pass statistics."""
        with self._lock:
            return self._stats.model_copy(deep=True)

    def to_text(self, row: dict[str, Any]) -> str:
        """Format ``row`` as training text for ``info.detected_format``."""
        return to_text(row, self.info)

    def cancel(self) -> None:
        """Ask the active pass to stop at the next row/batch boundary."""
        self._cancel.set()

    def iter_rows(self, cursor: StreamCursor | None = None, split: str | None = None, val_fraction: float = 0.0,
                  seed: int = 0) -> Iterator[tuple[dict[str, Any], StreamCursor]]:
        """Yield ``(row, cursor_after_row)`` for valid rows, optionally restricted to a hash split.

        Resuming from any yielded cursor produces exactly the rows after it.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        keep: Callable[[dict[str, Any]], bool] | None = None
        if split is not None:
            keep = lambda row: split_of(row, val_fraction, seed) == split
        return self._pass(cursor, keep)

    def iter_texts(self, cursor: StreamCursor | None = None, split: str | None = None, val_fraction: float = 0.0,
                   seed: int = 0) -> Iterator[tuple[str, StreamCursor]]:
        """Like :meth:`iter_rows` but yields ``(training_text, cursor_after_row)``."""
        for row, cur in self.iter_rows(cursor, split, val_fraction, seed):
            yield self.to_text(row), cur

    def preview(self, n: int = 20) -> DatasetPreview:
        """First ``n`` rows (``0 <= n <= 1000``), JSON-safe, long strings truncated to 500 chars.

        Does not change ``stats``.
        """
        if not 0 <= n <= PREVIEW_MAX_ROWS:
            raise ValueError(f"n must be between 0 and {PREVIEW_MAX_ROWS}")
        rows: list[dict[str, Any]] = []
        truncated = False
        if n:
            gen = iter_with_sink(self.connector, None, lambda rec: None)
            try:
                for batch, _ in gen:
                    for row in batch.slice(0, n - len(rows)).to_pylist():
                        safe = json.loads(json.dumps(row, default=str, ensure_ascii=False))
                        safe, cut = _truncate(safe)
                        truncated |= cut
                        rows.append(safe)
                    if len(rows) >= n:
                        break
            finally:
                gen.close()  # type: ignore[attr-defined]
        columns = [c.name for c in self.info.columns]
        for r in rows:
            columns += [k for k in r if k not in columns]
        return DatasetPreview(columns=columns, rows=rows, truncated_values=truncated)

    # ------------------------------------------------------------------ scan
    def scan(self, progress_cb: Callable[[StreamStats], None] | None = None) -> StreamStats:
        """Stream the whole source once, validating and counting rows without keeping them.

        On exhaustion sets ``info.cardinality = Cardinality(rows, exact=True, method="scan")``
        where ``rows`` is the number of valid rows. Raises ``DatasetEmpty`` /
        ``DatasetMalformed`` if no valid row exists. Returns the final stats
        (state ``"cancelled"`` if :meth:`cancel` was called; cardinality unchanged then).
        """
        last = 0.0
        for _ in self._pass(None, None):
            if progress_cb is not None:
                now = time.monotonic()
                if now - last >= 0.2:
                    last = now
                    progress_cb(self.stats)
        final = self.stats
        if final.state == "exhausted":
            try:
                ensure_usable(final.rows_seen, final.malformed_rows, final.malformed_examples,
                              self.info.spec.name or self.info.spec.path or "?")
            except Exception as e:
                self._set(state="failed", error=f"{type(e).__name__}: {e}")
                self._publish(force=True)
                raise
            self.info = self.info.model_copy(
                update={"cardinality": Cardinality(rows=final.rows_seen, exact=True, method="scan")})
        if progress_cb is not None:
            progress_cb(self.stats)
        return self.stats

    # ------------------------------------------------------------------ internals
    def _set(self, **fields: Any) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(self._stats, k, v)

    def _publish(self, force: bool = False) -> None:
        if self.events is None:
            return
        now = time.monotonic()
        if not force and now - self._last_publish < self.publish_interval:
            return
        self._last_publish = now
        try:
            self.events.publish("dataset", {"dataset_id": self.info.dataset_id, **self.stats.model_dump()})
        except Exception:  # an event sink must never break streaming
            log.exception("failed to publish dataset event")

    def _pass(self, cursor: StreamCursor | None,
              keep: Callable[[dict[str, Any]], bool] | None) -> Iterator[tuple[dict[str, Any], StreamCursor]]:
        self._cancel.clear()
        validator = RowValidator(self.info)
        total = self.connector.total_bytes()
        kind = self.connector.kind
        prev = cursor if cursor is not None else StreamCursor(kind=kind)
        start_bytes = int(prev.state.get("bytes", 0))
        with self._lock:
            self._stats = StreamStats(state="streaming", bytes_seen=start_bytes,
                                      progress=min(1.0, start_bytes / total) if total else None)
        self._publish(force=True)

        def sink(rec: MalformedRecord) -> None:
            with self._lock:
                validator.record_malformed(rec.describe())
                self._stats.malformed_rows = validator.malformed_rows
                self._stats.malformed_examples = list(validator.malformed_examples)

        def cancelled() -> bool:
            if self._cancel.is_set():
                self._set(state="cancelled", checksum=validator.checksum)
                self._publish(force=True)
                return True
            return False

        gen = iter_with_sink(self.connector, cursor, sink)
        nbytes = start_bytes
        try:
            for batch, after in gen:
                if cancelled():
                    return
                rows = batch.to_pylist()
                n = len(rows)
                base = prev.rows_consumed
                if after.rows_consumed != base + n:
                    raise RuntimeError(f"connector cursor mismatch: {after.rows_consumed} != {base} + {n}")
                nbytes = int(after.state["bytes"]) if "bytes" in after.state else nbytes + batch.nbytes
                for i, row in enumerate(rows):
                    row_cur = after if i == n - 1 else StreamCursor(kind=kind, rows_consumed=base + i + 1,
                                                                    state=prev.state)
                    with self._lock:
                        ok = validator.check(row, f"row {base + i}")
                        if ok:
                            self._stats.rows_seen += 1
                        else:
                            self._stats.malformed_rows = validator.malformed_rows
                            self._stats.malformed_examples = list(validator.malformed_examples)
                    if ok and (keep is None or keep(row)):
                        yield row, row_cur
                        if cancelled():
                            return
                prev = after
                with self._lock:
                    s = self._stats
                    s.batches_seen += 1
                    s.bytes_seen = nbytes
                    s.checksum = validator.checksum
                    s.progress = min(1.0, nbytes / total) if total else None
                self._publish()
            self._set(state="exhausted", checksum=validator.checksum,
                      progress=1.0 if total else None)
            self._publish(force=True)
        except GeneratorExit:  # consumer stopped early: not a failure
            self._set(state="idle", checksum=validator.checksum)
            raise
        except BaseException as e:
            self._set(state="failed", error=f"{type(e).__name__}: {e}", checksum=validator.checksum)
            self._publish(force=True)
            raise
        finally:
            gen.close()  # type: ignore[attr-defined]
