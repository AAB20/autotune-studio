"""StreamingDataset: resume, cancel, deterministic split, scan, failure, events, thread-safety."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from autotune_studio.connectors import HuggingFaceConnector, make_connector
from autotune_studio.contracts import Cardinality, DatasetSpec, StreamStats
from autotune_studio.data_engine import StreamingDataset, open_dataset, rows_checksum, split_of
from autotune_studio.errors import DatasetEmpty
from autotune_studio.interfaces import StreamCursor, StreamingDatasetProtocol

from .helpers import (
    SMALL,
    dataset_from,
    make_rows,
    small_budget,
    write_csv,
    write_json_array,
    write_jsonl,
    write_parquet,
)

N = 2000
WRITERS = {"csv": write_csv, "jsonl": write_jsonl, "json": write_json_array, "parquet": write_parquet}


def _dataset(tmp_path: Path, kind: str, rows: list[dict[str, Any]], events: Any = None) -> StreamingDataset:
    if kind == "huggingface":
        p = write_jsonl(tmp_path / "hf.jsonl", rows)
        spec = DatasetSpec(source="huggingface", name="json")
        return dataset_from(HuggingFaceConnector("json", SMALL, data_files=str(p)), spec, events)
    p = WRITERS[kind](tmp_path / f"d.{kind}", rows)
    spec = DatasetSpec(source=kind, path=str(p))
    return dataset_from(make_connector(spec, SMALL), spec, events)


def _roundtrip(c: StreamCursor) -> StreamCursor:
    return StreamCursor.from_dict(json.loads(json.dumps(c.to_dict())))


@pytest.mark.parametrize("kind", ["csv", "jsonl", "json", "parquet", "huggingface"])
def test_resume_from_cursor_equals_remainder(tmp_path: Path, kind: str) -> None:
    ds = _dataset(tmp_path, kind, make_rows(N))
    assert isinstance(ds, StreamingDatasetProtocol)
    picks = {1, 437, 1000, 1999}  # 1-based row positions, mid-batch and near the end
    cursors: dict[int, StreamCursor] = {}
    boundary: StreamCursor | None = None
    full: list[dict[str, Any]] = []
    for i, (row, cur) in enumerate(ds.iter_rows(), start=1):
        full.append(row)
        if i in picks:
            cursors[i] = _roundtrip(cur)
        if boundary is None and cur.state.get("base_rows", -1) == cur.rows_consumed and cur.rows_consumed > 0 \
                or (boundary is None and kind in ("csv", "json", "parquet") and i == 600):
            boundary = _roundtrip(cur)
    assert len(full) == N
    first_stats = ds.stats
    assert first_stats.state == "exhausted" and first_stats.rows_seen == N
    assert first_stats.checksum == rows_checksum(full)
    cursors[boundary.rows_consumed] = boundary
    for k, cur in cursors.items():
        assert cur.rows_consumed == k
        resumed = (row for row, _ in ds.iter_rows(cursor=cur))
        assert rows_checksum(resumed) == rows_checksum(full[k:]), f"resume at {k}"
        assert ds.stats.rows_seen == N - k
    # second full pass -> identical checksum
    assert rows_checksum(r for r, _ in ds.iter_rows()) == first_stats.checksum
    assert ds.stats.checksum == first_stats.checksum


def test_resume_jsonl_with_malformed_lines_counts_once(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    lines = [json.dumps({"text": f"t{i}" + "x" * 100}) for i in range(1500)]
    lines.insert(10, "{bad")
    lines.insert(1200, "{bad again")
    p.write_text("\n".join(lines) + "\n")
    ds = open_dataset(DatasetSpec(source="jsonl", path=str(p)), budget=small_budget())
    full, cur_at = [], None
    for i, (row, cur) in enumerate(ds.iter_rows(), start=1):
        full.append(row)
        if i == 700:
            cur_at = _roundtrip(cur)
    assert len(full) == 1500 and ds.stats.malformed_rows == 2
    assert cur_at is not None
    assert rows_checksum(r for r, _ in ds.iter_rows(cursor=cur_at)) == rows_checksum(full[700:])
    assert ds.stats.malformed_rows == 1  # only the bad line after the cursor


def test_cancel_stops_iteration(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, "jsonl", make_rows(N))
    it = ds.iter_rows()
    for _ in range(10):
        next(it)
    ds.cancel()
    assert list(it) == []
    st = ds.stats
    assert st.state == "cancelled" and st.rows_seen < N
    # a new pass clears the cancel flag
    assert sum(1 for _ in ds.iter_rows()) == N


def test_cancel_from_other_thread_while_reading_stats(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, "jsonl", make_rows(20000))

    def worker() -> None:
        it = ds.iter_rows()
        for _ in it:
            time.sleep(0.0005)

    t = threading.Thread(target=worker)
    t.start()
    deadline = time.monotonic() + 10
    while ds.stats.rows_seen < 50 and time.monotonic() < deadline:
        time.sleep(0.005)
    snap = ds.stats  # read while the other thread iterates
    assert snap.state == "streaming" and snap.rows_seen >= 50
    ds.cancel()
    t.join(10)
    assert not t.is_alive()
    assert ds.stats.state == "cancelled" and ds.stats.rows_seen < 20000
    assert ds.info.cardinality.rows is None


def test_cancelled_scan_keeps_cardinality_unknown(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, "jsonl", make_rows(5000))
    st = ds.scan(progress_cb=lambda s: ds.cancel())
    assert st.state == "cancelled" and st.rows_seen < 5000
    assert ds.info.cardinality.rows is None


def test_deterministic_split(tmp_path: Path) -> None:
    rows = make_rows(N)
    ds = _dataset(tmp_path, "parquet", rows)
    train1 = [r["id"] for r, _ in ds.iter_rows(split="train", val_fraction=0.2, seed=7)]
    val1 = [r["id"] for r, _ in ds.iter_rows(split="validation", val_fraction=0.2, seed=7)]
    train2 = [r["id"] for r, _ in ds.iter_rows(split="train", val_fraction=0.2, seed=7)]
    val2 = [r["id"] for r, _ in ds.iter_rows(split="validation", val_fraction=0.2, seed=7)]
    assert train1 == train2 and val1 == val2
    assert set(train1).isdisjoint(val1)
    assert sorted(train1 + val1) == list(range(N))
    assert 0.15 < len(val1) / N < 0.25
    other_seed = [r["id"] for r, _ in ds.iter_rows(split="validation", val_fraction=0.2, seed=8)]
    assert other_seed != val1
    assert all(split_of(rows[i], 0.2, 7) == "validation" for i in val1)
    assert [r["id"] for r, _ in ds.iter_rows(split="validation", val_fraction=0.0)] == []
    with pytest.raises(ValueError):
        ds.iter_rows(split="test")


def test_split_resume(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, "jsonl", make_rows(N))
    train = list(ds.iter_rows(split="train", val_fraction=0.3, seed=1))
    k = len(train) // 2
    cur = _roundtrip(train[k - 1][1])
    rest = [r for r, _ in ds.iter_rows(cursor=cur, split="train", val_fraction=0.3, seed=1)]
    assert rows_checksum(rest) == rows_checksum(r for r, _ in train[k:])


@pytest.mark.parametrize("kind", ["jsonl", "csv", "parquet"])
def test_scan_sets_cardinality(tmp_path: Path, kind: str) -> None:
    ds = _dataset(tmp_path, kind, make_rows(1234))
    calls: list[StreamStats] = []
    st = ds.scan(progress_cb=calls.append)
    assert st.state == "exhausted" and st.rows_seen == 1234 and st.progress == 1.0
    assert ds.info.cardinality == Cardinality(rows=1234, exact=True, method="scan")
    assert calls and calls[-1].state == "exhausted"
    assert st.bytes_seen > 0 and st.batches_seen >= 1


def test_scan_of_emptied_file_raises(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, "jsonl", make_rows(10))
    Path(ds.info.spec.path or "").write_text("")
    with pytest.raises(DatasetEmpty):
        ds.scan()
    assert ds.stats.state == "failed"


class _FailingConnector:
    kind = "fail"

    def __init__(self) -> None:
        self.ok = pa.RecordBatch.from_pylist([{"text": f"r{i}"} for i in range(5)])
        self.calls = 0

    def iter_batches(self, cursor: StreamCursor | None = None) -> Iterator[tuple[pa.RecordBatch, StreamCursor]]:
        self.calls += 1  # first call (discovery) succeeds, later calls fail after one batch
        yield self.ok, StreamCursor(kind="fail", rows_consumed=5)
        if self.calls > 1:
            raise OSError("disk went away")

    def cardinality(self) -> Cardinality:
        return Cardinality()

    def total_bytes(self) -> int | None:
        return None


def test_failure_distinct_from_exhaustion() -> None:
    conn = _FailingConnector()
    spec = DatasetSpec(source="jsonl", path="x")
    from autotune_studio.data_engine import discover

    ds = StreamingDataset(discover(conn, spec, 1024), conn)
    got = []
    with pytest.raises(OSError, match="disk went away"):
        for row, _ in ds.iter_rows():
            got.append(row)
    assert len(got) == 5
    st = ds.stats
    assert st.state == "failed" and st.error and "disk went away" in st.error
    assert st.progress is None  # total size unknown


class _Sink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def publish(self, type: str, data: dict[str, Any]) -> None:
        self.events.append((type, data))


def test_events_published_and_throttled(tmp_path: Path) -> None:
    sink = _Sink()
    ds = _dataset(tmp_path, "jsonl", make_rows(N), events=sink)
    ds.publish_interval = 3600  # only forced (state-change) events
    ds.scan()
    assert [e[0] for e in sink.events] == ["dataset", "dataset"]
    assert sink.events[0][1]["state"] == "streaming"
    assert sink.events[-1][1]["state"] == "exhausted"
    assert sink.events[-1][1]["dataset_id"] == ds.info.dataset_id
    sink.events.clear()
    ds.publish_interval = 0.0
    ds.scan()
    assert len(sink.events) > 2  # periodic updates when not throttled


def test_preview(tmp_path: Path) -> None:
    rows = [{"text": "x" * 2000, "n": i} for i in range(50)]
    ds = _dataset(tmp_path, "jsonl", rows)
    before = ds.stats
    pv = ds.preview(5)
    assert len(pv.rows) == 5 and pv.columns == ["text", "n"]
    assert pv.truncated_values and len(pv.rows[0]["text"]) < 600
    assert ds.stats == before  # preview does not touch stats
    assert ds.preview(0).rows == []
    with pytest.raises(ValueError):
        ds.preview(10**6)


def test_early_break_sets_idle(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, "jsonl", make_rows(100))
    it = ds.iter_rows()
    next(it)
    it.close()
    assert ds.stats.state == "idle"


def test_iter_texts(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, "jsonl", make_rows(3))
    texts = [t for t, _ in ds.iter_texts()]
    assert texts[0].startswith("### Instruction:\nquestion 0")
