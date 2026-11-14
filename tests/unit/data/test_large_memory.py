"""Streams a generated >= 200 MB file and checks traced Python memory stays far below file size."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import pyarrow as pa
import pytest

from autotune_studio.contracts import DatasetSpec
from autotune_studio.data_engine import MemoryBudget, open_dataset
from autotune_studio.data_engine.memory_manager import MiB

TARGET_BYTES = 200 * MiB


def _write_big_jsonl(path: Path) -> int:
    pad = "lorem ipsum dolor sit amet " * 36  # ~1 KB per row
    rows = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        chunk: list[str] = []
        while f.tell() < TARGET_BYTES:
            for _ in range(2000):
                chunk.append(json.dumps({"id": rows, "text": f"{rows} {pad}"}))
                rows += 1
            f.write("\n".join(chunk) + "\n")
            chunk.clear()
    return rows


@pytest.mark.slow
def test_stream_200mb_jsonl_with_bounded_memory(tmp_path: Path) -> None:
    path = tmp_path / "big.jsonl"
    n_rows = _write_big_jsonl(path)
    size = path.stat().st_size
    assert size >= TARGET_BYTES

    budget = MemoryBudget(total_bytes=256 * MiB, batch_bytes=4 * MiB, sample_bytes=4 * MiB,
                          shuffle_buffer_bytes=16 * MiB)
    arrow_peak = 0
    # Warm up: the first Arrow <-> Python conversion imports ~400 modules (one-time,
    # ~25 MiB, independent of data size). Exclude that from the data-path measurement.
    pa.RecordBatch.from_pylist([{"id": 1, "text": "x"}]).to_pylist()
    t0 = time.perf_counter()
    tracemalloc.start()
    try:
        ds = open_dataset(DatasetSpec(source="jsonl", path=str(path)), budget=budget)
        count = 0
        for _row, _cur in ds.iter_rows():
            count += 1
            if count % 20000 == 0:
                arrow_peak = max(arrow_peak, pa.total_allocated_bytes())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    elapsed = time.perf_counter() - t0

    assert count == n_rows
    st = ds.stats
    assert st.state == "exhausted" and st.rows_seen == n_rows and st.malformed_rows == 0
    assert st.progress == 1.0 and st.bytes_seen == size
    print(f"\nfile={size / MiB:.0f} MiB rows={n_rows} python_peak={peak / MiB:.1f} MiB "
          f"arrow_sampled_peak={arrow_peak / MiB:.1f} MiB elapsed={elapsed:.1f}s")
    assert peak < 0.25 * size, f"traced peak {peak / MiB:.1f} MiB vs file {size / MiB:.0f} MiB"
    assert arrow_peak < 0.25 * size
