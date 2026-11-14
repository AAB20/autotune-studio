"""Dataset streaming at scale through the Studio facade (zero row configuration, bounded memory).

Generates ~150 MB JSONL and ~100 MB CSV files, then checks: unknown cardinality before the scan,
exact afterwards, every row counted, checksum stable across passes, process RSS growth far below
the file size, and the files are byte-identical afterwards.
"""

from __future__ import annotations

import gc
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
import pytest

from ..conftest import sha256_file

pytestmark = pytest.mark.slow

MiB = 1024 * 1024
PAD = "lorem ipsum dolor sit amet " * 36  # ~1 KB per row


def _write_jsonl(path: Path, target: int) -> int:
    rows = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        while f.tell() < target:
            f.write("".join(json.dumps({"id": rows + i, "text": f"{rows + i} {PAD}"}) + "\n" for i in range(4000)))
            rows += 4000
    return rows


def _write_csv(path: Path, target: int) -> int:
    rows = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("instruction,output\n")
        while f.tell() < target:
            f.write("".join(f"question {rows + i},answer {rows + i} {PAD}\n" for i in range(4000)))
            rows += 4000
    return rows


class PeakRSS:
    """Samples this process' RSS in a background thread; ``peak - baseline`` is the growth."""

    def __init__(self, interval: float = 0.02) -> None:
        self.proc = psutil.Process()
        self.interval = interval
        gc.collect()
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            self._stop.wait(self.interval)

    def __enter__(self) -> PeakRSS:
        self._t.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def growth(self) -> int:
        return self.peak - self.baseline


@pytest.mark.parametrize("fmt, target, writer", [
    ("jsonl", 150 * MiB, _write_jsonl),
    ("csv", 150 * MiB, _write_csv),
])
def test_scan_large_file_exact_count_stable_checksum_bounded_rss(
    studio, tmp_path: Path, fmt: str, target: int, writer: Callable[[Path, int], int]
) -> None:
    path = tmp_path / "data" / f"big.{fmt}"
    path.parent.mkdir(parents=True)
    n_rows = writer(path, target)
    size = path.stat().st_size
    assert size >= target
    file_hash = sha256_file(path)

    with PeakRSS() as mem:
        info = studio.load_dataset(fmt, path=str(path))  # no row count supplied anywhere
        assert info.cardinality.rows is None and not info.cardinality.exact
        assert info.sampled_rows < n_rows  # discovery read a byte-bounded sample only

        s1 = studio.scan_dataset(info.dataset_id)
    assert s1.state == "exhausted", s1
    assert s1.rows_seen == n_rows
    assert s1.malformed_rows == 0
    assert s1.checksum
    after = studio.get_dataset(info.dataset_id).info.cardinality
    assert after.rows == n_rows and after.exact and after.method == "scan"
    assert studio.list_datasets()[-1].cardinality.rows == n_rows

    # Memory: bounded, not proportional to the file size.
    msg = f"RSS grew {mem.growth / MiB:.0f} MiB for a {size / MiB:.0f} MiB {fmt} file"
    if fmt == "jsonl":
        assert mem.growth < size * 0.5, msg
    else:
        # pyarrow's CSV reader has a fixed ~150-235 MiB RSS overhead that plateaus with file size (measured
        # 146/235/233 MiB for 100/300/600 MB files on the dev box), so the bound is the engine's own data
        # memory budget rather than a fraction of this (small) file.
        from autotune_studio.data_engine import MemoryBudget

        assert mem.growth < MemoryBudget.from_system().total_bytes + 64 * MiB, msg

    s2 = studio.scan_dataset(info.dataset_id)
    assert s2.rows_seen == n_rows and s2.checksum == s1.checksum

    assert sha256_file(path) == file_hash


def test_background_scan_progress_events_and_cancel(studio, tmp_path: Path) -> None:
    path = tmp_path / "data" / "big.jsonl"
    path.parent.mkdir(parents=True)
    _write_jsonl(path, 60 * MiB)
    info = studio.load_dataset("jsonl", path=str(path))
    seq0 = studio.events.last_seq
    studio.scan_dataset(info.dataset_id, background=True)
    from .conftest import wait_until

    wait_until(lambda: studio.dataset_stats(info.dataset_id).rows_seen > 0, timeout=60, interval=0.005)
    st = studio.cancel_dataset(info.dataset_id)
    t = studio._scan_threads[info.dataset_id]
    t.join(30)
    assert not t.is_alive()
    final = studio.dataset_stats(info.dataset_id)
    assert final.state in ("cancelled", "exhausted"), final
    if final.state == "cancelled":
        assert studio.get_dataset(info.dataset_id).info.cardinality.rows is None  # unchanged on cancel
    assert st is not None
    evs = [e for e in studio.events.replay(seq0)[0] if e.type == "dataset"]
    assert evs and all(e.data["dataset_id"] == info.dataset_id for e in evs)
