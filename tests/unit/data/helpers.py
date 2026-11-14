"""Synthetic dataset helpers for data-engine tests (files are written under tmp_path)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from autotune_studio.contracts import DatasetSpec
from autotune_studio.data_engine import MemoryBudget, StreamingDataset, discover
from autotune_studio.interfaces import Connector

SMALL = 64 * 1024  # connectors clamp batch_bytes to >= 64 KiB


def make_rows(n: int, pad: int = 150) -> list[dict[str, Any]]:
    """Instruction-style rows with a stable payload (~pad bytes each)."""
    return [{"id": i, "instruction": f"question {i} " + "q" * (pad // 2), "output": f"answer {i} " + "a" * (pad // 2)}
            for i in range(n)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def write_parquet(path: Path, rows: list[dict[str, Any]], row_group_size: int = 300) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)
    return path


def small_budget() -> MemoryBudget:
    return MemoryBudget(total_bytes=64 * 1024 * 1024, batch_bytes=SMALL, sample_bytes=SMALL,
                        shuffle_buffer_bytes=SMALL)


def dataset_from(connector: Connector, spec: DatasetSpec, events: Any = None) -> StreamingDataset:
    info = discover(connector, spec, SMALL)
    return StreamingDataset(info, connector, budget=small_budget(), events=events)
