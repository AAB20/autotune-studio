"""Regression: training must work while a WebSocket client is subscribed to the event bus.

``datasets.IterableDataset.from_generator`` fingerprints the generator by pickling it. The streaming
dataset holds the EventBus, whose subscribers hold asyncio loops (unpicklable). Found in the browser
E2E check: "TypeError: cannot pickle '_OverlappedFuture' object".
"""

from __future__ import annotations

import asyncio
import json
import pickle
import threading

import pytest

from autotune_studio.contracts import DatasetSource, DatasetSpec
from autotune_studio.core.events import EventBus
from autotune_studio.data_engine import open_dataset
from autotune_studio.training.data import DataPosition, PositionRegistry, TrainStream


def _dataset(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("".join(json.dumps({"text": f"row {i}"}) + "\n" for i in range(10)), encoding="utf-8")
    bus = EventBus()
    loop = asyncio.new_event_loop()
    bus._subscribers.add((loop, asyncio.Queue()))  # what a connected WebSocket client looks like
    ds = open_dataset(DatasetSpec(source=DatasetSource.JSONL, path=str(p)), events=bus)
    ds._unpicklable = threading.Lock()  # make sure nothing tries to copy the live object graph
    return ds, loop


def test_train_stream_is_hashable_with_live_subscribers(tmp_path):
    datasets = pytest.importorskip("datasets")
    ds, loop = _dataset(tmp_path)
    try:
        stream = TrainStream(ds, lambda row: [1, 2, 3], PositionRegistry(), DataPosition(0, None), 0.0, 0,
                             threading.Event())
        it = datasets.IterableDataset.from_generator(stream)
        first = next(iter(it))
        assert first["input_ids"] == [1, 2, 3]
        assert pickle.dumps(stream)  # hashing state only
        with pytest.raises(RuntimeError):
            pickle.loads(pickle.dumps(stream))
    finally:
        loop.close()
