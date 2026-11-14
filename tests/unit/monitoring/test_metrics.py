"""ResourceSampler, MetricsStore, EtaEstimator."""

from __future__ import annotations

import threading
import time

import pytest

from autotune_studio.contracts import MetricPoint, ResourceSnapshot
from autotune_studio.monitoring import EtaEstimator, MetricsStore, ResourceMonitor, ResourceSampler


class FakeSink:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []
        self.lock = threading.Lock()
        self.got = threading.Event()

    def publish(self, type, data):
        with self.lock:
            self.events.append((type, data))
        self.got.set()


class FakeMonitor:
    def __init__(self, fail_first: bool = False):
        self.calls = 0
        self.fail_first = fail_first

    def snapshot(self):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("boom")
        return ResourceSnapshot(cpu_pct=1.0, ram_used_bytes=1, ram_total_bytes=2)


def _wait_for(pred, timeout=5.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.01)
    return pred()


# --------------------------------------------------------------------------- sampler
def test_sampler_publishes_and_stops_quickly():
    sink = FakeSink()
    s = ResourceSampler(ResourceMonitor(probe_gpu=False), sink, interval=0.05)
    s.start()
    s.start()  # idempotent: still one thread
    assert _wait_for(lambda: len(sink.events) >= 3)
    t0 = time.perf_counter()
    s.stop()
    assert time.perf_counter() - t0 < 1.0
    assert not s.running
    s.stop()  # idempotent
    n = len(sink.events)
    time.sleep(0.15)
    assert len(sink.events) == n
    typ, data = sink.events[0]
    assert typ == "resource"
    ResourceSnapshot.model_validate(data)
    assert sum(t.name == "autotune-resource-sampler" for t in threading.enumerate()) == 0


def test_sampler_stop_is_prompt_with_long_interval():
    sink = FakeSink()
    s = ResourceSampler(FakeMonitor(), sink, interval=10.0)
    s.start()
    assert sink.got.wait(2.0)  # first sample is immediate
    t0 = time.perf_counter()
    s.stop()
    assert time.perf_counter() - t0 < 1.0  # stop wakes the wait instead of sleeping out 10 s
    assert not s.running


def test_sampler_survives_monitor_errors_and_restarts():
    sink = FakeSink()
    mon = FakeMonitor(fail_first=True)
    s = ResourceSampler(mon, sink, interval=0.02)
    s.start()
    assert _wait_for(lambda: bool(sink.events))
    s.stop()
    assert mon.calls >= 2
    s.start()
    assert s.running
    s.stop()
    assert not s.running


def test_sampler_rejects_bad_interval():
    with pytest.raises(ValueError):
        ResourceSampler(FakeMonitor(), FakeSink(), interval=0)


# --------------------------------------------------------------------------- store
def test_store_basic():
    st = MetricsStore(max_points=10)
    assert st.latest() is None and st.list() == []
    for i in range(1, 6):
        st.add(MetricPoint(step=i, loss=float(i)))
    assert [p.step for p in st.list()] == [1, 2, 3, 4, 5]
    assert [p.step for p in st.list(since_step=3)] == [4, 5]
    assert st.latest().step == 5


def test_store_bounded_and_downsamples_keeping_first_and_last():
    st = MetricsStore(max_points=100)
    for i in range(1, 10_001):
        st.add(MetricPoint(step=i, loss=1.0 / i))
        assert len(st) <= 100
    steps = [p.step for p in st.list()]
    assert steps[0] == 1
    assert steps[-1] == 10_000 == st.latest().step
    assert steps == sorted(steps) and len(set(steps)) == len(steps)
    assert steps[-5:] == [9996, 9997, 9998, 9999, 10_000]  # recent region at full resolution
    assert len(steps) >= 50


def test_store_rewind_on_resume():
    st = MetricsStore(max_points=50)
    for i in range(1, 11):
        st.add(MetricPoint(step=i))
    st.add(MetricPoint(step=6, loss=0.5))  # resumed from a checkpoint at step 5
    assert [p.step for p in st.list()] == [1, 2, 3, 4, 5, 6]
    assert st.latest().loss == 0.5


def test_store_thread_safety():
    st = MetricsStore(max_points=64)
    counter = iter(range(1, 100_000))
    lock = threading.Lock()

    def worker():
        for _ in range(500):
            with lock:
                st.add(MetricPoint(step=next(counter)))
            st.list()
            st.latest()

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(st) <= 64
    assert st.latest().step == 2000


def test_store_min_size():
    with pytest.raises(ValueError):
        MetricsStore(max_points=2)


# --------------------------------------------------------------------------- ETA
def test_eta_unknown_cases():
    e = EtaEstimator(max_steps=None)
    e.update(1, 0.0)
    e.update(2, 1.0)
    assert e.eta_seconds() is None  # max_steps unknown
    e2 = EtaEstimator(max_steps=100)
    assert e2.eta_seconds() is None
    e2.update(1, 0.0)
    assert e2.eta_seconds() is None  # single observation
    e2.update(1, 5.0)  # duplicate step ignored
    assert e2.eta_seconds() is None


def test_eta_constant_rate():
    e = EtaEstimator(max_steps=100)
    for s in range(11):
        e.update(s, s * 0.5)  # 2 steps/s
    assert e.steps_per_second() == pytest.approx(2.0)
    assert e.eta_seconds() == pytest.approx(45.0)  # 90 steps left


def test_eta_window_tracks_recent_rate_and_done():
    e = EtaEstimator(max_steps=40, window=5)
    t = 0.0
    for s in range(10):
        e.update(s, t)
        t += 1.0  # 1 step/s
    for s in range(10, 20):
        e.update(s, t)
        t += 0.25  # 4 steps/s
    assert e.steps_per_second() == pytest.approx(4.0)
    assert e.eta_seconds() == pytest.approx(21 / 4.0)
    e.update(40, t + 1)
    assert e.eta_seconds() == 0.0


def test_eta_reset_on_rewind_and_set_max_steps():
    e = EtaEstimator()
    e.update(10, 0.0)
    e.update(20, 10.0)
    e.update(5, 11.0)  # resume rewound the step counter
    assert e.steps_per_second() is None
    e.update(15, 16.0)
    e.set_max_steps(25)
    assert e.eta_seconds() == pytest.approx(5.0)
