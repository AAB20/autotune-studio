"""Background resource sampler, bounded metric history and ETA estimation.

Owned by: gpu-memory-optimization.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from ..contracts import MetricPoint
from ..interfaces import EventSink, ResourceMonitorProtocol

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- sampler
class ResourceSampler:
    """Daemon thread publishing ``"resource"`` events every ``interval`` seconds.

    Event data is ``monitor.snapshot().model_dump()``. The first sample is
    published immediately after :meth:`start`. :meth:`start` and :meth:`stop`
    are idempotent; a stopped sampler may be started again. :meth:`stop` wakes
    the thread at once, so it returns well within one interval. Exceptions from
    the monitor or sink are logged and never kill the thread.
    """

    def __init__(self, monitor: ResourceMonitorProtocol, sink: EventSink, interval: float = 2.0) -> None:
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self.monitor = monitor
        self.sink = sink
        self.interval = float(interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run, args=(self._stop,), name="autotune-resource-sampler", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Signal the thread and join it (default timeout: one interval + 1 s)."""
        with self._lock:
            thread, self._thread = self._thread, None
            self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(self.interval + 1.0 if timeout is None else timeout)

    def sample_once(self) -> None:
        """Take and publish one snapshot synchronously (errors are logged, not raised)."""
        try:
            snap = self.monitor.snapshot()
            self.sink.publish("resource", snap.model_dump())
        except Exception as exc:  # noqa: BLE001
            log.warning("resource sampling failed: %s", exc)

    def _run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.sample_once()
            stop.wait(self.interval)


# --------------------------------------------------------------------------- metric history
class MetricsStore:
    """Thread-safe, bounded history of :class:`MetricPoint` for charts.

    Never holds more than ``max_points``. When full, the *older half* is thinned
    by keeping every second point (the very first point is always kept); the
    newer half stays at full resolution and the latest point is always kept.
    Repeated thinning makes resolution degrade gracefully with age.

    Resume handling: a point whose ``step`` is <= the latest stored step means
    training rewound to a checkpoint; stored points with ``step >= point.step``
    are replaced so the curve stays monotonic in step.
    """

    def __init__(self, max_points: int = 5000) -> None:
        if max_points < 4:
            raise ValueError("max_points must be >= 4")
        self.max_points = int(max_points)
        self._lock = threading.Lock()
        self._points: list[MetricPoint] = []

    def add(self, point: MetricPoint) -> None:
        with self._lock:
            pts = self._points
            if pts and point.step <= pts[-1].step:
                # Rewind (resume from checkpoint): drop the superseded tail.
                i = len(pts)
                while i > 0 and pts[i - 1].step >= point.step:
                    i -= 1
                del pts[i:]
            if len(pts) >= self.max_points:
                self._downsample()
            pts.append(point)

    def _downsample(self) -> None:
        pts = self._points
        split = len(pts) // 2
        self._points = pts[:split:2] + pts[split:]

    def list(self, since_step: int | None = None) -> list[MetricPoint]:
        """Copy of stored points; with ``since_step``, only points with ``step > since_step``."""
        with self._lock:
            if since_step is None:
                return list(self._points)
            return [p for p in self._points if p.step > since_step]

    def latest(self) -> MetricPoint | None:
        with self._lock:
            return self._points[-1] if self._points else None

    def clear(self) -> None:
        with self._lock:
            self._points = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._points)


# --------------------------------------------------------------------------- ETA
class EtaEstimator:
    """Estimate remaining training time from recent ``(step, timestamp)`` observations.

    The rate is measured over a sliding window of the last ``window`` updates
    (steps per second between the oldest and newest observation in the window).
    :meth:`eta_seconds` returns None when ``max_steps`` is unknown or fewer than
    two observations with progress in both step and time exist. If the step goes
    backwards (resume) the window is reset.
    """

    def __init__(self, max_steps: int | None = None, window: int = 50) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.max_steps = max_steps
        self._lock = threading.Lock()
        self._obs: deque[tuple[int, float]] = deque(maxlen=window)

    def set_max_steps(self, max_steps: int | None) -> None:
        with self._lock:
            self.max_steps = max_steps

    def reset(self) -> None:
        with self._lock:
            self._obs.clear()

    def update(self, step: int, timestamp: float | None = None) -> None:
        ts = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            if self._obs and step < self._obs[-1][0]:
                self._obs.clear()
            if self._obs and step == self._obs[-1][0]:
                return  # duplicate step: keep the earliest timestamp
            self._obs.append((int(step), ts))

    def steps_per_second(self) -> float | None:
        with self._lock:
            return self._rate()

    def _rate(self) -> float | None:
        if len(self._obs) < 2:
            return None
        (s0, t0), (s1, t1) = self._obs[0], self._obs[-1]
        if s1 <= s0 or t1 <= t0:
            return None
        return (s1 - s0) / (t1 - t0)

    def eta_seconds(self) -> float | None:
        with self._lock:
            if not self.max_steps or not self._obs:
                return None
            remaining = self.max_steps - self._obs[-1][0]
            if remaining <= 0:
                return 0.0
            rate = self._rate()
            if rate is None:
                return None
            return remaining / rate
