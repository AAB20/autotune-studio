"""Thread-safe event bus with bounded replay buffer. Owned by: Chief Architect.

Producers (training thread, data engine, monitors) call :meth:`EventBus.publish`
from any thread. Async consumers (WebSocket handlers) call :meth:`subscribe`.
Reconnecting clients pass the last ``seq`` they saw and receive missed events
from the replay buffer (bounded; a ``gap`` flag is set if events were evicted).
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from ..contracts import Event


class EventBus:
    def __init__(self, replay_size: int = 2000) -> None:
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._buffer: deque[Event] = deque(maxlen=replay_size)
        self._subscribers: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Event]]] = set()

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._buffer[-1].seq if self._buffer else 0

    def publish(self, type: str, data: dict[str, Any] | None = None) -> Event:
        with self._lock:
            ev = Event(seq=next(self._seq), type=type, data=data or {})  # type: ignore[arg-type]
            self._buffer.append(ev)
            subs = list(self._subscribers)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(_put_nowait_drop_oldest, q, ev)
            except RuntimeError:  # loop closed
                with self._lock:
                    self._subscribers.discard((loop, q))
        return ev

    def replay(self, since_seq: int = 0) -> tuple[list[Event], bool]:
        """Return events with seq > since_seq and whether some were already evicted (gap)."""
        with self._lock:
            events = [e for e in self._buffer if e.seq > since_seq]
            oldest = self._buffer[0].seq if self._buffer else None
        gap = bool(since_seq and oldest is not None and oldest > since_seq + 1)
        return events, gap

    async def subscribe(self, since_seq: int | None = None, maxsize: int = 1000) -> AsyncIterator[Event]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.add((loop, q))
            backlog = [e for e in self._buffer if since_seq is not None and e.seq > since_seq]
        try:
            last = 0
            for e in backlog:
                last = e.seq
                yield e
            while True:
                e = await q.get()
                if e.seq <= last:
                    continue
                last = e.seq
                yield e
        finally:
            with self._lock:
                self._subscribers.discard((loop, q))


def _put_nowait_drop_oldest(q: asyncio.Queue[Event], ev: Event) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(ev)
