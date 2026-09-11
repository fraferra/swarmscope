"""Async, batched, best-effort event writer.

Instrumentation that can block a swarm is worse than no instrumentation, so:
- ``emit`` is a single ``deque.append`` (atomic under the GIL, no lock, no
  condition variable, no writer wake-up). If the bounded deque is full the
  event is dropped and counted (``dropped``); the count is surfaced in every
  report.
- a daemon thread wakes every ``flush_interval`` seconds, drains everything
  in one go and writes it in batches. Store errors are logged and swallowed.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Sequence

from ..store.base import Store
from .events import Event

log = logging.getLogger("swarmscope")


class EventBuffer:
    def __init__(
        self,
        store: Store,
        *,
        batch_size: int = 512,
        flush_interval: float = 0.25,
        max_queue: int = 100_000,
    ) -> None:
        self.store = store
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue = max_queue
        self._q: deque[Event] = deque()
        self.dropped = 0
        self.written = 0
        self.errors = 0
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._drain_lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="swarmscope-writer", daemon=True)
        self._thread.start()

    # public --------------------------------------------------------------
    def emit(self, event: Event) -> None:
        q = self._q
        if len(q) >= self.max_queue:
            self.dropped += 1
            d = self.dropped
            if d in (1, 100, 10_000) or d % 100_000 == 0:
                log.warning("swarmscope: event buffer full, dropped %d events so far", d)
            return
        q.append(event)

    def emit_many(self, events: Sequence[Event]) -> None:
        for e in events:
            self.emit(e)

    def flush(self, timeout: float = 5.0) -> None:
        """Synchronously hand everything emitted so far to the store."""
        self._drain()
        try:
            self.store.flush()
        except Exception:  # pragma: no cover
            log.exception("swarmscope: store.flush failed")

    def close(self, timeout: float = 5.0) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout)
        self._drain()

    @property
    def pending(self) -> int:
        return len(self._q)

    # internals -------------------------------------------------------------
    def _drain(self) -> None:
        with self._drain_lock:
            q = self._q
            while q:
                batch: list[Event] = []
                pop = q.popleft
                try:
                    for _ in range(self.batch_size):
                        batch.append(pop())
                except IndexError:
                    pass
                self._write(batch)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.flush_interval)
            self._wake.clear()
            if self._q:
                self._drain()

    def _write(self, batch: list[Event]) -> None:
        if not batch:
            return
        try:
            self.store.write(batch)
            self.written += len(batch)
        except Exception:
            self.errors += 1
            if self.errors <= 3 or self.errors % 1000 == 0:
                log.exception("swarmscope: store.write failed (%d events lost)", len(batch))
