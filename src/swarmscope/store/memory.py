"""In-memory store. Useful for tests and short-lived scripts."""
from __future__ import annotations

import array
import threading
import time
from typing import Any, Iterable, Sequence

from ..core.events import Event
from .base import RouteOutcome, RouteStat, RunInfo, VectorHit, cosine


class MemoryStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: list[Event] = []
        self._runs: dict[str, RunInfo] = {}
        self._vectors: dict[str, dict[str, Any]] = {}
        self._judge: dict[str, dict[str, Any]] = {}
        self._calib: dict[str, dict[str, Any]] = {}
        self._outcomes: dict[tuple[str, str], RouteOutcome] = {}
        self._routes: dict[str, RouteStat] = {}

    # runs
    def begin_run(self, run_id, name, started_at, meta):
        with self._lock:
            self._runs[run_id] = RunInfo(run_id, name, started_at, None, dict(meta))

    def end_run(self, run_id, ended_at):
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].ended_at = ended_at

    def runs(self):
        with self._lock:
            out = []
            for r in self._runs.values():
                r.event_count = sum(1 for e in self._events if e.run_id == r.run_id)
                out.append(r)
            return sorted(out, key=lambda r: r.started_at or 0, reverse=True)

    def run(self, run_id):
        with self._lock:
            r = self._runs.get(run_id)
            if r is None and any(e.run_id == run_id for e in self._events):
                r = RunInfo(run_id, None, None, None, {})
            if r:
                r.event_count = sum(1 for e in self._events if e.run_id == run_id)
            return r

    # events
    def write(self, events: Sequence[Event]) -> None:
        with self._lock:
            self._events.extend(events)

    def events(self, run_id, types: Iterable[str] | None = None, agent_id=None, limit=None):
        ts = set(types) if types else None
        with self._lock:
            out = [
                e for e in self._events
                if e.run_id == run_id
                and (ts is None or e.type in ts)
                and (agent_id is None or e.agent_id == agent_id)
            ]
        out.sort(key=lambda e: (e.ts, e.event_id))
        return out[:limit] if limit else out

    # vectors
    def upsert_vector(self, run_id, claim_id, vector, text, kind, metadata, agent_id):
        with self._lock:
            self._vectors[claim_id] = dict(
                run_id=run_id, vector=array.array("f", vector), text=text, kind=kind,  # packed: 4 B/dim
                metadata=dict(metadata), agent_id=agent_id,
            )

    def upsert_vectors(self, rows):
        for r in rows:
            self.upsert_vector(*r)

    def search_vectors(self, vector, k, run_id=None, kind=None):
        with self._lock:
            rows = list(self._vectors.items())
        hits = []
        for cid, r in rows:
            if run_id is not None and r["run_id"] != run_id:
                continue
            if kind is not None and r["kind"] != kind:
                continue
            hits.append(VectorHit(cid, cosine(vector, r["vector"]), r["text"], r["kind"],
                                  r["metadata"], r["run_id"], r["agent_id"]))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def vector_count(self, run_id=None):
        with self._lock:
            return sum(1 for r in self._vectors.values() if run_id is None or r["run_id"] == run_id)

    # judge cache
    def judge_cache_get(self, key):
        return self._judge.get(key)

    def judge_cache_put(self, key, value):
        self._judge[key] = dict(value, ts=time.time())

    # reputation
    def route_outcomes_put(self, rows):
        with self._lock:
            for r in rows:
                self._outcomes[(r.request_id, r.artifact_id)] = r

    def route_outcomes(self, request_ids=None, limit=None):
        ids = set(request_ids) if request_ids is not None else None
        with self._lock:
            out = [o for o in self._outcomes.values() if ids is None or o.request_id in ids]
        out.sort(key=lambda o: o.ts, reverse=True)
        return out[:limit] if limit else out

    def route_stats_add(self, route_key, route, success, failure, ts):
        with self._lock:
            st = self._routes.get(route_key)
            if st is None:
                st = self._routes[route_key] = RouteStat(route_key, list(route), 0.0, 0.0, ts)
            st.successes += success
            st.failures += failure
            st.last_ts = max(st.last_ts, ts)

    def route_stats(self, limit=None):
        with self._lock:
            out = sorted(self._routes.values(), key=lambda s: (s.successes / (s.n or 1), s.n), reverse=True)
        return out[:limit] if limit else out

    def route_stats_clear(self):
        with self._lock:
            self._routes.clear()
            self._outcomes.clear()

    # calibrations
    def calibration_get(self, workload):
        return self._calib.get(workload)

    def calibration_put(self, workload, value):
        self._calib[workload] = dict(value)

    def flush(self):
        pass

    def close(self):
        pass
