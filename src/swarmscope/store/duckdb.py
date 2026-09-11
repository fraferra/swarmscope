"""DuckDB backend: analytics-friendly single file, fast scans over large runs.

Vector search is a brute-force cosine scan computed in SQL over ``FLOAT[]``
columns (DuckDB's ``list_cosine_similarity``), which is fine to ~10^6 claims.
Requires the ``duckdb`` extra.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Iterable, Sequence

from ..core.events import Event
from .base import RunInfo, VectorHit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (run_id VARCHAR PRIMARY KEY, name VARCHAR, started_at DOUBLE, ended_at DOUBLE, meta VARCHAR);
CREATE TABLE IF NOT EXISTS events (
  event_id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, type VARCHAR NOT NULL, ts DOUBLE NOT NULL,
  group_id VARCHAR, agent_id VARCHAR, parent_id VARCHAR, data VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_run_type ON events(run_id, type, ts);
CREATE TABLE IF NOT EXISTS claim_vectors (
  claim_id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, agent_id VARCHAR, kind VARCHAR, text VARCHAR,
  metadata VARCHAR, dim INTEGER, vec FLOAT[]
);
CREATE TABLE IF NOT EXISTS judge_cache (key VARCHAR PRIMARY KEY, value VARCHAR, ts DOUBLE);
CREATE TABLE IF NOT EXISTS calibrations (workload VARCHAR PRIMARY KEY, value VARCHAR, ts DOUBLE);
"""


class DuckDBStore:
    def __init__(self, path: str = "swarmscope.duckdb") -> None:
        import duckdb  # lazy

        self.path = path
        self._lock = threading.RLock()
        self._conn = duckdb.connect(path)
        for stmt in _SCHEMA.strip().split(";"):
            if stmt.strip():
                self._conn.execute(stmt)

    def _ex(self, sql: str, params: Sequence[Any] = ()) -> Any:
        with self._lock:
            return self._conn.execute(sql, list(params))

    # runs -------------------------------------------------------------------
    def begin_run(self, run_id, name, started_at, meta):
        self._ex("INSERT OR REPLACE INTO runs VALUES (?,?,?,NULL,?)", (run_id, name, started_at, json.dumps(meta, default=repr)))

    def end_run(self, run_id, ended_at):
        self._ex("UPDATE runs SET ended_at=? WHERE run_id=?", (ended_at, run_id))

    def runs(self):
        rows = self._ex("""SELECT r.run_id, r.name, r.started_at, r.ended_at, r.meta,
                                  (SELECT COUNT(*) FROM events e WHERE e.run_id=r.run_id)
                           FROM runs r ORDER BY r.started_at DESC""").fetchall()
        known = {r[0] for r in rows}
        orphan = self._ex("SELECT run_id, MIN(ts), MAX(ts), COUNT(*) FROM events GROUP BY run_id").fetchall()
        out = [RunInfo(r[0], r[1], r[2], r[3], json.loads(r[4] or "{}"), r[5]) for r in rows]
        out += [RunInfo(o[0], None, o[1], o[2], {}, o[3]) for o in orphan if o[0] not in known]
        return out

    def run(self, run_id):
        return next((r for r in self.runs() if r.run_id == run_id), None)

    # events -----------------------------------------------------------------
    def write(self, events: Sequence[Event]) -> None:
        if not events:
            return
        rows = [(e.event_id, e.run_id, e.type, e.ts, e.group_id, e.agent_id, e.parent_id,
                 json.dumps(e.to_dict(), default=repr, separators=(",", ":"))) for e in events]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?)", rows)

    def events(self, run_id, types: Iterable[str] | None = None, agent_id=None, limit=None):
        sql, params = "SELECT data FROM events WHERE run_id=?", [run_id]
        if types:
            ts = list(types)
            sql += f" AND type IN ({','.join('?' * len(ts))})"
            params += ts
        if agent_id is not None:
            sql += " AND agent_id=?"
            params.append(agent_id)
        sql += " ORDER BY ts, event_id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [Event.from_dict(json.loads(r[0])) for r in self._ex(sql, params).fetchall()]

    # vectors ----------------------------------------------------------------
    def upsert_vector(self, run_id, claim_id, vector, text, kind, metadata, agent_id):
        self.upsert_vectors([(run_id, claim_id, vector, text, kind, metadata, agent_id)])

    def upsert_vectors(self, rows):
        params = [(cid, rid, aid, kind, text, json.dumps(md, default=repr), len(vec), [float(x) for x in vec])
                  for rid, cid, vec, text, kind, md, aid in rows]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO claim_vectors VALUES (?,?,?,?,?,?,?,?)", params)

    def search_vectors(self, vector, k, run_id=None, kind=None):
        q = [float(x) for x in vector]
        sql = ("SELECT claim_id, run_id, agent_id, kind, text, metadata, "
               "list_cosine_similarity(vec, ?::FLOAT[]) AS s FROM claim_vectors WHERE dim=?")
        params: list[Any] = [q, len(q)]
        if run_id is not None:
            sql += " AND run_id=?"
            params.append(run_id)
        if kind is not None:
            sql += " AND kind=?"
            params.append(kind)
        sql += f" ORDER BY s DESC NULLS LAST LIMIT {int(k)}"
        return [VectorHit(r[0], float(r[6] or 0.0), r[4], r[3], json.loads(r[5] or "{}"), r[1], r[2])
                for r in self._ex(sql, params).fetchall()]

    def vector_count(self, run_id=None):
        if run_id is None:
            return self._ex("SELECT COUNT(*) FROM claim_vectors").fetchone()[0]
        return self._ex("SELECT COUNT(*) FROM claim_vectors WHERE run_id=?", (run_id,)).fetchone()[0]

    # judge cache / calibrations -----------------------------------------------
    def judge_cache_get(self, key):
        row = self._ex("SELECT value FROM judge_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def judge_cache_put(self, key, value):
        self._ex("INSERT OR REPLACE INTO judge_cache VALUES (?,?,?)", (key, json.dumps(value, default=repr), time.time()))

    def calibration_get(self, workload):
        row = self._ex("SELECT value FROM calibrations WHERE workload=?", (workload,)).fetchone()
        return json.loads(row[0]) if row else None

    def calibration_put(self, workload, value):
        self._ex("INSERT OR REPLACE INTO calibrations VALUES (?,?,?)", (workload, json.dumps(value, default=repr), time.time()))

    def flush(self):
        pass

    def close(self):
        with self._lock:
            self._conn.close()
