"""Postgres backend: the shared store for multi-process / multi-host swarms.

Uses ``pgvector`` for ANN search when the extension is installed (an HNSW
index is created on the vector column); otherwise falls back to a Python
cosine scan over the candidate rows. Requires the ``postgres`` extra
(``psycopg[binary]``). URL: ``postgresql://user:pass@host/db``.

Concurrency: one connection per store, guarded by a lock. Each process in a
swarm opens its own ``Swarmscope`` (and store) pointed at the same URL; the
claim store's read path then sees claims from every process, which is what
makes the dedup gate a live coordination substrate rather than a per-process
cache.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Iterable, Sequence

from ..core.events import Event
from .base import RouteOutcome, RouteStat, RunInfo, VectorHit, cosine

_SCHEMA = """
CREATE TABLE IF NOT EXISTS swarmscope_runs (
  run_id TEXT PRIMARY KEY, name TEXT, started_at DOUBLE PRECISION, ended_at DOUBLE PRECISION, meta JSONB
);
CREATE TABLE IF NOT EXISTS swarmscope_events (
  event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, type TEXT NOT NULL, ts DOUBLE PRECISION NOT NULL,
  group_id TEXT, agent_id TEXT, parent_id TEXT, data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ss_events_run_type ON swarmscope_events(run_id, type, ts);
CREATE INDEX IF NOT EXISTS ix_ss_events_run_agent ON swarmscope_events(run_id, agent_id);
CREATE TABLE IF NOT EXISTS swarmscope_judge_cache (key TEXT PRIMARY KEY, value JSONB, ts DOUBLE PRECISION);
CREATE TABLE IF NOT EXISTS swarmscope_calibrations (workload TEXT PRIMARY KEY, value JSONB, ts DOUBLE PRECISION);
CREATE TABLE IF NOT EXISTS swarmscope_route_outcomes (
  request_id TEXT, artifact_id TEXT, run_id TEXT, route_key TEXT, route JSONB, accepted BOOLEAN,
  weight DOUBLE PRECISION, ts DOUBLE PRECISION, source TEXT, PRIMARY KEY (request_id, artifact_id)
);
CREATE TABLE IF NOT EXISTS swarmscope_route_stats (
  route_key TEXT PRIMARY KEY, route JSONB, successes DOUBLE PRECISION, failures DOUBLE PRECISION, last_ts DOUBLE PRECISION
);
"""

_VECTORS_PLAIN = """
CREATE TABLE IF NOT EXISTS swarmscope_claim_vectors (
  claim_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, agent_id TEXT, kind TEXT, text TEXT, metadata JSONB,
  dim INTEGER, vec REAL[]
);
CREATE INDEX IF NOT EXISTS ix_ss_vec_run ON swarmscope_claim_vectors(run_id, kind);
"""

_VECTORS_PGVECTOR = """
CREATE TABLE IF NOT EXISTS swarmscope_claim_vectors (
  claim_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, agent_id TEXT, kind TEXT, text TEXT, metadata JSONB,
  dim INTEGER, vec vector({dim})
);
CREATE INDEX IF NOT EXISTS ix_ss_vec_run ON swarmscope_claim_vectors(run_id, kind);
CREATE INDEX IF NOT EXISTS ix_ss_vec_hnsw ON swarmscope_claim_vectors USING hnsw (vec vector_cosine_ops);
"""


class PostgresStore:
    def __init__(self, url: str, *, dim: int = 512, use_pgvector: bool | None = None) -> None:
        import psycopg  # lazy

        self.url = url
        self.dim = dim
        self._lock = threading.RLock()
        self._conn = psycopg.connect(url, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA)
            self.pgvector = False
            if use_pgvector is not False:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    self.pgvector = True
                except Exception:
                    if use_pgvector:
                        raise
                    self._conn.rollback() if not self._conn.autocommit else None
            cur.execute(_VECTORS_PGVECTOR.format(dim=dim) if self.pgvector else _VECTORS_PLAIN)
        if self.pgvector:
            from pgvector.psycopg import register_vector  # type: ignore

            register_vector(self._conn)

    def _ex(self, sql: str, params: Sequence[Any] = (), *, fetch: str | None = None) -> Any:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, list(params))
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return None

    # runs -------------------------------------------------------------------
    def begin_run(self, run_id, name, started_at, meta):
        self._ex("""INSERT INTO swarmscope_runs(run_id,name,started_at,ended_at,meta) VALUES (%s,%s,%s,NULL,%s)
                    ON CONFLICT (run_id) DO UPDATE SET name=EXCLUDED.name, started_at=EXCLUDED.started_at, meta=EXCLUDED.meta""",
                 (run_id, name, started_at, json.dumps(meta, default=repr)))

    def end_run(self, run_id, ended_at):
        self._ex("UPDATE swarmscope_runs SET ended_at=%s WHERE run_id=%s", (ended_at, run_id))

    def runs(self):
        rows = self._ex("""SELECT r.run_id, r.name, r.started_at, r.ended_at, r.meta,
                                  (SELECT COUNT(*) FROM swarmscope_events e WHERE e.run_id=r.run_id)
                           FROM swarmscope_runs r ORDER BY r.started_at DESC NULLS LAST""", fetch="all")
        known = {r[0] for r in rows}
        orphan = self._ex("SELECT run_id, MIN(ts), MAX(ts), COUNT(*) FROM swarmscope_events GROUP BY run_id", fetch="all")
        out = [RunInfo(r[0], r[1], r[2], r[3], r[4] or {}, r[5]) for r in rows]
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
        with self._lock, self._conn.cursor() as cur:
            cur.executemany("""INSERT INTO swarmscope_events(event_id,run_id,type,ts,group_id,agent_id,parent_id,data)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT (event_id) DO NOTHING""", rows)

    def events(self, run_id, types: Iterable[str] | None = None, agent_id=None, limit=None):
        sql, params = "SELECT data FROM swarmscope_events WHERE run_id=%s", [run_id]
        if types:
            sql += " AND type = ANY(%s)"
            params.append(list(types))
        if agent_id is not None:
            sql += " AND agent_id=%s"
            params.append(agent_id)
        sql += " ORDER BY ts, event_id"
        if limit:
            sql += " LIMIT %s"
            params.append(int(limit))
        return [Event.from_dict(r[0]) for r in self._ex(sql, params, fetch="all")]

    # vectors ----------------------------------------------------------------
    def upsert_vector(self, run_id, claim_id, vector, text, kind, metadata, agent_id):
        self.upsert_vectors([(run_id, claim_id, vector, text, kind, metadata, agent_id)])

    def upsert_vectors(self, rows):
        if not rows:
            return
        params = []
        for rid, cid, vec, text, kind, md, aid in rows:
            v = [float(x) for x in vec]
            if self.pgvector:
                import numpy as np  # pgvector adapts numpy arrays

                v = np.asarray(v, dtype=np.float32)
            params.append((cid, rid, aid, kind, text, json.dumps(md, default=repr), len(vec), v))
        with self._lock, self._conn.cursor() as cur:
            cur.executemany("""INSERT INTO swarmscope_claim_vectors(claim_id,run_id,agent_id,kind,text,metadata,dim,vec)
                               VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s,%s)
                               ON CONFLICT (claim_id) DO UPDATE SET text=EXCLUDED.text, metadata=EXCLUDED.metadata,
                               vec=EXCLUDED.vec, kind=EXCLUDED.kind""", params)

    def search_vectors(self, vector, k, run_id=None, kind=None):
        where, params = ["dim=%s"], [len(vector)]
        if run_id is not None:
            where.append("run_id=%s")
            params.append(run_id)
        if kind is not None:
            where.append("kind=%s")
            params.append(kind)
        if self.pgvector:
            import numpy as np

            q = np.asarray([float(x) for x in vector], dtype=np.float32)
            rows = self._ex(f"""SELECT claim_id, run_id, agent_id, kind, text, metadata, 1 - (vec <=> %s) AS s
                                FROM swarmscope_claim_vectors WHERE {' AND '.join(where)} ORDER BY vec <=> %s LIMIT %s""",
                            [q] + params + [q, int(k)], fetch="all")
            return [VectorHit(r[0], float(r[6]), r[4], r[3], r[5] or {}, r[1], r[2]) for r in rows]
        rows = self._ex(f"SELECT claim_id, run_id, agent_id, kind, text, metadata, vec FROM swarmscope_claim_vectors "
                        f"WHERE {' AND '.join(where)}", params, fetch="all")
        scored = sorted(((r, cosine(vector, r[6])) for r in rows), key=lambda p: p[1], reverse=True)[:k]
        return [VectorHit(r[0], s, r[4], r[3], r[5] or {}, r[1], r[2]) for r, s in scored]

    def vector_count(self, run_id=None):
        if run_id is None:
            return self._ex("SELECT COUNT(*) FROM swarmscope_claim_vectors", fetch="one")[0]
        return self._ex("SELECT COUNT(*) FROM swarmscope_claim_vectors WHERE run_id=%s", (run_id,), fetch="one")[0]

    # judge cache / calibrations -----------------------------------------------
    def judge_cache_get(self, key):
        row = self._ex("SELECT value FROM swarmscope_judge_cache WHERE key=%s", (key,), fetch="one")
        return row[0] if row else None

    def judge_cache_put(self, key, value):
        self._ex("""INSERT INTO swarmscope_judge_cache VALUES (%s,%s::jsonb,%s)
                    ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, ts=EXCLUDED.ts""",
                 (key, json.dumps(value, default=repr), time.time()))

    def calibration_get(self, workload):
        row = self._ex("SELECT value FROM swarmscope_calibrations WHERE workload=%s", (workload,), fetch="one")
        return row[0] if row else None

    def calibration_put(self, workload, value):
        self._ex("""INSERT INTO swarmscope_calibrations VALUES (%s,%s::jsonb,%s)
                    ON CONFLICT (workload) DO UPDATE SET value=EXCLUDED.value, ts=EXCLUDED.ts""",
                 (workload, json.dumps(value, default=repr), time.time()))

    # reputation ---------------------------------------------------------------
    def route_outcomes_put(self, rows):
        if not rows:
            return
        with self._lock, self._conn.cursor() as cur:
            cur.executemany("""INSERT INTO swarmscope_route_outcomes VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
                               ON CONFLICT (request_id, artifact_id) DO UPDATE SET accepted=EXCLUDED.accepted,
                               weight=EXCLUDED.weight, ts=EXCLUDED.ts, source=EXCLUDED.source""",
                            [(r.request_id, r.artifact_id, r.run_id, r.route_key, json.dumps(r.route), bool(r.accepted),
                              r.weight, r.ts, r.source) for r in rows])

    def route_outcomes(self, request_ids=None, limit=None):
        sql, params = "SELECT request_id, artifact_id, run_id, route_key, route, accepted, weight, ts, source FROM swarmscope_route_outcomes", []
        if request_ids is not None:
            ids = list(request_ids)
            if not ids:
                return []
            sql += " WHERE request_id = ANY(%s)"
            params.append(ids)
        sql += " ORDER BY ts DESC"
        if limit:
            sql += " LIMIT %s"
            params.append(int(limit))
        return [RouteOutcome(r[0], r[1], r[2], r[3], r[4], bool(r[5]), r[6], r[7], r[8] or "")
                for r in self._ex(sql, params, fetch="all")]

    def route_stats_add(self, route_key, route, success, failure, ts):
        self._ex("""INSERT INTO swarmscope_route_stats VALUES (%s,%s::jsonb,%s,%s,%s)
                    ON CONFLICT (route_key) DO UPDATE SET successes=swarmscope_route_stats.successes+EXCLUDED.successes,
                    failures=swarmscope_route_stats.failures+EXCLUDED.failures,
                    last_ts=GREATEST(swarmscope_route_stats.last_ts, EXCLUDED.last_ts)""",
                 (route_key, json.dumps(list(route)), success, failure, ts))

    def route_stats(self, limit=None):
        sql = ("SELECT route_key, route, successes, failures, last_ts FROM swarmscope_route_stats "
               "ORDER BY successes / (successes + failures + 1e-9) DESC, successes + failures DESC")
        params: list[Any] = []
        if limit:
            sql += " LIMIT %s"
            params.append(int(limit))
        return [RouteStat(r[0], r[1], r[2], r[3], r[4]) for r in self._ex(sql, params, fetch="all")]

    def route_stats_clear(self):
        self._ex("DELETE FROM swarmscope_route_stats")
        self._ex("DELETE FROM swarmscope_route_outcomes")

    def flush(self):
        pass

    def close(self):
        with self._lock:
            self._conn.close()
