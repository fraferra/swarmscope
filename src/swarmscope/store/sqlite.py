"""SQLite backend. ``pip install swarmscope`` and go.

Vector search uses ``sqlite-vec`` when the extension is importable and
loadable, otherwise a brute-force cosine scan (numpy-accelerated if
available). Brute force is fine up to ~10^5 claims per run; beyond that
install the ``vec`` extra.
"""
from __future__ import annotations

import array
import json
import sqlite3
import threading
import time
from typing import Any, Iterable, Sequence

from ..core.events import Event
from .base import RouteOutcome, RouteStat, RunInfo, VectorHit, cosine

try:  # optional acceleration
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:  # optional: ~5x faster serialisation on the writer thread (``perf`` extra)
    import orjson as _orjson

    def _dumps(obj: Any) -> str:
        return _orjson.dumps(obj, default=repr).decode()
except Exception:  # pragma: no cover
    def _dumps(obj: Any) -> str:
        return json.dumps(obj, default=repr, separators=(",", ":"))

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, name TEXT, started_at REAL, ended_at REAL, meta TEXT
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  type TEXT NOT NULL,
  ts REAL NOT NULL,
  group_id TEXT, agent_id TEXT, parent_id TEXT,
  data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_run_type ON events(run_id, type, ts);
CREATE INDEX IF NOT EXISTS ix_events_run_agent ON events(run_id, agent_id);
CREATE TABLE IF NOT EXISTS claim_vectors (
  claim_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, agent_id TEXT,
  kind TEXT, text TEXT, metadata TEXT, dim INTEGER, vec BLOB
);
CREATE INDEX IF NOT EXISTS ix_vec_run ON claim_vectors(run_id, kind);
CREATE TABLE IF NOT EXISTS judge_cache (key TEXT PRIMARY KEY, value TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS calibrations (workload TEXT PRIMARY KEY, value TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS route_outcomes (
  request_id TEXT, artifact_id TEXT, run_id TEXT, route_key TEXT, route TEXT, accepted INTEGER, weight REAL,
  ts REAL, source TEXT, PRIMARY KEY (request_id, artifact_id)
);
CREATE INDEX IF NOT EXISTS ix_route_outcomes_req ON route_outcomes(request_id);
CREATE TABLE IF NOT EXISTS route_stats (
  route_key TEXT PRIMARY KEY, route TEXT, successes REAL, failures REAL, last_ts REAL
);
"""


_ROWS_PER_STMT = 500  # 8 params/row -> 4000 params, well under SQLITE_MAX_VARIABLE_NUMBER
_INSERT_PREFIX = ("INSERT OR REPLACE INTO events(event_id,run_id,type,ts,group_id,agent_id,parent_id,data) VALUES ")
_ROW_PLACEHOLDER = "(?,?,?,?,?,?,?,?)"


def _pack(vec: Sequence[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes) -> list[float]:
    a = array.array("f")
    a.frombytes(blob)
    return a.tolist()


class SQLiteStore:
    def __init__(self, path: str = "swarmscope.db", *, use_sqlite_vec: bool | None = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.executescript(_SCHEMA)
        self._vec = False
        if use_sqlite_vec is not False:
            self._vec = self._try_load_sqlite_vec(require=bool(use_sqlite_vec))

    # ------------------------------------------------------------------
    def _try_load_sqlite_vec(self, require: bool) -> bool:
        try:
            import sqlite_vec  # type: ignore

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except Exception as exc:  # pragma: no cover - depends on env
            if require:
                raise RuntimeError("sqlite-vec requested but not loadable") from exc
            return False

    # runs ---------------------------------------------------------------
    def begin_run(self, run_id, name, started_at, meta):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs(run_id,name,started_at,ended_at,meta) VALUES(?,?,?,NULL,?)",
                (run_id, name, started_at, json.dumps(meta, default=repr)),
            )

    def end_run(self, run_id, ended_at):
        with self._lock:
            self._conn.execute("UPDATE runs SET ended_at=? WHERE run_id=?", (ended_at, run_id))

    def runs(self):
        with self._lock:
            rows = self._conn.execute(
                """SELECT r.run_id, r.name, r.started_at, r.ended_at, r.meta,
                          (SELECT COUNT(*) FROM events e WHERE e.run_id=r.run_id)
                   FROM runs r ORDER BY r.started_at DESC"""
            ).fetchall()
            known = {r[0] for r in rows}
            orphan = self._conn.execute(
                "SELECT run_id, MIN(ts), MAX(ts), COUNT(*) FROM events GROUP BY run_id"
            ).fetchall()
        out = [RunInfo(r[0], r[1], r[2], r[3], json.loads(r[4] or "{}"), r[5]) for r in rows]
        out += [RunInfo(o[0], None, o[1], o[2], {}, o[3]) for o in orphan if o[0] not in known]
        return out

    def run(self, run_id):
        return next((r for r in self.runs() if r.run_id == run_id), None)

    # events -------------------------------------------------------------
    def write(self, events: Sequence[Event]) -> None:
        if not events:
            return
        rows = []
        for e in events:
            d = e.to_dict()
            rows.append((e.event_id, e.run_id, e.type, e.ts, e.group_id, e.agent_id, e.parent_id, _dumps(d)))
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                # Multi-row INSERTs: one sqlite3_step (one GIL release/acquire) per
                # chunk instead of per row, which is what makes the writer thread
                # cheap for the swarm's main thread.
                for i in range(0, len(rows), _ROWS_PER_STMT):
                    chunk = rows[i:i + _ROWS_PER_STMT]
                    self._conn.execute(_INSERT_PREFIX + ",".join([_ROW_PLACEHOLDER] * len(chunk)),
                                       [v for r in chunk for v in r])
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def events(self, run_id, types: Iterable[str] | None = None, agent_id=None, limit=None):
        sql = "SELECT data FROM events WHERE run_id=?"
        params: list[Any] = [run_id]
        if types:
            ts = list(types)
            sql += f" AND type IN ({','.join('?' * len(ts))})"
            params += ts
        if agent_id is not None:
            sql += " AND agent_id=?"
            params.append(agent_id)
        sql += " ORDER BY ts, event_id"
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [Event.from_dict(json.loads(r[0])) for r in rows]

    # vectors ------------------------------------------------------------
    def upsert_vector(self, run_id, claim_id, vector, text, kind, metadata, agent_id):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO claim_vectors(claim_id,run_id,agent_id,kind,text,metadata,dim,vec)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (claim_id, run_id, agent_id, kind, text, json.dumps(metadata, default=repr),
                 len(vector), _pack(vector)),
            )

    def upsert_vectors(self, rows):
        if not rows:
            return
        params = [(cid, rid, aid, kind, text, json.dumps(md, default=repr), len(vec), _pack(vec))
                  for rid, cid, vec, text, kind, md, aid in rows]
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO claim_vectors(claim_id,run_id,agent_id,kind,text,metadata,dim,vec)"
                    " VALUES(?,?,?,?,?,?,?,?)", params)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def search_vectors(self, vector, k, run_id=None, kind=None):
        sql = "SELECT claim_id, run_id, agent_id, kind, text, metadata, vec FROM claim_vectors WHERE 1=1"
        params: list[Any] = []
        if run_id is not None:
            sql += " AND run_id=?"
            params.append(run_id)
        if kind is not None:
            sql += " AND kind=?"
            params.append(kind)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []
        if _np is not None:
            q = _np.asarray(vector, dtype=_np.float32)
            mat = _np.stack([_np.frombuffer(r[6], dtype=_np.float32) for r in rows])
            qn = _np.linalg.norm(q) or 1.0
            mn = _np.linalg.norm(mat, axis=1)
            mn[mn == 0] = 1.0
            scores = (mat @ q) / (mn * qn)
            idx = _np.argsort(-scores)[:k]
            pairs = [(rows[i], float(scores[i])) for i in idx]
        else:
            scored = [(r, cosine(vector, _unpack(r[6]))) for r in rows]
            scored.sort(key=lambda p: p[1], reverse=True)
            pairs = scored[:k]
        return [VectorHit(r[0], s, r[4], r[3], json.loads(r[5] or "{}"), r[1], r[2]) for r, s in pairs]

    def vector_count(self, run_id=None):
        with self._lock:
            if run_id is None:
                return self._conn.execute("SELECT COUNT(*) FROM claim_vectors").fetchone()[0]
            return self._conn.execute(
                "SELECT COUNT(*) FROM claim_vectors WHERE run_id=?", (run_id,)).fetchone()[0]

    # judge cache --------------------------------------------------------
    def judge_cache_get(self, key):
        with self._lock:
            row = self._conn.execute("SELECT value FROM judge_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def judge_cache_put(self, key, value):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO judge_cache(key,value,ts) VALUES(?,?,?)",
                               (key, json.dumps(value, default=repr), time.time()))

    # reputation ---------------------------------------------------------
    def route_outcomes_put(self, rows):
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO route_outcomes VALUES (?,?,?,?,?,?,?,?,?)",
                [(r.request_id, r.artifact_id, r.run_id, r.route_key, json.dumps(r.route), int(r.accepted),
                  r.weight, r.ts, r.source) for r in rows])

    def route_outcomes(self, request_ids=None, limit=None):
        sql, params = "SELECT request_id, artifact_id, run_id, route_key, route, accepted, weight, ts, source FROM route_outcomes", []
        if request_ids is not None:
            ids = list(request_ids)
            if not ids:
                return []
            sql += f" WHERE request_id IN ({','.join('?' * len(ids))})"
            params += ids
        sql += " ORDER BY ts DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [RouteOutcome(r[0], r[1], r[2], r[3], json.loads(r[4]), bool(r[5]), r[6], r[7], r[8] or "") for r in rows]

    def route_stats_add(self, route_key, route, success, failure, ts):
        with self._lock:
            self._conn.execute(
                """INSERT INTO route_stats(route_key, route, successes, failures, last_ts) VALUES (?,?,?,?,?)
                   ON CONFLICT(route_key) DO UPDATE SET successes=successes+excluded.successes,
                   failures=failures+excluded.failures, last_ts=MAX(last_ts, excluded.last_ts)""",
                (route_key, json.dumps(list(route)), success, failure, ts))

    def route_stats(self, limit=None):
        sql = ("SELECT route_key, route, successes, failures, last_ts FROM route_stats "
               "ORDER BY successes / (successes + failures + 1e-9) DESC, successes + failures DESC")
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [RouteStat(r[0], json.loads(r[1]), r[2], r[3], r[4]) for r in rows]

    def route_stats_clear(self):
        with self._lock:
            self._conn.execute("DELETE FROM route_stats")
            self._conn.execute("DELETE FROM route_outcomes")

    # calibrations -------------------------------------------------------
    def calibration_get(self, workload):
        with self._lock:
            row = self._conn.execute("SELECT value FROM calibrations WHERE workload=?", (workload,)).fetchone()
        return json.loads(row[0]) if row else None

    def calibration_put(self, workload, value):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO calibrations(workload,value,ts) VALUES(?,?,?)",
                               (workload, json.dumps(value, default=repr), time.time()))

    def flush(self):
        pass

    def close(self):
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.close()
