# Storage backends

Every backend implements the same `Store` protocol (events, runs, claim vectors, judge cache, calibrations) and passes the same conformance test (`tests/test_stores.py`). Pick by URL:

| URL | Backend | Extra | Use when |
|---|---|---|---|
| `sqlite:///path.db` (default) | SQLite, WAL mode | none | single host; `pip install` and go. Multiple processes on one host can share the file. |
| `duckdb:///path.duckdb` | DuckDB | `duckdb` | large runs you want to analyse with SQL; vector search is an in-SQL cosine scan |
| `postgresql://user:pass@host/db` | Postgres, pgvector when available | `postgres` | multi-host swarms; the shared claim store that makes dedup a live coordination substrate |
| `memory://` | in-process | none | tests, notebooks |

```python
sdk = ss.init("postgresql://swarm:swarm@db.internal/swarmscope")
```

## Vector search

- SQLite: `sqlite-vec` when loadable (`vec` extra), otherwise a brute-force cosine scan (numpy-accelerated if present). Fine to ~10^5 claims per run.
- DuckDB: `list_cosine_similarity` over `FLOAT[]` columns. Fine to ~10^6.
- Postgres: pgvector `<=>` with an HNSW index when `CREATE EXTENSION vector` succeeds; otherwise a Python cosine scan over the candidate rows. The vector column's dimension is fixed at table creation (`PostgresStore(url, dim=...)`, default 512 to match the hash embedder); use a fresh schema when changing embedders.

## Sharing a store across processes (live coordination)

The claim store's read path queries the store, not process memory, so any two processes writing to the same store see each other's claims within one writer tick (50 ms) plus store latency. Combine with lineage headers so the parent/child relationship survives the boundary:

```python
# dispatcher
headers = sdk.headers()
queue.put({"job": job, "swarm": headers})

# worker process
sdk = ss.init(os.environ["SWARMSCOPE_STORE"])         # same URL
with sdk.attach_headers(msg["swarm"]):
    with sdk.agent_scope(role="worker"):
        hit = sdk.claim(approach, kind="approach")    # sees claims from every other worker
```

`GatePolicy(scope="global")` widens recall from the current run to every run in the store (cross-run memory: "has anyone *ever* tried this?").

`examples/multiprocess_swarm.py` runs this end to end with a SQLite file and a process pool.

## Operational notes

- Writes are batched and best-effort in every backend; the buffer never blocks the swarm, and drops are counted in `sdk.stats["buffer"]`.
- Postgres uses one connection per `Swarmscope` instance under a lock. For hundreds of worker processes, put PgBouncer in front.
- Tables are prefixed `swarmscope_` in Postgres so the store can live in an existing database.
- Run `swarmscope export --format jsonl` to move a run between backends.
