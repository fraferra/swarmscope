"""Storage protocol. Backends: SQLite (default), in-memory (tests).

Postgres/pgvector and DuckDB backends implement the same protocol; only
:class:`SQLiteStore` ships in this release.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from ..core.events import Event


@dataclass(slots=True)
class RunInfo:
    run_id: str
    name: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    event_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "name": self.name, "started_at": self.started_at,
                "ended_at": self.ended_at, "meta": self.meta, "event_count": self.event_count}


@dataclass(slots=True)
class VectorHit:
    claim_id: str
    score: float
    text: str
    kind: str
    metadata: dict[str, Any]
    run_id: str
    agent_id: str | None = None


@runtime_checkable
class Store(Protocol):
    """Every method must be safe to call from any thread."""

    # --- runs ---------------------------------------------------------
    def begin_run(self, run_id: str, name: str | None, started_at: float, meta: dict[str, Any]) -> None: ...
    def end_run(self, run_id: str, ended_at: float) -> None: ...
    def runs(self) -> list[RunInfo]: ...
    def run(self, run_id: str) -> RunInfo | None: ...

    # --- events -------------------------------------------------------
    def write(self, events: Sequence[Event]) -> None: ...
    def events(
        self,
        run_id: str,
        types: Iterable[str] | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[Event]: ...

    # --- claim vectors (L2) ------------------------------------------
    def upsert_vector(
        self, run_id: str, claim_id: str, vector: Sequence[float], text: str,
        kind: str, metadata: dict[str, Any], agent_id: str | None,
    ) -> None: ...
    def upsert_vectors(self, rows: Sequence[tuple]) -> None:
        """Batch form of :meth:`upsert_vector`; rows are its positional args."""
        for r in rows:
            self.upsert_vector(*r)

    def search_vectors(
        self, vector: Sequence[float], k: int, run_id: str | None = None,
        kind: str | None = None,
    ) -> list[VectorHit]: ...
    def vector_count(self, run_id: str | None = None) -> int: ...

    # --- judge cache (L2) --------------------------------------------
    def judge_cache_get(self, key: str) -> dict[str, Any] | None: ...
    def judge_cache_put(self, key: str, value: dict[str, Any]) -> None: ...

    # --- calibrations (L3) -------------------------------------------
    def calibration_get(self, workload: str) -> dict[str, Any] | None: ...
    def calibration_put(self, workload: str, value: dict[str, Any]) -> None: ...

    def flush(self) -> None: ...
    def close(self) -> None: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def open_store(url: str | Store | None) -> Store:
    """``sqlite:///path.db`` | ``duckdb:///path.duckdb`` | ``postgresql://...`` | ``memory://`` |
    a Store instance | None (default sqlite)."""
    from .memory import MemoryStore
    from .sqlite import SQLiteStore

    if url is None:
        return SQLiteStore("swarmscope.db")
    if isinstance(url, Store):
        return url
    if url.startswith("memory://") or url == ":memory:":
        return MemoryStore()
    if url.startswith(("postgresql://", "postgres://")):
        from .postgres import PostgresStore

        return PostgresStore(url)
    if url.startswith("duckdb:///"):
        from .duckdb import DuckDBStore

        return DuckDBStore(url[len("duckdb:///"):])
    if url.endswith(".duckdb"):
        from .duckdb import DuckDBStore

        return DuckDBStore(url)
    if url.startswith("sqlite:///"):
        return SQLiteStore(url[len("sqlite:///"):])
    if url.startswith("sqlite://"):
        return SQLiteStore(url[len("sqlite://"):])
    if url.endswith((".db", ".sqlite", ".sqlite3")):
        return SQLiteStore(url)
    raise ValueError(f"unsupported store url: {url!r}")
