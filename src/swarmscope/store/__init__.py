from .base import RunInfo, Store, VectorHit, open_store
from .memory import MemoryStore
from .sqlite import SQLiteStore



def __getattr__(name):  # lazy: optional backends
    if name == "DuckDBStore":
        from .duckdb import DuckDBStore

        return DuckDBStore
    if name == "PostgresStore":
        from .postgres import PostgresStore

        return PostgresStore
    raise AttributeError(name)


__all__ = ["Store", "RunInfo", "VectorHit", "open_store", "MemoryStore", "SQLiteStore", "DuckDBStore", "PostgresStore"]
