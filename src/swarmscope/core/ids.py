"""ID generation and hashing helpers.

IDs are time-ordered so range scans on ``event_id`` roughly follow emission
order. Generation is on the hot path (one id per event) so it is a hex
timestamp + process-wide counter + per-process random suffix: no syscalls,
no Python loops, ~1 µs.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from typing import Any

_PROC = os.urandom(4).hex()  # distinguishes processes that share a store
_COUNTER = itertools.count()  # ``next`` on a C iterator is atomic under the GIL


def new_id(prefix: str = "") -> str:
    """Return a lexically time-sortable id: ``<prefix><ts_us hex><proc><counter hex>``."""
    return f"{prefix}{time.time_ns() // 1000:012x}{_PROC}{next(_COUNTER) & 0xFFFFFF:06x}"


try:  # optional ``perf`` extra: ~5x faster canonical serialisation
    import orjson as _orjson

    def _canon(obj: Any) -> bytes:
        return _orjson.dumps(obj, option=_orjson.OPT_SORT_KEYS, default=repr)
except Exception:  # pragma: no cover
    def _canon(obj: Any) -> bytes:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":")).encode("utf-8")


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic content hash. Non-JSON-serialisable objects hash their repr."""
    try:
        payload = _canon(obj)
    except Exception:  # pragma: no cover - defensive
        payload = repr(obj).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()[:length]


def hash_bytes(payload: bytes | str, length: int = 16) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()[:length]


def config_hash(config: dict[str, Any] | None) -> str:
    return stable_hash(config or {}, 12)
