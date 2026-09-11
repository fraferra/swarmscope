"""Embedders for the claim store and the reputation router.

The default :class:`HashEmbedder` is dependency-free and deterministic — a
feature-hashed bag of word unigrams/bigrams and character trigrams. It is
*weak* (lexical, not semantic) but good enough to exercise the pipeline and
to catch near-verbatim repeats. Semantic options, fastest first:

- :class:`Model2VecEmbedder` — static embeddings, ~0.05 ms per text on CPU,
  no torch; the recommended low-latency semantic choice (``model2vec`` extra)
- :class:`SentenceTransformerEmbedder` — ~5–20 ms per text on CPU
- :class:`OpenAIEmbedder` — network round trip, 50–300 ms

Every embedder is wrapped in :class:`CachedEmbedder` by the SDK: an LRU in
memory plus persistence in the store, so repeated texts cost nothing and
the cache is shared across processes. Pick one with a spec string::

    Swarmscope(store, embedder="model2vec:minishlab/potion-base-8M")
    Swarmscope(store, embedder="openai:text-embedding-3-small")
    Swarmscope(store, embedder="st:all-MiniLM-L6-v2")
    Swarmscope(store, embedder="hash")            # default
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import OrderedDict
from zlib import crc32
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

log = logging.getLogger("swarmscope.embed")

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "be", "that",
         "this", "it", "we", "for", "on", "by", "with", "as", "at", "from", "via"}


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


class HashEmbedder:
    name = "hash-v1"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        dim = self.dim
        v = [0.0] * dim
        t = normalize_text(text)
        words = [w for w in _WORD.findall(t) if w not in _STOP]
        # crc32 is deterministic across processes and C-fast; bit 16 picks the sign.
        for w in words:
            h = crc32(b"w:" + w.encode())
            v[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
        for a, b in zip(words, words[1:]):
            h = crc32(b"b:" + a.encode() + b"_" + b.encode())
            v[h % dim] += 1.5 if (h >> 16) & 1 else -1.5
        if len(words) < 4:  # char trigrams only for short texts; words+bigrams carry the signal otherwise
            compact = t.replace(" ", "_").encode()
            for i in range(len(compact) - 2):
                h = crc32(compact[i:i + 3], 99)  # seed distinguishes char features from word features
                v[h % dim] += 0.5 if (h >> 16) & 1 else -0.5
        n = math.hypot(*v) or 1.0
        return [x / n for x in v]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class CallableEmbedder:
    def __init__(self, fn: Callable[[Sequence[str]], Sequence[Sequence[float]]], dim: int, name: str = "callable") -> None:
        self._fn, self.dim, self.name = fn, dim, name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


class OpenAIEmbedder:
    """Uses the OpenAI embeddings endpoint. Requires the ``openai`` extra."""

    def __init__(self, model: str = "text-embedding-3-small", client=None, dim: int | None = None) -> None:
        if client is None:
            from openai import OpenAI  # lazy

            client = OpenAI()
        self._client = client
        self.model = model
        self.name = f"openai:{model}"
        self.dim = dim or (1536 if "small" in model else 3072)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        return [list(d.embedding) for d in resp.data]


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self._m = SentenceTransformer(model)
        self.name = f"st:{model}"
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._m.encode(list(texts), normalize_embeddings=True)]

    def warmup(self) -> None:
        self.embed(["warmup"])


class Model2VecEmbedder:
    """Static (non-contextual) embeddings distilled from a sentence transformer.
    Microsecond-scale inference, no torch. Requires the ``model2vec`` extra."""

    def __init__(self, model: str = "minishlab/potion-base-8M") -> None:
        from model2vec import StaticModel  # lazy

        self._m = StaticModel.from_pretrained(model)
        self.model = model
        self.name = f"model2vec:{model}"
        self.dim = int(self._m.dim) if hasattr(self._m, "dim") else len(self.embed(["x"])[0])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import numpy as np

        vecs = self._m.encode(list(texts))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(float).tolist()

    def warmup(self) -> None:
        self.embed(["warmup"])


class CachedEmbedder:
    """LRU + store-persisted cache in front of any embedder.

    Keyed by normalised text and embedder name, so switching embedders never
    mixes vector spaces. The store layer (``judge_cache`` table, ``emb:`` keys)
    makes the cache shared across processes and restarts; misses are written
    back on a background thread so a lookup never waits on persistence.
    """

    def __init__(self, inner: Embedder, store=None, *, maxsize: int = 50_000, persist: bool = True) -> None:
        self.inner = inner
        self.store = store if persist else None
        self.name = inner.name
        self.dim = inner.dim
        self.maxsize = maxsize
        self._lru: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.store_hits = 0

    def _key(self, text: str) -> str:
        return "emb:" + hashlib.blake2b(f"{self.name}\x00{normalize_text(text)}".encode(), digest_size=16).hexdigest()

    def _lru_get(self, key: str) -> list[float] | None:
        with self._lock:
            v = self._lru.get(key)
            if v is not None:
                self._lru.move_to_end(key)
            return v

    def _lru_put(self, key: str, vec: list[float]) -> None:
        with self._lock:
            self._lru[key] = vec
            if len(self._lru) > self.maxsize:
                self._lru.popitem(last=False)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, k in enumerate(keys):
            v = self._lru_get(k)
            if v is not None:
                out[i] = v
                self.hits += 1
            else:
                missing.append(i)
        if missing and self.store is not None:
            still: list[int] = []
            for i in missing:
                try:
                    row = self.store.judge_cache_get(keys[i])
                except Exception:
                    row = None
                if row and "v" in row and len(row["v"]) == self.dim:
                    out[i] = row["v"]
                    self._lru_put(keys[i], row["v"])
                    self.store_hits += 1
                else:
                    still.append(i)
            missing = still
        if missing:
            self.misses += len(missing)
            vecs = self.inner.embed([texts[i] for i in missing])
            for i, v in zip(missing, vecs):
                out[i] = v
                self._lru_put(keys[i], v)
            if self.store is not None:
                pairs = [(keys[i], v) for i, v in zip(missing, vecs)]
                threading.Thread(target=self._persist, args=(pairs,), daemon=True).start()
        return out  # type: ignore[return-value]

    def _persist(self, pairs) -> None:
        for k, v in pairs:
            try:
                self.store.judge_cache_put(k, {"v": v})
            except Exception:  # pragma: no cover
                log.debug("swarmscope: embedding cache write failed", exc_info=True)

    def warmup(self) -> None:
        w = getattr(self.inner, "warmup", None)
        if w:
            w()

    @property
    def stats(self) -> dict[str, Any]:
        return {"embedder": self.name, "dim": self.dim, "hits": self.hits, "store_hits": self.store_hits,
                "misses": self.misses, "lru_size": len(self._lru)}


def embedder_from_spec(spec: str | Embedder | None) -> Embedder:
    """``"hash"`` | ``"model2vec[:model]"`` | ``"st:<model>"`` | ``"openai[:model]"`` | an Embedder."""
    if spec is None or spec == "hash":
        return HashEmbedder()
    if not isinstance(spec, str):
        return spec
    kind, _, model = spec.partition(":")
    if kind == "model2vec":
        return Model2VecEmbedder(model or "minishlab/potion-base-8M")
    if kind in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2")
    if kind == "openai":
        return OpenAIEmbedder(model or "text-embedding-3-small")
    if kind == "hash":
        return HashEmbedder(int(model) if model else 512)
    raise ValueError(f"unknown embedder spec {spec!r}; use hash | model2vec[:model] | st:<model> | openai[:model]")
