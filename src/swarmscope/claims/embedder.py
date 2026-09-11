"""Embedders for the claim store.

The default :class:`HashEmbedder` is dependency-free and deterministic — a
feature-hashed bag of word unigrams/bigrams and character trigrams. It is
*weak* (lexical, not semantic) but good enough to exercise the pipeline and
to catch near-verbatim repeats. For real semantic recall use
:class:`OpenAIEmbedder` or :class:`SentenceTransformerEmbedder`.
"""
from __future__ import annotations

import math
import re
from zlib import crc32
from typing import Callable, Protocol, Sequence, runtime_checkable

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
