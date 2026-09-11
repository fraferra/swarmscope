"""Stage-2 equivalence judges with a pair-hash verdict cache.

Cosine similarity is not equivalence. The judge decides whether two claims
are *the same idea*. Verdicts are cached by a normalised, order-independent
pair hash so that in a large swarm, where the same handful of ideas recur,
judge calls per claim trend toward zero.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from .embedder import normalize_text


@dataclass(slots=True)
class JudgeResult:
    equivalent: bool
    confidence: float = 1.0
    rationale: str | None = None
    cached: bool = False
    judge: str = ""


@runtime_checkable
class EquivalenceJudge(Protocol):
    name: str

    def judge(self, a: str, b: str, meta_a: dict[str, Any], meta_b: dict[str, Any]) -> JudgeResult: ...


def pair_key(judge_name: str, a: str, b: str) -> str:
    na, nb = sorted((normalize_text(a), normalize_text(b)))
    return hashlib.sha256(f"{judge_name}\x00{na}\x00{nb}".encode()).hexdigest()


class NoJudge:
    """Stage 2 disabled: matches are reported as *unjudged* cosine candidates."""

    name = "none"

    def judge(self, a, b, meta_a, meta_b) -> JudgeResult:  # pragma: no cover - trivial
        raise NotImplementedError


class CallableJudge:
    def __init__(self, fn: Callable[[str, str, dict, dict], bool | tuple[bool, float] | JudgeResult], name: str = "callable") -> None:
        self._fn, self.name = fn, name

    def judge(self, a, b, meta_a, meta_b) -> JudgeResult:
        r = self._fn(a, b, meta_a, meta_b)
        if isinstance(r, JudgeResult):
            r.judge = self.name
            return r
        if isinstance(r, tuple):
            return JudgeResult(bool(r[0]), float(r[1]), judge=self.name)
        return JudgeResult(bool(r), judge=self.name)


_PROMPT = """You judge whether two research claims made by different agents are the SAME idea.
Two claims are equivalent if pursuing one would rediscover the other: same approach/technique,
same assumptions, same target. Different assumptions (e.g. "assumes A" vs "assumes not A") are NOT equivalent.

Claim 1: {a}
Metadata 1: {ma}

Claim 2: {b}
Metadata 2: {mb}

Reply with JSON only: {{"equivalent": true|false, "confidence": 0..1, "rationale": "<one sentence>"}}"""


class OpenAIJudge:
    """LLM judge over the OpenAI chat API. Requires the ``openai`` extra."""

    def __init__(self, model: str = "gpt-4o-mini", client=None) -> None:
        if client is None:
            from openai import OpenAI  # lazy

            client = OpenAI()
        self._client = client
        self.model = model
        self.name = f"openai:{model}"

    def judge(self, a, b, meta_a, meta_b) -> JudgeResult:
        prompt = _PROMPT.format(a=a, b=b, ma=json.dumps(meta_a, default=repr), mb=json.dumps(meta_b, default=repr))
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return JudgeResult(bool(data.get("equivalent")), float(data.get("confidence", 0.5)),
                           data.get("rationale"), judge=self.name)


class CachedJudge:
    """Wraps any judge with the store-backed pair-hash cache."""

    def __init__(self, inner: EquivalenceJudge, store) -> None:
        self.inner = inner
        self.store = store
        self.calls = 0
        self.hits = 0
        self.name = inner.name

    def judge(self, a, b, meta_a, meta_b) -> JudgeResult:
        key = pair_key(self.inner.name, a, b)
        cached = self.store.judge_cache_get(key)
        if cached is not None:
            self.hits += 1
            return JudgeResult(cached["equivalent"], cached.get("confidence", 1.0),
                               cached.get("rationale"), cached=True, judge=self.inner.name)
        self.calls += 1
        r = self.inner.judge(a, b, meta_a, meta_b)
        self.store.judge_cache_put(key, {"equivalent": r.equivalent, "confidence": r.confidence,
                                         "rationale": r.rationale})
        return r
