"""Claim store: "has anyone already tried this?"

Pipeline for a lookup:
  0. structured-metadata prefilter (cheap, reliable for "assumes A" vs "assumes ¬A")
  1. ANN recall over embeddings → top-k candidates
  2. judge equivalence on candidates above ``judge_min_score``, cached by pair hash

The gate is **advisory**: :class:`ClaimHit` reports matches and a score and the
caller decides. Every suppression a caller reports is logged with the claim
that caused it. The read path has a hard latency budget and fails open.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from ..store.base import Store, VectorHit
from .embedder import Embedder, HashEmbedder
from .judge import CachedJudge, EquivalenceJudge, JudgeResult, NoJudge

log = logging.getLogger("swarmscope.claims")

_NEG = ("¬", "not ", "!", "~")


def _norm_assumption(a: str) -> tuple[str, bool]:
    s = a.strip().lower()
    neg = False
    for n in _NEG:
        if s.startswith(n):
            s = s[len(n):].strip()
            neg = not neg
    return s, neg


def default_prefilter(meta_a: dict[str, Any], meta_b: dict[str, Any]) -> bool:
    """Return False when metadata proves the claims are *not* the same idea.

    Contradictory ``assumes`` entries (``A`` vs ``¬A``/``not A``) exclude a
    candidate. Everything else passes; embeddings + judge take it from there.
    """
    aa = {(_norm_assumption(x)) for x in meta_a.get("assumes", []) or []}
    bb = {(_norm_assumption(x)) for x in meta_b.get("assumes", []) or []}
    for name, neg in aa:
        if (name, not neg) in bb:
            return False
    return True


@dataclass(slots=True)
class Match:
    claim_id: str
    text: str
    score: float  # cosine similarity
    kind: str
    metadata: dict[str, Any]
    agent_id: str | None
    run_id: str
    judged: bool = False
    equivalent: bool | None = None  # None when unjudged
    judge_confidence: float | None = None
    rationale: str | None = None


@dataclass(slots=True)
class ClaimHit:
    """Result of ``sdk.claim``. Advisory — nothing here blocks the caller."""

    claim_id: str
    similar: list[Match] = field(default_factory=list)
    top_score: float = 0.0
    #: True when policy says a reasonable agent would skip: a judged-equivalent
    #: match, or (with no judge) cosine above ``threshold``.
    suggest_skip: bool = False
    judged: bool = False
    timed_out: bool = False
    latency_ms: float = 0.0
    arm: str | None = None
    judge_calls: int = 0
    _on_suppress: Callable[["ClaimHit", str | None], None] | None = field(default=None, repr=False)

    @property
    def best(self) -> Match | None:
        return self.similar[0] if self.similar else None

    def suppress(self, reason: str | None = None) -> None:
        """Tell the SDK you acted on this advice and skipped the work."""
        if self._on_suppress:
            self._on_suppress(self, reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_score": self.top_score, "suggest_skip": self.suggest_skip, "judged": self.judged,
            "timed_out": self.timed_out, "latency_ms": self.latency_ms, "judge_calls": self.judge_calls,
            "matches": [{"claim_id": m.claim_id, "score": m.score, "equivalent": m.equivalent,
                         "judge_confidence": m.judge_confidence, "agent_id": m.agent_id}
                        for m in self.similar[:5]],
        }


@dataclass(slots=True)
class GatePolicy:
    #: cosine above which an unjudged candidate counts as "similar"
    threshold: float = 0.90
    #: cosine above which we bother calling the judge
    judge_min_score: float = 0.60
    top_k: int = 8
    #: max judge calls per lookup (cost control)
    max_judge_calls: int = 3
    #: read-path budget; on expiry the lookup fails open with ``timed_out=True``
    latency_budget_ms: float = 200.0
    #: "advisory" (default) | "off" (no lookups, record only)
    mode: str = "advisory"
    #: scope of recall: "run" (default) or "global" (all runs in the store)
    scope: str = "run"


class ClaimStore:
    def __init__(
        self,
        store: Store,
        *,
        embedder: Embedder | None = None,
        judge: EquivalenceJudge | None = None,
        policy: GatePolicy | None = None,
        prefilter: Callable[[dict[str, Any], dict[str, Any]], bool] | None = default_prefilter,
    ) -> None:
        self.store = store
        self.embedder = embedder or HashEmbedder()
        self.policy = policy or GatePolicy()
        self.prefilter = prefilter or (lambda a, b: True)
        self._judge: CachedJudge | None = None
        if judge is not None and not isinstance(judge, NoJudge):
            self._judge = CachedJudge(judge, store)
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swarmscope-claims")
        self._lock = threading.Lock()
        self.lookups = 0
        self.timeouts = 0
        # Write path: a deque drained by one thread that embeds in batches (one
        # embedder call per batch — cheap for the hash embedder, essential for
        # API embedders). ``record`` is a single append; nothing to lock.
        self._pending: deque[tuple] = deque()
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._write_loop, name="swarmscope-claims-writer", daemon=True)
        self._writer.start()

    # --- stats -----------------------------------------------------------
    @property
    def judge_stats(self) -> dict[str, Any]:
        if not self._judge:
            return {"judge": None, "calls": 0, "cache_hits": 0, "lookups": self.lookups,
                    "calls_per_lookup": 0.0}
        return {
            "judge": self._judge.name, "calls": self._judge.calls, "cache_hits": self._judge.hits,
            "lookups": self.lookups,
            "calls_per_lookup": (self._judge.calls / self.lookups) if self.lookups else 0.0,
        }

    # --- write path (fire and forget) -------------------------------------
    def record(self, run_id: str, claim_id: str, text: str, kind: str, metadata: dict[str, Any],
               agent_id: str | None, vector: list[float] | None = None) -> None:
        self._pending.append((run_id, claim_id, text, kind, metadata, agent_id, vector))

    def drain(self) -> None:
        """Embed and store everything recorded so far (called by the writer thread and on close)."""
        with self._lock:
            q = self._pending
            while q:
                batch = []
                try:
                    for _ in range(64):  # small batches keep the embedding transient small
                        batch.append(q.popleft())
                except IndexError:
                    pass
                try:
                    need = [i for i, b in enumerate(batch) if b[6] is None]
                    vecs = self.embedder.embed([batch[i][2] for i in need]) if need else []
                    for i, v in zip(need, vecs):
                        b = batch[i]
                        batch[i] = b[:6] + (v,)
                    rows = [(run_id, claim_id, vec, text, kind, metadata, agent_id)
                            for run_id, claim_id, text, kind, metadata, agent_id, vec in batch]
                    up = getattr(self.store, "upsert_vectors", None)
                    if up is not None:
                        up(rows)
                    else:  # minimal Store implementations
                        for r in rows:
                            self.store.upsert_vector(*r)
                except Exception:
                    log.exception("swarmscope: claim record failed (%d claims lost)", len(batch))

    def _write_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(0.05)
            if self._pending:
                self.drain()

    # --- read path (budgeted, fails open) ---------------------------------
    def lookup(self, run_id: str, claim_id: str, text: str, kind: str, metadata: dict[str, Any],
               budget_ms: float | None = None) -> tuple[ClaimHit, list[float] | None]:
        t0 = time.perf_counter()
        hit = ClaimHit(claim_id=claim_id)
        if self.policy.mode == "off":
            return hit, None
        self.lookups += 1
        budget = (budget_ms if budget_ms is not None else self.policy.latency_budget_ms) / 1000.0
        fut = self._pool.submit(self._lookup_sync, run_id, text, kind, metadata, hit, t0 + budget)
        vec: list[float] | None = None
        try:
            vec = fut.result(timeout=budget)
        except FutTimeout:
            hit.timed_out = True
            self.timeouts += 1
        except Exception:
            log.exception("swarmscope: claim lookup failed (failing open)")
            hit.timed_out = True
        hit.latency_ms = (time.perf_counter() - t0) * 1000
        return hit, vec

    def _lookup_sync(self, run_id, text, kind, metadata, hit: ClaimHit, deadline: float) -> list[float]:
        p = self.policy
        if self._pending:  # read-your-writes: claims recorded moments ago must be searchable now
            self.drain()
        vec = self.embedder.embed([text])[0]
        scope_run = run_id if p.scope == "run" else None
        cands: list[VectorHit] = self.store.search_vectors(vec, p.top_k + 1, run_id=scope_run, kind=kind)
        cands = [c for c in cands if c.claim_id != hit.claim_id]
        matches: list[Match] = []
        for c in cands:
            if not self.prefilter(metadata, c.metadata):
                continue
            matches.append(Match(c.claim_id, c.text, c.score, c.kind, c.metadata, c.agent_id, c.run_id))
        matches.sort(key=lambda m: m.score, reverse=True)
        judge_calls = 0
        if self._judge is not None:
            for m in matches:
                if m.score < p.judge_min_score or judge_calls >= p.max_judge_calls:
                    break
                if time.perf_counter() > deadline:
                    break
                before = self._judge.calls
                try:
                    r: JudgeResult = self._judge.judge(text, m.text, metadata, m.metadata)
                except Exception:
                    log.exception("swarmscope: judge failed for a candidate")
                    continue
                judge_calls += self._judge.calls - before
                m.judged, m.equivalent, m.judge_confidence, m.rationale = True, r.equivalent, r.confidence, r.rationale
            hit.judged = any(m.judged for m in matches)
            # judged-equivalent matches first, then by score
            matches.sort(key=lambda m: ((m.equivalent or False), m.score), reverse=True)
        hit.similar = matches
        hit.judge_calls = judge_calls
        hit.top_score = matches[0].score if matches else 0.0
        if hit.judged:
            hit.suggest_skip = any(m.equivalent for m in matches)
        else:
            hit.suggest_skip = hit.top_score >= p.threshold
        return vec

    # --- programmatic query API (L4) ------------------------------------------
    def query(self, text: str, *, k: int = 10, run_id: str | None = None, kind: str | None = None,
              metadata: dict[str, Any] | None = None, judge: bool = False) -> list[Match]:
        """Search the claim store without registering a claim.

        Synchronous and unbudgeted (this is an analysis/coordination call, not
        the agent hot path). ``run_id=None`` searches every run in the store,
        which with a shared Postgres/SQLite store means every process in the
        swarm — the live coordination substrate. ``judge=True`` runs stage 2
        on candidates above ``judge_min_score`` (cached).
        """
        self.drain()
        vec = self.embedder.embed([text])[0]
        cands = self.store.search_vectors(vec, k, run_id=run_id, kind=kind)
        meta = metadata or {}
        out = [Match(c.claim_id, c.text, c.score, c.kind, c.metadata, c.agent_id, c.run_id)
               for c in cands if self.prefilter(meta, c.metadata)]
        if judge and self._judge is not None:
            for m in out:
                if m.score < self.policy.judge_min_score:
                    continue
                r = self._judge.judge(text, m.text, meta, m.metadata)
                m.judged, m.equivalent, m.judge_confidence, m.rationale = True, r.equivalent, r.confidence, r.rationale
            out.sort(key=lambda m: ((m.equivalent or False), m.score), reverse=True)
        return out

    def close(self) -> None:
        self._stop.set()
        self._writer.join(2.0)
        self.drain()
        self._pool.shutdown(wait=True)
