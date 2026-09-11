"""Reputation router: "who handled requests like this well before?"

``Router.advise(text)`` embeds the request, recalls similar past requests
(vectors stored under kind ``__request__``), gathers the verdict outcomes of
the routes that handled them, folds in each route's global record as a
prior, and ranks routes with a bandit policy. The result is advice; the
caller decides whether to reroute.

Learning is incremental: ``record_outcome`` is called by the SDK on every
verdict for an artifact produced under a request. ``rebuild`` recomputes
everything from the event log (needed after post-hoc verdicts such as
``swarmscope review``, or when changing policy weights).
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..claims.embedder import Embedder, HashEmbedder
from ..core.events import Artifact, Request, Verdict
from ..core.ids import stable_hash
from ..store.base import RouteOutcome, RouteStat, Store
from .bandit import BanditPolicy, RouteScore, beta_ci

log = logging.getLogger("swarmscope.reputation")

REQUEST_KIND = "__request__"


def request_vector_kind(kind: str | None) -> str:
    """Requests are recalled within their caller-declared ``kind`` (category)."""
    return REQUEST_KIND if not kind or kind == "request" else f"{REQUEST_KIND}:{kind}"


def route_key(route: Sequence[str]) -> str:
    return stable_hash(list(route), 16)


def agent_identity(role: str | None, name: str | None, model: str | None, config_hash: str | None) -> str:
    """Stable identity of an agent across runs: what it is, not which instance it was."""
    return f"{role or name or 'agent'}|{model or ''}|{config_hash or ''}"


@dataclass(slots=True)
class SimilarRequest:
    request_id: str
    text: str
    similarity: float
    run_id: str
    outcomes: list[RouteOutcome] = field(default_factory=list)


@dataclass
class RoutingAdvice:
    """Advisory result of ``sdk.request``. Also a context manager: entering it
    tags everything produced inside with the request id so outcomes are learned."""

    request_id: str
    text: str
    kind: str
    routes: list[RouteScore]
    similar: list[SimilarRequest]
    policy: str
    cold_start: bool
    latency_ms: float = 0.0
    unit: str = "sequence"
    _enter: Any = field(default=None, repr=False)
    _token: Any = field(default=None, repr=False)

    @property
    def recommended(self) -> RouteScore | None:
        return self.routes[0] if self.routes else None

    @property
    def recommended_agent(self) -> str | None:
        r = self.recommended
        return r.agent if r else None

    def by_agent(self) -> list[RouteScore]:
        """Aggregate route scores by final agent identity (for callers that pick an agent, not a path)."""
        agg: dict[str, list[RouteScore]] = defaultdict(list)
        for r in self.routes:
            agg[r.agent].append(r)
        out = []
        for agent, rs in agg.items():
            ls = sum(r.local_successes for r in rs)
            lf = sum(r.local_failures for r in rs)
            gs = sum(r.global_successes for r in rs)
            gf = sum(r.global_failures for r in rs)
            best = max(rs, key=lambda r: r.score)
            out.append(RouteScore((agent,), route_key([agent]), (ls + 1) / (ls + lf + 2), best.score,
                                  *beta_ci(1 + ls, 1 + lf), ls, lf, gs, gf,
                                  sum(r.similar_requests for r in rs),
                                  max(r.mean_similarity for r in rs)))
        out.sort(key=lambda r: r.score, reverse=True)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "policy": self.policy, "cold_start": self.cold_start,
                "latency_ms": self.latency_ms, "unit": self.unit,
                "routes": [r.to_dict() for r in self.routes[:10]],
                "similar": [{"request_id": s.request_id, "similarity": s.similarity,
                             "accepted": sum(1 for o in s.outcomes if o.accepted), "outcomes": len(s.outcomes)}
                            for s in self.similar[:10]]}

    # context manager: scope work to this request
    def __enter__(self) -> "RoutingAdvice":
        if self._enter is not None:
            self._token = self._enter(self.request_id)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            from ..core import context as ctx

            ctx.reset_context(self._token)
            self._token = None


@dataclass
class RouterPolicy(BanditPolicy):
    #: similar requests to recall
    top_k: int = 20
    #: ignore past requests below this cosine similarity (evidence is similarity-weighted anyway,
    #: so this mainly bounds recall cost; paraphrases score low under the lexical default embedder)
    min_similarity: float = 0.05
    #: also consider the best globally-rated routes even if none of them handled a similar request
    global_candidates: int = 5
    #: "sequence" ranks full agent paths; "agent" ranks by final agent identity
    unit: str = "sequence"
    #: recall scope: "global" (all runs; the default — reputation is cross-run) or "run"
    scope: str = "global"


class Router:
    def __init__(self, store: Store, *, embedder: Embedder | None = None, policy: RouterPolicy | None = None,
                 seed: int | None = None) -> None:
        self.store = store
        self.embedder = embedder or HashEmbedder()
        self.policy = policy or RouterPolicy()
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.advised = 0
        self.outcomes_recorded = 0
        #: agent identities this process knows about: the bandit's arms, including untried ones
        self.known_identities: set[str] = set()

    def register_identity(self, identity: str) -> None:
        self.known_identities.add(identity)

    # ------------------------------------------------------------------ advise
    def advise(self, text: str, *, request_id: str, kind: str = "request", metadata: dict[str, Any] | None = None,
               run_id: str | None = None, candidates: Iterable[str] | None = None) -> RoutingAdvice:
        """Rank routes for ``text``. ``candidates`` are agent identities (or full routes as
        ``"a|m| > b|m|"`` strings / tuples) that must be considered even without evidence;
        by default every identity registered in this process is a candidate, so untried
        agents get explored instead of being invisible to the bandit."""
        t0 = time.perf_counter()
        p = self.policy
        self.advised += 1
        vec = self.embedder.embed([text])[0]
        scope_run = run_id if p.scope == "run" else None
        hits = self.store.search_vectors(vec, p.top_k + 1, run_id=scope_run, kind=request_vector_kind(kind))
        hits = [h for h in hits if h.claim_id != request_id and h.score >= p.min_similarity]
        outcomes = self.store.route_outcomes([h.claim_id for h in hits]) if hits else []
        by_req: dict[str, list[RouteOutcome]] = defaultdict(list)
        for o in outcomes:
            by_req[o.request_id].append(o)
        similar = [SimilarRequest(h.claim_id, h.text, h.score, h.run_id, by_req.get(h.claim_id, [])) for h in hits]

        now = time.time()
        local: dict[str, dict[str, Any]] = {}
        for s in similar:
            for o in s.outcomes:
                key = self._unit_key(o.route)
                d = local.setdefault(key, {"route": self._unit_route(o.route), "s": 0.0, "f": 0.0, "reqs": set(), "sims": []})
                w = s.similarity * o.weight * p.decay(now - o.ts)
                d["s" if o.accepted else "f"] += w
                d["reqs"].add(o.request_id)
                d["sims"].append(s.similarity)

        glob: dict[str, tuple[list[str], float, float]] = {}
        for st in self.store.route_stats():
            key = self._unit_key(st.route)
            r, gs, gf = glob.get(key, (self._unit_route(st.route), 0.0, 0.0))
            glob[key] = (r, gs + st.successes, gf + st.failures)
        cands = dict(local)
        for key, (route, gs, gf) in sorted(glob.items(), key=lambda kv: -(kv[1][1] / (kv[1][1] + kv[1][2] + 1e-9)))[: p.global_candidates]:
            cands.setdefault(key, {"route": route, "s": 0.0, "f": 0.0, "reqs": set(), "sims": []})
        # untried arms: explicit candidates and every identity this process has registered
        universe: list[tuple[str, ...]] = []
        for c in (candidates if candidates is not None else self.known_identities):
            if isinstance(c, str):
                universe.append(tuple(x.strip() for x in c.split(">")) if ">" in c else (c,))
            else:
                universe.append(tuple(c))
        for route in universe:
            r = self._unit_route(route)
            cands.setdefault(route_key(r), {"route": r, "s": 0.0, "f": 0.0, "reqs": set(), "sims": []})
        candidates = cands

        total_n = sum(d["s"] + d["f"] for d in candidates.values())
        scores: list[RouteScore] = []
        with self._lock:
            for key, d in candidates.items():
                _, gs, gf = glob.get(key, (d["route"], 0.0, 0.0))
                a, b = p.posterior(d["s"], d["f"], gs, gf)
                lo, hi = beta_ci(a, b)
                scores.append(RouteScore(tuple(d["route"]), key, a / (a + b), p.score(a, b, total_n, self._rng), lo, hi,
                                         d["s"], d["f"], gs, gf, len(d["reqs"]),
                                         (sum(d["sims"]) / len(d["sims"])) if d["sims"] else 0.0))
        scores = [s for s in scores if (s.local_successes + s.local_failures + s.global_successes + s.global_failures) >= p.min_evidence]
        scores.sort(key=lambda s: s.score, reverse=True)
        adv = RoutingAdvice(request_id, text, kind, scores, similar, p.policy, cold_start=not scores,
                            latency_ms=(time.perf_counter() - t0) * 1000, unit=p.unit)
        return adv

    def _unit_route(self, route: Sequence[str]) -> list[str]:
        return list(route) if self.policy.unit == "sequence" else list(route[-1:])

    def _unit_key(self, route: Sequence[str]) -> str:
        return route_key(self._unit_route(route))

    # ---------------------------------------------------------------- learning
    def index_request(self, run_id: str, request_id: str, text: str, metadata: dict[str, Any], agent_id: str | None,
                      vector: list[float] | None = None, kind: str = "request") -> None:
        vec = vector if vector is not None else self.embedder.embed([text])[0]
        try:
            self.store.upsert_vector(run_id, request_id, vec, text, request_vector_kind(kind), metadata, agent_id)
        except Exception:
            log.exception("swarmscope: could not index request")

    def record_outcome(self, *, request_id: str, artifact_id: str, run_id: str, route: Sequence[str], accepted: bool,
                       source: str, confidence: float, ts: float | None = None) -> None:
        """Fold one verdict into the route's reputation (idempotent per artifact)."""
        if not route:
            return
        w = self.policy.source_weights.get(source, 0.5) * max(0.0, min(1.0, confidence))
        if w <= 0:
            return
        ts = ts or time.time()
        key = route_key(route)
        try:
            prior = {(o.request_id, o.artifact_id): o for o in self.store.route_outcomes([request_id])}
            old = prior.get((request_id, artifact_id))
            if old is not None:  # verdict revised: undo the old contribution
                self.store.route_stats_add(old.route_key, old.route, -old.weight if old.accepted else 0.0,
                                           0.0 if old.accepted else -old.weight, ts)
            self.store.route_outcomes_put([RouteOutcome(request_id, artifact_id, run_id, key, list(route), accepted, w, ts, source)])
            self.store.route_stats_add(key, route, w if accepted else 0.0, 0.0 if accepted else w, ts)
            self.outcomes_recorded += 1
        except Exception:
            log.exception("swarmscope: could not record route outcome")

    def rebuild(self, run_ids: Iterable[str] | None = None) -> int:
        """Recompute outcomes and stats from the event log (all runs by default)."""
        self.store.route_stats_clear()
        n = 0
        ids = list(run_ids) if run_ids is not None else [r.run_id for r in self.store.runs()]
        for rid in ids:
            arts: dict[str, Artifact] = {}
            reqs: dict[str, Request] = {}
            latest: dict[str, Verdict] = {}
            for e in self.store.events(rid, types=["artifact", "request", "verdict"]):
                if isinstance(e, Artifact):
                    arts[e.artifact_id] = e
                elif isinstance(e, Request):
                    reqs[e.request_id] = e
                elif isinstance(e, Verdict) and e.status != "pending":
                    cur = latest.get(e.artifact_id)
                    if cur is None or e.ts >= cur.ts:
                        latest[e.artifact_id] = e
            for rq in reqs.values():
                self.index_request(rid, rq.request_id, rq.text, rq.metadata, rq.agent_id, kind=rq.kind)
            for art_id, v in latest.items():
                art = arts.get(art_id)
                if art is None or not art.request_id or not art.route:
                    continue
                self.record_outcome(request_id=art.request_id, artifact_id=art_id, run_id=rid, route=art.route,
                                    accepted=(v.status == "accepted"), source=v.source, confidence=v.confidence, ts=v.ts)
                n += 1
        return n

    def leaderboard(self, limit: int = 20, unit: str | None = None) -> list[RouteScore]:
        """Global reputation table (no request context)."""
        unit = unit or self.policy.unit
        agg: dict[str, tuple[list[str], float, float, float]] = {}
        for st in self.store.route_stats():
            r = list(st.route) if unit == "sequence" else list(st.route[-1:])
            key = route_key(r)
            _, s, f, ts = agg.get(key, (r, 0.0, 0.0, 0.0))
            agg[key] = (r, s + st.successes, f + st.failures, max(ts, st.last_ts))
        out = []
        for key, (r, s, f, _) in agg.items():
            a, b = self.policy.prior_a + s, self.policy.prior_b + f
            lo, hi = beta_ci(a, b)
            out.append(RouteScore(tuple(r), key, a / (a + b), a / (a + b), lo, hi, 0.0, 0.0, s, f))
        out.sort(key=lambda x: (x.mean, x.global_successes + x.global_failures), reverse=True)
        return out[:limit]

    @property
    def stats(self) -> dict[str, Any]:
        return {"advised": self.advised, "outcomes_recorded": self.outcomes_recorded, "policy": self.policy.policy,
                "unit": self.policy.unit}
