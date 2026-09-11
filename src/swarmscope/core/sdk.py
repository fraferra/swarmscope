"""The reference implementation. Framework adapters are thin shims onto this.

Typical use::

    import swarmscope as ss
    sdk = ss.init("sqlite:///swarm.db")

    @sdk.agent(role="worker", group="search")
    def worker(task): ...

    with sdk.run(name="proof-search") as run:
        for t in tasks:
            worker(t)
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

from ..attribution.pricing import DEFAULT_PRICING, PricingTable
from ..claims.embedder import Embedder
from ..claims.judge import EquivalenceJudge
from ..claims.store import ClaimHit, ClaimStore, GatePolicy, Match
from ..reputation.router import Router, RouterPolicy, RoutingAdvice, agent_identity
from ..store.base import Store, open_store
from . import context as ctx
from .buffer import EventBuffer
from .events import (UNKNOWN, AgentEnd, AgentStart, Artifact, Claim, Consolidation, Event,
                     Generation, Message, Request, Suppression, ToolCall, Verdict)
from .ids import config_hash, hash_bytes, new_id, stable_hash

log = logging.getLogger("swarmscope")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass(slots=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    content_hash: str
    agent_id: str | None
    run_id: str


@dataclass(slots=True)
class RunHandle:
    run_id: str
    name: str | None
    started_at: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Gated/ungated A/B for the dedup gate. Assignment is deterministic by agent id."""

    name: str = "dedup-gate"
    arms: tuple[str, ...] = ("gated", "ungated")
    gated_fraction: float = 0.5

    def assign(self, agent_id: str | None) -> str:
        if not agent_id:
            return self.arms[0]
        h = int(stable_hash(f"{self.name}:{agent_id}", 8), 16) / 0xFFFFFFFF
        return self.arms[0] if h < self.gated_fraction else self.arms[1]


class Swarmscope:
    def __init__(
        self,
        store: str | Store | None = None,
        *,
        pricing: PricingTable | None = None,
        retain_content: bool = True,
        embedder: Embedder | None = None,
        judge: EquivalenceJudge | None = None,
        gate: GatePolicy | None = None,
        experiment: ExperimentConfig | None = None,
        router: RouterPolicy | None = None,
        service_name: str = "swarmscope",
        batch_size: int = 256,
        flush_interval: float = 0.25,
        max_queue: int = 100_000,
    ) -> None:
        self.store: Store = open_store(store)
        self.pricing = pricing or DEFAULT_PRICING
        self.retain_content = retain_content
        self.service_name = service_name
        self.buffer = EventBuffer(self.store, batch_size=batch_size, flush_interval=flush_interval,
                                  max_queue=max_queue)
        self.claims = ClaimStore(self.store, embedder=embedder, judge=judge, policy=gate)
        self.reputation = Router(self.store, embedder=self.claims.embedder, policy=router)
        self.experiment = experiment
        #: artifact_id -> (route, request_id, run_id) for in-process verdict → reputation updates
        self._artifact_meta: dict[str, tuple[tuple[str, ...], str | None, str]] = {}
        self._consolidators: dict[str, Callable[..., Any]] = {}
        self._exporters: list[Callable[[Event], None]] = []
        self._default_run: RunHandle | None = None
        self._lock = threading.Lock()
        self.unknown_lineage_events = 0
        self.total_events = 0

    # ------------------------------------------------------------------ core
    def emit(self, event: Event) -> None:
        self.total_events += 1
        if event.agent_id == UNKNOWN or event.parent_id == UNKNOWN:
            self.unknown_lineage_events += 1
        self.buffer.emit(event)
        if self._exporters:
            for exp in self._exporters:
                try:
                    exp(event)
                except Exception:  # never let an exporter hurt the swarm
                    log.exception("swarmscope: exporter failed")

    def add_exporter(self, fn: Callable[[Event], None]) -> None:
        """Register a synchronous per-event hook (used by the OTLP exporter)."""
        self._exporters.append(fn)

    def flush(self, timeout: float = 5.0) -> None:
        self.buffer.flush(timeout)
        self.claims.drain()

    def close(self) -> None:
        self.buffer.close()
        self.claims.close()
        self.store.close()

    def __enter__(self) -> "Swarmscope":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ runs
    @contextmanager
    def run(self, name: str | None = None, *, run_id: str | None = None,
            meta: dict[str, Any] | None = None, workload: str | None = None) -> Iterator[RunHandle]:
        rid = run_id or new_id("run_")
        meta = dict(meta or {})
        if workload:
            meta["workload"] = workload
        handle = RunHandle(rid, name, time.time(), meta)
        try:
            self.store.begin_run(rid, name, handle.started_at, meta)
        except Exception:
            log.exception("swarmscope: begin_run failed")
        baggage = {"workload": workload} if workload else {}
        with ctx.use(ctx.RunContext(run_id=rid, baggage=baggage)):
            try:
                yield handle
            finally:
                self.flush()
                try:
                    self.store.end_run(rid, time.time())
                except Exception:
                    log.exception("swarmscope: end_run failed")

    def _run_id(self) -> str:
        c = ctx.current()
        if c.run_id:
            return c.run_id
        # No active run: events still get captured under an implicit run so
        # nothing is silently lost; lineage is UNKNOWN by construction.
        with self._lock:
            if self._default_run is None:
                rid = new_id("run_")
                self._default_run = RunHandle(rid, "<implicit>", time.time())
                try:
                    self.store.begin_run(rid, "<implicit>", self._default_run.started_at, {})
                except Exception:
                    pass
        return self._default_run.run_id

    def _base(self, *, run_level: bool = False, **kw) -> dict[str, Any]:
        """Lineage columns for a new event.

        Work events (generation, tool, message, claim, artifact) emitted outside
        any agent get ``agent_id=UNKNOWN``: their owner was lost. Run-level
        events (consolidation, verdicts issued by a verifier/CLI) legitimately
        have no agent and get ``None``.
        """
        c = ctx.current()
        agent = c.agent_id if (c.agent_id or run_level) else UNKNOWN
        base = dict(run_id=c.run_id or self._run_id(), group_id=c.group_id, agent_id=agent,
                    parent_id=c.parent_agent_id)
        base.update(kw)
        return base

    @contextmanager
    def group(self, group_id: str) -> Iterator[str]:
        with ctx.scoped(group_id=group_id):
            yield group_id

    # ---------------------------------------------------------------- agents
    @contextmanager
    def agent_scope(self, *, role: str | None = None, group: str | None = None, model: str | None = None,
                    config: dict[str, Any] | None = None, name: str | None = None,
                    agent_id: str | None = None, parent_id: str | None = None,
                    identity: str | None = None) -> Iterator[str]:
        """Explicit agent boundary. ``parent_id`` overrides the inherited lineage
        (pass :data:`UNKNOWN` when you *know* it was lost). ``identity`` overrides
        the stable identity used for reputation (default: role|model|config hash)."""
        aid = agent_id or new_id("ag_")
        c = ctx.current()
        parent = parent_id if parent_id is not None else (c.agent_id or c.parent_agent_id)
        gid = group or c.group_id
        run_id = c.run_id or self._run_id()
        chash = config_hash(config) if config else None
        ident = identity or agent_identity(role, name, model, chash)
        self.reputation.known_identities.add(ident)
        self.emit(AgentStart(run_id=run_id, group_id=gid, agent_id=aid, parent_id=parent, role=role, model=model,
                             config_hash=chash, name=name, attrs={"identity": ident}))
        t0 = time.perf_counter()
        status, err = "ok", None
        # Direct RunContext construction + set/reset: cheaper than replace() inside a nested context manager.
        token = ctx.set_context(ctx.RunContext(run_id, gid, aid, parent, c.causes, self._arm_for(aid, c), c.baggage,
                                               c.path + (ident,), c.request_id))
        try:
            yield aid
        except BaseException as e:  # noqa: BLE001
            status, err = ("cancelled" if isinstance(e, asyncio.CancelledError) else "error"), repr(e)[:500]
            raise
        finally:
            ctx.reset_context(token)
            self.emit(AgentEnd(run_id=run_id, group_id=gid, agent_id=aid, parent_id=parent, status=status, error=err,
                               duration_ms=(time.perf_counter() - t0) * 1000))

    def _arm_for(self, agent_id: str, c: ctx.RunContext) -> str | None:
        if c.arm:
            return c.arm
        return self.experiment.assign(agent_id) if self.experiment else None

    def agent(self, fn: F | None = None, *, role: str | None = None, group: str | None = None,
              model: str | None = None, config: dict[str, Any] | None = None, name: str | None = None,
              identity: str | None = None) -> Any:
        """Decorator marking a function (sync or async) as an agent invocation."""

        def deco(f: F) -> F:
            nm = name or f.__name__
            # Register the arm at decoration time so the router can explore it before it has ever run.
            self.reputation.register_identity(identity or agent_identity(role, nm, model, config_hash(config) if config else None))
            if inspect.iscoroutinefunction(f):
                @functools.wraps(f)
                async def aw(*a, **kw):
                    with self.agent_scope(role=role, group=group, model=model, config=config, name=nm, identity=identity):
                        return await f(*a, **kw)
                return aw  # type: ignore[return-value]

            @functools.wraps(f)
            def w(*a, **kw):
                with self.agent_scope(role=role, group=group, model=model, config=config, name=nm, identity=identity):
                    return f(*a, **kw)
            return w  # type: ignore[return-value]

        return deco(fn) if fn is not None else deco

    # ----------------------------------------------------------- generations
    def generation(self, *, model: str | None, input_tokens: int = 0, output_tokens: int = 0,
                   cached_input_tokens: int = 0, reasoning_tokens: int = 0, latency_ms: float | None = None,
                   cost_usd: float | None = None, system: str | None = None, request_id: str | None = None,
                   batch: bool = False, **attrs) -> Generation:
        if cost_usd is None:
            cost_usd = self.pricing.cost(model, input_tokens, output_tokens, cached_input_tokens,
                                         reasoning_tokens, batch)
        ev = Generation(**self._base(), model=model, system=system, input_tokens=input_tokens,
                        output_tokens=output_tokens, cached_input_tokens=cached_input_tokens,
                        reasoning_tokens=reasoning_tokens, latency_ms=latency_ms, cost_usd=cost_usd,
                        request_id=request_id, batch=batch, attrs=attrs)
        self.emit(ev)
        return ev

    # -------------------------------------------------------------- messages
    def message(self, payload: Any = None, *, to: Sequence[str] | str | None = None,
                causes: Sequence[str] | None = None, retain: bool | None = None, **attrs) -> str:
        """Record a message from the current agent. Returns the message id."""
        c = ctx.current()
        recipients = [to] if isinstance(to, str) else list(to or [])
        all_causes = list(c.causes) + list(causes or [])
        keep = self.retain_content if retain is None else retain
        try:
            raw = json.dumps(payload, default=repr)
        except Exception:
            raw = repr(payload)
        ev = Message(**self._base(), sender=c.agent_id or UNKNOWN, recipients=recipients,
                     payload_ref=hash_bytes(raw), payload=payload if keep else None,
                     causes=all_causes, size_bytes=len(raw), attrs=attrs)
        self.emit(ev)
        return ev.event_id

    @contextmanager
    def link(self, cause: str | Sequence[str] | None = None, *, parent: str | None = None,
             run_id: str | None = None, group_id: str | None = None) -> Iterator[None]:
        """Attach causal/lineage information for out-of-band handoffs.

        ``cause`` = message id(s) that led to the work done inside the block.
        ``parent`` = agent id of the sender when the handoff crossed a process
        boundary and contextvars could not carry it.
        """
        causes = [cause] if isinstance(cause, str) else list(cause or [])
        c = ctx.current()
        changes: dict[str, Any] = {"causes": tuple(c.causes) + tuple(causes)}
        if parent is not None:
            changes["parent_agent_id"] = parent
        if run_id is not None:
            changes["run_id"] = run_id
        if group_id is not None:
            changes["group_id"] = group_id
        with ctx.scoped(**changes):
            yield

    receive = link  # alias: ``with sdk.receive(msg_id): ...``

    def headers(self) -> dict[str, str]:
        """Lineage headers to send with an out-of-band handoff."""
        return ctx.current().to_headers()

    attach_headers = staticmethod(ctx.attach_headers)

    # ----------------------------------------------------------------- tools
    def tool_call(self, name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
        t0 = time.perf_counter()
        err = None
        result = None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            err = repr(e)[:500]
            raise
        finally:
            self.emit(ToolCall(**self._base(), tool=name, args_hash=stable_hash([args, kwargs]),
                               result_hash=None if err else stable_hash(result),
                               latency_ms=(time.perf_counter() - t0) * 1000, error=err))

    def tool(self, fn: F | None = None, *, name: str | None = None) -> Any:
        def deco(f: F) -> F:
            nm = name or f.__name__
            if inspect.iscoroutinefunction(f):
                @functools.wraps(f)
                async def aw(*a, **kw):
                    t0 = time.perf_counter()
                    err, result = None, None
                    try:
                        result = await f(*a, **kw)
                        return result
                    except Exception as e:
                        err = repr(e)[:500]
                        raise
                    finally:
                        self.emit(ToolCall(**self._base(), tool=nm, args_hash=stable_hash([a, kw]),
                                           result_hash=None if err else stable_hash(result),
                                           latency_ms=(time.perf_counter() - t0) * 1000, error=err))
                return aw  # type: ignore[return-value]

            @functools.wraps(f)
            def w(*a, **kw):
                return self.tool_call(nm, f, *a, **kw)
            return w  # type: ignore[return-value]

        return deco(fn) if fn is not None else deco

    # ---------------------------------------------------------------- claims
    def claim(self, text: str, *, kind: str = "hypothesis", status: str = "exploring",
              metadata: dict[str, Any] | None = None, inputs: Sequence[str] | None = None,
              lookup: bool | None = None, budget_ms: float | None = None, **attrs) -> ClaimHit:
        """Register a claim and ask "has anyone already tried this?".

        Returns a :class:`ClaimHit`. Advisory: nothing is blocked. Under an
        experiment, agents in the ``ungated`` arm get an empty hit (the claim
        is still recorded so both arms are measured identically).
        """
        metadata = dict(metadata or {})
        c = ctx.current()
        base = self._base()
        run_id = base["run_id"]
        claim_id = new_id("cl_")
        arm = c.arm if c.arm else (self.experiment.assign(c.agent_id) if self.experiment else None)
        do_lookup = lookup if lookup is not None else (arm != "ungated")
        vec = None
        if do_lookup:
            hit, vec = self.claims.lookup(run_id, claim_id, text, kind, metadata, budget_ms)
        else:
            hit = ClaimHit(claim_id=claim_id)
        hit.arm = arm
        hit._on_suppress = self._suppress
        ev = Claim(**base, claim_id=claim_id, text=text, kind=kind, status=status, metadata=metadata,
                   dedup=hit.to_dict() if do_lookup else {"skipped": True}, arm=arm,
                   inputs=list(c.causes) + list(inputs or []), attrs=attrs)
        self.emit(ev)
        self.claims.record(run_id, claim_id, text, kind, metadata, base["agent_id"], vec)
        return hit

    def _suppress(self, hit: ClaimHit, reason: str | None) -> None:
        best = hit.best
        self.emit(Suppression(**self._base(), claim_id=hit.claim_id,
                              suppressed_by=best.claim_id if best else "", score=hit.top_score, reason=reason))

    def search_claims(self, text: str, *, k: int = 10, run_id: str | None = "current", kind: str | None = None,
                      metadata: dict[str, Any] | None = None, judge: bool = False) -> list[Match]:
        """Query the claim store without registering a claim. ``run_id="current"``
        (default) scopes to the active run; ``None`` searches every run in the store."""
        rid = self._run_id() if run_id == "current" else run_id
        return self.claims.query(text, k=k, run_id=rid, kind=kind, metadata=metadata, judge=judge)

    def update_claim(self, claim_id: str, status: str, **attrs) -> None:
        """Emit a status transition for an existing claim (e.g. ``dead_end``)."""
        self.emit(Claim(**self._base(), claim_id=claim_id, status=status, kind=attrs.pop("kind", "update"),
                        attrs=attrs))

    # ------------------------------------------------------------- artifacts
    def artifact(self, content: Any = None, *, kind: str = "artifact", inputs: Sequence[str] | None = None,
                 retain: bool | None = None, **attrs) -> ArtifactRef:
        c = ctx.current()
        base = self._base()
        keep = self.retain_content if retain is None else retain
        try:
            raw = json.dumps(content, default=repr)
        except Exception:
            raw = repr(content)
        ev = Artifact(**base, kind=kind, content_hash=hash_bytes(raw), content=content if keep else None,
                      inputs=list(c.causes) + list(inputs or []), size_bytes=len(raw), attrs=attrs,
                      route=list(c.path), request_id=c.request_id)
        self.emit(ev)
        if c.request_id and c.path:
            if len(self._artifact_meta) > 50_000:
                self._artifact_meta.clear()
            self._artifact_meta[ev.artifact_id] = (c.path, c.request_id, base["run_id"])
        return ArtifactRef(ev.artifact_id, kind, ev.content_hash or "", base["agent_id"], base["run_id"])

    def verdict(self, artifact: ArtifactRef | str, *, status: str, source: str, confidence: float = 1.0,
                evidence: dict[str, Any] | None = None, inferred: bool = False, run_id: str | None = None,
                **attrs) -> Verdict:
        if source not in ("human", "verifier", "judge", "downstream"):
            raise ValueError(f"verdict source must be human|verifier|judge|downstream, got {source!r}")
        if status not in ("accepted", "rejected", "pending"):
            raise ValueError(f"verdict status must be accepted|rejected|pending, got {status!r}")
        aid = artifact.artifact_id if isinstance(artifact, ArtifactRef) else artifact
        base = self._base(run_level=True)
        if run_id:
            base["run_id"] = run_id
        elif isinstance(artifact, ArtifactRef):
            base["run_id"] = artifact.run_id
        ev = Verdict(**base, artifact_id=aid, status=status, source=source, confidence=float(confidence),
                     evidence=dict(evidence or {}), inferred=inferred, attrs=attrs)
        self.emit(ev)
        meta = self._artifact_meta.get(aid) if status != "pending" else None
        if meta is not None:
            route, req_id, rid = meta
            if req_id:
                self.reputation.record_outcome(request_id=req_id, artifact_id=aid, run_id=rid, route=route,
                                               accepted=(status == "accepted"), source=source,
                                               confidence=float(confidence), ts=ev.ts)
        return ev

    # ---------------------------------------------------------------- requests
    def request(self, text: str, *, kind: str = "request", metadata: dict[str, Any] | None = None,
                candidates: Sequence[str] | None = None, **attrs) -> RoutingAdvice:
        """Ask the reputation router who should handle this work.

        Returns :class:`RoutingAdvice` (advisory: routes ranked by a bandit
        policy over past verdicts on similar requests). Use it as a context
        manager so artifacts produced inside are tied to the request and their
        verdicts update the routes' reputation::

            with sdk.request("prove lemma 3", kind="proof") as adv:
                agent = pick(adv.recommended_agent)   # your decision
                ...
        """
        metadata = dict(metadata or {})
        base = self._base(run_level=True)
        req_id = new_id("rq_")
        adv = self.reputation.advise(text, request_id=req_id, kind=kind, metadata=metadata, run_id=base["run_id"],
                                     candidates=candidates)
        self.emit(Request(**base, request_id=req_id, text=text, kind=kind, metadata=metadata,
                          advice=adv.to_dict(), attrs=attrs))
        self.reputation.index_request(base["run_id"], req_id, text, metadata, base["agent_id"], kind=kind)

        def _enter(rid: str):
            c = ctx.current()
            return ctx.set_context(replace(c, request_id=rid))

        adv._enter = _enter
        return adv

    # ---------------------------------------------------------- consolidation
    def consolidator(self, fn: F | None = None, *, name: str | None = None,
                     check_determinism: bool = False) -> Any:
        """Mark the aggregation step. Inputs and output are recorded verbatim.

        The wrapped function must accept ``contributions`` — a list of
        :class:`Contribution` (or ``(agent_id, value)`` pairs / dicts) — as its
        first positional argument, and must be a pure function of them for
        replay to be valid. With ``check_determinism=True`` the function is
        called twice on first use and the outputs compared; a mismatch marks
        the record non-replayable and ablation falls back to online proxies.
        """

        def deco(f: F) -> F:
            nm = name or f.__name__
            self._consolidators[nm] = f

            @functools.wraps(f)
            def w(contributions, *a, **kw):
                inputs = [Contribution.coerce(x).to_dict() for x in contributions]
                replayable, reason, deterministic = True, None, None
                try:
                    json.dumps(inputs)
                except Exception as e:
                    replayable, reason = False, f"inputs not JSON-serialisable: {e!r}"
                if a or kw:
                    replayable, reason = False, "extra positional/keyword args are not recorded"
                out = f(contributions, *a, **kw)
                if check_determinism and replayable:
                    try:
                        out2 = f(contributions, *a, **kw)
                        deterministic = stable_hash(out) == stable_hash(out2)
                        if not deterministic:
                            replayable, reason = False, "non-deterministic output on identical inputs"
                    except Exception as e:  # pragma: no cover
                        deterministic, replayable, reason = False, False, f"re-run failed: {e!r}"
                self.emit(Consolidation(**self._base(run_level=True), name=nm, inputs=inputs, output=out,
                                        output_hash=stable_hash(out), deterministic=deterministic,
                                        replayable=replayable, reason=reason))
                return out

            return w  # type: ignore[return-value]

        return deco(fn) if fn is not None else deco

    def consolidator_fn(self, name: str) -> Callable[..., Any] | None:
        return self._consolidators.get(name)

    # ---------------------------------------------------------------- stats
    @property
    def stats(self) -> dict[str, Any]:
        return {
            "events": self.total_events,
            "unknown_lineage_events": self.unknown_lineage_events,
            "unknown_lineage_fraction": (self.unknown_lineage_events / self.total_events) if self.total_events else 0.0,
            "buffer": {"pending": self.buffer.pending, "written": self.buffer.written,
                       "dropped": self.buffer.dropped, "errors": self.buffer.errors},
            "claims": self.claims.judge_stats,
            "reputation": self.reputation.stats,
            "unpriced_models": dict(self.pricing.unpriced),
        }


@dataclass(slots=True)
class Contribution:
    """One agent's input to a consolidation step."""

    agent_id: str
    value: Any
    group_id: str | None = None
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "group_id": self.group_id, "value": self.value, "weight": self.weight}

    @classmethod
    def coerce(cls, x: Any) -> "Contribution":
        if isinstance(x, Contribution):
            return x
        if isinstance(x, dict) and "agent_id" in x:
            return cls(x["agent_id"], x.get("value"), x.get("group_id"), x.get("weight", 1.0))
        if isinstance(x, (tuple, list)) and len(x) >= 2:
            return cls(str(x[0]), x[1], x[2] if len(x) > 2 else None)
        raise TypeError(f"cannot coerce {type(x).__name__} to Contribution")


def contributions_from(store: Store, run_id: str, *, kind: str | None = None) -> list[Contribution]:
    """Build consolidation inputs from the run's recorded artifacts."""
    from .events import Artifact as _A

    out = []
    for e in store.events(run_id, types=["artifact"]):
        assert isinstance(e, _A)
        if kind and e.kind != kind:
            continue
        out.append(Contribution(e.agent_id or UNKNOWN, e.content, e.group_id))
    return out
