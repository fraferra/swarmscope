"""Run context propagated through ``contextvars``.

Anything that runs in-process and in the same async task tree inherits
``run_id`` / ``group_id`` / ``agent_id`` automatically. For out-of-band
handoffs (queues, subprocesses, HTTP) callers serialise :meth:`RunContext.to_headers`
on one side and :func:`attach_headers` on the other, or use ``sdk.link``.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Iterator

from .events import UNKNOWN

HEADER_PREFIX = "x-swarmscope-"


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str | None = None
    group_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    #: Message ids that causally precede whatever is emitted next.
    causes: tuple[str, ...] = field(default_factory=tuple)
    #: Experiment arm for the dedup gate, if any.
    arm: str | None = None
    #: Free-form baggage propagated with the context (e.g. workload class).
    baggage: dict[str, str] = field(default_factory=dict)
    #: Stable agent identities from the root to the current agent (the route so far).
    path: tuple[str, ...] = field(default_factory=tuple)
    #: Active ``sdk.request`` this work answers, if any.
    request_id: str | None = None

    @property
    def has_agent(self) -> bool:
        return self.agent_id is not None

    def to_headers(self) -> dict[str, str]:
        h = {}
        for k in ("run_id", "group_id", "agent_id", "arm"):
            v = getattr(self, k)
            if v:
                h[HEADER_PREFIX + k.replace("_", "-")] = v
        if self.causes:
            h[HEADER_PREFIX + "causes"] = ",".join(self.causes)
        if self.path:
            h[HEADER_PREFIX + "path"] = "\x1f".join(self.path)
        if self.request_id:
            h[HEADER_PREFIX + "request-id"] = self.request_id
        return h

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "RunContext":
        lower = {k.lower(): v for k, v in headers.items()}

        def g(name: str) -> str | None:
            return lower.get(HEADER_PREFIX + name)

        causes = tuple(x for x in (g("causes") or "").split(",") if x)
        path = tuple(x for x in (g("path") or "").split("\x1f") if x)
        return cls(
            run_id=g("run-id"),
            group_id=g("group-id"),
            # The remote side's agent becomes our parent; we have no agent yet.
            parent_agent_id=g("agent-id"),
            causes=causes,
            arm=g("arm"),
            path=path,
            request_id=g("request-id"),
        )


_ctx: contextvars.ContextVar[RunContext] = contextvars.ContextVar(
    "swarmscope_ctx", default=RunContext()
)


def current() -> RunContext:
    return _ctx.get()


def set_context(ctx: RunContext) -> contextvars.Token:
    return _ctx.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    _ctx.reset(token)


@contextmanager
def use(ctx: RunContext) -> Iterator[RunContext]:
    token = _ctx.set(ctx)
    try:
        yield ctx
    finally:
        _ctx.reset(token)


@contextmanager
def scoped(**changes) -> Iterator[RunContext]:
    """Derive a child context from the current one with ``changes`` applied."""
    with use(replace(current(), **changes)) as ctx:
        yield ctx


@contextmanager
def attach_headers(headers: dict[str, str]) -> Iterator[RunContext]:
    """Adopt lineage from transport headers produced by :meth:`RunContext.to_headers`."""
    with use(RunContext.from_headers(headers)) as ctx:
        yield ctx


def lineage_parent() -> str | None:
    """Parent agent id for an agent started now.

    ``None`` means a root agent (no parent, by design). Adapters that *know*
    lineage was lost (e.g. an event bus that reports an agent with no owning
    crew) pass :data:`UNKNOWN` explicitly; we never guess here.
    """
    ctx = current()
    return ctx.agent_id or ctx.parent_agent_id


__all__ = [
    "RunContext", "current", "set_context", "reset_context", "use", "scoped",
    "attach_headers", "lineage_parent", "UNKNOWN", "HEADER_PREFIX",
]
