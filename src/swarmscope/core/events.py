"""Event schema.

The wire format follows OpenTelemetry GenAI semantic conventions wherever
OTel already names a concept (``gen_ai.request.model``,
``gen_ai.usage.input_tokens`` ...). Swarm-specific attributes live under the
``swarm.*`` namespace. :mod:`swarmscope.surfaces.otlp` maps these onto spans.

Six event families: AgentStart/AgentEnd, Generation, Message, ToolCall,
Claim, Artifact + Verdict. Every event carries lineage columns
(``run_id``, ``group_id``, ``agent_id``, ``parent_id``). When lineage cannot
be determined we record :data:`UNKNOWN`, never a guess.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from operator import attrgetter
from typing import Any, ClassVar, Literal

from .ids import new_id

#: Sentinel for lineage we could not determine. Reported, never hidden.
UNKNOWN = "unknown"

EventType = Literal[
    "agent_start", "agent_end", "generation", "message", "tool_call",
    "claim", "artifact", "verdict", "consolidation", "suppression", "request",
]

VerdictStatus = Literal["accepted", "rejected", "pending"]
#: Verdict provenance. Never collapse these into a single confidence number.
VerdictSource = Literal["human", "verifier", "judge", "downstream"]
ClaimKind = Literal["hypothesis", "approach", "result", "dead_end", "artifact_ref"]
ClaimStatus = Literal["exploring", "done", "abandoned", "suppressed"]


@dataclass(slots=True)
class Event:
    """Base event. Subclasses add typed fields; everything else goes in ``attrs``."""

    type: ClassVar[str] = "event"

    run_id: str
    event_id: str = field(default_factory=lambda: new_id("ev_"))
    ts: float = field(default_factory=time.time)
    group_id: str | None = None
    agent_id: str | None = None
    #: Parent *agent* id for lineage; UNKNOWN when it could not be determined.
    parent_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # Shallow and C-level: ``asdict`` deep-copies every nested container and is ~10x slower.
        names, getter = _FIELD_ACCESS[type(self)]
        d = dict(zip(names, getter(self)))
        d["type"] = self.type
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        d = dict(d)
        t = d.pop("type", None)
        klass = EVENT_TYPES.get(t, Event)
        names = {f.name for f in fields(klass)}
        known = {k: v for k, v in d.items() if k in names}
        extra = {k: v for k, v in d.items() if k not in names}
        if extra:
            known.setdefault("attrs", {}).update(extra)
        return klass(**known)  # type: ignore[arg-type]


@dataclass(slots=True)
class AgentStart(Event):
    type: ClassVar[str] = "agent_start"
    role: str | None = None
    model: str | None = None
    config_hash: str | None = None
    name: str | None = None


@dataclass(slots=True)
class AgentEnd(Event):
    type: ClassVar[str] = "agent_end"
    status: str = "ok"  # ok | error | cancelled
    error: str | None = None
    duration_ms: float | None = None
    #: Hash of the agent's return value, for replay / dedup of identical outputs.
    result_hash: str | None = None


@dataclass(slots=True)
class Generation(Event):
    type: ClassVar[str] = "generation"
    model: str | None = None
    system: str | None = None  # gen_ai.system, e.g. "openai", "anthropic"
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float | None = None
    cost_usd: float | None = None
    request_id: str | None = None
    batch: bool = False


@dataclass(slots=True)
class Message(Event):
    type: ClassVar[str] = "message"
    sender: str | None = None
    recipients: list[str] = field(default_factory=list)
    payload_ref: str | None = None  # hash or store key; never the payload itself by default
    payload: Any = None  # optionally retained verbatim (needed for replay)
    #: Causal parent message ids (from ``sdk.link``/``sdk.receive``).
    causes: list[str] = field(default_factory=list)
    size_bytes: int | None = None


@dataclass(slots=True)
class ToolCall(Event):
    type: ClassVar[str] = "tool_call"
    tool: str | None = None
    args_hash: str | None = None
    result_hash: str | None = None
    latency_ms: float | None = None
    error: str | None = None


@dataclass(slots=True)
class Claim(Event):
    type: ClassVar[str] = "claim"
    claim_id: str = field(default_factory=lambda: new_id("cl_"))
    text: str = ""
    kind: str = "hypothesis"
    status: str = "exploring"
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Dedup lookup outcome recorded alongside the claim (top score, matches).
    dedup: dict[str, Any] = field(default_factory=dict)
    #: Experiment arm this claim was issued under ("gated" | "ungated" | None).
    arm: str | None = None
    #: Ids of message/claim/artifact inputs this claim derived from.
    inputs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Artifact(Event):
    type: ClassVar[str] = "artifact"
    artifact_id: str = field(default_factory=lambda: new_id("ar_"))
    kind: str = "artifact"
    content_hash: str | None = None
    content: Any = None  # retained verbatim when ``retain_content`` is on
    #: Ids (artifacts, claims, messages, agents) this artifact was derived from.
    inputs: list[str] = field(default_factory=list)
    size_bytes: int | None = None
    #: Stable agent identities from the run root to the producer (the "route").
    route: list[str] = field(default_factory=list)
    #: Request this artifact answers, if produced under ``sdk.request``.
    request_id: str | None = None


@dataclass(slots=True)
class Verdict(Event):
    type: ClassVar[str] = "verdict"
    verdict_id: str = field(default_factory=lambda: new_id("vd_"))
    artifact_id: str = ""
    status: str = "pending"
    source: str = "human"
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)
    inferred: bool = False


@dataclass(slots=True)
class Consolidation(Event):
    """Verbatim record of a consolidation/aggregation step for replay (L3)."""

    type: ClassVar[str] = "consolidation"
    consolidation_id: str = field(default_factory=lambda: new_id("co_"))
    name: str = ""
    #: List of ``{"agent_id": ..., "group_id": ..., "value": <json>}``.
    inputs: list[dict[str, Any]] = field(default_factory=list)
    output: Any = None
    output_hash: str | None = None
    deterministic: bool | None = None
    replayable: bool = True
    reason: str | None = None


@dataclass(slots=True)
class Request(Event):
    """A unit of work being routed: "who should handle this?" (reputation layer)."""

    type: ClassVar[str] = "request"
    request_id: str = field(default_factory=lambda: new_id("rq_"))
    text: str = ""
    kind: str = "request"
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Routing advice summary recorded alongside (top routes, policy, cold_start).
    advice: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Suppression(Event):
    """Logged every time a caller acts on dedup advice by skipping work."""

    type: ClassVar[str] = "suppression"
    claim_id: str = ""
    suppressed_by: str = ""
    score: float | None = None
    reason: str | None = None


_FIELD_ACCESS: dict[type, tuple[tuple[str, ...], Any]] = {}
for _cls in (Event, AgentStart, AgentEnd, Generation, Message, ToolCall, Claim, Artifact, Verdict, Consolidation,
             Suppression, Request):
    _names = tuple(f.name for f in fields(_cls))
    _FIELD_ACCESS[_cls] = (_names, attrgetter(*_names))

EVENT_TYPES: dict[str, type[Event]] = {
    c.type: c
    for c in (AgentStart, AgentEnd, Generation, Message, ToolCall, Claim,
              Artifact, Verdict, Consolidation, Suppression, Request)
}

__all__ = [
    "UNKNOWN", "Event", "AgentStart", "AgentEnd", "Generation", "Message",
    "ToolCall", "Claim", "Artifact", "Verdict", "Consolidation", "Suppression", "Request",
    "EVENT_TYPES", "EventType", "VerdictStatus", "VerdictSource", "ClaimKind",
    "ClaimStatus",
]
