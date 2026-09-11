"""Lineage + data-flow graph over a run's events.

Nodes: agents, messages, claims, artifacts. Edges point in the direction of
information flow:

- child agent → parent agent (a sub-agent's output returns to its parent)
- sender agent → message → each recipient agent
- message → message (causal parents), message → claim/artifact (inputs)
- producer agent → artifact / claim; input artifacts/claims → artifact

Waste analysis walks these edges *backwards* from accepted artifacts.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core.events import (UNKNOWN, AgentEnd, AgentStart, Artifact, Claim, Event, Generation, Message,
                           ToolCall, Verdict)


@dataclass(slots=True)
class AgentNode:
    agent_id: str
    parent_id: str | None = None
    group_id: str | None = None
    role: str | None = None
    name: str | None = None
    model: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    status: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_generations: int = 0
    generations: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    children: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LineageGraph:
    def __init__(self, events: Iterable[Event]) -> None:
        self.agents: dict[str, AgentNode] = {}
        self.messages: dict[str, Message] = {}
        self.claims: dict[str, Claim] = {}
        self.artifacts: dict[str, Artifact] = {}
        self.verdicts: dict[str, list[Verdict]] = defaultdict(list)
        self.generations: list[Generation] = []
        #: reverse edges: node -> set(nodes that flowed INTO it)
        self._pred: dict[str, set[str]] = defaultdict(set)
        self.event_count = 0
        self.unknown_lineage_events = 0
        self.run_started: float | None = None
        self.run_ended: float | None = None
        for e in events:
            self._add(e)
        for a in self.agents.values():
            if a.parent_id and a.parent_id in self.agents:
                self.agents[a.parent_id].children.append(a.agent_id)

    # ------------------------------------------------------------------
    def _agent(self, aid: str | None) -> AgentNode | None:
        if not aid:
            return None
        node = self.agents.get(aid)
        if node is None:
            node = self.agents[aid] = AgentNode(aid)
        return node

    def _add(self, e: Event) -> None:
        self.event_count += 1
        if e.agent_id == UNKNOWN or e.parent_id == UNKNOWN:
            self.unknown_lineage_events += 1
        self.run_started = e.ts if self.run_started is None else min(self.run_started, e.ts)
        self.run_ended = e.ts if self.run_ended is None else max(self.run_ended, e.ts)

        if isinstance(e, AgentStart):
            n = self._agent(e.agent_id)
            assert n is not None
            n.parent_id, n.group_id, n.role, n.name, n.model = e.parent_id, e.group_id, e.role, e.name, e.model
            n.started_at = e.ts
            if e.parent_id and e.parent_id != UNKNOWN:
                self._pred[e.parent_id].add(e.agent_id)  # child's output flows to parent
        elif isinstance(e, AgentEnd):
            n = self._agent(e.agent_id)
            if n:
                n.ended_at, n.status = e.ts, e.status
        elif isinstance(e, Generation):
            self.generations.append(e)
            n = self._agent(e.agent_id)
            if n:
                n.generations += 1
                n.input_tokens += e.input_tokens
                n.output_tokens += e.output_tokens
                n.cached_input_tokens += e.cached_input_tokens
                if e.cost_usd is None:
                    n.unpriced_generations += 1
                else:
                    n.cost_usd += e.cost_usd
        elif isinstance(e, ToolCall):
            n = self._agent(e.agent_id)
            if n:
                n.tool_calls += 1
                if e.error:
                    n.tool_errors += 1
        elif isinstance(e, Message):
            self.messages[e.event_id] = e
            if e.sender:
                self._pred[e.event_id].add(e.sender)
            for c in e.causes:
                self._pred[e.event_id].add(c)
            for r in e.recipients:
                self._pred[r].add(e.event_id)
                self._agent(r)
        elif isinstance(e, Claim):
            if e.claim_id in self.claims and not e.text:
                self.claims[e.claim_id].status = e.status  # status update
                return
            self.claims[e.claim_id] = e
            if e.agent_id:
                self._pred[e.claim_id].add(e.agent_id)
            for i in e.inputs:
                self._pred[e.claim_id].add(i)
        elif isinstance(e, Artifact):
            self.artifacts[e.artifact_id] = e
            if e.agent_id:
                self._pred[e.artifact_id].add(e.agent_id)
            for i in e.inputs:
                self._pred[e.artifact_id].add(i)
        elif isinstance(e, Verdict):
            self.verdicts[e.artifact_id].append(e)

    # ------------------------------------------------------------------
    @property
    def unknown_lineage_fraction(self) -> float:
        return self.unknown_lineage_events / self.event_count if self.event_count else 0.0

    def roots(self) -> list[AgentNode]:
        return [a for a in self.agents.values() if not a.parent_id or a.parent_id not in self.agents]

    def ancestors(self, start: Iterable[str]) -> set[str]:
        """All nodes (any kind) that information flowed from into ``start``."""
        seen: set[str] = set()
        dq = deque(start)
        while dq:
            n = dq.popleft()
            if n in seen:
                continue
            seen.add(n)
            dq.extend(self._pred.get(n, ()))
        return seen

    def contributing_agents(self, artifact_ids: Iterable[str]) -> set[str]:
        return {n for n in self.ancestors(artifact_ids) if n in self.agents}

    def accepted_artifacts(self, sources: Iterable[str] | None = None, min_confidence: float = 0.0,
                           include_inferred: bool = False) -> dict[str, Verdict]:
        """Latest non-pending verdict per artifact, filtered by provenance."""
        srcs = set(sources) if sources else None
        out: dict[str, Verdict] = {}
        for aid, vs in self.verdicts.items():
            cands = [v for v in vs if v.status != "pending" and (srcs is None or v.source in srcs)
                     and v.confidence >= min_confidence and (include_inferred or not v.inferred)]
            if not cands:
                continue
            latest = max(cands, key=lambda v: v.ts)
            if latest.status == "accepted":
                out[aid] = latest
        return out

    def verdict_sources(self) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for vs in self.verdicts.values():
            for v in vs:
                c[v.source + (":inferred" if v.inferred else "")] += 1
        return dict(c)

    def totals(self) -> dict[str, Any]:
        it = sum(a.input_tokens for a in self.agents.values())
        ot = sum(a.output_tokens for a in self.agents.values())
        return {
            "agents": len(self.agents), "generations": len(self.generations),
            "input_tokens": it, "output_tokens": ot, "tokens": it + ot,
            "cached_input_tokens": sum(a.cached_input_tokens for a in self.agents.values()),
            "cost_usd": sum(a.cost_usd for a in self.agents.values()),
            "unpriced_generations": sum(a.unpriced_generations for a in self.agents.values()),
            "messages": len(self.messages), "claims": len(self.claims), "artifacts": len(self.artifacts),
            "verdicts": sum(len(v) for v in self.verdicts.values()),
            "unknown_lineage_fraction": self.unknown_lineage_fraction,
            "duration_s": (self.run_ended - self.run_started) if self.run_started and self.run_ended else 0.0,
        }
