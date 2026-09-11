"""Cost rollup along lineage: agent → group → run, plus per-model breakdown."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..store.base import Store
from .graph import AgentNode, LineageGraph


@dataclass(slots=True)
class CostReport:
    run_id: str
    totals: dict[str, Any]
    per_agent: dict[str, dict[str, Any]]
    per_group: dict[str, dict[str, Any]]
    per_model: dict[str, dict[str, Any]]
    #: cost including descendants (subtree rollup), keyed by agent id
    subtree: dict[str, float]
    tree: list[dict[str, Any]]
    unknown_lineage_fraction: float
    unpriced_generations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "totals": self.totals, "per_group": self.per_group,
            "per_model": self.per_model, "unknown_lineage_fraction": self.unknown_lineage_fraction,
            "unpriced_generations": self.unpriced_generations, "tree": self.tree,
            "per_agent": self.per_agent,
        }


def _agent_row(a: AgentNode) -> dict[str, Any]:
    return {
        "agent_id": a.agent_id, "parent_id": a.parent_id, "group_id": a.group_id, "role": a.role,
        "name": a.name, "model": a.model, "status": a.status, "generations": a.generations,
        "input_tokens": a.input_tokens, "output_tokens": a.output_tokens,
        "cached_input_tokens": a.cached_input_tokens, "tokens": a.tokens, "cost_usd": a.cost_usd,
        "unpriced_generations": a.unpriced_generations, "tool_calls": a.tool_calls, "tool_errors": a.tool_errors,
        "duration_ms": ((a.ended_at - a.started_at) * 1000) if a.started_at and a.ended_at else None,
    }


def cost_rollup(store: Store, run_id: str, graph: LineageGraph | None = None) -> CostReport:
    g = graph or LineageGraph(store.events(run_id))
    per_agent = {aid: _agent_row(a) for aid, a in g.agents.items()}

    per_group: dict[str, dict[str, Any]] = defaultdict(lambda: {"agents": 0, "tokens": 0, "cost_usd": 0.0,
                                                                "generations": 0})
    for a in g.agents.values():
        row = per_group[a.group_id or "<none>"]
        row["agents"] += 1
        row["tokens"] += a.tokens
        row["cost_usd"] += a.cost_usd
        row["generations"] += a.generations

    per_model: dict[str, dict[str, Any]] = defaultdict(lambda: {"generations": 0, "input_tokens": 0,
                                                                "output_tokens": 0, "cached_input_tokens": 0,
                                                                "cost_usd": 0.0, "unpriced": 0})
    for gen in g.generations:
        row = per_model[gen.model or "<none>"]
        row["generations"] += 1
        row["input_tokens"] += gen.input_tokens
        row["output_tokens"] += gen.output_tokens
        row["cached_input_tokens"] += gen.cached_input_tokens
        if gen.cost_usd is None:
            row["unpriced"] += 1
        else:
            row["cost_usd"] += gen.cost_usd

    # subtree rollup (post-order over the agent tree)
    subtree: dict[str, float] = {}
    subtree_tokens: dict[str, int] = {}

    def visit(aid: str, seen: set[str]) -> float:
        if aid in subtree:
            return subtree[aid]
        if aid in seen:  # defensive: cycles from bad adapters
            return 0.0
        seen.add(aid)
        a = g.agents[aid]
        total = a.cost_usd + sum(visit(c, seen) for c in a.children)
        subtree[aid] = total
        subtree_tokens[aid] = a.tokens + sum(subtree_tokens.get(c, 0) for c in a.children)
        return total

    for aid in g.agents:
        visit(aid, set())

    def tree_node(a: AgentNode, depth: int = 0) -> dict[str, Any]:
        if depth > 200:
            return {"agent_id": a.agent_id, "truncated": True}
        return {"agent_id": a.agent_id, "name": a.name or a.role or a.agent_id, "group_id": a.group_id,
                "cost_usd": a.cost_usd, "subtree_cost_usd": subtree.get(a.agent_id, a.cost_usd),
                "tokens": a.tokens, "subtree_tokens": subtree_tokens.get(a.agent_id, a.tokens),
                "children": [tree_node(g.agents[c], depth + 1) for c in a.children]}

    tree = [tree_node(r) for r in g.roots()]
    totals = g.totals()
    return CostReport(run_id, totals, per_agent, dict(per_group), dict(per_model), subtree, tree,
                      g.unknown_lineage_fraction, totals["unpriced_generations"])
