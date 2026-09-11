"""Waste ratio and cost per accepted artifact.

*Waste ratio* = tokens spent by agents whose output never reached an
accepted artifact ÷ total tokens. Requires an acceptance signal; without
verdicts the ratio is **undefined** and reported as such — never as 0.

Verdict sources are kept separate throughout. ``sources`` filters which
provenance counts as "accepted"; the default excludes ``downstream`` and
inferred verdicts so a Lean-verified accept is never averaged with a guess.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core.events import UNKNOWN, Verdict
from ..store.base import Store
from .graph import LineageGraph

DEFAULT_SOURCES = ("human", "verifier", "judge")


def percentiles(xs: list[float], ps=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0)) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    out = {}
    for p in ps:
        idx = min(len(s) - 1, max(0, round(p * (len(s) - 1))))
        out[f"p{int(p * 100)}"] = s[idx]
    out["mean"] = statistics.fmean(s)
    return out


@dataclass(slots=True)
class WasteReport:
    run_id: str
    defined: bool
    reason: str | None
    sources: tuple[str, ...]
    total_tokens: int = 0
    wasted_tokens: int = 0
    total_cost_usd: float = 0.0
    wasted_cost_usd: float = 0.0
    waste_ratio: float | None = None
    waste_ratio_cost: float | None = None
    accepted_artifacts: int = 0
    contributing_agents: int = 0
    total_agents: int = 0
    unknown_lineage_agents: int = 0
    unknown_lineage_fraction: float = 0.0
    #: cost of the backward-reachable set for each accepted artifact
    cost_per_accepted: dict[str, float] = field(default_factory=dict)
    cost_per_accepted_dist: dict[str, float] = field(default_factory=dict)
    by_source: dict[str, dict[str, Any]] = field(default_factory=dict)
    verdict_sources: dict[str, int] = field(default_factory=dict)
    wasted_agents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__slots__}  # type: ignore[attr-defined]
        d["sources"] = list(self.sources)
        d["wasted_agents"] = self.wasted_agents[:50]
        return d


def waste_report(store: Store, run_id: str, *, sources: Iterable[str] = DEFAULT_SOURCES,
                 min_confidence: float = 0.0, include_inferred: bool = False,
                 graph: LineageGraph | None = None) -> WasteReport:
    g = graph or LineageGraph(store.events(run_id))
    srcs = tuple(sources)
    rep = WasteReport(run_id, True, None, srcs, verdict_sources=g.verdict_sources())
    totals = g.totals()
    rep.total_tokens, rep.total_cost_usd, rep.total_agents = totals["tokens"], totals["cost_usd"], len(g.agents)
    unknown = [a for a in g.agents.values() if a.agent_id == UNKNOWN or a.parent_id == UNKNOWN]
    rep.unknown_lineage_agents = len(unknown)
    rep.unknown_lineage_fraction = g.unknown_lineage_fraction

    if not g.verdicts:
        rep.defined, rep.reason = False, "no verdicts recorded: waste ratio is undefined without an acceptance signal"
        return rep
    accepted = g.accepted_artifacts(srcs, min_confidence, include_inferred)
    rep.accepted_artifacts = len(accepted)
    if not accepted:
        rep.defined = False
        rep.reason = f"no accepted artifacts from sources {list(srcs)} (have: {rep.verdict_sources})"
        return rep

    contributing = g.contributing_agents(accepted.keys())
    # Agents with unknown lineage cannot be proven wasted; count them separately, not as waste.
    unknown_ids = {a.agent_id for a in unknown}
    wasted = [a for a in g.agents.values() if a.agent_id not in contributing and a.agent_id not in unknown_ids]
    rep.contributing_agents = len(contributing)
    rep.wasted_tokens = sum(a.tokens for a in wasted)
    rep.wasted_cost_usd = sum(a.cost_usd for a in wasted)
    rep.wasted_agents = [a.agent_id for a in wasted]
    measurable_tokens = rep.total_tokens - sum(g.agents[a].tokens for a in unknown_ids if a in g.agents)
    measurable_cost = rep.total_cost_usd - sum(g.agents[a].cost_usd for a in unknown_ids if a in g.agents)
    rep.waste_ratio = (rep.wasted_tokens / measurable_tokens) if measurable_tokens else None
    rep.waste_ratio_cost = (rep.wasted_cost_usd / measurable_cost) if measurable_cost else None

    for art_id in accepted:
        agents = g.contributing_agents([art_id])
        rep.cost_per_accepted[art_id] = sum(g.agents[a].cost_usd for a in agents if a in g.agents)
    rep.cost_per_accepted_dist = percentiles(list(rep.cost_per_accepted.values()))

    # per-source breakdown so nobody averages a verifier with a judge
    for src in sorted({v.source for vs in g.verdicts.values() for v in vs}):
        acc = g.accepted_artifacts([src], min_confidence, include_inferred=True)
        contrib = g.contributing_agents(acc.keys()) if acc else set()
        w = sum(a.tokens for a in g.agents.values() if a.agent_id not in contrib and a.agent_id not in unknown_ids)
        rep.by_source[src] = {"accepted_artifacts": len(acc), "contributing_agents": len(contrib),
                              "waste_ratio": (w / measurable_tokens) if measurable_tokens and acc else None}
    return rep


def infer_downstream_verdicts(store: Store, run_id: str, *, sources: Iterable[str] = DEFAULT_SOURCES,
                              graph: LineageGraph | None = None) -> list[Verdict]:
    """Heuristic fallback: artifacts consumed by an accepted artifact get an
    *inferred* ``downstream`` accept. Clearly labelled; excluded by default."""
    g = graph or LineageGraph(store.events(run_id))
    accepted = g.accepted_artifacts(sources)
    if not accepted:
        return []
    upstream = g.ancestors(accepted.keys())
    out = []
    for art_id, art in g.artifacts.items():
        if art_id in accepted or art_id not in upstream:
            continue
        if any(v.source == "downstream" for v in g.verdicts.get(art_id, [])):
            continue
        out.append(Verdict(run_id=run_id, group_id=art.group_id, agent_id=art.agent_id, parent_id=art.parent_id,
                           artifact_id=art_id, status="accepted", source="downstream", confidence=0.5,
                           evidence={"consumed_by": [a for a in accepted if art_id in g.ancestors([a])][:10]},
                           inferred=True))
    if out:
        store.write(out)
    return out
