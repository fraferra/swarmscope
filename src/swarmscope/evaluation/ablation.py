"""Retrospective ablation: value vs. k, with confidence intervals.

Swarm search is heavy-tailed. Everything here reports **P(success) vs k** and
the score distribution; the mean is provided but is never the headline.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from ..attribution.graph import LineageGraph
from .replay import Fidelity, ReplayHarness, Scorer, as_score


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def quantiles(xs: list[float], qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    out = {}
    for q in qs:
        pos = q * (len(s) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        out[f"q{int(q * 100)}"] = s[lo] + (s[hi] - s[lo]) * (pos - lo)
    return out


@dataclass(slots=True)
class KPoint:
    k: int
    n_samples: int
    successes: int
    p_success: float
    ci_low: float
    ci_high: float
    score_mean: float
    score_quantiles: dict[str, float]
    cost_usd_mean: float | None = None
    tokens_mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}  # type: ignore[attr-defined]


@dataclass(slots=True)
class SaturationFit:
    """Independent-lottery model: P(success | k) = 1 - (1 - q)^k."""

    q: float  # per-unit success probability
    log_likelihood: float
    #: k at which the model reaches ``target`` of its asymptote (1.0)
    knee_k: int | None

    def predict(self, k: int) -> float:
        return 1 - (1 - self.q) ** k


@dataclass(slots=True)
class AblationCurve:
    run_id: str
    unit: str  # "agent" | "group"
    n_units: int
    points: list[KPoint]
    fidelity: Fidelity | None
    fit: SaturationFit | None = None
    knee_k: int | None = None
    notes: list[str] = field(default_factory=list)
    replayable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "unit": self.unit, "n_units": self.n_units, "replayable": self.replayable,
            "points": [p.to_dict() for p in self.points], "knee_k": self.knee_k, "notes": self.notes,
            "fidelity": None if self.fidelity is None else {
                "identical": self.fidelity.identical, "gap": self.fidelity.gap,
                "original_score": self.fidelity.original_score, "replay_score": self.fidelity.replay_score},
            "fit": None if self.fit is None else {"q": self.fit.q, "knee_k": self.fit.knee_k,
                                                   "log_likelihood": self.fit.log_likelihood},
        }

    def marginal_cost_per_accepted(self) -> list[dict[str, Any]]:
        """Δcost / ΔP(success) between consecutive k. Heavy-tailed friendly."""
        out = []
        for a, b in zip(self.points, self.points[1:]):
            dp = b.p_success - a.p_success
            dc = (b.cost_usd_mean or 0) - (a.cost_usd_mean or 0)
            out.append({"from_k": a.k, "to_k": b.k, "delta_p": dp, "delta_cost_usd": dc,
                        "cost_per_unit_p": (dc / dp) if dp > 1e-9 else None})
        return out


def fit_saturation(ks: list[int], successes: list[int], ns: list[int], target: float = 0.95) -> SaturationFit:
    best_q, best_ll = 0.0, -math.inf
    for i in range(1, 1000):
        q = i / 1000.0
        ll = 0.0
        for k, s, n in zip(ks, successes, ns):
            p = min(1 - 1e-12, max(1e-12, 1 - (1 - q) ** k))
            ll += s * math.log(p) + (n - s) * math.log(1 - p)
        if ll > best_ll:
            best_q, best_ll = q, ll
    knee = None
    if 0 < best_q < 1:
        knee = int(math.ceil(math.log(1 - target) / math.log(1 - best_q)))
    return SaturationFit(best_q, best_ll, knee)


def _subtree_costs(g: LineageGraph, agent_ids: list[str]) -> tuple[dict[str, float], dict[str, int]]:
    cost: dict[str, float] = {}
    toks: dict[str, int] = {}

    def visit(aid: str, seen: set[str]) -> tuple[float, int]:
        if aid in cost:
            return cost[aid], toks[aid]
        if aid in seen or aid not in g.agents:
            return 0.0, 0
        seen.add(aid)
        a = g.agents[aid]
        c, t = a.cost_usd, a.tokens
        for ch in a.children:
            cc, ct = visit(ch, seen)
            c, t = c + cc, t + ct
        cost[aid], toks[aid] = c, t
        return c, t

    for aid in agent_ids:
        visit(aid, set())
    return cost, toks


def default_ks(n: int) -> list[int]:
    ks = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    out = [k for k in ks if k < n]
    if n not in out:
        out.append(n)
    return out


def ablate(
    harness: ReplayHarness,
    scorer: Scorer,
    *,
    ks: list[int] | None = None,
    samples: int = 30,
    seed: int = 0,
    success_threshold: float = 1.0,
    unit: str = "agent",
    graph: LineageGraph | None = None,
    knee_target: float = 0.95,
) -> AblationCurve:
    """Sample subsets of size k, replay the consolidation, score, fit value vs k."""
    rng = random.Random(seed)
    units = harness.agent_ids if unit == "agent" else harness.group_ids
    curve = AblationCurve(harness.run_id, unit, len(units), [], None)
    if not harness.replayable:
        curve.replayable = False
        curve.notes.append(f"consolidation not replayable ({harness.reason}); use online proxies or a "
                           "prospective dose-response run")
        return curve
    curve.fidelity = harness.fidelity(scorer)
    if not curve.fidelity.identical:
        curve.notes.append("replay of the full set differs from the recorded output; treat the curve as "
                           f"approximate (score gap={curve.fidelity.gap})")

    g = graph or LineageGraph(harness.store.events(harness.run_id))
    # A contributing agent's cost includes its sub-agents (subtree rollup).
    cost_of, toks_of = _subtree_costs(g, harness.agent_ids)

    def agents_for(sel: list[str]) -> list[str]:
        if unit == "agent":
            return sel
        gs = set(sel)
        return [a for a, grp in harness.group_of.items() if grp in gs]

    ks = sorted({min(k, len(units)) for k in (ks or default_ks(len(units))) if k > 0})  # clamp + dedupe
    for k in ks:
        scores, costs, toks = [], [], []
        n = 1 if k == len(units) else samples
        for _ in range(n):
            sel = units if k == len(units) else rng.sample(units, k)
            agents = agents_for(sel)
            out = harness.replay(agents)
            scores.append(as_score(scorer(out)))
            costs.append(sum(cost_of.get(a, 0.0) for a in agents))
            toks.append(sum(toks_of.get(a, 0) for a in agents))
        succ = sum(1 for s in scores if s >= success_threshold)
        # The full set is a single deterministic replay: its CI is a point.
        lo, hi = (succ / n, succ / n) if k == len(units) else wilson(succ, n)
        curve.points.append(KPoint(k, n, succ, succ / n, lo, hi, sum(scores) / n, quantiles(scores),
                                   sum(costs) / n if costs else None, sum(toks) / n if toks else None))
    if curve.points:
        curve.fit = fit_saturation([p.k for p in curve.points], [p.successes for p in curve.points],
                                   [p.n_samples for p in curve.points], knee_target)
        pmax = max(p.p_success for p in curve.points)
        for p in curve.points:  # empirical knee: first k whose CI reaches knee_target·pmax
            if p.ci_high >= knee_target * pmax and p.p_success >= knee_target * pmax:
                curve.knee_k = p.k
                break
    return curve
