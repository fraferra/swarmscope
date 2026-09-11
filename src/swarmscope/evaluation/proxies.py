"""Online proxies: cheap, continuous, and what actually goes on the dashboard.

Ablation, Shapley and dose-response exist to *calibrate* these per workload.
:class:`Calibration` stores the fitted relationship and warns when a run
drifts outside the regime where the calibration was measured.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..attribution.graph import LineageGraph
from ..attribution.waste import DEFAULT_SOURCES, waste_report
from ..store.base import Store


@dataclass(slots=True)
class ProxyReport:
    run_id: str
    agents: int
    agent_hours: float
    claims: int
    novel_claims: int
    novel_claim_rate_per_agent_hour: float | None
    dedup_hit_rate: float | None
    dedup_hit_rate_over_time: list[dict[str, Any]]
    suppressions: int
    coverage_entropy: float | None  # normalised [0,1] over (kind, technique)
    coverage_clusters: int
    time_to_first_accepted_s: float | None
    waste_ratio: float | None
    waste_defined: bool
    unknown_lineage_fraction: float
    judge_calls_per_claim: float | None = None
    claim_timeouts: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}  # type: ignore[attr-defined]


def online_proxies(store: Store, run_id: str, *, window_s: float = 60.0, similarity_threshold: float = 0.9,
                   sources=DEFAULT_SOURCES, graph: LineageGraph | None = None) -> ProxyReport:
    events = store.events(run_id)
    g = graph or LineageGraph(events)
    agent_hours = sum(((a.ended_at or g.run_ended or 0) - a.started_at) for a in g.agents.values()
                      if a.started_at) / 3600.0
    claims = [c for c in g.claims.values() if c.text]
    hits = 0
    novel = 0
    timeouts = 0
    judge_calls = 0
    buckets: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    t0 = g.run_started or 0.0
    for c in claims:
        d = c.dedup or {}
        if d.get("skipped"):
            continue
        timeouts += 1 if d.get("timed_out") else 0
        judge_calls += int(d.get("judge_calls", 0))
        is_hit = bool(d.get("suggest_skip")) or float(d.get("top_score", 0.0)) >= similarity_threshold
        b = buckets[int((c.ts - t0) // window_s)]
        b[0] += 1
        if is_hit:
            hits += 1
            b[1] += 1
        else:
            novel += 1
    looked_up = sum(b[0] for b in buckets.values())
    over_time = [{"t_start_s": k * window_s, "claims": v[0], "hits": v[1],
                  "hit_rate": v[1] / v[0] if v[0] else None} for k, v in sorted(buckets.items())]
    suppressions = sum(1 for e in events if e.type == "suppression")

    keys = Counter((c.kind, str((c.metadata or {}).get("technique", ""))) for c in claims)
    entropy = None
    if len(keys) > 1:
        n = sum(keys.values())
        h = -sum((v / n) * math.log(v / n) for v in keys.values())
        entropy = h / math.log(len(keys))
    elif keys:
        entropy = 0.0

    accepted = g.accepted_artifacts(sources)
    ttfa = None
    if accepted and g.run_started is not None:
        ttfa = min(v.ts for v in accepted.values()) - g.run_started
    wr = waste_report(store, run_id, sources=sources, graph=g)

    rep = ProxyReport(run_id, len(g.agents), agent_hours, len(claims), novel,
                      (novel / agent_hours) if agent_hours > 0 else None,
                      (hits / looked_up) if looked_up else None, over_time, suppressions, entropy, len(keys), ttfa,
                      wr.waste_ratio, wr.defined, g.unknown_lineage_fraction,
                      (judge_calls / looked_up) if looked_up else None, timeouts)
    if not wr.defined:
        rep.warnings.append(wr.reason or "waste ratio undefined")
    if g.unknown_lineage_fraction > 0.2:
        rep.warnings.append(f"{g.unknown_lineage_fraction:.0%} of events have unknown lineage; attribution is partial")
    return rep


@dataclass(slots=True)
class Calibration:
    """Linear fit measured_value ≈ a·proxy + b, with the proxy range it was fit on."""

    workload: str
    proxy: str
    a: float
    b: float
    resid_std: float
    x_min: float
    x_max: float
    n: int

    @classmethod
    def fit(cls, workload: str, proxy: str, xs: list[float], ys: list[float]) -> "Calibration":
        n = len(xs)
        if n < 2 or len(ys) != n:
            raise ValueError("need at least two (proxy, value) pairs")
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        b = my - a * mx
        resid = [y - (a * x + b) for x, y in zip(xs, ys)]
        std = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
        return cls(workload, proxy, a, b, std, min(xs), max(xs), n)

    def predict(self, x: float) -> tuple[float, str | None]:
        warn = None
        if not (self.x_min <= x <= self.x_max):
            warn = (f"proxy {self.proxy}={x:.4g} is outside the calibrated range "
                    f"[{self.x_min:.4g}, {self.x_max:.4g}] for workload {self.workload!r}")
        return self.a * x + self.b, warn

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}  # type: ignore[attr-defined]

    def save(self, store: Store) -> None:
        existing = store.calibration_get(self.workload) or {}
        existing[self.proxy] = self.to_dict()
        store.calibration_put(self.workload, existing)

    @classmethod
    def load(cls, store: Store, workload: str, proxy: str) -> "Calibration | None":
        d = (store.calibration_get(workload) or {}).get(proxy)
        return cls(**d) if d else None
