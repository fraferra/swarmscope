"""Monte-Carlo Shapley over agents or groups, with confidence intervals.

Exact Shapley is combinatorially infeasible. We use permutation sampling,
stratified by group (each permutation interleaves groups so no group is
systematically early or late), truncate a permutation once the running
value is within ``epsilon`` of the grand value for ``patience`` consecutive
steps (remaining players get marginal 0, as in TMC-Shapley), and always
report a CI. Group-level is the default: tens of groups are far better conditioned
than thousands of agents.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .replay import ReplayHarness, Scorer, as_score


@dataclass(slots=True)
class ShapleyValue:
    unit: str
    mean: float
    ci_low: float
    ci_high: float
    n: int
    std: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}  # type: ignore[attr-defined]


@dataclass(slots=True)
class ShapleyResult:
    run_id: str
    level: str
    permutations: int
    values: dict[str, ShapleyValue]
    grand_value: float
    truncated_fraction: float
    replay_calls: int
    notes: list[str] = field(default_factory=list)

    def ranked(self) -> list[ShapleyValue]:
        return sorted(self.values.values(), key=lambda v: v.mean, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "level": self.level, "permutations": self.permutations,
                "grand_value": self.grand_value, "truncated_fraction": self.truncated_fraction,
                "replay_calls": self.replay_calls, "notes": self.notes,
                "values": [v.to_dict() for v in self.ranked()]}


def _stratified_permutation(rng: random.Random, groups: dict[str, list[str]]) -> list[str]:
    queues = {g: rng.sample(m, len(m)) for g, m in groups.items()}
    order = list(queues)
    out: list[str] = []
    while queues:
        rng.shuffle(order)
        for g in list(order):
            q = queues.get(g)
            if not q:
                queues.pop(g, None)
                order.remove(g)
                continue
            out.append(q.pop())
    return out


def shapley(
    harness: ReplayHarness,
    scorer: Scorer,
    *,
    level: str = "group",
    permutations: int = 100,
    epsilon: float = 1e-3,
    patience: int = 1,
    seed: int = 0,
    z: float = 1.96,
) -> ShapleyResult:
    rng = random.Random(seed)
    if level == "group":
        units = harness.group_ids
        groups = {"all": list(units)}  # stratification is moot at group level
        members = defaultdict(list)
        for a, g in harness.group_of.items():
            members[g].append(a)

        def value(sel: set[str]) -> float:
            agents = [a for g in sel for a in members[g]]
            return as_score(scorer(harness.replay(agents)))
    else:
        units = harness.agent_ids
        groups = defaultdict(list)
        for a, g in harness.group_of.items():
            groups[g].append(a)

        def value(sel: set[str]) -> float:
            return as_score(scorer(harness.replay(sel)))

    res = ShapleyResult(harness.run_id, level, permutations, {}, 0.0, 0.0, 0)
    if not harness.replayable:
        res.notes.append(f"consolidation not replayable ({harness.reason})")
        return res
    empty = value(set())
    grand = value(set(units))
    res.grand_value = grand
    contrib: dict[str, list[float]] = {u: [] for u in units}
    truncated = 0
    for _ in range(permutations):
        perm = _stratified_permutation(rng, groups)
        prev, sel, near_grand = empty, set(), 0
        for i, u in enumerate(perm):
            if near_grand >= patience:
                for rest in perm[i:]:
                    contrib[rest].append(0.0)
                truncated += 1
                break
            sel.add(u)
            cur = value(sel)
            contrib[u].append(cur - prev)
            prev = cur
            near_grand = near_grand + 1 if abs(grand - cur) < epsilon else 0
    for u, xs in contrib.items():
        n = len(xs)
        mean = sum(xs) / n if n else 0.0
        var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(var)
        half = z * std / math.sqrt(n) if n else 0.0
        res.values[u] = ShapleyValue(u, mean, mean - half, mean + half, n, std)
    res.truncated_fraction = truncated / permutations if permutations else 0.0
    res.replay_calls = harness.calls
    if any(v.ci_high - v.ci_low > 0.5 * max(1e-9, abs(grand - empty)) for v in res.values.values()):
        res.notes.append("some CIs are wide relative to the grand value; increase permutations")
    return res
