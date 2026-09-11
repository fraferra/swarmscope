"""Bandit arithmetic for route reputation.

Each route is an arm with a Beta posterior over its success probability.
Evidence comes in two flavours:

- **local**: outcomes of *similar* past requests handled by this route,
  weighted by similarity (and verdict-source weight, and optional time decay)
- **global**: all outcomes for this route, used as a prior of strength
  ``prior_strength`` so a route with no local evidence still competes

Policies: ``thompson`` (sample the posterior; exploration for free), ``ucb``
(mean + confidence bonus), ``greedy`` (posterior mean).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RouteScore:
    route: tuple[str, ...]
    route_key: str
    #: posterior mean P(success)
    mean: float
    #: policy score used for ranking (Thompson sample / UCB / mean)
    score: float
    ci_low: float
    ci_high: float
    local_successes: float
    local_failures: float
    global_successes: float
    global_failures: float
    #: number of similar past requests this route handled
    similar_requests: int = 0
    mean_similarity: float = 0.0
    #: mean cost of the route's past requests, when known
    cost_usd_mean: float | None = None

    @property
    def n(self) -> float:
        return self.local_successes + self.local_failures

    @property
    def agent(self) -> str:
        """The final agent identity of the route."""
        return self.route[-1] if self.route else ""

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__slots__}  # type: ignore[attr-defined]
        d["route"] = list(self.route)
        d["n"] = self.n
        return d


def beta_ci(a: float, b: float, z: float = 1.96) -> tuple[float, float]:
    """Normal approximation to the Beta(a, b) credible interval (fine for a+b ≥ ~3;
    clipped to [0, 1] otherwise)."""
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    half = z * math.sqrt(var)
    return max(0.0, mean - half), min(1.0, mean + half)


@dataclass
class BanditPolicy:
    #: "thompson" | "ucb" | "greedy"
    policy: str = "thompson"
    #: how many pseudo-observations the global rate contributes as a prior
    prior_strength: float = 2.0
    #: Beta(1,1) base prior, i.e. an unseen route is a coin flip
    prior_a: float = 1.0
    prior_b: float = 1.0
    #: UCB exploration constant
    ucb_c: float = 1.0
    #: optional exponential decay of evidence by age (seconds); None = no decay
    half_life_s: float | None = None
    #: evidence weight by verdict provenance — never averaged blindly
    source_weights: dict[str, float] = field(default_factory=lambda: {
        "human": 1.0, "verifier": 1.0, "judge": 0.7, "downstream": 0.3})
    #: routes must have at least this much total evidence to be *recommended*
    min_evidence: float = 0.0

    def decay(self, age_s: float) -> float:
        if not self.half_life_s or age_s <= 0:
            return 1.0
        return 0.5 ** (age_s / self.half_life_s)

    def posterior(self, local_s: float, local_f: float, global_s: float, global_f: float) -> tuple[float, float]:
        p_global = (global_s + self.prior_a) / (global_s + global_f + self.prior_a + self.prior_b)
        a = self.prior_a + local_s + self.prior_strength * p_global
        b = self.prior_b + local_f + self.prior_strength * (1.0 - p_global)
        return a, b

    def score(self, a: float, b: float, total_n: float, rng: random.Random) -> float:
        mean = a / (a + b)
        if self.policy == "thompson":
            return rng.betavariate(a, b)
        if self.policy == "ucb":
            n = a + b - self.prior_a - self.prior_b
            if n <= 0:
                return 1.0  # untried arms first
            return mean + self.ucb_c * math.sqrt(2.0 * math.log(max(total_n, 1.0) + 1.0) / n)
        return mean
