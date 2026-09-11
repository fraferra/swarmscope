"""Pricing table.

Rates are USD per 1M tokens. The bundled defaults are a *snapshot* and will
drift; override them with :meth:`PricingTable.set`, a JSON file via
:meth:`PricingTable.load`, or the ``SWARMSCOPE_PRICING`` env var (path).
Self-hosted models use :class:`SelfHosted` ($/GPU-hour ÷ tokens/GPU-hour).

Model lookup is longest-prefix so dated snapshots (``gpt-4o-2024-08-06``)
resolve to their family entry.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Rate:
    input: float
    output: float
    cached_input: float | None = None  # defaults to ``input`` when None
    reasoning: float | None = None  # defaults to ``output``
    batch_multiplier: float = 0.5  # applied when Generation.batch is True

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
             reasoning_tokens: int = 0, batch: bool = False) -> float:
        cached_rate = self.input if self.cached_input is None else self.cached_input
        reasoning_rate = self.output if self.reasoning is None else self.reasoning
        uncached = max(0, input_tokens - cached_input_tokens)
        usd = (uncached * self.input + cached_input_tokens * cached_rate
               + output_tokens * self.output + reasoning_tokens * reasoning_rate) / 1_000_000
        return usd * self.batch_multiplier if batch else usd


@dataclass(frozen=True, slots=True)
class SelfHosted:
    """Cost model for self-hosted inference: amortised GPU time per token."""

    usd_per_gpu_hour: float
    tokens_per_gpu_hour: float  # measured throughput (input+output)

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
             reasoning_tokens: int = 0, batch: bool = False) -> float:
        toks = input_tokens + output_tokens + reasoning_tokens
        return toks / self.tokens_per_gpu_hour * self.usd_per_gpu_hour


# Snapshot defaults. Verify against your provider's current price list.
_DEFAULTS: dict[str, Rate] = {
    "gpt-4o": Rate(2.50, 10.00, 1.25),
    "gpt-4o-mini": Rate(0.15, 0.60, 0.075),
    "gpt-4.1": Rate(2.00, 8.00, 0.50),
    "gpt-4.1-mini": Rate(0.40, 1.60, 0.10),
    "gpt-4.1-nano": Rate(0.10, 0.40, 0.025),
    "gpt-5": Rate(1.25, 10.00, 0.125),
    "gpt-5-mini": Rate(0.25, 2.00, 0.025),
    "o3": Rate(2.00, 8.00, 0.50),
    "o3-mini": Rate(1.10, 4.40, 0.55),
    "o4-mini": Rate(1.10, 4.40, 0.275),
    "claude-opus-4": Rate(15.00, 75.00, 1.50),
    "claude-sonnet-4": Rate(3.00, 15.00, 0.30),
    "claude-haiku-4-5": Rate(1.00, 5.00, 0.10),
    "claude-3-5-haiku": Rate(0.80, 4.00, 0.08),
    "text-embedding-3-small": Rate(0.02, 0.0, 0.02),
    "text-embedding-3-large": Rate(0.13, 0.0, 0.13),
}


class PricingTable:
    def __init__(self, rates: Mapping[str, Rate | SelfHosted] | None = None, *, use_defaults: bool = True) -> None:
        self._rates: dict[str, Rate | SelfHosted] = dict(_DEFAULTS) if use_defaults else {}
        if rates:
            self._rates.update(rates)
        env = os.environ.get("SWARMSCOPE_PRICING")
        if env and os.path.exists(env):
            self.load(env)
        self.unpriced: dict[str, int] = {}
        self._cache: dict[str, Rate | SelfHosted | None] = {}

    def set(self, model: str, rate: Rate | SelfHosted) -> None:
        self._rates[model] = rate
        self._cache.clear()

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        for model, r in data.items():
            if "usd_per_gpu_hour" in r:
                self._rates[model] = SelfHosted(**r)
            else:
                self._rates[model] = Rate(**r)
        self._cache.clear()

    def resolve(self, model: str | None) -> Rate | SelfHosted | None:
        if not model:
            return None
        try:
            return self._cache[model]
        except KeyError:
            pass
        r = self._resolve(model)
        if len(self._cache) < 10_000:
            self._cache[model] = r
        return r

    def _resolve(self, model: str) -> Rate | SelfHosted | None:
        m = model.lower()
        if m in self._rates:
            return self._rates[m]
        # strip provider prefixes like "openai/" or "models/"
        if "/" in m:
            m = m.rsplit("/", 1)[-1]
        best = None
        for key in self._rates:
            if m.startswith(key) and (best is None or len(key) > len(best)):
                best = key
        return self._rates[best] if best else None

    def cost(self, model: str | None, input_tokens: int, output_tokens: int,
             cached_input_tokens: int = 0, reasoning_tokens: int = 0, batch: bool = False) -> float | None:
        rate = self.resolve(model)
        if rate is None:
            key = model or "<none>"
            self.unpriced[key] = self.unpriced.get(key, 0) + 1
            return None
        return rate.cost(input_tokens, output_tokens, cached_input_tokens, reasoning_tokens, batch)

    def to_dict(self) -> dict[str, Any]:
        return {k: v.__dict__ if hasattr(v, "__dict__") else {
            f: getattr(v, f) for f in v.__slots__} for k, v in self._rates.items()}


DEFAULT_PRICING = PricingTable()
