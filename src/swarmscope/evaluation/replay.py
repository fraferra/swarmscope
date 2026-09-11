"""Replay harness over a recorded consolidation step.

Ablation replays *only* the consolidation/aggregation over subsets of the
contributing agents — one consolidation call per sample, not a rerun of the
swarm. This requires the consolidation to have been recorded verbatim via
``@sdk.consolidator`` and to be a deterministic function of its inputs.
When it is not, :attr:`ReplayHarness.replayable` is False and callers must
fall back to online proxies; we never silently produce a curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..core.events import Consolidation
from ..core.ids import stable_hash
from ..core.sdk import Contribution, Swarmscope
from ..store.base import Store

Scorer = Callable[[Any], float | bool]


def as_score(x: float | bool) -> float:
    return 1.0 if x is True else 0.0 if x is False else float(x)


@dataclass(slots=True)
class Fidelity:
    original_hash: str | None
    replay_hash: str
    identical: bool
    original_score: float | None
    replay_score: float | None

    @property
    def gap(self) -> float | None:
        if self.original_score is None or self.replay_score is None:
            return None
        return abs(self.original_score - self.replay_score)


class ReplayHarness:
    def __init__(self, store: Store, run_id: str, consolidator: Callable[..., Any] | None = None, *,
                 sdk: Swarmscope | None = None, name: str | None = None) -> None:
        self.store = store
        self.run_id = run_id
        recs = [e for e in store.events(run_id, types=["consolidation"]) if isinstance(e, Consolidation)]
        if name:
            recs = [r for r in recs if r.name == name]
        if not recs:
            raise LookupError(f"run {run_id} has no recorded consolidation" + (f" named {name!r}" if name else "")
                              + "; wrap your aggregation step with @sdk.consolidator")
        self.record: Consolidation = recs[-1]
        fn = consolidator or (sdk.consolidator_fn(self.record.name) if sdk else None)
        if fn is None:
            raise LookupError(f"no consolidator function for {self.record.name!r}: pass consolidator=... "
                              "or the Swarmscope instance that registered it")
        self.fn = fn
        self.contributions = [Contribution.coerce(x) for x in self.record.inputs]
        self.agent_ids = [c.agent_id for c in self.contributions]
        self.group_of = {c.agent_id: (c.group_id or "<none>") for c in self.contributions}
        self.group_ids = sorted(set(self.group_of.values()))
        self.calls = 0

    @property
    def replayable(self) -> bool:
        return bool(self.record.replayable)

    @property
    def reason(self) -> str | None:
        return self.record.reason

    def replay(self, subset: Iterable[str]) -> Any:
        """Re-run the consolidation over contributions from ``subset`` agent ids."""
        keep = set(subset)
        self.calls += 1
        return self.fn([c for c in self.contributions if c.agent_id in keep])

    def replay_groups(self, groups: Iterable[str]) -> Any:
        gs = set(groups)
        return self.replay(a for a, g in self.group_of.items() if g in gs)

    def fidelity(self, scorer: Scorer | None = None) -> Fidelity:
        """Replay the full set and compare with the recorded original."""
        out = self.replay(self.agent_ids)
        h = stable_hash(out)
        orig_score = as_score(scorer(self.record.output)) if scorer else None
        rep_score = as_score(scorer(out)) if scorer else None
        return Fidelity(self.record.output_hash, h, h == self.record.output_hash, orig_score, rep_score)
