"""swarmscope — instrumentation for multi-agent swarms.

Quick start::

    import swarmscope as ss
    sdk = ss.init("sqlite:///swarm.db")

    @sdk.agent(role="worker")
    def worker(task):
        sdk.generation(model="gpt-4o-mini", input_tokens=120, output_tokens=40)
        hit = sdk.claim("try approach X", kind="approach")
        art = sdk.artifact({"answer": 42}, kind="answer")
        sdk.verdict(art, status="accepted", source="verifier")

    with sdk.run("demo"):
        worker(1)
"""
from __future__ import annotations

from ._version import __version__
from .attribution import (DEFAULT_PRICING, CostReport, LineageGraph, PricingTable, Rate, SelfHosted, WasteReport,
                          cost_rollup, infer_downstream_verdicts, waste_report)
from .claims import (CallableEmbedder, CallableJudge, ClaimHit, GatePolicy, HashEmbedder, Match, OpenAIEmbedder,
                     OpenAIJudge)
from .core import (UNKNOWN, ArtifactRef, Contribution, ExperimentConfig, RunHandle, Swarmscope, context,
                   contributions_from)
from .core.events import (AgentEnd, AgentStart, Artifact, Claim, Consolidation, Event, Generation, Message,
                          Suppression, ToolCall, Verdict)
from .evaluation import (AblationCurve, Calibration, ProxyReport, ReplayHarness, ShapleyResult, ablate,
                         online_proxies, shapley)
from .store import MemoryStore, SQLiteStore, Store, open_store

_global: Swarmscope | None = None


def init(store: str | Store | None = None, **kw) -> Swarmscope:
    """Create (and register as global) an SDK instance. See :class:`Swarmscope`."""
    global _global
    if _global is not None:
        try:
            _global.close()
        except Exception:
            pass
    _global = Swarmscope(store, **kw)
    return _global


def get() -> Swarmscope:
    """The global SDK, creating a default SQLite-backed one if needed."""
    global _global
    if _global is None:
        _global = Swarmscope()
    return _global


# Module-level conveniences delegating to the global SDK.
def run(*a, **kw):
    return get().run(*a, **kw)


def agent(*a, **kw):
    return get().agent(*a, **kw)


def tool(*a, **kw):
    return get().tool(*a, **kw)


def claim(*a, **kw):
    return get().claim(*a, **kw)


def artifact(*a, **kw):
    return get().artifact(*a, **kw)


def verdict(*a, **kw):
    return get().verdict(*a, **kw)


def message(*a, **kw):
    return get().message(*a, **kw)


def link(*a, **kw):
    return get().link(*a, **kw)


def generation(*a, **kw):
    return get().generation(*a, **kw)


def consolidator(*a, **kw):
    return get().consolidator(*a, **kw)


__all__ = [
    "__version__", "init", "get", "Swarmscope", "run", "agent", "tool", "claim", "artifact", "verdict", "message",
    "link", "generation", "consolidator", "UNKNOWN", "ArtifactRef", "Contribution", "ExperimentConfig",
    "RunHandle", "context", "contributions_from", "Event", "AgentStart", "AgentEnd", "Generation", "Message",
    "ToolCall", "Claim", "Artifact", "Verdict", "Consolidation", "Suppression", "Store", "MemoryStore",
    "SQLiteStore", "open_store", "PricingTable", "Rate", "SelfHosted", "DEFAULT_PRICING", "LineageGraph",
    "CostReport", "WasteReport", "cost_rollup", "waste_report", "infer_downstream_verdicts", "ClaimHit", "Match",
    "GatePolicy", "HashEmbedder", "CallableEmbedder", "OpenAIEmbedder", "CallableJudge", "OpenAIJudge",
    "ReplayHarness", "AblationCurve", "ablate", "ShapleyResult", "shapley", "ProxyReport", "online_proxies",
    "Calibration",
]
