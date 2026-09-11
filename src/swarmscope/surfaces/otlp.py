"""OTLP export so everything lands in Grafana / Datadog / Langfuse / Phoenix.

Attribute names follow the OpenTelemetry GenAI semantic conventions where
they exist (``gen_ai.request.model``, ``gen_ai.usage.input_tokens`` ...) and
``swarm.*`` for the rest. Requires the ``otlp`` extra.

Two modes:
- live: ``attach_otlp(sdk, endpoint=...)`` registers a per-event exporter
- batch: ``export_run(store, run_id, endpoint=...)`` replays a stored run
"""
from __future__ import annotations

import logging
from typing import Any

from ..core.events import (AgentEnd, AgentStart, Artifact, Claim, Consolidation, Event, Generation, Message,
                           Suppression, ToolCall, Verdict)

log = logging.getLogger("swarmscope.otlp")


def event_attributes(e: Event) -> dict[str, Any]:
    """Flat OTel attribute dict for an event (also used by the JSON export)."""
    a: dict[str, Any] = {
        "swarm.run_id": e.run_id, "swarm.event_id": e.event_id, "swarm.event_type": e.type,
    }
    if e.group_id:
        a["swarm.group_id"] = e.group_id
    if e.agent_id:
        a["gen_ai.agent.id"] = e.agent_id
    if e.parent_id:
        a["swarm.parent_agent_id"] = e.parent_id
    if isinstance(e, AgentStart):
        a.update({"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": e.name or "",
                  "swarm.agent.role": e.role or "", "swarm.agent.config_hash": e.config_hash or ""})
        if e.model:
            a["gen_ai.request.model"] = e.model
    elif isinstance(e, AgentEnd):
        a.update({"swarm.agent.status": e.status, "swarm.agent.duration_ms": e.duration_ms or 0.0})
        if e.error:
            a["error.type"] = e.error
    elif isinstance(e, Generation):
        a.update({"gen_ai.operation.name": "chat", "gen_ai.system": e.system or "", "gen_ai.request.model": e.model or "",
                  "gen_ai.usage.input_tokens": e.input_tokens, "gen_ai.usage.output_tokens": e.output_tokens,
                  "gen_ai.usage.cached_input_tokens": e.cached_input_tokens,
                  "swarm.usage.reasoning_tokens": e.reasoning_tokens, "swarm.cost_usd": e.cost_usd if e.cost_usd is not None else -1.0})
        if e.request_id:
            a["gen_ai.response.id"] = e.request_id
    elif isinstance(e, ToolCall):
        a.update({"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": e.tool or "",
                  "swarm.tool.args_hash": e.args_hash or "", "swarm.tool.latency_ms": e.latency_ms or 0.0})
        if e.error:
            a["error.type"] = e.error
    elif isinstance(e, Message):
        a.update({"swarm.message.sender": e.sender or "", "swarm.message.recipients": list(e.recipients),
                  "swarm.message.causes": list(e.causes), "swarm.message.payload_ref": e.payload_ref or ""})
    elif isinstance(e, Claim):
        a.update({"swarm.claim.id": e.claim_id, "swarm.claim.kind": e.kind, "swarm.claim.status": e.status,
                  "swarm.claim.top_score": float((e.dedup or {}).get("top_score", 0.0)),
                  "swarm.claim.suggest_skip": bool((e.dedup or {}).get("suggest_skip", False))})
        if e.arm:
            a["swarm.experiment.arm"] = e.arm
    elif isinstance(e, Artifact):
        a.update({"swarm.artifact.id": e.artifact_id, "swarm.artifact.kind": e.kind,
                  "swarm.artifact.content_hash": e.content_hash or ""})
    elif isinstance(e, Verdict):
        a.update({"swarm.artifact.id": e.artifact_id, "swarm.verdict.status": e.status, "swarm.verdict.source": e.source,
                  "swarm.verdict.confidence": e.confidence, "swarm.verdict.inferred": e.inferred})
    elif isinstance(e, Consolidation):
        a.update({"swarm.consolidation.name": e.name, "swarm.consolidation.inputs": len(e.inputs),
                  "swarm.consolidation.replayable": e.replayable})
    elif isinstance(e, Suppression):
        a.update({"swarm.claim.id": e.claim_id, "swarm.suppressed_by": e.suppressed_by, "swarm.score": e.score or 0.0})
    for k, v in (e.attrs or {}).items():
        if isinstance(v, (str, int, float, bool)):
            a[f"swarm.attr.{k}"] = v
    return a


class OTLPExporter:
    """Turns the event stream into spans. Agent spans stay open between
    AgentStart and AgentEnd; everything else is a point-in-time span."""

    def __init__(self, endpoint: str | None = None, service_name: str = "swarmscope", *, provider=None,
                 headers: dict[str, str] | None = None, console: bool = False) -> None:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        self._trace = trace
        if provider is None:
            provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            if console:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers)))
        self.provider = provider
        self.tracer = provider.get_tracer("swarmscope")
        self._open: dict[str, Any] = {}  # agent_id -> span
        self._run_spans: dict[str, Any] = {}

    def _parent_ctx(self, e: Event):
        span = self._open.get(e.parent_id or "") if isinstance(e, AgentStart) else self._open.get(e.agent_id or "")
        if span is None:
            span = self._run_spans.get(e.run_id)
        return self._trace.set_span_in_context(span) if span is not None else None

    def __call__(self, e: Event) -> None:
        ns = int(e.ts * 1e9)
        attrs = event_attributes(e)
        if e.run_id not in self._run_spans:
            self._run_spans[e.run_id] = self.tracer.start_span(f"run {e.run_id}", start_time=ns,
                                                               attributes={"swarm.run_id": e.run_id})
        if isinstance(e, AgentStart):
            span = self.tracer.start_span(f"agent {e.name or e.role or e.agent_id}", context=self._parent_ctx(e),
                                          start_time=ns, attributes=attrs)
            self._open[e.agent_id or ""] = span
            return
        if isinstance(e, AgentEnd):
            span = self._open.pop(e.agent_id or "", None)
            if span is not None:
                span.set_attributes(attrs)
                if e.status != "ok":
                    from opentelemetry.trace import Status, StatusCode

                    span.set_status(Status(StatusCode.ERROR, e.error or e.status))
                span.end(end_time=ns)
            return
        start = ns
        if isinstance(e, (Generation, ToolCall)) and e.latency_ms:
            start = ns - int(e.latency_ms * 1e6)
        name = {"generation": f"chat {getattr(e, 'model', '')}", "tool_call": f"execute_tool {getattr(e, 'tool', '')}"}.get(e.type, e.type)
        span = self.tracer.start_span(name, context=self._parent_ctx(e), start_time=start, attributes=attrs)
        span.end(end_time=ns)

    def close(self) -> None:
        for span in list(self._open.values()):
            span.end()
        for span in self._run_spans.values():
            span.end()
        self._open.clear()
        self._run_spans.clear()
        try:
            self.provider.force_flush()
        except Exception:  # pragma: no cover
            pass


def attach_otlp(sdk, endpoint: str | None = None, **kw) -> OTLPExporter:
    exp = OTLPExporter(endpoint, sdk.service_name, **kw)
    sdk.add_exporter(exp)
    return exp


def export_run(store, run_id: str, endpoint: str | None = None, **kw) -> int:
    exp = OTLPExporter(endpoint, **kw)
    n = 0
    for e in store.events(run_id):
        exp(e)
        n += 1
    exp.close()
    return n
