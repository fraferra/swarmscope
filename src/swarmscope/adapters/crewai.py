"""CrewAI adapter: event-bus subscriber plus ``step_callback``.

Crew → run, Task → group, Agent → agent map cleanly. CrewAI's event bus has
changed names across versions; we subscribe defensively and *report* which
event classes we could bind so adapter drift is visible in ``stats()``.

Usage::

    from swarmscope.adapters.crewai import instrument_crewai
    binding = instrument_crewai(sdk)
    crew = Crew(..., step_callback=binding.step_callback)
    with sdk.run("my-crew"):
        crew.kickoff()
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..core import context as ctx
from ..core.events import UNKNOWN, AgentEnd, AgentStart, Generation, ToolCall
from ..core.ids import new_id, stable_hash
from ..core.sdk import Swarmscope

log = logging.getLogger("swarmscope.adapters.crewai")

_EVENT_NAMES = {
    "agent_start": ("AgentExecutionStartedEvent",),
    "agent_end": ("AgentExecutionCompletedEvent",),
    "agent_error": ("AgentExecutionErrorEvent",),
    "task_start": ("TaskStartedEvent",),
    "task_end": ("TaskCompletedEvent", "TaskFailedEvent"),
    "llm_start": ("LLMCallStartedEvent",),
    "llm_end": ("LLMCallCompletedEvent",),
    "llm_error": ("LLMCallFailedEvent",),
    "tool_start": ("ToolUsageStartedEvent",),
    "tool_end": ("ToolUsageFinishedEvent",),
    "tool_error": ("ToolUsageErrorEvent",),
}


@dataclass
class CrewAIBinding:
    sdk: Swarmscope
    bound: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    _agents: dict[Any, tuple[str, Any, float]] = field(default_factory=dict)
    #: open agents in start order: (key, agent_id, group_id, parent_id). The bus may dispatch
    #: handlers outside the caller's contextvars, so attribution never relies on them alone.
    _open: list[tuple[Any, str, str | None, str | None]] = field(default_factory=list)
    #: recently closed agents, because the 1.x bus may dispatch a tool/LLM handler after the
    #: agent-completed handler for the same execution
    _closed: dict[Any, tuple[str, str | None, str | None]] = field(default_factory=dict)
    _tasks: dict[Any, str] = field(default_factory=dict)
    _llm_t0: dict[Any, float] = field(default_factory=dict)
    _tool_t0: dict[Any, float] = field(default_factory=dict)

    # --- helpers ---------------------------------------------------------------
    def _key(self, obj) -> Any:
        return getattr(obj, "id", None) or id(obj)

    def _owner(self, source, event) -> tuple[str, str | None, str | None]:
        """(agent_id, group_id, parent_id) for a generation/tool event."""
        c = ctx.current()
        if c.agent_id:
            return c.agent_id, c.group_id, c.parent_agent_id
        agent = getattr(event, "agent", None) or getattr(event, "from_agent", None)
        task = getattr(event, "task", None) or getattr(event, "from_task", None)
        for key, aid, gid, parent in reversed(self._open):
            if agent is not None and key[0] == self._key(agent) and (task is None or key[1] == self._key(task)):
                return aid, gid, parent
        for key, aid, gid, parent in reversed(self._open):
            if key[0] == self._key(source):
                return aid, gid, parent
        if self._open:  # single-threaded crews: the most recently started agent owns the call
            _, aid, gid, parent = self._open[-1]
            return aid, gid, parent
        for key, owner in reversed(list(self._closed.items())):  # late dispatch after agent end
            if (agent is not None and key[0] == self._key(agent)) or key[0] == self._key(source):
                return owner
        return UNKNOWN, c.group_id, None

    def _group_for(self, task) -> str | None:
        if task is None:
            return ctx.current().group_id
        k = self._key(task)
        if k not in self._tasks:
            self._tasks[k] = getattr(task, "name", None) or getattr(task, "description", "task")[:40]
        return self._tasks[k]

    # --- agent lifecycle ---------------------------------------------------------
    def on_agent_start(self, source, event) -> None:
        agent = getattr(event, "agent", None) or source
        task = getattr(event, "task", None)
        key = (self._key(agent), self._key(task))
        c = ctx.current()
        aid = new_id("ag_")
        parent = c.agent_id or (UNKNOWN if c.run_id is None else None)
        gid = self._group_for(task)
        self.sdk.emit(AgentStart(run_id=self.sdk._run_id(), group_id=gid, agent_id=aid, parent_id=parent,
                                 role=getattr(agent, "role", None), name=getattr(agent, "role", None),
                                 model=str(getattr(getattr(agent, "llm", None), "model", None) or "") or None,
                                 attrs={"framework": "crewai"}))
        token = ctx.set_context(ctx.RunContext(run_id=self.sdk._run_id(), group_id=gid, agent_id=aid,
                                               parent_agent_id=parent, causes=c.causes, arm=c.arm))
        self._agents[key] = (aid, token, time.perf_counter())
        self._open.append((key, aid, gid, parent))

    def _agent_end(self, source, event, status: str, error: str | None) -> None:
        agent = getattr(event, "agent", None) or source
        task = getattr(event, "task", None)
        key = (self._key(agent), self._key(task))
        entry = self._agents.pop(key, None)
        if not entry:
            return
        aid, token, t0 = entry
        for o in self._open:
            if o[0] == key:
                self._closed[key] = (o[1], o[2], o[3])
        if len(self._closed) > 10_000:
            for k in list(self._closed)[:5_000]:
                del self._closed[k]
        self._open = [o for o in self._open if o[0] != key]
        self.sdk.emit(AgentEnd(run_id=self.sdk._run_id(), group_id=self._group_for(task), agent_id=aid,
                               status=status, error=error, duration_ms=(time.perf_counter() - t0) * 1000,
                               result_hash=stable_hash(getattr(event, "output", None))))
        try:
            ctx.reset_context(token)
        except ValueError:
            pass

    def on_agent_end(self, source, event):
        self._agent_end(source, event, "ok", None)

    def on_agent_error(self, source, event):
        self._agent_end(source, event, "error", repr(getattr(event, "error", None))[:500])

    # --- llm ---------------------------------------------------------------------
    def on_llm_start(self, source, event):
        self._llm_t0[id(source)] = time.perf_counter()

    def on_llm_end(self, source, event):
        t0 = self._llm_t0.pop(id(source), None)
        usage = getattr(event, "usage", None) or getattr(event, "token_usage", None) or {}
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        model = getattr(event, "model", None) or getattr(source, "model", None)
        self._generation(source, event, model=str(model) if model else None,
                         input_tokens=int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
                         output_tokens=int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
                         cached_input_tokens=int(usage.get("cached_prompt_tokens", 0) or 0),
                         latency_ms=((time.perf_counter() - t0) * 1000) if t0 else None,
                         attrs={"usage_reported": bool(usage)})

    def on_llm_error(self, source, event):
        t0 = self._llm_t0.pop(id(source), None)
        self._generation(source, event, model=None, latency_ms=((time.perf_counter() - t0) * 1000) if t0 else None,
                         attrs={"error": repr(getattr(event, "error", None))[:200]})

    def _generation(self, source, event, *, model, input_tokens=0, output_tokens=0, cached_input_tokens=0,
                    latency_ms=None, attrs=None):
        aid, gid, parent = self._owner(source, event)
        self.sdk.emit(Generation(run_id=self.sdk._run_id(), group_id=gid, agent_id=aid, parent_id=parent, model=model,
                                 system="crewai", input_tokens=input_tokens, output_tokens=output_tokens,
                                 cached_input_tokens=cached_input_tokens, latency_ms=latency_ms,
                                 cost_usd=self.sdk.pricing.cost(model, input_tokens, output_tokens, cached_input_tokens),
                                 attrs=attrs or {}))

    # --- tools -------------------------------------------------------------------
    def on_tool_start(self, source, event):
        self._tool_t0[id(source)] = time.perf_counter()

    def _tool_end(self, source, event, error):
        t0 = self._tool_t0.pop(id(source), None)
        aid, gid, parent = self._owner(source, event)
        self.sdk.emit(ToolCall(run_id=self.sdk._run_id(), group_id=gid, agent_id=aid, parent_id=parent,
                               tool=getattr(event, "tool_name", None) or "tool",
                               args_hash=stable_hash(getattr(event, "tool_args", None)),
                               result_hash=None if error else stable_hash(getattr(event, "output", None)),
                               latency_ms=((time.perf_counter() - t0) * 1000) if t0 else None, error=error))

    def on_tool_end(self, source, event):
        self._tool_end(source, event, None)

    def on_tool_error(self, source, event):
        self._tool_end(source, event, repr(getattr(event, "error", None))[:500])

    # --- step_callback (works on every CrewAI version) ------------------------------
    def step_callback(self, step: Any) -> None:
        """Pass as ``Crew(step_callback=binding.step_callback)``. Records tool
        steps when the event bus is unavailable; a no-op otherwise."""
        if self.bound.get("tool_end"):
            return
        tool = getattr(step, "tool", None)
        if tool:
            aid, gid, parent = self._owner(None, step)
            self.sdk.emit(ToolCall(run_id=self.sdk._run_id(), group_id=gid, agent_id=aid, parent_id=parent, tool=str(tool),
                                   args_hash=stable_hash(getattr(step, "tool_input", None)),
                                   result_hash=stable_hash(getattr(step, "result", None))))

    def stats(self) -> dict[str, Any]:
        return {"bound": dict(self.bound), "missing": list(self.missing)}


# Where event classes live, by CrewAI generation. Searched in order.
_EVENT_MODULES = (
    "crewai.events.types.agent_events", "crewai.events.types.llm_events", "crewai.events.types.tool_usage_events",
    "crewai.events.types.task_events", "crewai.events",  # >= 1.0
    "crewai.utilities.events",  # 0.x
)


def _find_bus():
    for modname in ("crewai.events", "crewai.utilities.events"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        bus = getattr(mod, "crewai_event_bus", None)
        if bus is not None:
            return bus
    return None


def _find_event(name: str):
    for modname in _EVENT_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        cls = getattr(mod, name, None)
        if cls is not None:
            return cls
    return None


def instrument_crewai(sdk: Swarmscope) -> CrewAIBinding:
    binding = CrewAIBinding(sdk)
    crewai_event_bus = _find_bus()
    if crewai_event_bus is None:  # pragma: no cover - extra not installed
        log.warning("swarmscope: CrewAI event bus unavailable; only step_callback will record")
        binding.missing = list(_EVENT_NAMES)
        return binding

    handlers = {
        "agent_start": binding.on_agent_start, "agent_end": binding.on_agent_end,
        "agent_error": binding.on_agent_error, "llm_start": binding.on_llm_start, "llm_end": binding.on_llm_end,
        "llm_error": binding.on_llm_error, "tool_start": binding.on_tool_start, "tool_end": binding.on_tool_end,
        "tool_error": binding.on_tool_error,
    }
    for logical, handler in handlers.items():
        cls = None
        for name in _EVENT_NAMES[logical]:
            cls = _find_event(name)
            if cls is not None:
                break
        if cls is None:
            binding.missing.append(logical)
            continue
        crewai_event_bus.on(cls)(handler)
        binding.bound[logical] = cls.__name__
    if binding.missing:
        log.warning("swarmscope: CrewAI adapter could not bind %s (version drift?)", binding.missing)
    return binding
