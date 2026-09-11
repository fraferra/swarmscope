"""LangChain / LangGraph adapter.

``SwarmscopeCallbackHandler`` records generations (token usage from
``usage_metadata`` or ``llm_output``), tool calls, and agent boundaries.

Lineage honesty: bare LangChain gives us chain/agent boundaries but no
message causality between agents; LangGraph is better — each node execution
(``metadata["langgraph_node"]``) becomes an agent and channel writes become
messages when ``track_langgraph_messages=True``. Anything the framework does
not tell us is recorded as UNKNOWN, not guessed.

Usage::

    handler = SwarmscopeCallbackHandler(sdk)
    graph.invoke(state, config={"callbacks": [handler]})
"""
from __future__ import annotations

import logging
import time
from typing import Any

from ..core import context as ctx
from ..core.events import UNKNOWN, AgentEnd, AgentStart, Message, ToolCall
from ..core.ids import config_hash, new_id, stable_hash
from ..core.sdk import Swarmscope

log = logging.getLogger("swarmscope.adapters.langchain")

try:
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore
except Exception:  # pragma: no cover - extra not installed
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


def _usage_from_response(response) -> tuple[dict[str, int], str | None]:
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0}
    model = None
    llm_output = getattr(response, "llm_output", None) or {}
    model = llm_output.get("model_name") or llm_output.get("model")
    tu = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if tu:
        usage["input_tokens"] = int(tu.get("prompt_tokens", tu.get("input_tokens", 0)) or 0)
        usage["output_tokens"] = int(tu.get("completion_tokens", tu.get("output_tokens", 0)) or 0)
        d = tu.get("prompt_tokens_details") or {}
        usage["cached_input_tokens"] = int(d.get("cached_tokens", 0) or 0)
    for gens in getattr(response, "generations", []) or []:
        for gen in gens:
            msg = getattr(gen, "message", None)
            um = getattr(msg, "usage_metadata", None) if msg is not None else None
            if um:
                usage["input_tokens"] = int(um.get("input_tokens", 0) or 0)
                usage["output_tokens"] = int(um.get("output_tokens", 0) or 0)
                usage["cached_input_tokens"] = int((um.get("input_token_details") or {}).get("cache_read", 0) or 0)
                usage["reasoning_tokens"] = int((um.get("output_token_details") or {}).get("reasoning", 0) or 0)
            rm = getattr(msg, "response_metadata", None) if msg is not None else None
            if rm and not model:
                model = rm.get("model_name") or rm.get("model")
    return usage, model


class SwarmscopeCallbackHandler(BaseCallbackHandler):
    """Attach to any LangChain runnable. One handler per Swarmscope instance."""

    raise_error = False
    run_inline = True  # keep contextvars propagation deterministic

    def __init__(self, sdk: Swarmscope, *, agent_chains: bool = True, track_langgraph_messages: bool = True) -> None:
        super().__init__()
        self.sdk = sdk
        self.agent_chains = agent_chains
        self.track_langgraph_messages = track_langgraph_messages
        # run_id (LangChain UUID) -> (agent_id, token, t0)
        self._agents: dict[Any, tuple[str, Any, float, str | None]] = {}
        self._llm_t0: dict[Any, float] = {}
        self._tool_t0: dict[Any, tuple[float, str, str]] = {}

    # ---- agents (chains / LangGraph nodes) --------------------------------
    def _is_agent(self, serialized, metadata, kwargs) -> tuple[bool, str | None]:
        node = (metadata or {}).get("langgraph_node")
        if node:
            return True, node
        name = kwargs.get("name") or ((serialized or {}).get("name") if serialized else None)
        ids = (serialized or {}).get("id") or []
        cls = ids[-1] if ids else ""
        if self.agent_chains and (("Agent" in cls) or (name and "agent" in name.lower())):
            return True, name or cls
        return False, None

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        is_agent, name = self._is_agent(serialized, metadata, kwargs)
        if not is_agent:
            return
        c = ctx.current()
        parent = c.agent_id
        if parent is None and parent_run_id is not None and parent_run_id in self._agents:
            parent = self._agents[parent_run_id][0]
        aid = new_id("ag_")
        gid = (metadata or {}).get("swarm_group") or c.group_id
        self.sdk.emit(AgentStart(run_id=self.sdk._run_id(), group_id=gid, agent_id=aid, parent_id=parent, role=name,
                                 name=name, config_hash=config_hash({"tags": tags or []}),
                                 attrs={"framework": "langgraph" if (metadata or {}).get("langgraph_node") else "langchain",
                                        "lc_run_id": str(run_id)}))
        token = ctx.set_context(ctx.RunContext(run_id=self.sdk._run_id(), group_id=gid, agent_id=aid,
                                               parent_agent_id=parent, causes=c.causes, arm=c.arm, baggage=c.baggage))
        self._agents[run_id] = (aid, token, time.perf_counter(), gid)

    def _end_agent(self, run_id, status: str, error: str | None, outputs=None):
        entry = self._agents.pop(run_id, None)
        if not entry:
            return
        aid, token, t0, gid = entry
        if self.track_langgraph_messages and outputs is not None and status == "ok":
            try:
                self.sdk.emit(Message(run_id=self.sdk._run_id(), group_id=gid, agent_id=aid, sender=aid,
                                      recipients=[], payload_ref=stable_hash(outputs),
                                      payload=outputs if self.sdk.retain_content else None,
                                      attrs={"kind": "langgraph_channel_write"}))
            except Exception:  # pragma: no cover
                pass
        self.sdk.emit(AgentEnd(run_id=self.sdk._run_id(), group_id=gid, agent_id=aid, status=status, error=error,
                               duration_ms=(time.perf_counter() - t0) * 1000))
        try:
            ctx.reset_context(token)
        except ValueError:  # context was entered in a different task; best effort
            pass

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._end_agent(run_id, "ok", None, outputs)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end_agent(run_id, "error", repr(error)[:500])

    # ---- generations -------------------------------------------------------
    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._llm_t0[run_id] = time.perf_counter()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._llm_t0[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        t0 = self._llm_t0.pop(run_id, None)
        usage, model = _usage_from_response(response)
        self.sdk.generation(model=model, latency_ms=((time.perf_counter() - t0) * 1000) if t0 else None,
                            system="langchain", **usage)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        t0 = self._llm_t0.pop(run_id, None)
        self.sdk.generation(model=None, latency_ms=((time.perf_counter() - t0) * 1000) if t0 else None,
                            system="langchain", error=repr(error)[:200])

    # ---- tools -------------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._tool_t0[run_id] = (time.perf_counter(), name, stable_hash(inputs if inputs is not None else input_str))

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        t0, name, ah = self._tool_t0.pop(run_id, (None, "tool", None))
        self.sdk.emit(ToolCall(**self.sdk._base(), tool=name, args_hash=ah, result_hash=stable_hash(output),
                               latency_ms=((time.perf_counter() - t0) * 1000) if t0 else None))

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        t0, name, ah = self._tool_t0.pop(run_id, (None, "tool", None))
        self.sdk.emit(ToolCall(**self.sdk._base(), tool=name, args_hash=ah, error=repr(error)[:500],
                               latency_ms=((time.perf_counter() - t0) * 1000) if t0 else None))


def instrument_langchain(sdk: Swarmscope, **kw) -> SwarmscopeCallbackHandler:
    return SwarmscopeCallbackHandler(sdk, **kw)
