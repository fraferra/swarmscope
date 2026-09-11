"""CrewAI contract test: binds to the event bus if crewai is installed; otherwise
checks the step_callback fallback and that missing hooks are *reported*."""
import pytest

import swarmscope as ss
from swarmscope.adapters.crewai import CrewAIBinding, instrument_crewai


def test_fallback_without_crewai_reports_missing():
    sdk = ss.Swarmscope("memory://", flush_interval=0.01)
    try:
        b = instrument_crewai(sdk)
        crewai = pytest.importorskip("crewai") if False else None  # noqa: F841
        if b.missing and not b.bound:
            class Step:
                tool, tool_input, result = "search", {"q": 1}, "ok"

            with sdk.run("c") as run:
                b.step_callback(Step())
            assert sdk.store.events(run.run_id, types=["tool_call"])[0].tool == "search"
            assert b.stats()["missing"]
        else:
            assert "agent_start" in b.bound
    finally:
        sdk.close()


def test_synthetic_event_bus_sequence():
    """Drive the binding with duck-typed events (what the bus would send)."""
    sdk = ss.Swarmscope("memory://", flush_interval=0.01)
    b = CrewAIBinding(sdk)

    class Agent:
        id, role, llm = "a1", "researcher", type("L", (), {"model": "gpt-4o"})()

    class Task:
        id, name = "t1", "research"

    class Ev:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    agent, task = Agent(), Task()
    with sdk.run("crew") as run:
        b.on_agent_start(agent, Ev(agent=agent, task=task))
        b.on_llm_start(agent, Ev())
        b.on_llm_end(agent, Ev(model="gpt-4o", usage={"prompt_tokens": 50, "completion_tokens": 5}))
        b.on_tool_start(agent, Ev())
        b.on_tool_end(agent, Ev(tool_name="search", tool_args={"q": 1}, output="ok"))
        b.on_agent_end(agent, Ev(agent=agent, task=task, output="done"))
    starts = sdk.store.events(run.run_id, types=["agent_start"])
    assert starts[0].role == "researcher" and starts[0].group_id == "research" and starts[0].model == "gpt-4o"
    g = sdk.store.events(run.run_id, types=["generation"])[0]
    assert g.agent_id == starts[0].agent_id and g.input_tokens == 50
    t = sdk.store.events(run.run_id, types=["tool_call"])[0]
    assert t.agent_id == starts[0].agent_id and t.tool == "search"
    assert sdk.store.events(run.run_id, types=["agent_end"])[0].status == "ok"
    sdk.close()


def test_real_event_bus_binding():
    """With CrewAI installed: bind to the real bus and emit real event instances through it."""
    crewai = pytest.importorskip("crewai")
    from swarmscope.adapters.crewai import _find_bus, _find_event

    bus = _find_bus()
    assert bus is not None
    sdk = ss.Swarmscope("memory://", flush_interval=0.01)
    try:
        b = instrument_crewai(sdk)
        assert not b.missing, f"unbound CrewAI events: {b.missing}"
        AS, AC = _find_event("AgentExecutionStartedEvent"), _find_event("AgentExecutionCompletedEvent")
        LS, LC = _find_event("LLMCallStartedEvent"), _find_event("LLMCallCompletedEvent")
        TS, TF = _find_event("ToolUsageStartedEvent"), _find_event("ToolUsageFinishedEvent")

        class Agent:
            id, role, llm = "a1", "researcher", type("L", (), {"model": "gpt-4o"})()

        class Task:
            id, name = "t1", "research"

        agent, task = Agent(), Task()
        mk = lambda cls, **kw: cls.model_construct(**kw) if hasattr(cls, "model_construct") else cls(**kw)
        with sdk.run("crew") as run:
            bus.emit(agent, mk(AS, agent=agent, task=task, type="agent_execution_started"))
            bus.emit(agent, mk(LS, type="llm_call_started"))
            bus.emit(agent, mk(LC, type="llm_call_completed", model="gpt-4o",
                               usage={"prompt_tokens": 40, "completion_tokens": 4}))
            bus.emit(agent, mk(TS, type="tool_usage_started", tool_name="search", tool_args={"q": 1}))
            bus.emit(agent, mk(TF, type="tool_usage_finished", tool_name="search", tool_args={"q": 1}, output="ok"))
            bus.emit(agent, mk(AC, agent=agent, task=task, type="agent_execution_completed", output="done"))
            try:
                bus.flush()
            except Exception:
                pass
        sdk.flush()
        starts = sdk.store.events(run.run_id, types=["agent_start"])
        assert len(starts) == 1 and starts[0].role == "researcher" and starts[0].group_id == "research"
        gens = sdk.store.events(run.run_id, types=["generation"])
        assert gens and gens[0].agent_id == starts[0].agent_id and gens[0].input_tokens == 40
        tools = sdk.store.events(run.run_id, types=["tool_call"])
        assert tools and tools[0].tool == "search" and tools[0].agent_id == starts[0].agent_id
        assert sdk.store.events(run.run_id, types=["agent_end"])[0].status == "ok"
    finally:
        sdk.close()
