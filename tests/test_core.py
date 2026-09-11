import asyncio
import json

import pytest

import swarmscope as ss
from swarmscope.core import context as ctx
from swarmscope.core.events import UNKNOWN, AgentEnd, AgentStart, Event, Generation, ToolCall


def test_event_roundtrip():
    g = Generation(run_id="r", model="gpt-4o", input_tokens=10, output_tokens=5, attrs={"x": 1})
    d = g.to_dict()
    assert d["type"] == "generation"
    back = Event.from_dict(json.loads(json.dumps(d)))
    assert isinstance(back, Generation) and back.model == "gpt-4o" and back.attrs == {"x": 1}


def test_agent_lineage_nested(sdk):
    with sdk.run("t") as run:
        @sdk.agent(role="parent")
        def parent():
            @sdk.agent(role="child")
            def child():
                return ctx.current().agent_id
            return ctx.current().agent_id, child()

        pid, cid = parent()
    sdk.flush()
    evs = sdk.store.events(run.run_id, types=["agent_start"])
    by_id = {e.agent_id: e for e in evs}
    assert by_id[pid].parent_id is None  # root
    assert by_id[cid].parent_id == pid
    ends = sdk.store.events(run.run_id, types=["agent_end"])
    assert {e.status for e in ends} == {"ok"}
    assert all(isinstance(e, AgentEnd) and e.duration_ms is not None for e in ends)


def test_agent_error_status(sdk):
    with sdk.run("t") as run:
        @sdk.agent
        def boom():
            raise ValueError("x")

        with pytest.raises(ValueError):
            boom()
    ends = sdk.store.events(run.run_id, types=["agent_end"])
    assert ends[0].status == "error" and "ValueError" in ends[0].error


async def test_async_agent_and_tool(sdk):
    with sdk.run("t") as run:
        @sdk.tool
        async def fetch(x):
            await asyncio.sleep(0)
            return x * 2

        @sdk.agent(group="g")
        async def worker(x):
            return await fetch(x)

        assert await asyncio.gather(worker(1), worker(2)) == [2, 4]
    starts = sdk.store.events(run.run_id, types=["agent_start"])
    tools = sdk.store.events(run.run_id, types=["tool_call"])
    assert len(starts) == 2 and len(tools) == 2
    # each tool call attributed to its own agent
    assert {t.agent_id for t in tools} == {s.agent_id for s in starts}
    assert all(t.group_id == "g" for t in tools)


def test_generation_outside_agent_is_unknown(sdk):
    with sdk.run("t") as run:
        sdk.generation(model="gpt-4o-mini", input_tokens=100, output_tokens=10)
    ev = sdk.store.events(run.run_id, types=["generation"])[0]
    assert ev.agent_id == UNKNOWN
    assert sdk.stats["unknown_lineage_fraction"] == 1.0
    assert ev.cost_usd is not None and ev.cost_usd > 0


def test_unpriced_model_cost_none(sdk):
    with sdk.run("t"):
        ev = sdk.generation(model="my-secret-model", input_tokens=1, output_tokens=1)
    assert ev.cost_usd is None
    assert sdk.pricing.unpriced["my-secret-model"] == 1


def test_link_and_headers(sdk):
    with sdk.run("t") as run:
        with sdk.agent_scope(role="a") as a:
            mid = sdk.message({"hi": 1}, to=["worker-1"])
            headers = sdk.headers()
        assert headers["x-swarmscope-agent-id"] == a
        # simulate an out-of-band worker
        with ctx.use(ctx.RunContext()):
            with sdk.attach_headers(headers):
                with sdk.link(cause=mid):
                    with sdk.agent_scope(role="b") as b:
                        art = sdk.artifact("x")
    starts = {e.agent_id: e for e in sdk.store.events(run.run_id, types=["agent_start"])}
    assert starts[b].parent_id == a
    arts = sdk.store.events(run.run_id, types=["artifact"])
    assert mid in arts[0].inputs


def test_tool_error_recorded(sdk):
    with sdk.run("t") as run:
        @sdk.tool(name="t1")
        def bad():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            bad()
    tc = sdk.store.events(run.run_id, types=["tool_call"])[0]
    assert isinstance(tc, ToolCall) and tc.tool == "t1" and "RuntimeError" in tc.error


def test_buffer_drops_when_full_and_counts():
    s = ss.Swarmscope("memory://", max_queue=5, flush_interval=10)
    for _ in range(50):
        s.emit(Generation(run_id="r"))
    assert s.buffer.dropped > 0
    assert s.stats["buffer"]["dropped"] == s.buffer.dropped
    s.close()


def test_implicit_run_when_no_run_active(sdk):
    sdk.generation(model="gpt-4o", input_tokens=1, output_tokens=1)
    sdk.flush()
    runs = sdk.store.runs()
    assert runs and runs[0].name == "<implicit>"


def test_sqlite_persistence(tmp_path):
    path = tmp_path / "p.db"
    s = ss.Swarmscope(f"sqlite:///{path}", flush_interval=0.01)
    with s.run("persist") as run:
        with s.agent_scope(role="w"):
            s.generation(model="gpt-4o", input_tokens=5, output_tokens=5)
    s.close()
    s2 = ss.SQLiteStore(str(path))
    assert s2.run(run.run_id).name == "persist"
    evs = s2.events(run.run_id)
    assert [e.type for e in evs] == ["agent_start", "generation", "agent_end"]
    s2.close()


def test_verdict_validation(sdk):
    with sdk.run("t"):
        art = sdk.artifact("a")
        with pytest.raises(ValueError):
            sdk.verdict(art, status="accepted", source="vibes")
        with pytest.raises(ValueError):
            sdk.verdict(art, status="maybe", source="human")


def test_experiment_arm_assignment_deterministic():
    exp = ss.ExperimentConfig(gated_fraction=0.5)
    arms = {a: exp.assign(a) for a in [f"ag_{i}" for i in range(200)]}
    assert set(arms.values()) == {"gated", "ungated"}
    assert all(exp.assign(a) == arm for a, arm in arms.items())
    frac = sum(1 for v in arms.values() if v == "gated") / len(arms)
    assert 0.35 < frac < 0.65
