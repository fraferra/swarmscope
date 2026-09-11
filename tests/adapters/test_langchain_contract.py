"""Contract tests for the LangChain/LangGraph callback handler using langchain-core fakes."""
import pytest

lc = pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

import swarmscope as ss
from swarmscope.adapters.langchain import SwarmscopeCallbackHandler


@pytest.fixture
def sdk():
    s = ss.Swarmscope("memory://", flush_interval=0.01)
    yield s
    s.close()


def _llm():
    msg = AIMessage(content="hi", usage_metadata={"input_tokens": 11, "output_tokens": 2, "total_tokens": 13},
                    response_metadata={"model_name": "gpt-4o-mini"})
    return FakeMessagesListChatModel(responses=[msg, msg, msg])


def test_llm_usage_recorded(sdk):
    h = SwarmscopeCallbackHandler(sdk)
    with sdk.run("lc") as run:
        with sdk.agent_scope(role="w") as a:
            _llm().invoke("x", config={"callbacks": [h]})
    g = sdk.store.events(run.run_id, types=["generation"])[0]
    assert g.agent_id == a and (g.input_tokens, g.output_tokens) == (11, 2) and g.model == "gpt-4o-mini"
    assert g.cost_usd is not None


def test_tool_calls_recorded(sdk):
    h = SwarmscopeCallbackHandler(sdk)

    @tool
    def add(a: int, b: int) -> int:
        """add"""
        return a + b

    with sdk.run("lc") as run:
        with sdk.agent_scope(role="w") as ag:
            assert add.invoke({"a": 1, "b": 2}, config={"callbacks": [h]}) == 3
    tc = sdk.store.events(run.run_id, types=["tool_call"])[0]
    assert tc.tool == "add" and tc.agent_id == ag and tc.error is None


def test_langgraph_node_metadata_becomes_agent(sdk):
    h = SwarmscopeCallbackHandler(sdk)
    llm = _llm()
    node = RunnableLambda(lambda x: llm.invoke("q").content, name="research")
    with sdk.run("lg") as run:
        out = node.invoke("in", config={"callbacks": [h], "metadata": {"langgraph_node": "research"}})
        assert out == "hi"
    starts = sdk.store.events(run.run_id, types=["agent_start"])
    assert len(starts) == 1 and starts[0].name == "research" and starts[0].attrs["framework"] == "langgraph"
    gens = sdk.store.events(run.run_id, types=["generation"])
    assert gens[0].agent_id == starts[0].agent_id  # generation inside the node is attributed to it
    ends = sdk.store.events(run.run_id, types=["agent_end"])
    assert ends[0].status == "ok"
    msgs = sdk.store.events(run.run_id, types=["message"])
    assert msgs and msgs[0].attrs["kind"] == "langgraph_channel_write"


def test_plain_chain_is_not_an_agent(sdk):
    h = SwarmscopeCallbackHandler(sdk)
    with sdk.run("lc") as run:
        RunnableLambda(lambda x: x, name="format").invoke("in", config={"callbacks": [h]})
    assert not sdk.store.events(run.run_id, types=["agent_start"])
