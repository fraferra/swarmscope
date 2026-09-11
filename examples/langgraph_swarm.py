"""LangGraph nodes as agents. Needs the ``langchain`` extra (+ langgraph, an LLM provider).

    python examples/langgraph_swarm.py
"""
import swarmscope as ss
from swarmscope.adapters.langchain import SwarmscopeCallbackHandler

sdk = ss.init("sqlite:///swarmscope.db")
handler = SwarmscopeCallbackHandler(sdk)

if __name__ == "__main__":
    from typing import TypedDict

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, StateGraph

    llm = FakeMessagesListChatModel(responses=[
        AIMessage(content="draft", usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}),
        AIMessage(content="final", usage_metadata={"input_tokens": 80, "output_tokens": 12, "total_tokens": 92}),
    ])

    class State(TypedDict, total=False):
        text: str

    def research(state: State) -> State:  # each node becomes an agent (metadata.langgraph_node)
        return {"text": llm.invoke("research").content}

    def write(state: State) -> State:
        out = llm.invoke(state["text"]).content
        art = sdk.artifact(out, kind="report")
        sdk.verdict(art, status="accepted", source="human")
        return {"text": out}

    g = StateGraph(State)
    g.add_node("research", research)
    g.add_node("write", write)
    g.set_entry_point("research")
    g.add_edge("research", "write")
    g.add_edge("write", END)
    app = g.compile()

    with sdk.run("langgraph-demo") as run:
        print(app.invoke({}, config={"callbacks": [handler]}))
    rep = ss.cost_rollup(sdk.store, run.run_id)
    print({k: rep.totals[k] for k in ("agents", "generations", "tokens", "cost_usd", "unknown_lineage_fraction")})
    print("agents:", [(n["name"], n["tokens"]) for n in rep.tree])
    sdk.close()
