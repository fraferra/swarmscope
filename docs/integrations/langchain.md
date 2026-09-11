# Integrating swarmscope with LangChain and LangGraph

One callback handler covers both. LangGraph gives you materially better lineage than bare LangChain, and this guide is explicit about the difference.

## Install

```bash
pip install "swarmscope[langchain]"      # langchain-core; add langgraph yourself
```

## Step 1: create the handler

```python
import swarmscope as ss
from swarmscope.adapters.langchain import SwarmscopeCallbackHandler

sdk = ss.init("sqlite:///swarm.db")
handler = SwarmscopeCallbackHandler(sdk)
```

One handler per `Swarmscope` instance. It is safe to share across runnables and threads.

## Step 2: attach it

Per invocation:

```python
chain.invoke(inputs, config={"callbacks": [handler]})
graph.invoke(state, config={"callbacks": [handler]})
```

Or globally, so every runnable in the process reports:

```python
from langchain_core.globals import set_llm_cache  # unrelated; shown to note the pattern
from langchain_core.callbacks.manager import  ...
```

The recommended global route is LangChain's `callbacks` config on the compiled graph:

```python
app = graph.compile().with_config({"callbacks": [handler]})
```

## What gets recorded

| LangChain event | swarmscope event | Notes |
|---|---|---|
| `on_chat_model_start` / `on_llm_end` | `Generation` | tokens from `AIMessage.usage_metadata` (preferred) or `llm_output.token_usage`; cached and reasoning tokens from `input_token_details` / `output_token_details` |
| `on_llm_error` | `Generation` with `attrs.error` | zero tokens |
| `on_tool_start` / `on_tool_end` / `on_tool_error` | `ToolCall` | args and result are hashed, not stored |
| `on_chain_start` for an agent (see below) | `AgentStart` | plus `AgentEnd` on chain end/error |
| LangGraph node output | `Message` | `attrs.kind = "langgraph_channel_write"`; payload retained if `retain_content=True` |

## LangGraph: nodes are agents

When a runnable executes with `metadata["langgraph_node"]` set (LangGraph sets this for every node), the handler opens an agent scope named after the node. Everything the node does, including model calls and tools, is attributed to it. Nested subgraphs become child agents. The node's output is recorded as a `Message`, which is what lets waste analysis trace an accepted artifact back through the graph.

```python
from langgraph.graph import StateGraph, END

def research(state):        # becomes agent "research"
    return {"notes": llm.invoke(state["q"]).content}

def write(state):           # becomes agent "write"
    out = llm.invoke(state["notes"]).content
    art = sdk.artifact(out, kind="report")
    sdk.verdict(art, status="accepted", source="human")
    return {"report": out}

g = StateGraph(dict)
g.add_node("research", research); g.add_node("write", write)
g.set_entry_point("research"); g.add_edge("research", "write"); g.add_edge("write", END)
app = g.compile()

with sdk.run("report"):
    app.invoke({"q": "..."}, config={"callbacks": [handler]})
```

Groups: set `metadata={"swarm_group": "team-a"}` on the config to assign nodes to a group, or wrap the invocation in `with sdk.group("team-a"):`.

### Fan-out with `Send`

Each `Send` target executes as its own node run and therefore its own agent, so a map-reduce over 200 workers produces 200 agents under the dispatching node. Ablation and Shapley work over these directly if your reduce step is wrapped with `@sdk.consolidator`:

```python
@sdk.consolidator
def reduce(contributions):            # contributions: list of Contribution(agent_id, value)
    return best_of([c.value for c in contributions])

def reduce_node(state):
    return {"answer": reduce([ss.Contribution(w["agent_id"], w["out"]) for w in state["workers"]])}
```

To know each worker's `agent_id` inside the worker node, read it from context:

```python
from swarmscope.core import context as ctx
def worker(state):
    ...
    return {"workers": [{"agent_id": ctx.current().agent_id, "out": out}]}
```

## Bare LangChain: what you get and what you do not

Without LangGraph, the handler treats a chain as an agent only when its class name or run name contains `Agent` (`AgentExecutor`, a `RunnableLambda(name="research-agent")`, ...). Plain chains are not agents; their model calls are attributed to the nearest enclosing agent scope, or `unknown` if there is none.

Bare LangChain does **not** expose which agent sent what to whom, so there is no message causality between agents. The lineage graph has the spawn tree only. Practically:

- cost rollup and per-agent tokens: full fidelity
- waste ratio: computed over the spawn tree, so an agent counts as contributing only if it (or a descendant) produced the accepted artifact or sent an explicit `sdk.message`
- ablation and Shapley: need `@sdk.consolidator`, which is framework-independent

Add causality yourself where it matters:

```python
msg = sdk.message(result, to=["writer"])
with sdk.link(cause=msg):
    writer_chain.invoke(...)
```

Turn agent detection off with `SwarmscopeCallbackHandler(sdk, agent_chains=False)` and drive scopes yourself with `@sdk.agent`.

## Claims from inside nodes

The dedup gate is framework-agnostic:

```python
def explore(state):
    hit = sdk.claim(state["hypothesis"], kind="hypothesis",
                    metadata={"technique": state["technique"]})
    if hit.suggest_skip:
        hit.suppress("already explored")
        return {"skip": True}
    ...
```

## Async and threads

The handler sets `run_inline = True` so callbacks execute on the calling task, which keeps `contextvars` consistent. LangChain's own threaded executors (`RunnableParallel` in sync mode) copy context, so attribution survives them.

## Verify the integration

After one run:

```bash
swarmscope inspect          # every node should appear in the lineage tree; unknown lineage should be ~0%
swarmscope cost --by agent
```

If nodes are missing, the handler was not on the config. If unknown lineage is high, model calls are happening outside nodes (for example in a plain `chain.invoke` before the graph starts).

## Version notes

Tested against `langchain-core` 0.3.x. `usage_metadata` on `AIMessage` exists from 0.2; older versions fall back to `llm_output`, which some providers do not populate. LangGraph metadata key `langgraph_node` has been stable since 0.1.
