# Integrating swarmscope with the OpenAI SDK (and the Agents SDK)

This guide covers the OpenAI Python SDK v1+: Chat Completions, the Responses API, embeddings, Assistants, and anything built on top of the same client, including the OpenAI Agents SDK.

## How the adapter works

swarmscope does **not** monkey-patch `client.chat.completions.create`. It wraps the `httpx` transport underneath the client. Every request that hits `/chat/completions`, `/responses`, `/completions`, `/embeddings` or `/messages` is timed, and the response body's `usage` block becomes a `Generation` event. Because the wire format is the stable contract, the adapter survives SDK refactors and covers every surface the SDK exposes with one implementation.

What the adapter records per call:

| Field | Source |
|---|---|
| `model` | response `model` (the dated snapshot), falling back to the request body |
| `input_tokens`, `output_tokens` | `usage.prompt_tokens/completion_tokens` or `usage.input_tokens/output_tokens` |
| `cached_input_tokens` | `usage.prompt_tokens_details.cached_tokens` or `input_tokens_details.cached_tokens` |
| `reasoning_tokens` | `usage.completion_tokens_details.reasoning_tokens` or `output_tokens_details.reasoning_tokens` |
| `latency_ms` | wall time of the HTTP round trip |
| `cost_usd` | from the pricing table; `None` when the model is unpriced |
| `request_id` | response `id` |

What it does **not** do: capture prompts or completions. Content stays out of the store unless you put it there yourself with `sdk.artifact(...)` or `sdk.message(...)`.

## Install

```bash
pip install "swarmscope[openai]"
```

## Step 1: instrument the client

Two options.

**Patch all clients created from now on** (simplest; do it once at startup, before any `OpenAI()` is constructed):

```python
import swarmscope as ss
from swarmscope.adapters.openai_transport import instrument_openai

sdk = ss.init("sqlite:///swarm.db")
instrument_openai(sdk)

from openai import OpenAI, AsyncOpenAI   # both are covered
client = OpenAI()
```

**Wrap one existing client** (when you do not control construction, or want some clients untraced):

```python
from swarmscope.adapters.openai_transport import wrap_client

client = wrap_client(sdk, OpenAI())
aclient = wrap_client(sdk, AsyncOpenAI())
```

`wrap_client` also accepts a bare `httpx.Client`. Wrapping is idempotent.

## Step 2: give generations an owner

A generation recorded outside any agent scope gets `agent_id="unknown"`. It is still counted in totals, but it cannot be attributed, and the unknown-lineage fraction on every report goes up. Put the calls inside an agent:

```python
@sdk.agent(role="solver", group="math")
def solve(question: str) -> str:
    r = client.chat.completions.create(model="gpt-4o-mini",
                                       messages=[{"role": "user", "content": question}])
    return r.choices[0].message.content
```

Attribution flows through `contextvars`, so this works across `await`, thread pools that copy context (`asyncio.to_thread` does; raw `threading.Thread` does not, see below), and nested agents. Async functions can be decorated the same way.

If you cannot use the decorator, use the context manager:

```python
with sdk.agent_scope(role="solver", group="math") as agent_id:
    ...
```

## Step 3: streaming

Usage for a streamed response only exists in the final SSE chunk, and only if you ask for it:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini", messages=msgs,
    stream=True, stream_options={"include_usage": True},
)
for chunk in stream:
    ...
```

The adapter tees the byte stream and records the generation when the stream is exhausted or closed, whichever comes first. Without `include_usage` the generation is recorded with zero tokens and `attrs["stream_usage_missing"] = True`, so the gap is visible in `swarmscope inspect` rather than silently under-counted.

The Responses API streams usage in its `response.completed` event; that is handled too.

## Step 4: the OpenAI Agents SDK

The Agents SDK uses the same `OpenAI`/`AsyncOpenAI` client, so `instrument_openai(sdk)` captures every model call. What it does not know is which SDK "agent" made the call. Two ways to add that:

**Wrap each `Runner.run` in an agent scope:**

```python
from agents import Agent, Runner

triage = Agent(name="triage", instructions="...")

@sdk.agent(role="triage")
async def run_triage(text: str):
    return await Runner.run(triage, text)
```

**Or use the Agents SDK's lifecycle hooks to open scopes per agent** (covers handoffs):

```python
from agents import RunHooks
from swarmscope.core import context as ctx

class SwarmscopeHooks(RunHooks):
    def __init__(self, sdk):
        self.sdk, self._tokens = sdk, {}

    async def on_agent_start(self, context, agent):
        scope = self.sdk.agent_scope(role=agent.name, name=agent.name)
        agent_id = scope.__enter__()
        self._tokens[id(context)] = scope

    async def on_agent_end(self, context, agent, output):
        scope = self._tokens.pop(id(context), None)
        if scope:
            scope.__exit__(None, None, None)

    async def on_handoff(self, context, from_agent, to_agent):
        self.sdk.message({"handoff": to_agent.name}, to=[to_agent.name])

result = await Runner.run(triage, text, hooks=SwarmscopeHooks(sdk))
```

Handoffs become `Message` events so waste analysis can follow work from the agent that produced the accepted artifact back to the agents that fed it.

## Step 5: verdicts

Without verdicts you get token accounting and nothing else. Wherever you already validate output, emit a verdict:

```python
art = sdk.artifact(answer, kind="answer")
sdk.verdict(art, status="accepted" if check(answer) else "rejected",
            source="verifier", evidence={"check": "unit-tests"})
```

Use `source="judge"` when an LLM decided, `source="human"` for review, `source="verifier"` only for an objective oracle. They are reported separately and never averaged.

## Retries, errors, and what counts as spend

Each HTTP attempt is one generation. A 429 or 5xx that the SDK retries is recorded with `attrs["error"] = True` and `attrs["http_status"]`, with zero tokens. That is deliberate: retries are real latency and, for some providers, real spend. Filter them out with `attrs.error` if you only want successful calls.

## Multiple processes and worker pools

`contextvars` do not cross process boundaries or raw threads. Carry lineage explicitly:

```python
# producer
headers = sdk.headers()                       # {"x-swarmscope-run-id": ..., "x-swarmscope-agent-id": ...}
queue.put({"task": t, "swarm": headers})

# consumer (another process; its own Swarmscope pointed at the same store)
with sdk.attach_headers(job["swarm"]):
    with sdk.agent_scope(role="worker"):
        client.chat.completions.create(...)
```

The consumer's agent gets the producer's agent as `parent_id`. Both processes must write to the same store (SQLite in WAL mode on a shared filesystem, or Postgres).

## Pricing

Bundled rates are a snapshot. Check `sdk.stats["unpriced_models"]` after a run and add anything missing:

```python
from swarmscope import Rate
sdk.pricing.set("gpt-4o-2024-11-20", Rate(input=2.5, output=10.0, cached_input=1.25))
```

Or point `SWARMSCOPE_PRICING` at a JSON file with the same fields per model.

## Uninstrumenting

```python
from swarmscope.adapters.openai_transport import uninstrument_openai
uninstrument_openai()     # restores the original constructors; already-wrapped clients stay wrapped
```

## Troubleshooting

- **`unknown lineage` is high.** Calls are happening outside an agent scope. Wrap the call site, or check that your executor copies context (`concurrent.futures.ThreadPoolExecutor` does not by default; use `contextvars.copy_context().run`).
- **Generations show zero tokens.** Streaming without `stream_options={"include_usage": True}`, or a proxy that strips `usage`.
- **`cost_usd` is `None`.** Unpriced model; see Pricing.
- **Nothing is recorded.** `instrument_openai(sdk)` was called after the client was constructed. Use `wrap_client` on the existing instance.
