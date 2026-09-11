# Adapters

Every adapter is a thin shim onto the raw SDK (`Swarmscope.emit` and the
`agent_scope` context). What each one can and cannot tell you about lineage:

## OpenAI SDK (incl. Agents SDK, Assistants) — `adapters/openai_transport.py`

Wraps the `httpx` transport, not the client surface. One implementation covers
Chat Completions, Responses, Assistants, embeddings and anything the Agents
SDK does over the same client. Attribution comes from the enclosing
`@sdk.agent` / `agent_scope`; outside one, generations carry `agent_id=unknown`.

- Streaming: usage is read from the final SSE chunk. Pass
  `stream_options={"include_usage": True}`; otherwise the generation is
  recorded with zero tokens and `attrs.stream_usage_missing=True`.
- Retries: each HTTP attempt is a generation (a failed attempt is recorded
  with `attrs.error=True`); that is the real spend.
- `instrument_openai(sdk)` patches constructors for future clients;
  `instrument_openai(sdk, client)` wraps one.

## LangChain / LangGraph — `adapters/langchain.py`

`SwarmscopeCallbackHandler(BaseCallbackHandler)`.

- Generations from `usage_metadata` (preferred) or `llm_output.token_usage`.
- Tools from `on_tool_*`.
- **Agents**: LangGraph node executions (`metadata.langgraph_node`) become
  agents, and each node's output is recorded as a message
  (`kind=langgraph_channel_write`). In bare LangChain only chains whose class
  or name contains "Agent" are treated as agents. There is no message
  causality between LangChain agents — the framework does not expose it — so
  waste analysis on bare LangChain sees only the spawn tree.

## CrewAI — `adapters/crewai.py`

Event-bus subscriber (`crewai_event_bus.on(...)`) plus `step_callback`.
Crew → run, Task → group, Agent → agent. Event class names have changed across
CrewAI versions; `instrument_crewai` binds what it finds and lists the rest in
`binding.missing`. Check it: a silently empty run is exactly the failure mode
this project exists to prevent.

## Raw — `Swarmscope`

`@sdk.agent`, `@sdk.tool`, `sdk.message`, `sdk.link`, `sdk.headers()` /
`sdk.attach_headers()`. The reference implementation and the only path that
records full message causality.

## Contract tests

`tests/adapters/` run against pinned versions in CI (`adapters-pinned`) and
nightly against each framework's `main` (`adapters-nightly`). Bump the pins
deliberately; treat a red nightly as a heads-up, not a build break.
