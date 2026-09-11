# Integrating swarmscope with CrewAI

CrewAI's Crew / Task / Agent map directly onto swarmscope's run / group / agent, which makes this the most natural of the three integrations. The adapter subscribes to CrewAI's event bus and falls back to `step_callback` where the bus is unavailable.

## Install

```bash
pip install "swarmscope[crewai]"
```

## Step 1: bind before building the crew

```python
import swarmscope as ss
from swarmscope.adapters.crewai import instrument_crewai

sdk = ss.init("sqlite:///swarm.db")
binding = instrument_crewai(sdk)
print(binding.stats())     # {'bound': {...}, 'missing': [...]}
```

**Look at `missing`.** CrewAI has renamed its event classes across releases. The adapter binds every event it can find and lists the rest. An empty `bound` dict means the bus was not importable and only `step_callback` will record anything. A silently empty run is exactly the failure this project exists to catch, so check this once at startup.

## Step 2: pass the step callback and run inside `sdk.run`

```python
from crewai import Agent, Crew, Task

researcher = Agent(role="Researcher", goal="...", backstory="...", llm="gpt-4o-mini")
writer = Agent(role="Writer", goal="...", backstory="...", llm="gpt-4o-mini")

research = Task(description="Find ...", agent=researcher, expected_output="notes")
draft = Task(description="Write ...", agent=writer, expected_output="report", context=[research])

crew = Crew(agents=[researcher, writer], tasks=[research, draft],
            step_callback=binding.step_callback)

with sdk.run("weekly-report", workload="fuzzy-synthesis"):
    result = crew.kickoff()
```

`step_callback` is a no-op when the tool events are bound; it exists so tool usage is still recorded on versions where the bus does not emit them.

## What gets recorded

| CrewAI event | swarmscope event |
|---|---|
| `AgentExecutionStartedEvent` / `...CompletedEvent` / `...ErrorEvent` | `AgentStart` / `AgentEnd`, with `role`, `model`, and the task name as `group_id` |
| `LLMCallStartedEvent` / `LLMCallCompletedEvent` / `LLMCallFailedEvent` | `Generation` (tokens from the event's `usage` when CrewAI provides it) |
| `ToolUsageStartedEvent` / `...FinishedEvent` / `...ErrorEvent` | `ToolCall` |
| `step_callback` (fallback) | `ToolCall` |

Each agent execution within a task becomes one swarmscope agent. If the same CrewAI `Agent` runs two tasks, that is two swarmscope agents in two groups, which is what you want for per-task cost.

### Token usage caveat

Whether `LLMCallCompletedEvent` carries usage depends on the CrewAI version and the LLM backend. If your generations show zero tokens, do one of:

- **Instrument the underlying OpenAI client as well.** `instrument_openai(sdk)` at startup captures usage at the transport layer regardless of what CrewAI reports, and the enclosing CrewAI agent scope attributes it. This is the recommended setup for OpenAI-backed crews. Do not worry about double counting: when both fire, the CrewAI-side generation has `attrs["usage_reported"] = False` and zero tokens.
- Read `crew.usage_metrics` after `kickoff()` and emit one aggregate generation per crew. Coarse, but better than nothing.

## Hierarchical crews and delegation

In `Process.hierarchical` the manager agent delegates via tools. Delegated executions arrive as nested agent-started events while the manager's scope is active, so they become children of the manager in the lineage tree and cost rolls up to it. Nothing extra is required.

## Flows

CrewAI Flows call `crew.kickoff()` from flow steps. Wrap each step you care about as an agent so the flow structure appears above the crews:

```python
from crewai.flow.flow import Flow, start, listen

class Pipeline(Flow):
    @start()
    @sdk.agent(role="research-stage")
    def research(self):
        return research_crew.kickoff().raw

    @listen(research)
    @sdk.agent(role="write-stage")
    def write(self, notes):
        return write_crew.kickoff(inputs={"notes": notes}).raw
```

Order matters: put `@sdk.agent` innermost so the scope wraps the actual execution.

## Verdicts and artifacts

Task outputs are the natural artifacts. Record them at the end and attach whatever acceptance signal you have:

```python
with sdk.run("weekly-report"):
    result = crew.kickoff()
    for task_out in result.tasks_output:
        art = sdk.artifact(task_out.raw, kind=task_out.name or "task-output")
    final = sdk.artifact(result.raw, kind="report")
    # objective check if you have one, else queue it for a human:
    #   swarmscope review
```

If a task has a `guardrail` that validates output, that is a verifier; emit `sdk.verdict(art, status=..., source="verifier")` from it.

## Claims: dedup across a large crew

For crews with many parallel agents on the same problem (`async_execution=True` tasks, or many crews under a flow), register the approach each agent takes as a claim from a tool, so agents can see what others already tried:

```python
from crewai.tools import tool

@tool("register_approach")
def register_approach(approach: str) -> str:
    """Record the approach you are about to try; returns similar prior attempts."""
    hit = sdk.claim(approach, kind="approach")
    if not hit.similar:
        return "No prior attempts."
    return "Similar prior attempts:\n" + "\n".join(f"- {m.text} (score {m.score:.2f})" for m in hit.similar[:3])
```

The gate stays advisory: the agent reads the list and decides. Every time it decides to skip, have it call a second tool that invokes `hit.suppress(reason)` so the suppression is logged.

## Verify

```bash
swarmscope inspect      # one agent per task execution, grouped by task name
swarmscope cost --by group
```

`binding.stats()["missing"]` non-empty plus a lineage tree that is only tool calls means the event bus did not bind; upgrade or pin CrewAI, or file the event names you see.

## Version notes

Written against the `crewai.utilities.events` bus (CrewAI ≥ 0.86). Event class names are looked up by name at bind time; see `_EVENT_NAMES` in the adapter source to add aliases for a new release. The contract test runs nightly against CrewAI `main` to catch renames early.
