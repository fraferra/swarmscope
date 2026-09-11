# swarmscope

Open, framework-agnostic instrumentation for multi-agent swarms. Zero core dependencies.

It answers three questions nobody can answer once agent counts get large:

1. **Where did the tokens go, and which spend produced output anyone kept?** — cost rollup along lineage and a *waste ratio* (tokens on branches that never reached an accepted artifact).
2. **How much of the run was N agents rediscovering the same dead end?** — a claim store with two-stage semantic dedup (ANN recall → cached LLM-judge equivalence), advisory by design.
3. **Would 500 agents have done as well as 5,000?** — retrospective ablation by replaying only the consolidation step, group-level Monte-Carlo Shapley with CIs, and cheap online proxies calibrated against them.

All three are queries over one substrate: a typed event stream with stable lineage and an explicit notion of what counted as a result (`Artifact` + `Verdict`).

> **The hard dependency:** without an acceptance signal none of this works. If you never emit a `Verdict`, cost attribution degrades to token counting and the waste ratio is reported as *undefined* — never as zero.

## Install

```bash
pip install swarmscope                # core: sqlite store, raw decorator API, CLI, dashboard
pip install "swarmscope[openai]"      # httpx-transport adapter for the OpenAI SDK / Agents SDK
pip install "swarmscope[langchain]"   # LangChain / LangGraph callback handler
pip install "swarmscope[crewai]"      # CrewAI event-bus adapter
pip install "swarmscope[otlp]"        # OTLP export to Grafana / Datadog / Langfuse
pip install "swarmscope[duckdb]"      # DuckDB store
pip install "swarmscope[postgres]"    # Postgres/pgvector store: the shared store for multi-process swarms
pip install "swarmscope[model2vec]"   # low-latency semantic embedder for dedup and routing
```

Integration guides: [OpenAI SDK / Agents SDK](docs/integrations/openai.md) · [LangChain / LangGraph](docs/integrations/langchain.md) · [CrewAI](docs/integrations/crewai.md) · [storage backends](docs/storage.md) · [all docs](docs/README.md)

## Sixty-second tour

```python
import swarmscope as ss

sdk = ss.init("sqlite:///swarm.db")

@sdk.tool
def lean_check(proof: str) -> bool: ...

@sdk.agent(role="prover", group="search")
def prover(problem):
    sdk.generation(model="gpt-4o-mini", input_tokens=900, output_tokens=200)   # adapters do this for you
    hit = sdk.claim("bound vorticity via Grönwall under assumption A",
                    kind="approach", metadata={"assumes": ["A"], "technique": "gronwall"})
    if hit.suggest_skip:                     # advisory — you decide
        hit.suppress("already explored by " + hit.best.agent_id)
        return None
    sketch = "..."
    art = sdk.artifact(sketch, kind="lemma")
    sdk.verdict(art, status="accepted" if lean_check(sketch) else "rejected",
                source="verifier", evidence={"lean_exit": 0})
    return sketch

@sdk.consolidator
def consolidate(contributions):
    return max((c.value for c in contributions if c.value), default=None)

with sdk.run("proof-search", workload="formal-proof"):
    results = [(f"ag_{i}", prover(p)) for i, p in enumerate(problems)]
    consolidate(results)
```

Then:

```bash
swarmscope inspect          # totals, lineage tree, unknown-lineage fraction
swarmscope cost --by group  # rollup agent → group → run, per-model breakdown
swarmscope waste            # waste ratio + cost-per-accepted-artifact distribution, by verdict source
swarmscope dedup            # dedup hit rate over time, judge calls/claim, experiment arms
swarmscope claims -q "..."  # semantic search over the claim store (add --all-runs for cross-run memory)
swarmscope reputation       # route reputation leaderboard; -q "..." for routing advice
swarmscope review           # CLI verdict queue for artifacts with no verdict
swarmscope ablate --consolidator mymod:consolidate --scorer mymod:score --shapley
swarmscope serve            # local dashboard on :8765
swarmscope export --format otlp --endpoint http://localhost:4318/v1/traces
```

## Architecture

```
L4  Surfaces      CLI · local dashboard · OTLP export
L3  Evaluation    replay harness · ablation curves · Shapley (CIs) · online proxies + calibration
L2  Claim store   embeddings · two-stage dedup · advisory gate · gated/ungated experiment
L1  Attribution   cost rollup · waste ratio · Artifact/Verdict model · review queue
L0  Core          event schema (OTel GenAI names) · contextvars lineage · SQLite store · adapters
```

Each layer is independently useful. L0–L1 stand on their own at five agents.

### L0 — lineage

`run_id / group_id / agent_id` propagate automatically through in-process async calls via `contextvars`. For out-of-band handoffs:

```python
headers = sdk.headers()                       # sender side (queue, HTTP, subprocess)
with sdk.attach_headers(headers): ...         # receiver side
with sdk.link(cause=msg_id, parent=agent_id): ...   # or explicit
```

When lineage cannot be determined it is recorded as `UNKNOWN`, never guessed, and the unknown-lineage fraction is printed on every report. A run at 40% unknown lineage is shown as partially measured, not as measured.

Writes are async, batched and best-effort: the buffer never blocks the swarm; if it overflows, events are dropped and *counted* (`sdk.stats["buffer"]["dropped"]`).

### L1 — verdicts have provenance

```python
sdk.verdict(art, status="accepted", source="verifier", confidence=1.0, evidence={"lean_exit": 0})
```

Sources: `human`, `verifier` (objective oracle), `judge` (LLM), `downstream` (consumed by an accepted artifact; always labelled *inferred*). They are never averaged. `waste_report(..., sources=[...])` filters by source; the default excludes `downstream`.

### L2 — the gate is advisory, and that is a design commitment

`sdk.claim()` returns matches and a score; the caller decides. Every acted-on suppression is logged with the claim that caused it. `ExperimentConfig` runs gated and ungated arms (deterministic assignment by agent id) so you can measure on *your* workload whether the gate helps. If it hurts on most workloads, that is a finding worth publishing.

Read path: 200 ms budget, fails open. Write path: fire-and-forget. Judge verdicts are cached by normalised pair-hash so judge calls per claim trend to zero in a long run.

The store is the coordination substrate: point every process at the same SQLite file or Postgres database and `sdk.claim()` sees what other processes registered moments ago, with no messaging between workers. `sdk.search_claims()` and `swarmscope claims -q` query it without registering anything. See `examples/multiprocess_swarm.py`.

The default embedder is a dependency-free feature-hash — lexical, deterministic, weak. For semantic recall pass `embedder="model2vec"` (~50 µs per text, no torch), `"st:<model>"`, `"openai:<model>"`, or your own. Every embedder is cached in memory and in the store, and every lookup runs under a budget that fails open, so a slow embedder costs recall, never swarm latency. See [docs/embeddings.md](docs/embeddings.md).

### L3 — P(success) vs k, never a mean

Swarm search is heavy-tailed. Every curve reports P(success) with Wilson CIs and score quantiles; means are present but never the headline. Ablation needs a replayable consolidation (`@sdk.consolidator` records inputs/outputs verbatim and can check determinism); when it is not replayable the harness says so and you fall back to online proxies. Replay fidelity (full-set replay vs original) is reported on every curve.

### Reputation: route similar requests to whoever handled them well

```python
with sdk.request("translate this clause to French", kind="translation") as adv:
    agent = pick(adv.recommended_agent) if not adv.cold_start else default_agent   # your call
    art = sdk.artifact(agent(...))
    sdk.verdict(art, status=..., source="verifier")     # updates the route's reputation
```

Every route (the sequence of stable agent identities that produced an artifact) is a bandit arm with a Beta posterior learned from verdicts on similar past requests, plus its global record as a prior. Thompson sampling by default, so untried agents get explored; UCB and greedy available. Advisory, with credible intervals on every score. `swarmscope reputation` shows the leaderboard and `swarmscope serve` has a bandit page (`/reputation`) with the learnt posteriors, their densities, evidence over time, and an "ask the bandit" box. See [docs/reputation.md](docs/reputation.md) and `examples/reputation_routing.py`.

## Adapters

| Framework | Mechanism | Lineage quality |
|---|---|---|
| OpenAI SDK / Agents SDK | `httpx` transport wrapper (`instrument_openai`) | generations attributed to the enclosing `@sdk.agent`; streaming needs `stream_options={"include_usage": True}` |
| LangChain | `BaseCallbackHandler` | chains flagged as agents; no inter-agent message causality |
| LangGraph | same handler; nodes → agents, channel writes → messages | better than bare LangChain |
| CrewAI | event-bus subscriber + `step_callback` (verified on CrewAI 1.x) | Crew/Task/Agent → run/group/agent; binding reports what it could not hook |
| Raw | `@sdk.agent`, `@sdk.tool`, `sdk.message`, `sdk.link` | reference implementation |

Adapter contract tests run against pinned versions in CI and nightly against each framework's `main` (`.github/workflows/ci.yml`). Adapter drift is the most likely way this project rots.

## Overhead

Budget: < 3 % wall-clock and < 5 MB RSS per 1k agents for the in-process buffer. `benchmarks/bench_overhead.py` enforces it and CI fails on regression.

## Validation

`python -m validation.run` runs the three reference workloads (objective verifier, semi-objective judge, fuzzy human) at k ∈ {1, 5, 25, 100, 500}, five independent runs per k, and compares the prospective dose-response with the retrospective replay-ablation curve, reports group Shapley with CIs, waste, and a gated-vs-ungated dedup experiment. Results for the simulated solver are in [docs/results.md](docs/results.md); `--solver openai` runs the same protocol against a real model. The protocol is in [docs/validation.md](docs/validation.md).

## Status

v0.1 — every layer of the plan (`docs/plan.md`) is implemented and exercised end to end: four storage backends with a shared conformance test, three adapters with contract tests against pinned and nightly framework versions, the validation harness, and the local dashboard. Not yet done: real-model validation curves (needs an API key and budget), TypeScript.
