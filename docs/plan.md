# Swarm Instrumentation SDK — Implementation Plan

**Working name:** `swarmscope` (placeholder)
**Scope:** An open, framework-agnostic Python SDK for (1) observability and cost attribution, (2) semantic dedup and shared claim memory, and (3) marginal-value evaluation of multi-agent runs.
**Targets:** LangChain / LangGraph, OpenAI SDK (incl. Agents SDK), CrewAI, plus a raw decorator API for homegrown loops.
**Status:** Design brief for implementation handoff. Nothing built yet.

---

## 1. The thesis, stated plainly

Teams are starting to run agent counts where nobody can answer three basic questions:

1. Where did the tokens go, and which spend produced output anyone kept?
2. How much of the run was N agents rediscovering the same dead end?
3. Would 500 agents have done as well as 5,000?

These look like three products. They are one product, because all three require the same missing substrate: **a typed event stream of agent activity with stable lineage and an explicit notion of what counted as a result.** Build that once; the three capabilities are queries over it.

The corollary is a hard dependency worth naming up front: **without an acceptance signal, none of this works.** If a user can't tell the system which artifacts were kept, cost attribution degrades to token counting and marginal-value evaluation is impossible. Making `Verdict` cheap to emit is the central adoption problem, not a detail.

### Non-goals

- Not an orchestration framework. We instrument other people's swarms.
- Not a transport or messaging layer. That problem is solved.
- Not a replacement for LangSmith / Langfuse / Phoenix tracing. We emit OTel and integrate.
- Not a hosted service in v1. Local-first, self-hostable, no account required.

---

## 2. Architecture

Five layers. Each is independently useful, which matters because layers 0–1 will have 100× the users of layer 3.

```
┌─────────────────────────────────────────────────┐
│ L4  Surfaces: CLI · local dashboard · OTLP export│
├─────────────────────────────────────────────────┤
│ L3  Evaluation: replay harness · ablation curves │
│     · Shapley estimator · online proxies         │
├─────────────────────────────────────────────────┤
│ L2  Claim store: embeddings · two-stage dedup    │
│     · advisory gate                              │
├─────────────────────────────────────────────────┤
│ L1  Attribution: cost rollup · waste ratio       │
│     · artifact/verdict model                     │
├─────────────────────────────────────────────────┤
│ L0  Core: event schema · lineage · adapters      │
└─────────────────────────────────────────────────┘
```

### L0 — Core schema and adapters

Build on **OpenTelemetry GenAI semantic conventions** rather than inventing a wire format. Extend with swarm-specific attributes; anything OTel already names, we use its name.

Six event types:

| Event | Carries |
|---|---|
| `AgentStart` / `AgentEnd` | agent_id, group_id, parent_id, role, model, config hash |
| `Generation` | prompt/completion tokens, model, latency, cached-token counts, cost |
| `Message` | sender, recipients, payload ref, causal parent message ids |
| `ToolCall` | tool name, args hash, result hash, latency, error |
| `Claim` | see L2 |
| `Artifact` + `Verdict` | see L1 |

**Lineage is the hard part.** None of the three target frameworks give consistent parent/child agent attribution, and message causality is almost never recorded. Approach:

- `contextvars`-based run context propagates `run_id`, `group_id`, `agent_id` automatically through in-process async calls.
- Explicit `sdk.link(cause=msg_id)` for out-of-band handoffs (queues, subprocesses, HTTP).
- When lineage can't be determined, record `parent_id=UNKNOWN` rather than guessing. Report the unknown-lineage fraction on every dashboard; a run at 40% unknown lineage should not be silently presented as if it were measured.

**Adapter strategy** — differs per framework, deliberately:

- **OpenAI SDK:** wrap at the `httpx` transport layer, not the client surface. Catches the Agents SDK, Assistants, and direct calls with one implementation, survives most SDK refactors.
- **LangChain / LangGraph:** `BaseCallbackHandler` for generations/tools; LangGraph channel writes for message edges. LangGraph gives better lineage than bare LangChain — document the difference honestly.
- **CrewAI:** event bus subscriber plus `step_callback`. Crew/Task/Agent map cleanly onto run/group/agent, so this is the easiest of the three.
- **Raw:** `@sdk.agent` / `@sdk.tool` decorators and a context manager. This should be the reference implementation; adapters are thin shims onto it.

Each adapter ships with **contract tests** run against pinned framework versions in CI, plus a nightly job against `main` of each framework. Adapter drift is the single most likely cause of this project rotting.

**Storage:** pluggable. Default SQLite (+ `sqlite-vec` for L2) so `pip install` and go. Postgres/pgvector and DuckDB backends for scale. Writes are async, batched, and best-effort — instrumentation that can block a swarm is worse than no instrumentation.

**Overhead budget:** <3% wall-clock, <5MB RSS per 1k agents for the in-process buffer. Enforce with a benchmark in CI that fails the build on regression.

### L1 — Cost attribution

Per-call token cost is trivial. The useful quantities are:

- **Rollup along lineage**: cost per agent → group → task → run, with a pricing table that supports cached-input rates, batch discounts, and self-hosted $/GPU-hour.
- **Waste ratio**: fraction of tokens spent on branches whose output never reached an accepted artifact. This requires reachability analysis backwards from accepted artifacts through the message/claim graph. It is the single most quotable number this SDK produces.
- **Cost per accepted artifact**, with a distribution, not a mean.

The `Artifact` / `Verdict` model:

```python
art = sdk.artifact(content=proof_sketch, kind="lemma")
sdk.verdict(art, status="accepted", source="lean_verifier",
            confidence=1.0, evidence={"lean_exit": 0})
```

Verdict sources: `human`, `verifier` (objective oracle), `judge` (LLM), `downstream` (was consumed by an accepted artifact). Never collapse these into one confidence number — a Lean-verified accept and an LLM-judge accept are different claims about the world, and a dashboard that averages them is lying. Store the source, filter by it in every query.

Provide three ways to supply verdicts, in decreasing order of value: verifier hook, CLI review queue (`swarmscope review`), and heuristic downstream-reachability as a fallback with a clear "inferred" label.

### L2 — Claim store and semantic dedup

Agents emit claims; the store answers "has anyone already tried this?"

```python
hit = sdk.claim(
    text="Vorticity stays bounded under assumption A via Grönwall",
    kind="approach",          # hypothesis | approach | result | dead_end | artifact_ref
    status="exploring",
    metadata={"assumes": ["A"], "technique": "gronwall"},
)
if hit.similar and hit.top_score > 0.9:
    ...  # agent decides — advisory, not enforced
```

**Two-stage matching**, because pure cosine similarity is not equivalence:

1. ANN recall over embeddings → top-k candidates (fast, ~10ms local).
2. LLM-judge equivalence on candidates, with verdicts cached by normalized pair-hash. The cache is what makes this affordable: in a 10k-agent swarm the same handful of ideas recur constantly, so judge calls per claim approach zero as the run progresses.

Structured metadata is a pre-filter before stage 1 where available — cheaper and more reliable than embeddings for things like "assumes A" vs "assumes ¬A", which embed almost identically and mean opposite things.

**The gate is advisory by default, and this is a design commitment, not a hedge.** A dedup gate that hard-blocks work suppresses exactly the diversity that makes swarm search worth doing; two agents pursuing "the same" approach often diverge at step 40. So: return the matches and a score, let the caller decide, log every suppression with the claim that caused it, and ship an experiment mode that runs gated and ungated arms so users can measure whether the gate helped on *their* workload. If it turns out to hurt on most workloads, that is a publishable finding and we should publish it.

Read path has a hard latency budget (default 200ms) and **fails open**. Write path is fire-and-forget.

### L3 — Marginal-value evaluation

The headline question: did agent 5,000 pay for itself? Four methods, deliberately spanning cheap/noisy to expensive/defensible.

**(a) Retrospective ablation via replay — the workhorse.** Take a completed run. Sample subsets of size *k* from the contributing agents. Replay only the consolidation/aggregation step over each subset. Score the output. Fit value vs. *k*.

Cost: one consolidation call per sample, not a rerun of the swarm. This is the method that makes the whole feature affordable, and it imposes a requirement back on L0: **consolidation inputs and outputs must be recorded verbatim and be deterministically replayable.** Ship a `@sdk.consolidator` decorator that enforces this. Where the user's aggregation isn't replayable, the SDK must say so and fall back to (d).

**(b) Monte-Carlo Shapley over agents or groups.** Exact Shapley is combinatorially infeasible; use permutation sampling with stratification by group, truncate a permutation once marginal contribution falls below ε, and **always report confidence intervals**. An uncertainty-free Shapley number over 10k agents is a fabrication. Group-level Shapley (tens of groups) is far better conditioned than agent-level and should be the default.

**(c) Prospective dose-response.** Run the same task at k ∈ {10, 50, 200, 1000, …}, fixed seeds where the stack allows, fit a saturation curve, report the knee and marginal cost per additional accepted result. Expensive; positioned as a calibration exercise a team runs once per workload class, not per run.

**(d) Online proxies.** What actually goes on the dashboard: novel-claim rate per agent-hour, dedup hit rate over time, coverage entropy across the hypothesis space, time-to-first-accepted-artifact, waste ratio. Cheap and continuous. Methods (a)–(c) exist to *calibrate* these proxies per workload; the SDK should store the fitted relationship and warn when a run drifts outside the regime where the calibration holds.

**The statistical point that has to survive into the UI.** Swarm search is heavy-tailed: often one agent finds the thing and the other 9,999 are the price of the lottery ticket. Mean-based metrics are actively misleading here. Everything in L3 reports **P(success) vs. k** and the full distribution, never expected value alone. A team that sizes a swarm off a mean will systematically under-provision. If we get one thing right in this project, it should be this.

### L4 — Surfaces

- `swarmscope` CLI: `run`, `inspect`, `cost`, `dedup`, `ablate`, `review`, `export`.
- Local dashboard (FastAPI + a single-page frontend, read-only over the store). Lineage tree, cost sunburst, dedup heatmap over time, value-vs-k curve with CIs.
- OTLP exporter so everything lands in Grafana / Datadog / Langfuse. Users should not have to choose between us and their existing tracing.
- Query API over the claim store for programmatic use.

---

## 3. Milestones

| Phase | Weeks | Deliverable | Done when |
|---|---|---|---|
| M0 Core | 1–3 | Event schema, lineage, SQLite store, raw decorator API, OpenAI transport adapter | A 50-agent toy swarm produces a complete lineage tree; overhead <3% |
| M1 Attribution | 4–7 | LangChain + CrewAI adapters, artifact/verdict model, cost rollup, waste ratio, `review` CLI | Waste ratio computed on a real CrewAI workload; contract tests green on all three frameworks |
| M2 Dedup | 8–12 | Claim store, two-stage matcher, judge cache, advisory gate, gated/ungated experiment mode | Dedup hit rate reported on a ≥500-agent run; judge calls/claim trending to <0.05 |
| M3 Evaluation | 13–18 | Replay harness, ablation curves, group-level Shapley w/ CIs, online proxies | Value-vs-k curve with CIs produced end-to-end on the reference workloads |
| M4 Surfaces | 19–22 | Dashboard, OTLP export, docs, published benchmark results | Someone outside the team instruments their own swarm from docs alone |

Roughly five months for a small team. M0–M1 is the part that has standalone value and should be released on its own; don't hold it for M3.

---

## 4. Validation

Pick three reference workloads spanning the acceptance-signal spectrum, since the honest finding may be that the SDK's value varies enormously across them:

1. **Objective verifier** — e.g. competitive-programming or formal-proof tasks where correctness is machine-checkable. Best case for the methodology; use it to establish that the ablation curves are trustworthy where ground truth exists.
2. **Semi-objective** — multi-hop research QA with a gold answer set. Judge-based verdicts, measurable judge error.
3. **Fuzzy** — an open-ended synthesis/report task with human acceptance. The realistic enterprise case, and the one where the methods will look worst. Report it anyway.

Run each at k ∈ {1, 5, 25, 100, 500} and publish the curves. **The published curves are the project's credibility artifact and its best marketing.** If they show that most workloads saturate at k=50, that is a more valuable public contribution than the SDK itself, and should be framed that way rather than buried.

---

## 5. Risks, honestly

- **The market may not exist yet.** Swarms at the scale that motivated this are currently run by maybe a dozen labs. Mitigation is a design constraint, not a marketing line: the SDK must be genuinely useful at 5 agents, where the waste-ratio and cost-attribution features stand on their own. If it only pays off at 1,000, it ships to nobody.
- **Acceptance signals are the adoption bottleneck.** Everything past L1 assumes users will define verdicts. Many won't. Invest disproportionately in the verifier hook and the review CLI, and degrade gracefully and visibly when verdicts are absent.
- **Adapter maintenance is the ongoing tax.** Three fast-moving frameworks. Pinned contract tests plus nightly runs against `main`; expect to spend real engineering time here indefinitely.
- **Replay fidelity.** Nondeterminism (sampling, tool state, time) makes ablation approximate. Quantify it: replay the full set and compare to the original outcome; report that gap as a confidence bound on every curve.
- **The dedup gate may be net-harmful.** Genuinely uncertain. This is why it ships advisory-only with an experiment mode rather than on-by-default.
- **Commoditization from above.** Langfuse, LangSmith, Arize and the framework vendors could absorb L0–L1 as features. L2 and L3 are the defensible part; L0–L1 are table stakes we build to earn the right to them. Plan accordingly — don't over-invest in polishing tracing.

---

## 6. Open questions for whoever picks this up

1. Does the claim store need to be a live coordination substrate (agents read/write during the run) or is post-hoc analysis sufficient for v1? Live is much more valuable and much harder. Current lean: post-hoc in M2, live behind a flag in M3.
2. Group-level or agent-level Shapley as the default? Group is better conditioned; agent-level is what people will ask for.
3. Is OTel GenAI mature enough to build the schema on, or do we need an extension namespace we then have to maintain? Needs a week of investigation before M0 locks the schema.
4. Python-only for v1, or TypeScript from the start? LangChain and CrewAI usage skews Python; the OpenAI SDK does not.
