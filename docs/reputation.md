# Reputation and routing

**Question it answers:** "Requests like this one — who handled them well before, and should I send this one there?"

swarmscope keeps a reputation for every *route*, and `sdk.request()` turns that into routing advice using a contextual multi-armed bandit. Like the dedup gate, it is advisory: you get a ranked list with uncertainty, and you decide.

## Concepts

| Term | Meaning |
|---|---|
| **Identity** | A stable name for *what* an agent is, not which instance ran: `role\|model\|config_hash` by default, or `identity="..."` on `@sdk.agent` / `agent_scope`. |
| **Route** | The sequence of identities from the run root to the agent that produced an artifact, e.g. `coordinator\|gpt-4o\| → prover\|gpt-4o-mini\|`. Propagates through `contextvars` and through lineage headers across processes, so multi-process sequences are captured. |
| **Request** | A unit of work being routed. `sdk.request(text)` embeds it and records a `Request` event. Entering the returned advice as a context manager tags every artifact produced inside with the request id. |
| **Outcome** | A verdict on such an artifact, attributed to the artifact's route, weighted by verdict source (`human` 1.0, `verifier` 1.0, `judge` 0.7 × confidence, `downstream` 0.3). |

## Usage

```python
with sdk.request("translate this clause to French", kind="translation") as adv:
    if adv.cold_start:
        agent = default_agent
    else:
        best = adv.recommended                    # RouteScore
        print(best.route, best.mean, (best.ci_low, best.ci_high), best.similar_requests)
        agent = agents_by_identity[best.agent]    # your decision; adv.by_agent() aggregates by final agent
    out = agent(...)                              # runs inside the request: artifacts link to it
    art = sdk.artifact(out)
    sdk.verdict(art, status=..., source="verifier")   # this updates the route's reputation
```

Every `RouteScore` carries the posterior mean, a 95% credible interval, the local evidence (similar requests), the global evidence (all requests), and the policy score used for ranking. **Read the interval.** A route with mean 0.9 and interval [0.4, 1.0] has been tried once.

`sdk.search_claims` / `swarmscope claims` are the read side for claims; for reputation use:

```bash
swarmscope reputation                      # leaderboard of routes, store-wide
swarmscope reputation --unit agent         # aggregated by final agent identity
swarmscope reputation -q "translate this"  # advice for a query, no request recorded
swarmscope reputation --rebuild            # recompute from the event log (after `swarmscope review`)
```

## The bandit

Each route is an arm with a Beta posterior over P(accepted).

0. **Arms.** A bandit must know its arms. Every identity seen by `@sdk.agent` (at decoration time) or `agent_scope` is registered as a candidate, so an agent that has never run is explored instead of being invisible. Pass `candidates=[...]` (identities, or sequences as `"a|m| > b|m|"`) to restrict or extend the set for one request.
1. **Recall.** The request is embedded and the top-k similar past requests of the same `kind` (cosine ≥ `min_similarity`) are fetched from the store, across all runs by default (`scope="global"`). `kind` is your categorisation; requests of different kinds never inform each other.
2. **Local evidence.** For each route that handled a similar request, successes and failures are summed with weight `similarity × source_weight × decay(age)`.
3. **Global prior.** The route's all-time record contributes `prior_strength` pseudo-observations at its global success rate, so a route with no local history still competes, and the top `global_candidates` routes are always in the candidate set.
4. **Score.** `thompson` (default) samples the posterior, which explores under-tried routes for free; `ucb` adds an explicit confidence bonus; `greedy` ranks by posterior mean.

```python
from swarmscope import RouterPolicy
sdk = ss.init(store, router=RouterPolicy(policy="thompson", unit="sequence", prior_strength=2.0,
                                          half_life_s=7*24*3600, min_similarity=0.3))
```

`unit="agent"` ranks by final agent identity instead of the full sequence; use it when your caller picks an agent, not a pipeline.

## Learning and consistency

- Verdicts issued in-process update reputation immediately (`sdk.stats["reputation"]`).
- A revised verdict on the same artifact replaces the earlier contribution, never double-counts.
- `pending` verdicts carry no evidence.
- Verdicts issued later by another process (review queue, offline verifier) are picked up by `swarmscope reputation --rebuild`, which recomputes outcomes and stats from the event log. Reputation is fully derivable from events, so the tables can always be rebuilt.
- With a shared store (SQLite file or Postgres), every process contributes to and reads the same reputation.

## What it is not

- It does not reroute anything itself. There is no hook that overrides your dispatch; you read the advice.
- It does not know *why* a route succeeded. If two routes differ only by a prompt you did not put in `config`, they share an identity and their outcomes are pooled. Put what matters into `config` (hashed into the identity) or set `identity` explicitly.
- Similarity is only as good as the embedder. The default hash embedder is lexical; for real semantic recall use `OpenAIEmbedder` or `SentenceTransformerEmbedder` (the router shares the claim store's embedder).

`examples/reputation_routing.py` runs a simulated swarm where a Thompson-sampling dispatcher learns which specialist to route each request type to, and prints the acceptance rate per round against a random dispatcher.
