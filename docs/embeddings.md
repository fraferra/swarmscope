# Embedders and latency

Both the claim store (dedup) and the reputation router (routing) embed short texts on the hot path. This page covers how to pick a semantic embedder without paying for it in swarm latency.

## Choosing one

```python
sdk = ss.init(store, embedder="model2vec")                          # recommended semantic default
sdk = ss.init(store, embedder="model2vec:minishlab/potion-base-32M")
sdk = ss.init(store, embedder="st:all-MiniLM-L6-v2")                 # sentence-transformers (torch)
sdk = ss.init(store, embedder="openai:text-embedding-3-small")
sdk = ss.init(store, embedder=MyEmbedder())                          # anything with .name, .dim, .embed(texts)
export SWARMSCOPE_EMBEDDER=model2vec                                 # or via the environment
```

| Spec | Semantic | Latency per text (CPU) | Needs | Notes |
|---|---|---|---|---|
| `hash` (default) | no, lexical | ~30 µs | nothing | deterministic; catches near-verbatim repeats; paraphrases score low (~0.1) |
| `model2vec[:model]` | yes | ~50 µs | `swarmscope[model2vec]` | static embeddings distilled from a sentence transformer; ~30 MB model, loads in seconds; paraphrases ~0.5 |
| `st:<model>` | yes | 5–20 ms | `swarmscope[embeddings]` (torch) | best local quality |
| `openai[:model]` | yes | 50–300 ms | `swarmscope[openai]`, API key | network round trip; cache and budget matter most here |

Measured on this repo's routing example (`examples/reputation_routing.py`), the semantic embedder raises Thompson-routed acceptance from 0.77 to 0.81 against an oracle of 0.88, because evidence from paraphrased requests is no longer diluted.

**Changing embedders changes the vector space.** Vectors are cached and stored under the embedder's name, so nothing gets mixed, but claims and requests indexed under the old embedder are not recalled by the new one. Use a fresh store, or `swarmscope reputation --rebuild` to re-index requests.

## How latency stays low

1. **Cache.** The SDK wraps every embedder in `CachedEmbedder`: an in-memory LRU (50k entries) plus persistence in the store, so a repeated text costs a dictionary lookup and the cache is shared across processes and restarts. Misses are written back on a background thread. `sdk.stats["embedder"]` reports hits, store hits, and misses.
2. **Budgets that fail open.** `sdk.claim()` has a 200 ms read budget (`GatePolicy.latency_budget_ms`) and `sdk.request()` a 250 ms budget (`RouterPolicy.latency_budget_ms`, or `budget_ms=` per call). On expiry you get an empty result flagged `timed_out=True` and your code takes its default path. Neither ever blocks the swarm on an embedder.
3. **Nothing synchronous after the answer.** Indexing a request and recording a claim happen on background threads, batched. Only the lookup you asked for runs inline.
4. **Warmup.** Local models are loaded on a background thread at `Swarmscope(...)` construction (`warm=False` to disable), so the first real call does not pay the load.
5. **Precomputed vectors.** If you already embed the text for your own purposes, pass it: `sdk.request(text, vector=v)`, `sdk.claim(text, vector=v)`. The embedder is not called.

For the OpenAI embedder specifically: batch where you can (the write path already batches up to 64 claims per call), expect the first lookup of each distinct text to cost a round trip, and set `budget_ms` to what your dispatch loop can afford. The budget is a ceiling on your latency, not on the provider's.

## Writing your own

```python
class MyEmbedder:
    name = "mine-v1"   # part of the cache key: bump it when the model changes
    dim = 384
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    def warmup(self) -> None: ...   # optional
```

Return L2-normalised vectors; scores are cosine similarities. Pass the instance as `embedder=`.
