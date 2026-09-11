import time

import pytest

import swarmscope as ss
from swarmscope.claims import CachedEmbedder, CallableEmbedder, HashEmbedder, embedder_from_spec
from swarmscope.reputation import RouterPolicy


class Counting:
    name, dim = "counting", 4

    def __init__(self, delay=0.0):
        self.calls, self.texts, self.delay = 0, 0, delay

    def embed(self, texts):
        self.calls += 1
        self.texts += len(texts)
        if self.delay:
            time.sleep(self.delay)
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]


def test_cached_embedder_lru_and_batching():
    inner = Counting()
    e = CachedEmbedder(inner, None, maxsize=2)
    assert e.embed(["a", "bb"]) == [[1.0, 1, 0, 0], [2.0, 1, 0, 0]]
    assert inner.calls == 1 and inner.texts == 2
    e.embed(["a", "BB "])  # normalised text hits the cache
    assert inner.calls == 1 and e.hits == 2
    e.embed(["ccc"])  # evicts the oldest ("a")
    e.embed(["a"])
    assert inner.calls == 3 and e.stats["lru_size"] == 2
    e.embed(["x", "a", "y"])  # mixed: only misses go to the inner embedder, in one batch
    assert inner.calls == 4 and inner.texts == 2 + 1 + 1 + 2


def test_cache_persists_in_store_across_instances(tmp_path):
    store = ss.SQLiteStore(str(tmp_path / "e.db"))
    inner1 = Counting()
    e1 = CachedEmbedder(inner1, store)
    e1.embed(["hello world"])
    time.sleep(0.05)  # background persist
    inner2 = Counting()
    e2 = CachedEmbedder(inner2, store)
    assert e2.embed(["hello world"]) == [[11.0, 1, 0, 0]] and inner2.calls == 0 and e2.store_hits == 1
    # a different embedder name never reads another's vectors
    e3 = CachedEmbedder(CallableEmbedder(lambda ts: [[9.0, 0, 0, 0]] * len(ts), 4, "other"), store)
    assert e3.embed(["hello world"]) == [[9.0, 0, 0, 0]]
    store.close()


def test_spec_parsing():
    assert isinstance(embedder_from_spec(None), HashEmbedder)
    assert embedder_from_spec("hash:64").dim == 64
    custom = Counting()
    assert embedder_from_spec(custom) is custom
    with pytest.raises(ValueError):
        embedder_from_spec("nope:x")


def test_sdk_wraps_embedder_and_shares_it(sdk):
    assert isinstance(sdk.embedder, CachedEmbedder)
    assert sdk.claims.embedder is sdk.embedder and sdk.reputation.embedder is sdk.embedder
    with sdk.run("e"):
        sdk.request("same text", kind="k")
        sdk.claim("same text")
    assert sdk.stats["embedder"]["hits"] >= 1


def test_request_budget_fails_open():
    slow = Counting(delay=0.3)
    sdk = ss.Swarmscope("memory://", embedder=slow, embedding_cache=False, router=RouterPolicy(latency_budget_ms=20))
    with sdk.run("slow"):
        t0 = time.perf_counter()
        adv = sdk.request("anything")
        assert adv.timed_out and adv.cold_start and (time.perf_counter() - t0) < 0.25
        assert sdk.reputation.stats["timeouts"] == 1
        # a precomputed vector skips the embedder entirely
        adv2 = sdk.request("anything", vector=[1.0, 1.0, 0.0, 0.0])
        assert not adv2.timed_out
    sdk.close()


def test_precomputed_vector_for_claims(sdk):
    with sdk.run("v"):
        v = sdk.embedder.embed(["custom vector claim"])[0]
        with sdk.agent_scope() as a:
            sdk.claim("custom vector claim", vector=v)
        sdk.flush()
        hit = sdk.claim("custom vector claim", vector=v)
        assert hit.similar and hit.best.agent_id == a and hit.top_score > 0.99


def test_model2vec_semantic_recall():
    pytest.importorskip("model2vec")
    try:
        e = embedder_from_spec("model2vec")
    except Exception as exc:  # no network / no model cache
        pytest.skip(f"model2vec model unavailable: {exc}")
    from swarmscope.store.base import cosine

    a, b, c = e.embed(["translate the clause to French", "translate this paragraph into German",
                       "write a python function that sorts"])
    assert cosine(a, b) > cosine(a, c) + 0.2
    assert e.dim == len(a)
    t0 = time.perf_counter()
    e.embed(["one short request"] * 50)
    assert (time.perf_counter() - t0) < 0.5
