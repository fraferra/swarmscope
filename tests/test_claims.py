import time

import swarmscope as ss
from swarmscope.claims import CallableJudge, ClaimStore, GatePolicy, HashEmbedder, default_prefilter
from swarmscope.claims.judge import pair_key


def test_hash_embedder_similarity():
    e = HashEmbedder()
    a, b, c = e.embed(["sieve primes below 100", "sieve primes below one hundred", "gradient descent on relaxation"])
    from swarmscope.store.base import cosine

    assert cosine(a, b) > cosine(a, c)
    assert abs(cosine(a, a) - 1.0) < 1e-6


def test_prefilter_contradictory_assumptions():
    assert default_prefilter({"assumes": ["A"]}, {"assumes": ["A"]})
    assert not default_prefilter({"assumes": ["A"]}, {"assumes": ["¬A"]})
    assert not default_prefilter({"assumes": ["bounded"]}, {"assumes": ["not bounded"]})
    assert default_prefilter({}, {"assumes": ["A"]})


def test_claim_lookup_advisory(sdk):
    with sdk.run("c"):
        with sdk.agent_scope() as a1:
            h1 = sdk.claim("bound vorticity via Gronwall under A", kind="approach", metadata={"assumes": ["A"]})
            assert not h1.similar and not h1.suggest_skip
        sdk.flush()  # drains the async claim writer deterministically
        with sdk.agent_scope() as a2:
            h2 = sdk.claim("bound vorticity via Gronwall under A", kind="approach", metadata={"assumes": ["A"]})
            assert h2.similar and h2.top_score > 0.99 and h2.suggest_skip
            assert h2.best.agent_id == a1
            h2.suppress("dup")
        sdk.flush()
        with sdk.agent_scope():
            h3 = sdk.claim("bound vorticity via Gronwall under A", kind="approach", metadata={"assumes": ["¬A"]})
            assert not h3.similar  # prefilter excludes contradictory assumption
    sdk.flush()
    evs = sdk.store.events(sdk.store.runs()[0].run_id, types=["suppression"])
    assert len(evs) == 1 and evs[0].suppressed_by == h1.claim_id


def test_judge_cache_makes_calls_trend_to_zero():
    calls = {"n": 0}

    def fn(a, b, ma, mb):
        calls["n"] += 1
        return a.lower().replace("one hundred", "100") == b.lower().replace("one hundred", "100")

    sdk = ss.Swarmscope("memory://", judge=CallableJudge(fn), gate=GatePolicy(judge_min_score=0.3))
    with sdk.run("j"):
        with sdk.agent_scope():
            sdk.claim("sieve primes below 100", kind="approach")
        sdk.flush()
        for _ in range(5):
            with sdk.agent_scope():
                h = sdk.claim("sieve primes below one hundred", kind="approach")
                assert h.judged and h.best.equivalent and h.suggest_skip
            sdk.flush()
    st = sdk.claims.judge_stats
    # Each new claim is compared against all prior ones, but identical pairs hit the cache.
    assert st["calls"] <= 2 and st["cache_hits"] >= 4
    assert st["calls_per_lookup"] < 0.5
    sdk.close()


def test_lookup_fails_open_on_timeout():
    class Slow:
        name, dim = "slow", 4

        def embed(self, texts):
            time.sleep(0.3)
            return [[1, 0, 0, 0] for _ in texts]

    sdk = ss.Swarmscope("memory://", embedder=Slow(), gate=GatePolicy(latency_budget_ms=20))
    with sdk.run("t"):
        t0 = time.perf_counter()
        h = sdk.claim("x")
        assert h.timed_out and not h.similar
        assert (time.perf_counter() - t0) < 0.25
    sdk.close()


def test_experiment_ungated_arm_skips_lookup():
    sdk = ss.Swarmscope("memory://", experiment=ss.ExperimentConfig(gated_fraction=0.0))  # all ungated
    with sdk.run("e") as run:
        with sdk.agent_scope():
            h = sdk.claim("anything")
            assert h.arm == "ungated" and not h.similar
    sdk.flush()
    ev = sdk.store.events(run.run_id, types=["claim"])[0]
    assert ev.arm == "ungated" and ev.dedup == {"skipped": True}
    sdk.close()


def test_pair_key_symmetric():
    assert pair_key("j", "A b", "c") == pair_key("j", "c", "a  B")


def test_sqlite_vector_search(tmp_path):
    store = ss.SQLiteStore(str(tmp_path / "v.db"))
    e = HashEmbedder(64)
    for i, t in enumerate(["alpha beta", "alpha gamma", "zeta eta"]):
        store.upsert_vector("r", f"c{i}", e.embed([t])[0], t, "k", {}, "a")
    hits = store.search_vectors(e.embed(["alpha beta"])[0], k=2, run_id="r")
    assert hits[0].claim_id == "c0" and hits[0].score > 0.99
    assert store.vector_count("r") == 3
    store.close()


def test_query_api_and_cli(tmp_path, capsys):
    from swarmscope.surfaces.cli import main

    path = tmp_path / "q.db"
    sdk = ss.Swarmscope(f"sqlite:///{path}", flush_interval=0.01)
    with sdk.run("q") as run:
        with sdk.agent_scope() as a:
            sdk.claim("sieve primes below 100", kind="approach")
            sdk.claim("gradient descent on a relaxation", kind="approach")
        hits = sdk.search_claims("sieve primes below one hundred")
        assert hits and hits[0].agent_id == a and "sieve" in hits[0].text
        assert sdk.search_claims("sieve primes", kind="result") == []
        assert sdk.search_claims("sieve primes", run_id=None)  # store-wide
    sdk.close()
    main(["--store", str(path), "claims", run.run_id])
    out = capsys.readouterr().out
    assert "sieve" in out and "gradient" in out
    main(["--store", str(path), "claims", "--all-runs", "--query", "primes sieve", "--limit", "1"])
    assert "sieve" in capsys.readouterr().out
