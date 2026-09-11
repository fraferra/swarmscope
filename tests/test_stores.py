"""Backend conformance: every store passes the same scenario."""
import os

import pytest

import swarmscope as ss
from swarmscope.claims import HashEmbedder
from swarmscope.store.base import open_store

PG_URL = os.environ.get("SWARMSCOPE_PG_URL")


def _backends(tmp_path):
    yield "memory://"
    yield f"sqlite:///{tmp_path / 's.db'}"
    try:
        import duckdb  # noqa: F401
        yield f"duckdb:///{tmp_path / 'd.duckdb'}"
    except ImportError:
        pass
    if PG_URL:
        yield PG_URL


@pytest.fixture(params=["memory", "sqlite", "duckdb", "postgres"])
def store(request, tmp_path):
    kind = request.param
    if kind == "memory":
        url = "memory://"
    elif kind == "sqlite":
        url = f"sqlite:///{tmp_path / 's.db'}"
    elif kind == "duckdb":
        pytest.importorskip("duckdb")
        url = f"duckdb:///{tmp_path / 'd.duckdb'}"
    else:
        if not PG_URL:
            pytest.skip("set SWARMSCOPE_PG_URL to test the Postgres backend")
        url = PG_URL
    st = open_store(url)
    yield st
    st.close()


def test_store_scenario(store):
    sdk = ss.Swarmscope(store, flush_interval=0.01)
    with sdk.run("conf", meta={"k": 1}) as run:
        with sdk.group("g"):
            with sdk.agent_scope(role="w") as a:
                sdk.generation(model="gpt-4o", input_tokens=10, output_tokens=5)
                h1 = sdk.claim("bound vorticity via gronwall", kind="approach", metadata={"assumes": ["A"]})
                sdk.flush()
                h2 = sdk.claim("bound vorticity via gronwall", kind="approach", metadata={"assumes": ["A"]})
                art = sdk.artifact({"x": 1})
                sdk.verdict(art, status="accepted", source="verifier")
    sdk.flush()
    assert h2.similar and h2.best.claim_id == h1.claim_id and h2.top_score > 0.99
    info = store.run(run.run_id)
    assert info and info.name == "conf" and info.meta == {"k": 1} and info.event_count >= 7
    evs = store.events(run.run_id)
    assert [e.type for e in evs][:2] == ["agent_start", "generation"]
    assert store.events(run.run_id, types=["claim"], agent_id=a)[0].agent_id == a
    assert store.events(run.run_id, limit=2).__len__() == 2
    assert store.vector_count(run.run_id) == 2
    hits = store.search_vectors(HashEmbedder().embed(["bound vorticity via gronwall"])[0], k=1, run_id=run.run_id)
    assert hits and hits[0].score > 0.99 and hits[0].metadata == {"assumes": ["A"]}
    assert store.search_vectors([0.0] * 512, k=1, run_id=run.run_id, kind="nope") == []
    store.judge_cache_put("k", {"equivalent": True, "confidence": 0.9})
    assert store.judge_cache_get("k")["equivalent"] is True and store.judge_cache_get("missing") is None
    store.calibration_put("w", {"a": 1})
    assert store.calibration_get("w") == {"a": 1}
    rep = ss.waste_report(store, run.run_id)
    assert rep.defined and rep.waste_ratio == 0.0
    # reputation tables
    from swarmscope.store.base import RouteOutcome

    store.route_outcomes_put([RouteOutcome("rq1", "ar1", run.run_id, "k1", ["a|m|"], True, 1.0, 1.0, "verifier")])
    store.route_outcomes_put([RouteOutcome("rq1", "ar1", run.run_id, "k1", ["a|m|"], False, 0.5, 2.0, "human")])  # upsert
    outs = store.route_outcomes(["rq1"])
    assert len(outs) == 1 and outs[0].accepted is False and outs[0].weight == 0.5 and outs[0].route == ["a|m|"]
    assert store.route_outcomes([]) == [] and len(store.route_outcomes()) == 1
    store.route_stats_add("k1", ["a|m|"], 1.0, 0.0, 1.0)
    store.route_stats_add("k1", ["a|m|"], 0.0, 0.5, 3.0)
    store.route_stats_add("k2", ["b|m|"], 0.0, 1.0, 1.0)
    st = store.route_stats()
    assert [x.route_key for x in st] == ["k1", "k2"] and st[0].successes == 1.0 and st[0].failures == 0.5 and st[0].last_ts == 3.0
    assert store.route_stats(limit=1)[0].route_key == "k1"
    store.route_stats_clear()
    assert store.route_stats() == [] and store.route_outcomes() == []
    sdk.buffer.close()
    sdk.claims.close()


def test_open_store_urls(tmp_path):
    assert type(open_store("memory://")).__name__ == "MemoryStore"
    assert type(open_store(f"sqlite:///{tmp_path/'a.db'}")).__name__ == "SQLiteStore"
    pytest.importorskip("duckdb")
    assert type(open_store(f"duckdb:///{tmp_path/'a.duckdb'}")).__name__ == "DuckDBStore"
    with pytest.raises(ValueError):
        open_store("redis://x")
