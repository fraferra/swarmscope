import random

import pytest

import swarmscope as ss
from swarmscope.core import context as ctx
from swarmscope.reputation import BanditPolicy, Router, RouterPolicy, beta_ci, route_key


def test_beta_ci_and_posterior():
    lo, hi = beta_ci(1, 1)
    assert lo < 0.5 < hi
    p = BanditPolicy(prior_strength=2.0)
    a, b = p.posterior(local_s=3, local_f=0, global_s=10, global_f=10)
    assert a > b  # local evidence dominates the flat global prior
    a0, b0 = p.posterior(0, 0, 8, 2)  # no local: prior pulled toward global 80%
    assert a0 / (a0 + b0) > 0.6
    rng = random.Random(0)
    assert p.score(3, 1, 10, rng) != p.score(3, 1, 10, rng)  # thompson samples vary
    assert BanditPolicy(policy="greedy").score(3, 1, 10, rng) == 0.75
    assert BanditPolicy(policy="ucb").score(1, 1, 10, rng) == 1.0  # untried arms first
    assert p.decay(0) == 1.0 and BanditPolicy(half_life_s=10).decay(10) == pytest.approx(0.5)


def _serve(sdk, text, agent_role, ok, model="m"):
    with sdk.request(text, kind="task") as adv:
        with sdk.agent_scope(role="coordinator", model="big"):
            with sdk.agent_scope(role=agent_role, model=model):
                art = sdk.artifact({"by": agent_role})
                sdk.verdict(art, status="accepted" if ok else "rejected", source="verifier")
    return adv


def test_learns_to_route_similar_requests(sdk):
    sdk.reputation.policy.policy = "greedy"
    with sdk.run("learn"):
        first = sdk.request("translate this contract clause to French")
        assert first.cold_start and first.recommended is None
        # "legal" agent succeeds on translation-type requests, "coder" fails on them
        for i in range(6):
            _serve(sdk, f"translate contract clause {i} to French", "legal", ok=True)
            _serve(sdk, f"translate contract clause {i} to German", "coder", ok=False)
        # coder is great at code requests
        for i in range(6):
            _serve(sdk, f"write a python function for task {i}", "coder", ok=True)
        sdk.flush()
        adv = sdk.request("translate this contract clause to Spanish", kind="task")
        assert not adv.cold_start
        assert adv.recommended.route[-1].startswith("legal|")
        assert adv.recommended.local_successes > 0 and adv.recommended.similar_requests >= 3
        assert adv.recommended.route == ("coordinator|big|", "legal|m|")  # full sequence by default
        code = sdk.request("write a python function that sorts a list", kind="task")
        assert code.recommended.route[-1].startswith("coder|")
        # agent-level aggregation is available too
        agents = adv.by_agent()
        assert agents[0].route == ("legal|m|",)
        # the global leaderboard knows both routes
        lb = sdk.reputation.leaderboard(unit="agent")
        assert {r.route[0].split("|")[0] for r in lb} >= {"legal", "coder"}
        stats = sdk.stats["reputation"]
        assert stats["outcomes_recorded"] == 18 and stats["advised"] >= 20


def test_verdict_revision_and_source_weights(sdk):
    with sdk.run("rev"):
        with sdk.request("do x") as adv:
            with sdk.agent_scope(role="a"):
                art = sdk.artifact(1)
                sdk.verdict(art, status="accepted", source="judge", confidence=0.5)  # weight 0.7*0.5
        st = sdk.store.route_stats()
        assert st[0].successes == pytest.approx(0.35) and st[0].failures == 0
        # a human later rejects it: the old contribution is undone, not double-counted
        sdk.verdict(art, status="rejected", source="human")
        st = sdk.store.route_stats()
        assert st[0].successes == pytest.approx(0.0) and st[0].failures == pytest.approx(1.0)
        # pending verdicts carry no evidence
        with sdk.request("do y") as adv2:
            with sdk.agent_scope(role="b"):
                sdk.verdict(sdk.artifact(2), status="pending", source="human")
        assert len(sdk.store.route_stats()) == 1


def test_reputation_persists_across_runs_and_rebuild(tmp_path):
    path = tmp_path / "rep.db"
    sdk = ss.Swarmscope(f"sqlite:///{path}", router=RouterPolicy(policy="greedy"), flush_interval=0.01)
    with sdk.run("day1"):
        for i in range(4):
            _serve(sdk, f"summarise earnings call {i}", "analyst", ok=True)
    sdk.close()
    sdk2 = ss.Swarmscope(f"sqlite:///{path}", router=RouterPolicy(policy="greedy"), flush_interval=0.01)
    with sdk2.run("day2"):
        adv = sdk2.request("summarise the Q3 earnings call", kind="task")
        assert not adv.cold_start and adv.recommended.route[-1].startswith("analyst|")
        assert adv.recommended.global_successes == pytest.approx(4.0)
    # rebuild from the event log reproduces the same stats
    before = {s.route_key: (s.successes, s.failures) for s in sdk2.store.route_stats()}
    n = sdk2.reputation.rebuild()
    after = {s.route_key: (s.successes, s.failures) for s in sdk2.store.route_stats()}
    assert n == 4 and after == before
    sdk2.close()


def test_post_hoc_verdict_counts_after_rebuild(tmp_path):
    path = tmp_path / "ph.db"
    sdk = ss.Swarmscope(f"sqlite:///{path}", flush_interval=0.01)
    with sdk.run("r") as run:
        with sdk.request("classify ticket") as adv:
            with sdk.agent_scope(role="triager"):
                art = sdk.artifact("bug")
    sdk.close()
    assert not ss.SQLiteStore(str(path)).route_stats()
    # a reviewer accepts it later from another process (e.g. `swarmscope review`)
    later = ss.Swarmscope(f"sqlite:///{path}", flush_interval=0.01)
    later.verdict(art.artifact_id, status="accepted", source="human", run_id=run.run_id)
    later.flush()
    assert later.reputation.rebuild() == 1
    lb = later.reputation.leaderboard()
    assert lb and lb[0].route == ("triager||",) and lb[0].global_successes == 1.0
    later.close()


def test_route_propagates_through_headers(sdk):
    with sdk.run("h"):
        with sdk.request("job") as adv:
            with sdk.agent_scope(role="dispatcher", model="big"):
                headers = sdk.headers()
        assert headers["x-swarmscope-request-id"] == adv.request_id
        with ctx.use(ctx.RunContext()):
            with sdk.attach_headers(headers):
                with sdk.agent_scope(role="worker", model="small"):
                    art = sdk.artifact("out")
                    sdk.verdict(art, status="accepted", source="verifier")
    st = sdk.store.route_stats()
    assert st[0].route == ["dispatcher|big|", "worker|small|"] and st[0].successes == 1.0
    ev = sdk.store.events(sdk.store.runs()[0].run_id, types=["artifact"])[0]
    assert ev.request_id == adv.request_id and ev.route == ["dispatcher|big|", "worker|small|"]


def test_agent_unit_and_ucb(sdk):
    sdk.reputation.policy = RouterPolicy(policy="ucb", unit="agent")
    with sdk.run("u"):
        _serve(sdk, "fix the flaky test", "fixer", ok=True)
        _serve(sdk, "fix the broken test", "fixer", ok=True)
        _serve(sdk, "fix the failing test", "breaker", ok=False)
        adv = sdk.request("fix the test that fails", kind="task")
        assert adv.unit == "agent" and all(len(r.route) == 1 for r in adv.routes)
        assert adv.recommended.route == ("fixer|m|",)
        assert adv.recommended.score >= adv.recommended.mean  # UCB adds a bonus


def test_custom_identity_and_min_similarity(sdk):
    sdk.reputation.policy = RouterPolicy(policy="greedy", min_similarity=0.99)
    with sdk.run("id"):
        with sdk.request("alpha beta gamma") as adv:
            with sdk.agent_scope(role="x", identity="team-blue/v3"):
                sdk.verdict(sdk.artifact(1), status="accepted", source="verifier")
        adv = sdk.request("something entirely unrelated to that")
        # no similar request passes the threshold, but the global prior still surfaces the route
        assert not adv.similar and adv.routes and adv.routes[0].route == ("team-blue/v3",)
        assert adv.routes[0].local_successes == 0 and adv.routes[0].global_successes == 1


def test_cli_reputation(tmp_path, capsys):
    from swarmscope.surfaces.cli import main

    path = tmp_path / "c.db"
    sdk = ss.Swarmscope(f"sqlite:///{path}", flush_interval=0.01)
    with sdk.run("c"):
        _serve(sdk, "draft a press release", "writer", ok=True)
        _serve(sdk, "draft a blog post", "writer", ok=True)
    sdk.close()
    main(["--store", str(path), "reputation"])
    out = capsys.readouterr().out
    assert "writer" in out and "coordinator" in out
    main(["--store", str(path), "reputation", "--unit", "agent", "--json"])
    import json

    data = json.loads(capsys.readouterr().out)
    assert data["leaderboard"][0]["route"] == ["writer|m|"]
    main(["--store", str(path), "reputation", "-q", "draft an announcement", "--policy", "greedy"])
    assert "writer" in capsys.readouterr().out
    main(["--store", str(path), "reputation", "--rebuild"])
    assert "rebuilt" in capsys.readouterr().out


def test_untried_registered_agents_are_explored(sdk):
    """A bandit must know its arms: decorated agents are candidates before they ever run."""
    sdk.reputation.policy = RouterPolicy(policy="greedy")
    a = sdk.agent(role="alpha", model="m")(lambda: "a")
    b = sdk.agent(role="beta", model="m")(lambda: "b")
    with sdk.run("explore"):
        adv = sdk.request("anything")
        assert not adv.cold_start and {r.route[0].split("|")[0] for r in adv.routes} == {"alpha", "beta"}
        assert all(r.n == 0 and r.mean == 0.5 for r in adv.routes)
        # explicit candidates override the registry
        adv2 = sdk.request("anything", candidates=["gamma|x|", "alpha|m| > delta|m|"])
        assert {r.route for r in adv2.routes} == {("gamma|x|",), ("alpha|m|", "delta|m|")}


def test_kind_scopes_recall(sdk):
    sdk.reputation.policy = RouterPolicy(policy="greedy")
    with sdk.run("kinds"):
        _serve(sdk, "handle item 1", "specialist", ok=True)  # kind="task" inside _serve
        adv_same = sdk.request("handle item 2", kind="task")
        adv_other = sdk.request("handle item 2", kind="other")
        assert adv_same.similar and not adv_other.similar
