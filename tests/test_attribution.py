import swarmscope as ss
from swarmscope.attribution import LineageGraph, Rate, SelfHosted, cost_rollup, infer_downstream_verdicts, waste_report


def test_pricing_prefix_and_cached():
    t = ss.PricingTable()
    assert t.resolve("gpt-4o-2024-08-06") is t.resolve("gpt-4o")
    assert t.resolve("openai/gpt-4o-mini") is t.resolve("gpt-4o-mini")
    full = t.cost("gpt-4o", 1_000_000, 0)
    cached = t.cost("gpt-4o", 1_000_000, 0, cached_input_tokens=1_000_000)
    assert full == 2.5 and cached == 1.25
    assert t.cost("gpt-4o", 1_000_000, 0, batch=True) == 1.25
    t.set("llama-local", SelfHosted(usd_per_gpu_hour=2.0, tokens_per_gpu_hour=1_000_000))
    assert t.cost("llama-local", 500_000, 0) == 1.0
    t.set("custom", Rate(1.0, 2.0))
    assert t.cost("custom-v2", 1_000_000, 1_000_000) == 3.0


def _build(sdk, *, accept_from_group="g0"):
    with sdk.run("w") as run:
        arts = {}
        for g in range(3):
            with sdk.group(f"g{g}"):
                with sdk.agent_scope(role="lead", name=f"lead{g}") as lead:
                    with sdk.agent_scope(role="worker") as w:
                        sdk.generation(model="gpt-4o", input_tokens=1000, output_tokens=1000)  # $0.0125
                    sdk.generation(model="gpt-4o", input_tokens=1000, output_tokens=1000)
                    arts[f"g{g}"] = sdk.artifact({"g": g}, kind="answer")
        sdk.verdict(arts[accept_from_group], status="accepted", source="verifier")
        sdk.verdict(arts["g1"], status="rejected", source="judge", confidence=0.7)
    sdk.flush()
    return run.run_id


def test_cost_rollup_subtree(sdk):
    rid = _build(sdk)
    rep = cost_rollup(sdk.store, rid)
    assert rep.totals["agents"] == 6
    assert abs(rep.totals["cost_usd"] - 6 * 0.0125) < 1e-9
    leads = [n for n in rep.tree]
    assert len(leads) == 3
    for n in leads:
        assert abs(n["subtree_cost_usd"] - 0.025) < 1e-9
        assert abs(n["cost_usd"] - 0.0125) < 1e-9
    assert rep.per_group["g0"]["agents"] == 2
    assert rep.per_model["gpt-4o"]["generations"] == 6
    assert rep.unknown_lineage_fraction == 0.0


def test_waste_ratio_reachability(sdk):
    rid = _build(sdk)
    rep = waste_report(sdk.store, rid)
    assert rep.defined
    # only g0 (lead + worker) contributed: 2 of 6 agents, so 4/6 tokens wasted
    assert rep.contributing_agents == 2
    assert abs(rep.waste_ratio - 4 / 6) < 1e-9
    assert rep.accepted_artifacts == 1
    assert abs(rep.cost_per_accepted_dist["p50"] - 0.025) < 1e-9
    assert set(rep.by_source) == {"verifier", "judge"}
    assert rep.by_source["judge"]["waste_ratio"] is None  # judge accepted nothing
    assert rep.verdict_sources == {"verifier": 1, "judge": 1}


def test_waste_undefined_without_verdicts(sdk):
    with sdk.run("nv") as run:
        with sdk.agent_scope():
            sdk.generation(model="gpt-4o", input_tokens=10, output_tokens=10)
            sdk.artifact("a")
    rep = waste_report(sdk.store, run.run_id)
    assert not rep.defined and rep.waste_ratio is None and "undefined" in rep.reason


def test_source_filter_never_mixes(sdk):
    rid = _build(sdk)
    only_judge = waste_report(sdk.store, rid, sources=["judge"])
    assert not only_judge.defined  # judge issued no accept
    human_verifier = waste_report(sdk.store, rid, sources=["human", "verifier"])
    assert human_verifier.defined


def test_downstream_inference_labelled(sdk):
    with sdk.run("d") as run:
        with sdk.agent_scope() as a:
            lemma = sdk.artifact("lemma", kind="lemma")
        with sdk.agent_scope() as b:
            proof = sdk.artifact("proof", kind="proof", inputs=[lemma.artifact_id])
        sdk.verdict(proof, status="accepted", source="verifier")
    sdk.flush()
    new = infer_downstream_verdicts(sdk.store, run.run_id)
    assert len(new) == 1 and new[0].inferred and new[0].source == "downstream"
    assert new[0].artifact_id == lemma.artifact_id
    # excluded by default, included on request
    assert waste_report(sdk.store, run.run_id).accepted_artifacts == 1
    assert waste_report(sdk.store, run.run_id, sources=["verifier", "downstream"],
                        include_inferred=True).accepted_artifacts == 2


def test_unknown_lineage_not_counted_as_waste(sdk):
    with sdk.run("u") as run:
        with sdk.agent_scope() as a:
            sdk.generation(model="gpt-4o", input_tokens=100, output_tokens=0)
            art = sdk.artifact("x")
        sdk.verdict(art, status="accepted", source="human")
        sdk.generation(model="gpt-4o", input_tokens=100, output_tokens=0)  # orphan
    rep = waste_report(sdk.store, run.run_id)
    assert rep.unknown_lineage_agents == 1
    assert rep.waste_ratio == 0.0  # the orphan is excluded from the denominator, not blamed
    assert rep.unknown_lineage_fraction > 0
