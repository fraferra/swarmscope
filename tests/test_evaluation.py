import pytest

import swarmscope as ss
from swarmscope import Contribution
from swarmscope.evaluation import Calibration, ReplayHarness, ablate, online_proxies, shapley, wilson


def _swarm(sdk, n_groups=4, per_group=5, seed=1):
    import random

    rng = random.Random(seed)

    def consolidate(contribs):
        vals = [c.value for c in contribs if c.value is not None]
        return max(vals) if vals else 0

    wrapped = sdk.consolidator(consolidate, check_determinism=True)
    with sdk.run("ev") as run:
        contribs = []
        for g in range(n_groups):
            with sdk.group(f"g{g}"):
                for _ in range(per_group):
                    with sdk.agent_scope(role="w") as a:
                        sdk.generation(model="gpt-4o-mini", input_tokens=100, output_tokens=10)
                        v = 10 if (g == 0 and rng.random() < 0.5) else rng.randint(0, 5)
                        art = sdk.artifact(v)
                        sdk.verdict(art, status="accepted" if v == 10 else "rejected", source="verifier")
                        contribs.append(Contribution(a, v, f"g{g}"))
        wrapped(contribs)
    sdk.flush()
    return run.run_id, consolidate


def test_wilson_bounds():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 0.4
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo > 0.6


def test_replay_fidelity_and_ablation(sdk):
    rid, fn = _swarm(sdk)
    h = ReplayHarness(sdk.store, rid, fn)
    assert h.replayable and len(h.agent_ids) == 20 and len(h.group_ids) == 4
    f = h.fidelity(lambda out: out == 10)
    assert f.identical and f.gap == 0.0
    curve = ablate(h, lambda out: out == 10, ks=[1, 5, 20], samples=20, seed=3)
    ks = [p.k for p in curve.points]
    assert ks == [1, 5, 20]
    ps = [p.p_success for p in curve.points]
    assert ps[0] <= ps[1] <= ps[2] == 1.0
    assert curve.points[-1].ci_low == curve.points[-1].ci_high == 1.0
    assert all(p.ci_low <= p.p_success <= p.ci_high for p in curve.points)
    assert curve.fit is not None and 0 < curve.fit.q < 1
    assert curve.points[-1].cost_usd_mean > curve.points[0].cost_usd_mean
    d = curve.to_dict()
    assert d["points"][0]["score_quantiles"]


def test_group_shapley_identifies_group0(sdk):
    rid, fn = _swarm(sdk)
    h = ReplayHarness(sdk.store, rid, fn)
    res = shapley(h, lambda out: out == 10, level="group", permutations=40, seed=0)
    top = res.ranked()[0]
    assert top.unit == "g0" and top.mean > 0.9
    assert all(v.ci_low <= v.mean <= v.ci_high for v in res.values.values())
    assert abs(sum(v.mean for v in res.values.values()) - res.grand_value) < 1e-9  # efficiency
    agent = shapley(h, lambda out: out == 10, level="agent", permutations=20, seed=0)
    assert len(agent.values) == 20


def test_non_replayable_consolidator_is_flagged(sdk):
    import random

    def noisy(contribs):
        return random.random()

    wrapped = sdk.consolidator(noisy, check_determinism=True)
    with sdk.run("nr") as run:
        wrapped([Contribution("a", 1)])
    sdk.flush()
    h = ReplayHarness(sdk.store, run.run_id, noisy)
    assert not h.replayable and "non-deterministic" in h.reason
    curve = ablate(h, lambda o: o > 0.5)
    assert not curve.replayable and not curve.points and curve.notes
    res = shapley(h, lambda o: o > 0.5)
    assert not res.values and res.notes


def test_missing_consolidation_raises(sdk):
    with sdk.run("none") as run:
        pass
    with pytest.raises(LookupError):
        ReplayHarness(sdk.store, run.run_id, lambda c: 0)


def test_online_proxies_and_calibration(sdk):
    rid, _ = _swarm(sdk)
    p = online_proxies(sdk.store, rid)
    assert p.agents == 20 and p.waste_defined and p.time_to_first_accepted_s is not None
    assert p.claims == 0 and p.dedup_hit_rate is None
    cal = Calibration.fit("objective", "novel_claim_rate", [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert abs(cal.a - 0.1) < 1e-9
    y, warn = cal.predict(2.5)
    assert warn is None and abs(y - 0.25) < 1e-9
    _, warn = cal.predict(9.0)
    assert warn and "outside" in warn
    cal.save(sdk.store)
    assert Calibration.load(sdk.store, "objective", "novel_claim_rate").n == 3
