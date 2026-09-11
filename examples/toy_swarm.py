"""A 50-agent toy swarm exercising every layer end to end (no network).

Ten groups × five prover agents search for a number with a known property.
Each agent "generates" (fake tokens), registers a claim about the approach it
takes, may skip work on dedup advice, produces an artifact, and a verifier
issues a verdict. A consolidator picks the best accepted artifact. Then we run
cost rollup, waste ratio, dedup stats, ablation, Shapley, and online proxies.

Run:  python examples/toy_swarm.py [store_url]
Then: SWARMSCOPE_STORE=<store_url> swarmscope ablate --consolidator examples.toy_swarm:consolidate \
          --scorer examples.toy_swarm:verify --shapley --unit group
"""
from __future__ import annotations

import random
import sys
import time

import swarmscope as ss
from swarmscope import Contribution

TARGET = 97  # the "hard" answer; only some approaches find it
APPROACHES = [
    ("sieve primes below 100", {"technique": "sieve", "assumes": ["bounded"]}),
    ("sieve primes below 100 using a bit array", {"technique": "sieve", "assumes": ["bounded"]}),
    ("random search over integers", {"technique": "random", "assumes": ["bounded"]}),
    ("trial division upward from 90", {"technique": "trial", "assumes": ["bounded"]}),
    ("trial division upward from ninety", {"technique": "trial", "assumes": ["bounded"]}),
    ("gradient descent on a continuous relaxation", {"technique": "gd", "assumes": ["not bounded"]}),
]


def verify(x: int | None) -> bool:
    return x == TARGET


def consolidate(contributions):
    """Aggregation step: replayable, deterministic. Importable for ``swarmscope ablate``."""
    answers = [c.value for c in contributions if c.value is not None]
    return TARGET if TARGET in answers else (max(answers) if answers else None)


def main(store_url: str = "memory://") -> None:
    rng = random.Random(7)
    sdk = ss.init(store_url, gate=ss.GatePolicy(threshold=0.85), retain_content=True)

    @sdk.tool
    def check(x: int) -> bool:
        return verify(x)

    @sdk.agent(role="prover")
    def prover(i: int) -> int | None:
        text, meta = rng.choice(APPROACHES)
        sdk.generation(model="gpt-4o-mini", input_tokens=rng.randint(400, 1200), output_tokens=rng.randint(50, 300),
                       cached_input_tokens=rng.randint(0, 200), latency_ms=rng.uniform(200, 900))
        hit = sdk.claim(text, kind="approach", metadata=meta)
        if hit.suggest_skip and rng.random() < 0.5:
            hit.suppress("dedup advice")
            sdk.message({"skipped": text}, to=[])
            return None
        # some techniques find the target, some don't
        found = {"sieve": 0.6, "trial": 0.5, "random": 0.15, "gd": 0.0}[meta["technique"]]
        answer = TARGET if rng.random() < found else rng.randint(2, 96)
        sdk.generation(model="gpt-4o-mini", input_tokens=rng.randint(200, 600), output_tokens=rng.randint(20, 120))
        art = sdk.artifact({"answer": answer, "via": meta["technique"]}, kind="answer")
        ok = check(answer)
        sdk.verdict(art, status="accepted" if ok else "rejected", source="verifier", confidence=1.0,
                    evidence={"verify": ok})
        return answer

    consolidate_recorded = sdk.consolidator(check_determinism=True)(consolidate)

    t0 = time.perf_counter()
    with sdk.run("toy-swarm", workload="objective-verifier") as run:
        contributions = []
        for g in range(10):
            with sdk.group(f"g{g}"):
                for i in range(5):
                    with sdk.agent_scope(role="coordinator", name=f"coord-{g}") as coord:
                        ans = prover(g * 5 + i)
                        contributions.append(Contribution(coord, ans, f"g{g}"))
        consolidate_recorded(contributions)
    sdk.flush()
    wall = time.perf_counter() - t0
    store = sdk.store
    rid = run.run_id

    cost = ss.cost_rollup(store, rid)
    waste = ss.waste_report(store, rid)
    prox = ss.online_proxies(store, rid, window_s=0.05)
    print(f"run {rid} in {wall * 1000:.0f} ms")
    print(f"agents={cost.totals['agents']} generations={cost.totals['generations']} tokens={cost.totals['tokens']} "
          f"cost=${cost.totals['cost_usd']:.4f} unknown_lineage={cost.unknown_lineage_fraction:.1%}")
    print(f"waste ratio={waste.waste_ratio:.1%} accepted={waste.accepted_artifacts} "
          f"contributing={waste.contributing_agents}/{waste.total_agents} cost/accepted p50=${waste.cost_per_accepted_dist['p50']:.4f}")
    print(f"dedup hit rate={prox.dedup_hit_rate:.1%} novel/agent-h={prox.novel_claim_rate_per_agent_hour:.0f} "
          f"suppressions={prox.suppressions} entropy={prox.coverage_entropy:.2f} judge calls/claim={prox.judge_calls_per_claim}")

    harness = ss.ReplayHarness(store, rid, consolidate)
    curve = ss.ablate(harness, verify, ks=[1, 2, 5, 10, 25, 50], samples=40, unit="agent")
    print("value vs k (P(success) [95% CI]):")
    for p in curve.points:
        print(f"  k={p.k:<3d} p={p.p_success:.2f} [{p.ci_low:.2f},{p.ci_high:.2f}] cost=${p.cost_usd_mean:.4f}")
    print(f"knee_k={curve.knee_k} fit_q={curve.fit.q:.3f} fit_knee={curve.fit.knee_k} fidelity_identical={curve.fidelity.identical}")
    shap = ss.shapley(harness, verify, level="group", permutations=60)
    print("group shapley (top 5):")
    for v in shap.ranked()[:5]:
        print(f"  {v.unit:<4s} {v.mean:+.3f} [{v.ci_low:+.3f},{v.ci_high:+.3f}]")
    print("sdk stats:", sdk.stats)
    sdk.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "memory://")
