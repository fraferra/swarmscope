"""Bandit routing: a dispatcher learns which specialist handles which request type.

Three specialists with different (hidden) success rates per request type.
A Thompson-sampling dispatcher uses ``sdk.request`` advice; a random
dispatcher ignores it. Both are scored by a verifier. Watch the routed
acceptance rate climb over rounds while the random one stays flat.

    python examples/reputation_routing.py
"""
from __future__ import annotations

import random

import swarmscope as ss

TYPES = {
    "translate": ["translate the clause to French", "translate this paragraph into German", "render the notice in Spanish"],
    "code": ["write a python function that sorts", "implement a parser for csv", "fix the failing unit test"],
    "summarise": ["summarise the earnings call", "condense this report to a paragraph", "give me the gist of the memo"],
}
# hidden ground truth: P(accepted) per (specialist, request type)
SKILL = {
    "linguist": {"translate": 0.9, "code": 0.1, "summarise": 0.5},
    "coder": {"translate": 0.1, "code": 0.85, "summarise": 0.3},
    "editor": {"translate": 0.4, "code": 0.2, "summarise": 0.9},
}


def run(policy: str | None, rounds: int = 12, per_round: int = 30, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    sdk = ss.Swarmscope("memory://", router=ss.RouterPolicy(policy=policy or "thompson"))
    specialists = {name: sdk.agent(role=name, model="m")(lambda req, n=name: n) for name in SKILL}
    rates = []
    with sdk.run(f"routing-{policy or 'random'}"):
        for r in range(rounds):
            accepted = 0
            for _ in range(per_round):
                rtype = rng.choice(list(TYPES))
                text = rng.choice(TYPES[rtype]) + f" #{rng.randint(0, 999)}"
                with sdk.request(text, kind=rtype) as adv:
                    if policy is None or adv.cold_start:
                        name = rng.choice(list(SKILL))
                    else:
                        name = adv.recommended_agent.split("|")[0]
                    specialists[name](text)  # the agent scope records the route
                    with sdk.agent_scope(role=name, model="m"):  # produce inside the specialist's scope
                        art = sdk.artifact({"by": name})
                        ok = rng.random() < SKILL[name][rtype]
                        sdk.verdict(art, status="accepted" if ok else "rejected", source="verifier")
                accepted += ok
            rates.append(accepted / per_round)
    sdk.close()
    return rates


if __name__ == "__main__":
    random_rates = run(None)
    ts_rates = run("thompson")
    ucb_rates = run("ucb")
    print("round   random  thompson  ucb")
    for i, (a, b, c) in enumerate(zip(random_rates, ts_rates, ucb_rates), 1):
        print(f"{i:>5}   {a:.2f}    {b:.2f}      {c:.2f}")
    print(f"\nmean of last 4 rounds: random={sum(random_rates[-4:]) / 4:.2f} "
          f"thompson={sum(ts_rates[-4:]) / 4:.2f} ucb={sum(ucb_rates[-4:]) / 4:.2f}  (oracle ≈ 0.88)")
