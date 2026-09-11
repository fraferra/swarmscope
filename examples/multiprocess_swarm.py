"""Live coordination across processes: a shared store + lineage headers.

A dispatcher spawns N worker processes. Each worker opens its own Swarmscope
on the same SQLite file, adopts the dispatcher's lineage from headers, and
registers the approach it is about to try. Because lookups hit the shared
store, workers see each other's claims and most duplicates are skipped —
without any messaging between workers.

    python examples/multiprocess_swarm.py
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import tempfile

import swarmscope as ss

APPROACHES = ["sieve", "trial division", "random search", "gradient descent", "sieve with bit array"]


def worker(args):
    store_url, headers, i = args
    sdk = ss.Swarmscope(store_url, gate=ss.GatePolicy(threshold=0.8, scope="run"))
    rng = random.Random(i)
    with sdk.attach_headers(headers):
        with sdk.agent_scope(role="worker", group=f"g{i % 4}") as aid:
            approach = rng.choice(APPROACHES)
            hit = sdk.claim(approach, kind="approach")
            if hit.suggest_skip:
                hit.suppress(f"seen from {hit.best.agent_id}")
                out = None
            else:
                sdk.generation(model="gpt-4o-mini", input_tokens=500, output_tokens=100)
                out = 97 if approach.startswith("sieve") and rng.random() < 0.7 else rng.randint(2, 96)
                art = sdk.artifact({"answer": out}, kind="answer")
                sdk.verdict(art, status="accepted" if out == 97 else "rejected", source="verifier")
    sdk.close()
    return aid, out, hit.suggest_skip


def main() -> None:
    path = os.path.join(tempfile.gettempdir(), f"swarmscope-mp-{os.getpid()}.db")
    url = f"sqlite:///{path}"
    sdk = ss.Swarmscope(url)
    with sdk.run("multiprocess") as run:
        with sdk.agent_scope(role="dispatcher") as dispatcher:
            headers = sdk.headers()
            sdk.flush()
            with mp.get_context("spawn").Pool(4) as pool:
                results = pool.map(worker, [(url, headers, i) for i in range(24)])
    sdk.flush()
    rep = ss.cost_rollup(sdk.store, run.run_id)
    waste = ss.waste_report(sdk.store, run.run_id)
    prox = ss.online_proxies(sdk.store, run.run_id)
    kids = [n for n in rep.tree if n["agent_id"] == dispatcher][0]["children"]
    print(f"run {run.run_id}: {rep.totals['agents']} agents, {len(kids)} workers under the dispatcher, "
          f"unknown lineage {rep.unknown_lineage_fraction:.0%}")
    print(f"skipped on dedup advice: {sum(1 for r in results if r[2])}/{len(results)}  suppressions={prox.suppressions}")
    print(f"waste ratio {waste.waste_ratio:.0%}, accepted {waste.accepted_artifacts}")
    sdk.close()
    os.remove(path)


if __name__ == "__main__":
    sys.exit(main())
