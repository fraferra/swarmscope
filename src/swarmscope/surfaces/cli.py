"""``swarmscope`` CLI: run · inspect · cost · waste · dedup · ablate · review · proxies · export · serve."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import runpy
import sys
from typing import Any, Callable

from .._version import __version__
from ..attribution.graph import LineageGraph
from ..attribution.rollup import cost_rollup
from ..attribution.waste import DEFAULT_SOURCES, infer_downstream_verdicts, waste_report
from ..core.events import Artifact, Verdict
from ..evaluation.ablation import ablate
from ..evaluation.proxies import online_proxies
from ..evaluation.replay import ReplayHarness
from ..evaluation.shapley import shapley
from ..store.base import open_store

DEFAULT_STORE = os.environ.get("SWARMSCOPE_STORE", "sqlite:///swarmscope.db")


def _load(spec: str) -> Callable[..., Any]:
    """``package.module:function``"""
    mod, _, name = spec.partition(":")
    if not name:
        raise SystemExit(f"expected module:function, got {spec!r}")
    cwd = os.getcwd()
    if cwd not in sys.path:  # resolve user modules relative to the working directory, like ``python -m``
        sys.path.insert(0, cwd)
    try:
        return getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"cannot load {spec!r}: {exc}") from exc


def _out(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{k}:")
                print("  " + json.dumps(v, default=str)[:2000])
            else:
                print(f"{k}: {v}")
    else:
        print(obj)


def _resolve_run(store, run_id: str | None) -> str:
    if run_id and run_id != "latest":
        return run_id
    runs = store.runs()
    if not runs:
        raise SystemExit("no runs in store")
    return runs[0].run_id


# ---------------------------------------------------------------- commands
def cmd_runs(a):
    store = open_store(a.store)
    rows = store.runs()
    if a.json:
        return _out([r.to_dict() for r in rows], True)
    for r in rows:
        print(f"{r.run_id}  {r.name or '-':24s}  events={r.event_count:<7d}  "
              f"{'open' if r.ended_at is None else 'done'}")


def cmd_run(a):
    """Execute a Python script inside ``sdk.run`` with the global SDK pointed at --store."""
    import swarmscope as ss

    sdk = ss.init(a.store)
    sys.argv = [a.script] + a.args
    with sdk.run(a.name or os.path.basename(a.script)) as run:
        print(f"swarmscope: run {run.run_id}", file=sys.stderr)
        runpy.run_path(a.script, run_name="__main__")
    sdk.close()


def cmd_inspect(a):
    store = open_store(a.store)
    rid = _resolve_run(store, a.run)
    g = LineageGraph(store.events(rid))
    rep = cost_rollup(store, rid, g)
    if a.json:
        return _out(rep.to_dict(), True)
    t = rep.totals
    print(f"run {rid}")
    for k in ("agents", "generations", "tokens", "cost_usd", "messages", "claims", "artifacts", "verdicts", "duration_s"):
        print(f"  {k:12s} {t[k]}")
    frac = rep.unknown_lineage_fraction
    flag = "  <-- attribution is partial" if frac > 0.2 else ""
    print(f"  unknown lineage: {frac:.1%}{flag}")
    if rep.unpriced_generations:
        print(f"  unpriced generations: {rep.unpriced_generations} (add rates to the pricing table)")

    def walk(nodes, depth=0):
        for n in nodes[: a.limit]:
            print("  " * (depth + 1) + f"- {n['name']} [{n['group_id'] or '-'}] ${n['subtree_cost_usd']:.4f} {n['subtree_tokens']} tok")
            walk(n["children"], depth + 1)

    print("lineage:")
    walk(rep.tree)


def cmd_cost(a):
    store = open_store(a.store)
    rid = _resolve_run(store, a.run)
    rep = cost_rollup(store, rid)
    table = {"agent": rep.per_agent, "group": rep.per_group, "model": rep.per_model}[a.by]
    if a.json:
        return _out({"run_id": rid, "by": a.by, "rows": table, "totals": rep.totals}, True)
    print(f"run {rid}  total ${rep.totals['cost_usd']:.4f}  ({rep.totals['tokens']} tokens; "
          f"unknown lineage {rep.unknown_lineage_fraction:.1%})")
    rows = sorted(table.items(), key=lambda kv: -kv[1].get("cost_usd", 0))
    for key, row in rows[: a.limit]:
        print(f"  {key[:40]:40s} ${row.get('cost_usd', 0):.4f}  tokens={row.get('tokens', row.get('input_tokens', 0) + row.get('output_tokens', 0))}")


def cmd_waste(a):
    store = open_store(a.store)
    rid = _resolve_run(store, a.run)
    if a.infer_downstream:
        n = infer_downstream_verdicts(store, rid, sources=a.sources)
        print(f"inferred {n and len(n)} downstream verdicts (labelled inferred; excluded unless --include-inferred)")
    rep = waste_report(store, rid, sources=a.sources, include_inferred=a.include_inferred)
    if a.json:
        return _out(rep.to_dict(), True)
    print(f"run {rid}")
    if not rep.defined:
        print(f"  waste ratio: UNDEFINED — {rep.reason}")
        print("  supply verdicts via sdk.verdict(...), `swarmscope review`, or --infer-downstream (heuristic)")
        return
    print(f"  waste ratio (tokens): {rep.waste_ratio:.1%}   (cost): {rep.waste_ratio_cost:.1%}" if rep.waste_ratio_cost is not None else f"  waste ratio (tokens): {rep.waste_ratio:.1%}")
    print(f"  accepted artifacts: {rep.accepted_artifacts} from sources {list(rep.sources)}")
    print(f"  contributing agents: {rep.contributing_agents}/{rep.total_agents}  unknown-lineage agents: {rep.unknown_lineage_agents}")
    d = rep.cost_per_accepted_dist
    if d:
        print(f"  cost per accepted artifact: p50=${d['p50']:.4f} p90=${d['p90']:.4f} max=${d['p100']:.4f} (mean=${d['mean']:.4f})")
    print("  by verdict source:")
    for s, r in rep.by_source.items():
        wr = "—" if r["waste_ratio"] is None else f"{r['waste_ratio']:.1%}"
        print(f"    {s:10s} accepted={r['accepted_artifacts']:<4d} contributing={r['contributing_agents']:<5d} waste={wr}")


def cmd_dedup(a):
    store = open_store(a.store)
    rid = _resolve_run(store, a.run)
    p = online_proxies(store, rid, window_s=a.window)
    if a.json:
        return _out(p.to_dict(), True)
    print(f"run {rid}")
    print(f"  claims: {p.claims}  novel: {p.novel_claims}  suppressions: {p.suppressions}")
    print(f"  dedup hit rate: {'—' if p.dedup_hit_rate is None else f'{p.dedup_hit_rate:.1%}'}   "
          f"judge calls/claim: {'—' if p.judge_calls_per_claim is None else f'{p.judge_calls_per_claim:.3f}'}   "
          f"timeouts: {p.claim_timeouts}")
    print(f"  coverage entropy: {'—' if p.coverage_entropy is None else f'{p.coverage_entropy:.2f}'} over {p.coverage_clusters} clusters")
    if p.dedup_hit_rate_over_time:
        print("  hit rate over time:")
        for w in p.dedup_hit_rate_over_time[: a.limit]:
            bar = "#" * int(40 * (w["hit_rate"] or 0))
            print(f"    t+{w['t_start_s']:>7.0f}s {w['hits']:>4d}/{w['claims']:<4d} {bar}")
    # experiment arms, if any
    arms: dict[str, dict[str, int]] = {}
    for e in store.events(rid, types=["claim"]):
        arm = getattr(e, "arm", None)
        if arm:
            arms.setdefault(arm, {"claims": 0, "hits": 0})
            arms[arm]["claims"] += 1
            arms[arm]["hits"] += 1 if (e.dedup or {}).get("suggest_skip") else 0
    if arms:
        print("  experiment arms:", arms)


def cmd_claims(a):
    """List claims for a run, or semantic-search the claim store."""
    import swarmscope as ss

    store = open_store(a.store)
    rid = None if a.all_runs else _resolve_run(store, a.run)
    if a.query:
        cs = ss.Swarmscope(store).claims
        hits = cs.query(a.query, k=a.limit, run_id=rid, kind=a.kind)
        if a.json:
            return _out([m.__dict__ if hasattr(m, "__dict__") else {k: getattr(m, k) for k in m.__slots__} for m in hits], True)
        for m in hits:
            print(f"  {m.score:.3f}  {m.claim_id}  [{m.kind}] {m.text[:100]}  agent={m.agent_id}")
        return
    if rid is None:
        raise SystemExit("--all-runs requires --query")
    claims = [e for e in store.events(rid, types=["claim"]) if getattr(e, "text", "")]
    if a.kind:
        claims = [c for c in claims if c.kind == a.kind]
    if a.json:
        return _out([c.to_dict() for c in claims[: a.limit]], True)
    for c in claims[: a.limit]:
        d = c.dedup or {}
        flag = "DUP " if d.get("suggest_skip") else "    "
        print(f"  {flag}{c.claim_id}  [{c.kind}/{c.status}] {c.text[:90]}  top={d.get('top_score', 0):.2f} agent={c.agent_id}")


def cmd_reputation(a):
    """Route reputation leaderboard, or routing advice for a query."""
    import swarmscope as ss
    from swarmscope.reputation import RouterPolicy

    store = open_store(a.store)
    sdk = ss.Swarmscope(store, router=RouterPolicy(policy=a.policy, unit=a.unit))
    try:
        if a.rebuild:
            n = sdk.reputation.rebuild()
            print(f"rebuilt reputation from the event log: {n} outcomes")
        if a.query:
            adv = sdk.reputation.advise(a.query, request_id="_cli", kind="request")
            if a.json:
                return _out(adv.to_dict(), True)
            print(f"routing advice ({adv.policy}, {adv.unit}; {len(adv.similar)} similar past requests"
                  f"{'; cold start' if adv.cold_start else ''}):")
            for r in adv.routes[: a.limit]:
                print(f"  score={r.score:.3f} mean={r.mean:.2f} [{r.ci_low:.2f},{r.ci_high:.2f}] "
                      f"local={r.local_successes:.1f}/{r.n:.1f} global={r.global_successes:.1f}/"
                      f"{r.global_successes + r.global_failures:.1f} sim={r.mean_similarity:.2f}  {' → '.join(r.route)}")
            return
        lb = sdk.reputation.leaderboard(limit=a.limit, unit=a.unit)
        if a.json:
            return _out({"unit": a.unit, "leaderboard": [r.to_dict() for r in lb]}, True)
        if not lb:
            print("no reputation yet: wrap work in `with sdk.request(...)` and emit verdicts")
            return
        print(f"route reputation ({a.unit}); P(success) with 95% CI, evidence-weighted:")
        for r in lb:
            print(f"  {r.mean:.2f} [{r.ci_low:.2f},{r.ci_high:.2f}]  n={r.global_successes + r.global_failures:5.1f}  "
                  f"{' → '.join(r.route)}")
    finally:
        sdk.buffer.close()
        sdk.claims.close()


def cmd_proxies(a):
    store = open_store(a.store)
    rid = _resolve_run(store, a.run)
    _out(online_proxies(store, rid, window_s=a.window).to_dict(), a.json)


def cmd_ablate(a):
    store = open_store(a.store)
    rid = _resolve_run(store, a.run)
    fn = _load(a.consolidator)
    scorer = _load(a.scorer)
    h = ReplayHarness(store, rid, fn, name=a.name)
    ks = [int(x) for x in a.ks.split(",")] if a.ks else None
    curve = ablate(h, scorer, ks=ks, samples=a.samples, seed=a.seed, unit=a.unit,
                   success_threshold=a.threshold)
    store.calibration_put(f"ablation:{rid}", curve.to_dict())
    result: dict[str, Any] = {"ablation": curve.to_dict()}
    if a.shapley:
        result["shapley"] = shapley(h, scorer, level=a.unit if a.unit in ("agent", "group") else "group",
                                    permutations=a.permutations, seed=a.seed).to_dict()
    if a.json:
        return _out(result, True)
    print(f"run {rid}  unit={curve.unit} n={curve.n_units} replayable={curve.replayable}")
    for n in curve.notes:
        print(f"  note: {n}")
    if curve.fidelity:
        print(f"  fidelity: identical={curve.fidelity.identical} gap={curve.fidelity.gap}")
    print("  k     P(success)  95% CI          score p50   cost")
    for p in curve.points:
        print(f"  {p.k:<5d} {p.p_success:>8.2f}   [{p.ci_low:.2f}, {p.ci_high:.2f}]   {p.score_quantiles.get('q50', 0):>7.2f}   "
              f"${(p.cost_usd_mean or 0):.4f}")
    print(f"  knee_k (empirical): {curve.knee_k}   fit q={curve.fit.q if curve.fit else None} knee={curve.fit.knee_k if curve.fit else None}")
    if a.shapley:
        print("  shapley:")
        for v in result["shapley"]["values"][:20]:
            print(f"    {v['unit'][:30]:30s} {v['mean']:+.3f}  [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]  n={v['n']}")
        for n in result["shapley"]["notes"]:
            print(f"    note: {n}")


def cmd_review(a):
    """Interactive verdict queue: artifacts without a human/verifier/judge verdict."""
    import swarmscope as ss

    store = open_store(a.store)
    rid = _resolve_run(store, a.run)
    g = LineageGraph(store.events(rid))
    judged = {aid for aid, vs in g.verdicts.items() if any(v.source != "downstream" and not v.inferred for v in vs)}
    queue = [art for art in g.artifacts.values() if (a.all or art.artifact_id not in judged)
             and (not a.kind or art.kind == a.kind)]
    if not queue:
        print("nothing to review")
        return
    sdk = ss.Swarmscope(store)
    print(f"{len(queue)} artifacts to review (a=accept r=reject s=skip q=quit)")
    try:
        for i, art in enumerate(queue, 1):
            content = json.dumps(art.content, default=str) if art.content is not None else f"<content not retained; hash {art.content_hash}>"
            print(f"\n[{i}/{len(queue)}] {art.artifact_id} kind={art.kind} agent={art.agent_id} group={art.group_id}")
            print("  " + content[: a.width])
            while True:
                ans = input("  verdict> ").strip().lower()
                if ans in ("a", "r", "s", "q"):
                    break
            if ans == "q":
                break
            if ans == "s":
                continue
            sdk.verdict(art.artifact_id, status="accepted" if ans == "a" else "rejected", source="human",
                        confidence=1.0, evidence={"reviewer": os.environ.get("USER", "cli")}, run_id=rid)
    finally:
        sdk.flush()
        sdk.buffer.close()
    print("done")


def cmd_export(a):
    store = open_store(a.store)
    rid = _resolve_run(store, a.run)
    if a.format == "jsonl":
        out = open(a.out, "w") if a.out else sys.stdout
        n = 0
        for e in store.events(rid):
            out.write(json.dumps(e.to_dict(), default=str) + "\n")
            n += 1
        if a.out:
            out.close()
            print(f"wrote {n} events to {a.out}")
    elif a.format == "otel-json":
        from .otlp import event_attributes

        out = open(a.out, "w") if a.out else sys.stdout
        for e in store.events(rid):
            out.write(json.dumps({"name": e.type, "ts": e.ts, "attributes": event_attributes(e)}, default=str) + "\n")
        if a.out:
            out.close()
    else:
        from .otlp import export_run

        n = export_run(store, rid, endpoint=a.endpoint, console=a.endpoint is None)
        print(f"exported {n} events via OTLP{' to ' + a.endpoint if a.endpoint else ' (console)'}")


def cmd_serve(a):
    from .dashboard import serve

    serve(a.store, a.host, a.port)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swarmscope", description="Instrumentation for multi-agent swarms")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--store", default=DEFAULT_STORE, help="store url (default: $SWARMSCOPE_STORE or sqlite:///swarmscope.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, run=True):
        if run:
            sp.add_argument("run", nargs="?", default="latest", help="run id (default: latest)")
        sp.add_argument("--json", action="store_true")
        sp.add_argument("--limit", type=int, default=50)

    s = sub.add_parser("runs", help="list runs"); common(s, run=False); s.set_defaults(fn=cmd_runs)
    s = sub.add_parser("run", help="execute a script inside a swarmscope run")
    s.add_argument("script"); s.add_argument("args", nargs=argparse.REMAINDER); s.add_argument("--name")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("inspect", help="totals + lineage tree"); common(s); s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("cost", help="cost rollup"); common(s)
    s.add_argument("--by", choices=["agent", "group", "model"], default="group"); s.set_defaults(fn=cmd_cost)
    s = sub.add_parser("waste", help="waste ratio and cost per accepted artifact"); common(s)
    s.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    s.add_argument("--include-inferred", action="store_true")
    s.add_argument("--infer-downstream", action="store_true", help="heuristic: mark consumed artifacts as inferred-accepted")
    s.set_defaults(fn=cmd_waste)
    s = sub.add_parser("dedup", help="dedup hit rate, judge economics, experiment arms"); common(s)
    s.add_argument("--window", type=float, default=60.0); s.set_defaults(fn=cmd_dedup)
    s = sub.add_parser("claims", help="list claims or semantic-search the claim store"); common(s)
    s.add_argument("--query", "-q"); s.add_argument("--kind"); s.add_argument("--all-runs", action="store_true")
    s.set_defaults(fn=cmd_claims)
    s = sub.add_parser("reputation", help="route reputation (bandit over past verdicts)"); common(s, run=False)
    s.add_argument("--query", "-q"); s.add_argument("--unit", choices=["sequence", "agent"], default="sequence")
    s.add_argument("--policy", choices=["thompson", "ucb", "greedy"], default="greedy")
    s.add_argument("--rebuild", action="store_true", help="recompute from the event log (after post-hoc verdicts)")
    s.set_defaults(fn=cmd_reputation)
    s = sub.add_parser("proxies", help="online proxies"); common(s)
    s.add_argument("--window", type=float, default=60.0); s.set_defaults(fn=cmd_proxies)
    s = sub.add_parser("ablate", help="retrospective ablation via replay (+ optional Shapley)"); common(s)
    s.add_argument("--consolidator", required=True, help="module:function wrapped by @sdk.consolidator")
    s.add_argument("--scorer", required=True, help="module:function mapping consolidation output -> score/bool")
    s.add_argument("--name", help="consolidation name if several were recorded")
    s.add_argument("--ks"); s.add_argument("--samples", type=int, default=30); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--unit", choices=["agent", "group"], default="agent"); s.add_argument("--threshold", type=float, default=1.0)
    s.add_argument("--shapley", action="store_true"); s.add_argument("--permutations", type=int, default=100)
    s.set_defaults(fn=cmd_ablate)
    s = sub.add_parser("review", help="interactive verdict queue"); common(s)
    s.add_argument("--kind"); s.add_argument("--all", action="store_true"); s.add_argument("--width", type=int, default=600)
    s.set_defaults(fn=cmd_review)
    s = sub.add_parser("export", help="export a run"); common(s)
    s.add_argument("--format", choices=["jsonl", "otel-json", "otlp"], default="jsonl")
    s.add_argument("--out"); s.add_argument("--endpoint"); s.set_defaults(fn=cmd_export)
    s = sub.add_parser("serve", help="local dashboard"); s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765); s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
