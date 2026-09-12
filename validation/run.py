"""Run the validation protocol and write docs/results.md.

For each workload and each k: run the swarm ``reps`` times (prospective
dose-response), record every run with swarmscope, then on the largest-k run
produce the retrospective ablation curve and compare it with the prospective
points (the replay-fidelity gap). Also runs the gated/ungated dedup experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import Any

import swarmscope as ss
from swarmscope.evaluation.ablation import ablate, wilson
from swarmscope.evaluation.replay import ReplayHarness
from swarmscope.evaluation.shapley import shapley

from .workloads import SOLVERS, WORKLOADS, Workload


def run_swarm(sdk: ss.Swarmscope, workload: Workload, solver, k: int, seed: int, *, task_idx: int = 0,
              experiment: bool = False, workers: int = 1) -> tuple[str, bool | float, dict[str, Any]]:
    """One swarm of k agents on one task. ``workers > 1`` runs agents concurrently (each on a copy
    of the current context so lineage stays correct); agent seeds are fixed up front so results
    do not depend on scheduling."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    task = workload.tasks[task_idx % len(workload.tasks)]
    rng = random.Random(seed)
    consolidate = sdk.consolidator(name=f"{workload.name}-consolidate", check_determinism=True)(workload.consolidate)
    with sdk.run(f"{workload.name}-k{k}-s{seed}", workload=workload.name,
                 meta={"k": k, "seed": seed, "task": task.task_id, "solver": solver.name}) as run:
        groups = max(1, min(10, k // 5))
        seeds = [rng.random() for _ in range(k)]

        def one(i: int) -> ss.Contribution:
            with sdk.group(f"g{i % groups}"):
                with sdk.agent_scope(role="solver") as aid:
                    out = solver.solve(workload, task, random.Random(seeds[i]), sdk)
                    art = sdk.artifact(out, kind="candidate")
                    accepted = None
                    if workload.check is not None:
                        code_or_ans = out["code"] if isinstance(out, dict) and "code" in out else out
                        accepted = workload.check(task, code_or_ans)
                        sdk.verdict(art, status="accepted" if accepted else "rejected", source=workload.verdict_source,
                                    confidence=1.0 if workload.verdict_source == "verifier" else 0.8)
                    value = dict(out, accepted=accepted) if isinstance(out, dict) else out
                    return ss.Contribution(aid, value, f"g{i % groups}")

        if workers > 1 and k > 1:
            ctx_ = contextvars.copy_context()
            with ThreadPoolExecutor(min(workers, k)) as pool:
                contribs = list(pool.map(lambda i: ctx_.copy().run(one, i), range(k)))
        else:
            contribs = [one(i) for i in range(k)]
        final = consolidate(contribs)
        score = workload.score(task, final)
        if workload.name == "fuzzy":  # no human in the loop here: label the proxy honestly
            with sdk.agent_scope(role="consolidator"):
                fart = sdk.artifact(final, kind="report")
                sdk.verdict(fart, status="accepted" if score >= 0.99 else "pending", source="judge", confidence=0.5,
                            evidence={"note": "rubric self-report; run `swarmscope review` for human verdicts"})
    sdk.flush()
    return run.run_id, score, {"task": task.task_id}


def arm_report(store, run_id: str, source: str) -> dict[str, dict[str, float]]:
    """Per-arm outcome: claims, hit rate, suppressions, tokens, accepted artifacts, cost per accepted."""
    from swarmscope.attribution.graph import LineageGraph

    g = LineageGraph(store.events(run_id))
    arm_of: dict[str, str] = {}
    arms: dict[str, dict[str, float]] = {}
    for c in g.claims.values():
        arm = c.arm or "none"
        arm_of[c.agent_id or ""] = arm
        d = arms.setdefault(arm, {"claims": 0, "hits": 0, "suppressions": 0, "agents": 0, "tokens": 0,
                                  "cost_usd": 0.0, "accepted": 0})
        d["claims"] += 1
        d["hits"] += 1 if (c.dedup or {}).get("suggest_skip") else 0
    for e in store.events(run_id, types=["suppression"]):
        arms.setdefault(arm_of.get(e.agent_id or "", "none"), {}).setdefault("suppressions", 0)
        arms[arm_of.get(e.agent_id or "", "none")]["suppressions"] += 1
    for aid, a in g.agents.items():
        arm = arm_of.get(aid)
        if arm is None:
            continue
        arms[arm]["agents"] += 1
        arms[arm]["tokens"] += a.tokens
        arms[arm]["cost_usd"] += a.cost_usd
    accepted = g.accepted_artifacts([source])
    for art_id in accepted:
        art = g.artifacts[art_id]
        arm = arm_of.get(art.agent_id or "")
        if arm:
            arms[arm]["accepted"] += 1
    for d in arms.values():
        d["cost_per_accepted"] = (d["cost_usd"] / d["accepted"]) if d.get("accepted") else None
    return arms


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=value lines; no dependency). Existing env wins."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip().removeprefix("export ").strip(), v.strip().strip("'\"")
                os.environ.setdefault(k, v)
    except FileNotFoundError:
        pass


def main(argv=None) -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=list(SOLVERS), default="simulated")
    ap.add_argument("--workload", choices=list(WORKLOADS) + ["all"], default="all")
    ap.add_argument("--ks", default="1,5,25,100,500")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--store", default="sqlite:///validation.db")
    ap.add_argument("--out", default="docs/results.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write raw results here")
    ap.add_argument("--model", default=None, help="model for the openai solver (default gpt-4o-mini)")
    ap.add_argument("--model-map", default=None,
                    help="per-workload model override, e.g. objective=gpt-4o-mini,fuzzy=gpt-4.1-mini")
    ap.add_argument("--max-usd", type=float, default=5.0, help="abort before starting if the estimate exceeds this")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation for the openai solver")
    ap.add_argument("--workers", type=int, default=None, help="concurrent agents per swarm (default 8 for openai, 1 otherwise)")
    ap.add_argument("--resume", action="store_true", help="reuse swarms already completed in --store (same solver/seed)")
    a = ap.parse_args(argv)

    ks = [int(x) for x in a.ks.split(",")]
    names = list(WORKLOADS) if a.workload == "all" else [a.workload]
    total_calls = len(names) * (a.reps * sum(ks) + max(ks))  # + one experiment run per workload
    if a.solver == "openai":
        est = total_calls * 250 / 1e6 * 0.4  # ~250 tokens/call at blended gpt-4o-mini rates (rough)
        print(f"openai solver: ~{total_calls} model calls, rough estimate ${est:.2f} (model={a.model or 'gpt-4o-mini'})",
              file=sys.stderr)
        if est > a.max_usd:
            raise SystemExit(f"estimate ${est:.2f} exceeds --max-usd {a.max_usd}; reduce --ks/--reps or raise the cap")
        if not a.yes and sys.stdin.isatty():
            if input("proceed? [y/N] ").strip().lower() != "y":
                raise SystemExit("aborted")
        solver = SOLVERS["openai"](a.model)
        # Preflight one tiny call so quota/auth problems abort here, not after thousands of failed generations.
        try:
            solver._client.chat.completions.create(model=solver.model, max_tokens=1,
                                                   messages=[{"role": "user", "content": "ok"}])
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"openai preflight failed ({type(exc).__name__}): {str(exc)[:300]}") from exc
    else:
        solver = SOLVERS[a.solver]()
    workers = a.workers if a.workers is not None else (8 if a.solver == "openai" else 1)
    model_map = dict(kv.split("=", 1) for kv in a.model_map.split(",")) if a.model_map else {}
    results: dict[str, Any] = {"solver": a.solver, "model": getattr(solver, "model", None), "ks": ks, "reps": a.reps,
                               "workers": workers, "workloads": {}}
    t_all = time.time()
    done_calls = 0
    for name in names:
        wl = WORKLOADS[name]
        if name in model_map and hasattr(solver, "model"):
            solver.model = model_map[name]
        wl_model = getattr(solver, "model", None)
        tag = f"{a.solver}:{wl_model}" if wl_model else a.solver
        sdk = ss.Swarmscope(a.store, gate=ss.GatePolicy(threshold=0.85))
        prospective: dict[int, list[float]] = {}
        costs: dict[int, list[float]] = {}
        last_run = None
        for k in ks:
            for r in range(a.reps):
                seed = a.seed * 1000 + k * 10 + r
                t_run = time.time()
                key = f"vrun:{tag}:{name}-k{k}-s{seed}"
                prior = sdk.store.calibration_get(key) if a.resume else None
                if prior:
                    rid, score, cost, resumed = prior["run_id"], prior["score"], prior["cost"], True
                else:
                    rid, score, _ = run_swarm(sdk, wl, solver, k, seed, task_idx=r, workers=workers)
                    cost = ss.cost_rollup(sdk.store, rid).totals["cost_usd"]
                    sdk.store.calibration_put(key, {"run_id": rid, "score": float(score), "cost": cost})
                    resumed = False
                prospective.setdefault(k, []).append(float(score))
                costs.setdefault(k, []).append(cost)
                last_run = rid
                done_calls += k
                print(f"[{name}] k={k:<4d} rep={r} score={float(score):.2f} cost=${cost:.4f} "
                      f"{time.time() - t_run:.1f}s  ({done_calls}/{total_calls} calls){' [resumed]' if resumed else ''}",
                      file=sys.stderr, flush=True)
        # retrospective ablation on the largest run
        harness = ReplayHarness(sdk.store, last_run, wl.consolidate, name=f"{wl.name}-consolidate")
        task = wl.tasks[(a.reps - 1) % len(wl.tasks)]
        curve = ablate(harness, lambda out, t=task: wl.score(t, out), ks=ks, samples=40, seed=a.seed,
                       success_threshold=0.99)
        shap = shapley(harness, lambda out, t=task: wl.score(t, out), level="group", permutations=60, seed=a.seed)
        sdk.store.calibration_put(f"ablation:{last_run}", curve.to_dict())
        prox = ss.online_proxies(sdk.store, last_run)
        waste = ss.waste_report(sdk.store, last_run, sources=[wl.verdict_source])

        # dedup experiment: gated vs ungated arms on the largest k
        exp_sdk = ss.Swarmscope(a.store, gate=ss.GatePolicy(threshold=0.85), experiment=ss.ExperimentConfig())
        key = f"vrun:{tag}:{name}-experiment-k{max(ks)}"
        prior = exp_sdk.store.calibration_get(key) if a.resume else None
        if prior:
            rid_exp = prior["run_id"]
        else:
            rid_exp, _, _ = run_swarm(exp_sdk, wl, solver, max(ks), a.seed + 999, workers=workers)
            exp_sdk.store.calibration_put(key, {"run_id": rid_exp, "score": None, "cost": None})
        arms = arm_report(exp_sdk.store, rid_exp, wl.verdict_source)
        exp_sdk.close()

        points = []
        for k in ks:
            xs = prospective[k]
            succ = sum(1 for x in xs if x >= 0.99)
            lo, hi = wilson(succ, len(xs))
            points.append({"k": k, "p_success": succ / len(xs), "ci": [lo, hi], "n": len(xs),
                           "score_mean": statistics.fmean(xs), "cost_usd_mean": statistics.fmean(costs[k])})
        results["workloads"][name] = {
            "verdict_source": wl.verdict_source, "model": wl_model, "prospective": points, "ablation": curve.to_dict(),
            "shapley": shap.to_dict(), "waste": waste.to_dict(), "proxies": prox.to_dict(), "dedup_arms": arms,
            "largest_run": last_run,
        }
        sdk.close()
    results["elapsed_s"] = time.time() - t_all
    results["solver_retries"] = getattr(solver, "retries", 0)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1, default=str)
    write_report(results, a.out)
    print(f"wrote {a.out} ({results['elapsed_s']:.0f}s)")
    return 0


def write_report(res: dict[str, Any], path: str) -> None:
    L: list[str] = []
    L.append("# Validation results\n")
    L.append(f"Solver: `{res['solver']}`{(' (' + res['model'] + ')') if res.get('model') else ''} · k ∈ {res['ks']} · "
             f"{res['reps']} independent runs per k · generated by `python -m validation.run` in {res.get('elapsed_s', 0):.0f}s.\n")
    if res["solver"] == "openai":
        L.append("> **Real-model curves.** Each agent is one `gpt-4o-mini`-class call at temperature 1.0 with a distinct "
                 "seed; verdicts come from the workload's verifier/judge; the fuzzy workload's verdicts are a keyword "
                 "rubric proxy (`source=judge`, confidence 0.5) until a human reviews them.\n")
    if res.get("findings"):
        L.append("## Findings\n")
        for item in res["findings"]:
            L.append(f"- {item}")
        L.append("")
    if res.get("spend_note"):
        L.append(f"_{res['spend_note']}_\n")
    if res["solver"] == "simulated":
        L.append("> **These curves come from the simulated solver**: a deterministic stand-in whose per-agent "
                 "success probability is fixed per approach. They validate the *methodology* (do retrospective "
                 "ablation curves recover the prospective dose-response? are the CIs calibrated?), not any real "
                 "model. Re-run with `--solver openai` to produce real-model curves for your workload.\n")
    for name, w in res["workloads"].items():
        L.append(f"\n## Workload: {name}  (verdict source: `{w['verdict_source']}`"
                 f"{(', model: `' + w['model'] + '`') if w.get('model') else ''})\n")
        L.append("### P(success) vs k — prospective (independent runs) vs retrospective (replay ablation)\n")
        L.append("| k | prospective P(success) | 95% CI | retrospective P(success) | 95% CI | mean cost USD |")
        L.append("|---|---|---|---|---|---|")
        retro = {p["k"]: p for p in w["ablation"]["points"]}
        for p in w["prospective"]:
            r = retro.get(p["k"])
            rp = f"{r['p_success']:.2f}" if r else "—"
            rci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]" if r else "—"
            L.append(f"| {p['k']} | {p['p_success']:.2f} | [{p['ci'][0]:.2f}, {p['ci'][1]:.2f}] | {rp} | {rci} | {p['cost_usd_mean']:.3g} |")
        if w.get("per_task"):
            L.append("\n### Per task at the largest k (agent accuracy vs consolidated outcome)\n")
            L.append("| task | agents correct | P(agent correct) | consolidated correct? |\n|---|---|---|---|")
            for t in w["per_task"]:
                if t.get("accepted") is None:
                    L.append(f"| {t['task']} | not measured | — | {'yes (proxy)' if t['score'] >= 0.99 else 'no (proxy)'} |")
                else:
                    L.append(f"| {t['task']} | {t['accepted']}/{t['agents']} | {t['agent_rate']:.2f} | "
                             f"{'yes' if t['score'] >= 0.99 else 'no'} |")
        ab = w["ablation"]
        fit = ab.get("fit") or {}
        if w.get("retro_pooled_runs"):
            L.append(f"\nRetrospective column: replay ablation pooled over the {w['retro_pooled_runs']} runs at the largest k "
                     f"(one per task/rep), so both columns average over the same tasks.")
        L.append(f"\nEmpirical knee (first k whose CI reaches 95% of max): **{ab.get('knee_k')}**; "
                 f"independent-lottery fit q={fit.get('q', 0):.3f} → knee≈{fit.get('knee_k')}. "
                 f"Replay fidelity: identical={ab['fidelity']['identical'] if ab.get('fidelity') else None}.")
        gaps = [abs(p["p_success"] - retro[p["k"]]["p_success"]) for p in w["prospective"] if p["k"] in retro]
        if gaps:
            L.append(f"Max |prospective − retrospective| gap: **{max(gaps):.2f}** (this is the number that says whether "
                     f"replay ablation can stand in for re-running the swarm on this workload).")
        L.append("\n### Waste and cost per accepted artifact (largest run)\n")
        ws = w["waste"]
        if ws["defined"]:
            d = ws["cost_per_accepted_dist"]
            L.append(f"Waste ratio {ws['waste_ratio']:.1%} (tokens); {ws['accepted_artifacts']} accepted of "
                     f"{ws['total_agents']} agents; cost per accepted artifact p50={d.get('p50', 0):.3g} "
                     f"p90={d.get('p90', 0):.3g} max={d.get('p100', 0):.3g} USD.")
        else:
            L.append(f"Waste ratio undefined: {ws['reason']}")
        L.append("\n### Group Shapley (largest run, 60 permutations)\n")
        L.append("| group | value | 95% CI |\n|---|---|---|")
        for v in w["shapley"]["values"][:6]:
            L.append(f"| {v['unit']} | {v['mean']:+.3f} | [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] |")
        L.append("\n### Dedup gate experiment (largest k, deterministic arm assignment by agent)\n")
        L.append("| arm | agents | claims | hit rate | suppressions | tokens | accepted | cost / accepted USD |")
        L.append("|---|---|---|---|---|---|---|---|")
        for arm, d in w["dedup_arms"].items():
            rate = d["hits"] / d["claims"] if d.get("claims") else 0
            cpa = "—" if d.get("cost_per_accepted") is None else f"{d['cost_per_accepted']:.3g}"
            L.append(f"| `{arm}` | {int(d.get('agents', 0))} | {int(d.get('claims', 0))} | {rate:.0%} | "
                     f"{int(d.get('suppressions', 0))} | {int(d.get('tokens', 0))} | {int(d.get('accepted', 0))} | {cpa} |")
        if res["solver"] == "simulated":
            L.append("\nThe gate is advisory: simulated gated-arm agents skip ~70% of duplicate approaches. Compare accepted "
                     "artifacts and cost per accepted between arms; a gate that lowers the first is net-harmful on this workload.")
        else:
            L.append("\n**Uninformative with this solver.** The real-model solver registers one identical claim per agent "
                     "(it does not express its approach), so every claim after the first is a duplicate and no agent acts on "
                     "the advice (0 suppressions). The arms differ only by random assignment. A dedup experiment on real "
                     "models needs agents that emit distinct approach claims and honour `suggest_skip`.")
        px = w["proxies"]
        hit_rate = "—" if px["dedup_hit_rate"] is None else f"{px['dedup_hit_rate']:.0%}"
        entropy = "—" if px["coverage_entropy"] is None else f"{px['coverage_entropy']:.2f}"
        L.append(f"\nOnline proxies on the largest run: dedup hit rate {hit_rate}, coverage entropy {entropy}, "
                 f"unknown lineage {px['unknown_lineage_fraction']:.1%}.")
    L.append("\n## Reading these numbers\n")
    L.append("- Everything is P(success) with a CI, never a mean alone: swarm search is heavy-tailed and a team that "
             "sizes off a mean under-provisions.")
    L.append("- The prospective/retrospective gap is the honest cost of replay ablation. Where it is small, one "
             "large run plus replay replaces a k-sweep.")
    L.append("- The fuzzy workload's verdicts are self-reported rubric proxies until a human reviews them "
             "(`swarmscope review`); treat its curve as a lower bound on uncertainty, not on quality.")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def reanalyze(store_url: str, json_path: str, out: str) -> None:
    """Recompute every analysis section from the store (no model calls) and rewrite the report.

    Uses the per-swarm completion records to find run ids. The retrospective
    curve is pooled over all runs at the largest k so it averages the same
    tasks as the prospective points.
    """
    from swarmscope.attribution.graph import LineageGraph
    from swarmscope.evaluation.ablation import fit_saturation, quantiles, wilson
    from swarmscope.store.base import open_store

    with open(json_path) as f:
        res = json.load(f)
    store = open_store(store_url)
    ks = res["ks"]
    for name, w in res["workloads"].items():
        wl = WORKLOADS[name]
        tag = f"{res['solver']}:{w['model']}" if w.get("model") else res["solver"]
        runs: dict[tuple[int, int], dict[str, Any]] = {}
        for r in range(res["reps"]):
            for k in ks:
                seed = k * 10 + r  # a.seed == 0 for the published runs
                rec = store.calibration_get(f"vrun:{tag}:{name}-k{k}-s{seed}")
                if rec:
                    runs[(k, r)] = rec
        # pooled retrospective over the max-k runs
        kmax = max(ks)
        pooled: dict[int, list[float]] = {k: [] for k in ks}
        pooled_cost: dict[int, list[float]] = {k: [] for k in ks}
        fidelity_ok = True
        per_task = []
        for r in range(res["reps"]):
            rec = runs.get((kmax, r))
            if not rec:
                continue
            task = wl.tasks[r % len(wl.tasks)]
            h = ReplayHarness(store, rec["run_id"], wl.consolidate, name=f"{wl.name}-consolidate")
            curve = ablate(h, lambda out, t=task: wl.score(t, out), ks=ks, samples=40, seed=r, success_threshold=0.99)
            fidelity_ok = fidelity_ok and bool(curve.fidelity and curve.fidelity.identical)
            for pt in curve.points:
                pooled[pt.k] += [1.0] * pt.successes + [0.0] * (pt.n_samples - pt.successes)
                pooled_cost[pt.k].append(pt.cost_usd_mean or 0.0)
            g = LineageGraph(store.events(rec["run_id"]))
            if wl.check is None:  # no per-agent verdicts on this workload (human-reviewed): not measured
                per_task.append({"task": task.task_id, "agents": len(h.agent_ids), "accepted": None,
                                 "agent_rate": None, "score": float(rec["score"])})
            else:
                acc = g.accepted_artifacts([wl.verdict_source], include_inferred=True)
                per_task.append({"task": task.task_id, "agents": len(h.agent_ids), "accepted": len(acc),
                                 "agent_rate": len(acc) / max(1, len(h.agent_ids)), "score": float(rec["score"])})
        points = []
        for k in ks:
            xs = pooled[k]
            if not xs:
                continue
            succ = int(sum(xs))
            lo, hi = wilson(succ, len(xs))
            points.append({"k": k, "n_samples": len(xs), "successes": succ, "p_success": succ / len(xs),
                           "ci_low": lo, "ci_high": hi, "score_mean": succ / len(xs), "score_quantiles": quantiles(xs),
                           "cost_usd_mean": sum(pooled_cost[k]) / max(1, len(pooled_cost[k]))})
        fit = fit_saturation([p["k"] for p in points], [p["successes"] for p in points], [p["n_samples"] for p in points])
        pmax = max((p["p_success"] for p in points), default=0.0)
        knee = next((p["k"] for p in points if p["ci_high"] >= 0.95 * pmax and p["p_success"] >= 0.95 * pmax), None)
        w["ablation"] = {"points": points, "knee_k": knee, "notes": [], "replayable": True,
                         "fidelity": {"identical": fidelity_ok, "gap": 0.0 if fidelity_ok else None},
                         "fit": {"q": fit.q, "knee_k": fit.knee_k, "log_likelihood": fit.log_likelihood}}
        w["per_task"] = per_task
        w["retro_pooled_runs"] = len(per_task)
    with open(json_path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    write_report(res, out)
    store.close()


def regenerate(json_path: str, out: str) -> None:
    """Rewrite the markdown report from a saved results JSON (no model calls)."""
    with open(json_path) as f:
        write_report(json.load(f), out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--regenerate":
        regenerate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "docs/results.md")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--reanalyze":
        reanalyze(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "docs/results.md")
        sys.exit(0)
    sys.exit(main())
