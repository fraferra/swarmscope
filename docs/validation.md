# Validation protocol: reference workloads

The published value-vs-k curves are the project's credibility artifact. This
document fixes the protocol so the numbers are comparable across workloads and
over time. The harness is `validation/` (`python -m validation.run`); the
latest output is [results.md](results.md). With the default `simulated` solver
the whole protocol runs in ~10 s and validates the *methodology*; the
`openai` solver runs the same protocol against a real model.

## Three workloads, spanning the acceptance-signal spectrum

| # | Workload | Verdict source | Expected behaviour |
|---|---|---|---|
| 1 | **Objective verifier** — competitive-programming tasks with hidden tests, or Lean-checked lemmas | `verifier`, confidence 1.0 | best case; use it to establish that ablation curves are trustworthy where ground truth exists |
| 2 | **Semi-objective** — multi-hop research QA with a gold answer set | `judge` (LLM vs. gold), report judge error on a labelled subsample | measurable judge error; curves carry that error as an extra CI term |
| 3 | **Fuzzy** — open-ended synthesis/report with human acceptance | `human` via `swarmscope review` | the realistic enterprise case; methods will look worst here. Report it anyway. |

## Procedure per workload

1. Instrument the swarm with `@sdk.agent` / an adapter and wrap the aggregation
   step with `@sdk.consolidator(check_determinism=True)`.
2. Run at k ∈ {1, 5, 25, 100, 500} with fixed seeds where the stack allows
   (prospective dose-response). Each k at least 5 independent runs; more where
   P(success) is near 0 or 1.
3. For the k=500 run, produce the retrospective ablation curve
   (`swarmscope ablate --unit agent --samples 50`) and the group-level Shapley
   with CIs. Compare the ablation curve at k ∈ {1, 5, 25, 100} with the
   prospective runs: the gap is the replay-fidelity error, reported on the
   curve.
4. Record online proxies for every run and fit `Calibration` per workload
   class. Store with `Calibration.save`.
5. Run the dedup gate experiment (`ExperimentConfig`) and report per-arm
   accepted artifacts, tokens, and novel-claim rate. If the gate hurts, say so.

## What gets published

- P(success) vs k with 95% CIs (never the mean alone), per workload.
- Waste ratio and cost-per-accepted-artifact **distribution**, per verdict source.
- Dedup hit rate over time, judge calls per claim, and the gated/ungated comparison.
- Replay-fidelity gap and unknown-lineage fraction for every run.
- The knee. If most workloads saturate at k≈50, that is the headline.
