"""Reference-workload validation harness (see docs/validation.md).

Three workloads spanning the acceptance-signal spectrum, each with a pluggable
solver. The ``simulated`` solver is deterministic and free, so the whole
protocol runs in CI and produces the curves in docs/results.md; the ``openai``
solver swaps in a real model for the same protocol (``OPENAI_API_KEY``).

    python -m validation.run --solver simulated --ks 1,5,25,100,500 --reps 5
    python -m validation.run --solver openai --workload objective --ks 1,5,25
"""
