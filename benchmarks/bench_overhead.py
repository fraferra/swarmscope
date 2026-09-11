"""Overhead benchmark. Budget: <3% wall-clock, <5 MB RSS per 1k agents (in-process buffer).

Two workload modes, both run with and without instrumentation:

- ``cpu``: N agents on one thread, each burning ~2.5 ms of CPU (a hashing
  loop) and emitting 8 events — ~3,000 events per CPU-second. Hostile by
  design: a real agent emits a handful of events per *second* of wall time,
  and any background Python work is serialised against a CPU-bound main
  thread. Enforces the wall-clock budget on the SDK's main-thread cost
  (``memory://`` store).
- ``io``: N agents on a thread pool, each sleeping a few ms per step as if
  waiting on a model, emitting 8 events. This is what a swarm looks like;
  enforces the full budget (wall-clock + RSS) with the SQLite store so
  writer-thread and I/O costs are included.

Exits non-zero on budget regression so CI fails the build.

RSS is the buffer's *working set*: peak RSS sampled during the instrumented
run minus RSS just before it. Enforced with the sqlite store (events leave
the process); with ``memory://`` the store itself retains every event and
vector, so the figure is reported but not enforced.

    python benchmarks/bench_overhead.py [--agents 1000] [--work-us 2000] [--json]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time

import swarmscope as ss

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def rss_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def work(us: int) -> None:
    """Burn roughly ``us`` microseconds of CPU deterministically."""
    end = time.perf_counter() + us / 1e6
    h = b"x"
    while time.perf_counter() < end:
        h = hashlib.blake2b(h, digest_size=16).digest()


def run_uninstrumented(n: int, work_us: int, io_ms: float = 0.0, workers: int = 1) -> float:
    def agent(i: int):
        for _ in range(3):
            work(work_us // 3)
            if io_ms:
                time.sleep(io_ms / 1000)
        hashlib.sha256(b"tool").hexdigest()

    t0 = time.perf_counter()
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(agent, range(n)))
    else:
        for i in range(n):
            agent(i)
    return time.perf_counter() - t0


class RssSampler:
    def __init__(self, interval: float = 0.02):
        import threading

        self.peak = rss_mb()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(interval,), daemon=True)

    def _run(self, interval):
        while not self._stop.wait(interval):
            self.peak = max(self.peak, rss_mb())

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, rss_mb())


def run_instrumented(n: int, work_us: int, store_url: str, io_ms: float = 0.0, workers: int = 1) -> tuple[float, dict]:
    sdk = ss.Swarmscope(store_url, gate=ss.GatePolicy(mode="off"))  # dedup lookups are a feature, not overhead

    @sdk.tool
    def tool():
        return hashlib.sha256(b"tool").hexdigest()

    @sdk.agent(role="bench")
    def agent(i: int):
        for _ in range(3):
            work(work_us // 3)
            if io_ms:
                time.sleep(io_ms / 1000)
            sdk.generation(model="gpt-4o-mini", input_tokens=500, output_tokens=100)
        tool()
        sdk.claim(f"approach {i % 50}", kind="approach")
        sdk.artifact({"i": i})

    t0 = time.perf_counter()
    with sdk.run("bench"):
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(agent, range(n)))
        else:
            for i in range(n):
                agent(i)
    elapsed = time.perf_counter() - t0  # includes flush at run exit
    stats = sdk.stats
    sdk.close()
    return elapsed, stats


def bench(mode: str, agents: int, work_us: int, reps: int, store: str, io_ms: float, workers: int,
          max_overhead: float, max_rss: float) -> dict:
    kw = dict(io_ms=io_ms, workers=workers)
    base = [run_uninstrumented(agents, work_us, **kw) for _ in range(reps)]
    run_instrumented(min(agents, 200), work_us, store, **kw)  # warm imports/allocator before measuring RSS
    gc.collect()
    base_rss = rss_mb()
    peak_rss = base_rss
    inst, stats = [], {}
    for _ in range(reps):
        with RssSampler() as smp:
            t, stats = run_instrumented(agents, work_us, store, **kw)
        inst.append(t)
        peak_rss = max(peak_rss, smp.peak)
        gc.collect()
    # min-of-reps: timing noise is additive, so the minimum is the least-biased estimate of each cost
    b, i = min(base), min(inst)
    overhead = (i - b) / b
    events = stats.get("events", 0)
    rss_per_1k = (peak_rss - base_rss) / (agents / 1000)
    enforce_rss = not store.startswith("memory")  # an in-process store retains everything by design
    ok = overhead <= max_overhead and (not enforce_rss or rss_per_1k != rss_per_1k or rss_per_1k <= max_rss)
    return {
        "mode": mode, "store": store, "agents": agents, "workers": workers, "events": events, "baseline_s": b,
        "instrumented_s": i, "overhead_fraction": overhead, "per_event_us": (i - b) / max(1, events) * 1e6,
        "events_per_s": events / i if i else None, "rss_delta_mb_per_1k_agents": rss_per_1k, "rss_enforced": enforce_rss,
        "dropped": stats.get("buffer", {}).get("dropped"), "pass": ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpu", "io", "both"], default="both")
    ap.add_argument("--agents", type=int, default=1000)
    ap.add_argument("--work-us", type=int, default=2500, help="CPU work per agent (µs) in cpu mode")
    ap.add_argument("--io-work-us", type=int, default=300, help="CPU work per agent (µs) in io mode")
    ap.add_argument("--io-ms", type=float, default=20.0, help="simulated model latency per step in io mode (real calls: 300ms+)")
    ap.add_argument("--workers", type=int, default=32, help="concurrent agents in io mode")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--store", default=None, help="override store for both modes (default: memory:// for cpu, sqlite for io)")
    ap.add_argument("--max-overhead", type=float, default=0.03)
    ap.add_argument("--max-rss-mb-per-1k", type=float, default=5.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    results = []
    modes = ["cpu", "io"] if a.mode == "both" else [a.mode]
    for mode in modes:
        if mode == "cpu":
            store = a.store or "memory://"
            r = bench(mode, a.agents, a.work_us, a.reps, store, 0.0, 1, a.max_overhead, a.max_rss_mb_per_1k)
        else:
            store = a.store or f"sqlite:///{os.path.join(os.environ.get('TMPDIR', '/tmp'), f'swarmscope-bench-{os.getpid()}.db')}"
            r = bench(mode, a.agents, a.io_work_us, a.reps, store, a.io_ms, a.workers, a.max_overhead, a.max_rss_mb_per_1k)
        results.append(r)
        if not a.json:
            print(f"[{mode}] agents={r['agents']} workers={r['workers']} events={r['events']} store={store.split('/')[0]} "
                  f"baseline={r['baseline_s']:.3f}s instrumented={r['instrumented_s']:.3f}s "
                  f"overhead={r['overhead_fraction']:.2%} ({r['per_event_us']:.1f} µs/event @ {r['events_per_s']:.0f} events/s) "
                  f"rss=+{r['rss_delta_mb_per_1k_agents']:.2f} MB/1k agents{'' if r['rss_enforced'] else ' (in-process store; not enforced)'} "
                  f"dropped={r['dropped']} -> {'PASS' if r['pass'] else 'FAIL'}")
    if a.json:
        print(json.dumps(results, indent=2))
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
