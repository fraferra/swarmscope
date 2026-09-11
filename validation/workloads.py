"""The three reference workloads.

Each workload defines: a task set, a solver interface, a *verdict source*, and
a scorer over the consolidated output. Solvers get a seed so runs are
reproducible where the stack allows.
"""
from __future__ import annotations

import ast
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import swarmscope as ss


@dataclass
class Task:
    task_id: str
    prompt: str
    gold: Any
    meta: dict[str, Any] = field(default_factory=dict)


class Solver(Protocol):
    name: str

    def solve(self, workload: "Workload", task: Task, rng: random.Random, sdk: ss.Swarmscope) -> Any: ...


@dataclass
class Workload:
    name: str
    verdict_source: str  # verifier | judge | human
    tasks: list[Task]
    #: acceptance check for one artifact; None means "needs a human"
    check: Callable[[Task, Any], bool] | None
    #: how the swarm's outputs are consolidated (recorded verbatim for replay)
    consolidate: Callable[[list[ss.Contribution]], Any]
    #: score of the consolidated output: True/False or float
    score: Callable[[Task, Any], float | bool]
    #: per-agent success probability for the simulated solver, by "approach"
    sim_approaches: dict[str, float]


# --------------------------------------------------------------------------- 1. objective
_OBJECTIVE_TASKS = [
    Task("sum-evens", "Return a Python function f(xs) that returns the sum of the even numbers in xs.",
         [([1, 2, 3, 4], 6), ([], 0), ([2, 2, 2], 6), ([-2, 1], -2)]),
    Task("second-largest", "Return a Python function f(xs) returning the second largest distinct value (None if <2).",
         [([1, 3, 2], 2), ([5, 5, 5], None), ([9], None), ([1, 2, 3, 4], 3)]),
    Task("balanced", "Return a Python function f(s) that returns True iff the brackets ()[]{} in s are balanced.",
         [("()", True), ("([)]", False), ("", True), ("{[()]}", True), ("(", False)]),
    Task("rle", "Return a Python function f(s) that run-length encodes s as e.g. 'aaab' -> 'a3b1'.",
         [("aaab", "a3b1"), ("", ""), ("abc", "a1b1c1"), ("zzzz", "z4")]),
]


def _run_tests(task: Task, code: str) -> bool:
    """Objective verifier: execute candidate code against hidden tests (sandboxed namespace)."""
    if not isinstance(code, str) or "def f" not in code:
        return False
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False
        ns: dict[str, Any] = {"__builtins__": {"len": len, "range": range, "sorted": sorted, "set": set, "list": list,
                                               "sum": sum, "max": max, "min": min, "str": str, "int": int,
                                               "enumerate": enumerate, "reversed": reversed, "abs": abs, "None": None}}
        exec(compile(tree, "<cand>", "exec"), ns)  # noqa: S102 - validation harness, curated builtins
        f = ns.get("f")
        return all(f(inp) == exp for inp, exp in task.gold)
    except Exception:
        return False


# Simulated "approaches": each has a success probability; the swarm samples approaches.
_OBJ_SOLUTIONS = {
    "sum-evens": {"filter": "def f(xs):\n    return sum(x for x in xs if x % 2 == 0)",
                  "loop": "def f(xs):\n    t = 0\n    for x in xs:\n        if x % 2 == 0:\n            t += x\n    return t",
                  "wrong-odd": "def f(xs):\n    return sum(x for x in xs if x % 2)",},
    "second-largest": {"sorted-set": "def f(xs):\n    u = sorted(set(xs))\n    return u[-2] if len(u) > 1 else None",
                       "wrong-dup": "def f(xs):\n    s = sorted(xs)\n    return s[-2] if len(s) > 1 else None"},
    "balanced": {"stack": "def f(s):\n    st = []\n    p = {')': '(', ']': '[', '}': '{'}\n    for c in s:\n        if c in '([{':\n            st.append(c)\n        elif c in p:\n            if not st or st.pop() != p[c]:\n                return False\n    return not st",
                 "wrong-count": "def f(s):\n    return s.count('(') == s.count(')')"},
    "rle": {"scan": "def f(s):\n    out = ''\n    i = 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out += s[i] + str(j - i)\n        i = j\n    return out",
            "wrong-single": "def f(s):\n    return ''.join(c + '1' for c in s)"},
}


def _obj_consolidate(contribs):
    # pick any accepted candidate; the consolidator sees (code, accepted) pairs
    for c in contribs:
        v = c.value
        if isinstance(v, dict) and v.get("accepted"):
            return v["code"]
    return None


OBJECTIVE = Workload(
    name="objective", verdict_source="verifier", tasks=_OBJECTIVE_TASKS, check=_run_tests,
    consolidate=_obj_consolidate, score=lambda task, out: out is not None and _run_tests(task, out),
    sim_approaches={"good": 0.35, "bad": 0.0, "slow": 0.2},
)

# --------------------------------------------------------------------------- 2. semi-objective
_QA_TASKS = [
    Task("capital-river", "Which river flows through the capital of the country whose largest city is Istanbul?",
         "Ankara Çayı", {"aliases": ["ankara river", "ankara çayı", "ankara cayi"]}),
    Task("author-birth", "In which century was the author of 'The Name of the Rose' born?", "20th",
         {"aliases": ["20th", "twentieth", "1900s"]}),
    Task("element-discovery", "What is the atomic number of the element named after the discoverer of radium's home country?",
         "84", {"aliases": ["84", "polonium 84"]}),
    Task("moon-count", "How many moons does the planet with the Great Red Spot have that are larger than Mercury?", "1",
         {"aliases": ["1", "one", "ganymede"]}),
]


def _qa_judge(task: Task, answer: Any) -> bool:
    """Semi-objective: gold answer set with aliases. With the ``openai`` solver an LLM
    judge is used instead and its error measured on this alias set."""
    if not isinstance(answer, str):
        return False
    a = answer.strip().lower()
    return any(al in a for al in task.meta["aliases"])


def _qa_consolidate(contribs):
    from collections import Counter

    votes = Counter(str(c.value).strip().lower() for c in contribs if c.value)
    return votes.most_common(1)[0][0] if votes else None


SEMI_OBJECTIVE = Workload(
    name="semi-objective", verdict_source="judge", tasks=_QA_TASKS, check=_qa_judge,
    consolidate=_qa_consolidate, score=lambda task, out: out is not None and _qa_judge(task, out),
    sim_approaches={"hop-correct": 0.4, "hop-partial": 0.0, "guess": 0.05},
)

# --------------------------------------------------------------------------- 3. fuzzy
_SYNTH_TASKS = [
    Task("brief-swarm", "Write a 150-word brief on when adding agents to a swarm stops paying off.", None,
         {"rubric": ["mentions diminishing returns", "gives a concrete threshold or method", "under 200 words"]}),
    Task("brief-dedup", "Write a 150-word brief on why hard-blocking duplicate work can hurt swarm search.", None,
         {"rubric": ["mentions diversity", "mentions divergence after early steps", "under 200 words"]}),
]


def _synth_consolidate(contribs):
    # longest-common-rubric heuristic stands in for an editor merging drafts
    drafts = [c.value for c in contribs if isinstance(c.value, dict)]
    if not drafts:
        return None
    return max(drafts, key=lambda d: d.get("rubric_hits", 0))


def _synth_score(task: Task, out: Any) -> float:
    # fuzzy: the score is the *human* rubric fraction when reviewed, else the self-reported
    # rubric hits (clearly labelled as proxy in results)
    if not isinstance(out, dict):
        return 0.0
    return out.get("human_score", out.get("rubric_hits", 0) / len(task.meta["rubric"]))


FUZZY = Workload(
    name="fuzzy", verdict_source="human", tasks=_SYNTH_TASKS, check=None,
    consolidate=_synth_consolidate, score=_synth_score,
    sim_approaches={"rubric-aware": 0.2, "generic": 0.05},
)

WORKLOADS = {w.name: w for w in (OBJECTIVE, SEMI_OBJECTIVE, FUZZY)}


# --------------------------------------------------------------------------- solvers
class SimulatedSolver:
    """Deterministic stand-in for a model: samples an approach, succeeds with its probability.

    Heavy-tailed by construction (a few approaches carry all the success), which is
    the regime the plan says swarm search lives in.
    """

    name = "simulated"

    def solve(self, workload, task, rng, sdk):
        approach = rng.choice(list(workload.sim_approaches))
        p = workload.sim_approaches[approach]
        sdk.generation(model="sim-model", input_tokens=rng.randint(300, 900), output_tokens=rng.randint(50, 400),
                       cost_usd=0.001)
        hit = sdk.claim(f"{task.task_id}: {approach}", kind="approach", metadata={"technique": approach})
        # Honour the advisory gate (in the gated arm only): skip most duplicate approaches.
        if hit.suggest_skip and rng.random() < 0.7:
            hit.suppress("simulated agent followed dedup advice")
            return None
        ok = rng.random() < p
        if workload.name == "objective":
            sols = _OBJ_SOLUTIONS[task.task_id]
            good = [k for k in sols if not k.startswith("wrong")]
            bad = [k for k in sols if k.startswith("wrong")]
            code = sols[rng.choice(good)] if ok else sols[rng.choice(bad)]
            return {"code": code, "approach": approach}
        if workload.name == "semi-objective":
            # wrong answers are scattered (hallucinations vary), so a plurality vote can recover the gold
            return task.gold if ok else f"wrong-{rng.randint(0, 10**6)}"
        hits = len(task.meta["rubric"]) if ok else rng.randint(0, 1)
        return {"draft": f"[{approach}] draft", "rubric_hits": hits}


class OpenAISolver:
    """Real model. Needs OPENAI_API_KEY and the openai extra."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI
        from swarmscope.adapters.openai_transport import instrument_openai

        self.model = model
        self._client = OpenAI()
        self._instrumented = False
        self._instrument = instrument_openai

    def solve(self, workload, task, rng, sdk):
        if not self._instrumented:
            self._instrument(sdk, self._client)
            self._instrumented = True
        sdk.claim(f"{task.task_id}: attempt", kind="approach")
        sys_prompt = {
            "objective": "Reply with only Python code defining f. No imports, no prose.",
            "semi-objective": "Answer with the shortest possible phrase.",
            "fuzzy": "Write the brief. Plain prose.",
        }[workload.name]
        r = self._client.chat.completions.create(model=self.model, temperature=1.0, seed=rng.randint(0, 2**31),
                                                 messages=[{"role": "system", "content": sys_prompt},
                                                           {"role": "user", "content": task.prompt}])
        text = (r.choices[0].message.content or "").strip()
        if workload.name == "objective":
            code = text.strip("`").removeprefix("python").strip()
            return {"code": code, "approach": "llm"}
        if workload.name == "semi-objective":
            return text
        hits = sum(1 for r_ in task.meta["rubric"] if any(w in text.lower() for w in r_.split()[1:2]))
        return {"draft": text, "rubric_hits": hits}


SOLVERS: dict[str, Callable[[], Solver]] = {"simulated": SimulatedSolver, "openai": OpenAISolver}
