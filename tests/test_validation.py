"""The validation harness runs end to end on the simulated solver (fast) and reports every section."""
import pytest

from validation.run import main


def test_validation_protocol_smoke(tmp_path):
    out = tmp_path / "results.md"
    js = tmp_path / "results.json"
    assert main(["--ks", "1,5,10", "--reps", "2", "--store", "memory://", "--out", str(out), "--json", str(js)]) == 0
    text = out.read_text()
    for section in ("Workload: objective", "Workload: semi-objective", "Workload: fuzzy", "Group Shapley",
                    "Dedup gate experiment", "prospective P(success)", "Replay fidelity"):
        assert section in text
    import json

    data = json.loads(js.read_text())
    for w in data["workloads"].values():
        assert [p["k"] for p in w["ablation"]["points"]] == [1, 5, 10]
        assert w["ablation"]["fidelity"]["identical"]
        assert set(w["dedup_arms"]) >= {"gated", "ungated"}
