import json
import subprocess
import sys

import pytest

import swarmscope as ss
from swarmscope.surfaces.cli import main


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "cli.db"
    sdk = ss.Swarmscope(f"sqlite:///{path}", flush_interval=0.01)

    @sdk.consolidator
    def consolidate(contribs):
        return max((c.value for c in contribs), default=0)

    with sdk.run("cli-run") as run:
        contribs = []
        for g in range(2):
            with sdk.group(f"g{g}"):
                with sdk.agent_scope(role="w") as a:
                    sdk.generation(model="gpt-4o", input_tokens=100, output_tokens=50)
                    sdk.claim("try thing", kind="approach")
                    art = sdk.artifact(g)
                    if g == 1:
                        sdk.verdict(art, status="accepted", source="verifier")
                    contribs.append(ss.Contribution(a, g, f"g{g}"))
        consolidate(contribs)
    sdk.close()
    return str(path), run.run_id


def test_runs_inspect_cost_waste_dedup(db, capsys):
    path, rid = db
    main(["--store", path, "runs"])
    assert rid in capsys.readouterr().out
    main(["--store", path, "inspect", rid])
    out = capsys.readouterr().out
    assert "unknown lineage: 0.0%" in out and "lineage:" in out
    main(["--store", path, "cost", "--by", "group", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert set(data["rows"]) == {"g0", "g1"}
    main(["--store", path, "waste", rid])
    out = capsys.readouterr().out
    assert "waste ratio (tokens): 50.0%" in out
    main(["--store", path, "dedup", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["claims"] == 2


def test_export_jsonl(db, tmp_path, capsys):
    path, rid = db
    out = tmp_path / "e.jsonl"
    main(["--store", path, "export", rid, "--out", str(out)])
    lines = out.read_text().splitlines()
    assert len(lines) >= 8 and all(json.loads(l)["run_id"] == rid for l in lines)
    capsys.readouterr()
    main(["--store", path, "export", rid, "--format", "otel-json"])
    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first["attributes"]["swarm.run_id"] == rid


def test_ablate_cli(db, capsys):
    path, rid = db
    main(["--store", path, "ablate", rid, "--consolidator", "tests.helpers:consolidate",
          "--scorer", "tests.helpers:score", "--ks", "1,2", "--samples", "10", "--shapley", "--unit", "group",
          "--json"])
    data = json.loads(capsys.readouterr().out)
    assert [p["k"] for p in data["ablation"]["points"]] == [1, 2]
    assert data["ablation"]["points"][-1]["p_success"] == 1.0
    assert {v["unit"] for v in data["shapley"]["values"]} == {"g0", "g1"}
    # stored for the dashboard
    st = ss.SQLiteStore(path)
    assert st.calibration_get(f"ablation:{rid}")["points"]
    st.close()


def test_review_cli(db, monkeypatch, capsys):
    path, rid = db
    answers = iter(["a"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    main(["--store", path, "review", rid])
    assert "1 artifacts to review" in capsys.readouterr().out
    st = ss.SQLiteStore(path)
    vs = st.events(rid, types=["verdict"])
    assert any(v.source == "human" and v.status == "accepted" for v in vs)
    st.close()


def test_entrypoint_runs():
    r = subprocess.run([sys.executable, "-m", "swarmscope.surfaces.cli", "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and ss.__version__ in r.stdout
