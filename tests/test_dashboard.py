import json
import urllib.request

import swarmscope as ss
from swarmscope.surfaces.dashboard import serve


def test_dashboard_endpoints(tmp_path):
    path = tmp_path / "d.db"
    sdk = ss.Swarmscope(f"sqlite:///{path}", flush_interval=0.01)
    with sdk.run("dash") as run:
        with sdk.agent_scope(role="w"):
            sdk.generation(model="gpt-4o", input_tokens=10, output_tokens=10)
            art = sdk.artifact("x")
            sdk.verdict(art, status="accepted", source="human")
    sdk.close()
    httpd = serve(f"sqlite:///{path}", port=0, block=False)
    port = httpd.server_address[1]
    try:
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "swarmscope" in html and "unknown lineage" in html
        runs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs").read())
        assert runs[0]["run_id"] == run.run_id
        for what in ("cost", "waste", "proxies", "ablation", "events"):
            body = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/run/{run.run_id}/{what}").read())
            assert body is not None
        waste = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/run/{run.run_id}/waste").read())
        assert waste["defined"] and waste["waste_ratio"] == 0.0
        rep = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/reputation?unit=agent").read())
        assert rep["leaderboard"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_bandit_dashboard(tmp_path):
    path = tmp_path / "b.db"
    sdk = ss.Swarmscope(f"sqlite:///{path}", flush_interval=0.01)
    with sdk.run("b"):
        for i in range(3):
            with sdk.request(f"translate clause {i}", kind="t") as adv:
                with sdk.agent_scope(role="linguist", model="m"):
                    sdk.verdict(sdk.artifact(i), status="accepted" if i < 2 else "rejected", source="verifier")
    sdk.close()
    httpd = serve(f"sqlite:///{path}", port=0, block=False)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        html = urllib.request.urlopen(f"{base}/reputation").read().decode()
        assert "Learnt success probabilities" in html and "Ask the bandit" in html
        assert 'href="/reputation"' in urllib.request.urlopen(f"{base}/").read().decode()
        lb = json.loads(urllib.request.urlopen(f"{base}/api/reputation?unit=agent").read())["leaderboard"]
        assert lb[0]["route"] == ["linguist|m|"] and lb[0]["global_successes"] == 2.0 and lb[0]["global_failures"] == 1.0
        tl = json.loads(urllib.request.urlopen(f"{base}/api/reputation/outcomes").read())
        assert tl["total"] == 3 and tl["requests"] == 3 and all("ts" in o for o in tl["outcomes"])
        adv = json.loads(urllib.request.urlopen(f"{base}/api/reputation/advise?q=translate+this+clause&kind=t&policy=greedy").read())
        assert not adv["cold_start"] and adv["routes"][0]["route"] == ["linguist|m|"]
        assert adv["routes"][0]["local_successes"] > 0
        cold = json.loads(urllib.request.urlopen(f"{base}/api/reputation/advise?q=x&kind=other&unit=sequence").read())
        assert cold["routes"]  # known routes are candidates even for an unseen kind
    finally:
        httpd.shutdown()
        httpd.server_close()
