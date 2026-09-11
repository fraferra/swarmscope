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
