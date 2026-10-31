from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from poultry_ews.api.app import create_app
from poultry_ews.pipeline import Pipeline

from synth import make_config, moving_flock, write_video


def test_api_flow(tmp_path):
    cfg = make_config(tmp_path)
    pipe = Pipeline(cfg)
    dead = {"x": 500, "y": 100, "r": 10}
    t0 = datetime(2025, 11, 20, 9, 0)
    for i in range(4):
        pipe.process_clip("cam1", write_video(tmp_path / f"c{i}.mp4",
                                              moving_flock(10, seed=i + 3) + [dead], seed=i),
                          t0 + timedelta(minutes=5 * i))
    c = TestClient(create_app(cfg, pipe))

    assert "پایش" in c.get("/").text
    ov = c.get("/api/overview").json()
    sig = ov["pens"][0]["signals"]
    assert set(sig) == {"activity", "distribution", "mortality", "circadian"}
    assert sig["mortality"]["state"] == "critical"
    assert sig["mortality"]["candidates_alerted"] == 1

    m = c.get("/api/pens/pen1/metrics?names=activity_index,resource_frac&hours=2").json()
    assert len(m["series"]["activity_index"]) == 4

    hm = c.get("/api/pens/pen1/heatmap").json()
    assert hm["grid"] and len(hm["candidates"]) == 1 and hm["feeders"]

    cams = c.get("/api/pens/pen1/cameras").json()
    assert cams[0]["image"] and c.get(cams[0]["image"]).status_code == 200

    evs = c.get("/api/events?pen=pen1").json()
    ev = next(e for e in evs if e["rule_id"] == "mortality_candidate")
    assert ev["details"]["actions"]
    assert c.get("/media/" + ev["snapshots"][0]).status_code == 200

    # vet confirms the bird was dead -> candidate + event closed
    cid = ev["details"]["candidate_id"]
    r = c.post(f"/api/candidates/{cid}/resolve", json={"outcome": "dead", "by": "دکتر", "note": "نکروپسی"})
    assert r.json()["status"] == "confirmed_dead"
    assert all(e["id"] != ev["id"] for e in c.get("/api/events?pen=pen1&status=open").json())
    ov = c.get("/api/overview").json()
    assert ov["pens"][0]["signals"]["mortality"]["candidates_alerted"] == 0

    r = c.post("/api/daily-log", json={"pen": "pen1", "date": "2025-11-20", "deaths": 1})
    assert abs(r.json()["daily_mortality_pct"] - 1.0) < 1e-9
    assert c.post("/api/environment", json={"pen": "pen1", "temp_c": 31.4, "rh": 30}).json()["ok"]
    assert c.get("/api/pens/nope/metrics").status_code == 404


def test_basic_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("PEWS_USER", "farm")
    monkeypatch.setenv("PEWS_PASSWORD", "secret")
    cfg = make_config(tmp_path)
    c = TestClient(create_app(cfg))
    assert c.get("/api/overview").status_code == 401
    assert c.get("/api/overview", auth=("farm", "secret")).status_code == 200
