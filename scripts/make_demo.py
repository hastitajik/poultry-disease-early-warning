"""Build a demo database so the panel can be seen before real clips are processed.

    python scripts/make_demo.py            # writes demo/ (config + data)
    pews serve --config demo/configs/farm.yaml

It analyses one real sample frame (distribution + snapshot) and fills four days
of SIMULATED activity values with a daily rhythm, then simulates a morning
where activity drops and birds crowd at the walls. Nothing here is real data.
"""
from __future__ import annotations

import math
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from poultry_ews.config import load_config  # noqa: E402
from poultry_ews.pipeline import Pipeline  # noqa: E402
from poultry_ews.store import iso  # noqa: E402


def main(out: Path = REPO / "demo"):
    if out.exists():
        shutil.rmtree(out)
    (out / "configs" / "cameras").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "rules.yaml", out / "configs" / "rules.yaml")
    cal = out / "configs" / "cameras" / "cam1.yaml"
    shutil.copy(REPO / "configs" / "cameras" / "cam1.example.yaml", cal)
    farm = (REPO / "configs" / "farm.example.yaml").read_text(encoding="utf-8")
    farm = farm.replace('data_dir: "data"', f'data_dir: "{(out / "data").as_posix()}"')
    farm = farm.replace('"configs/cameras/cam1.example.yaml"', f'"{cal.as_posix()}"')
    (out / "configs" / "farm.yaml").write_text(farm, encoding="utf-8")
    cfg = load_config(out / "configs" / "farm.yaml")
    pipe = Pipeline(cfg)
    st = pipe.store
    rnd = random.Random(7)

    now = datetime(2025, 12, 9, 7, 0)
    start = now - timedelta(days=4)

    def act(ts: datetime) -> float:
        h = ts.hour + ts.minute / 60
        day = 0.10 + 0.12 * max(0.0, math.sin((h - 5) / 16 * math.pi)) if 5 <= h <= 21 else 0.05
        return max(0.01, day + rnd.gauss(0, 0.01))

    ts = start
    while ts < now - timedelta(minutes=30):
        age = cfg.age_days(ts)
        for name, val in (("activity_index", act(ts)), ("resource_frac", 0.31 + rnd.gauss(0, 0.02)),
                          ("periphery_frac", 0.27 + rnd.gauss(0, 0.02))):
            st.put_metric(cfg.flock_id, "pen1", ts, age, name, val)
        ts += timedelta(minutes=5)

    # the real sample frame -> occupancy map + snapshot for the last slot
    sample = REPO / "data" / "samples" / "cam1_2025-12-09_070206.jpg"
    # last 30 minutes: simulated trouble (activity down, birds at the walls)
    for i in range(6):
        t = now - timedelta(minutes=30) + timedelta(minutes=5 * i)
        age = cfg.age_days(t)
        pipe.process_clip("cam1", sample, t)  # computes distribution from the frame
        for name, val in (("activity_index", act(t) * 0.35), ("periphery_frac", 0.42 + 0.01 * i),
                          ("resource_frac", 0.20)):
            from poultry_ews.baseline import compute_baseline
            b = compute_baseline(st, cfg.baseline, cfg.flock_id, "pen1", name, t, age)
            st.put_metric(cfg.flock_id, "pen1", t, age, name, val, b.median, b.mad, b.z(val), b.n)
        pipe.alerts.evaluate("pen1", t, [f"slots/{t:%Y%m%d}/{t:%H%M}_cam1.jpg"])

    # a still-bird candidate near the right drinker
    cid = st.add_candidate("pen1", now - timedelta(minutes=150), 2.05, 3.55, 0.03, 150, True)
    cand = dict(st.q1("SELECT * FROM candidates WHERE id=?", (cid,)))
    cand["last_slot"] = iso(now)
    rows = st.camera_results("pen1", now - timedelta(minutes=5))
    snaps = pipe._annotate_candidate("pen1", now - timedelta(minutes=5), cand, rows)
    eid = pipe.alerts.raise_mortality_candidate("pen1", now - timedelta(minutes=5), cand, snaps)
    st.update_candidate(cid, status="alerted", event_id=eid, last_slot=iso(now - timedelta(minutes=5)))
    st.put_environment("pen1", now - timedelta(minutes=20), 31.4, 30)
    print(f"demo ready: pews serve --config {out / 'configs' / 'farm.yaml'}")


if __name__ == "__main__":
    main()
