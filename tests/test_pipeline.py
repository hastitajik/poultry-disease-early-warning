"""End-to-end tests on synthetic videos."""
from datetime import datetime, timedelta

import pytest

from poultry_ews.analyzer import analyze_clip
from poultry_ews.pipeline import Pipeline
from poultry_ews.store import iso

from synth import make_config, moving_flock, still_flock, write_video


@pytest.fixture()
def cfg(tmp_path):
    return make_config(tmp_path)


def test_activity_moving_vs_still(cfg, tmp_path):
    t = datetime(2025, 11, 20, 8, 0)
    mv = analyze_clip(cfg, "cam1", write_video(tmp_path / "m.mp4", moving_flock()), t)
    st = analyze_clip(cfg, "cam1", write_video(tmp_path / "s.mp4", still_flock()), t)
    a_mv = mv.moving_px / mv.bird_px
    a_st = st.moving_px / st.bird_px
    assert a_mv > 0.4, a_mv
    assert a_st < 0.05, a_st
    # occupancy found roughly 12 birds' worth of floor
    assert mv.bird_px > 0 and st.occupancy.sum() > 0


def test_still_blobs_only_for_bird_sized_still_objects(cfg, tmp_path):
    t = datetime(2025, 11, 20, 8, 0)
    birds = moving_flock(10) + [{"x": 320, "y": 180, "r": 10}]
    res = analyze_clip(cfg, "cam1", write_video(tmp_path / "d.mp4", birds), t)
    assert len(res.blobs) == 1
    b = res.blobs[0]
    # image (320,180) = floor (1.0 m, 2.0 m) with the test calibration
    assert abs(b.x_m - 1.0) < 0.1 and abs(b.y_m - 2.0) < 0.1


def test_mortality_candidate_alert(cfg, tmp_path):
    """A bird that never moves for 15 min while the flock moves -> critical event."""
    pipe = Pipeline(cfg)
    dead = {"x": 500, "y": 100, "r": 10}
    t0 = datetime(2025, 11, 20, 9, 0)
    for i in range(4):
        clip = write_video(tmp_path / f"c{i}.mp4", moving_flock(10, seed=i + 3) + [dead], seed=i)
        pipe.process_clip("cam1", clip, t0 + timedelta(minutes=5 * i))
    ev = pipe.store.q("SELECT * FROM events WHERE rule_id='mortality_candidate'")
    assert len(ev) == 1 and ev[0]["severity"] == "critical"
    cands = pipe.store.candidates("pen1", ("alerted",))
    assert len(cands) == 1 and cands[0]["still_min"] >= 15
    # annotated snapshot saved
    import json
    snaps = json.loads(ev[0]["snapshots"])
    assert snaps and (cfg.data_dir / "media" / snaps[0]).exists()


def test_moving_bird_is_not_a_candidate(cfg, tmp_path):
    pipe = Pipeline(cfg)
    t0 = datetime(2025, 11, 20, 9, 0)
    for i in range(4):
        pipe.process_clip("cam1", write_video(tmp_path / f"c{i}.mp4", moving_flock(10, seed=i + 3), seed=i),
                          t0 + timedelta(minutes=5 * i))
    assert pipe.store.q("SELECT * FROM events WHERE rule_id='mortality_candidate'") == []


def test_reprocessing_same_slot_is_idempotent(cfg, tmp_path):
    pipe = Pipeline(cfg)
    clip = write_video(tmp_path / "x.mp4", moving_flock(8, seed=4) + [{"x": 500, "y": 100, "r": 10}])
    t = datetime(2025, 11, 20, 9, 0)
    pipe.process_clip("cam1", clip, t)
    pipe.fuse_slot("pen1", t)   # e.g. a second camera's clip arrived
    pipe.fuse_slot("pen1", t)
    cands = pipe.store.q("SELECT * FROM candidates")
    assert len(cands) == 1 and cands[0]["still_min"] == 5


def _seed_history(pipe, days=3, value=0.30, name="activity_index"):
    """Fake 3 previous days of normal metric values every 5 minutes."""
    st, cfg = pipe.store, pipe.cfg
    start = datetime(2025, 11, 20)
    for d in range(days):
        for k in range(24 * 12):
            ts = start + timedelta(days=d, minutes=5 * k)
            st.put_metric(cfg.flock_id, "pen1", ts, cfg.age_days(ts), name, value + 0.01 * ((k % 5) - 2))
    return start + timedelta(days=days)


def test_activity_drop_raises_event_after_persist(cfg, tmp_path):
    pipe = Pipeline(cfg)
    day = _seed_history(pipe)
    # the flock is now completely still -> activity_index ~ 0 at 10:00, 10:05, 10:10
    for i in range(3):
        clip = write_video(tmp_path / f"s{i}.mp4", still_flock(10, seed=i), seed=i)
        pipe.process_clip("cam1", clip, day.replace(hour=10) + timedelta(minutes=5 * i))
        evs = pipe.store.q("SELECT * FROM events WHERE rule_id='activity_drop'")
        assert len(evs) == (1 if i == 2 else 0), f"after {i + 1} slots: {len(evs)}"
    e = evs[0]
    assert e["severity"] == "critical" and e["z"] < -3
    r = pipe.store.latest_metric("pen1", "activity_index")
    assert r["n_baseline"] >= 6 and abs(r["baseline_median"] - 0.30) < 0.02


def test_learning_state_without_history(cfg, tmp_path):
    from poultry_ews.alerts import signal_states
    pipe = Pipeline(cfg)
    pipe.process_clip("cam1", write_video(tmp_path / "a.mp4", moving_flock()), datetime(2025, 11, 20, 8))
    s = signal_states(cfg, pipe.store, "pen1")
    assert s["activity"]["state"] == "learning"
    assert s["activity"]["metrics"]["activity_index"]["value"] > 0.3
    assert s["mortality"]["state"] == "ok"


def test_daily_mortality_rule(cfg):
    pipe = Pipeline(cfg)
    pct = pipe.daily_mortality("pen1", "2025-11-20", deaths=1, culls=0)
    assert abs(pct - 1.0) < 1e-9   # 1 of 100 birds
    ev = pipe.store.q("SELECT * FROM events WHERE rule_id='daily_mortality'")
    assert len(ev) == 1 and ev[0]["severity"] == "critical"
    pct2 = pipe.daily_mortality("pen1", "2025-11-21", deaths=0, culls=0)
    assert pct2 == 0.0


def test_image_input_gives_distribution_only(cfg, tmp_path):
    import cv2
    pipe = Pipeline(cfg)
    clip = write_video(tmp_path / "v.mp4", still_flock())
    cap = cv2.VideoCapture(str(clip))
    ok, frame = cap.read()
    cap.release()
    assert ok
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), frame)
    pipe.process_clip("cam1", img, datetime(2025, 11, 20, 8))
    assert pipe.store.latest_metric("pen1", "activity_index")["value"] is None
    assert pipe.store.latest_metric("pen1", "bird_coverage")["value"] > 0


def test_watch_once_reads_dvr_names(cfg, tmp_path):
    """DVR naming: date folder + 'HH.MM.SS-HH.MM.SS' file name."""
    from poultry_ews.pipeline import watch
    d = cfg.camera("cam1").input_dir / "2025-11-20"
    d.mkdir(parents=True)
    write_video(d / "05.45.00-05.50.00RO000.mp4", moving_flock())
    pipe = Pipeline(cfg)
    watch(pipe, once=True)
    r = pipe.store.q1("SELECT * FROM clips")
    assert r["status"] == "done" and r["slot_ts"] == "2025-11-20T05:45:00"
    watch(pipe, once=True)  # already done -> not processed twice
    assert len(pipe.store.q("SELECT * FROM camera_results")) == 1
