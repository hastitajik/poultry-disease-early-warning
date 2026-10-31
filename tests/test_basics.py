from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from poultry_ews.alerts import rule_level
from poultry_ews.baseline import BaselineStat
from poultry_ews.config import Rule, load_config
from poultry_ews.geometry import CORNER, DRINKER, FEEDER, OPEN, WALL, FloorCanvas
from poultry_ews.signals.circadian import rhythm_metrics
from poultry_ews.signals.distribution import distribution_metrics, fuse_occupancy
from poultry_ews.signals.mortality import merge_blobs
from poultry_ews.timeparse import TimestampError, parse_clip_time, slot_start

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- timestamps
@pytest.mark.parametrize("name,expected", [
    ("20251118184503423_fps15.mp4", datetime(2025, 11, 18, 18, 45, 3, 423000)),
    # export tools append a second (export) date: the FIRST one is the recording time
    ("20251117223036384_fps15.mp4_20260425_221353.395_1.png", datetime(2025, 11, 17, 22, 30, 36, 384000)),
    ("cam1_2025-12-09_05-47-54.mp4", datetime(2025, 12, 9, 5, 47, 54)),
])
def test_parse_names(name, expected):
    assert parse_clip_time(name) == expected


def test_time_range_uses_folder_date(tmp_path):
    p = tmp_path / "2025-12-09" / "05.45.00-05.50.00RO000.h264"
    assert parse_clip_time(p) == datetime(2025, 12, 9, 5, 45, 0)
    # the export date in the name must NOT be used
    p2 = tmp_path / "2025-12-09" / "05.45.00-05.50.00RO000.h264_20260426_222529.516_1.png"
    assert parse_clip_time(p2) == datetime(2025, 12, 9, 5, 45, 0)


def test_time_range_without_folder_date_fails(tmp_path):
    with pytest.raises(TimestampError):
        parse_clip_time(tmp_path / "x" / "05.45.00-05.50.00RO000.h264")


def test_custom_regex():
    rx = r"D(?P<year>\d{4})(?P<month>\d\d)(?P<day>\d\d)T(?P<hour>\d\d)(?P<minute>\d\d)"
    assert parse_clip_time("ch01_D20251209T0545.mp4", rx) == datetime(2025, 12, 9, 5, 45)


def test_slot_start():
    assert slot_start(datetime(2025, 12, 9, 5, 47, 54), 5) == datetime(2025, 12, 9, 5, 45)


# ---------------------------------------------------------------- config
def test_example_config_loads():
    cfg = load_config(REPO / "configs" / "farm.example.yaml")
    assert cfg.camera("cam1").calibration_for(datetime(2025, 12, 9)).valid_from.month == 12
    assert cfg.camera("cam1").calibration_for(datetime(2025, 11, 20)).valid_from.month == 11
    assert len(cfg.pen("pen1").at(datetime(2025, 11, 20)).drinkers) == 9
    assert len(cfg.pen("pen1").at(datetime(2025, 12, 20)).feeders) == 4
    assert {r.signal for r in cfg.rules} == {"activity", "distribution", "mortality", "circadian"}
    assert cfg.age_days(datetime(2025, 12, 9)) == 22
    assert 0.025 < cfg.bird_area_m2(22) < 0.03


# ---------------------------------------------------------------- geometry / distribution
def _pen(cfg):
    return cfg.pen("pen1").at(datetime(2025, 12, 9))


def test_zone_map_priorities():
    cfg = load_config(REPO / "configs" / "farm.example.yaml")
    pen = _pen(cfg)
    c = FloorCanvas(pen.width_m, pen.length_m, 50)
    z = c.zone_map(pen)
    assert z[0, 0] == CORNER
    assert z[z.shape[0] // 2, 2] == WALL
    f = pen.feeders[1]
    assert z[int(f.y * 50), int(f.x * 50)] == FEEDER
    d = pen.drinkers[0]
    assert z[int(d.y * 50), int(d.x * 50)] == DRINKER
    assert (z == OPEN).any()


def test_distribution_periphery_vs_resources():
    cfg = load_config(REPO / "configs" / "farm.example.yaml")
    pen = _pen(cfg)
    c = FloorCanvas(pen.width_m, pen.length_m, 50)
    z = c.zone_map(pen)
    cov = np.ones(z.shape, bool)
    at_walls = np.where((z == WALL) | (z == CORNER), 1.0, 0.0).astype(np.float32)
    at_food = np.where((z == FEEDER) | (z == DRINKER), 1.0, 0.0).astype(np.float32)
    m1 = distribution_metrics(at_walls, cov, z, 50)
    m2 = distribution_metrics(at_food, cov, z, 50)
    assert m1["periphery_frac"] > 0.99 and m1["resource_frac"] < 0.01
    assert m2["resource_frac"] > 0.99
    uniform = distribution_metrics(np.full(z.shape, 0.3, np.float32), cov, z, 50)
    clumped = np.zeros(z.shape, np.float32)
    clumped[:20, :20] = 1
    assert uniform["dispersion_entropy"] > distribution_metrics(clumped, cov, z, 50)["dispersion_entropy"]
    assert abs(uniform["bird_coverage"] - 0.3) < 1e-6


def test_fuse_occupancy_averages_overlap():
    a = np.full((4, 4), 1.0, np.float32)
    b = np.full((4, 4), 0.0, np.float32)
    ca = np.zeros((4, 4), bool); ca[:, :3] = True
    cb = np.zeros((4, 4), bool); cb[:, 2:] = True
    occ, cov = fuse_occupancy([a, b], [ca, cb])
    assert cov.all()
    assert occ[0, 0] == 1.0 and occ[0, 3] == 0.0 and occ[0, 2] == 0.5


def test_merge_blobs():
    out = merge_blobs([(1.0, 1.0, 0.02), (1.05, 1.02, 0.03), (2.0, 2.0, 0.02)], 0.2)
    assert len(out) == 2


# ---------------------------------------------------------------- circadian
def test_rhythm_metrics():
    usual = {h: 0.3 + 0.2 * np.sin((h - 6) / 24 * 2 * np.pi) for h in range(24)}
    same = rhythm_metrics(dict(usual), usual, 18)
    assert same["rhythm_corr"] > 0.99 and abs(same["rhythm_amplitude"] - 1) < 1e-6
    flat = {h: 0.3 + 0.02 * np.sin((h - 6) / 24 * 2 * np.pi) for h in range(24)}
    assert rhythm_metrics(flat, usual, 18)["rhythm_amplitude"] < 0.2
    shifted = {h: usual[(h + 12) % 24] for h in range(24)}
    assert rhythm_metrics(shifted, usual, 18)["rhythm_corr"] < -0.9
    assert rhythm_metrics({h: 1 for h in range(10)}, usual, 18)["rhythm_corr"] is None


# ---------------------------------------------------------------- rules / baseline
def test_rule_levels():
    low = Rule("r", "activity", "m", "zscore", warn=2, crit=3, direction="low")
    assert rule_level(low, 0.1, -3.5) == "critical"
    assert rule_level(low, 0.1, -2.2) == "warning"
    assert rule_level(low, 0.1, 2.5) == "ok"
    assert rule_level(low, 0.1, None) is None
    lt = Rule("c", "circadian", "rhythm_corr", "threshold", warn=0.5, crit=0.2, op="<")
    assert rule_level(lt, 0.1, None) == "critical"
    assert rule_level(lt, 0.4, None) == "warning"
    assert rule_level(lt, 0.9, None) == "ok"
    gt = Rule("d", "mortality", "daily", "threshold", warn=0.2, crit=0.5, op=">")
    assert rule_level(gt, 0.6, None) == "critical" and rule_level(gt, 0.1, None) == "ok"


def test_baseline_z_floor():
    b = BaselineStat(median=0.30, mad=0.0, n=10)
    # zero MAD must not explode: scale floored at 5% of the median
    assert abs(b.z(0.285) - (-1.0)) < 1e-9
    assert BaselineStat(None, None, 2).z(0.3) is None
