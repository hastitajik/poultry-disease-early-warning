"""Synthetic pen videos for tests: brown litter, white elliptical 'birds'."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

W, H = 640, 360       # image size; calibration maps it to a 2 m x 4 m floor (top-down, rotated)
FPS = 10


def _reflect(v: float, lo: float, hi: float) -> float:
    span = hi - lo
    m = (v - lo) % (2 * span)
    return lo + (m if m < span else 2 * span - m)


def write_video(path: Path, birds: list[dict], seconds: float = 6.0, seed: int = 0) -> Path:
    """birds: {"x","y" (px), "vx","vy" (px/s), "r" (px)} ; vx=vy=0 -> still bird."""
    rng = np.random.default_rng(seed)
    base = np.zeros((H, W, 3), np.uint8)
    base[:] = (60, 95, 130)  # litter (BGR)
    noise = rng.integers(-12, 12, (H, W, 1), dtype=np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened(), "mp4v writer not available"
    n = int(seconds * FPS)
    for i in range(n):
        t = i / FPS
        f = base.copy()
        for b in birds:
            x = _reflect(b["x"] + b.get("vx", 0) * t, 20, W - 20)
            y = _reflect(b["y"] + b.get("vy", 0) * t, 20, H - 20)
            r = b.get("r", 10)
            cv2.ellipse(f, (int(x), int(y)), (r, int(r * 0.7)), 0, 0, 360, (235, 240, 245), -1)
            cv2.ellipse(f, (int(x) + 2, int(y)), (max(2, r // 3), max(2, r // 4)), 0, 0, 360,
                        (200, 210, 220), -1)  # texture so flow can lock on
        vw.write(f)
    vw.release()
    return path


def moving_flock(n: int = 12, speed: float = 40, seed: int = 1, r: int = 10) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ang = rng.uniform(0, 2 * np.pi)
        out.append({"x": rng.uniform(60, W - 60), "y": rng.uniform(60, H - 60),
                    "vx": speed * np.cos(ang), "vy": speed * np.sin(ang), "r": r})
    return out


def still_flock(n: int = 12, seed: int = 1, r: int = 10) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"x": rng.uniform(60, W - 60), "y": rng.uniform(60, H - 60), "r": r} for _ in range(n)]


FARM_YAML = """
farm: {{name: test, data_dir: "{data}", db_file: t.sqlite}}
flock:
  id: test-flock
  placement_date: "2025-11-01"
  initial_count: 100
  bird_area_m2_by_age: {{0: 0.0075, 60: 0.0075}}   # 10x7 px ellipse at ~170 px/m
pens:
  - id: pen1
    name: Pen 1
    width_m: 2.0
    length_m: 4.0
    wall_band_m: 0.3
    corner_size_m: 0.5
    feeders: [{{x: 1.0, y: 1.0, r: 0.4}}]
    drinkers: [{{x: 1.0, y: 3.0, r: 0.4}}]
    cameras: [cam1]
cameras:
  - id: cam1
    pen: pen1
    calibration: "{cal}"
    input_dir: "{data}/raw/cam1"
    source_fps: 10
analysis:
  work_width: 640
  analysis_fps: 5
  floor_px_per_m: 80
  slot_minutes: 5
  segmenter: {{kind: color, contrast: 25, v_min: 180, s_max: 60}}
  mortality:
    alert_after_min: 15
    min_flock_activity: 0.02
    min_area_factor: 0.3
    max_area_factor: 3.0
  circadian: {{min_hours: 18, baseline_days: [1, 3]}}
baseline: {{hour_window: 1, days_back: [1, 3], min_samples: 6}}
"""

# image is 640x360 (landscape); floor 2 m (x) x 4 m (y). The camera sees the floor rotated:
# image x -> floor y, image y -> floor x.
CAL_YAML = """
camera_id: cam1
versions:
  - valid_from: "2025-01-01"
    image_size: [640, 360]
    image_points: [[0, 0], [640, 0], [640, 360], [0, 360]]
    floor_points: [[0.0, 0.0], [0.0, 4.0], [2.0, 4.0], [2.0, 0.0]]
"""


def make_config(tmp: Path):
    from poultry_ews.config import load_config
    (tmp / "configs" / "cameras").mkdir(parents=True, exist_ok=True)
    cal = tmp / "configs" / "cameras" / "cam1.yaml"
    cal.write_text(CAL_YAML)
    farm = tmp / "configs" / "farm.yaml"
    farm.write_text(FARM_YAML.format(data=(tmp / "data").as_posix(), cal=cal.as_posix()))
    repo_rules = Path(__file__).resolve().parents[1] / "configs" / "rules.yaml"
    (tmp / "configs" / "rules.yaml").write_text(repo_rules.read_text(encoding="utf-8"), encoding="utf-8")
    return load_config(farm)
