"""Configuration loading (farm.yaml, camera calibrations, rules.yaml)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


DEFAULT_ANALYSIS: dict[str, Any] = {
    "work_width": 1152,
    "analysis_fps": 2.0,
    "floor_px_per_m": 100,
    "slot_minutes": 5,
    "segmenter": {"kind": "color", "contrast": 18, "v_min": 140, "s_max": 120,
                  "weights": "models/broiler-seg.pt", "conf": 0.35,
                  "external_suffix": ".dets.jsonl"},
    "activity": {"moving_speed_m_s": 0.04},
    "mortality": {"still_speed_m_s": 0.015, "max_moving_frac": 0.05,
                  "min_presence_frac": 0.90, "min_area_factor": 0.4,
                  "max_area_factor": 2.5, "match_radius_m": 0.20,
                  "alert_after_min": 120, "min_flock_activity": 0.03},
    "circadian": {"min_hours": 18, "baseline_days": [2, 7]},
}

DEFAULT_BASELINE = {"hour_window": 1, "days_back": [1, 4],
                    "other_flocks_age_window": 2, "min_samples": 6}


@dataclass
class Circle:
    x: float
    y: float
    r: float


@dataclass
class PenConfig:
    id: str
    name: str
    width_m: float
    length_m: float
    wall_band_m: float = 0.3
    corner_size_m: float = 0.6
    feeders: list[Circle] = field(default_factory=list)
    drinkers: list[Circle] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    # (valid_from, feeders, drinkers) — equipment moves during the flock
    layouts: list[tuple[date, list[Circle], list[Circle]]] = field(default_factory=list)

    def at(self, when: date | datetime) -> "PenConfig":
        """Copy of this pen with the equipment layout valid at `when`."""
        if not self.layouts:
            return self
        d = when.date() if isinstance(when, datetime) else when
        valid = [lay for lay in self.layouts if lay[0] <= d] or [min(self.layouts, key=lambda x: x[0])]
        _, feeders, drinkers = max(valid, key=lambda x: x[0])
        return PenConfig(self.id, self.name, self.width_m, self.length_m, self.wall_band_m,
                         self.corner_size_m, feeders, drinkers, self.cameras, self.layouts)


@dataclass
class CalibrationVersion:
    valid_from: date
    image_size: tuple[int, int]               # (w, h) full resolution
    image_points: np.ndarray                  # (N,2) px
    floor_points: np.ndarray                  # (N,2) metres
    roi: np.ndarray                           # polygon px
    exclude: list[Any]                        # polygons (N,2) or ("circle", cx, cy, r)


@dataclass
class CameraConfig:
    id: str
    pen: str
    calibration_file: Path
    input_dir: Path
    source_fps: float = 15.0
    filename_regex: str = ""
    versions: list[CalibrationVersion] = field(default_factory=list)

    def calibration_for(self, when: date | datetime) -> CalibrationVersion:
        d = when.date() if isinstance(when, datetime) else when
        valid = [v for v in self.versions if v.valid_from <= d]
        if not valid:
            raise ValueError(f"camera {self.id}: no calibration valid on {d}")
        return max(valid, key=lambda v: v.valid_from)


@dataclass
class Rule:
    id: str
    signal: str
    metric: str
    kind: str                    # zscore | threshold
    warn: float
    crit: float
    direction: str = "low"       # zscore only
    op: str = ">"                # threshold only
    persist: int = 1
    cooldown_min: int = 60


@dataclass
class Config:
    root: Path
    farm: dict
    flock: dict
    pens: list[PenConfig]
    cameras: list[CameraConfig]
    analysis: dict
    baseline: dict
    notifications: dict
    rules: list[Rule]

    # ---- helpers --------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        p = Path(self.farm.get("data_dir", "data"))
        return p if p.is_absolute() else self.root / p

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.farm.get("db_file", "pews.sqlite")

    @property
    def placement_date(self) -> date:
        return _to_date(self.flock["placement_date"])

    @property
    def flock_id(self) -> str:
        return str(self.flock["id"])

    def age_days(self, when: datetime | date) -> int:
        d = when.date() if isinstance(when, datetime) else when
        return (d - self.placement_date).days

    def bird_area_m2(self, age_days: float) -> float:
        table = self.flock.get("bird_area_m2_by_age") or DEFAULT_BIRD_AREA
        ages = np.array(sorted(float(k) for k in table))
        vals = np.array([float(table[k]) for k in sorted(table, key=float)])
        return float(np.interp(age_days, ages, vals))

    def camera(self, cam_id: str) -> CameraConfig:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        raise KeyError(f"unknown camera {cam_id!r}")

    def pen(self, pen_id: str) -> PenConfig:
        for p in self.pens:
            if p.id == pen_id:
                return p
        raise KeyError(f"unknown pen {pen_id!r}")


DEFAULT_BIRD_AREA = {0: 0.0025, 7: 0.006, 14: 0.015, 21: 0.025, 28: 0.035, 35: 0.045, 42: 0.055}


def _to_date(v: Any) -> date:
    if isinstance(v, date):
        return v
    return datetime.strptime(str(v), "%Y-%m-%d").date()


def _parse_exclude(items: list) -> list:
    out = []
    for it in items or []:
        if isinstance(it, dict) and "circle" in it:
            cx, cy, r = it["circle"]
            out.append(("circle", float(cx), float(cy), float(r)))
        else:
            out.append(np.asarray(it, dtype=np.float64))
    return out


def load_calibration(path: Path) -> list[CalibrationVersion]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    versions = []
    for v in raw.get("versions", []):
        ip = np.asarray(v["image_points"], dtype=np.float64)
        fp = np.asarray(v["floor_points"], dtype=np.float64)
        if len(ip) < 4 or len(ip) != len(fp):
            raise ValueError(f"{path}: need >=4 matching image/floor points")
        w, h = v["image_size"]
        roi = np.asarray(v.get("roi") or [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        versions.append(CalibrationVersion(
            valid_from=_to_date(v.get("valid_from", "1970-01-01")),
            image_size=(int(w), int(h)), image_points=ip, floor_points=fp,
            roi=roi, exclude=_parse_exclude(v.get("exclude"))))
    if not versions:
        raise ValueError(f"{path}: no calibration versions")
    return versions


def load_rules(path: Path) -> list[Rule]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rules = []
    for r in raw.get("rules", []):
        rules.append(Rule(
            id=r["id"], signal=r["signal"], metric=r["metric"], kind=r.get("kind", "zscore"),
            warn=float(r["warn"]), crit=float(r["crit"]),
            direction=r.get("direction", "low"), op=r.get("op", ">"),
            persist=int(r.get("persist", 1)), cooldown_min=int(r.get("cooldown_min", 60))))
    return rules


def load_config(path: str | Path, rules_path: str | Path | None = None) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # repo root = parent of configs/ when the file lives there, else the file's directory
    root = path.parent.parent if path.parent.name == "configs" else path.parent

    def rel(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else root / q

    pens = []
    for p in raw.get("pens", []):
        pens.append(PenConfig(
            id=str(p["id"]), name=str(p.get("name", p["id"])),
            width_m=float(p["width_m"]), length_m=float(p["length_m"]),
            wall_band_m=float(p.get("wall_band_m", 0.3)),
            corner_size_m=float(p.get("corner_size_m", 0.6)),
            feeders=[Circle(**c) for c in p.get("feeders", [])],
            drinkers=[Circle(**c) for c in p.get("drinkers", [])],
            cameras=[str(c) for c in p.get("cameras", [])],
            layouts=[(_to_date(lay.get("valid_from", "1970-01-01")),
                      [Circle(**c) for c in lay.get("feeders", [])],
                      [Circle(**c) for c in lay.get("drinkers", [])])
                     for lay in p.get("layouts", [])]))

    cams = []
    for c in raw.get("cameras", []):
        cal_file = rel(c["calibration"])
        cams.append(CameraConfig(
            id=str(c["id"]), pen=str(c["pen"]), calibration_file=cal_file,
            input_dir=rel(c.get("input_dir", f"data/raw/{c['id']}")),
            source_fps=float(c.get("source_fps", 15)),
            filename_regex=c.get("filename_regex") or "",
            versions=load_calibration(cal_file)))

    pen_ids = {p.id for p in pens}
    for c in cams:
        if c.pen not in pen_ids:
            raise ValueError(f"camera {c.id} refers to unknown pen {c.pen}")

    rules_file = Path(rules_path) if rules_path else root / "configs" / "rules.yaml"
    rules = load_rules(rules_file) if rules_file.exists() else []

    return Config(
        root=root, farm=raw.get("farm", {}), flock=raw.get("flock", {}), pens=pens, cameras=cams,
        analysis=_deep_merge(DEFAULT_ANALYSIS, raw.get("analysis", {})),
        baseline=_deep_merge(DEFAULT_BASELINE, raw.get("baseline", {})),
        notifications=raw.get("notifications", {}) or {}, rules=rules)
