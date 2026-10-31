"""Interactive calibration with an OpenCV window (needs a desktop; install
`opencv-python`, not the headless build).

Steps (instructions are printed in the terminal):
  1. click 4 floor points whose position in metres you know
     (default: the 4 inner corners of the pen: far-left, far-right, near-right, near-left)
  2. click the analysis ROI polygon (floor + bottom of walls), Enter to finish
  3. click exclusion polygons (feeders, OSD text, thermometer...), n = next polygon, Enter = done
  4. click feeder / drinker centres, f / d to switch, Enter = done -> printed in metres
The result is appended as a new `version` to the camera calibration YAML.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import cv2
import numpy as np
import yaml


class _Clicker:
    def __init__(self, win: str, img: np.ndarray, scale: float):
        self.win, self.base, self.scale = win, img, scale
        self.points: list[tuple[int, int]] = []
        self.done_polys: list[list[tuple[int, int]]] = []
        cv2.setMouseCallback(win, self._cb)

    def _cb(self, ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x / self.scale), int(y / self.scale)))
            self.draw()

    def draw(self):
        img = self.base.copy()
        s = self.scale
        for poly in self.done_polys:
            cv2.polylines(img, [np.int32([(p[0] * s, p[1] * s) for p in poly])], True, (0, 200, 255), 2)
        for i, p in enumerate(self.points):
            q = (int(p[0] * s), int(p[1] * s))
            cv2.circle(img, q, 5, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (q[0] + 6, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(self.points) > 1:
            cv2.polylines(img, [np.int32([(p[0] * s, p[1] * s) for p in self.points])], False, (0, 0, 255), 1)
        cv2.imshow(self.win, img)


def _collect(clk: _Clicker, n: int | None = None, multi: bool = False) -> list:
    clk.points = []
    clk.draw()
    polys = []
    while True:
        k = cv2.waitKey(30) & 0xFF
        if n is not None and len(clk.points) >= n:
            return clk.points[:n]
        if k == ord("u") and clk.points:
            clk.points.pop()
            clk.draw()
        if multi and k == ord("n") and len(clk.points) >= 3:
            polys.append(clk.points)
            clk.done_polys.append(clk.points)
            clk.points = []
            clk.draw()
        if k in (13, 10):  # Enter
            if multi:
                if len(clk.points) >= 3:
                    polys.append(clk.points)
                return polys
            return clk.points
        if k == 27:
            raise KeyboardInterrupt


def run_calibration(image_path: str | Path, out_yaml: str | Path, camera_id: str,
                    width_m: float, length_m: float, valid_from: date) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"cannot read {image_path}")
    h, w = img.shape[:2]
    scale = min(1.0, 1400 / w)
    view = cv2.resize(img, (int(w * scale), int(h * scale)))
    win = "pews calibrate"
    cv2.namedWindow(win)
    clk = _Clicker(win, view, scale)

    print("1) Click 4 floor points: far-left, far-right, near-right, near-left corner of the pen floor.")
    print("   (u = undo, Esc = abort). If a corner is outside the image, click any point whose")
    print("   position in metres you know and type it when asked.")
    pts = _collect(clk, n=4)
    defaults = [(0, 0), (width_m, 0), (width_m, length_m), (0, length_m)]
    floor = []
    for i, d in enumerate(defaults):
        s = input(f"   point {i + 1} in metres x,y [{d[0]},{d[1]}]: ").strip()
        floor.append(tuple(float(v) for v in s.split(",")) if s else d)

    print("2) Click the ROI polygon (floor + base of the walls). Enter = done.")
    roi = _collect(clk)
    print("3) Click exclusion polygons (white feeders, OSD clock, thermometer).")
    print("   n = next polygon, Enter = done (Enter right away = none).")
    excl = _collect(clk, multi=True)

    H, _ = cv2.findHomography(np.float32(pts), np.float32(floor), 0)
    print("4) Click feeder centres (on the floor, at the base). Enter = done.")
    feeders = _collect(clk)
    print("   Click drinker centres. Enter = done.")
    drinkers = _collect(clk)
    cv2.destroyWindow(win)

    def to_m(p):
        q = cv2.perspectiveTransform(np.float64([[p]]), H)[0, 0]
        return round(float(q[0]), 2), round(float(q[1]), 2)

    version = {
        "valid_from": valid_from.isoformat(), "image_size": [w, h],
        "image_points": [list(p) for p in pts], "floor_points": [list(p) for p in floor],
        "roi": [list(p) for p in roi] if len(roi) >= 3 else None,
        "exclude": [[list(p) for p in poly] for poly in excl],
    }
    out = Path(out_yaml)
    data = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else None
    data = data or {"camera_id": camera_id, "versions": []}
    data["versions"] = [v for v in data.get("versions", []) if str(v.get("valid_from")) != version["valid_from"]]
    data["versions"].append(version)
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"saved calibration version {valid_from} -> {out}")
    if feeders or drinkers:
        print("\nPaste into farm.yaml (pen layouts), adjust r (zone radius, m):")
        print(f"      - valid_from: \"{valid_from}\"")
        print("        feeders:")
        for p in feeders:
            x, y = to_m(p)
            print(f"          - {{x: {x}, y: {y}, r: 0.45}}")
        print("        drinkers:")
        for p in drinkers:
            x, y = to_m(p)
            print(f"          - {{x: {x}, y: {y}, r: 0.40}}")
    return version
