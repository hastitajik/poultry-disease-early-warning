"""Normalised baseline: "what is normal for THIS age at THIS hour of day?"

For a metric at (slot, age, hour) the reference sample is
  * the same flock, previous `days_back` days, hour of day +- `hour_window`
  * plus older flocks at the same age +- `other_flocks_age_window`, same hours
Robust statistics (median, MAD) so a single bad day does not move "normal".
Below `min_samples` the signal is in "learning" state (no z-score).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .store import Store, iso


@dataclass
class BaselineStat:
    median: float | None
    mad: float | None
    n: int

    def z(self, value: float | None) -> float | None:
        if value is None or self.median is None:
            return None
        scale = 1.4826 * (self.mad or 0.0)
        # floor the scale so a very stable baseline does not turn noise into alarms
        scale = max(scale, 0.05 * abs(self.median), 1e-6)
        return (value - self.median) / scale


def _hours(h: int, w: int) -> list[int]:
    return sorted({(h + k) % 24 for k in range(-w, w + 1)})


def compute_baseline(store: Store, cfg_baseline: dict, flock_id: str, pen: str, name: str,
                     slot: datetime, age_days: int) -> BaselineStat:
    w = int(cfg_baseline.get("hour_window", 1))
    d0, d1 = cfg_baseline.get("days_back", [1, 4])
    hours = _hours(slot.hour, w)
    marks = ",".join("?" * len(hours))

    day = slot.replace(hour=0, minute=0, second=0, microsecond=0)
    since = day - timedelta(days=int(d1))
    until = day - timedelta(days=int(d0) - 1)  # exclusive upper bound = end of day-d0
    rows = store.q(
        f"SELECT value FROM metrics WHERE flock_id=? AND pen=? AND name=? AND value IS NOT NULL "
        f"AND slot_ts>=? AND slot_ts<? AND hour IN ({marks})",
        (flock_id, pen, name, iso(since), iso(until), *hours))
    vals = [r["value"] for r in rows]

    aw = int(cfg_baseline.get("other_flocks_age_window", 2))
    rows2 = store.q(
        f"SELECT value FROM metrics WHERE flock_id!=? AND pen=? AND name=? AND value IS NOT NULL "
        f"AND age_days BETWEEN ? AND ? AND hour IN ({marks})",
        (flock_id, pen, name, age_days - aw, age_days + aw, *hours))
    vals += [r["value"] for r in rows2]

    if len(vals) < int(cfg_baseline.get("min_samples", 6)):
        return BaselineStat(None, None, len(vals))
    arr = np.asarray(vals, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return BaselineStat(med, mad, len(vals))
