"""Signal 4 — Circadian pattern.

Healthy broilers follow a daily rhythm (more active in the morning, less in
the afternoon, rest at night). Sick birds lose it; Roberts, Cain & Dawkins
(2012) predicted welfare problems 1-2 days ahead from such changes.

Every slot we compare the last 24 h of hourly activity with the usual hourly
profile (median over the previous days):

  rhythm_corr       Pearson correlation of the two hourly profiles (1 = same shape)
  rhythm_amplitude  today's peak-to-trough / usual peak-to-trough (1 = same strength)

NOTE: if the lighting programme changes (e.g. dark hours added), the rhythm
changes for a non-disease reason — record lighting changes for the vet.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np


def hourly_means(rows: list[tuple[datetime, float]]) -> dict[int, float]:
    acc: dict[int, list[float]] = defaultdict(list)
    for ts, v in rows:
        if v is not None:
            acc[ts.hour].append(float(v))
    return {h: float(np.mean(v)) for h, v in acc.items()}


def baseline_profile(rows: list[tuple[datetime, float]]) -> dict[int, float]:
    """Median over days of each day's hourly mean."""
    per_day: dict[tuple, list[float]] = defaultdict(list)
    for ts, v in rows:
        if v is not None:
            per_day[(ts.date(), ts.hour)].append(float(v))
    by_hour: dict[int, list[float]] = defaultdict(list)
    for (_, h), vals in per_day.items():
        by_hour[h].append(float(np.mean(vals)))
    return {h: float(np.median(v)) for h, v in by_hour.items()}


def _spread(vals: np.ndarray) -> float:
    return float(np.percentile(vals, 90) - np.percentile(vals, 10))


def rhythm_metrics(recent: dict[int, float], usual: dict[int, float], min_hours: int
                   ) -> dict[str, float | None]:
    hours = sorted(set(recent) & set(usual))
    if len(hours) < min_hours:
        return {"rhythm_corr": None, "rhythm_amplitude": None}
    a = np.array([recent[h] for h in hours])
    b = np.array([usual[h] for h in hours])
    corr = None
    if a.std() > 1e-9 and b.std() > 1e-9:
        corr = float(np.corrcoef(a, b)[0, 1])
    sb = _spread(b)
    amp = _spread(a) / sb if sb > 1e-9 else None
    return {"rhythm_corr": corr, "rhythm_amplitude": amp}
