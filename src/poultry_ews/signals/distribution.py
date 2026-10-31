"""Signal 2 — Spatial distribution.

From the mean bird-occupancy map of a slot (floor plane, 0..1 per pixel):

  frac_<zone>        share of all bird mass in feeder / drinker / wall / corner / open
  resource_frac      feeder + drinker   (drops when birds stop eating/drinking)
  periphery_frac     wall + corner      (rises when birds huddle at the edges)
  dispersion_entropy 0..1, how evenly birds are spread over 25 cm cells
  bird_coverage      share of visible floor covered by birds (grows with age)

Newcastle disease / influenza: fewer birds at feed & water, more in corners,
often before clear neurological signs (see the project PDF, signal 2).
"""
from __future__ import annotations

import numpy as np

from ..geometry import CORNER, DRINKER, FEEDER, OPEN, WALL, ZONE_NAMES


def fuse_occupancy(maps: list[np.ndarray], coverages: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Average overlapping cameras; returns (occupancy, union coverage)."""
    num = np.zeros_like(maps[0], dtype=np.float32)
    den = np.zeros_like(maps[0], dtype=np.float32)
    for m, c in zip(maps, coverages):
        c = c.astype(np.float32)
        num += m.astype(np.float32) * c
        den += c
    occ = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return occ, den > 0


def distribution_metrics(occ: np.ndarray, coverage: np.ndarray, zone_map: np.ndarray,
                         ppm: float, cell_m: float = 0.25) -> dict[str, float | None]:
    occ = np.where(coverage, occ, 0.0)
    total = float(occ.sum())
    out: dict[str, float | None] = {}
    if total <= 1e-6:
        for z in ZONE_NAMES.values():
            out[f"frac_{z}"] = None
        out.update(resource_frac=None, periphery_frac=None, dispersion_entropy=None,
                   bird_coverage=0.0 if coverage.any() else None)
        return out

    for label, name in ZONE_NAMES.items():
        out[f"frac_{name}"] = float(occ[zone_map == label].sum()) / total
    out["resource_frac"] = (out["frac_feeder"] or 0) + (out["frac_drinker"] or 0)
    out["periphery_frac"] = (out["frac_wall"] or 0) + (out["frac_corner"] or 0)

    # entropy over cells that the cameras can see (>50% covered)
    cell = max(1, int(round(cell_m * ppm)))
    rows, cols = occ.shape
    r2, c2 = rows // cell, cols // cell
    if r2 and c2:
        o = occ[:r2 * cell, :c2 * cell].reshape(r2, cell, c2, cell).sum(axis=(1, 3))
        cv = coverage[:r2 * cell, :c2 * cell].reshape(r2, cell, c2, cell).mean(axis=(1, 3)) > 0.5
        p = o[cv]
        n = p.size
        if n > 1 and p.sum() > 0:
            p = p / p.sum()
            nz = p[p > 0]
            out["dispersion_entropy"] = float(-(nz * np.log(nz)).sum() / np.log(n))
        else:
            out["dispersion_entropy"] = None
    else:
        out["dispersion_entropy"] = None
    out["bird_coverage"] = total / float(coverage.sum())
    return out


__all__ = ["fuse_occupancy", "distribution_metrics", "OPEN", "WALL", "CORNER", "DRINKER", "FEEDER"]
