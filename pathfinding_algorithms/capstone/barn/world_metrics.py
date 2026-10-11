#!/usr/bin/env python3
"""Measure navigation difficulty for every BARN world.

Writes barn/world_metrics.json, used by the poster's difficulty panels
and by barn/WORLDS.md.

Per world:
    occupancy   fraction of interior cells holding an obstacle
    min_gap_m   narrowest passage on the widest route across the world
    open_frac   fraction of free cells with >= 0.5 m clearance
    score       0-1 difficulty, higher is harder

BARN grids are walled on all four borders, so the wall ring is stripped
before measuring; the traverse runs along the long axis.
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label

HERE = Path(__file__).parent
GRIDS = HERE / 'grids'
WORLDS = HERE / 'worlds_color'
OUT = HERE / 'world_metrics.json'

RESOLUTION_M = 0.15        # BARN grid cell size
CLEARANCE_OPEN_M = 0.5     # "room to manoeuvre" threshold
BISECT_STEPS = 14


def widest_bottleneck(free, clearance):
    """Largest clearance that still connects the left edge to the right.

    Binary search on a clearance threshold: keep only cells at least
    that clear, and check whether the two ends remain connected.
    """
    lo, hi, best = 0.0, float(clearance.max()), 0.0
    for _ in range(BISECT_STEPS):
        mid = (lo + hi) / 2.0
        labelled, _ = label(free & (clearance >= mid))
        left = set(labelled[:, 0][labelled[:, 0] > 0].tolist())
        right = set(labelled[:, -1][labelled[:, -1] > 0].tolist())
        if left & right:
            best, lo = mid, mid
        else:
            hi = mid
    return best


def measure(grid_path):
    return measure_grid(np.load(grid_path))


def measure_grid(grid):
    """Metrics of one walled grid (long axis = axis 1); ring stripped here."""
    grid = np.asarray(grid).astype(bool)[1:-1, 1:-1]
    clearance = distance_transform_edt(~grid) * RESOLUTION_M
    free = ~grid
    return {
        'occ': float(grid.mean()),
        'min_gap_m': widest_bottleneck(free, clearance) * 2.0,
        'open_frac': (float((clearance[free] >= CLEARANCE_OPEN_M).mean())
                      if free.any() else 0.0),
    }


def normalize(values, invert=False):
    lo, hi = values.min(), values.max()
    scaled = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
    return 1.0 - scaled if invert else scaled


def tier_for_quantile(quantile):
    """Tier name for a position in the ranked set (0 = easiest)."""
    return ('EASY' if quantile < 0.25 else
            'MODERATE' if quantile < 0.55 else
            'HARD' if quantile < 0.85 else 'EXTREME')


def place(metrics, rows):
    """Score one grid's metrics on the scale the measured set defines.

    Uses the set's own min/max for each term (clipped, so a grid outside
    the set's range saturates rather than extrapolates) and returns
    (score, rank, tier) as if the grid had been ranked among `rows`."""
    def unit(key, invert=False):
        values = np.array([r[key] for r in rows])
        lo, hi = values.min(), values.max()
        v = float(np.clip((metrics[key] - lo) / (hi - lo), 0.0, 1.0))
        return 1.0 - v if invert else v
    score = (0.45 * unit('occ') + 0.35 * unit('min_gap_m', invert=True)
             + 0.20 * unit('open_frac', invert=True))
    below = sum(r['score'] < score for r in rows)
    return score, below + 1, tier_for_quantile(below / len(rows))


def main():
    rows = []
    for grid_path in sorted(GRIDS.glob('grid_*.npy')):
        idx = int(re.search(r'grid_(\d+)', grid_path.name).group(1))
        if not (WORLDS / f'world_{idx:03d}.world').exists():
            continue
        rows.append({'world': idx, **measure(grid_path)})

    score = (0.45 * normalize(np.array([r['occ'] for r in rows]))
             + 0.35 * normalize(np.array([r['min_gap_m'] for r in rows]),
                                invert=True)
             + 0.20 * normalize(np.array([r['open_frac'] for r in rows]),
                                invert=True))
    for row, value in zip(rows, score):
        row['score'] = float(value)

    rows.sort(key=lambda r: r['score'])
    total = len(rows)
    for rank, row in enumerate(rows):
        row['tier'] = tier_for_quantile(rank / total)
        row['rank'] = rank + 1

    OUT.write_text(json.dumps(rows, indent=1))
    print(f'{total} worlds measured -> {OUT}')
    for key in (20, 410, 768, 305):
        row = next((r for r in rows if r['world'] == key), None)
        if row:
            print(f"  world {row['world']:3d}: occ {row['occ']*100:5.1f}%  "
                  f"gap {row['min_gap_m']:.2f} m  "
                  f"open {row['open_frac']*100:5.1f}%  "
                  f"{row['tier']}  rank {row['rank']}/{total}")


if __name__ == '__main__':
    main()
