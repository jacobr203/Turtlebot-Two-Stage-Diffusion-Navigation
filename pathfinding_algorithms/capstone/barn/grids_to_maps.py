#!/usr/bin/env python3
"""Render BARN grids into map_server maps (for Nav2 on BARN worlds).

The committed grids are exact (cylinders live on the 0.15 m lattice), so a
rasterized map beats SLAMing the world first. Each occupied cell paints its
0.15 m footprint at 0.05 m resolution. The canvas is padded well past the
arena (x [-5.0, 0.5], y [-1.0, 14.5], all free ground plane in the worlds)
because Nav2's global costmap is sized to the static map — the official
goal (-2.25, 13) sits 3.4 m beyond the arena's open end and must be inside
map bounds to be accepted.

Maps are regenerable artifacts (gitignored); run.sh nav2 auto-generates a
missing map for a BARN world via this script.

  .venv/bin/python barn/grids_to_maps.py --idx 306
  .venv/bin/python barn/grids_to_maps.py --all
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ARENA_ORIGIN = (-4.425, 0.075)   # lattice cell (0,0) centre, = common.grid
CELL = 0.15
RES = 0.05
X_MIN, X_MAX = -5.0, 0.5
Y_MIN, Y_MAX = -1.0, 14.5
CYL_HALF = 0.075


def render(grid_file, out_dir):
    occ = np.load(grid_file)
    W = int(round((X_MAX - X_MIN) / RES))
    H = int(round((Y_MAX - Y_MIN) / RES))
    img = np.full((H, W), 254, dtype=np.uint8)   # free
    for i, j in zip(*np.nonzero(occ)):
        cx = ARENA_ORIGIN[0] + i * CELL
        cy = ARENA_ORIGIN[1] + j * CELL
        px0 = int(np.floor((cx - CYL_HALF - X_MIN) / RES))
        px1 = int(np.ceil((cx + CYL_HALF - X_MIN) / RES))
        py0 = int(np.floor((cy - CYL_HALF - Y_MIN) / RES))
        py1 = int(np.ceil((cy + CYL_HALF - Y_MIN) / RES))
        # PGM row 0 is the TOP of the map (max y)
        img[H - py1:H - py0, px0:px1] = 0
    idx = grid_file.stem.split('_')[1]
    pgm = out_dir / f'map_{idx}.pgm'
    with open(pgm, 'wb') as f:
        f.write(f'P5\n{W} {H}\n255\n'.encode())
        f.write(img.tobytes())
    (out_dir / f'map_{idx}.yaml').write_text(
        f'image: map_{idx}.pgm\n'
        f'mode: trinary\n'
        f'resolution: {RES}\n'
        f'origin: [{X_MIN}, {Y_MIN}, 0.0]\n'
        f'negate: 0\n'
        f'occupied_thresh: 0.65\n'
        f'free_thresh: 0.196\n')
    return idx, int(occ.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, nargs='*', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    out_dir = HERE / 'maps'
    out_dir.mkdir(exist_ok=True)
    if args.all:
        files = sorted(HERE.glob('grids/grid_*.npy'))
    elif args.idx:
        files = [HERE / 'grids' / f'grid_{i:03d}.npy' for i in args.idx]
    else:
        ap.error('give --idx N [N ...] or --all')
    for f in files:
        if not f.exists():
            print(f'[skip] {f} missing')
            continue
        idx = f.stem.split('_')[1]
        if not args.force and (out_dir / f'map_{idx}.yaml').exists():
            continue
        idx, n = render(f, out_dir)
        print(f'map_{idx}: {n} cylinders -> maps/map_{idx}.pgm/.yaml')


if __name__ == '__main__':
    main()
