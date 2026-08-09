#!/usr/bin/env python3
"""Build barn/metadata.yaml: per-world facts + the seeded train/holdout split.

Per world:
  n_cylinders        obstacle count (walls included)
  optimal_len_barn   length of BARN's shipped reference path (start/goal
                     endpoints included, as the challenge metric computes it)
  optimal_len_ours   length of OUR A* path on the Burger-inflated 0.05 m
                     planning grid — the path-efficiency denominator used in
                     evaluation (consistent with the robot that actually runs)
  min_clearance      min cylinder-center distance along our path [m]
  split              train | holdout (40 holdout worlds, seed 42 — the
                     evaluation set; never used for expert data)

Usage: .venv/bin/python barn/build_metadata.py [--inflate 0.21] [--seed 42]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import grid as G  # noqa: E402

HERE = Path(__file__).parent
N_HOLDOUT = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inflate', type=float, default=G.DEFAULT_INFLATE_RADIUS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grids-dir', type=Path, default=HERE / 'grids')
    ap.add_argument('--paths-dir', type=Path,
                    default=HERE / 'raw/jackal_helper/worlds/BARN/path_files')
    args = ap.parse_args()

    grids = sorted(args.grids_dir.glob('grid_*.npy'),
                   key=lambda p: int(p.stem.split('_')[1]))
    if not grids:
        sys.exit(f'no grids under {args.grids_dir} — run world_to_grid.py first')

    rng = np.random.default_rng(args.seed)
    holdout = set(sorted(rng.choice(len(grids), N_HOLDOUT, replace=False).tolist()))

    worlds = {}
    unplannable = []
    t0 = time.time()
    for gf in grids:
        idx = int(gf.stem.split('_')[1])
        occ = np.load(gf)
        cyl = G.arena_grid_to_cylinders(occ)
        planner = G.Planner(cyl, inflate_radius=args.inflate)
        path = planner.plan(G.START_POSE[:2], G.GOAL_XY)
        entry = {
            'world': f'worlds/world_{idx:03d}.world',
            'world_fast': f'worlds_fast/world_{idx:03d}.world',
            'grid': f'grids/grid_{idx:03d}.npy',
            'n_cylinders': int(occ.sum()),
            'optimal_len_barn': round(G.reference_path_length(
                args.paths_dir / f'path_{idx}.npy'), 3),
            'split': 'holdout' if idx in holdout else 'train',
        }
        if path is None:
            unplannable.append(idx)
            entry['optimal_len_ours'] = None
            entry['min_clearance'] = None
        else:
            entry['optimal_len_ours'] = round(G.path_length(path), 3)
            entry['min_clearance'] = round(planner.min_clearance(path), 3)
        worlds[idx] = entry

    meta = {
        'source': 'Daffan/the-barn-challenge @ bf5a226f6088ec96bf0d2dbee3253a8ea6119b83 (MIT)',
        'constants': {
            'cell': G.CELL,
            'cylinder_radius': G.CYL_RADIUS,
            'arena_origin': list(G.ARENA_ORIGIN),
            'start_pose': list(G.START_POSE),
            'goal': list(G.GOAL_XY),
            'success_radius': G.SUCCESS_RADIUS,
            'inflate_radius': args.inflate,
            'split_seed': args.seed,
        },
        'holdout': sorted(holdout),
        'unplannable': unplannable,
        'worlds': worlds,
    }
    out = HERE / 'metadata.yaml'
    with open(out, 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    lens = [w['optimal_len_ours'] for w in worlds.values()
            if w['optimal_len_ours'] is not None]
    clear = [w['min_clearance'] for w in worlds.values()
             if w['min_clearance'] is not None]
    print(f'metadata: {out} ({len(worlds)} worlds, {time.time()-t0:.1f}s)')
    print(f'holdout: {len(holdout)} worlds   unplannable at inflate='
          f'{args.inflate}: {len(unplannable)} {unplannable[:20]}')
    print(f'optimal_len_ours: min {min(lens):.2f}  max {max(lens):.2f}  '
          f'mean {np.mean(lens):.2f} m')
    print(f'min_clearance: min {min(clear):.3f}  mean {np.mean(clear):.3f} m')


if __name__ == '__main__':
    main()
