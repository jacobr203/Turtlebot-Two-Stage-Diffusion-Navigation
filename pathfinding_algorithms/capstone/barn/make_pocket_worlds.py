#!/usr/bin/env python3
"""Generate pocket-approach worlds (indices >= 1000) for the corridor-pocket
training top-up (2026-08-03, color-test NEEDS-TRAINING note in
notes/progress.md).

Each world is the standard BARN arena (wall lattice reused verbatim) holding
2-3 C-shaped pockets plus light scatter clutter. A pocket is a rectangular
cylinder ring with a 3-cell mouth (0.45 m center gap = 0.30 m surface gap —
the canonical BARN squeeze the planner passes at 0.21 m inflate and the wide
0.24 m gate refuses): the exact stress-tier geometry where the deployed seek
stack's veto clusters and wedges happened (world 768 session).

Per pocket the yaml records TWO expert legs keyed to the color-test approach
machinery:
  enter: outside start -> the 0.55 m stand-off pose in front of the pocket's
         inner back wall (the seek stack's staged approach pose)
  exit:  that pose -> an outside waypoint on a different bearing (the
         post-FOUND pocket exit — absent from all prior training data; a
         terminal wedge in the 2026-08-03 run 3)
Legs are verified plannable with the production route convention
(plan_route: wide inflate preferred, clearance-centered fallback at 0.21).

Outputs are drop-in for the toolchain, like make_dense_worlds.py:
  worlds/world_NNNN.world  worlds_fast/world_NNNN.world  grids/grid_NNNN.npy
  pocket_worlds.yaml       per-world split ('train' / 'holdout', every 8th)
                           + legs + geometry stats

  .venv/bin/python barn/make_pocket_worlds.py --count 80
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import grid as G                          # noqa: E402
from make_dense_worlds import (wall_and_field, load_template,  # noqa: E402
                               write_world, SPAWN_CELL, SPAWN_CLEAR)

HERE = Path(__file__).parent

MOUTH_CELLS = 3          # 0.45 m center gap -> 0.30 m surface gap
APPROACH_D = 0.55        # stand-off pose distance from the face surface [m]
ROUTE_LEN = (2.0, 5.0)   # accepted leg route lengths [m]
SCATTER_P = (0.02, 0.05)  # Bernoulli clutter range outside pockets

# grid-axis mouth directions: (di, dj) pointing OUT of the pocket
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def carve_pocket(occ, field, rng):
    """Place one pocket ring; returns (cells_mask, face_xy, out_dir_xy,
    approach_xy) in world coords, or None if no placement fits."""
    w = int(rng.integers(4, 7))       # interior width  [cells] 0.60-0.90 m
    h = int(rng.integers(5, 8))       # interior depth  [cells] 0.75-1.05 m
    di, dj = DIRS[int(rng.integers(4))]
    for _ in range(60):
        # pocket rect in grid coords: (w+2) across, (h+2) deep incl. walls
        if dj == 0:      # mouth faces +-x: rect is (h+2) x (w+2) in (i, j)
            ni, nj = h + 2, w + 2
        else:
            ni, nj = w + 2, h + 2
        i0 = int(rng.integers(1, occ.shape[0] - ni - 1))
        j0 = int(rng.integers(1, occ.shape[1] - nj - 1))
        rect = np.s_[i0:i0 + ni, j0:j0 + nj]
        # need the rect + 1-cell margin fully in the free field band
        m = np.s_[max(i0 - 1, 0):i0 + ni + 1, max(j0 - 1, 0):j0 + nj + 1]
        if not field[m].all() or occ[m].any():
            continue
        ring = np.zeros((ni, nj), dtype=np.uint8)
        ring[0, :] = ring[-1, :] = 1
        ring[:, 0] = ring[:, -1] = 1
        # open the mouth: middle MOUTH_CELLS of the out-facing side
        if dj == 0:
            row = -1 if di > 0 else 0
            a = 1 + (nj - 2 - MOUTH_CELLS) // 2
            ring[row, a:a + MOUTH_CELLS] = 0
        else:
            col = -1 if dj > 0 else 0
            a = 1 + (ni - 2 - MOUTH_CELLS) // 2
            ring[a:a + MOUTH_CELLS, col] = 0
        occ[rect] |= ring

        # world-frame geometry
        ox, oy = G.ARENA_ORIGIN
        cell_xy = lambda i, j: (ox + i * G.CELL, oy + j * G.CELL)
        # inner back-wall face center: wall opposite the mouth
        if dj == 0:
            bi = i0 if di > 0 else i0 + ni - 1
            bj = j0 + nj // 2
        else:
            bi = i0 + ni // 2
            bj = j0 if dj > 0 else j0 + nj - 1
        bx, by = cell_xy(bi, bj)
        # face surface sits one cylinder radius toward the interior
        fx = bx + di * G.CYL_RADIUS
        fy = by + dj * G.CYL_RADIUS
        approach = (fx + di * APPROACH_D, fy + dj * APPROACH_D)
        return rect, (fx, fy), (di, dj), approach
    return None


def sample_leg(planner, free_xy, fixed, rng, fixed_is_goal, jyaw=0.15):
    """One verified leg. fixed_is_goal: outside start -> fixed (enter);
    else fixed -> outside goal (exit). Returns leg dict or None."""
    for _ in range(40):
        p = tuple(free_xy[rng.integers(len(free_xy))])
        if planner.clearance_at(p) < 0.35:
            continue
        s, g = (p, fixed) if fixed_is_goal else (fixed, p)
        d = np.hypot(g[0] - s[0], g[1] - s[1])
        if not (ROUTE_LEN[0] * 0.6 <= d <= ROUTE_LEN[1]):
            continue
        _, path, _ = G.plan_route(planner_cyl(planner), s, g)
        if path is None:
            continue
        if not (ROUTE_LEN[0] <= G.path_length(path) <= ROUTE_LEN[1] * 1.3):
            continue
        k = min(len(path) - 1, 6)
        yaw = float(np.arctan2(path[k][1] - s[1], path[k][0] - s[0])) \
            + float(rng.uniform(-jyaw, jyaw))
        return dict(sx=round(float(s[0]), 4), sy=round(float(s[1]), 4),
                    syaw=round(yaw, 4),
                    gx=round(float(g[0]), 4), gy=round(float(g[1]), 4),
                    kind='enter' if fixed_is_goal else 'exit')
    return None


def planner_cyl(planner):
    return planner._cylinders


def build_world(rng, walls, field):
    """One world draw: pockets + scatter. Returns (occ, legs, stats) or
    None when the draw fails verification."""
    occ = walls.copy()
    n_pockets = int(rng.integers(2, 4))
    pockets = []
    for _ in range(n_pockets):
        pk = carve_pocket(occ, field & (occ == 0), rng)
        if pk is not None:
            pockets.append(pk)
    if len(pockets) < 2:
        return None

    # light scatter outside the pockets (margin already enforced at carve
    # time for the ring; keep a bubble around each approach pose free)
    scatter = (field & (occ == 0)) \
        & (np.random.default_rng(rng.integers(2**31)).random(occ.shape)
           < rng.uniform(*SCATTER_P))
    occ = np.where(scatter, 1, occ).astype(np.uint8)
    ii, jj = np.mgrid[0:occ.shape[0], 0:occ.shape[1]]
    bubble = np.hypot(ii - SPAWN_CELL[0], jj - SPAWN_CELL[1]) <= SPAWN_CLEAR
    occ[bubble & (walls == 0)] = 0

    cyl = G.arena_grid_to_cylinders(occ)
    planner = G.Planner(cyl, inflate_radius=G.DEFAULT_INFLATE_RADIUS)
    planner._cylinders = cyl
    fi, fj = np.nonzero(~planner.blocked)
    free_xy = np.stack([planner.x0 + fi * planner.res,
                        planner.y0 + fj * planner.res], axis=1)

    legs = []
    for _, face, (di, dj), approach in pockets:
        if planner.clearance_at(approach) < 0.24:
            continue          # scatter landed too close to the stand-off
        enter = sample_leg(planner, free_xy, approach, rng, fixed_is_goal=True)
        exit_ = sample_leg(planner, free_xy, approach, rng, fixed_is_goal=False)
        if enter and exit_:
            legs += [enter, exit_]
    if len(legs) < 4:         # >= 2 pockets fully usable
        return None
    stats = dict(pockets=len(legs) // 2,
                 cylinders=int(occ.sum()),
                 mouth_gap_m=round(MOUTH_CELLS * G.CELL - 2 * G.CYL_RADIUS, 3))
    return occ, legs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=80)
    ap.add_argument('--seed', type=int, default=20260803)
    ap.add_argument('--start-idx', type=int, default=1000)
    ap.add_argument('--max-draws', type=int, default=2000)
    args = ap.parse_args()

    walls, field = wall_and_field()
    stripped, cyl_template = load_template()
    (HERE / 'worlds_fast').mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    yaml_file = HERE / 'pocket_worlds.yaml'
    meta = yaml.safe_load(open(yaml_file))['worlds'] \
        if yaml_file.exists() else {}
    idx = args.start_idx
    made = 0
    for _ in range(args.max_draws):
        if made >= args.count:
            break
        out = build_world(rng, walls, field)
        if out is None:
            continue
        occ, legs, stats = out
        write_world(idx, occ, stripped, cyl_template)
        meta[idx] = dict(split='holdout' if made % 8 == 7 else 'train',
                         seed=args.seed, legs=legs, **stats)
        print(f'world_{idx}: {stats["pockets"]} pockets, {len(legs)} legs, '
              f'{stats["cylinders"]} cylinders [{meta[idx]["split"]}]',
              flush=True)
        idx += 1
        made += 1

    yaml_file.write_text(yaml.safe_dump({'worlds': meta}, sort_keys=True))
    n_train = sum(1 for w in meta.values() if w['split'] == 'train')
    print(f'[pocket] {made} worlds written (total {len(meta)}: '
          f'{n_train} train / {len(meta) - n_train} holdout) -> {yaml_file}')
    if made < args.count:
        print(f'[pocket] WARNING: only {made}/{args.count} accepted draws')


if __name__ == '__main__':
    main()
