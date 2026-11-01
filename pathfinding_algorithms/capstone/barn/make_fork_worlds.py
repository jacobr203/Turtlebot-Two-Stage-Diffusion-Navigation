#!/usr/bin/env python3
"""Generate multipath worlds (indices 400+): even cylinder clutter with several routes.

Accepted only when >= MIN_ROUTES distinct routes exist within COST_PARITY of the best,
so A* commits to one of several genuine options: the multimodality the diffuser trains on.

Usage (from capstone/): .venv/bin/python barn/make_fork_worlds.py [--n 48]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import grid as G  # noqa: E402

BARN = Path(__file__).resolve().parent
FORK_BASE = 400
CELL = 0.15
NI, NJ = 30, 64                 # BARN arena lattice
ORIGIN = G.ARENA_ORIGIN         # world xy of lattice cell (0, 0)
HOLDOUT = 8

P_BLOCK = (0.30, 0.46)          # per-world block occupancy range
P_DOUBLE = (0.02, 0.08)         # extra clutter: second cylinder per block
START_CLEAR = 0.45              # [m] free pocket around the spawn
MIN_ROUTES = 3                  # distinct near-equal routes required
COST_PARITY = 0.18              # max relative length gap to the best route
ROUTE_SEP = 0.40                # [m] midpoint separation for distinctness


def build_even_field(rng):
    """Jittered-grid clutter: one cylinder per 2x2 block with prob p, some doubled."""
    p = rng.uniform(*P_BLOCK)
    pd = rng.uniform(*P_DOUBLE)
    occ = np.zeros((NI, NJ), dtype=np.uint8)
    for bi in range(0, NI, 2):
        for bj in range(0, NJ, 2):
            cells = [(bi + a, bj + b) for a in (0, 1) for b in (0, 1)
                     if bi + a < NI and bj + b < NJ]
            picks = 1 if rng.random() < p else 0
            if picks and rng.random() < pd:
                picks = 2
            for c in range(picks):
                i, j = cells[rng.integers(len(cells))]
                occ[i, j] = 1
    # perimeter fence: otherwise A* routes skirt the boundary, not the clutter
    occ[0, :] = occ[-1, :] = 1
    occ[:, 0] = occ[:, -1] = 1
    sx, sy = G.START_POSE[:2]
    for i in range(NI):
        for j in range(NJ):
            if occ[i, j] and np.hypot(ORIGIN[0] + i * CELL - sx,
                                      ORIGIN[1] + j * CELL - sy) < START_CLEAR:
                occ[i, j] = 0
    return occ


def grid_to_cyls(occ):
    ii, jj = np.nonzero(occ)
    return np.stack([ORIGIN[0] + ii * CELL, ORIGIN[1] + jj * CELL], axis=1)


GOAL_MIN_D = 2.0                # [m] minimum start->goal separation
ROUTE_LEN_TARGET = 6.0          # [m] best-route length held ~constant
ROUTE_LEN_TOL = 0.75            # across worlds
ROUTE_LEN_TOL_HARD = 1.6        # wider for behind/side goals, which rarely hit the band


DIRS4 = ('above', 'behind', 'left', 'right')


def _direction(dx, dy):
    if dy >= abs(dx):
        return 'above'
    if dy <= -abs(dx):
        return 'behind'
    return 'left' if dx < 0 else 'right'


def goal_candidates(occ, rng, want_dir):
    """Shuffled open interior pockets in the wanted direction from the start."""
    sx, sy = G.START_POSE[:2]
    cands = []
    for i in range(1, NI - 1):
        for j in range(1, NJ - 1):
            if occ[i-1:i+2, j-1:j+2].sum() == 0:
                x = ORIGIN[0] + i * CELL
                y = ORIGIN[1] + j * CELL
                # goals outside the planner's window are unplannable
                (bx0, bx1), (by0, by1) = G.PLAN_BOUNDS
                if not (bx0 + 0.15 <= x <= bx1 - 0.15
                        and by0 + 0.15 <= y <= by1 - 0.15):
                    continue
                d = np.hypot(x - sx, y - sy)
                if not (GOAL_MIN_D <= d
                        <= ROUTE_LEN_TARGET + ROUTE_LEN_TOL_HARD):
                    continue
                if _direction(x - sx, y - sy) != want_dir:
                    continue
                cands.append((x, y))
    rng.shuffle(cands)
    return cands


X_LO = ORIGIN[0] + 2 * CELL     # interior band: routes may not skirt the
X_HI = ORIGIN[0] + (NI - 3) * CELL   # field's side edges
INTERIOR_FRAC = 0.85            # share of a route that must stay in the band


def _interior(route):
    x = route[:, 0]
    return ((x >= X_LO) & (x <= X_HI)).mean() >= INTERIOR_FRAC


def distinct_routes(cyls, goal, kmax=6):
    """Distinct interior routes near the best, found by plugging each and replanning."""
    _, base, _ = G.plan_route(cyls, G.START_POSE[:2], goal)
    if base is None or not _interior(base):
        return []
    routes = [base]
    best = G.path_length(base)
    for _ in range(kmax - 1):
        plugs = []
        for r in routes:
            for frac in (0.25, 0.5, 0.75):
                m = r[int(len(r) * frac)]
                plugs += [[m[0] + dx, m[1] + dy]
                          for dx in (-CELL, 0, CELL)
                          for dy in (-CELL, 0, CELL)]
        _, alt, _ = G.plan_route(np.concatenate([cyls, np.array(plugs)]),
                                 G.START_POSE[:2], goal)
        if alt is None:
            break
        if abs(G.path_length(alt) - best) / best > COST_PARITY:
            break
        mid = alt[len(alt) // 2]
        seps = [np.hypot(r[:, 0] - mid[0], r[:, 1] - mid[1]).min()
                for r in routes]
        if min(seps) < ROUTE_SEP:
            break
        routes.append(alt)
    return [r for r in routes if _interior(r)]


def cylinder_template(world_text):
    """(header-with-cylinders-stripped, one-cylinder template)."""
    a = world_text.index('    <model name="unit_cylinder_')
    b = world_text.index('</model>', a) + len('</model>')
    block = world_text[a:b]
    import re
    block = re.sub(r'unit_cylinder_\d+', 'unit_cylinder_{n}', block)
    block = re.sub(r'<pose frame="">[-0-9.]+ [-0-9.]+ ',
                   '<pose frame="">{x:.6f} {y:.6f} ', block, count=1)
    txt = world_text
    while True:
        try:
            a = txt.index('    <model name="unit_cylinder_')
        except ValueError:
            break
        b = txt.index('</model>', a) + len('</model>')
        txt = txt[:a] + txt[b:]
    return txt, block


def emit_world(template_txt, block, cyls, out_path):
    models = ''.join(
        block.format(n=k, x=x, y=y) for k, (x, y) in enumerate(cyls))
    tail = template_txt.rindex('</world>')
    out_path.write_text(
        template_txt[:tail] + models + template_txt[tail:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--base', type=int, default=FORK_BASE,
                    help='first world index (mass train pool: 500)')
    ap.add_argument('--holdout', type=int, default=HOLDOUT)
    ap.add_argument('--out-yaml', default='fork_worlds.yaml')
    args = ap.parse_args()

    tmpl_eval, block_eval = cylinder_template(
        (BARN / 'worlds' / 'world_000.world').read_text())
    tmpl_fast, block_fast = cylinder_template(
        (BARN / 'worlds_fast' / 'world_000.world').read_text())

    meta, made, tries = [], 0, 0
    rng_root = np.random.default_rng(args.seed)
    while made < args.n and tries < args.n * 200:
        tries += 1
        seed = int(rng_root.integers(0, 2**31))
        rng = np.random.default_rng(seed)
        occ = build_even_field(rng)
        cyls = grid_to_cyls(occ)
        if len(cyls) < 150:
            continue
        want_dir = DIRS4[made % len(DIRS4)]   # even direction mix
        routes = None
        goal = None
        planners = [G.Planner(cyls, inflate_radius=r)
                    for r in dict.fromkeys((G.WIDE_INFLATE_RADIUS,
                                            G.DEFAULT_INFLATE_RADIUS))]
        for cand in goal_candidates(occ, rng, want_dir)[:8]:
            base = None
            for pl in planners:
                try:
                    base = pl.plan_centered(G.START_POSE[:2], cand)
                except ValueError:
                    base = None
                if base is not None:
                    break
            if base is None or not _interior(base):
                continue
            blen = G.path_length(base)
            if want_dir == 'above':
                # constant-difficulty rule
                if abs(blen - ROUTE_LEN_TARGET) > ROUTE_LEN_TOL:
                    continue
            else:
                # a 6 m band is unreachable behind/beside the start in even
                # clutter, so require winding relative to the straight line
                euclid = float(np.hypot(cand[0] - G.START_POSE[0],
                                        cand[1] - G.START_POSE[1]))
                if not (1.05 * euclid <= blen <= 1.9 * euclid):
                    continue
            # short behind/side routes rarely admit 3 near-parity alternatives
            need = MIN_ROUTES if want_dir == 'above' else 2
            rts = distinct_routes(cyls, cand)
            if len(rts) >= need:
                goal, routes = cand, rts
                break
        if goal is None:
            continue
        # the world must admit colour squares with reachable approach pockets
        from place_color_targets import pick_targets
        if pick_targets(cyls, np.random.default_rng(seed + 1), 3) is None:
            continue
        idx = args.base + made
        emit_world(tmpl_eval, block_eval, cyls,
                   BARN / 'worlds' / f'world_{idx}.world')
        emit_world(tmpl_fast, block_fast, cyls,
                   BARN / 'worlds_fast' / f'world_{idx}.world')
        np.save(BARN / 'grids' / f'grid_{idx}.npy', occ)
        meta.append({'idx': idx, 'seed': seed,
                     'goal_dir': want_dir,
                     'n_routes': len(routes),
                     'route_costs': [round(float(G.path_length(r)), 2)
                                     for r in routes],
                     'goal': [round(goal[0], 3), round(goal[1], 3)],
                     'n_cylinders': int(len(cyls))})
        made += 1

    if made < args.n:
        sys.exit(f'only {made}/{args.n} multipath worlds after {tries} '
                 'tries — loosen density or the route gates')
    holdout = [m['idx'] for m in meta[-args.holdout:]] if args.holdout else []
    (BARN / args.out_yaml).write_text(yaml.safe_dump({
        'generator': 'multipath-v3', 'generator_seed': args.seed,
        'n': args.n, 'min_routes': MIN_ROUTES,
        'cost_parity_max': COST_PARITY,
        'holdout': holdout,
        'train': [m['idx'] for m in meta if m['idx'] not in holdout],
        'worlds': meta}, sort_keys=False))
    print(f'{made} multipath worlds -> {args.base}..{args.base+made-1} '
          f'({tries} tries); holdout: {holdout}')
    dens = [m['n_cylinders'] for m in meta]
    print(f'cylinders per world: {min(dens)}..{max(dens)} '
          f'(mean {sum(dens)/len(dens):.0f}); routes: '
          f'{[m["n_routes"] for m in meta[:10]]}...')


if __name__ == '__main__':
    main()
