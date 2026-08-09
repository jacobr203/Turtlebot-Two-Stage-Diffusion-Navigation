#!/usr/bin/env python3
"""Generate dense-clutter BARN-style worlds (indices >= 300).

BARN's 300 worlds top out at 24.9% obstacle-field density (mean 12.7%).
These worlds push 25-40% to simulate heavily cluttered environments:
obstacles are seeded Bernoulli(p_seed) on the exact BARN field band
(28x30 cells; the 156-cell wall lattice is reused verbatim) and clumped
with one cellular-automaton pass (3x3 neighbourhood count >= 5 ->
occupied — the classic cave rule). Clumping is what keeps high density
navigable: blobs with channels between them, like furniture, rather
than salt-and-pepper chokepoints (pure Bernoulli is 0/60 plannable at
25% field density; CA at the same density accepts ~70%).

Every accepted world is verified PLANNABLE with the capstone Planner at
the TB3 inflation radius (0.21 m) — the same feasibility gate the
benchmark metadata applies to worlds 0-299.

Outputs are drop-in for the whole toolchain (run.sh grid auto-detection,
the two-stage subgoal planner, trial.py all key on world_NNN/grid_NNN):
  worlds/world_NNN.world        eval variant (RTF 1.0, ros_state plugin)
  worlds_fast/world_NNN.world   collection variant (unbounded RTF)
  grids/grid_NNN.npy            30x64 occupancy incl. walls (committed)
  dense_worlds.yaml             per-world density / cylinders / optimal len

Reproducible from --seed; .world files stay gitignored (regenerate by
re-running), grids + this script + the yaml are committed. NOT part of
the BARN benchmark: never mixed into metadata.yaml's train/holdout
splits — these are stress-test worlds.

Styles: 'dense' (CA-clumped blobs, tiered 25-40% density), 'winding'
(alternating baffles force S-curves), 'scatter' (furniture-scale pieces
dropped uniformly at random over the spawn-adjacent band — the cluttered-
indoor look: no CA clumping, no corridor structure; blocks rather than
salt-and-pepper singletons keep equal density plannable).

  .venv/bin/python barn/make_dense_worlds.py                   # 6 worlds, 300-305
  .venv/bin/python barn/make_dense_worlds.py --style winding   # 306-309
  .venv/bin/python barn/make_dense_worlds.py --style scatter   # 310+
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import grid as G  # noqa: E402

HERE = Path(__file__).parent

# (density_lo, density_hi, p_seed): windows are ACCEPTED field density;
# p_seed is the Bernoulli rate that lands in the window after one CA pass
# (measured acceptance at 0.21 m inflation: ~70% / ~27% / a few %).
TIERS = [
    ('dense',   0.25, 0.30, 0.45),
    ('denser',  0.30, 0.34, 0.50),
    ('extreme', 0.34, 0.40, 0.55),
]

# ── winding style ────────────────────────────────────────────────────────────
# Serpentine worlds: obstacles start just ahead of the SPAWN (BARN's own
# field begins 2+ m away from it), moderate density, and a route that must
# snake. Alternating baffles (partial cylinder walls anchored to a side
# wall, opening on the other side) guarantee the S-curves; light CA scatter
# between them adds clutter. Acceptance is by PATH GEOMETRY, not density:
# turn_count = string-pulled-path heading changes >= TURN_DEG, tortuosity =
# optimal length / 10.0 m straight line.
W_J_LO, W_J_HI = 23, 63          # fill band: y 3.5 m (ahead of spawn) -> 9.5
W_I_LO, W_I_HI = 1, 28           # interior columns between the side walls
SPAWN_CELL = (14.5, 19.5)        # START_POSE in grid coords
SPAWN_CLEAR = 3.2                # cells kept free around the spawn (~0.5 m)
W_BAFFLE_J = (25, 61)            # baffle anchor range (j)
W_BAFFLE_GAP = 6                 # min j spacing between baffles (0.9 m)
W_SCATTER_P = 0.28               # Bernoulli seed for the between-baffle clutter
W_DENSITY = (0.13, 0.20)         # 'cluttered but not too dense' window
W_MIN_TURNS = 6
W_MIN_TORT = 1.30
TURN_DEG = 25.0

# ── scatter style ────────────────────────────────────────────────────────────
# Cluttered-indoor worlds: furniture-scale pieces dropped uniformly at random
# over the same spawn-adjacent band as the winding style — no CA clumping and
# no baffle structure, so the distribution is genuinely random. Pieces are
# cell offsets on the lattice (0.15 m cells): singletons (small items),
# dominoes and 3-runs (benches/shelves), 2x2 blocks (tables/boxes) and Ls
# (corner furniture). Pieces may touch and merge into larger blobs. Same
# 0.21 m plannability gate; mild twist floors just to reject the occasional
# straight-shot layout.
S_PIECES = [
    [(0, 0)],
    [(0, 0), (1, 0)], [(0, 0), (0, 1)],
    [(0, 0), (1, 0), (2, 0)], [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (1, 1)],
]
S_DENSITY = (0.12, 0.20)         # accepted band density window
S_MIN_TURNS = 2
S_MIN_TORT = 1.05
# Indoor clutter should be dense but not impassable: acceptance also
# requires a route at 0.24 m inflation (i.e. through >= 0.30 m-class gaps),
# so nav2-scale planners always have a viable corridor. Winding worlds keep
# their deliberate 0.225 m squeezes; the official feasibility gate for
# metrics stays DEFAULT_INFLATE_RADIUS everywhere. Random draws almost
# never satisfy this by luck (0/4 in the first batch) — it must be gated.
S_WIDE_INFLATE = 0.24

CYL_RE = re.compile(r'\s*<model name=["\']unit_cylinder_\d+["\']>.*?</model>',
                    re.S)


def wall_and_field():
    """Wall lattice (occupied in EVERY benchmark grid) and the field band
    (occupied in ANY benchmark grid, walls excluded). Only worlds 0-299
    count — dense grids >= 300 must not feed back into the masks."""
    files = sorted(HERE.glob('grids/grid_[0-2][0-9][0-9].npy'))
    if len(files) != 300:
        sys.exit(f'expected 300 benchmark grids, found {len(files)}')
    gs = np.stack([np.load(f) for f in files])
    walls = np.minimum.reduce(gs)
    field = (np.maximum.reduce(gs) == 1) & (walls == 0)
    return walls.astype(np.uint8), field


def generate(rng, walls, field, p_seed):
    """One Bernoulli seed + one CA clump pass, walls held fixed."""
    occ = walls.copy()
    occ[field] = rng.random(int(field.sum())) < p_seed
    n = uniform_filter(occ.astype(float), size=3, mode='constant', cval=1.0) * 9
    occ = np.where(field, (n >= 5).astype(np.uint8), occ).astype(np.uint8)
    return occ


def winding_band(walls):
    """Boolean mask of the extended fill band (spawn-adjacent to goal end)."""
    band = np.zeros_like(walls, dtype=bool)
    band[W_I_LO:W_I_HI + 1, W_J_LO:W_J_HI + 1] = True
    band &= walls == 0
    return band


def generate_winding(rng, walls, band):
    """Alternating baffles + light CA scatter; spawn bubble kept free."""
    occ = walls.copy()

    # baffles: partial walls anchored to alternating sides, opening on the
    # other side — the route must snake through the offset openings
    n_baffles = int(rng.integers(5, 8))
    js, j = [], W_BAFFLE_J[0]
    while len(js) < n_baffles and j <= W_BAFFLE_J[1]:
        js.append(j)
        j += W_BAFFLE_GAP + int(rng.integers(0, 3))
    side_left = bool(rng.integers(0, 2))
    width = W_I_HI - W_I_LO + 1
    for j in js:
        length = int(round(width * rng.uniform(0.60, 0.78)))
        thick = 2 if rng.random() < 0.3 else 1
        rows = (range(W_I_LO, W_I_LO + length) if side_left
                else range(W_I_HI - length + 1, W_I_HI + 1))
        for i in rows:
            occ[i, j:j + thick] = 1
        side_left = not side_left

    # scatter between the baffles (CA-clumped, applied to free band cells)
    scatter = np.zeros_like(occ)
    free = band & (occ == 0)
    scatter[free] = rng.random(int(free.sum())) < W_SCATTER_P
    n = uniform_filter(scatter.astype(float), size=3, mode='constant') * 9
    occ = np.where(free, np.maximum((n >= 5).astype(np.uint8), 0), occ)

    # spawn bubble: the robot must materialize collision-free with room to go
    ii, jj = np.mgrid[0:occ.shape[0], 0:occ.shape[1]]
    bubble = np.hypot(ii - SPAWN_CELL[0], jj - SPAWN_CELL[1]) <= SPAWN_CLEAR
    occ[bubble & (walls == 0)] = 0
    return occ.astype(np.uint8)


def generate_scatter(rng, walls, band):
    """Uniformly random furniture pieces until a target band density is hit."""
    occ = walls.copy()
    target = rng.uniform(*S_DENSITY)
    for _ in range(600):
        if occ[band].mean() >= target:
            break
        piece = S_PIECES[int(rng.integers(len(S_PIECES)))]
        i0 = int(rng.integers(W_I_LO, W_I_HI + 1))
        j0 = int(rng.integers(W_J_LO, W_J_HI + 1))
        cells = [(i0 + di, j0 + dj) for di, dj in piece]
        if not all(0 <= i < occ.shape[0] and 0 <= j < occ.shape[1]
                   and band[i, j] for i, j in cells):
            continue
        for i, j in cells:
            occ[i, j] = 1

    # spawn bubble: the robot must materialize collision-free with room to go
    ii, jj = np.mgrid[0:occ.shape[0], 0:occ.shape[1]]
    bubble = np.hypot(ii - SPAWN_CELL[0], jj - SPAWN_CELL[1]) <= SPAWN_CLEAR
    occ[bubble & (walls == 0)] = 0
    return occ.astype(np.uint8)


def path_twist(path):
    """(turn_count, total_turn_deg, tortuosity) of a planner path."""
    d = np.diff(np.asarray(path), axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    keep = seg > 1e-6
    heads = np.arctan2(d[keep, 1], d[keep, 0])
    dh = np.degrees(np.abs(np.angle(np.exp(1j * np.diff(heads)))))
    tort = G.path_length(np.asarray(path)) / 10.0   # straight spawn->goal
    return int((dh >= TURN_DEG).sum()), float(dh.sum()), float(tort)


def load_template():
    src = HERE / 'worlds' / 'world_000.world'
    if not src.exists():
        sys.exit(f'{src} missing — run download_barn.sh + fixup_worlds.py first')
    text = src.read_text()
    m = CYL_RE.search(text)
    cyl_template = m.group(0)
    stripped = CYL_RE.sub('', text)
    if '<real_time_update_rate>1000' not in stripped:
        sys.exit('template: expected RTF cap 1000 in eval variant')
    return stripped, cyl_template


def world_text(stripped, cyl_template, occ):
    blocks = []
    for k, (x, y) in enumerate(G.arena_grid_to_cylinders(occ)):
        b = cyl_template.replace('unit_cylinder_0', f'unit_cylinder_{k}', 1)
        b = re.sub(r'<pose frame="">[^<]*</pose>',
                   f'<pose frame="">{x:.6f} {y:.6f} 0.000000 0.000000 '
                   f'0.000000 0.000000</pose>', b, count=1)
        blocks.append(b)
    anchor = stripped.index('<state world_name=')
    return stripped[:anchor] + ''.join(blocks) + '\n    ' + stripped[anchor:]


def write_world(idx, occ, stripped, cyl_template):
    text = world_text(stripped, cyl_template, occ)
    (HERE / 'worlds' / f'world_{idx}.world').write_text(text)
    (HERE / 'worlds_fast' / f'world_{idx}.world').write_text(
        text.replace('<real_time_update_rate>1000',
                     '<real_time_update_rate>0', 1))
    np.save(HERE / 'grids' / f'grid_{idx}.npy', occ)


def next_free_idx():
    taken = [int(f.stem.split('_')[1]) for f in HERE.glob('grids/grid_*.npy')]
    return max(max(taken) + 1, 300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--style', choices=['dense', 'winding', 'scatter'],
                    default='dense')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--start-idx', type=int, default=None,
                    help='default: next free index >= 300')
    ap.add_argument('--per-tier', type=int, default=2,
                    help='dense style: worlds per density tier')
    ap.add_argument('--count', type=int, default=4,
                    help='winding/scatter styles: number of worlds')
    ap.add_argument('--max-draws', type=int, default=800)
    args = ap.parse_args()

    walls, field = wall_and_field()
    stripped, cyl_template = load_template()
    (HERE / 'worlds_fast').mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    idx = args.start_idx if args.start_idx is not None else next_free_idx()
    meta = {}

    if args.style == 'dense':
        for name, lo, hi, p_seed in TIERS:
            found = 0
            for _ in range(args.max_draws):
                occ = generate(rng, walls, field, p_seed)
                d = float(occ[field].mean())
                if not (lo <= d < hi):
                    continue
                planner = G.Planner(G.arena_grid_to_cylinders(occ),
                                    inflate_radius=G.DEFAULT_INFLATE_RADIUS)
                path = planner.plan(G.START_POSE[:2], G.GOAL_XY)
                if path is None:
                    continue
                write_world(idx, occ, stripped, cyl_template)
                meta[idx] = dict(style='dense', tier=name, seed=args.seed,
                                 field_density=round(d, 4),
                                 cylinders=int(occ.sum()),
                                 optimal_len_ours=round(G.path_length(path), 3))
                print(f"world_{idx}: {name}  density {d*100:.1f}%  "
                      f"{int(occ.sum())} cylinders  "
                      f"optimal {meta[idx]['optimal_len_ours']} m")
                idx += 1
                found += 1
                if found == args.per_tier:
                    break
            if found < args.per_tier:
                print(f"[warn] tier '{name}': only {found}/{args.per_tier} "
                      f"plannable worlds in {args.max_draws} draws")
    else:
        band = winding_band(walls)
        gen, (d_lo, d_hi), min_turns, min_tort = {
            'winding': (generate_winding, W_DENSITY, W_MIN_TURNS, W_MIN_TORT),
            'scatter': (generate_scatter, S_DENSITY, S_MIN_TURNS, S_MIN_TORT),
        }[args.style]
        found = 0
        for _ in range(args.max_draws):
            occ = gen(rng, walls, band)
            d = float(occ[band].mean())
            if not (d_lo <= d < d_hi):
                continue
            planner = G.Planner(G.arena_grid_to_cylinders(occ),
                                inflate_radius=G.DEFAULT_INFLATE_RADIUS)
            path = planner.plan(G.START_POSE[:2], G.GOAL_XY)
            if path is None:
                continue
            turns, total_deg, tort = path_twist(path)
            if turns < min_turns or tort < min_tort:
                continue
            if args.style == 'scatter':
                wide = G.Planner(G.arena_grid_to_cylinders(occ),
                                 inflate_radius=S_WIDE_INFLATE)
                try:
                    if wide.plan(G.START_POSE[:2], G.GOAL_XY) is None:
                        continue
                except ValueError:   # start/goal inside the fatter inflation
                    continue
            write_world(idx, occ, stripped, cyl_template)
            meta[idx] = dict(style=args.style, seed=args.seed,
                             band_density=round(d, 4),
                             cylinders=int(occ.sum()),
                             optimal_len_ours=round(G.path_length(path), 3),
                             turns_ge_25deg=turns,
                             total_turn_deg=round(total_deg, 1),
                             tortuosity=round(tort, 3))
            print(f"world_{idx}: {args.style}  density {d*100:.1f}%  "
                  f"{int(occ.sum())} cyl  optimal "
                  f"{meta[idx]['optimal_len_ours']} m  "
                  f"turns {turns}  twist {total_deg:.0f} deg  tort {tort:.2f}")
            idx += 1
            found += 1
            if found == args.count:
                break
        if found < args.count:
            print(f"[warn] {args.style}: only {found}/{args.count} accepted "
                  f"in {args.max_draws} draws")

    out = HERE / 'dense_worlds.yaml'
    existing = yaml.safe_load(out.read_text()) if out.exists() else {}
    worlds = existing.get('worlds', {}) if isinstance(existing, dict) else {}
    worlds.update(meta)
    out.write_text(yaml.safe_dump(
        dict(inflate_radius=G.DEFAULT_INFLATE_RADIUS, worlds=worlds),
        sort_keys=True))
    print(f'wrote {out} ({len(meta)} new, {len(worlds)} total)')


if __name__ == '__main__':
    main()
