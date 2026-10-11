#!/usr/bin/env python3
"""Generate color-target variants of the BARN worlds (color_test_plan.md §2).

For each input world, place one 2"x2" scavenger square per requested color
(--colors, default light_green,purple,orange; lime_green and sky_blue also
available) on the sides of distinct cylinders, facing free space:

  barn/worlds/world_NNN.world  ->  barn/worlds_color/world_NNN.world
                                   barn/targets/targets_NNN.yaml

Placement is seeded per world (base seed + idx) and constraint-checked on
the same grid machinery the expert uses:

  - approach pocket: the pose 0.45 m in front of the square (and the
    midpoint) has enough clearance for the robot;
  - reachable: A* from the canonical start pose to the approach pose
    succeeds at the standard inflate radius;
  - separation: squares are >= MIN_SEP apart, one per cylinder.

The squares are VISUAL-ONLY models (no <collision>): the camera sees them,
the lidar and physics do not — navigation is unchanged, per the plan.
Mounting height is BRIO_HEIGHT_M +/-15% (uniform), per the agreed wiggle
room. The targets yaml is ground truth for scoring VALIDATION and analysis
only — the robot never reads it.

Usage (from capstone/, via .venv/bin/python):
  python barn/place_color_targets.py --idx 306
  python barn/place_color_targets.py --all
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import grid as G  # noqa: E402

BARN = Path(__file__).resolve().parent
BRIO_HEIGHT_M = 0.12          # burger_brio camera_link z (model.sdf)
HEIGHT_JITTER = 0.15          # +/-15% mounting-height wiggle room
SQUARE = 0.0508               # 2" side
CYL_R = 0.075
APPROACH_DIST = 0.45          # seeker goal pose, in front of the square
MIN_SEP = 2.0                 # between squares [m]
START = (-2.25, 3.0)          # canonical BARN start (metadata.yaml)
BASE_SEED = 42

COLORS = {                    # name -> (r, g, b) 0-1, plan §1
    'light_green': (0.565, 0.933, 0.565),
    'purple':      (0.502, 0.0, 0.502),
    'orange':      (1.0, 0.498, 0.0),
    'lime_green':  (0.196, 0.804, 0.196),   # #32CD32
    'sky_blue':    (0.424, 0.651, 0.804),   # #6CA6CD, slightly-darker sky blue
}
DEFAULT_COLORS = ('light_green', 'purple', 'orange')

MODEL_TMPL = """    <model name="color_target_{name}">
      <static>1</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.4f}</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box><size>0.002 {side} {side}</size></box>
          </geometry>
          <material>
            <ambient>{r} {g} {b} 1</ambient>
            <diffuse>{r} {g} {b} 1</diffuse>
            <emissive>{re} {ge} {be} 1</emissive>
          </material>
        </visual>
      </link>
    </model>
"""

# Perimeter walls: BARN courses are open cylinder fields — a frontier-
# exploring seeker walks straight out of one into the infinite void
# (observed live 2026-07-20, robot at (-7.8, 16.6)). Real rooms are walled,
# which is exactly what makes frontier exploration terminate; the color
# variants model that. Lidar-visible, collidable, 0.3 m tall.
WALL_TMPL = """    <model name="color_arena_wall_{name}">
      <static>1</static>
      <pose>{x:.3f} {y:.3f} 0.15 0 0 {yaw:.4f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{length:.3f} 0.05 0.3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{length:.3f} 0.05 0.3</size></box></geometry>
          <material>
            <ambient>0.35 0.35 0.35 1</ambient>
            <diffuse>0.35 0.35 0.35 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""
WALL_MARGIN = 1.2   # [m] beyond the cylinder/start/goal extent


def wall_blocks(cylinders):
    """Four perimeter walls around cylinders + start + goal (+margin)."""
    xs = np.concatenate([cylinders[:, 0], [START[0], -2.25]])
    ys = np.concatenate([cylinders[:, 1], [START[1], 13.0]])
    x0, x1 = xs.min() - WALL_MARGIN, xs.max() + WALL_MARGIN
    y0, y1 = ys.min() - WALL_MARGIN, ys.max() + WALL_MARGIN
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    return ''.join([
        WALL_TMPL.format(name='south', x=cx, y=y0, yaw=0.0, length=w + 0.1),
        WALL_TMPL.format(name='north', x=cx, y=y1, yaw=0.0, length=w + 0.1),
        WALL_TMPL.format(name='west', x=x0, y=cy, yaw=1.5708, length=h + 0.1),
        WALL_TMPL.format(name='east', x=x1, y=cy, yaw=1.5708, length=h + 0.1),
    ])


def pick_targets(cylinders, rng, n):
    """Choose n (cylinder_xy, outward_angle) pairs satisfying the plan's
    constraints. Returns list of dicts or None if the world can't host n."""
    planner = G.Planner(cylinders, inflate_radius=G.DEFAULT_INFLATE_RADIUS)
    # squares must face the INSIDE of the maze: the approach pocket has to
    # sit within the cylinder field's extent (+small slack), not out in the
    # wall-margin ring an edge cylinder may
    # still host a square, but only facing inward)
    in_lo = cylinders.min(axis=0) - 0.20
    in_hi = cylinders.max(axis=0) + 0.20
    order = rng.permutation(len(cylinders))
    chosen = []
    for ci in order:
        if len(chosen) == n:
            break
        cx, cy = cylinders[ci]
        if any(math.hypot(cx - t['cx'], cy - t['cy']) < MIN_SEP
               for t in chosen):
            continue
        for theta in rng.permutation(8) * (math.pi / 4):
            dx, dy = math.cos(theta), math.sin(theta)
            approach = (cx + (CYL_R + APPROACH_DIST) * dx,
                        cy + (CYL_R + APPROACH_DIST) * dy)
            if not (in_lo[0] <= approach[0] <= in_hi[0]
                    and in_lo[1] <= approach[1] <= in_hi[1]):
                continue        # faces the outside of the maze
            mid = (cx + (CYL_R + APPROACH_DIST / 2) * dx,
                   cy + (CYL_R + APPROACH_DIST / 2) * dy)
            # pocket: nearest cylinder CENTER must leave robot room. The
            # host cylinder itself is the binding constraint at the
            # midpoint (0.30 m from center) — require the rest of the
            # pocket to be as open as the host allows plus margin.
            if planner.clearance_at(approach) < CYL_R + 0.30:
                continue
            if planner.clearance_at(mid) < CYL_R + 0.20:
                continue
            try:
                if planner.plan(START, approach) is None:
                    continue
            except ValueError:      # approach outside the arena bounds
                continue
            chosen.append({'cx': float(cx), 'cy': float(cy),
                           'theta': float(theta),
                           'approach': [round(float(approach[0]), 3),
                                        round(float(approach[1]), 3)]})
            break
    return chosen if len(chosen) == n else None


def build_world(idx, worlds_dir, out_worlds, out_targets, use_colors):
    src = worlds_dir / f'world_{idx:03d}.world'
    if not src.exists():
        return f'world_{idx:03d}: missing'
    cylinders = G.parse_world_cylinders(src)
    if len(cylinders) < len(use_colors):
        return f'world_{idx:03d}: <{len(use_colors)} cylinders'
    rng = np.random.default_rng(BASE_SEED + idx)
    picks = pick_targets(cylinders, rng, len(use_colors))
    if picks is None:
        return f'world_{idx:03d}: could not place {len(use_colors)} targets'

    colors = list(use_colors)
    rng.shuffle(colors)
    blocks, gt = [], []
    for color, t in zip(colors, picks):
        r, g, b = COLORS[color]
        z = BRIO_HEIGHT_M * rng.uniform(1 - HEIGHT_JITTER, 1 + HEIGHT_JITTER)
        # face center sits just proud of the cylinder surface
        fx = t['cx'] + (CYL_R + 0.002) * math.cos(t['theta'])
        fy = t['cy'] + (CYL_R + 0.002) * math.sin(t['theta'])
        blocks.append(MODEL_TMPL.format(
            name=color, x=fx, y=fy, z=z, yaw=t['theta'], side=SQUARE,
            r=r, g=g, b=b, re=0.35 * r, ge=0.35 * g, be=0.35 * b))
        gt.append({'color': color, 'x': round(float(fx), 4),
                   'y': round(float(fy), 4), 'z': round(float(z), 4),
                   'normal_yaw': round(t['theta'], 4),
                   'approach': t['approach'],
                   'cylinder': [t['cx'], t['cy']]})

    text = src.read_text()
    tail = text.rindex('</world>')
    (out_worlds / src.name).write_text(
        text[:tail] + ''.join(blocks) + wall_blocks(cylinders) + text[tail:])
    (out_targets / f'targets_{idx:03d}.yaml').write_text(yaml.safe_dump(
        {'world': idx, 'seed': BASE_SEED + idx, 'brio_height': BRIO_HEIGHT_M,
         'targets': gt}, sort_keys=False))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, nargs='+')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--worlds-dir', default=BARN / 'worlds', type=Path)
    ap.add_argument('--colors', default=','.join(DEFAULT_COLORS),
                    help='comma-separated square colors to place '
                         f'(known: {", ".join(COLORS)}). Regenerating with a '
                         'different set OVERWRITES the worlds_color/targets '
                         'output for those indices.')
    args = ap.parse_args()
    if not args.idx and not args.all:
        ap.error('need --idx or --all')
    use_colors = [c.strip() for c in args.colors.split(',') if c.strip()]
    unknown = [c for c in use_colors if c not in COLORS]
    if unknown or not use_colors:
        ap.error(f'unknown colors {unknown} (known: {list(COLORS)})')

    out_worlds = BARN / 'worlds_color'
    out_targets = BARN / 'targets'
    out_worlds.mkdir(exist_ok=True)
    out_targets.mkdir(exist_ok=True)

    if args.all:
        idxs = sorted(int(p.stem.split('_')[1])
                      for p in args.worlds_dir.glob('world_*.world'))
    else:
        idxs = args.idx
    fails = []
    for i in idxs:
        err = build_world(i, args.worlds_dir, out_worlds, out_targets,
                          use_colors)
        if err:
            fails.append(err)
            print(err)
    print(f'{len(idxs) - len(fails)}/{len(idxs)} worlds generated '
          f'-> {out_worlds}')
    if fails:
        print(f'{len(fails)} FAILED — these worlds have no color variant; '
              'the test runner must refuse them rather than fall back '
              'to the plain world')


if __name__ == '__main__':
    main()
