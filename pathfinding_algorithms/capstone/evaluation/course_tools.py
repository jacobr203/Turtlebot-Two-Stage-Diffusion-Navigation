#!/usr/bin/env python3
"""Design, measure and verify the physical courses for the real-robot trials.

A course layout (common/course.py) is the single description of a course:
the build sheet, the two-stage policy's prior map, and the source of the
optimal route length and timeout used in scoring.

  measure    difficulty on the BARN tier scale, optimal route, timeout
  render     top-down PNG: lattice, walls, tubes, start, goal circle, route
  marks      tape-mark list for building (tube centres in cm from the start)
  sheet      one printable build sheet (Markdown) for several courses
  check-map  compare a Cartographer map of the BUILT course with the layout
  design     search random layouts for one that lands in a target tier

Colour-task arenas (common/arena.py) — a room bounded by furniture, with
thin poles and coloured squares:

  arena-from-map   pole centres + boundary from a Cartographer map of the
                   built arena -> a new arena file (prune non-course blobs)
  arena-measure    free area, open space, narrowest gaps, square pockets,
                   pole tape marks, and a figure per placement

  .venv/bin/python evaluation/course_tools.py measure evaluation/courses/A.yaml
  .venv/bin/python evaluation/course_tools.py design --tier MODERATE \\
      --tubes 22 --name B --id 9002 -o evaluation/courses/B.yaml
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from barn import world_metrics as W                           # noqa: E402
from common import arena as A                                 # noqa: E402
from common import course as C                                # noqa: E402
from common import grid as G                                  # noqa: E402

CAPSTONE = Path(__file__).resolve().parents[1]
COURSES_DIR = CAPSTONE / 'evaluation' / 'courses'
RENDER_DIR = CAPSTONE / 'results' / 'real' / 'courses'

TUBE_RADIUS_M = 0.075
# check-map: a built tube shows up as occupied cells on its lidar-facing
# surface, TUBE_RADIUS_M from the design centre. Allow half a map cell of
# quantization plus a few centimetres of placement error either way.
MAP_SURFACE_TOL_M = 0.07
STRAY_CLEAR_M = 0.20 # occupied cells this far from every design obstacle
# design: the clutter field starts clear of the robot's nose and ends
# before the success circle, so succeeding means crossing all of it
FIELD_START_X = 0.6
GOAL_CLEAR_M = 1.2
DESIGN_TRIES = 4000
# arena-from-map: a 4.4 cm pole covers 1-3 cells of a 5 cm map; anything
# wider than this is boundary (furniture, walls)
POLE_MAX_EXTENT_M = 0.16
POLE_MAX_CELLS = 12
POLE_EXCLUDE_M = 0.10 # a pole's own cells, when measuring its gaps
SQUARE_POCKET_M = 0.6 # plan: >= 0.6 m of free floor in front of a square
ROBOT_WIDTH_M = 0.178
ARENA_DIR = CAPSTONE / 'evaluation' / 'arenas'
ARENA_RENDER_DIR = CAPSTONE / 'results' / 'real' / 'arenas'
# Narrowest passage a design may contain, on the world_metrics scale.
# 0.42 m is BARN's typical HARD-tier gap and the tightest the 0.21 m
# planning inflation passes; below it lies the EXTREME 0.30 m pinch.
DESIGN_MIN_GAP_M = 0.42


def _set_rows():
    return json.loads(W.OUT.read_text())


def measure(course):
    """Difficulty metrics, tier placement and route for one course."""
    m = W.measure_grid(C.arena_grid(course))
    score, rank, tier = W.place(m, _set_rows())
    path, length, inflate = C.optimal_route(course)
    return {
        'name': course['name'], 'id': course['id'],
        'tubes': len(course['tubes']),
        'occ': m['occ'], 'min_gap_m': m['min_gap_m'],
        'open_frac': m['open_frac'], 'score': score,
        'rank': rank, 'tier': tier, 'tier_target': course['tier_target'],
        'route': path, 'optimal_len_m': length, 'inflate_m': inflate,
        'timeout_s': C.timeout_s(length) if length else None,
    }


def _print_measure(r):
    total = len(_set_rows())
    print(f"course {r['name']} (id {r['id']}): {r['tubes']} tubes")
    print(f"  occupancy {r['occ'] * 100:.1f} %   min gap {r['min_gap_m']:.2f} m"
          f"   open space {r['open_frac'] * 100:.1f} %")
    target = (f"   (target {r['tier_target']})"
              if r['tier_target'] else '')
    print(f"  score {r['score']:.3f} -> {r['tier']}, would rank "
          f"{r['rank']}/{total}{target}")
    if r['route'] is None:
        print('  NO ROUTE start -> goal at any inflation: not traversable')
    else:
        print(f"  optimal route {r['optimal_len_m']:.2f} m "
              f"(inflation {r['inflate_m']} m), timeout {r['timeout_s']:.0f} s")


def render(course, out_png, result=None, map_points=None):
    """Top-down figure of a course, optionally with a SLAM map overlaid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    r = result or measure(course)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    p = course['lattice_m']
    xs = np.arange(round(course['back_x'] / p), round(course['end_x'] / p) + 1) * p
    ys = np.arange(-round(course['side_y'] / p), round(course['side_y'] / p) + 1) * p
    gx, gy = np.meshgrid(xs, ys)
    ax.plot(gx.ravel(), gy.ravel(), '.', color='0.85', ms=2, zorder=0)
    if map_points is not None and len(map_points):
        ax.plot(map_points[:, 0], map_points[:, 1], 's', color='#d62728',
                ms=1.5, alpha=0.6, label='SLAM map (occupied)', zorder=1)
    b, s, e = course['back_x'], course['side_y'], course['end_x']
    ax.plot([e, b, b, e], [s, s, -s, -s], color='0.2', lw=3, label='walls')
    for k, (x, y) in enumerate(course['tubes']):
        ax.add_patch(Circle((x, y), TUBE_RADIUS_M, color='#8c6d31', zorder=2))
        ax.annotate(f't{k:02d}', (x, y), fontsize=6, ha='center',
                    va='center', color='white', zorder=3)
    ax.add_patch(Circle(course['goal'], 1.0, fill=False, ls='--',
                        color='#2ca02c', label='success circle (1 m)'))
    ax.plot(*course['goal'], '*', color='#2ca02c', ms=12)
    ax.annotate('', xy=(0.3, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2, color='#1f77b4'))
    ax.plot(0, 0, 'o', color='#1f77b4', ms=8, label='start')
    if r['route'] is not None:
        ax.plot(r['route'][:, 0], r['route'][:, 1], color='#1f77b4', lw=1.2,
                label=f"A* route {r['optimal_len_m']:.2f} m")
    ax.set_aspect('equal')
    ax.set_xlabel('x forward from start [m]')
    ax.set_ylabel('y left [m]')
    ax.set_title(f"Course {course['name']}: {r['tier']} "
                 f"(score {r['score']:.2f}, min gap {r['min_gap_m']:.2f} m, "
                 f"{r['tubes']} tubes)")
    ax.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def marks(course):
    """Tape marks sorted along the course: (id, x_cm, y_cm)."""
    rows = [(tid, round(x * 100), round(y * 100))
            for tid, (x, y) in zip(C.tube_ids(course), course['tubes'])]
    return sorted(rows, key=lambda r: (r[1], r[2]))


def sheet(courses):
    """Printable build sheet: parts, fixed marks, and every tube per course."""
    out = ['# Course build sheet', '',
           'All positions in cm from the START mark: x forward (toward the '
           'goal), y to the robot\'s LEFT. Tubes: 15 cm diameter, '
           '35-40 cm tall, matte, not black. Put a small tape cross at each '
           'centre.', '',
           f"Tubes needed (courses are built one at a time): "
           f"**{max(len(c['tubes']) for c in courses)}**", '']
    for c in courses:
        r = measure(c)
        b, sy, e = (round(c['back_x'] * 100), round(c['side_y'] * 100),
                    round(c['end_x'] * 100))
        gx, gy = (round(v * 100) for v in c['goal'])
        out += [f"## Course {c['name']} — {r['tier']} "
                f"({len(c['tubes'])} tubes, min gap {r['min_gap_m']:.2f} m, "
                f"route {r['optimal_len_m']:.2f} m, timeout "
                f"{r['timeout_s']:.0f} s)", '',
                f'- Back wall: x = {b}, from y = -{sy} to +{sy} '
                '(operator stands behind it)',
                f'- Side walls: y = ±{sy}, from x = {b} to {e}; far end open',
                '- Start: robot centre on (0, 0), facing +x — tape the '
                'footprint outline and a heading line',
                f'- Goal: cross at ({gx}, {gy}); success circle radius 100 '
                '(tape dots)', '',
                '| tube | x | y |', '|---|---:|---:|']
        out += [f'| {tid} | {x} | {y} |' for tid, x, y in marks(c)]
        out += ['', f"Figure: results/real/courses/{c['name']}.png", '']
    return '\n'.join(out)


def check_map(course, map_yaml):
    """Per-tube placement check and stray-obstacle list from a built map."""
    from scipy.spatial import cKDTree
    from deploy.subgoal_planner import _map_yaml_to_points
    pts = _map_yaml_to_points(map_yaml)
    tree = cKDTree(pts)
    tubes = []
    for tid, (x, y) in zip(C.tube_ids(course), course['tubes']):
        d, _ = tree.query((x, y))
        ok = abs(d - TUBE_RADIUS_M) <= MAP_SURFACE_TOL_M
        tubes.append((tid, x, y, float(d), ok))
    inside = ((pts[:, 0] > course['back_x'] + 0.1)
              & (pts[:, 0] < course['end_x'])
              & (np.abs(pts[:, 1]) < course['side_y'] - 0.1))
    if len(course['tubes']):
        d_tube, _ = cKDTree(course['tubes']).query(pts)
    else:
        d_tube = np.full(len(pts), np.inf)
    stray = pts[inside & (d_tube > STRAY_CLEAR_M)]
    return tubes, stray, pts


def _write_layout(path, course_dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tubes = course_dict.pop('tubes')
    text = yaml.safe_dump(course_dict, sort_keys=False)
    text += 'tubes:\n' + ''.join(f'  - [{x:.2f}, {y:.2f}]\n' for x, y in tubes)
    path.write_text(text)


def design(tier, n_tubes, name, cid, seed, walls, goal,
           min_gap=DESIGN_MIN_GAP_M):
    """Random lattice layouts in the clutter band until one lands in `tier`
    and is traversable. Returns (layout dict, measure result) or None."""
    rng = np.random.default_rng(seed)
    p = G.CELL
    x_hi = round((goal[0] - GOAL_CLEAR_M) / p) * p
    xs = np.arange(round(FIELD_START_X / p), round(x_hi / p) + 1) * p
    ys = np.arange(-round(walls['side_y'] / p) + 1,
                   round(walls['side_y'] / p)) * p
    cells = np.array([(x, y) for x in xs for y in ys])
    best = None
    for _ in range(DESIGN_TRIES):
        pick = cells[rng.choice(len(cells), size=n_tubes, replace=False)]
        layout = {'name': name, 'id': cid, 'tier_target': tier,
                  'lattice_m': p, 'walls': walls,
                  'clutter_x': [FIELD_START_X, round(float(x_hi), 2)],
                  'start': [0.0, 0.0, 0.0],
                  'goal': list(goal),
                  'tubes': [[round(float(x), 2), round(float(y), 2)]
                            for x, y in pick]}
        course = C.from_dict(layout)
        m = W.measure_grid(C.arena_grid(course))
        _, _, got = W.place(m, _set_rows())
        if got != tier or m['min_gap_m'] < min_gap - 1e-6:
            continue
        r = measure(course)
        if r['route'] is None:
            continue
        # prefer the design whose route is least trivial: longest detour
        if best is None or r['optimal_len_m'] > best[1]['optimal_len_m']:
            best = (layout, r)
            if r['optimal_len_m'] > 1.08 * math.hypot(*goal):
                break
    return best


def extract_poles(map_yaml, with_evidence=False):
    """Centres of small islands of non-free cells: the arena's thin poles.

    A 4.4 cm pole is narrower than one 5 cm map cell. Cartographer marks
    it OCCUPIED where beams hit it, but beams grazing past also mark its
    cell free, so many poles end up UNKNOWN (between the thresholds)
    instead. Either way the pole is a 1-3 cell island of not-free cells
    entirely surrounded by free floor; the island's mean sits within about
    half a cell of the centre. Anything larger, or connected to the walls
    or the unexplored outside, is boundary. Chair and table legs look the
    same, so the list needs a human prune.

    Returns (N, 2) centres, or with with_evidence also a list saying
    whether each island held an occupied cell ('occupied') or only
    unknown ones ('unknown', weaker)."""
    from scipy.ndimage import find_objects, label
    occupied, free, res, origin, h = A.read_map(map_yaml)
    lab, _ = label(~free, structure=np.ones((3, 3)))
    found = []
    for k, sl in enumerate(find_objects(lab), start=1):
        if (sl[0].start == 0 or sl[1].start == 0
                or sl[0].stop == lab.shape[0] or sl[1].stop == lab.shape[1]):
            continue # touches the map edge: unexplored outside, not a pole
        rows, cols = np.nonzero(lab[sl] == k)
        extent = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * res
        if extent <= POLE_MAX_EXTENT_M and len(rows) <= POLE_MAX_CELLS:
            rr, cc = rows + sl[0].start, cols + sl[1].start
            xy = A.cells_to_xy(rr, cc, res, origin, h).mean(axis=0)
            kind = 'occupied' if occupied[rr, cc].any() else 'unknown'
            found.append((round(xy[0], 2), round(xy[1], 2), kind))
    found.sort()
    poles = np.array([(x, y) for x, y, _ in found]).reshape(-1, 2)
    return (poles, [kd for *_, kd in found]) if with_evidence else poles


def arena_from_map(map_yaml, name, arena_id, pole_d, out_dir=ARENA_DIR):
    """Write <name>.yaml + a copy of the map into out_dir."""
    import shutil
    map_yaml = Path(map_yaml)
    info = yaml.safe_load(open(map_yaml))
    out_dir.mkdir(parents=True, exist_ok=True)
    pgm = out_dir / f'{name}_map.pgm'
    shutil.copyfile(map_yaml.parent / info['image'], pgm)
    info['image'] = pgm.name
    (out_dir / f'{name}_map.yaml').write_text(yaml.safe_dump(info))
    poles, kinds = extract_poles(map_yaml, with_evidence=True)
    text = (f'name: {name}\nid: {arena_id}\nmap: {name}_map.yaml\n'
            f'pole_diameter_m: {pole_d}\n'
            '# extracted from the map; delete any line that is not a course\n'
            '# pole (chair/table legs look the same), then tape-check a few\n'
            'poles:\n'
            + ''.join(f'  - [{x:.2f}, {y:.2f}]  # p{k:02d}'
                      + ('  (unknown cells only: weaker evidence)'
                         if kind == 'unknown' else '') + '\n'
                      for k, ((x, y), kind) in enumerate(zip(poles, kinds)))
            + 'placements:\n'
              '  # one list per square arrangement; x, y in m from the start\n'
              '  # mark, facing_deg = direction the face points (0 = +x)\n'
              '  P1: []\n'
              '  # - {color: purple, x: 2.10, y: -0.80, facing_deg: 90, '
              'mount: wall}\n'
              '  # - {color: orange, x: 1.20, y: 0.37, facing_deg: 180, '
              'mount: pole, pole: p03}\n')
    out = out_dir / f'{name}.yaml'
    out.write_text(text)
    return out, poles


def _ray_free(occupied, res, origin, h, poles, pole_r, x, y, ang,
              max_m=3.0, skip_m=0.08):
    """Free distance from (x, y) along ang before a map cell or pole."""
    step = res / 2
    d = skip_m
    while d < max_m:
        px, py = x + d * math.cos(ang), y + d * math.sin(ang)
        c = int((px - origin[0]) / res)
        r = h - 1 - int((py - origin[1]) / res)
        if not (0 <= r < occupied.shape[0] and 0 <= c < occupied.shape[1]):
            return d
        if occupied[r, c]:
            return d
        if len(poles) and (np.hypot(*(poles - (px, py)).T) < pole_r).any():
            return d
        d += step
    return max_m


def measure_arena(arena):
    """Descriptive geometry of an arena and each placement's squares."""
    from scipy.ndimage import distance_transform_edt
    occupied, free, res, origin, h = A.read_map(arena['map'])
    clearance = distance_transform_edt(~occupied) * res
    out = {'free_area_m2': float(free.sum() * res * res),
           'open_frac': float((clearance[free] >= 0.5).mean())
           if free.any() else 0.0,
           'poles': len(arena['poles'])}
    out['pole_density'] = out['poles'] / max(out['free_area_m2'], 1e-9)
    rows, cols = np.nonzero(occupied)
    occ_xy = A.cells_to_xy(rows, cols, res, origin, h)
    r = arena['pole_diameter_m'] / 2
    gaps = []
    ids = A.pole_ids(arena)
    for k, p in enumerate(arena['poles']):
        for j in range(k + 1, len(arena['poles'])):
            q = arena['poles'][j]
            gaps.append((float(np.hypot(*(p - q)) - 2 * r),
                         f'{ids[k]}-{ids[j]}'))
        d = np.hypot(*(occ_xy - p).T)
        d = d[d > POLE_EXCLUDE_M]
        if len(d):
            gaps.append((float(d.min() - r - res / 2), f'{ids[k]}-boundary'))
    out['gaps'] = sorted(gaps)
    out['squares'] = {}
    for name, squares in arena['placements'].items():
        rows_ = []
        for s in squares:
            pocket = _ray_free(occupied, res, origin, h, arena['poles'], r,
                               s['x'], s['y'], math.radians(s['facing_deg']))
            rows_.append((s['color'], s['mount'], pocket))
        out['squares'][name] = rows_
    return out


def render_arena(arena, placement, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    occupied, free, res, origin, h = A.read_map(arena['map'])
    img = np.full(occupied.shape, 0.75)
    img[free], img[occupied] = 1.0, 0.15
    extent = (origin[0], origin[0] + occupied.shape[1] * res,
              origin[1], origin[1] + h * res)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, extent=extent,
              interpolation='nearest')
    r = arena['pole_diameter_m'] / 2
    for pid, (x, y) in zip(A.pole_ids(arena), arena['poles']):
        ax.add_patch(Circle((x, y), max(r, 0.03), color='#8c6d31'))
        ax.annotate(pid, (x, y), xytext=(3, 3), textcoords='offset points',
                    fontsize=6)
    swatch = {'light_green': '#90ee90', 'purple': '#800080',
              'orange': '#ff7f00', 'lime_green': '#32cd32',
              'sky_blue': '#6ca6cd'}
    for s in arena['placements'].get(placement, []):
        a = math.radians(s['facing_deg'])
        ax.plot(s['x'], s['y'], 's', ms=9, color=swatch[s['color']],
                mec='black')
        ax.annotate('', xy=(s['x'] + 0.35 * math.cos(a),
                            s['y'] + 0.35 * math.sin(a)),
                    xytext=(s['x'], s['y']),
                    arrowprops=dict(arrowstyle='->', color=swatch[s['color']]))
    ax.annotate('', xy=(0.3, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2, color='#1f77b4'))
    ax.plot(0, 0, 'o', color='#1f77b4', ms=8)
    ax.set_title(f"Arena {arena['name']}, placement {placement}")
    ax.set_xlabel('x forward from start [m]')
    ax.set_ylabel('y left [m]')
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for cmd in ('measure', 'render', 'marks', 'sheet'):
        s = sub.add_parser(cmd)
        s.add_argument('layout', nargs='+')
    sub.choices['sheet'].add_argument(
        '-o', '--out', type=Path, default=RENDER_DIR / 'build_sheet.md')
    s = sub.add_parser('check-map')
    s.add_argument('layout')
    s.add_argument('map_yaml')
    s = sub.add_parser('arena-from-map')
    s.add_argument('map_yaml')
    s.add_argument('--name', required=True)
    s.add_argument('--id', type=int, required=True)
    s.add_argument('--pole-diameter', type=float, default=0.04445)
    s = sub.add_parser('arena-measure')
    s.add_argument('arena')
    s = sub.add_parser('design')
    s.add_argument('--tier', required=True,
                   choices=['EASY', 'MODERATE', 'HARD', 'EXTREME'])
    s.add_argument('--tubes', type=int, required=True)
    s.add_argument('--name', required=True)
    s.add_argument('--id', type=int, required=True)
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--min-gap', type=float, default=DESIGN_MIN_GAP_M)
    s.add_argument('--side-y', type=float, default=1.35)
    s.add_argument('--back-x', type=float, default=-0.6)
    s.add_argument('--end-x', type=float, default=4.8)
    s.add_argument('--goal', type=float, nargs=2, default=(4.2, 0.0))
    s.add_argument('-o', '--out', type=Path)
    args = ap.parse_args()

    if args.cmd == 'measure':
        for f in args.layout:
            _print_measure(measure(C.load(f)))
    elif args.cmd == 'render':
        for f in args.layout:
            course = C.load(f)
            out = render(course, RENDER_DIR / f"{course['name']}.png")
            print(f'wrote {out}')
    elif args.cmd == 'marks':
        for f in args.layout:
            course = C.load(f)
            print(f"course {course['name']}: tube centres from the start "
                  f"mark (x forward, y left)")
            print('  id     x_cm   y_cm')
            for tid, x, y in marks(course):
                print(f'  {tid}  {x:6d} {y:6d}')
    elif args.cmd == 'sheet':
        courses = [C.load(f) for f in args.layout]
        for course in courses:
            render(course, RENDER_DIR / f"{course['name']}.png")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(sheet(courses))
        print(f'wrote {args.out} (+ figures)')
    elif args.cmd == 'check-map':
        course = C.load(args.layout)
        tubes, stray, pts = check_map(course, args.map_yaml)
        bad = [t for t in tubes if not t[4]]
        for tid, x, y, d, ok in tubes:
            flag = 'ok' if ok else 'CHECK: missing or displaced'
            print(f'  {tid} ({x:5.2f}, {y:5.2f})  nearest occupied '
                  f'{d * 100:5.1f} cm  {flag}')
        print(f'{len(tubes) - len(bad)}/{len(tubes)} tubes where the layout '
              f'says; {len(stray)} stray occupied cells inside the arena')
        if len(stray):
            cl = np.unique(np.round(stray / 0.15) * 0.15, axis=0)
            print('  stray clusters near: ' + ', '.join(
                f'({x:.2f}, {y:.2f})' for x, y in cl[:10]))
        out = render(course, RENDER_DIR / f"{course['name']}_map_check.png",
                     map_points=pts)
        print(f'overlay: {out}')
        sys.exit(0 if not bad else 1)
    elif args.cmd == 'arena-from-map':
        out, poles = arena_from_map(args.map_yaml, args.name, args.id,
                                    args.pole_diameter)
        arena = A.load(out)
        png = render_arena(arena, 'P1', ARENA_RENDER_DIR / f'{args.name}.png')
        print(f'{len(poles)} pole candidates -> {out}\nfigure: {png}\n'
              'next: delete non-course blobs, tape-check a few poles, then '
              'add the squares under placements')
    elif args.cmd == 'arena-measure':
        arena = A.load(args.arena)
        m = measure_arena(arena)
        print(f"arena {arena['name']} (id {arena['id']}): "
              f"{m['free_area_m2']:.1f} m2 free, open space "
              f"{m['open_frac'] * 100:.0f} %, {m['poles']} poles "
              f"({m['pole_density']:.2f} per m2)")
        tight = [g for g in m['gaps'] if g[0] < 0.4]
        print("  narrowest gaps: " + ', '.join(
            f'{w * 100:.0f} cm ({where})' for w, where in m['gaps'][:5]))
        print(f"  gaps under 40 cm: {len(tight)} (robot is "
              f"{ROBOT_WIDTH_M * 100:.1f} cm wide)")
        for name, rows in m['squares'].items():
            print(f'  placement {name}:' + ('' if rows else ' no squares yet'))
            for color, mount, pocket in rows:
                flag = '' if pocket >= SQUARE_POCKET_M else \
                    f'  <- under {SQUARE_POCKET_M} m of free floor in front'
                print(f'    {color:12s} {mount:5s} pocket {pocket:.2f} m{flag}')
            png = render_arena(arena, name, ARENA_RENDER_DIR /
                               f"{arena['name']}_{name}.png")
            print(f'    figure: {png}')
        print('  pole tape marks (cm from the start mark, x fwd, y left):')
        for pid, (x, y) in zip(A.pole_ids(arena), arena['poles']):
            print(f'    {pid} {x * 100:6.0f} {y * 100:6.0f}')
    elif args.cmd == 'design':
        walls = {'back_x': args.back_x, 'side_y': args.side_y,
                 'end_x': args.end_x}
        best = design(args.tier, args.tubes, args.name, args.id, args.seed,
                      walls, tuple(args.goal), args.min_gap)
        if best is None:
            sys.exit(f'no traversable {args.tier} layout with {args.tubes} '
                     f'tubes in {DESIGN_TRIES} tries — change --tubes')
        layout, r = best
        _print_measure(r)
        out = args.out or COURSES_DIR / f'{args.name}.yaml'
        _write_layout(out, layout)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
