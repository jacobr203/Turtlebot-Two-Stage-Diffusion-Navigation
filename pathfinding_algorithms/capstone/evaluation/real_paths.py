#!/usr/bin/env python3
"""Paths of real-robot colour runs in the arena frame, and their figures.

Each run's live SLAM starts wherever the robot was placed, so its trace
(slam_trace.csv) is in that run's own frame. At the start of every run the
runner localizes the robot on the arena map by matching its lidar scan
(localize() here) and stores the start pose; composing it with the trace
puts every run into the arena frame, where the notes and poles are. Each
run then leaves:

  path_arena.csv      t, x, y, yaw  (t = seconds since the controller's
                      first command; negative before it)
  path_events.json    start pose and match quality, where each note was
                      first seen and found, where collisions were logged
  path.png / path.pdf the run's path on the arena map

and this command draws every run of a placement side by side by model:

  .venv/bin/python evaluation/real_paths.py --run-id colour_v1 \\
      --arena evaluation/arenas/room1.yaml --placement P1 --lighting day
      -> results/real/colour_v1/figures/paths_room1_P1_day.png/.pdf
         (and re-renders each run's own figure)

Localization: a likelihood field on the map's free-space edges plus the
poles; a candidate pose scores the fraction of scan points within 6 cm of
an edge. On room1 the true pose scores 98-99 % and the best pose more than
30 cm away about 60 %.
"""

import argparse
import csv
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arena as A                                 # noqa: E402

CAPSTONE = Path(__file__).resolve().parents[1]
LIDAR_X = -0.032 # base_scan sits behind base_footprint on the Burger
MAX_RANGE_M = 4.0
INLIER_M = 0.06
COARSE_INLIER_M = 0.08
PRIOR_RADIUS_M = 0.5
YAW_STEP_DEG = 4.0
GOOD_MATCH = 0.80 # below this near the prior, search the whole map

# Figure style (dataviz reference palette, light mode, validated:
# blue/orange CVD dE 24.7, normal-vision dE 33.6, both >= 3:1)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
UNKNOWN = '#eceae6'
WALL = '#9c9a94'   # context: recedes behind the paths
POLE = '#6f6d68'   # darker than walls: what the robot threads between
MODEL_COLOR = {'two_stage': '#2a78d6', 'single_stage': '#eb6834'}
MODEL_LABEL = {'two_stage': 'Two-stage (seek)',
               'single_stage': 'Single-stage (wander)'}
# the physical notes' colours as the robot's camera sees them
NOTE_COLOR = {'sky_blue': '#2db4c2', 'lime_green': '#c0d370',
              'light_green': '#90ee90', 'purple': '#800080',
              'orange': '#ff7f00'}
NOTE_LABEL = {'sky_blue': 'Sky-blue note', 'lime_green': 'Lime note',
              'light_green': 'Light-green square', 'purple': 'Purple square',
              'orange': 'Orange square'}
START_COLOR = '#f2d53c' # the yellow sticky note the robot starts on


# ── localization ────────────────────────────────────────────────────────────
def scan_points(scans):
    """Base-frame xy of every valid return in a list of LaserScans."""
    pts = []
    for s in scans:
        r = np.asarray(s.ranges, dtype=np.float64)
        a = s.angle_min + np.arange(len(r)) * s.angle_increment
        ok = np.isfinite(r) & (r > s.range_min) & (r < MAX_RANGE_M)
        pts.append(np.stack([LIDAR_X + r[ok] * np.cos(a[ok]),
                             r[ok] * np.sin(a[ok])], axis=1))
    return np.vstack(pts) if pts else np.zeros((0, 2))


@functools.lru_cache(maxsize=4)
def _field(map_path, poles_key):
    """Distance to the nearest free-space edge or pole, per map cell."""
    from scipy.ndimage import binary_dilation, distance_transform_edt
    occupied, free, res, origin, h = A.read_map(map_path)
    edge = ~free & binary_dilation(free, np.ones((3, 3)))
    for x, y in np.asarray(poles_key).reshape(-1, 2):
        r, c = h - 1 - int((y - origin[1]) / res), int((x - origin[0]) / res)
        if 0 <= r < h and 0 <= c < edge.shape[1]:
            edge[r, c] = True
    return distance_transform_edt(~edge) * res, free, res, origin, h


def _scorer(arena, pts):
    field, free, res, origin, h = _field(
        str(arena['map']), tuple(map(tuple, arena['poles'].round(3))))

    def score(x, y, yaw, tol=INLIER_M):
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + c * pts[:, 0] - s * pts[:, 1]
        wy = y + s * pts[:, 0] + c * pts[:, 1]
        cc = ((wx - origin[0]) / res).astype(int)
        rr = h - 1 - ((wy - origin[1]) / res).astype(int)
        ins = (cc >= 0) & (cc < field.shape[1]) & (rr >= 0) & (rr < h)
        d = np.full(len(wx), np.inf)
        d[ins] = field[rr[ins], cc[ins]]
        return float((d < tol).mean()) if len(d) else 0.0
    return score, free, res, origin, h


def localize(arena, pts, prior=(0.0, 0.0), radius=PRIOR_RADIUS_M):
    """Pose (x, y, yaw) of base_footprint in the arena frame from a scan.

    Searches every free cell within `radius` of the prior at all headings,
    falling back to the whole map when nothing near the prior matches
    well, then refines. Returns a dict with the pose, the fraction of scan
    points on the map ('match') and the best score of any pose more than
    30 cm or 30 deg away ('rival'): a large gap means the answer is
    unambiguous."""
    score, free, res, origin, h = _scorer(arena, pts)
    rows, cols = np.nonzero(free)
    cells = A.cells_to_xy(rows, cols, res, origin, h)
    yaws = np.radians(np.arange(-180.0, 180.0, YAW_STEP_DEG))

    def search(candidates):
        return sorted(((score(x, y, yaw, COARSE_INLIER_M), x, y, yaw)
                       for x, y in candidates for yaw in yaws), reverse=True)

    near = cells[np.hypot(*(cells - prior).T) <= radius] \
        if prior is not None else cells[:0]
    cands = search(near) if len(near) else []
    if not cands or cands[0][0] < GOOD_MATCH:
        cands = search(cells[::4])
    best = cands[0]
    for step, ang in ((0.02, 2.0), (0.01, 0.5)):
        _, bx, by, byaw = best
        for dx in np.arange(-4, 5) * step:
            for dy in np.arange(-4, 5) * step:
                for da in np.radians(np.arange(-4, 5) * ang):
                    sc = score(bx + dx, by + dy, byaw + da)
                    if sc > best[0]:
                        best = (sc, bx + dx, by + dy, byaw + da)
    match, x, y, yaw = best
    rival = max((c[0] for c in cands
                 if math.hypot(c[1] - x, c[2] - y) > 0.3
                 or abs(math.remainder(c[3] - yaw, 2 * math.pi)) > 0.5),
                default=0.0)
    return {'x': x, 'y': y, 'yaw': math.remainder(yaw, 2 * math.pi),
            'match': match, 'rival': rival}


# ── paths and events ────────────────────────────────────────────────────────
def compose(start, trace):
    """Trace poses (relative to the run's start) -> arena frame.

    start: (x, y, yaw); trace: (N, 3) of x, y, yaw."""
    trace = np.asarray(trace, dtype=np.float64).reshape(-1, 3)
    c, s = math.cos(start[2]), math.sin(start[2])
    return np.stack([start[0] + c * trace[:, 0] - s * trace[:, 1],
                     start[1] + s * trace[:, 0] + c * trace[:, 1],
                     np.arctan2(np.sin(trace[:, 2] + start[2]),
                                np.cos(trace[:, 2] + start[2]))], axis=1)


def position_at(t, times, xy):
    """Interpolated position on a timed path (clamped at its ends)."""
    return (float(np.interp(t, times, xy[:, 0])),
            float(np.interp(t, times, xy[:, 1])))


def save_run(trial_dir, start, mon, meta):
    """Write path_arena.csv and path_events.json for one colour run.

    mon: the run's ColourMonitor (monotonic times); meta: model, verdict
    and the like, copied into the events file for the figures."""
    trial_dir = Path(trial_dir)
    if not mon.slam_trace or start is None:
        return None
    t0 = mon.t0 if mon.t0 is not None else mon.slam_trace[0][0]
    tr = np.asarray(mon.slam_trace, dtype=np.float64)
    times = tr[:, 0] - t0
    arena_xy = compose((start['x'], start['y'], start['yaw']), tr[:, 1:4])
    with open(trial_dir / 'path_arena.csv', 'w') as f:
        f.write('t,x,y,yaw\n')
        for t, (x, y, yaw) in zip(times, arena_xy):
            f.write(f'{t:.3f},{x:.4f},{y:.4f},{yaw:.4f}\n')

    def at(t):
        x, y = position_at(t - t0, times, arena_xy)
        return {'t': round(t - t0, 2), 'x': round(x, 3), 'y': round(y, 3)}
    events = {
        **meta,
        'start': {k: round(v, 4) for k, v in start.items()},
        'seen': {c: at(t) for c, t in mon.seen_t.items()},
        'found': {c: at(t) for c, t in mon.found_t.items()},
        'collisions': [at(t) for t in mon.collision_t],
    }
    (trial_dir / 'path_events.json').write_text(json.dumps(events, indent=1))
    return trial_dir / 'path_arena.csv'


def load_run(trial_dir):
    """(times, (N, 3) arena poses, events dict) for a saved run, or None."""
    trial_dir = Path(trial_dir)
    path, ev = trial_dir / 'path_arena.csv', trial_dir / 'path_events.json'
    if not (path.exists() and ev.exists()):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    arr = np.array([[float(r[k]) for k in ('t', 'x', 'y', 'yaw')]
                    for r in rows])
    return arr[:, 0], arr[:, 1:], json.loads(ev.read_text())


# ── figures ─────────────────────────────────────────────────────────────────
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_aspect('equal')   # no grid: on a map the ticks give the scale
    # equal aspect leaves a panel shorter than its slot; pin it to the top
    # so the slack falls at the bottom, where the tight save trims it
    ax.set_anchor('N')
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.set_xlabel('Distance along x from the start mark (m)', color=INK_2, fontsize=8)
    ax.set_ylabel('Distance along y (m)', color=INK_2, fontsize=8)


def _draw_arena(ax, arena, placement, label_notes=True, notes=None):
    """notes: squares to draw instead of the placement's recorded ones
    (e.g. where the detections put them); facing_deg is optional."""
    from matplotlib.colors import ListedColormap
    occupied, free, res, origin, h = A.read_map(arena['map'])
    img = np.zeros(occupied.shape, dtype=int)   # 0 unknown
    img[free], img[occupied] = 1, 2
    extent = (origin[0], origin[0] + occupied.shape[1] * res,
              origin[1], origin[1] + h * res)
    ax.imshow(img, cmap=ListedColormap([UNKNOWN, SURFACE, WALL]), vmin=0,
              vmax=2, extent=extent, interpolation='nearest', zorder=0)
    r = max(arena['pole_diameter_m'] / 2, 0.025)
    from matplotlib.patches import Circle
    for x, y in arena['poles']:
        ax.add_patch(Circle((x, y), r, color=POLE, zorder=1, lw=0))
    ax.plot(0, 0, 's', ms=9, color=START_COLOR, mec=INK_2, mew=0.8,
            zorder=5)
    if label_notes:
        ax.annotate('Start', (0, 0), xytext=(6, -10),
                    textcoords='offset points', fontsize=7, color=INK_2)
    for s in (arena['placements'].get(placement, []) if notes is None
              else notes):
        ax.plot(s['x'], s['y'], 's', ms=9, color=NOTE_COLOR[s['color']],
                mec=INK, mew=0.8, zorder=6)
        a = math.radians(s.get('facing_deg', 90.0))
        if 'facing_deg' in s:
            ax.annotate('', xy=(s['x'] + 0.25 * math.cos(a),
                                s['y'] + 0.25 * math.sin(a)),
                        xytext=(s['x'], s['y']), zorder=6,
                        arrowprops=dict(arrowstyle='-|>', color=INK_2,
                                        lw=0.8, shrinkA=4, shrinkB=0))
        if label_notes: # opposite the facing arrow, so neither hides the other
            ox, oy = -math.cos(a), -math.sin(a)
            ax.annotate(NOTE_LABEL[s['color']], (s['x'], s['y']),
                        xytext=(12 * ox, 12 * oy), textcoords='offset points',
                        ha='left' if ox > 0.3 else 'right' if ox < -0.3
                        else 'center',
                        va='bottom' if oy > 0.3 else 'top' if oy < -0.3
                        else 'center', fontsize=7, color=INK_2, zorder=7)
    (x0, x1), (y0, y1) = arena_bounds(arena)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def arena_bounds(arena, pad=0.25):
    """((x0, x1), (y0, y1)) framing the arena rather than the whole map
    image. An opening first drops thin strips of free cells the lidar saw
    through gaps under furniture."""
    from scipy.ndimage import binary_opening
    occupied, free, res, origin, h = A.read_map(arena['map'])
    body = binary_opening(free, np.ones((5, 5)))
    fr, fc = np.nonzero(body if body.any() else free)
    xy = A.cells_to_xy(fr, fc, res, origin, h)
    return ((xy[:, 0].min() - pad, xy[:, 0].max() + pad),
            (xy[:, 1].min() - pad, xy[:, 1].max() + pad))


def _panel_size(arena, width=5.2):
    """(width, height) in inches for one map panel at the arena's aspect,
    plus room for a title block."""
    (x0, x1), (y0, y1) = arena_bounds(arena)
    return width, width * (y1 - y0) / (x1 - x0) + 0.9


def _draw_path(ax, xy, events, color, alpha=1.0, end_dot=True):
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.6, alpha=alpha,
            solid_joinstyle='round', solid_capstyle='round', zorder=3)
    if end_dot:
        ax.plot(xy[-1, 0], xy[-1, 1], 'o', ms=6, color=color,
                mec=SURFACE, mew=1.5, alpha=alpha, zorder=4)
    for c, e in events.get('seen', {}).items():
        ax.plot(e['x'], e['y'], 'o', ms=7, mfc='none',
                mec=NOTE_COLOR.get(c, INK_2), mew=1.6, zorder=5)
    for c, e in events.get('found', {}).items():
        ax.plot(e['x'], e['y'], 'o', ms=7, color=NOTE_COLOR.get(c, INK_2),
                mec=INK, mew=0.8, zorder=5)
    for e in events.get('collisions', []):
        ax.plot(e['x'], e['y'], 'x', ms=7, color=INK, mew=1.6, zorder=5)


def _count(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _legend_handles(models):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MODEL_COLOR[m], lw=1.6,
                      label=MODEL_LABEL[m]) for m in models]
    handles += [
        Line2D([], [], ls='', marker='o', ms=7, mfc='none', mec=INK_2,
               mew=1.6, label='Note first seen (ring in its colour)'),
        Line2D([], [], ls='', marker='o', ms=7, color=INK_2, mec=INK,
               mew=0.8, label='Note found (fill in its colour)'),
        Line2D([], [], ls='', marker='x', ms=7, color=INK, mew=1.6,
               label='Collision'),
        Line2D([], [], ls='', marker='s', ms=8, color=START_COLOR,
               mec=INK_2, mew=0.8, label='Start mark'),
    ]
    return handles


def _save(fig, out_base):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    # bbox 'tight' keeps outside legends and axis labels in the file
    kw = dict(facecolor=SURFACE, bbox_inches='tight', pad_inches=0.12)
    fig.savefig(out_base.with_suffix('.png'), dpi=200, **kw)
    fig.savefig(out_base.with_suffix('.pdf'), **kw)


def render_run(arena, placement, trial_dir):
    """path.png/.pdf for one saved run; returns the PNG path or None."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    loaded = load_run(trial_dir)
    if loaded is None:
        return None
    times, xy, ev = loaded
    model = ev.get('model', 'two_stage')
    w, hgt = _panel_size(arena)
    fig, ax = plt.subplots(figsize=(w + 2.2, hgt), facecolor=SURFACE,
                           layout='constrained')
    _style(ax)
    _draw_arena(ax, arena, placement)
    run = times >= 0
    _draw_path(ax, xy[run] if run.any() else xy, ev,
               MODEL_COLOR.get(model, INK_2))
    found = len(ev.get('found', {}))
    n = len(arena['placements'].get(placement, []))
    light = f", {ev['lighting']} lighting" if ev.get('lighting') else ''
    ax.set_title(f"{MODEL_LABEL.get(model, model)}, placement {placement}"
                 f"{light}, round {ev.get('round', '?')}", loc='left',
                 fontsize=10,
                 color=INK, pad=16)
    ax.text(0.0, 1.015,
            f"{ev.get('verdict', '?').capitalize()}: {found}/{n} notes found in "
            f"{ev.get('total_s', '?')} s, path {ev.get('path_m', '?')} m, "
            f"{_count(len(ev.get('collisions', [])), 'collision')}; start "
            f"located on the map at {ev['start']['match'] * 100:.0f}% scan "
            "match", transform=ax.transAxes, fontsize=7, color=INK_2,
            va='bottom')
    ax.legend(handles=_legend_handles([model]), loc='upper left',
              bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False,
              labelcolor=INK_2)
    out = Path(trial_dir) / 'path'
    _save(fig, out)
    plt.close(fig)
    return out.with_suffix('.png')


def render_placement(arena, placement, runs, out_base, lighting=None):
    """Two panels, one per model, every counted run of the placement.

    runs: list of (model, times, xy, events)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = [m for m in ('two_stage', 'single_stage')
              if any(r[0] == m for r in runs)]
    if not models:
        return None
    w, hgt = _panel_size(arena)
    fig, axes = plt.subplots(1, len(models),
                             figsize=(w * len(models) + 2.2, hgt + 0.3),
                             sharex=True, sharey=True, facecolor=SURFACE,
                             squeeze=False, layout='constrained')
    for ax, m in zip(axes[0], models):
        _style(ax)
        _draw_arena(ax, arena, placement, label_notes=(ax is axes[0][0]))
        mine = [r for r in runs if r[0] == m]
        for _, times, xy, ev in mine:
            run = times >= 0
            _draw_path(ax, xy[run] if run.any() else xy, ev, MODEL_COLOR[m],
                       alpha=0.55)
        found = sum(1 for *_, ev in mine
                    if len(ev.get('found', {})) == len(
                        arena['placements'].get(placement, [])))
        ax.set_title(f'{MODEL_LABEL[m]}: {found}/{len(mine)} runs found '
                     'every note', loc='left', fontsize=9, color=INK)
    axes[0][-1].legend(handles=_legend_handles(models), loc='upper left',
                       bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False,
                       labelcolor=INK_2)
    light = f', {lighting} lighting' if lighting else ''
    fig.suptitle(f"Paths driven, arena {arena['name']}, placement "
                 f"{placement}{light}", x=0.01, ha='left', fontsize=11,
                 color=INK)
    _save(fig, out_base)
    plt.close(fig)
    return Path(out_base).with_suffix('.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--arena', required=True)
    ap.add_argument('--placement', required=True)
    ap.add_argument('--lighting', required=True,
                    help='day or night: conditions are never mixed in a figure')
    ap.add_argument('--shakedown', action='store_true')
    args = ap.parse_args()
    arena = A.load(args.arena)
    run_dir = (CAPSTONE / 'results' / 'real'
               / (f'{args.run_id}_shakedown' if args.shakedown
                  else args.run_id))
    rows = list(csv.DictReader(open(run_dir / 'colour_trials.csv')))
    counted = ('success', 'timeout', 'stuck', 'crash')
    runs = []
    for d in sorted((run_dir / 'trials').glob(
            f"t*_{arena['name']}_{args.placement}_*")):
        loaded = load_run(d)
        if loaded is None:
            continue
        times, xy, ev = loaded
        if ev.get('verdict') not in counted and not args.shakedown:
            continue
        if ev.get('lighting') != args.lighting:
            continue
        render_run(arena, args.placement, d)
        runs.append((ev.get('model'), times, xy, ev))
    out = render_placement(
        arena, args.placement, runs,
        run_dir / 'figures'
        / f"paths_{arena['name']}_{args.placement}_{args.lighting}",
        lighting=args.lighting)
    print(f'{len(runs)} runs ({len(rows)} rows in the CSV) -> {out} '
          '(+ .pdf, and path.png/.pdf in each run folder)')


if __name__ == '__main__':
    main()
