#!/usr/bin/env python3
"""World figures for the dissertation, drawn from the committed grids.

  .venv/bin/python report/make_world_figures.py -> report/figures/*.png

  barn_worlds.png    standard (easiest / hardest holdout), dense, winding,
                     scatter and multipath worlds, with the A* route
  dead_end.png       straight line to the goal vs the A* route
  colour_worlds.png  colour-task worlds 020 and 768 with their squares

Each world is drawn lying on its side: the robot starts on the left and
the goal is to the right (BARN's +y runs left to right).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import Circle, Rectangle               # noqa: E402
import numpy as np                                             # noqa: E402
import yaml                                                    # noqa: E402

CAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAP))
from common.grid import plan_route, arena_grid_to_cylinders    # noqa: E402

OUT = CAP / 'report' / 'figures'
BARN = CAP / 'barn'
META = yaml.safe_load(open(BARN / 'metadata.yaml'))['constants']
CELL, R = META['cell'], META['cylinder_radius']
ORIGIN = tuple(META['arena_origin'])
START = tuple(META['start_pose'][:2])
GOAL = tuple(META['goal'])

SURFACE, INK, INK_2 = '#fcfcfb', '#0b0b0b', '#52514e'
OBST = '#6f6d68'
ROUTE = '#2a78d6'            # two-stage blue: the planned route
DIRECT = '#eb6834'           # orange: the straight line to the goal
START_C = '#f2d53c'
SQUARE = {'light_green': '#90ee90', 'purple': '#800080', 'orange': '#ff7f00'}
SQUARE_LABEL = {'light_green': 'Light green', 'purple': 'Purple',
                'orange': 'Orange'}


def cylinders(idx):
    return arena_grid_to_cylinders(np.load(BARN / 'grids' / f'grid_{idx:03d}.npy'),
                                   ORIGIN)


def draw_world(ax, cyl, start=START, goal=GOAL, route=None, direct=False,
               title='', ylim=None):
    """World on its side: plot (y, -x) so start is left, goal right."""
    ax.set_facecolor(SURFACE)
    for x, y in cyl:
        ax.add_patch(Circle((y, -x), R, color=OBST, lw=0))
    x0, x1 = ORIGIN[0] - R, ORIGIN[0] + 29 * CELL + R
    y0, y1 = ORIGIN[1] - R, ORIGIN[1] + 63 * CELL + R
    ax.add_patch(Rectangle((y0, -x1), y1 - y0, x1 - x0, fill=False,
                           ec=INK_2, lw=0.6, ls=':'))
    if direct:
        ax.plot([start[1], goal[1]], [-start[0], -goal[0]], color=DIRECT,
                lw=1.6, ls='--', zorder=3)
    if route is not None:
        ax.plot(route[:, 1], -route[:, 0], color=ROUTE, lw=1.8, zorder=4)
    ax.plot(start[1], -start[0], 's', ms=7, color=START_C, mec=INK,
            mew=0.7, zorder=5)
    if goal is not None:
        ax.add_patch(Circle((goal[1], -goal[0]), META['success_radius'],
                            fill=False, ec=INK, lw=0.9, ls='--', zorder=5))
        ax.plot(goal[1], -goal[0], '*', ms=10, color=INK, zorder=5)
    ax.set_aspect('equal')
    ax.set_xlim(*(ylim or (y0 - 0.3, max(y1, (goal or START)[1] + 1.2) + 0.2)))
    ax.set_ylim(-x1 - 0.3, -x0 + 0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc='left', fontsize=8.5, color=INK)


def route_for(cyl, start=START, goal=GOAL):
    _, path, _ = plan_route(cyl, start, goal)
    return None if path is None else np.asarray(path)


def legend(fig, handles):
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles),
               fontsize=8, frameon=False, labelcolor=INK_2)


def handles(direct=False):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls='', marker='o', ms=5, color=OBST, label='Obstacle (0.15 m cylinder)'),
         Line2D([], [], ls='', marker='s', ms=7, color=START_C, mec=INK, label='Start'),
         Line2D([], [], ls='', marker='*', ms=9, color=INK, label='Goal (1 m success circle)'),
         Line2D([], [], color=ROUTE, lw=1.8, label='A* route')]
    if direct:
        h.append(Line2D([], [], color=DIRECT, lw=1.6, ls='--',
                        label='Straight line to the goal'))
    return h


def fig_barn_worlds():
    fork = {w['idx']: w for w in yaml.safe_load(
        open(BARN / 'fork_worlds.yaml'))['worlds']}
    panels = [
        (228, 'a) BARN 228: easiest evaluation world (12/12 two-stage successes)'),
        (265, 'b) BARN 265: hardest evaluation world (3/12 two-stage successes)'),
        (302, 'c) Dense stress world 302 (training only)'),
        (306, 'd) Winding (serpentine) world 306 (training only)'),
        (310, 'e) Scatter stress world 310 (training only)'),
        (400, 'f) Multipath fork world 400 (training only, interior goal)'),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 6.4), facecolor=SURFACE,
                             layout='constrained')
    for ax, (idx, title) in zip(axes.flat, panels):
        cyl = cylinders(idx)
        goal = tuple(fork[idx]['goal']) if idx in fork else GOAL
        draw_world(ax, cyl, goal=goal,
                   route=route_for(cyl, goal=goal), title=title,
                   ylim=(-0.3, 14.3))
    legend(fig, handles())
    fig.savefig(OUT / 'barn_worlds.png', dpi=220, facecolor=SURFACE,
                bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def first_block_on_line(cyl, start, goal, clearance=0.21):
    """Distance along the straight start->goal line to the first point
    closer than `clearance` to a cylinder centre minus its radius."""
    d = np.linspace(0, 1, 2000)
    pts = np.outer(1 - d, start) + np.outer(d, goal)
    dist = np.min(np.linalg.norm(pts[:, None, :] - cyl[None], axis=2), axis=1) - R
    hit = np.nonzero(dist < clearance)[0]
    return None if not len(hit) else pts[hit[0]]


def fig_dead_end(idx=265):
    cyl = cylinders(idx)
    route = route_for(cyl)
    fig, ax = plt.subplots(figsize=(9.2, 2.8), facecolor=SURFACE,
                           layout='constrained')
    draw_world(ax, cyl, route=route, direct=True,
               title=f'BARN evaluation world {idx}: the straight line towards '
                     'the goal is blocked; the A* route detours through an open gap',
               ylim=(-0.3, 14.3))
    blk = first_block_on_line(cyl, START, GOAL)
    if blk is not None:
        ax.plot(blk[1], -blk[0], 'X', ms=11, color=DIRECT, mec=INK, mew=0.8,
                zorder=6)
        ax.annotate('first blocked point on the straight line',
                    (blk[1], -blk[0]), xytext=(-175, 48),
                    textcoords='offset points', fontsize=7.5, color=INK_2,
                    arrowprops=dict(arrowstyle='-', color=INK_2, lw=0.6))
    from matplotlib.lines import Line2D
    h = handles(direct=True) + [Line2D([], [], ls='', marker='X', ms=8,
                                       color=DIRECT, mec=INK,
                                       label='First blocked point')]
    legend(fig, h)
    fig.savefig(OUT / 'dead_end.png', dpi=220, facecolor=SURFACE,
                bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    return blk


def fig_colour_worlds():
    rows = [(20, 'Colour-task world 020: easy (1.4% occupied, narrowest gap on the best route 1.50 m)'),
            (768, 'Colour-task world 768: hard (9.2% occupied, narrowest gap on the best route 0.42 m)')]
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.6), facecolor=SURFACE,
                             layout='constrained')
    for ax, (idx, title) in zip(axes, rows):
        cyl = cylinders(idx)
        draw_world(ax, cyl, goal=None, title=title, ylim=(-0.3, 10.0))
        tg = yaml.safe_load(open(BARN / 'targets' / f'targets_{idx:03d}.yaml'))
        for t in tg['targets']:
            ax.plot(t['y'], -t['x'], 's', ms=9, color=SQUARE[t['color']],
                    mec=INK, mew=0.8, zorder=6)
            nx, ny = np.cos(t['normal_yaw']), np.sin(t['normal_yaw'])
            ax.annotate('', xy=(t['y'] + 0.45 * ny, -(t['x'] + 0.45 * nx)),
                        xytext=(t['y'], -t['x']), zorder=6,
                        arrowprops=dict(arrowstyle='-|>', color=INK_2, lw=0.8))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls='', marker='o', ms=5, color=OBST, label='Obstacle'),
         Line2D([], [], ls='', marker='s', ms=7, color=START_C, mec=INK, label='Start')]
    h += [Line2D([], [], ls='', marker='s', ms=8, color=SQUARE[c], mec=INK,
                 label=f'{SQUARE_LABEL[c]} square (arrow: the way it faces)')
          for c in ('light_green', 'purple', 'orange')]
    legend(fig, h)
    fig.savefig(OUT / 'colour_worlds.png', dpi=220, facecolor=SURFACE,
                bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_barn_worlds()
    print('dead-end blocked point:', fig_dead_end())
    fig_colour_worlds()
    print('figures ->', OUT)
