"""Physical test courses for the real-robot evaluation (see evaluation/courses/A.yaml).

A course is a BARN-style walled arena of cardboard tubes, listed in the robot's
start frame (origin at the start pose, +x forward, +y left).
"""

import math
from pathlib import Path

import numpy as np
import yaml

from common import grid as G

COURSE_ID_MIN = 9000
LATTICE_TOL_M = 0.01 # tape-measured placement is snapped within this
# Burger half-diagonal (0.113 m) + tube radius: closer already touches the robot.
START_CLEAR_M = 0.19
# Planning window margin around the arena; the goal may sit past end_x.
BOUNDS_MARGIN_M = 0.3
GOAL_MARGIN_M = 1.0
# Same ratio as the sim matrix: 240 s on BARN's ~10.3 m optimal routes.
TIMEOUT_S_PER_M = 24.0
TIMEOUT_ROUND_S = 10.0


class CourseError(ValueError):
    """A layout file that cannot describe a buildable course."""


def _on_lattice(v, pitch):
    return abs(v / pitch - round(v / pitch)) * pitch <= LATTICE_TOL_M


def load(path):
    """Read and validate a layout file; returns a dict with numpy tubes."""
    path = Path(path).resolve()
    return from_dict(yaml.safe_load(open(path)), path)


def from_dict(raw, path=None):
    """Validate an already-parsed layout."""
    if not isinstance(raw, dict) or 'tubes' not in raw:
        raise CourseError(f'{path}: not a course layout (no "tubes" key)')
    pitch = float(raw.get('lattice_m', G.CELL))
    walls = raw['walls']
    back_x, side_y, end_x = (float(walls['back_x']), float(walls['side_y']),
                             float(walls['end_x']))
    course = {
        'path': path,
        'name': str(raw['name']),
        'id': int(raw['id']),
        'tier_target': raw.get('tier_target'),
        'lattice_m': pitch,
        'back_x': back_x, 'side_y': side_y, 'end_x': end_x,
        'clutter_x': tuple(float(v) for v in raw['clutter_x']),
        'start': tuple(float(v) for v in raw.get('start', (0.0, 0.0, 0.0))),
        'goal': tuple(float(v) for v in raw['goal']),
        'tubes': np.asarray(raw['tubes'] or [], dtype=np.float64).reshape(-1, 2),
    }
    _validate(course)
    return course


def _validate(c):
    pitch = c['lattice_m']
    if c['id'] < COURSE_ID_MIN:
        raise CourseError(f"id {c['id']} < {COURSE_ID_MIN} (BARN index range)")
    if c['start'] != (0.0, 0.0, 0.0):
        raise CourseError('start must be (0, 0, 0): the start frame IS the '
                          'odometry frame after bringup')
    for name in ('back_x', 'side_y', 'end_x'):
        if not _on_lattice(c[name], pitch):
            raise CourseError(f'wall {name}={c[name]} is off the '
                              f'{pitch} m lattice')
    if not (c['back_x'] < 0.0 < c['end_x'] and c['side_y'] > 0.0):
        raise CourseError('start must lie inside the walls')
    cx0, cx1 = c['clutter_x']
    if not (_on_lattice(cx0, pitch) and _on_lattice(cx1, pitch)
            and c['back_x'] < cx0 < cx1 < c['end_x']):
        raise CourseError(f'clutter_x {c["clutter_x"]} must be lattice '
                          'values between the back wall and the far end')
    gx, gy = c['goal']
    if not (0.0 < gx and abs(gy) < c['side_y']):
        raise CourseError(f'goal {c["goal"]} is not ahead of the start and '
                          'between the side walls')
    for k, (x, y) in enumerate(c['tubes']):
        if not (_on_lattice(x, pitch) and _on_lattice(y, pitch)):
            raise CourseError(f'tube {k} ({x}, {y}) is off the {pitch} m '
                              f'lattice by more than {LATTICE_TOL_M} m')
        if not (cx0 - LATTICE_TOL_M <= x <= cx1 + LATTICE_TOL_M
                and abs(y) < c['side_y']):
            raise CourseError(f'tube {k} ({x}, {y}) is outside the clutter '
                              'field')
        if math.hypot(x, y) < START_CLEAR_M:
            raise CourseError(f'tube {k} ({x}, {y}) overlaps the start pose')
    snapped = np.round(c['tubes'] / pitch).astype(int)
    if len({tuple(p) for p in snapped}) != len(snapped):
        raise CourseError('two tubes share one lattice point')
    c['tubes'] = snapped * pitch


def tube_ids(course):
    """Stable labels t00, t01, ... in layout order (tape marks, collisions)."""
    return [f't{k:02d}' for k in range(len(course['tubes']))]


def wall_points(course):
    """Lattice points along the back and side walls (the open end has none)."""
    p = course['lattice_m']
    xs = np.arange(round(course['back_x'] / p),
                   round(course['end_x'] / p) + 1) * p
    ys = np.arange(-round(course['side_y'] / p),
                   round(course['side_y'] / p) + 1) * p
    back = np.stack([np.full_like(ys, course['back_x']), ys], axis=1)
    left = np.stack([xs, np.full_like(xs, course['side_y'])], axis=1)
    right = np.stack([xs, np.full_like(xs, -course['side_y'])], axis=1)
    return np.unique(np.vstack([back, left, right]).round(6), axis=0)


def obstacle_points(course):
    """Cylinder centres the A* planner inflates around: tubes + walls."""
    return np.vstack([course['tubes'], wall_points(course)])


def planning_bounds(course):
    """((x_min, x_max), (y_min, y_max)) window for common.grid.Planner."""
    x_hi = max(course['end_x'], course['goal'][0] + GOAL_MARGIN_M)
    return ((course['back_x'] - BOUNDS_MARGIN_M, x_hi + BOUNDS_MARGIN_M),
            (-course['side_y'] - BOUNDS_MARGIN_M,
             course['side_y'] + BOUNDS_MARGIN_M))


def arena_grid(course):
    """Clutter-field occupancy grid in BARN-grid orientation.

    Shape (n_lateral, n_forward); axis 1 runs start -> goal. Difficulty is
    measured over the clutter field only, because open start and goal zones
    would dilute every metric. The border ring is set because world_metrics
    strips one ring before measuring."""
    p = course['lattice_m']
    j0 = round(course['clutter_x'][0] / p) - 1
    n_fwd = round(course['clutter_x'][1] / p) + 1 - j0 + 1
    i0 = -round(course['side_y'] / p)
    n_lat = 2 * round(course['side_y'] / p) + 1
    occ = np.zeros((n_lat, n_fwd), dtype=np.uint8)
    occ[0, :] = occ[-1, :] = 1
    occ[:, 0] = occ[:, -1] = 1
    for x, y in course['tubes']:
        occ[round(y / p) - i0, round(x / p) - j0] = 1
    return occ


def optimal_route(course):
    """A* route start -> goal (see common.grid.plan_route).

    Returns (path, length_m, inflate_used); path is None when the course
    cannot be traversed at any inflation radius."""
    _, path, used = G.plan_route(
        obstacle_points(course), course['start'][:2], course['goal'],
        bounds=planning_bounds(course))
    if path is None:
        return None, None, None
    return path, G.path_length(path), used


def timeout_s(optimal_len_m):
    """Trial timeout for a course of this optimal length."""
    raw = TIMEOUT_S_PER_M * optimal_len_m
    return math.ceil(raw / TIMEOUT_ROUND_S) * TIMEOUT_ROUND_S


def is_layout_file(path):
    """True for a course layout (vs a map_server map) — both are YAML."""
    try:
        raw = yaml.safe_load(open(path))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(raw, dict) and 'tubes' in raw
