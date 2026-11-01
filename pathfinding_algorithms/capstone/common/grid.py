"""Occupancy-grid utilities for BARN worlds: parsing, A*, path post-processing.

A BARN world is a 30 x 64 cylinder lattice with an open goal side. Planning uses a
0.05 m grid because at 0.15 m a Burger inflation radius closes every 3-cell gap.
"""

import heapq
import math
import time
import xml.etree.ElementTree as ET

import numpy as np

CELL = 0.15 # BARN lattice pitch [m]
CYL_RADIUS = 0.075 # obstacle cylinder radius [m]
ARENA_ORIGIN = (-4.425, 0.075) # world xy of arena cell (0, 0)
START_POSE = (-2.25, 3.0, 1.5708) # spawn (x, y, yaw) for worlds 0-299
GOAL_XY = (-2.25, 13.0)
SUCCESS_RADIUS = 1.0

# Reference-path frame (path_coord_to_gazebo_coord in the challenge run.py)
PATH_X_SHIFT = -CYL_RADIUS - 30 * CYL_RADIUS * 2 # 4.575
PATH_Y_SHIFT = CYL_RADIUS + 5 # 5.075

# Burger: 138 x 178 mm footprint -> half-diagonal 0.113 m. Inflation keeps the
# planned path >= this far from cylinder CENTERS; 0.21 m = cylinder radius
# + half-diagonal + ~2 cm margin, and still passes 3-cell gaps (midline at
# 0.225 m). Collisions during collection are gated by the bumper sensor.
DEFAULT_INFLATE_RADIUS = 0.21
# Tried first, falling back to 0.21 m: shortest-path A* would otherwise take an
# avoidable 0.225 m squeeze.
WIDE_INFLATE_RADIUS = 0.24
# plan_centered: per-point uphill step and total cap [m].
CENTER_ASCEND_STEP = 0.025
CENTER_SHIFT_CAP = 0.15
# Points clearer than this stay on the raw A* line: in open field the uphill
# push never converges and distorts straight stretches.
CENTER_MAX_CLEARANCE = 0.45
PLAN_RESOLUTION = 0.05
PLAN_BOUNDS = ((-5.1, 0.6), (0.9, 14.6)) # (x_min, x_max), (y_min, y_max)


def plan_route(cylinders, start_xy, goal_xy, center_len=None,
               inflates=(WIDE_INFLATE_RADIUS, DEFAULT_INFLATE_RADIUS),
               resolution=PLAN_RESOLUTION, bounds=PLAN_BOUNDS):
    """Return the first centered route across inflate radii, widest first.

    Returns (planner, path, inflate_used); path is None if no radius works.
    """
    planner = path = used = None
    for inflate in inflates:
        planner = Planner(cylinders, inflate_radius=inflate,
                          resolution=resolution, bounds=bounds)
        path = planner.plan_centered(start_xy, goal_xy,
                                     center_len=center_len)
        used = inflate
        if path is not None:
            break
    return planner, path, used


def parse_world_cylinders(world_file):
    """Return (N, 2) array of cylinder center xy from a BARN .world file."""
    world = ET.parse(world_file).getroot().find('world')
    pts = []
    for model in world.findall('model'):
        if model.get('name', '').startswith('unit_cylinder'):
            x, y = map(float, model.find('pose').text.split()[:2])
            pts.append((x, y))
    return np.asarray(pts, dtype=np.float64)


def cylinders_to_arena_grid(cylinders, tol=1e-6):
    """Snap cylinder centers onto the BARN lattice.

    Returns (occ, origin): occ is a (ni, nj) uint8 array indexed [i, j] with
    i along +x and j along +y; origin is the world xy of cell (0, 0).
    Raises if any cylinder is off-lattice.
    """
    origin = (cylinders[:, 0].min(), cylinders[:, 1].min())
    ij = (cylinders - origin) / CELL
    err = np.abs(ij - np.round(ij)).max()
    if err > tol:
        raise ValueError(f'cylinders off the {CELL} m lattice by {err:.2e}')
    ij = np.round(ij).astype(int)
    occ = np.zeros((ij[:, 0].max() + 1, ij[:, 1].max() + 1), dtype=np.uint8)
    occ[ij[:, 0], ij[:, 1]] = 1
    return occ, origin


def arena_grid_to_cylinders(occ, origin=ARENA_ORIGIN):
    """Inverse of cylinders_to_arena_grid (lattice is exact, so lossless)."""
    ii, jj = np.nonzero(occ)
    return np.stack([origin[0] + ii * CELL, origin[1] + jj * CELL], axis=1)


def reference_path_length(path_file):
    """BARN's optimal-path length: its reference path plus official start and goal."""
    path = np.load(path_file).astype(np.float64)
    pts = np.stack([path[:, 0] * CELL + PATH_X_SHIFT,
                    path[:, 1] * CELL + PATH_Y_SHIFT], axis=1)
    pts = np.vstack([[START_POSE[:2]], pts, [GOAL_XY]])
    return path_length(pts)


def path_length(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


class Planner:
    """A* on a supersampled, obstacle-inflated grid around one BARN world."""

    def __init__(self, cylinders, inflate_radius=DEFAULT_INFLATE_RADIUS,
                 resolution=PLAN_RESOLUTION, bounds=PLAN_BOUNDS):
        self.res = float(resolution)
        self.inflate = float(inflate_radius)
        (x0, x1), (y0, y1) = bounds
        self.x0, self.y0 = float(x0), float(y0)
        self.ni = int(round((x1 - x0) / self.res)) + 1
        self.nj = int(round((y1 - y0) / self.res)) + 1
        self.cylinders = np.asarray(cylinders, dtype=np.float64)

        xs = self.x0 + np.arange(self.ni) * self.res
        ys = self.y0 + np.arange(self.nj) * self.res
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        centers = np.stack([gx.ravel(), gy.ravel()], axis=1)
        try:
            from scipy.spatial import cKDTree
            dist, _ = cKDTree(self.cylinders).query(centers)
        except ImportError:
            diff = centers[:, None, :] - self.cylinders[None, :, :]
            dist = np.sqrt((diff ** 2).sum(-1)).min(axis=1)
        self._clearance = dist.reshape(self.ni, self.nj).astype(np.float32)
        self.blocked = self._clearance <= self.inflate
        self.blocked[0, :] = self.blocked[-1, :] = True
        self.blocked[:, 0] = self.blocked[:, -1] = True
        # Shortcuts use a slightly fatter grid so they (plus the tracker's
        # corner-cutting) cannot graze the inflate boundary.
        self.blocked_smooth = self._clearance <= self.inflate + 0.03
        self.blocked_smooth[0, :] = self.blocked_smooth[-1, :] = True
        self.blocked_smooth[:, 0] = self.blocked_smooth[:, -1] = True

    def to_cell(self, xy):
        return (int(round((xy[0] - self.x0) / self.res)),
                int(round((xy[1] - self.y0) / self.res)))

    def to_world(self, ij):
        return (self.x0 + ij[0] * self.res, self.y0 + ij[1] * self.res)

    def clearance_at(self, xy):
        """Distance from xy to the nearest cylinder center [m]."""
        i, j = self.to_cell(xy)
        i = min(max(i, 0), self.ni - 1)
        j = min(max(j, 0), self.nj - 1)
        return float(self._clearance[i, j])

    def plan_centered(self, start_xy, goal_xy, center_len=None,
                      max_expansions=None):
        """A* start->goal with points pushed uphill on the clearance field.

        Raw A* grazes the inflate boundary at every turn, even in wide gaps.
        center_len, if given, centers only the first N metres. A blocked start
        cell is tolerated (jittered spawns can begin inside the inflation band).
        Returns (M, 2) float32 world-frame points ~one grid cell apart, or None
        if unreachable."""
        cells = _astar(self.blocked, self.to_cell(start_xy),
                       self.to_cell(goal_xy), max_expansions=max_expansions)
        if cells is None:
            return None
        pts = np.array([self.to_world(c) for c in cells], dtype=np.float64)
        n = len(pts) if center_len is None else \
            min(len(pts), int(center_len / self.res) + 2)
        if n >= 3:
            seg, orig = pts[:n].copy(), pts[:n].copy()
            active = self._nearest_dist(orig) < CENTER_MAX_CLEARANCE
            if active.any():
                seg = self._ascend(seg, orig, 10, active=active)
                seg[1:-1] = _smooth_polyline(seg, 5)[1:-1]
                seg = self._ascend(seg, orig, 5, active=active)
                # round off per-point jitter left by the two ascents
                seg[1:-1] = _smooth_polyline(seg, 1)[1:-1]
                pts[:n] = seg
        pts[0] = start_xy
        pts[-1] = goal_xy
        # No arc-length resampling: interpolation shaves margin off corners.
        # Drop near-duplicates, as ascents can merge cells onto one ridge point.
        seg = np.hypot(*np.diff(pts, axis=0).T)
        keep = np.concatenate([[True], seg > 0.02])
        keep[-1] = True
        return pts[keep].astype(np.float32)

    def _ascend(self, pts, orig, iters, cap=CENTER_SHIFT_CAP, active=None):
        """Push points uphill, <= cap from orig; `active` masks which may move."""
        dirs = np.vstack([[0.0, 0.0],
                          np.stack([np.cos(np.arange(8) * np.pi / 4),
                                    np.sin(np.arange(8) * np.pi / 4)],
                                   axis=1)]) * CENTER_ASCEND_STEP
        for _ in range(iters):
            cand = pts[:, None, :] + dirs[None, :, :]
            d = self._nearest_dist(cand.reshape(-1, 2)).reshape(len(pts), -1)
            shift = cand - orig[:, None, :]
            d[np.hypot(shift[:, :, 0], shift[:, :, 1]) > cap] = -1.0
            best = d.argmax(axis=1) # ties resolve to index 0 = stay put
            if active is not None:
                best[~active] = 0
            if not best.any():
                break
            pts = cand[np.arange(len(pts)), best]
        return pts

    def _nearest_dist(self, pts):
        """Distance from points to the nearest obstacle center."""
        pts = np.asarray(pts, dtype=np.float64)
        try:
            from scipy.spatial import cKDTree
            if not hasattr(self, '_kdtree'):
                self._kdtree = cKDTree(self.cylinders)
            return self._kdtree.query(pts)[0]
        except ImportError:
            diff = pts[:, None, :] - self.cylinders[None, :, :]
            return np.sqrt((diff ** 2).sum(-1)).min(axis=1)

    def plan(self, start_xy, goal_xy, spacing=0.05, max_expansions=None):
        """A* start->goal; (M, 2) waypoints every `spacing` m, or None."""
        start, goal = self.to_cell(start_xy), self.to_cell(goal_xy)
        for name, c in (('start', start), ('goal', goal)):
            if not (0 <= c[0] < self.ni and 0 <= c[1] < self.nj):
                raise ValueError(f'{name} {c} outside planning bounds')
            if self.blocked[c]:
                raise ValueError(f'{name} cell {c} is inside an obstacle')
        cells = _astar(self.blocked, start, goal,
                       max_expansions=max_expansions)
        if cells is None:
            return None
        cells = _string_pull(self.blocked_smooth, cells)
        pts = np.array([self.to_world(c) for c in cells], dtype=np.float64)
        pts[0], pts[-1] = start_xy, goal_xy
        return _resample(pts, spacing)

    def min_clearance(self, pts):
        """Minimum cylinder-center distance along a polyline [m]."""
        pts = np.asarray(pts, dtype=np.float64)
        try:
            from scipy.spatial import cKDTree
            dist, _ = cKDTree(self.cylinders).query(pts)
            return float(dist.min())
        except ImportError:
            diff = pts[:, None, :] - self.cylinders[None, :, :]
            return float(np.sqrt((diff ** 2).sum(-1)).min())


_SQRT2 = math.sqrt(2.0)
_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2)]


def _astar(blocked, start, goal, max_expansions=None, partial=False):
    """8-connected A* (no corner cutting). Returns list of (i, j) or None.

    max_expansions bounds the closed set: a failed search is exhaustive and on
    a grown live map can stall the robot's control thread for seconds.
    partial=True returns the path to the expanded cell nearest the goal instead
    of None when the goal is not reached."""
    ni, nj = blocked.shape
    g = {start: 0.0}
    came = {}
    h = lambda c: math.hypot(c[0] - goal[0], c[1] - goal[1])
    open_heap = [(h(start), start)]
    closed = set()
    best, best_h = start, h(start)

    def _trace(cur):
        path = [cur]
        while cur in came:
            cur = came[cur]
            path.append(cur)
        return path[::-1]

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            return _trace(cur)
        if cur in closed:
            continue
        closed.add(cur)
        n_closed = len(closed)
        if max_expansions is not None and n_closed > max_expansions:
            break
        if n_closed % 256 == 0:
            # Yield the GIL so this loop cannot starve the 4 Hz control thread.
            # sleep(0) because a nonzero sleep costs 1-10 ms on WSL2, and this
            # A* also runs on the control thread itself.
            time.sleep(0)
        hc = h(cur)
        if hc < best_h:
            best, best_h = cur, hc
        ci, cj = cur
        for di, dj, cost in _NEIGHBORS:
            ni_, nj_ = ci + di, cj + dj
            if not (0 <= ni_ < ni and 0 <= nj_ < nj) or blocked[ni_, nj_]:
                continue
            if di and dj and (blocked[ci + di, cj] or blocked[ci, cj + dj]):
                continue # no cutting corners diagonally
            cand = g[cur] + cost
            nxt = (ni_, nj_)
            if cand < g.get(nxt, math.inf):
                g[nxt] = cand
                came[nxt] = cur
                heapq.heappush(open_heap, (cand + h(nxt), nxt))
    if partial and best != start:
        return _trace(best)
    return None


def _line_free(blocked, a, b):
    """True if the a->b segment stays on free cells (sampled at 1/4 cell)."""
    steps = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) * 4))
    for t in np.linspace(0.0, 1.0, steps + 1):
        i = int(round(a[0] + (b[0] - a[0]) * t))
        j = int(round(a[1] + (b[1] - a[1]) * t))
        if blocked[i, j]:
            return False
    return True


def _string_pull(blocked, cells):
    """Greedy shortcutting: keep only waypoints needed for line-of-sight."""
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    anchor = 0
    while anchor < len(cells) - 1:
        far = anchor + 1
        for k in range(len(cells) - 1, anchor, -1):
            if _line_free(blocked, cells[anchor], cells[k]):
                far = k
                break
        out.append(cells[far])
        anchor = far
    return out


def _smooth_polyline(pts, iters):
    """Light Laplacian smoothing (endpoints pinned) — rounds cell jaggies."""
    out = np.asarray(pts, dtype=np.float64).copy()
    for _ in range(iters):
        out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out


def _resample(pts, spacing):
    """Arc-length uniform resampling of a polyline (keeps both endpoints)."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < spacing:
        return pts.astype(np.float32)
    n = int(total / spacing) + 1
    si = np.linspace(0.0, total, n + 1)
    out = np.stack([np.interp(si, s, pts[:, 0]),
                    np.interp(si, s, pts[:, 1])], axis=1)
    return out.astype(np.float32)
