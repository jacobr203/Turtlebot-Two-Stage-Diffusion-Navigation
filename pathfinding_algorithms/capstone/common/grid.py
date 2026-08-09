"""Occupancy-grid utilities for BARN worlds: parsing, A*, path post-processing.

BARN world geometry (verified against Daffan/the-barn-challenge @ bf5a226,
worlds 0-299):

  - every world is a 30 x 64 cell arena of static unit cylinders (r = 0.075 m)
    on an exact 0.15 m lattice with origin (-4.425, 0.075) at cell (0, 0);
  - columns i=0 and i=29 are full-height side walls, row j=0 is the back wall,
    and the top (goal side) is open -- the robot cannot route around the
    clutter field, the world itself enforces the corridor;
  - the robot spawns at (-2.25, 3.0, yaw 1.57) and the goal is (-2.25, 13.0)
    with a 1.0 m success radius (run.py, `world_idx < 300` branch);
  - the shipped reference paths (path_files/path_%d.npy) live in their own
    grid frame: gazebo = (x*0.15 - 4.575, y*0.15 + 5.075). That frame is
    offset from the world lattice; it is only used to compute BARN's
    reference path length, where the offset is irrelevant.

Planning happens on a supersampled 0.05 m grid: with 0.15 m cells a Burger-
scale inflation radius (~0.21 m) would close every 3-cell gap outright, while
on the fine grid the free midline through such a gap (0.225 m from either
cylinder center) survives.
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
# Wide-route preference (v2 expert + nav2 route, decisions #25/#27): plan at
# 0.24 m first (only >= 0.30 m-class gaps) and fall back to the 0.21 m gate —
# shortest-path A* would otherwise take an avoidable 0.225 m squeeze.
WIDE_INFLATE_RADIUS = 0.24
# Clearance-centering (plan_centered): per-point uphill step and total cap.
CENTER_ASCEND_STEP = 0.025
CENTER_SHIFT_CAP = 0.15
# Points already clearer than this stay on the raw A* line: in open field
# the uphill push never converges (it just chases distant obstacles) and
# distorts straight stretches — observed as ~180 deg reversals where the
# pinned goal met an ascended tail.
CENTER_MAX_CLEARANCE = 0.45
PLAN_RESOLUTION = 0.05
# Planning bounds cover spawn jitter, the arena, and the goal circle.
PLAN_BOUNDS = ((-5.1, 0.6), (0.9, 14.6)) # (x_min, x_max), (y_min, y_max)


def plan_route(cylinders, start_xy, goal_xy, center_len=None,
               inflates=(WIDE_INFLATE_RADIUS, DEFAULT_INFLATE_RADIUS),
               resolution=PLAN_RESOLUTION, bounds=PLAN_BOUNDS):
    """The v2 route convention shared by the expert collector, the deploy
    subgoal planner, and run.sh's nav2 goal sender: try each inflate radius
    in order (wide corridor preferred), return the first CENTERED route.

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
    Raises if any cylinder is off-lattice (never observed in the dataset).
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
    """BARN's own optimal-path length: reference path in its frame, with the
    official start prepended and goal appended (as the challenge's metric
    computation does)."""
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
    """A* on a supersampled, obstacle-inflated grid around one BARN world.

    Free cells are those whose center is farther than inflate_radius from
    every cylinder center; the boundary of the planning window is blocked so
    paths stay inside PLAN_BOUNDS.
    """

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
        # String-pull shortcuts are validated against a slightly fatter grid:
        # the raw A* cells keep clearance > inflate, but a shortcut segment
        # (and the tracker's small corner-cutting on top of it) can otherwise
        # graze the inflate boundary. Tight gaps stay passable — the pull
        # simply keeps the original cells where there is no spare margin.
        self.blocked_smooth = self._clearance <= self.inflate + 0.03
        self.blocked_smooth[0, :] = self.blocked_smooth[-1, :] = True
        self.blocked_smooth[:, 0] = self.blocked_smooth[:, -1] = True

    # - coordinate helpers -------------------------------------------------
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

    # - planning -----------------------------------------------------------
    def plan_centered(self, start_xy, goal_xy, center_len=None,
                      max_expansions=None):
        """A* start->goal, clearance-CENTERED (v2 expert route, decisions #27).

        Raw A* (and plan()'s string-pull) is taut: it grazes the inflate
        boundary wherever the corridor turns, even when the gap is much
        wider — survival then hangs on sensor noise (#24c). Here every raw
        A* point is pushed uphill on the distance-to-nearest-obstacle field
        (<= CENTER_SHIFT_CAP, which preserves the corridor basin: uphill
        motion moves away from obstacles by construction), with light
        Laplacian smoothing between two ascent passes. Genuinely tight
        midlines are already at their local maximum and stay put.

        center_len, if given, centers only the first N metres — live subgoal
        replanning at a few Hz only needs the local stretch. Unlike plan(),
        a blocked start cell is tolerated (jittered spawns can begin inside
        the inflation band; A* expands outward). Returns (M, 2) float32
        world-frame points ~one grid cell apart, or None if unreachable."""
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
                # the two ascents can leave per-point jitter; one gentle pass
                # rounds it off for a few mm of margin at most
                seg[1:-1] = _smooth_polyline(seg, 1)[1:-1]
                pts[:n] = seg
        pts[0] = start_xy
        pts[-1] = goal_xy
        # NO arc-length resampling here: interpolating between centered
        # points shaves margin back off at corners. Points stay ~one grid
        # cell apart (ascents shift them <= CENTER_SHIFT_CAP coherently);
        # every consumer (tracker, recorder, subgoal walk) measures real
        # segment lengths rather than assuming uniform spacing.
        # Ascents can merge adjacent cells onto the same ridge point — drop
        # near-duplicates (heading between them is numeric noise).
        seg = np.hypot(*np.diff(pts, axis=0).T)
        keep = np.concatenate([[True], seg > 0.02])
        keep[-1] = True
        return pts[keep].astype(np.float32)

    def _ascend(self, pts, orig, iters, cap=CENTER_SHIFT_CAP, active=None):
        """Push points uphill on the clearance field, <= cap from their
        original positions. `active` masks which points may move."""
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
        """Continuous distance from points to the nearest obstacle center."""
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
        """A* start->goal; returns (M, 2) world-frame waypoints (arc-length
        resampled at `spacing`) or None if no path exists."""
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

    max_expansions bounds the closed set: a FAILED search is exhaustive by
    definition, and on a grown live map that ran multi-second on the robot's
    control thread (observed: 10 s wheel-zeroed outages, 2026-08-04).
    partial=True returns the path to the expanded cell nearest the goal
    (by heuristic) instead of None when the goal is not reached — the
    frontier of reachable space facing the target."""
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
            # cooperative GIL yield: this pure-Python loop on
            # a worker thread STARVED the robot's 4 Hz control thread —
            # off-threading alone doesn't help under the GIL (observed:
            # 10-11 s tick stalls ending exactly at picker output,
            # through two "fix" generations).
            # sleep(0), not sleep(0.0005): this A* ALSO runs
            # on the control thread itself (nav subgoal planning), and
            # on WSL2 a nonzero sleep costs 1-10 ms real per yield —
            # the testing "fix" inflated the tick's own searches ~10x
            # and made the storms MORE frequent. sleep(0) releases the
            # GIL with no timer wait.
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
