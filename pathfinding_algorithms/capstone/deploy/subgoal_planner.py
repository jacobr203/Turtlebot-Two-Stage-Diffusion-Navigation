"""A* subgoal planners, the high-level stage of the two-stage architecture.

Each returns the point `subgoal_dist` m along the A* path to the goal, matching
the training conditioning. grid_file: BARN *.npy, course layout or map_server
*.yaml; with a *.yaml map, anchor spawn_* at the start pose in the map's frame.
"""

from pathlib import Path

import numpy as np
import yaml

from common import course
from common import grid as G


def _map_yaml_to_points(map_yaml):
    """Occupied-cell centers (world xy) of a map_server map."""
    map_yaml = Path(map_yaml)
    info = yaml.safe_load(open(map_yaml))
    pgm = map_yaml.parent / info['image']
    with open(pgm, 'rb') as f:
        magic = f.readline().split()[0]
        if magic != b'P5':
            raise ValueError(f'{pgm}: only binary P5 pgm supported')
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline())
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)

    occ_thresh = info.get('occupied_thresh', 0.65)
    if info.get('negate', 0):
        occupied = img / maxval > occ_thresh
    else:
        occupied = (maxval - img) / maxval > occ_thresh
    res = info['resolution']
    ox, oy = info['origin'][0], info['origin'][1]
    rows, cols = np.nonzero(occupied)
    # pgm row 0 is the map's TOP edge (y_max)
    return np.stack([ox + (cols + 0.5) * res,
                     oy + (h - rows - 0.5) * res], axis=1)


# Cap for exploratory route queries (route_min_clearance, reachable_toward):
# a failed A* is exhaustive and would block the control thread for seconds.
ROUTE_MAX_EXPANSIONS = 4000


def _free_start(planner, xy, search_r=0.35):
    """Nearest free planning cell to xy: the policy may drift inside the
    inflation radius, which A* refuses as a start."""
    i, j = planner.to_cell(xy)
    if not (0 <= i < planner.ni and 0 <= j < planner.nj):
        return None # outside the planning window (e.g. a bogus odom reading)
    if not planner.blocked[i, j]:
        return xy
    r_cells = int(search_r / planner.res)
    best, best_d = None, 1e9
    for di in range(-r_cells, r_cells + 1):
        for dj in range(-r_cells, r_cells + 1):
            ii, jj = i + di, j + dj
            if 0 <= ii < planner.ni and 0 <= jj < planner.nj \
                    and not planner.blocked[ii, jj]:
                d = di * di + dj * dj
                if d < best_d:
                    best, best_d = (ii, jj), d
    return planner.to_world(best) if best else None


class LiveSubgoalPlanner:
    """Subgoals on the live SLAM map, in the map frame. The goal is per-call
    and unknown cells are traversable, so routes can reach frontiers."""

    def __init__(self, subgoal_dist=1.2, rebuild_period=2.0,
                 inflate_radius=G.DEFAULT_INFLATE_RADIUS,
                 prefer_inflate=None):
        self.subgoal_dist = subgoal_dist
        self.rebuild_period = rebuild_period
        # prefer_inflate: a comfort radius tried first, because the 0.24 m
        # planner routes through pinches the policy and gate cannot thread
        radii = (G.WIDE_INFLATE_RADIUS, inflate_radius) if prefer_inflate \
            is None else (prefer_inflate, G.WIDE_INFLATE_RADIUS, inflate_radius)
        # last-resort squeeze tier: SLAM walls read fatter than truth, so real
        # 0.30 m gaps close on the live map. 0.18 still clears the robot
        # (0.075 cylinder + 0.089 half-diagonal = 0.164).
        radii = tuple(radii) + (0.18,)
        self.inflates = list(dict.fromkeys(radii))
        self.planners = []
        self._pending = None
        self._built = 0.0
        self._last = None

    def update_map(self, msg):
        self._pending = msg # latest wins; consumed on next subgoal

    def _maybe_rebuild(self):
        import threading
        import time
        if self._pending is None or getattr(self, '_building', False):
            return
        if self.planners and time.monotonic() - self._built < self.rebuild_period:
            return
        m, self._pending = self._pending, None
        grid = np.asarray(m.data, dtype=np.int8).reshape(
            m.info.height, m.info.width)
        rows, cols = np.nonzero(grid > 50) # occupied only; unknown = free
        if len(rows) < 3:
            return # map too young to plan on
        if len(rows) > 6000:
            # divergence guard: a sane course map has ~1.5-2.5k occupied cells;
            # a diverged pose graph explodes this and each rebuild costs ~10 s
            if not getattr(self, '_diverged_logged', False):
                self._diverged_logged = True
                print(f'[subgoal] map divergence suspected ({len(rows)} '
                      f'occupied cells) — planner rebuilds paused', flush=True)
            return
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        pts = np.stack([ox + (cols + 0.5) * res,
                        oy + (rows + 0.5) * res], axis=1)
        lo, hi = pts.min(axis=0) - 1.5, pts.max(axis=0) + 1.5
        bounds = ((float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])))

        # build off the control path: a Planner build can take seconds on a
        # large map, and inside the tick that trips the stall watchdog
        self._building = True

        def _build():
            try:
                planners = [G.Planner(pts, inflate_radius=r, bounds=bounds)
                            for r in self.inflates]
                self.planners = planners
                self._built = time.monotonic()
            finally:
                self._building = False

        threading.Thread(target=_build, daemon=True).start()

    def route_min_clearance(self, start_map_xy, goal_map_xy):
        """Minimum obstacle clearance [m] along the A* route, or None without
        a route. A routable goal is not always one the policy can thread."""
        self._maybe_rebuild()
        import time as _t
        t0_rc = _t.monotonic() # tier walk deadline
        for planner in self.planners:
            if _t.monotonic() - t0_rc > 0.6:
                break
            s = _free_start(planner, start_map_xy)
            g = _free_start(planner, goal_map_xy, search_r=0.5)
            if s is None or g is None:
                continue
            try:
                path = planner.plan(s, g,
                                    max_expansions=ROUTE_MAX_EXPANSIONS)
            except ValueError:
                continue
            if path is None:
                continue
            mc = float(min(planner.clearance_at(p) for p in path[::3]))
            # remembered for route_cost: a straight-line cost over-ranks poses
            # behind a wall
            self.last_route_len = float(np.hypot(
                *np.diff(np.asarray(path, dtype=np.float64),
                         axis=0).T).sum()) if len(path) > 1 else 0.0
            return mc
        self.last_route_len = None
        return None

    def route_cost(self, start_map_xy, goal_map_xy):
        """(min_clearance, path_length) of the A* route, or (None, None)."""
        mc = self.route_min_clearance(start_map_xy, goal_map_xy)
        return mc, (self.last_route_len if mc is not None else None)

    def reachable_toward(self, start_map_xy, goal_map_xy):
        """Farthest A*-reachable point toward an unroutable goal (map frame),
        or None. Trips aim here so they end in an arrival that grows the map."""
        self._maybe_rebuild()
        import time as _t
        t0_rt = _t.monotonic() # tier walk deadline: capped failures still sum
        for planner in self.planners:
            if _t.monotonic() - t0_rt > 0.8:
                break
            s = _free_start(planner, start_map_xy)
            if s is None:
                continue
            gi, gj = planner.to_cell(goal_map_xy)
            gcell = (int(np.clip(gi, 0, planner.ni - 1)),
                     int(np.clip(gj, 0, planner.nj - 1)))
            cells = G._astar(planner.blocked, planner.to_cell(s), gcell,
                             max_expansions=ROUTE_MAX_EXPANSIONS,
                             partial=True)
            if cells is None or len(cells) < 2:
                continue
            w = planner.to_world(cells[-1])
            return (float(w[0]), float(w[1]))
        return None

    def subgoal(self, robot_map_xy, goal_map_xy):
        """Map-frame subgoal along the A* route. A brief failure returns the
        last good subgoal; it expires after 4 s of failure or when the goal
        moves >0.6 m, and None hands over to the caller's straight-line carrot,
        so an unreachable goal cannot deadlock the robot on a stale subgoal."""
        import time
        self._maybe_rebuild()
        import time as _t
        t0_sg = _t.monotonic()
        for planner in self.planners:
            # runs on the control thread every tick; past this deadline the
            # caller's carrot fallback takes over
            if _t.monotonic() - t0_sg > 0.5:
                break
            start = _free_start(planner, robot_map_xy)
            goal = _free_start(planner, goal_map_xy, search_r=0.5)
            if start is None or goal is None:
                continue
            try:
                # capped: a failed tier costs ~0.1-0.2 s
                path = planner.plan_centered(
                    start, goal, center_len=self.subgoal_dist + 1.8,
                    max_expansions=6000)
            except ValueError: # outside planning bounds
                continue
            if path is None:
                continue
            seg = np.hypot(*np.diff(path.astype(np.float64), axis=0).T)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            j = min(int(np.searchsorted(s, self.subgoal_dist)), len(path) - 1)
            self._last = tuple(path[j])
            self._last_goal = tuple(goal_map_xy)
            self._last_ok_t = time.monotonic()
            return self._last
        if self._last is not None:
            moved = getattr(self, '_last_goal', None) is None or \
                np.hypot(goal_map_xy[0] - self._last_goal[0],
                         goal_map_xy[1] - self._last_goal[1]) > 0.6
            stale = time.monotonic() - getattr(self, '_last_ok_t', 0.0) > 4.0
            if moved or stale:
                self._last = None
        return self._last


class SubgoalPlanner:
    def __init__(self, grid_file, goal_xy, subgoal_dist=1.2,
                 inflate_radius=G.DEFAULT_INFLATE_RADIUS):
        grid_file = Path(grid_file)
        if grid_file.suffix == '.npy':
            points = G.arena_grid_to_cylinders(np.load(grid_file))
            bounds = G.PLAN_BOUNDS
        elif course.is_layout_file(grid_file):
            layout = course.load(grid_file)
            points = course.obstacle_points(layout)
            bounds = course.planning_bounds(layout)
        else:
            points = _map_yaml_to_points(grid_file)
            lo = np.minimum(points.min(axis=0), goal_xy) - 1.0
            hi = np.maximum(points.max(axis=0), goal_xy) + 1.0
            bounds = ((float(lo[0]), float(hi[0])),
                      (float(lo[1]), float(hi[1])))
        # wide corridor preferred, feasibility radius as fallback. Sticky: once
        # a radius fails we stop retrying it.
        self.planners = [G.Planner(points, inflate_radius=r, bounds=bounds)
                         for r in dict.fromkeys(
                             (G.WIDE_INFLATE_RADIUS, inflate_radius))]
        self._pk = 0
        self.goal = tuple(goal_xy)
        self.subgoal_dist = subgoal_dist
        self._last = None # last good subgoal (world frame)

    def subgoal(self, world_xy):
        """World-frame subgoal for the current pose, or the last good one if
        planning transiently fails. None only if it has never succeeded."""
        for k in range(self._pk, len(self.planners)):
            planner = self.planners[k]
            start = _free_start(planner, world_xy)
            if start is None:
                continue
            # center only the local stretch — the subgoal sits 1.2 m out
            path = planner.plan_centered(start, self.goal,
                                         center_len=self.subgoal_dist + 1.8)
            if path is None:
                continue
            self._pk = k
            seg = np.hypot(*np.diff(path.astype(np.float64), axis=0).T)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            j = min(int(np.searchsorted(s, self.subgoal_dist)),
                    len(path) - 1)
            self._last = tuple(path[j])
            break
        return self._last
