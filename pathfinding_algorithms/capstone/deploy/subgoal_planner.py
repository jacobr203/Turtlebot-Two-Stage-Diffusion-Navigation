"""Live A* subgoal planner — the two-stage architecture's high-level stage.

Wraps common.grid.Planner around a known occupancy grid and, given the
robot's current world pose, returns the point `subgoal_dist` m along the
remaining A* path to the goal — the same quantity the expert recorder
logged as the training conditioning.

grid_file may be:
  *.npy          a BARN arena grid (sim) — obstacle points are the exact
                 cylinder centers, planning bounds are the BARN course;
  course layout  a physical test course (common/course.py) — tube and
                 wall centres in the start frame, planned exactly like a
                 BARN grid; the real-robot benchmark's prior map;
  *.yaml / *.yml a map_server/cartographer map (real robot) — occupied
                 pgm cells become obstacle points and the planning bounds
                 come from the map extent (+ the goal, + 1 m margin).
                 The "world" frame is then the map's own frame; anchor the
                 policy node's spawn_* at the robot's start pose in it
                 (or start bringup at the origin and use spawn 0,0,0).

The Planner (KD-tree + clearance field) is built once; each replan is just
A* + string-pull (~10-30 ms on the 0.05 m grid).
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


# Cap for exploratory route queries (route_min_clearance /
# reachable_toward): a FAILED A* is exhaustive, and on a 40-min live map
# that blocked the control thread for up to 10 s (wheels zeroed,
# 2026-08-04 testing). 4000 expansions is generous for any genuine route
# on these maps and bounds a failure to well under the callers' budgets.
# Navigation planning (subgoal/plan_centered) stays uncapped.
ROUTE_MAX_EXPANSIONS = 4000


def _free_start(planner, xy, search_r=0.35):
    """Nearest free planning cell to xy — the policy may drift closer to an
    obstacle than the inflation radius allows as an A* start (shared by the
    static and live planners)."""
    i, j = planner.to_cell(xy)
    if not (0 <= i < planner.ni and 0 <= j < planner.nj):
        return None # pose outside the planning window (e.g. a transient
                      # bogus odom reading) — caller falls back to the
                      # last good subgoal
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
    """Two-stage subgoal planning on the LIVE Cartographer map (the 3-color
    scavenger seeker, color_test_plan.md §5 — identical in sim and real).

    Differences from the static SubgoalPlanner: the occupancy grid arrives
    at runtime (update_map, rebuild throttled — a Planner build is
    ~50-150 ms on a room-sized map), the goal is per-call (seek targets
    move as detections refine), and UNKNOWN cells are traversable — the
    seeker must route toward frontiers, and actual collision safety out
    there belongs to the safety gate. All coordinates are in the MAP
    frame; the caller owns the map<->odom<->world conversions.
    """

    def __init__(self, subgoal_dist=1.2, rebuild_period=2.0,
                 inflate_radius=G.DEFAULT_INFLATE_RADIUS,
                 prefer_inflate=None):
        self.subgoal_dist = subgoal_dist
        self.rebuild_period = rebuild_period
        # prefer_inflate: try an extra-wide comfort inflation FIRST — the
        # 0.24 m wide planner happily routes through ~0.26-0.35 m pinches
        # that the policy+gate cannot thread (2026-07-23: 16 recovery
        # cycles in one run). Reachability is kept by the fallbacks.
        radii = (G.WIDE_INFLATE_RADIUS, inflate_radius) if prefer_inflate \
            is None else (prefer_inflate, G.WIDE_INFLATE_RADIUS, inflate_radius)
        # last-resort squeeze tier (2026-08-04, world 768): cartographer
        # walls read fatter than truth, so real 0.30 m mouths choke below
        # the 0.21 m gate on the LIVE map — the whole northern frontier
        # (and the third square) was A*-unreachable at every standard
        # tier (frontier push 0/5 routable; 5/5 at 0.18). 0.18 still
        # clears the physical robot (0.075 cyl + 0.089 half-diagonal =
        # 0.164); the safety gate and waypoint veto own the real squeeze.
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
            # divergence guard (2026-07-23, world 613): a sane course map
            # has ~1.5-2.5k occupied cells; a diverged Cartographer pose
            # graph explodes this and each rebuild cost ~10 s — keep the
            # last good planner rather than churn on a broken map
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

        # rebuild OFF the control path (2026-07-23): Planner construction
        # on a grown late-run map took seconds INSIDE the 4 Hz control
        # tick, tripping the tick-stall watchdog (wheels zeroed mid-fan —
        # robot paralyzed at the swatch). Build in a daemon thread and
        # swap atomically; subgoal keeps using the previous planners.
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
        """Minimum obstacle clearance [m] along the A* route start->goal,
        or None when no route exists (or no planner yet). The seek explorer
        uses this to sweep wide-corridor faces before tight-pocket ones —
        an A*-routable goal is not necessarily a goal the POLICY can thread
        without collisions (observed: repeated wedging in ~0.3 m gaps)."""
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
            # remember the PATH LENGTH of the last successful route
            # (ranking viewpoints by straight-line
            # cost re-introduces the undefined-cost defect — a pose
            # behind a wall is cheap by ruler, expensive by path, so
            # the ruler over-ranks the near hemisphere that just
            # failed). Same A*, no extra planning work.
            self.last_route_len = float(np.hypot(
                *np.diff(np.asarray(path, dtype=np.float64),
                         axis=0).T).sum()) if len(path) > 1 else 0.0
            return mc
        self.last_route_len = None
        return None

    def route_cost(self, start_map_xy, goal_map_xy):
        """(min_clearance, path_length) for the A* route, or (None,
        None) when unroutable — the pair the viewpoint scorer ranks."""
        mc = self.route_min_clearance(start_map_xy, goal_map_xy)
        return mc, (self.last_route_len if mc is not None else None)

    def reachable_toward(self, start_map_xy, goal_map_xy):
        """Farthest A*-reachable point toward an unroutable goal — the
        frontier of reachable space facing it (map frame), or None.
        Best-effort face trips aim here instead of at the face itself:
        cycle-4 trips at unroutable viewpoints stalled the progress leash
        82% of the time (2026-08-04); a reachable approach point converts
        those into arrivals that grow the map at the blocking band."""
        self._maybe_rebuild()
        import time as _t
        t0_rt = _t.monotonic() # tier walk deadline: all
        # 4 tiers of capped failures still
        # summed to ~1 s per call
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
        """Map-frame subgoal along the A* route robot->goal, or the last
        good one on BRIEF transient failure. None until the first success —
        and None again once failures persist or the goal moves.

        The unconditional last-good fallback caused a standstill deadlock
        (2026-07-22): with an unreachable goal (phantom swatch estimate) or
        a collision-shifted map, planning fails EVERY tick, the robot
        reaches the stale subgoal and conditions the policy on a zero
        offset forever. Stale subgoals now expire after 4 s of continuous
        failure and whenever the goal moves >0.6 m from the one that
        produced them, so the caller's straight-line carrot takes over."""
        import time
        self._maybe_rebuild()
        import time as _t
        t0_sg = _t.monotonic()
        for planner in self.planners:
            # tier-walk deadline: this runs ON the control
            # thread every tick — toward an unreachable best-effort
            # goal all tiers fail exhaustively and the tick blocked for
            # the full walk (the testing storm residue). Past the
            # deadline the caller's carrot fallback takes over, which
            # is the designed unreachable-goal path anyway.
            if _t.monotonic() - t0_sg > 0.5:
                break
            start = _free_start(planner, robot_map_xy)
            goal = _free_start(planner, goal_map_xy, search_r=0.5)
            if start is None or goal is None:
                continue
            try:
                # capped (tightened 12k left the
                # worst tick ~4 s; 6k with sleep(0) yields ≈ 0.1-0.2 s
                # per failed tier)
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
        # v2 route convention (matches the literature collector): wide corridor
        # (0.24 m) preferred, feasibility gate as fallback, clearance-centered.
        # Sticky preference: once a radius fails from some pose we stop
        # retrying it (a world either has a wide route or it does not).
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
