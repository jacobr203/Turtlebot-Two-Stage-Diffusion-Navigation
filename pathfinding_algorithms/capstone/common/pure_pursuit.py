"""Pure-pursuit path tracker.

Used by the expert tracker (following the A* path during data collection)
and by the deployed policy (following the diffusion model's predicted
waypoints between replans). Kinematic limits match the Burger and the
existing DWA configuration (0.22 m/s, 2.84 rad/s).
"""

import math

import numpy as np

from .geometry import world_to_robot


class PurePursuit:
    def __init__(self, path, lookahead=0.25, v_max=0.22, w_max=2.84,
                 goal_tol=0.15, decel_radius=0.40, curve_slowdown=0.12):
        """path: (N, 2) world-frame polyline, densely spaced (<= lookahead)."""
        self.path = np.asarray(path, dtype=np.float64)
        if self.path.ndim != 2 or len(self.path) < 2:
            raise ValueError('path must be (N>=2, 2)')
        self.lookahead = lookahead
        self.v_max = v_max
        self.w_max = w_max
        self.goal_tol = goal_tol
        self.decel_radius = decel_radius
        self.curve_slowdown = curve_slowdown
        self._seg = np.hypot(*np.diff(self.path, axis=0).T)
        self._s = np.concatenate([[0.0], np.cumsum(self._seg)])
        self._idx = 0                # monotonic progress index
        self.cross_track = 0.0       # |lateral error| at the last step [m]

    @property
    def done_fraction(self):
        return self._s[self._idx] / max(self._s[-1], 1e-9)

    def step(self, x, y, yaw):
        """-> (v, w, done). Call at the control rate with the current pose."""
        pose = (x, y, yaw)

        # advance the nearest-point index (never backwards; bounded window
        # so a path that loops near itself can't teleport progress)
        window = slice(self._idx, min(self._idx + 80, len(self.path)))
        d = np.hypot(self.path[window, 0] - x, self.path[window, 1] - y)
        self._idx += int(np.argmin(d))
        self.cross_track = float(d.min())

        goal = self.path[-1]
        dist_goal = math.hypot(goal[0] - x, goal[1] - y)
        remaining = self._s[-1] - self._s[self._idx]
        if dist_goal < self.goal_tol or (self._idx >= len(self.path) - 1
                                         and remaining < self.goal_tol):
            return 0.0, 0.0, True

        # target: first path point >= lookahead ahead (arc length)
        target_s = self._s[self._idx] + self.lookahead
        t_idx = int(np.searchsorted(self._s, target_s))
        t_idx = min(t_idx, len(self.path) - 1)
        tx, ty = world_to_robot(self.path[t_idx], pose)

        heading = math.atan2(ty, tx)
        if tx <= 0.0 or abs(heading) > 1.2:
            # target behind/beside us: rotate in place toward it
            w = max(-self.w_max, min(self.w_max, 2.5 * heading))
            return 0.0, w, False

        ld = max(math.hypot(tx, ty), 1e-6)
        curvature = 2.0 * ty / (ld * ld)
        v = self.v_max / (1.0 + self.curve_slowdown * abs(curvature) ** 2)
        v = min(v, self.v_max * max(dist_goal, 0.05) / self.decel_radius) \
            if dist_goal < self.decel_radius else v
        w = max(-self.w_max, min(self.w_max, v * curvature))
        return v, w, False
