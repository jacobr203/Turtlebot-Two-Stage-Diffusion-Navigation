"""Per-trial state machines fed from ROS callbacks."""

import math

import numpy as np

from common import grid as G

from .config import (
    PATH_BUCKET_S, PATH_MIN_STEP_M, SCAN_GAP_LIMIT_S, SLAM_LOST_LIMIT_S,
    START_TIMEOUT_S)


class TrialMonitor:
    """Decides one trial's verdict; no ROS, so the rules test offline.

    Times are monotonic seconds; poses are in the start frame."""

    KEYS = {'c': 'collision', 'a': 'aborted', 'x': 'harness_error'}

    def __init__(self, goal_xy, timeout_s,
                 success_radius=G.SUCCESS_RADIUS,
                 start_timeout_s=START_TIMEOUT_S,
                 scan_gap_limit_s=SCAN_GAP_LIMIT_S,
                 slam_lost_limit_s=SLAM_LOST_LIMIT_S):
        self.goal = tuple(goal_xy)
        self.timeout_s = timeout_s
        self.radius = success_radius
        self.start_timeout_s = start_timeout_s
        self.scan_gap_limit_s = scan_gap_limit_s
        self.slam_lost_limit_s = slam_lost_limit_s
        self.start_goal = math.hypot(*self.goal)
        self.t_armed = self.t0 = self.t_end = None
        self.verdict, self.reason = None, ''
        self.pre_arm_cmds = 0
        self.last_scan = self.last_slam = None
        self.scan_gap_max = 0.0
        self.slam_trace, self.odom_trace = [], []
        self.min_goal = math.inf
        self.final_goal = None

    def arm(self, t):
        """The controller is running and has its goal: commands now count."""
        self.t_armed = t

    def on_cmd(self, t):
        if self.t_armed is None:
            self.pre_arm_cmds += 1
        elif self.t0 is None:
            self.t0 = t

    def on_scan(self, t):
        if self.t_armed is not None and self.last_scan is not None:
            self.scan_gap_max = max(self.scan_gap_max, t - self.last_scan)
        self.last_scan = t

    def on_slam(self, t, x, y, yaw):
        self.last_slam = t
        if self.verdict is not None:
            return
        self.slam_trace.append((t, x, y, yaw))
        d = math.hypot(x - self.goal[0], y - self.goal[1])
        self.min_goal = min(self.min_goal, d)
        self.final_goal = d
        if self.t0 is not None and d <= self.radius:
            self.finish('success', t, f'SLAM pose {d:.2f} m from goal')

    def on_odom(self, t, x, y, yaw):
        if self.verdict is not None:
            return
        self.odom_trace.append((t, x, y, yaw))

    def on_key(self, t, key):
        if key in self.KEYS:
            self.finish(self.KEYS[key], t, 'operator')

    def on_controller_exit(self, t):
        if self.t0 is None:
            self.finish('harness_error', t,
                        'controller exited before commanding')
        else:
            self.finish('crash', t, 'controller process exited')

    def check(self, t):
        """Time-based rules; returns the verdict once one is reached."""
        if self.verdict is not None or self.t_armed is None:
            return self.verdict
        if self.t0 is None:
            if t - self.t_armed > self.start_timeout_s:
                self.finish('harness_error', t, 'controller never commanded')
        elif (self.last_scan is not None
              and t - self.last_scan > self.scan_gap_limit_s):
            self.finish('harness_error', t,
                        f'lidar silent {t - self.last_scan:.1f} s')
        elif (self.last_slam is not None
              and t - self.last_slam > self.slam_lost_limit_s):
            self.finish('harness_error', t, 'SLAM pose lost')
        elif t - self.t0 > self.timeout_s:
            self.finish('timeout', t, f'{self.timeout_s:.0f} s elapsed')
        return self.verdict

    def finish(self, verdict, t, reason):
        if self.verdict is None:
            self.verdict, self.reason, self.t_end = verdict, reason, t

    def summary(self):
        ttg = (self.t_end - self.t0) if (self.t0 is not None
                                         and self.t_end is not None) else 0.0
        min_goal = self.min_goal if math.isfinite(self.min_goal) else None
        progress = (max(0.0, 1.0 - min_goal / self.start_goal)
                    if min_goal is not None else 0.0)
        return {
            'verdict': self.verdict, 'verdict_reason': self.reason,
            'time_to_goal_s': round(ttg, 2),
            'gt_path_len': round(trace_length(self.slam_trace, self.t0), 3),
            'odom_path_len': round(trace_length(self.odom_trace, self.t0), 3),
            'min_goal_dist_m': (round(min_goal, 3)
                                if min_goal is not None else ''),
            'final_goal_dist_m': (round(self.final_goal, 3)
                                  if self.final_goal is not None else ''),
            'progress': round(progress, 3),
            'scan_gap_max_ms': round(self.scan_gap_max * 1000, 1),
            'pre_goal_cmds': self.pre_arm_cmds,
            'odom_drift_m': self.odom_drift(),
        }

    def odom_drift(self):
        """Odometry vs SLAM position at the end of the trial [m]: what the
        policy believed against where the map says the robot was."""
        if not self.slam_trace or not self.odom_trace:
            return ''
        t, sx, sy, _ = self.slam_trace[-1]
        _, ox, oy, _ = min(self.odom_trace, key=lambda o: abs(o[0] - t))
        return round(math.hypot(sx - ox, sy - oy), 3)


def trace_length(trace, t0, bucket_s=PATH_BUCKET_S,
                 min_step_m=PATH_MIN_STEP_M):
    """Distance travelled along a (t, x, y, yaw) trace from t0 on."""
    if t0 is None:
        return 0.0
    buckets = {}
    kept = [(t, x, y) for t, x, y, _ in trace if t >= t0]
    if len(kept) < 2:
        return 0.0
    for t, x, y in kept:
        buckets.setdefault(int((t - t0) // bucket_s), []).append((x, y))
    # bucket means sit half a bucket inside each end; pin the true ends
    pts = ([np.array(kept[0][1:])]
           + [np.mean(buckets[k], axis=0) for k in sorted(buckets)]
           + [np.array(kept[-1][1:])])
    length, anchor = 0.0, None
    for p in pts:
        if anchor is None:
            anchor = p
            continue
        step = float(np.hypot(*(p - anchor)))
        if step >= min_step_m:
            length += step
            anchor = p
    return length


class ColourMonitor(TrialMonitor):
    """Colour run: success is every square FOUND, the time cap the timeout.

    Collisions are logged (c) without ending the run. w ends a wedged or
    unsafe run as a counted failure: an abort would let bad runs be re-run."""

    KEYS = {'a': 'aborted', 'x': 'harness_error', 'w': 'stuck'}

    def __init__(self, colors, cap_s, **kw):
        super().__init__((0.0, 0.0), timeout_s=cap_s, **kw)
        self.colors = list(colors)
        self.seen_t, self.found_t = {}, {}
        self.collision_t = []

    def on_slam(self, t, x, y, yaw):
        self.last_slam = t
        if self.verdict is None:
            self.slam_trace.append((t, x, y, yaw))

    def on_status(self, t, status):
        """One /color_test/status message (already JSON-decoded)."""
        if self.verdict is not None:
            return
        for c, state in status.get('states', {}).items():
            if c not in self.colors:
                continue
            if state in ('SEEN', 'FOUND'):
                self.seen_t.setdefault(c, t)
            if state == 'FOUND':
                self.found_t.setdefault(c, t)
        if self.t0 is not None and all(c in self.found_t
                                       for c in self.colors):
            self.finish('success', t, f'all {len(self.colors)} squares found')

    def on_key(self, t, key):
        if key == 'c':
            if self.verdict is None:
                self.collision_t.append(t)
        else:
            super().on_key(t, key)

    def summary(self):
        s = super().summary()
        t0 = self.t0

        def rel(t):
            return round(t - t0, 1) if t is not None and t0 is not None else ''
        out = {k: s[k] for k in ('verdict', 'verdict_reason',
                                 'scan_gap_max_ms', 'pre_goal_cmds',
                                 'odom_drift_m')}
        out.update({
            'total_s': s['time_to_goal_s'], 'path_m': s['gt_path_len'],
            'odom_path_m': s['odom_path_len'],
            'found_count': len(self.found_t), 'colors': ','.join(self.colors),
            'collisions': len(self.collision_t),
            'collision_times_s': ' '.join(str(rel(t))
                                          for t in self.collision_t),
        })
        for c in self.colors:
            out[f'{c}_seen_s'] = rel(self.seen_t.get(c))
            out[f'{c}_found_s'] = rel(self.found_t.get(c))
        return out


def write_traces(trial_dir, mon, t_ref, events):
    for name, trace in (('slam_trace.csv', mon.slam_trace),
                        ('odom_trace.csv', mon.odom_trace)):
        with open(trial_dir / name, 'w') as f:
            f.write('t,x,y,yaw\n')
            for t, x, y, yaw in trace:
                f.write(f'{t - t_ref:.3f},{x:.4f},{y:.4f},{yaw:.4f}\n')
    with open(trial_dir / 'events.csv', 'w') as f:
        f.write('t,event\n')
        for t, e in events:
            f.write(f'{t - t_ref:.2f},"{e}"\n')
