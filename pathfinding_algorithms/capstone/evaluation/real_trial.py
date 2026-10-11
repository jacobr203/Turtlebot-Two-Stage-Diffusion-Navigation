#!/usr/bin/env python3
"""Real-robot benchmark trials: one course, one operator, many trials.

The hardware counterpart of trial.py + run_matrix.py (protocol and rules:
notes/real_robot_data_plan.md §5). Every trial starts from a full reset:
all ROS processes on both machines are killed and the robot's bringup is
restarted, because odometry only zeroes at bringup and the policy's frame
(spawn 0,0,0) and the two-stage prior map both assume it is zero at the
start mark. Carrying the robot back does not turn its wheels, so without
the restart odometry would still hold the previous trial's end pose.

Ground truth comes from a fresh lidar-only Cartographer per trial (map
frame = start pose). Success is the SLAM pose entering the 1 m goal
circle; collisions are called by the operator (there is no bumper).

  check      Day-0 readiness: link, clock, sensor rates and gaps, battery
             (never moves the wheels; drive with teleop to measure motion)

Colour scavenger task in an arena (common/arena.py) — the real-robot
protocol: two-stage seek vs single-stage wander, 4 Hz, safety gate on,
the run ends when every square is FOUND or at the time cap:

  map-arena         map the built arena from the start mark (you teleop),
                    save it, and extract the poles into a new arena file
  colour-schedule   seeded run order for one arena placement
  colour-run        walk it with the operator
  colour-status     counted / pending runs and a per-model summary

Point-to-point benchmark on a course (common/course.py):

  schedule   write the seeded run order for one course (before the session)
  run        walk that schedule with the operator, one trial at a time
  status     counted / pending cells for a run

  .venv/bin/python evaluation/real_trial.py schedule \\
      --course evaluation/courses/A.yaml --run-id bench_v1
  .venv/bin/python evaluation/real_trial.py run \\
      --course evaluation/courses/A.yaml --run-id bench_v1
  .venv/bin/python evaluation/real_trial.py run ... --shakedown   # rehearsal

Operator keys during a point-to-point trial: c = collision, a = abort
(external interference), x = harness fault, space = stop now and classify
after. In a colour run a collision does not end the run (c logs it), and
w ends a wedged or unsafe run as a counted failure.
"""

import argparse
import csv
import json
import fcntl
import math
import os
import random
import re
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arena as A                                 # noqa: E402
from common import course as C                                # noqa: E402
from common import grid as G                                  # noqa: E402
from common.geometry import wrap_angle, yaw_from_quaternion   # noqa: E402

CAPSTONE = Path(__file__).resolve().parents[1]
WS = CAPSTONE.parent
WS_SETUP = WS / 'install' / 'setup.bash'
VENV_PY = CAPSTONE / '.venv' / 'bin' / 'python'
RESULTS = CAPSTONE / 'results' / 'real'
ROBOT_ENV = WS / 'robot.env'

CKPTS = {
    'two_stage': CAPSTONE / 'diffusion/runs/two_stage_v5/best.pt',
    'single_stage': CAPSTONE / 'diffusion/runs/single_stage_v5/best.pt',
}
MODELS = ('two_stage', 'single_stage', 'dwa')
HZ_LIST = (2.0, 3.0, 4.0)
ROUNDS = 3
COUNTED = ('success', 'collision', 'timeout', 'crash')
RERUN = ('aborted', 'harness_error')
POLICY_THREADS = 4 # matches the v5 sim matrix (eval_full_v5_*.yaml)

# Colour task (decided 2026-09-28: two-stage vs single-stage, 4 Hz, gate on)
COLOUR_MODELS = ('two_stage', 'single_stage')
COLOUR_BEHAVIOR = {'two_stage': 'seek', 'single_stage': 'wander'}
COLOUR_HZ = 4.0
COLOUR_ROUNDS = 5
DEFAULT_CAP_MIN = 15.0 # sim colour runs finished in 5-14 min
CAMERA_WAIT_S = 20.0
LOCALIZE_SCANS = 8 # stationary scans stacked for the start-pose match
START_OFF_M = 0.15 # counted runs must start this close to the mark...
START_OFF_DEG = 10.0 # ...and this close to its heading
START_MATCH_WARN = 0.85
SEEK_MIN_GOAL_DIST = 0.5 # run.sh's --min_goal_dist default
INTERRUPTED = 'Ctrl-C during the run'
COLOUR_BAG_TOPICS = ['/image_raw/compressed', '/camera_info', '/scan_camera',
                     '/color_targets', '/color_test/status']

# Trial rules
START_TIMEOUT_S = 45.0 # controller ready -> first command
SCAN_GAP_LIMIT_S = 2.0 # longer lidar silence = harness fault, not the model
SLAM_LOST_LIMIT_S = 3.0
# Path length: average the trace in short time buckets, then count only
# steps of at least PATH_MIN_STEP_M, so a parked robot's pose jitter adds
# nothing while a driving robot (0.2 m/s = 5 cm per bucket) loses nothing.
PATH_BUCKET_S = 0.25
PATH_MIN_STEP_M = 0.02
MONITOR_PERIOD_S = 0.1

# Bring-up and teardown
BRINGUP_TIMEOUT_S = 30.0
SENSOR_WAIT_S = 15.0
NODE_TIMEOUT_S = 60.0 # the policy loads torch + a checkpoint
SLAM_READY_TIMEOUT_S = 30.0
ODOM_ZERO_TOL_M = 0.02
ODOM_ZERO_TOL_RAD = 0.03
STOP_V_TOL = 0.01
STOP_W_TOL = 0.05
BAG_CLOSE_GRACE_S = 15.0
BAG_START_CHECK_S = 2.0 # a recorder that fails (e.g. folder exists) exits fast
DEFAULT_MIN_BATTERY_V = 11.1
# Day-0 check: raise the policy deadman when the lidar or odometry ever
# goes this quiet — the 0.6 s default would hold the wheels mid-trial.
DEADMAN_RAISE_GAP_S = 0.5
RAISED_DEADMAN_S = 1.0
CLOCK_OFFSET_OK_MS = 50.0
CLOCK_OFFSET_FAIL_MS = 1000.0

BAG_TOPICS = ['/scan', '/odom', '/tf', '/tf_static', '/cmd_vel', '/imu',
              '/battery_state', '/map', '/policy/waypoints',
              '/policy/subgoal', '/goal_pose', '/rosout']

# Bracketed patterns never match the pkill command line that carries them.
LOCAL_KILL_PATTERNS = [
    'policy_nod[e]', 'dwa_nod[e]', 'cartographer_nod[e]',
    'cartographer_occupancy_grid_nod[e]', 'scan_relay_nod[e]',
    'tf_relay_nod[e]', 'cartographer_only.launch.p[y]', 'bag recor[d]',
    'color_seeke[r]', 'color_test_manage[r]', 'camera_obstacle_nod[e]',
]
def detached(cmd, log):
    """A remote command that outlives the SSH call without holding it open.

    `a && b &` backgrounds the whole list, and the shell running it keeps
    the channel's stdout/stderr — so SSH waits for it forever (bringup
    never exits). Grouping the list and redirecting the GROUP releases
    the channel at once."""
    return f'({cmd}) > {log} 2>&1 </dev/null &'


ROBOT_KILL = detached("pkill -9 -f '/opt/ro[s]/'; "
                      "pkill -9 -f 'turtlebot[3]'; pkill -9 -f 'ros[2]'",
                      '/tmp/real_trial_kill.log')
ROBOT_ALIVE = "pgrep -af '/opt/ro[s]/|turtlebot[3]|ros[2] ' || true"
ROBOT_BRINGUP = detached(
    'source /opt/ros/humble/setup.bash && '
    'source $HOME/turtlebot3_ws/install/setup.bash && '
    'export TURTLEBOT3_MODEL=burger LDS_MODEL=LDS-02 '
    'RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && '
    'exec ros2 launch turtlebot3_bringup robot.launch.py',
    '/tmp/tbot_bringup.log')

ROW_FIELDS = [
    # sim-compatible (trial.py / aggregate.py)
    'model', 'world_idx', 'hz', 'trial', 'verdict', 'time_to_goal_s',
    'gt_path_len', 'collided_with', 'lat_sample_ms', 'lat_plan_ms',
    'lat_max_ms',
    # real-robot additions
    'run_id', 'course', 'session_id', 'order_idx', 't_start_wall', 'ckpt',
    'git_sha', 'git_dirty', 'odom_path_len', 'min_goal_dist_m',
    'final_goal_dist_m', 'progress', 'collision_source', 'verdict_reason',
    'scan_age_p95_ms', 'scan_gap_max_ms', 'deadman_lines', 'recovery_lines',
    'pre_goal_cmds', 'odom_drift_m', 'battery_v_start', 'battery_v_end',
    'ping_p95_ms', 'stop_verified', 'notes',
]
COLOUR_ROW_FIELDS = [
    'model', 'arena', 'placement', 'lighting', 'hz', 'trial', 'verdict',
    'total_s',
    'found_count', 'colors', 'path_m', 'odom_path_m', 'collisions',
    'collision_times_s',
    *[f'{c}_found_s' for c in A.KNOWN_COLORS],
    *[f'{c}_seen_s' for c in A.KNOWN_COLORS],
    'lat_sample_ms', 'lat_plan_ms', 'lat_max_ms', 'scan_age_p95_ms',
    'scan_gap_max_ms', 'deadman_lines', 'recovery_lines', 'gate_lines',
    'odom_drift_m', 'battery_v_start', 'battery_v_end', 'ping_p95_ms',
    'stop_verified', 'cap_s', 'false_found', 'start_x', 'start_y',
    'start_yaw_deg', 'start_match', 'run_id', 'world_idx',
    'session_id', 'order_idx', 't_start_wall', 'ckpt', 'git_sha',
    'git_dirty', 'verdict_reason', 'pre_goal_cmds', 'collisions_late',
    'notes',
]
SESSION_FIELDS = [
    'session_id', 'run_id', 'course', 'lighting', 'shakedown', 'start',
    'end',
    'git_sha', 'git_dirty', 'link', 'ping_p50_ms', 'ping_p95_ms',
    'clock_offset_ms', 'battery_v_start', 'trials_run', 'notes',
]


class HarnessError(RuntimeError):
    """Infrastructure failure: the trial is re-run, never counted."""


class BatteryLow(RuntimeError):
    """Below the swap threshold: no trial starts, no row is written."""


# ── schedule ────────────────────────────────────────────────────────────────
def make_schedule(course, run_id, rounds=ROUNDS, models=MODELS,
                  hz_list=HZ_LIST):
    """Randomized complete blocks: each round runs every (model, hz) cell
    once, in an order seeded by run id + course so it is reproducible and
    fixed before any result is seen."""
    rng = random.Random(f'{run_id}/{course["name"]}')
    rows, order = [], 0
    for rnd in range(rounds):
        cells = [(m, hz) for m in models for hz in hz_list]
        rng.shuffle(cells)
        for m, hz in cells:
            rows.append({'order_idx': order, 'course': course['name'],
                         'world_idx': course['id'], 'model': m, 'hz': hz,
                         'round': rnd})
            order += 1
    return rows


def cell_key(row):
    return (row['model'], int(row['world_idx']), float(row['hz']),
            int(row['round'] if 'round' in row else row['trial']))


def pending_cells(schedule, done_rows):
    """Cells still owed a counted verdict: never-attempted cells in schedule
    order first, then cells whose last attempt was aborted/harness_error."""
    last = {}
    for r in done_rows:
        last[cell_key(r)] = r['verdict']
    fresh = [c for c in schedule if cell_key(c) not in last]
    retry = [c for c in schedule if last.get(cell_key(c)) in RERUN]
    return fresh + retry


# ── verdict logic ───────────────────────────────────────────────────────────
class TrialMonitor:
    """Decides one trial's verdict from timestamped observations.

    Pure logic with no ROS, so the rules can be tested offline. Every time
    argument is a monotonic clock reading in seconds; poses are in the
    start frame."""

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
    """Verdict rules for a colour run.

    Success is the checkpoint manager reporting every square FOUND; the
    time cap is the timeout. Collisions do not end the run — the sim
    colour task counts them and carries on — so the operator logs each
    one (c). A robot wedged or doing something unsafe is stopped with w,
    which ends the run as a COUNTED failure: an abort would let bad runs
    be quietly re-run."""

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


# ── small parsers ───────────────────────────────────────────────────────────
def percentile(values, q):
    """q-th percentile (0-100) by linear interpolation; None if empty."""
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def latency_stats(lat_csv):
    """Mean sampling/planning latency, worst tick, p95 scan age."""
    try:
        rows = list(csv.DictReader(open(lat_csv)))
    except OSError:
        return {}
    if not rows:
        return {}
    samp = [float(r['sample_ms']) for r in rows]
    plan = [float(r['plan_ms']) for r in rows]
    out = {'lat_sample_ms': round(sum(samp) / len(samp), 1),
           'lat_plan_ms': round(sum(plan) / len(plan), 1),
           'lat_max_ms': round(max(s + p for s, p in zip(samp, plan)), 1)}
    ages = [float(r['scan_age_ms']) for r in rows if r.get('scan_age_ms')]
    if ages:
        out['scan_age_p95_ms'] = round(percentile(ages, 95), 1)
    return out


_PING_RTT = re.compile(r'time[=<]([\d.]+) ms')


def ping_rtts(text):
    return [float(m) for m in _PING_RTT.findall(text)]


def count_lines(path, needle):
    try:
        return sum(needle in line for line in open(path, errors='replace'))
    except OSError:
        return ''


def relative_pose(origin, pose):
    """pose expressed in the frame of origin (both (x, y, yaw))."""
    ox, oy, oyaw = origin
    dx, dy = pose[0] - ox, pose[1] - oy
    c, s = math.cos(-oyaw), math.sin(-oyaw)
    return (c * dx - s * dy, s * dx + c * dy, wrap_angle(pose[2] - oyaw))


def offset_from_return(remote_s, local_return_s):
    """Remote minus local clock, given the remote timestamp printed at the
    end of a call and the local time the call returned."""
    return remote_s - local_return_s


def next_trial_index(trials_dir):
    """First unused tNNN index: folders, not result rows, decide it, so a
    run interrupted before its row was written can never be reused (its
    bag folder would make the next recording fail)."""
    used = [int(d.name[1:4]) for d in Path(trials_dir).glob('t[0-9][0-9][0-9]_*')
            if d.is_dir()]
    return max(used) + 1 if used else 0



# The terminal as it was when the runner started (canonical, echoing).
# Every prompt restores it first: a child that inherited it and was killed
# mid-way can leave it non-blocking or in cbreak mode, after which input()
# reads end-of-file at once and the session ends with nobody touching it.
_TTY_SANE = None


def _remember_tty():
    global _TTY_SANE
    if sys.stdin.isatty():
        _TTY_SANE = termios.tcgetattr(sys.stdin.fileno())


def ask(prompt):
    """input(), on a terminal put back into the state it started in."""
    if sys.stdin.isatty():
        fd = sys.stdin.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        if flags & os.O_NONBLOCK:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
            print('  (terminal was left non-blocking; repaired)')
        if _TTY_SANE is not None and termios.tcgetattr(fd) != _TTY_SANE:
            termios.tcsetattr(fd, termios.TCSANOW, _TTY_SANE)
            print('  (terminal mode was changed; repaired)')
    return input(prompt)


def read_robot_env(path=ROBOT_ENV):
    """KEY=value pairs from robot.env; the shell environment overrides."""
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"\'')
    for k in ('ROBOT_IP', 'ROBOT_USER', 'ROBOT_PASS'):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def git_state():
    """(short sha, dirty) of the code under test (tracked files only)."""
    def git(*a):
        return subprocess.run(['git', '-C', str(WS), *a], capture_output=True,
                              text=True).stdout.strip()
    sha = git('rev-parse', '--short', 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=no', '--', str(WS))
    return sha, dirty


def add_late_collisions(path, n):
    """Contacts the operator saw but could not key in: added to the run's
    collision count, and kept in their own column because they have no
    time or place on the path."""
    with open(path, newline='') as f:
        header = next(csv.reader(f))
    if 'collisions_late' not in header:
        header.append('collisions_late')
    rows = read_rows(path)
    rows[-1]['collisions_late'] = n
    rows[-1]['collisions'] = int(rows[-1].get('collisions') or 0) + n
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, restval='')
        w.writeheader()
        w.writerows(rows)


def set_last_field(path, key, value):
    """Set one field of the row just appended (adding the column if new)."""
    with open(path, newline='') as f:
        header = next(csv.reader(f))
    if key not in header:
        header.append(key)
    rows = read_rows(path)
    rows[-1][key] = value
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, restval='')
        w.writeheader()
        w.writerows(rows)


def found_frames(bag_dir):
    """{colour: (frame BGR with the detector's box drawn, detection)} at
    the moment each colour was declared FOUND, read back from a run's bag:
    the first /color_test/status naming it FOUND, the last detection of it
    before that, and the camera frame nearest that detection."""
    import cv2
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    def reader(topics):
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id='sqlite3'),
               rosbag2_py.ConverterOptions('', ''))
        r.set_filter(rosbag2_py.StorageFilter(topics=topics))
        return r
    found, last = {}, {}
    r = reader(['/color_test/status', '/color_targets'])
    while r.has_next():
        topic, data, t = r.read_next()
        msg = json.loads(deserialize_message(data, String).data)
        if topic == '/color_targets':
            for d in msg.get('detections', []):
                if d['color'] not in found:
                    last[d['color']] = (t, d)
        else:
            for c, state in msg.get('states', {}).items():
                if state == 'FOUND' and c not in found and c in last:
                    found[c] = last[c]
    out = {}
    for c, (t_det, det) in found.items():
        r = reader(['/image_raw/compressed'])
        r.seek(max(0, t_det - 200_000_000))
        best = None
        while r.has_next():
            _, data, t = r.read_next()
            if best is None or abs(t - t_det) < abs(best[0] - t_det):
                best = (t, data)
            if t > t_det + 200_000_000:
                break
        if best is None:
            continue
        img = deserialize_message(best[1], CompressedImage)
        bgr = cv2.imdecode(np.frombuffer(bytes(img.data), np.uint8),
                           cv2.IMREAD_COLOR)
        if 'bbox' in det:
            x, y, w, h = det['bbox']
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 30), (252, 252, 251), -1)
        cv2.putText(bgr, f"{c} FOUND at {det['distance_m']:.2f} m", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (11, 11, 11), 2)
        out[c] = (bgr, det)
    return out


def review_finds(csv_path, row, trial_dir):
    """Show the operator the camera frame behind every FOUND, right after
    the run, and record any find they judge false in false_found. Never
    allowed to cost the session: a failure here is reported and skipped."""
    import cv2
    try:
        frames = found_frames(trial_dir / 'bag')
    except Exception as e: # review is a convenience, never fatal
        print(f'  (could not read the finds back from the bag: {e})')
        return
    if not frames:
        print('  no FOUND to review in this run')
        return
    for c, (img, _) in frames.items():
        cv2.imwrite(str(trial_dir / f'found_{c}.png'), img)
    pad = np.full((next(iter(frames.values()))[0].shape[0], 12, 3), 252,
                  np.uint8)
    combined = trial_dir / 'finds.png'
    tiles = []
    for img, _ in frames.values():
        tiles += [img, pad]
    cv2.imwrite(str(combined), np.hstack(tiles[:-1]))
    viewer = subprocess.Popen(['eog', str(combined)], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, start_new_session=True)
    if os.environ.get('WINDOWID'): # keyboard back to this terminal
        time.sleep(1.5)
        subprocess.run(['xdotool', 'windowactivate', os.environ['WINDOWID']],
                       capture_output=True)
    print(f"  finds shown in {combined.name}: {', '.join(frames)}")
    try:
        answer = ask('  any FALSE find? type its colour(s), comma separated, '
                       'or Enter if all are real > ').strip()
    except EOFError:
        answer = ''
    false = [c.strip() for c in answer.split(',') if c.strip() in frames]
    if false:
        set_last_field(csv_path, 'false_found', ','.join(false))
        print(f"  recorded false_found = {','.join(false)}")
    try:
        viewer.terminate()
    except Exception:
        pass


def set_last_notes(path, notes):
    """Fill in the notes of the row just appended."""
    with open(path, newline='') as f:
        header = next(csv.reader(f))
    rows = read_rows(path)
    rows[-1]['notes'] = notes
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, restval='')
        w.writeheader()
        w.writerows(rows)


def read_rows(path):
    try:
        return list(csv.DictReader(open(path)))
    except OSError:
        return []


def append_row(path, fields, row):
    """Append one row under the file's OWN header.

    Writing with the code's current field list under an older header
    shifted every later column (the first shakedown file had rows of 49,
    53 and 54 columns under a 49-column header). If the code has gained
    fields since the file was created, the file is first rewritten under
    the old header plus the new fields, so old and new rows stay aligned."""
    header = None
    if path.exists():
        with open(path, newline='') as f:
            header = next(csv.reader(f), None)
    if not header:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerow(row)
        return
    union = header + [k for k in fields if k not in header]
    if union != header:
        old = read_rows(path)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=union, restval='')
            w.writeheader()
            w.writerows(old)
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=union, restval='',
                       extrasaction='ignore').writerow(row)


# ── robot link ──────────────────────────────────────────────────────────────
class RobotLink:
    """SSH to the robot. The password travels in SSHPASS, never argv."""

    def __init__(self, env):
        if not env.get('ROBOT_IP') or not env.get('ROBOT_USER'):
            raise SystemExit('robot.env needs ROBOT_IP and ROBOT_USER '
                             '(copy robot.env.example to robot.env)')
        self.ip = env['ROBOT_IP']
        self.target = f"{env['ROBOT_USER']}@{env['ROBOT_IP']}"
        self.password = env.get('ROBOT_PASS', '')

    def _argv(self, cmd):
        # -n + every background command redirecting stdin from /dev/null:
        # a remote `nohup ... &` that still holds the channel's stdin keeps
        # the SSH session open until it exits, i.e. forever for bringup
        opts = ['-n', '-o', 'StrictHostKeyChecking=no',
                '-o', 'ConnectTimeout=5']
        if self.password:
            return ['sshpass', '-e', 'ssh', *opts, self.target, cmd]
        return ['ssh', '-o', 'BatchMode=yes', *opts, self.target, cmd]

    def run(self, cmd, timeout=20):
        """(returncode, stdout). A robot that is off or unreachable gives
        (124, '') instead of raising: teardown must always finish."""
        env = dict(os.environ, SSHPASS=self.password)
        try:
            res = subprocess.run(self._argv(cmd), capture_output=True,
                                 stdin=subprocess.DEVNULL, text=True,
                                 timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return 124, ''
        return res.returncode, res.stdout

    def kill_all(self):
        """Kill every ROS process on the robot and confirm none survive."""
        self.run(ROBOT_KILL)
        deadline = time.time() + 12
        while time.time() < deadline:
            time.sleep(1.5)
            _, out = self.run(ROBOT_ALIVE)
            if not out.strip():
                return
        raise HarnessError(f'robot ROS processes survived the kill:\n{out}')

    def start_bringup(self):
        self.run(ROBOT_BRINGUP)

    def start_camera(self, extra_params=''):
        """BRIO driver on the robot, as run.sh's ensure_camera starts it.
        extra_params pass through to the node (e.g. exposure / white
        balance locks: list them with `ros2 param list /v4l2_camera`)."""
        self.run(detached(
            'source /opt/ros/humble/setup.bash && '
            'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && '
            'exec ros2 run v4l2_camera v4l2_camera_node --ros-args '
            '-p video_device:=/dev/video0 -p image_size:=[640,480] '
            f'{extra_params}', '/tmp/tbot_camera.log'))

    def clock_offset_ms(self):
        """Robot wall clock minus PC wall clock.

        The robot reads its clock at the very end of the SSH call, after
        connection setup (up to ~1 s, and most of the call), so the call's
        midpoint is the wrong reference: it reported 664 ms for a ~30 ms
        offset. Compare against the moment the call returns instead; what
        remains is the return latency (~1 ms on the tether)."""
        rc, out = self.run('date +%s.%N')
        t_after = time.time()
        if rc != 0:
            return None
        return round(offset_from_return(float(out.strip()), t_after) * 1000, 1)


# ── ROS side ────────────────────────────────────────────────────────────────
class RealHarness:
    """Owned rclpy context on the real robot's domain, plus process control
    for everything launched on the PC. Mirrors SimHarness's interface where
    the two overlap (popen, spin, wait_for_node, subscribe)."""

    def __init__(self, domain=0):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from tf2_ros import Buffer, TransformListener

        env = os.environ.copy()
        env['PYTHONPATH'] = ':'.join(
            p for p in env.get('PYTHONPATH', '').split(':')
            if p and not p.rstrip('/').endswith('unit_tests'))
        env.update({'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
                    'CYCLONEDDS_URI': 'file:///home/turtlebot/cyclone_config.xml',
                    'TURTLEBOT3_MODEL': 'burger', 'LDS_MODEL': 'LDS-02',
                    'ROS_DOMAIN_ID': str(domain)})
        self.env = env
        self.procs = []
        self._rclpy = rclpy
        self._ctx = Context()
        rclpy.init(context=self._ctx, domain_id=domain)
        self.node = rclpy.create_node(f'real_trial_{os.getpid()}',
                                      context=self._ctx)
        self._exec = SingleThreadedExecutor(context=self._ctx)
        self._exec.add_node(self.node)
        self.tf = Buffer()
        self._tf_listener = TransformListener(self.tf, self.node,
                                              spin_thread=False)

    # processes
    def popen(self, cmd, log_path):
        full = (f'source /opt/ros/humble/setup.bash && '
                f'source {WS_SETUP} && {cmd}')
        proc = subprocess.Popen(
            full, shell=True, executable='/bin/bash',
            stdin=subprocess.DEVNULL, stdout=open(log_path, 'w'),
            stderr=subprocess.STDOUT, env=self.env, start_new_session=True)
        self.procs.append(proc)
        return proc

    @staticmethod
    def stop(proc, grace_s=0.5):
        """SIGINT the process group (nodes zero cmd_vel, bags close), then
        SIGKILL whatever is left after grace_s."""
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    def kill_local(self):
        for proc in self.procs:
            self.stop(proc, grace_s=0.2)
        self.procs.clear()
        for pat in LOCAL_KILL_PATTERNS:
            subprocess.run(['pkill', '-9', '-f', pat], capture_output=True)
        time.sleep(1.0)

    # graph
    def spin(self, seconds=0.1):
        deadline = time.monotonic() + seconds
        while True:
            self._exec.spin_once(timeout_sec=0.02)
            if time.monotonic() >= deadline:
                return

    def nodes(self):
        self.spin(0.3)
        return [n for n, _ in self.node.get_node_names_and_namespaces()]

    def wait_for_node(self, name, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if name in self.nodes():
                return True
        return False

    def subscribe(self, topic, msg_type, callback, depth=50):
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(depth=depth)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        return self.node.create_subscription(msg_type, topic, callback, qos)

    def unsubscribe(self, sub):
        self.node.destroy_subscription(sub)

    def wait_for_message(self, topic, msg_type, timeout_s):
        box = []
        sub = self.subscribe(topic, msg_type, box.append, depth=5)
        try:
            deadline = time.monotonic() + timeout_s
            while not box and time.monotonic() < deadline:
                self.spin(0.1)
        finally:
            self.unsubscribe(sub)
        return box[-1] if box else None

    def publish_goal(self, x, y, frame='odom', n=3):
        from geometry_msgs.msg import PoseStamped
        msg = PoseStamped()
        msg.header.frame_id = frame
        msg.pose.position.x, msg.pose.position.y = float(x), float(y)
        msg.pose.orientation.w = 1.0
        self._publish_matched('/goal_pose', msg, n)

    def zero_wheels(self):
        """Zeros on /cmd_vel once the robot's subscription is matched —
        a publish before discovery completes is silently dropped."""
        from geometry_msgs.msg import Twist
        self._publish_matched('/cmd_vel', Twist(), 10)

    def _publish_matched(self, topic, msg, n):
        pub = self.node.create_publisher(type(msg), topic, 10)
        try:
            deadline = time.monotonic() + 5.0
            while (pub.get_subscription_count() == 0
                   and time.monotonic() < deadline):
                self.spin(0.1)
            for _ in range(n):
                pub.publish(msg)
                self.spin(0.1)
        finally:
            self.node.destroy_publisher(pub)

    def robot_still(self, timeout_s=10.0):
        """True once /odom reports near-zero velocity."""
        from nav_msgs.msg import Odometry
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            m = self.wait_for_message('/odom', Odometry, 2.0)
            if m is None:
                continue
            if (abs(m.twist.twist.linear.x) < STOP_V_TOL
                    and abs(m.twist.twist.angular.z) < STOP_W_TOL):
                return True
        return False

    def lookup(self, parent, child):
        """Latest (x, y, yaw, stamp_ns) of child in parent, or None. The
        buffer keeps old transforms for seconds, so callers compare stamps
        to tell a live pose from a stale one."""
        from rclpy.time import Time
        from tf2_ros import TransformException
        try:
            tf = self.tf.lookup_transform(parent, child, Time())
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        stamp = tf.header.stamp.sec * 1_000_000_000 + tf.header.stamp.nanosec
        return (t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w), stamp)

    def close(self):
        self.kill_local()
        self.node.destroy_node()
        self._rclpy.shutdown(context=self._ctx)


class KeyReader:
    """Single keypresses from the terminal without Enter (cbreak mode)."""

    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def poll(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None

    def __exit__(self, *exc):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)


# ── one trial ───────────────────────────────────────────────────────────────
def reset_robot(h, robot, min_battery_v, say):
    """Full reset: both machines cleared, fresh bringup, odom verified zero.
    Returns the battery voltage."""
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import BatteryState, LaserScan
    h.kill_local()
    say('killing ROS on the robot...')
    robot.kill_all()
    say('starting bringup (odometry zeroes here)...')
    robot.start_bringup()
    if not h.wait_for_node('turtlebot3_node', BRINGUP_TIMEOUT_S):
        raise HarnessError('bringup did not come up')
    if h.wait_for_message('/scan', LaserScan, SENSOR_WAIT_S) is None:
        raise HarnessError('no /scan after bringup')
    odom = h.wait_for_message('/odom', Odometry, SENSOR_WAIT_S)
    if odom is None:
        raise HarnessError('no /odom after bringup')
    p, q = odom.pose.pose.position, odom.pose.pose.orientation
    yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
    if (math.hypot(p.x, p.y) > ODOM_ZERO_TOL_M
            or abs(yaw) > ODOM_ZERO_TOL_RAD):
        raise HarnessError(f'odometry not zeroed after bringup '
                           f'({p.x:.3f}, {p.y:.3f}, {yaw:.3f})')
    batt = h.wait_for_message('/battery_state', BatteryState, SENSOR_WAIT_S)
    volts = round(batt.voltage, 2) if batt is not None else None
    if volts is not None and volts < min_battery_v:
        raise BatteryLow(f'battery {volts} V < {min_battery_v} V')
    return volts


def start_controller(h, course, cell, trial_dir, deadman_s):
    model, hz = cell['model'], float(cell['hz'])
    gx, gy = course['goal']
    if model == 'dwa':
        proc = h.popen('ros2 run dwa_avoidance dwa_node --ros-args '
                       f'-p control_period:={1.0 / hz}',
                       trial_dir / 'controller.log')
        if not h.wait_for_node('dwa_node', NODE_TIMEOUT_S):
            raise HarnessError('dwa_node did not start')
        h.publish_goal(gx, gy)
        return proc
    args = (f'--ros-args -p use_sim_time:=false '
            f'-p checkpoint:={CKPTS[model]} -p behavior:=goto '
            f'-p control_rate_hz:={hz} -p denoiser:=ddpm '
            f'-p safety_gate:=false -p threads:={POLICY_THREADS} '
            f'-p sensor_deadman_s:={deadman_s} '
            f'-p spawn_x:=0.0 -p spawn_y:=0.0 -p spawn_yaw:=0.0 '
            f'-p goal_x:={gx} -p goal_y:={gy} '
            f'-p latency_log:={trial_dir / "lat.csv"}')
    if model == 'two_stage':
        args += f' -p grid_file:={course["path"]}'
    proc = h.popen(f'{VENV_PY} {CAPSTONE}/deploy/policy_node.py {args}',
                   trial_dir / 'controller.log')
    if not h.wait_for_node('diffusion_policy', NODE_TIMEOUT_S):
        raise HarnessError('policy node did not start')
    return proc


def stop_everything_moving(h, robot, controller, say):
    """Controller down, wheels zeroed, stop verified on /odom; restarts the
    motor controller as a last resort. Returns whether the stop verified."""
    h.stop(controller, grace_s=0.5)
    h.zero_wheels()
    if h.robot_still():
        return True
    say('robot still reports motion — restarting bringup to reset motors')
    robot.run(detached("pkill -9 -f 'turtlebot[3]'", '/dev/null'))
    time.sleep(2)
    robot.start_bringup()
    h.wait_for_node('turtlebot3_node', BRINGUP_TIMEOUT_S)
    h.zero_wheels()
    if h.robot_still():
        return True
    print('\n!!! ROBOT MAY STILL BE MOVING — USE ITS POWER SWITCH !!!\n')
    return False


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


class GotoTask:
    """Point-to-point benchmark trial on a course."""

    bag_topics = BAG_TOPICS
    key_help = 'c collision · a abort · x harness fault · space stop'

    def __init__(self, course, cell, opts):
        _, length, _ = C.optimal_route(course)
        self.course, self.cell, self.opts = course, cell, opts
        self.monitor = TrialMonitor(course['goal'], C.timeout_s(length))

    def start_sensors(self, h, robot, trial_dir, say):
        pass

    def start_task_nodes(self, h, trial_dir, say):
        pass

    def after_slam(self, h, trial_dir, say):
        pass

    def subscriptions(self, h):
        return []

    def start_controller(self, h, trial_dir):
        return start_controller(h, self.course, self.cell, trial_dir,
                                self.opts.deadman_s)

    def row(self, trial_dir, extra):
        if self.monitor.verdict == 'collision':
            extra['collided_with'] = ask(
                '  which obstacle? tube id (e.g. t07), "wall", or Enter to '
                'skip > ').strip()
        log = trial_dir / 'controller.log'
        return {'model': self.cell['model'], 'world_idx': self.course['id'],
                'hz': float(self.cell['hz']),
                'trial': int(self.cell['round']),
                **self.monitor.summary(), **extra,
                **latency_stats(trial_dir / 'lat.csv'),
                'collision_source': ('operator' if self.monitor.verdict
                                     == 'collision' else ''),
                'deadman_lines': count_lines(log, 'sensor deadman'),
                'recovery_lines': count_lines(log, 'gate recovery')}


class ColourTask:
    """Colour scavenger run in an arena: camera, detector and checkpoint
    manager as run.sh's real colour test starts them, with the policy in
    seek (two-stage) or wander (single-stage) mode and the gate on."""

    bag_topics = BAG_TOPICS + COLOUR_BAG_TOPICS
    key_help = ('c log a collision (run continues) · w wedged/unsafe: end '
                'as a failure · a abort · x harness fault · space stop')

    def __init__(self, arena, placement, cell, opts):
        self.arena, self.placement = arena, placement
        self.cell, self.opts = cell, opts
        self.colors = A.placement_colors(arena, placement)
        self.light = A.lighting_config(arena, opts.lighting, placement)
        self.monitor = ColourMonitor(self.colors, opts.cap_min * 60.0)
        self.start = None

    def after_slam(self, h, trial_dir, say):
        """Locate the robot on the arena map before it moves, so this
        run's path can be drawn in the arena frame (real_paths.py)."""
        from sensor_msgs.msg import LaserScan
        from evaluation import real_paths as P
        scans = [h.wait_for_message('/scan', LaserScan, SENSOR_WAIT_S)
                 for _ in range(LOCALIZE_SCANS)]
        pts = P.scan_points([sc for sc in scans if sc is not None])
        if len(pts) == 0:
            say('warning: no scan to locate the start on the arena map')
            return
        self.start = P.localize(self.arena, pts)
        st = self.start
        say(f"start located at ({st['x']:.2f}, {st['y']:.2f}), heading "
            f"{math.degrees(st['yaw']):.0f} deg: {st['match'] * 100:.0f}% of "
            f"the scan on the map (best rival {st['rival'] * 100:.0f}%)")
        if st['match'] < START_MATCH_WARN:
            say('warning: weak start match; this run\'s arena-frame path '
                'may be misplaced')
        off_m = math.hypot(st['x'], st['y'])
        off_deg = abs(math.degrees(st['yaw']))
        if off_m > START_OFF_M or off_deg > START_OFF_DEG:
            msg = (f'robot is {off_m * 100:.0f} cm / {off_deg:.0f} deg off the '
                   'start mark')
            if not self.opts.shakedown: # both models of a round must share a start
                raise HarnessError(f'{msg}: re-place it on the mark and re-run')
            say(f'warning: {msg} (practice run, continuing)')

    def start_sensors(self, h, robot, trial_dir, say):
        from sensor_msgs.msg import CompressedImage
        say('starting the BRIO camera')
        robot.start_camera(self.camera_args())
        if h.wait_for_message('/image_raw/compressed', CompressedImage,
                              CAMERA_WAIT_S) is None:
            raise HarnessError('camera did not start')
        h.popen('ros2 run dwa_camera_avoidance camera_obstacle_node '
                '--ros-args --params-file '
                f'{WS}/dwa_camera_avoidance/config/camera_obstacle_params.yaml',
                trial_dir / 'camera_obstacle.log')

    def camera_args(self):
        """The arena's locked camera controls as -p overrides; anything on
        the command line (--camera-params) is appended and wins."""
        locked = ' '.join(f'-p {k}:={str(v).lower() if isinstance(v, bool) else v}'
                          for k, v in self.light['camera_params'].items())
        return f'{locked} {self.opts.camera_params}'.strip()

    def start_task_nodes(self, h, trial_dir, say):
        cs = ','.join(self.colors)
        say(f'starting the detector and checkpoint manager ({cs})')
        show = ' -p publish_debug:=true' if self.opts.show else ''
        h.popen(f'{VENV_PY} {CAPSTONE}/deploy/color_seeker.py --ros-args '
                f"-p profile:={self.light['profile']} "
                f'-p colors:="\'{cs}\'"{show} '
                '-p image_topic:=/image_raw/compressed -p compressed:=true '
                '-p camera_info_topic:=/camera_info '
                f"-p square_size_m:={self.arena['target_height_m']}",
                trial_dir / 'color_seeker.log')
        h.popen(f'{VENV_PY} {CAPSTONE}/deploy/color_test_manager.py '
                f'--ros-args -p results_csv:={trial_dir / "color_results.csv"} '
                f'-p colors:="\'{cs}\'"', trial_dir / 'color_test.log')
        for node in ('color_seeker', 'color_test_manager'):
            if not h.wait_for_node(node, NODE_TIMEOUT_S):
                raise HarnessError(f'{node} did not start')
        if self.opts.show:
            open_viewers(h.env)

    def subscriptions(self, h):
        from std_msgs.msg import String

        def on_status(msg):
            try:
                self.monitor.on_status(time.monotonic(), json.loads(msg.data))
            except ValueError:
                pass
        return [h.subscribe('/color_test/status', String, on_status)]

    def _save_path(self, trial_dir, summary):
        """Arena-frame path, events and figure. Never allowed to cost
        the result row: any failure is reported and the run still logs."""
        from evaluation import real_paths as P
        try:
            meta = {'model': self.cell['model'],
                    'round': int(self.cell['round']),
                    'placement': self.placement,
                    'lighting': self.opts.lighting, **{
                        k: summary[k] for k in ('verdict', 'total_s', 'path_m',
                                                'found_count')}}
            if P.save_run(trial_dir, self.start, self.monitor, meta):
                P.render_run(self.arena, self.placement, trial_dir)
        except Exception as e: # a figure must never lose a run
            print(f'  warning: path figure failed: {e}')

    def start_controller(self, h, trial_dir):
        model = self.cell['model']
        args = (f'--ros-args -p use_sim_time:=false '
                f'-p checkpoint:={CKPTS[model]} '
                f'-p behavior:={COLOUR_BEHAVIOR[model]} '
                f'-p control_rate_hz:={COLOUR_HZ} -p denoiser:=ddpm '
                f'-p safety_gate:=true -p threads:={POLICY_THREADS} '
                f'-p path_scale:=1.0 '
                f'-p seek_min_goal_dist:={SEEK_MIN_GOAL_DIST} '
                f'-p sensor_deadman_s:={self.opts.deadman_s} '
                f'-p spawn_x:=0.0 -p spawn_y:=0.0 -p spawn_yaw:=0.0 '
                f'-p latency_log:={trial_dir / "lat.csv"}')
        proc = h.popen(f'{VENV_PY} {CAPSTONE}/deploy/policy_node.py {args}',
                       trial_dir / 'controller.log')
        if not h.wait_for_node('diffusion_policy', NODE_TIMEOUT_S):
            raise HarnessError('policy node did not start')
        return proc

    def row(self, trial_dir, extra):
        log = trial_dir / 'controller.log'
        summary = self.monitor.summary()
        start = {}
        if self.start is not None:
            start = {'start_x': round(self.start['x'], 3),
                     'start_y': round(self.start['y'], 3),
                     'start_yaw_deg': round(math.degrees(self.start['yaw']), 1),
                     'start_match': round(self.start['match'], 3)}
            self._save_path(trial_dir, summary)
        return {**start, 'lighting': self.opts.lighting,
                'model': self.cell['model'], 'arena': self.arena['name'],
                'world_idx': self.arena['id'], 'placement': self.placement,
                'hz': COLOUR_HZ, 'trial': int(self.cell['round']),
                'cap_s': self.opts.cap_min * 60.0,
                **summary, **extra,
                **latency_stats(trial_dir / 'lat.csv'),
                'deadman_lines': count_lines(log, 'sensor deadman'),
                'recovery_lines': count_lines(log, 'gate recovery'),
                'gate_lines': count_lines(log, 'safety gate:')}


_VIEWERS = []
VIEWER_W, VIEWER_H = 1000, 800
VIEWER_SLOTS = {'/image_raw/compressed': 0,  # camera: left slot
                '/color_debug': 1}            # detector: right slot


def _viewer_origin():
    """Top-left of the largest monitor (xrandr), where the viewers go;
    the runner's terminal belongs on another screen so they never cover
    it. (0, 0) when xrandr is unavailable."""
    try:
        out = subprocess.run(['xrandr', '--listmonitors'], capture_output=True,
                             text=True).stdout
    except FileNotFoundError:
        return 0, 0
    mons = [tuple(map(int, m)) for m in re.findall(
        r'(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)', out)]
    if not mons:
        return 0, 0
    w, h, x, y = max(mons, key=lambda m: m[0] * m[1])
    return x, y


def open_viewers(env):
    """Camera and detector windows, opened once per session. Outside the
    harness's process list on purpose: every run kills and restarts the
    ROS stack, and the windows reconnect instead of flashing on and off."""
    if any(v.poll() is None for v in _VIEWERS):
        return
    view_env = dict(env, LIBGL_ALWAYS_SOFTWARE='1')
    for topic in VIEWER_SLOTS:
        _VIEWERS.append(subprocess.Popen(
            ['bash', '-c', 'source /opt/ros/humble/setup.bash && exec ros2 '
             f'run rqt_image_view rqt_image_view {topic}'],
            env=view_env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True))
    import threading
    threading.Thread(target=_place_viewers, daemon=True).start()


def _place_viewers(timeout_s=30.0):
    """Put the viewers side by side: both open at the same default spot,
    so the second hides the first. Matches each window to its topic via
    the owning process's command line (xdotool + xprop; skipped quietly
    when they are not installed)."""
    placed, deadline = set(), time.monotonic() + timeout_s
    while len(placed) < len(VIEWER_SLOTS) and time.monotonic() < deadline:
        time.sleep(1.0)
        try:
            wins = subprocess.run(
                ['xdotool', 'search', '--name', 'rqt_image_view__ImageView'],
                capture_output=True, text=True).stdout.split()
        except FileNotFoundError:
            return
        for w in wins:
            pid = subprocess.run(['xprop', '-id', w, '_NET_WM_PID'],
                                 capture_output=True, text=True).stdout.split()
            try:
                cmd = Path(f'/proc/{pid[-1]}/cmdline').read_bytes().decode()
            except (OSError, IndexError):
                continue
            ox, oy = _viewer_origin()
            for topic, slot in VIEWER_SLOTS.items():
                if topic in cmd and topic not in placed:
                    subprocess.run(['xdotool', 'windowsize', w,
                                    str(VIEWER_W), str(VIEWER_H)])
                    subprocess.run(['xdotool', 'windowmove', w,
                                    str(ox + 40 + slot * (VIEWER_W + 20)),
                                    str(oy + 60)])
                    placed.add(topic)
    # give the keyboard back to the runner's terminal: operator keys only
    # reach it while it has focus (xterm exports its own WINDOWID)
    if os.environ.get('WINDOWID'):
        subprocess.run(['xdotool', 'windowactivate', os.environ['WINDOWID']],
                       capture_output=True)


def run_task(h, robot, task, trial_dir, opts):
    """One run from full reset to verified stop; returns the task's row
    (without session/bookkeeping fields)."""
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import BatteryState, LaserScan

    trial_dir.mkdir(parents=True, exist_ok=True)
    t_ref = time.monotonic()
    events = []
    mon = task.monitor

    def say(msg):
        events.append((time.monotonic(), msg))
        print(f'  [{time.monotonic() - t_ref:6.1f}s] {msg}')

    extra = {'battery_v_start': '', 'battery_v_end': '',
             'stop_verified': '', 'ping_p95_ms': '', 'collided_with': ''}
    controller = bag = ping = None
    subs = []
    batt = {}
    try:
        extra['battery_v_start'] = reset_robot(h, robot, opts.min_battery_v,
                                               say)
        say(f"battery {extra['battery_v_start']} V")
        task.start_sensors(h, robot, trial_dir, say)
        say('starting SLAM')
        h.popen('ros2 launch dwa_avoidance cartographer_only.launch.py '
                'config_basename:=turtlebot3_no_odom.lua',
                trial_dir / 'slam.log')
        deadline = time.monotonic() + SLAM_READY_TIMEOUT_S
        origin = None
        while origin is None and time.monotonic() < deadline:
            h.spin(0.2)
            origin = h.lookup('map', 'base_footprint')
        if origin is None:
            raise HarnessError('SLAM pose never appeared')
        last_stamp = origin[3]
        origin = origin[:3]
        task.after_slam(h, trial_dir, say)
        task.start_task_nodes(h, trial_dir, say)

        bag = h.popen(f'ros2 bag record -o {trial_dir / "bag"} '
                      + ' '.join(task.bag_topics), trial_dir / 'bag.log')
        deadline = time.monotonic() + BAG_START_CHECK_S
        while time.monotonic() < deadline and bag.poll() is None:
            h.spin(0.1)
        if bag.poll() is not None:
            tail = (trial_dir / 'bag.log').read_text().strip().splitlines()
            raise HarnessError('bag recorder exited at start: '
                               + (tail[-1] if tail else 'no output'))
        ping = subprocess.Popen(['ping', '-n', '-i', '0.2', robot.ip],
                                stdin=subprocess.DEVNULL,
                                stdout=open(trial_dir / 'ping.log', 'w'),
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        subs = task.subscriptions(h) + [
            h.subscribe('/cmd_vel', Twist,
                        lambda m: mon.on_cmd(time.monotonic())),
            h.subscribe('/scan', LaserScan,
                        lambda m: mon.on_scan(time.monotonic())),
            h.subscribe('/odom', Odometry, lambda m: mon.on_odom(
                time.monotonic(), m.pose.pose.position.x,
                m.pose.pose.position.y, yaw_from_quaternion(
                    m.pose.pose.orientation.x, m.pose.pose.orientation.y,
                    m.pose.pose.orientation.z, m.pose.pose.orientation.w))),
            h.subscribe('/battery_state', BatteryState,
                        lambda m: batt.__setitem__('v', m.voltage)),
        ]
        say(f"starting {task.cell['model']} @ {float(task.cell['hz']):g} Hz")
        controller = task.start_controller(h, trial_dir)
        mon.arm(time.monotonic())
        say(f'controller live — running  ({task.key_help})')
        stopped_early = False
        n_coll = 0
        with KeyReader() as keys:
            while mon.check(time.monotonic()) is None:
                h.spin(MONITOR_PERIOD_S)
                now = time.monotonic()
                pose = h.lookup('map', 'base_footprint')
                if pose is not None and pose[3] != last_stamp:
                    last_stamp = pose[3]
                    mon.on_slam(now, *relative_pose(origin, pose[:3]))
                if controller.poll() is not None:
                    mon.on_controller_exit(now)
                key = keys.poll()
                if key == ' ':
                    h.stop(controller, grace_s=0.3)
                    h.zero_wheels()
                    stopped_early = True
                    t_stop = now
                    break
                if key:
                    mon.on_key(now, key)
                if len(getattr(mon, 'collision_t', ())) > n_coll:
                    n_coll = len(mon.collision_t)
                    say(f'collision #{n_coll} logged')
        if stopped_early:
            choices = '/'.join(f'{k} {v}' for k, v in mon.KEYS.items())
            choice = ''
            while choice not in mon.KEYS:
                choice = ask(f'  stopped. classify: {choices} > '
                               ).strip().lower()[:1]
            mon.on_key(t_stop, choice)
        say(f'verdict: {mon.verdict} ({mon.reason})')
    except HarnessError as e:
        mon.finish('harness_error', time.monotonic(), str(e))
        say(f'harness error: {e}')
    except KeyboardInterrupt:
        # recorded, not lost: the row says what happened and the session
        # loop stops after writing it
        mon.finish('aborted', time.monotonic(), INTERRUPTED)
        say('interrupted (Ctrl-C) — stopping the robot')
    finally:
        # each step guarded on its own: one failure (a robot switched off
        # mid-teardown) must not skip closing the bag or writing the traces
        def step(fn):
            try:
                fn()
            except Exception as e: # report and carry on tearing down
                say(f'teardown step failed: {e}')
        for sub in subs:
            step(lambda sub=sub: h.unsubscribe(sub))
        if controller is not None:
            step(lambda: extra.__setitem__('stop_verified', stop_everything_moving(
                h, robot, controller, say)))
        if batt.get('v') is not None:
            extra['battery_v_end'] = round(batt['v'], 2)
        if ping is not None:
            def close_ping():
                h.stop(ping, grace_s=0.5)
                rtts = ping_rtts((trial_dir / 'ping.log').read_text())
                if rtts:
                    extra['ping_p95_ms'] = round(percentile(rtts, 95), 1)
            step(close_ping)
        step(lambda: h.stop(bag, grace_s=BAG_CLOSE_GRACE_S))
        step(h.kill_local)
        step(lambda: write_traces(trial_dir, mon, t_ref, events))
    return task.row(trial_dir, extra)


def run_trial(h, robot, course, cell, trial_dir, opts):
    """One point-to-point trial (see run_task)."""
    return run_task(h, robot, GotoTask(course, cell, opts), trial_dir, opts)


# ── commands ────────────────────────────────────────────────────────────────
def run_dir(run_id, shakedown):
    return RESULTS / (f'{run_id}_shakedown' if shakedown else run_id)


def load_schedule(course, run_id):
    path = run_dir(run_id, False) / f'schedule_{course["name"]}.csv'
    if not path.exists():
        raise SystemExit(f'no schedule at {path} — run the "schedule" '
                         'command first, before the session')
    rows = read_rows(path)
    for r in rows:
        r['hz'], r['round'] = float(r['hz']), int(r['round'])
        r['world_idx'], r['order_idx'] = int(r['world_idx']), int(r['order_idx'])
    return rows


def shakedown_cells(course, per_model=2, hz=4.0):
    """Rehearsal list (never counted): each model twice at 4 Hz."""
    return [{'order_idx': -1, 'course': course['name'],
             'world_idx': course['id'], 'model': m, 'hz': hz, 'round': k}
            for k in range(per_model) for m in MODELS]


def cmd_schedule(args):
    course = C.load(args.course)
    out = run_dir(args.run_id, False) / f'schedule_{course["name"]}.csv'
    if out.exists() and not args.force:
        raise SystemExit(f'{out} exists — the order is fixed once written '
                         '(--force only if no trial has run yet)')
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = make_schedule(course, args.run_id, rounds=args.rounds)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} cells -> {out}')


def cmd_status(args):
    course = C.load(args.course)
    schedule = load_schedule(course, args.run_id)
    done = [r for r in read_rows(run_dir(args.run_id, args.shakedown)
                                 / 'trials_w0.csv')
            if r['course'] == course['name']]
    pending = pending_cells(schedule, done)
    counted = [r for r in done if r['verdict'] in COUNTED]
    print(f"course {course['name']}: {len(counted)} counted, "
          f"{len(done) - len(counted)} re-run attempts, "
          f"{len(pending)}/{len(schedule)} pending")
    for m in MODELS:
        v = [r['verdict'] for r in counted if r['model'] == m]
        if v:
            print(f"  {m:13s} {v.count('success')}/{len(v)} success, "
                  f"{v.count('collision')} collision, "
                  f"{v.count('timeout')} timeout, {v.count('crash')} crash")


def preflight(args, layout_paths):
    """Refuse to start when the robot link, the code or the layout files
    are not what counted runs require."""
    up = subprocess.run(['ip', 'link', 'show', 'eth0'], capture_output=True,
                        text=True).stdout
    if 'state UP' not in up:
        raise SystemExit('eth0 is down: the CycloneDDS config cannot reach '
                         'the robot. Bring it up (or restart WSL) first.')
    if subprocess.run(['pgrep', '-f', 'run\\.s[h]'],
                      capture_output=True).returncode == 0:
        raise SystemExit('a run.sh session is running — stop it first; two '
                         'stacks would fight over /cmd_vel')
    for m, p in CKPTS.items():
        if not p.exists():
            raise SystemExit(f'missing checkpoint for {m}: {p}')
    sha, dirty = git_state()
    for path in layout_paths:
        tracked = subprocess.run(
            ['git', '-C', str(WS), 'ls-files', '--error-unmatch', str(path)],
            capture_output=True).returncode == 0
        if not tracked:
            dirty = (dirty + '\n' if dirty else '') + \
                f'?? {path} (layout file not committed)'
    if dirty and not (args.shakedown or args.allow_dirty):
        raise SystemExit('uncommitted changes in the code under test — '
                         'counted runs need the frozen commit:\n' + dirty)
    return sha, bool(dirty)


def deadman_for_gap(worst_gap_s):
    """Deadman setting for the worst scan/odom gap seen, or None when even
    the raised value would hold the wheels (fix the link first)."""
    if worst_gap_s <= DEADMAN_RAISE_GAP_S:
        return 0.6
    if worst_gap_s <= 0.9 * RAISED_DEADMAN_S:
        return RAISED_DEADMAN_S
    return None


def cmd_check(args):
    """Read-only readiness report; writes results/real/day0/check_*.txt."""
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import BatteryState, Imu, LaserScan
    lines, verdicts = [], []

    def report(ok, text):
        mark = {True: 'ok  ', False: 'FAIL', None: 'note'}[ok]
        lines.append(f'  {mark}  {text}')
        print(lines[-1])
        if ok is False:
            verdicts.append(text)

    print('PC')
    up = 'state UP' in subprocess.run(['ip', 'link', 'show', 'eth0'],
                                      capture_output=True, text=True).stdout
    report(up, 'eth0 up (CycloneDDS is pinned to it)')
    running = subprocess.run(['pgrep', '-f', 'run\\.s[h]'],
                             capture_output=True).returncode == 0
    report(not running, 'no run.sh session competing for the robot')
    for m, path in CKPTS.items():
        report(path.exists(), f'{m} checkpoint {path.parent.name}')
    sha, dirty = git_state()
    report(None, f'code at {sha}' + (' (UNCOMMITTED changes — fine for '
                                     'Day 0, not for counted trials)'
                                     if dirty else ' (clean)'))

    print('robot link')
    robot = RobotLink(read_robot_env())
    rtts = ping_rtts(subprocess.run(
        ['ping', '-n', '-c', '10', '-i', '0.2', robot.ip],
        capture_output=True, text=True).stdout)
    report(bool(rtts), f'ping {robot.ip}: ' + (
        f'p50 {percentile(rtts, 50):.1f} ms, p95 {percentile(rtts, 95):.1f} ms'
        if rtts else 'no reply'))
    if not rtts:
        return _write_check(lines, verdicts)
    rc, _ = robot.run('true')
    report(rc == 0, 'SSH login')
    offset = robot.clock_offset_ms()
    # Robot-stamped data is restamped on the PC (scan/tf relays) or timed on
    # arrival (deadman, bags, detector), so an offset only matters once it
    # is large; WSL2's clock alone is often ~0.2 s off NTP.
    report(None if offset is not None and abs(offset) < CLOCK_OFFSET_FAIL_MS
           and abs(offset) >= CLOCK_OFFSET_OK_MS
           else (offset is not None and abs(offset) < CLOCK_OFFSET_OK_MS),
           f'robot clock offset {offset} ms (ok < {CLOCK_OFFSET_OK_MS:.0f}, '
           f'fails at {CLOCK_OFFSET_FAIL_MS:.0f}; run `chronyc tracking` '
           'to see whether the PC or the robot is off)')

    print('ROS')
    h = RealHarness()
    try:
        alive = h.wait_for_node('turtlebot3_node', 5)
        if not alive and not args.no_bringup:
            print('  ...  bringup not running — starting it')
            robot.start_bringup()
            alive = h.wait_for_node('turtlebot3_node', BRINGUP_TIMEOUT_S)
        report(alive, 'turtlebot3_node (bringup) running')
        if not alive:
            return _write_check(lines, verdicts)
        topics = [('/scan', LaserScan), ('/odom', Odometry), ('/imu', Imu),
                  ('/battery_state', BatteryState)]
        if args.camera:
            from sensor_msgs.msg import CompressedImage
            print('  ...  starting the BRIO (colour runs load the Pi with it)')
            robot.start_camera(args.camera_params)
            cam = h.wait_for_message('/image_raw/compressed',
                                     CompressedImage, CAMERA_WAIT_S)
            report(cam is not None, 'BRIO camera publishing')
            topics.append(('/image_raw/compressed', CompressedImage))
        rx = {t: [] for t, _ in topics}
        last = {}
        subs = [h.subscribe(t, ty, (lambda t: lambda m: (
                    rx[t].append(time.monotonic()), last.__setitem__(t, m)))(t))
                for t, ty in topics]
        print(f'  ...  listening {args.seconds:.0f} s — to measure while '
              'driving, teleop the robot now in another terminal')
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            h.spin(0.2)
        for s in subs:
            h.unsubscribe(s)
        worst = 0.0
        for topic, stamps in rx.items():
            if len(stamps) < 2:
                report(False, f'{topic}: {len(stamps)} messages')
                continue
            gaps = [b - a for a, b in zip(stamps, stamps[1:])]
            rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            text = (f'{topic}: {rate:.1f} Hz, gap p99 '
                    f'{percentile(gaps, 99) * 1000:.0f} ms, max '
                    f'{max(gaps) * 1000:.0f} ms')
            report(True, text)
            if topic in ('/scan', '/odom'):
                worst = max(worst, max(gaps))
        batt = last.get('/battery_state')
        if batt is not None:
            report(batt.voltage >= DEFAULT_MIN_BATTERY_V,
                   f'battery {batt.voltage:.2f} V (swap threshold '
                   f'{DEFAULT_MIN_BATTERY_V} V)')
        odom = last.get('/odom')
        if odom is not None:
            p = odom.pose.pose.position
            report(None, f'odom pose ({p.x:.3f}, {p.y:.3f}) — only zero '
                   'right after a bringup restart, which every trial does')
        deadman = deadman_for_gap(worst)
        if deadman is None:
            report(False, f'worst scan/odom gap {worst * 1000:.0f} ms — too '
                   'long for any deadman setting; check the link/robot load')
        else:
            report(None, f'worst scan/odom gap {worst * 1000:.0f} ms -> use '
                   f'--deadman-s {deadman} (measure idle AND driving; the '
                   'larger answer wins, then keep it for every counted '
                   'trial)')
    finally:
        h.close()
    return _write_check(lines, verdicts)


def _write_check(lines, failures):
    out = RESULTS / 'day0'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"check_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt"
    path.write_text('\n'.join(lines) + '\n')
    print(f"\n{'READY' if not failures else f'{len(failures)} FAILED'} "
          f'-> {path}')
    return 1 if failures else 0


def cmd_map_arena(args):
    """Guided arena mapping: fresh bringup on the start mark (so the map
    frame is the start frame), Cartographer while the operator teleops,
    then save the map and extract the poles (course_tools.arena_from_map).
    The map is resampled into the robot's frame at SLAM start (offset
    AND rotation), so arena coordinates are measured from the start mark
    along the robot's heading."""
    import yaml
    from evaluation import course_tools as T
    up = 'state UP' in subprocess.run(['ip', 'link', 'show', 'eth0'],
                                      capture_output=True, text=True).stdout
    if not up:
        raise SystemExit('eth0 is down — bring it up first')
    robot = RobotLink(read_robot_env())
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CompressedImage, LaserScan
    raw = RESULTS / 'arenas' / f'{args.name}_raw'
    raw.parent.mkdir(parents=True, exist_ok=True)
    h = RealHarness()
    try:
        if not args.ready:
            ask('Robot on the start mark, facing +x (into the arena)? '
                  '[Enter] ')
        volts = reset_robot(h, robot, 0.0, print)
        if args.camera:
            print('starting the BRIO (gaps are measured under its load)')
            robot.start_camera(args.camera_params)
            if h.wait_for_message('/image_raw/compressed', CompressedImage,
                                  CAMERA_WAIT_S) is None:
                print('warning: camera did not start')
        print(f'battery {volts} V; starting SLAM')
        h.popen('ros2 launch dwa_avoidance cartographer_only.launch.py '
                'config_basename:=turtlebot3_no_odom.lua',
                raw.parent / f'{args.name}_slam.log')
        deadline = time.monotonic() + SLAM_READY_TIMEOUT_S
        origin = None
        while origin is None and time.monotonic() < deadline:
            h.spin(0.2)
            origin = h.lookup('map', 'base_footprint')
        if origin is None:
            raise SystemExit('SLAM pose never appeared — see '
                             f'{raw.parent / (args.name + "_slam.log")}')
        rx = {'/scan': [], '/odom': []}
        subs = [h.subscribe(t, ty, (lambda t: lambda m: rx[t].append(
                    time.monotonic()))(t))
                for t, ty in (('/scan', LaserScan), ('/odom', Odometry))]
        print('\nSLAM running. In another terminal:\n'
              '    ros2 run turtlebot3_teleop teleop_keyboard\n'
              'Drive slowly past every pole from more than one side and '
              'along the whole boundary.\nWatch the map in RViz if you like '
              '(rviz2, fixed frame "map", add /map).', flush=True)
        if args.done_file:
            done = Path(args.done_file)
            print(f'waiting for {done} to appear...', flush=True)
            while not done.exists():
                h.spin(0.5)
            done.unlink()
        else:
            import threading
            finished = threading.Event()
            threading.Thread(target=lambda: (ask(
                '[Enter] when the map looks complete > '), finished.set()),
                daemon=True).start()
            while not finished.is_set():
                h.spin(0.5)
        for sub in subs:
            h.unsubscribe(sub)
        h.zero_wheels()
        worst = 0.0
        for topic, stamps in rx.items():
            if len(stamps) > 1:
                gaps = [b - a for a, b in zip(stamps, stamps[1:])]
                worst = max(worst, max(gaps))
                print(f'{topic} while driving: '
                      f'{(len(stamps) - 1) / (stamps[-1] - stamps[0]):.1f} Hz,'
                      f' gap p99 {percentile(gaps, 99) * 1000:.0f} ms, max '
                      f'{max(gaps) * 1000:.0f} ms')
        rec = deadman_for_gap(worst)
        print(f'worst gap while driving {worst * 1000:.0f} ms -> '
              + (f'--deadman-s {rec}' if rec else 'too long for any deadman '
                 'setting; check the link'), flush=True)
        saver = h.popen(f'ros2 run nav2_map_server map_saver_cli -f {raw} '
                        '--ros-args -p save_map_timeout:=10.0',
                        raw.parent / f'{args.name}_saver.log')
        saver.wait(timeout=40)
        Path(f'{raw}_start_pose.yaml').write_text(yaml.safe_dump(
            {'x': origin[0], 'y': origin[1], 'yaw': origin[2],
             'note': 'robot pose in the raw map when SLAM started'}))
        print(f'SLAM start pose in the raw map: ({origin[0]:.3f}, '
              f'{origin[1]:.3f}, {origin[2]:.3f} rad) — resampling the map '
              'into the start frame')
        start_map = A.map_to_start_frame(f'{raw}.yaml', origin,
                                         Path(f'{raw}_start.yaml'))
    finally:
        h.close()
    out, poles = T.arena_from_map(start_map, args.name, args.id,
                                  args.pole_diameter)
    png = T.render_arena(A.load(out), 'P1',
                         T.ARENA_RENDER_DIR / f'{args.name}.png')
    print(f'\n{len(poles)} pole candidates -> {out}\nfigure: {png}\n'
          'next: delete any line that is not a course pole, tape-check a '
          'few poles against the figure, then add the squares under '
          'placements and run course_tools.py arena-measure')


def run_session(args, *, label, layout_paths, out, csv_path, fields,
                todo, make_task, prompt, result_line, after_row, review=None):
    """Operator loop shared by both tasks: preflight, session log, then one
    run per Enter until nothing is owed or the operator ends the session.

    todo(done_rows) -> cells still owed, in order; make_task(cell) -> a
    GotoTask or ColourTask. Rows are appended as each run finishes, so a
    session can end (or crash) at any point without losing results."""
    sha, dirty = preflight(args, layout_paths)
    out.mkdir(parents=True, exist_ok=True)
    robot = RobotLink(read_robot_env())
    session_id = datetime.now().strftime('%Y-%m-%d_%H%M')
    session = {'session_id': session_id, 'run_id': args.run_id,
               'course': label, 'lighting': getattr(args, 'lighting', ''),
               'shakedown': args.shakedown,
               'start': datetime.now().isoformat(timespec='seconds'),
               'git_sha': sha, 'git_dirty': dirty, 'link': args.link,
               'clock_offset_ms': robot.clock_offset_ms()}
    rtts = ping_rtts(subprocess.run(
        ['ping', '-n', '-c', '10', '-i', '0.2', robot.ip],
        capture_output=True, text=True).stdout)
    if rtts:
        session['ping_p50_ms'] = round(percentile(rtts, 50), 1)
        session['ping_p95_ms'] = round(percentile(rtts, 95), 1)
    print(f"session {session_id}: {label}, commit {sha}"
          f"{' (dirty)' if dirty else ''}, clock offset "
          f"{session['clock_offset_ms']} ms, ping p95 "
          f"{session.get('ping_p95_ms')} ms")

    h = RealHarness()
    runs = 0
    try:
        while True:
            pending = todo(read_rows(csv_path))
            if not pending:
                print('nothing left to run here.')
                break
            cell = pending[0]
            print(f"\nnext: #{cell['order_idx']} {cell['model']} @ "
                  f"{float(cell['hz']):g} Hz, round {cell['round']}  "
                  f"({len(pending)} left)")
            try:
                answer = ask(f'  {prompt} [Enter] start · type quit to end '
                               'the session > ').strip().lower()
            except EOFError:
                break
            if answer == 'quit': # a whole word: a stray key cannot end it
                break
            if answer:
                print(f'  ({answer!r} ignored: Enter starts the run, '
                      '"quit" ends the session)')
                continue
            tag = (f"t{next_trial_index(out / 'trials'):03d}_"
                   f"{label.replace('/', '_')}_{cell['model']}_"
                   f"{float(cell['hz']):g}hz_r{cell['round']}")
            t_wall = datetime.now().isoformat(timespec='seconds')
            try:
                row = run_task(h, robot, make_task(cell),
                               out / 'trials' / tag, args)
            except BatteryLow as e:
                print(f'  {e} — swap the battery, then continue')
                continue
            if session.get('battery_v_start') is None:
                session['battery_v_start'] = row.get('battery_v_start')
            row.update({'run_id': args.run_id, 'session_id': session_id,
                        'order_idx': cell['order_idx'], 't_start_wall': t_wall,
                        'ckpt': (CKPTS[cell['model']].parent.name
                                 if cell['model'] in CKPTS else 'dwa'),
                        'git_sha': sha, 'git_dirty': dirty})
            # saved BEFORE the notes prompt: a session that dies there (the
            # robot switched off, the window closed) must not lose the run
            append_row(csv_path, fields, row)
            runs += 1
            if review is not None:
                review(csv_path, row, out / 'trials' / tag)
            try:
                late = (ask('  contacts you could not key in during the run '
                              '(number, Enter for none) > ').strip()
                        if 'collisions' in row else '')
                notes = ask('  notes (Enter to skip) > ').strip()
            except EOFError:
                late, notes = '', ''
            if late.isdigit() and int(late) > 0:
                add_late_collisions(csv_path, int(late))
            if notes:
                set_last_notes(csv_path, notes)
            print(f'  -> {result_line(row)}  (logged to {csv_path.name})')
            after_row(row)
            if row.get('verdict_reason') == INTERRUPTED:
                print('session ended by Ctrl-C')
                break
    except KeyboardInterrupt:
        print('\ninterrupted — stopping the robot')
        h.zero_wheels()
    finally:
        h.close()
        session.update({'end': datetime.now().isoformat(timespec='seconds'),
                        'trials_run': runs})
        append_row(out / 'sessions.csv', SESSION_FIELDS, session)


def cmd_run(args):
    course = C.load(args.course)
    schedule = (shakedown_cells(course) if args.shakedown
                else load_schedule(course, args.run_id))

    def todo(rows):
        done = [r for r in rows if r['course'] == course['name']]
        return (schedule[len(done):] if args.shakedown
                else pending_cells(schedule, done))

    def after(row):
        if row['verdict'] == 'collision':
            print('  put any moved tube back on its mark')

    run_session(
        args, label=course['name'], layout_paths=[course['path']],
        out=run_dir(args.run_id, args.shakedown),
        csv_path=run_dir(args.run_id, args.shakedown) / 'trials_w0.csv',
        fields=ROW_FIELDS, todo=todo,
        make_task=lambda cell: GotoTask(course, cell, args),
        prompt='robot on the start mark, tubes on their marks?',
        result_line=lambda r: (f"{r['verdict']}  {r['time_to_goal_s']} s  "
                               f"path {r['gt_path_len']} m"),
        after_row=after)


# ── colour task commands ────────────────────────────────────────────────────
COLOUR_COUNTED = ('success', 'timeout', 'stuck', 'crash')


def make_colour_schedule(arena, placement, run_id, rounds=COLOUR_ROUNDS,
                         lighting='day'):
    """Each round runs both models once, in a seeded random order. One
    schedule per lighting condition, so a round's two runs always share the
    same light and the comparison stays paired."""
    rng = random.Random(f'{run_id}/{arena["name"]}/{placement}/{lighting}')
    rows = []
    for rnd in range(rounds):
        models = list(COLOUR_MODELS)
        rng.shuffle(models)
        for m in models:
            rows.append({'order_idx': len(rows), 'arena': arena['name'],
                         'world_idx': arena['id'], 'placement': placement,
                         'lighting': lighting, 'model': m, 'hz': COLOUR_HZ,
                         'round': rnd})
    return rows


def colour_key(row):
    return (row['model'], row['placement'], row.get('lighting', ''),
            int(row['round'] if 'round' in row else row['trial']))


def colour_pending(schedule, done_rows):
    """Owed runs in the pre-declared schedule order: never attempted, or
    last attempt aborted / harness fault. A re-run keeps its place, so a
    round's two runs stay together and the seeded order is what runs."""
    last = {colour_key(r): r['verdict'] for r in done_rows}
    return [c for c in schedule
            if colour_key(c) not in last or last[colour_key(c)] in RERUN]


def colour_schedule_path(arena, placement, run_id, lighting):
    return (run_dir(run_id, False)
            / f"schedule_colour_{arena['name']}_{placement}_{lighting}.csv")


def load_colour_schedule(arena, placement, run_id, lighting):
    path = colour_schedule_path(arena, placement, run_id, lighting)
    if not path.exists():
        raise SystemExit(f'no schedule at {path} — run colour-schedule first')
    rows = read_rows(path)
    for r in rows:
        r['hz'], r['round'] = float(r['hz']), int(r['round'])
        r['order_idx'] = int(r['order_idx'])
    return rows


def cmd_colour_schedule(args):
    arena = A.load(args.arena)
    colors = A.placement_colors(arena, args.placement)
    if not colors:
        raise SystemExit(f'placement {args.placement} has no squares yet')
    A.lighting_config(arena, args.lighting, args.placement)
    out = colour_schedule_path(arena, args.placement, args.run_id,
                               args.lighting)
    if out.exists() and not args.force:
        raise SystemExit(f'{out} exists — the order is fixed once written')
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = make_colour_schedule(arena, args.placement, args.run_id,
                                args.rounds, args.lighting)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs ({', '.join(colors)}) -> {out}")


def colour_summary(rows):
    """Per-model summary of counted colour runs."""
    out = []
    for m in COLOUR_MODELS:
        g = [r for r in rows if r['model'] == m
             and r['verdict'] in COLOUR_COUNTED]
        if not g:
            continue
        wins = [float(r['total_s']) for r in g if r['verdict'] == 'success']
        out.append({
            'model': m, 'runs': len(g), 'success': len(wins),
            'success_%': round(100 * len(wins) / len(g), 1),
            'median_total_s': (round(percentile(wins, 50), 1)
                               if wins else None),
            'mean_found': round(sum(int(r['found_count']) for r in g)
                                / len(g), 2),
            'collisions_per_run': round(sum(int(r['collisions'] or 0)
                                            for r in g) / len(g), 2),
            'median_path_m': round(percentile(
                [float(r['path_m']) for r in g], 50), 1),
        })
    return out


def cmd_colour_status(args):
    arena = A.load(args.arena)
    rows = [r for r in read_rows(run_dir(args.run_id, args.shakedown)
                                 / 'colour_trials.csv')
            if r['arena'] == arena['name']
            and r['placement'] == args.placement
            and r.get('lighting', '') == args.lighting]
    schedule = load_colour_schedule(arena, args.placement, args.run_id,
                                    args.lighting)
    pending = colour_pending(schedule, rows)
    counted = [r for r in rows if r['verdict'] in COLOUR_COUNTED]
    print(f"arena {arena['name']} {args.placement} ({args.lighting}): "
          f"{len(counted)} counted, "
          f"{len(rows) - len(counted)} re-run attempts, "
          f"{len(pending)}/{len(schedule)} pending")
    for srow in colour_summary(rows):
        print('  ' + '  '.join(f'{k} {v}' for k, v in srow.items()))


def practice_cells(arena, placement, lighting, models=None, repeat=1):
    """Uncounted shakedown runs: each chosen model `repeat` times, in the
    order given. The list belongs to one session, so practice can be
    repeated as often as wanted without touching counted results."""
    models = [m.strip() for m in (models or ','.join(COLOUR_MODELS)).split(',')
              if m.strip()]
    unknown = [m for m in models if m not in COLOUR_MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown} "
                         f"(have {', '.join(COLOUR_MODELS)})")
    return [{'order_idx': -1, 'arena': arena['name'], 'world_idx': arena['id'],
             'placement': placement, 'lighting': lighting, 'model': m,
             'hz': COLOUR_HZ, 'round': k}
            for k in range(repeat) for m in models]


def cmd_colour_run(args):
    arena = A.load(args.arena)
    colors = A.placement_colors(arena, args.placement)
    if not colors:
        raise SystemExit(f'placement {args.placement} has no squares yet')
    A.lighting_config(arena, args.lighting, args.placement)
    out = run_dir(args.run_id, args.shakedown)
    csv_path = out / 'colour_trials.csv'
    practice = [0] # runs made in THIS session (practice lists are per session)
    if args.shakedown:
        schedule = practice_cells(arena, args.placement, args.lighting,
                                  args.models, args.repeat)
    else:
        schedule = load_colour_schedule(arena, args.placement, args.run_id,
                                        args.lighting)

    def todo(rows):
        if args.shakedown:
            return schedule[practice[0]:]
        done = [r for r in rows if r['arena'] == arena['name']
                and r['placement'] == args.placement
                and r.get('lighting', '') == args.lighting]
        return colour_pending(schedule, done)

    def after(row):
        practice[0] += 1
        if int(row['collisions'] or 0):
            print('  put any moved pole back on its mark')

    layout_files = [arena['path']] + ([arena['map'], arena['map'].with_suffix(
        '.pgm')] if arena['map'] else [])
    run_session(
        args, label=f"{arena['name']}/{args.placement}/{args.lighting}",
        layout_paths=layout_files, out=out, csv_path=csv_path,
        fields=COLOUR_ROW_FIELDS, todo=todo,
        make_task=lambda cell: ColourTask(arena, args.placement, cell, args),
        review=review_finds,
        prompt=f"robot on the start mark, squares ({', '.join(colors)}) "
               f'in place, {args.lighting} lighting?',
        result_line=lambda r: (f"{r['verdict']}  {r['found_count']}/"
                               f"{len(colors)} found  {r['total_s']} s  "
                               f"path {r['path_m']} m  "
                               f"{r['collisions']} collisions"),
        after_row=after)


def main():
    _remember_tty()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--seconds', type=float, default=60.0)
    c.add_argument('--no-bringup', action='store_true',
                   help='report instead of starting bringup if it is down')
    c.add_argument('--camera', action='store_true',
                   help='start the BRIO first, so gaps are measured under '
                        'the colour runs\' load')
    c.add_argument('--camera-params', default='')
    for name in ('schedule', 'run', 'status'):
        s = sub.add_parser(name)
        s.add_argument('--course', required=True, help='layout yaml')
        s.add_argument('--run-id', required=True)
        if name != 'schedule':
            s.add_argument('--shakedown', action='store_true',
                           help='uncounted rehearsal trials, separate folder')
    sub.choices['schedule'].add_argument('--rounds', type=int, default=ROUNDS)
    sub.choices['schedule'].add_argument('--force', action='store_true')
    r = sub.choices['run']
    r.add_argument('--min-battery-v', type=float,
                   default=DEFAULT_MIN_BATTERY_V)
    r.add_argument('--deadman-s', type=float, default=0.6,
                   help='policy sensor-age deadman (decide on Day 0)')
    r.add_argument('--link', default='unknown',
                   help='robot network link for the session log '
                        '(e.g. cable, wifi)')
    r.add_argument('--allow-dirty', action='store_true',
                   help='permit counted trials on uncommitted code')
    m = sub.add_parser('map-arena')
    m.add_argument('--name', required=True)
    m.add_argument('--id', type=int, required=True, help='>= 9100')
    m.add_argument('--pole-diameter', type=float, default=0.04445)
    m.add_argument('--ready', action='store_true',
                   help='robot is already on the start mark; skip the prompt')
    m.add_argument('--done-file',
                   help='finish when this file appears instead of on Enter')
    m.add_argument('--camera', action='store_true',
                   help='run the BRIO while mapping (realistic load, and '
                        'lets you watch the feed)')
    m.add_argument('--camera-params', default='')
    for name in ('colour-schedule', 'colour-run', 'colour-status'):
        s = sub.add_parser(name)
        s.add_argument('--arena', required=True, help='arena yaml')
        s.add_argument('--placement', required=True)
        s.add_argument('--run-id', required=True)
        s.add_argument('--lighting', required=True,
                       help="lighting profile in the arena file (e.g. day, "
                            "night); picks the colour bands and camera lock")
        if name != 'colour-schedule':
            s.add_argument('--shakedown', action='store_true',
                           help='one uncounted run per model, own folder')
    cs = sub.choices['colour-schedule']
    cs.add_argument('--rounds', type=int, default=COLOUR_ROUNDS)
    cs.add_argument('--force', action='store_true')
    cr = sub.choices['colour-run']
    cr.add_argument('--cap-min', type=float, default=DEFAULT_CAP_MIN,
                    help='time cap per run [min]')
    cr.add_argument('--models', default=None,
                    help='with --shakedown: which models to practise, comma '
                         'separated (default: both)')
    cr.add_argument('--repeat', type=int, default=1,
                    help='with --shakedown: runs per model')
    cr.add_argument('--show', action='store_true',
                    help='open the camera and detector views (boxes, labels '
                         'and a live mask per colour) for the session')
    cr.add_argument('--camera-params', default='',
                    help='extra -p settings for v4l2_camera_node, e.g. '
                         'exposure / white-balance locks')
    cr.add_argument('--min-battery-v', type=float,
                    default=DEFAULT_MIN_BATTERY_V)
    cr.add_argument('--deadman-s', type=float, default=0.6,
                    help='policy sensor-age deadman (decide on Day 0)')
    cr.add_argument('--link', default='unknown')
    cr.add_argument('--allow-dirty', action='store_true')
    args = ap.parse_args()
    if args.cmd == 'check':
        sys.exit(cmd_check(args))
    if args.cmd == 'map-arena':
        return cmd_map_arena(args)
    {'schedule': cmd_schedule, 'run': cmd_run, 'status': cmd_status,
     'colour-schedule': cmd_colour_schedule, 'colour-run': cmd_colour_run,
     'colour-status': cmd_colour_status}[args.cmd](args)


if __name__ == '__main__':
    main()
