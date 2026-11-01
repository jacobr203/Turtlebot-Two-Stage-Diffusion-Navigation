"""The two trial types and the loop that runs one."""

import json
import math
import subprocess
import time

from common import arena as A
from common import course as C
from common.geometry import yaw_from_quaternion

from .config import (
    BAG_CLOSE_GRACE_S, BAG_START_CHECK_S, BAG_TOPICS, CAMERA_WAIT_S,
    CAPSTONE, CKPTS, COLOUR_BAG_TOPICS, COLOUR_BEHAVIOR, COLOUR_HZ,
    INTERRUPTED, LOCALIZE_SCANS, MONITOR_PERIOD_S, NODE_TIMEOUT_S,
    POLICY_THREADS, SEEK_MIN_GOAL_DIST, SENSOR_WAIT_S,
    SLAM_READY_TIMEOUT_S, START_MATCH_WARN, START_OFF_DEG, START_OFF_M,
    VENV_PY, WS)
from .errors import HarnessError
from .harness import reset_robot, start_controller, stop_everything_moving
from .metrics import (
    count_lines, latency_stats, percentile, ping_rtts, relative_pose)
from .monitors import ColourMonitor, TrialMonitor, write_traces
from .operator_io import KeyReader, ask, open_viewers

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
        run's path can be drawn in the arena frame (paths.py)."""
        from sensor_msgs.msg import LaserScan
        from evaluation.real import paths as P
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
        """The arena's locked camera controls as -p overrides; --camera-params
        is appended and wins."""
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
        """Arena-frame path, events and figure; a failure is only reported,
        so the run still logs."""
        from evaluation.real import paths as P
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
        # recorded: the session loop stops after writing this row
        mon.finish('aborted', time.monotonic(), INTERRUPTED)
        say('interrupted (Ctrl-C) — stopping the robot')
    finally:
        # steps guarded singly: a robot switched off mid-teardown must not
        # skip closing the bag or writing the traces
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
