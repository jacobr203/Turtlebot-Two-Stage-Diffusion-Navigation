#!/usr/bin/env python3
"""Closed-loop deployment of a trained diffusion checkpoint: a control timer
samples waypoints (two-stage checkpoints plan A* subgoals on grid_file) and a
20 Hz pure-pursuit tracker follows them. `latency_log` records their latency.

Run with the capstone venv python (see launch_policy.sh):
  .venv/bin/python deploy/policy_node.py --ros-args -p use_sim_time:=true \
      -p checkpoint:=diffusion/runs/two_stage_v2/best.pt \
      -p grid_file:=barn/grids/grid_000.npy -p control_rate_hz:=4.0 \
      -p spawn_x:=-2.25 -p spawn_y:=3.0 -p spawn_yaw:=1.5708 \
      -p goal_x:=-2.25 -p goal_y:=13.0
"""

import json
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMsg
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scan import resample_scan # noqa: E402
from diffusion.ddpm import DDPM # noqa: E402
from diffusion.checkpoint import load_policy # noqa: E402
from deploy.subgoal_planner import LiveSubgoalPlanner, SubgoalPlanner # noqa: E402
from deploy.policy import (GeometryMixin, ViewpointMixin, ExploreMixin, # noqa: E402
                           SeekMixin, SafetyMixin, TrackingMixin) # noqa: E402


class DiffusionPolicyNode(SeekMixin, ExploreMixin, ViewpointMixin,
                          SafetyMixin, TrackingMixin, GeometryMixin,
                          Node):
    def __init__(self):
        super().__init__('diffusion_policy')
        for name, default in [
                ('checkpoint', ''), ('grid_file', ''),
                ('behavior', 'goto'), # goto | wander (map completion)
                ('control_rate_hz', 4.0), ('denoiser', 'ddpm'),
                # a smaller torch pool trades a little mean latency for a much
                # shorter tail while contending with gzserver + Cartographer
                ('ddim_steps', 10), ('threads', 4), ('sample_k', 4),
                ('guide_scale', 0.4), # clearance guidance (0 = off)
                ('spawn_x', -2.25), ('spawn_y', 3.0), ('spawn_yaw', 1.5708),
                ('goal_x', -2.25), ('goal_y', 13.0), ('goal_tol', 0.30),
                ('subgoal_dist', 1.2), ('latency_log', ''),
                # sensor-age deadman: hold the wheels when scan or odom is
                # older than this; 0.6 s = 3 missed LDS frames at 5 Hz
                ('sensor_deadman_s', 0.6),
                ('tracker_lookahead', 0.12),
                # EXPERIMENTAL: stretch the denormalized waypoint path by this
                # factor (run.sh --path_scale). Out of the training
                # distribution: keep 1.0 for evaluation.
                ('path_scale', 1.0),
                # minimum distance of seek exploration goals while moving
                # (run.sh --min_goal_dist); from standstill the bound is 2.5 m
                ('seek_min_goal_dist', 0.5),
                # wander goals stay above goal_clip (3 m): a nearby goal at
                # zero velocity reads as "arrived" and the robot freezes
                ('wander_goal_dist', 3.5), ('wander_interval', 20.0),
                ('wander_min_goal_dist', 2.0), # [m] reject closer cells
                ('wander_goal_clearance', 0.30), # [m] margin from obstacles
                # classical scan-based gate between policy and wheels; the
                # evaluation harness disables it to measure the bare policies.
                # stop_dist still permits BARN's narrowest 0.30 m gaps.
                ('safety_gate', True),
                ('safety_stop_dist', 0.16), # [m] hard block below this
                ('safety_slow_band', 0.25), # [m] linear slowdown band
                ('safety_sector_deg', 50.0), # total forward cone width
                # gate recovery: braking alone wedges the robot against walls,
                # so after gate_stuck_s blocked, turn until the cone clears
                ('gate_stuck_s', 2.5),
                ('gate_recover_w', 0.8), # [rad/s] recovery spin
                ('gate_resume_clear', 0.30), # [m] cone min to resume
                ('gate_recover_timeout_s', 4.0),
                # recovery backs up before rotating (rotating against a surface
                # wedges the chassis). The stall trigger catches what the lidar
                # cannot see, e.g. couch overhangs.
                ('gate_backup_speed', 0.08), # [m/s] reverse speed
                ('gate_backup_s', 1.8), # [s] ~0.14 m of backup
                ('gate_rear_clear', 0.25), # [m] needed to back up
                ('gate_stall_s', 1.5), # cmd>0 vs odom~0 trigger
                # no-progress watchdog, for when the policy itself goes timid
                # and command-based triggers stay silent. The radius also
                # catches ~0.2-0.3 m dithering; cruise covers ~2 m per window.
                ('progress_radius', 0.35), # [m]
                ('progress_window_s', 8.0),
                ('progress_backup_mult', 2.5), # generous backup (~0.36 m)
                # mandatory post-backup turn: without it the rotate phase exits
                # once the cone reads clear and the robot drives straight back
                ('gate_min_turn_deg', 45.0)]:
            # dynamic typing: '-p goal_y:=13' arrives as INTEGER and a strict
            # DOUBLE declaration rejects it — accept both, python math copes
            self.declare_parameter(name, default,
                                   ParameterDescriptor(dynamic_typing=True))

        def p(name):
            v = self.get_parameter(name).value
            return float(v) if type(v) is int else v

        torch.set_num_threads(int(p('threads')))
        self.policy, ckpt = load_policy(p('checkpoint'))
        self.cfg = ckpt['config']
        self.stats = ckpt['stats']
        self.ddpm = DDPM(T=self.cfg['T'])
        self.conditioning = self.cfg['conditioning']
        self.get_logger().info(
            f"loaded {p('checkpoint')} ({self.conditioning}, "
            f"val ADE {ckpt.get('val_ade', -1)*100:.2f} cm) "
            f"@ {p('control_rate_hz')} Hz, denoiser {p('denoiser')}")

        self.anchor = (p('spawn_x'), p('spawn_y'), p('spawn_yaw'))
        self.goal = (p('goal_x'), p('goal_y'))
        self.goal_tol = p('goal_tol')
        self.denoiser = p('denoiser')
        self.ddim_steps = int(p('ddim_steps'))
        self.behavior = p('behavior')
        self.wander_goal_dist = p('wander_goal_dist')
        self.wander_interval = p('wander_interval')
        self._wander_next = -1.0
        self._wander_meta = None # active wander-goal commitment leash
        self._visited_cells = set() # 0.5 m map cells stood in (novelty bias)
        if self.behavior == 'wander':
            # wander explores unknown space with no grid for subgoals, so it
            # needs a goal-conditioned checkpoint
            if self.conditioning != 'goal':
                raise ValueError('behavior=wander needs a goal-conditioned '
                                 '(single_stage) checkpoint, got '
                                 f'{self.conditioning}')
            self.goal = None # picked from the first scan
        if self.behavior == 'seek':
            # plans on the live SLAM map: frontier exploration until a square
            # is seen, then an approach pose
            if self.conditioning not in ('subgoal', 'subgoal_goal'):
                raise ValueError('behavior=seek needs a subgoal-conditioned '
                                 '(two_stage) checkpoint, got '
                                 f'{self.conditioning}')
            self.goal = None # exploration picks it; detections retarget it
        self.subgoaler = None
        self.live_subgoaler = None
        if self.behavior == 'seek':
            self.live_subgoaler = LiveSubgoalPlanner(
                subgoal_dist=p('subgoal_dist'), prefer_inflate=0.30)
        elif self.conditioning in ('subgoal', 'subgoal_goal'):
            if not p('grid_file'):
                raise ValueError('two-stage checkpoint requires grid_file')
            self.subgoaler = SubgoalPlanner(p('grid_file'), self.goal,
                                            subgoal_dist=p('subgoal_dist'))

        self._scan = None
        self._odom = None
        self._map = None
        self._ranges_m = None
        self._near_goal_logged = False
        self._gate = p('safety_gate')
        self._gate_stop = p('safety_stop_dist')
        self._gate_band = p('safety_slow_band')
        if self.behavior == 'seek':
            # wider resting margins: at 0.16 m the hull corners can clip a
            # pillar on the next turn. _tight_ok relaxes them while driving.
            self._gate_stop = max(self._gate_stop, 0.21)
            self._gate_band = max(self._gate_band, 0.30)
        self._gate_half = int(round(p('safety_sector_deg') / 2.0))
        self._gate_stuck_s = p('gate_stuck_s')
        self._gate_recover_w = p('gate_recover_w')
        self._gate_resume = p('gate_resume_clear')
        self._gate_recover_timeout = p('gate_recover_timeout_s')
        self._gate_backup_v = p('gate_backup_speed')
        self._gate_backup_s = p('gate_backup_s')
        self._gate_rear_clear = p('gate_rear_clear')
        self._gate_stall_s = p('gate_stall_s')
        self._gate_blocked_since = None
        self._stall_since = None
        self._stall_recov_pose = None # wedge-escalation displacement ref
        self._stall_streak = 0
        self._last_cmd_v = 0.0
        self._scan_hist = deque(maxlen=40) # (wall_t, front ranges) ~2 s
        # near-field memory for the gate: obstacles closer than the lidar's
        # range_min return invalid, so a short bin stays short for ~1.2 s
        self._gate_win = deque(maxlen=8) # (wall_t, per-bin valid mins)
        self._gate_ranges = None
        # camera virtual scan: obstacles off the lidar plane. Gate-only, as the
        # policy must stay in its pure-lidar training distribution; inf fill
        # keeps unseen bins neutral in the min-fusion with lidar.
        self._cam_win = deque(maxlen=8)
        self._cam_gate = None
        self._cam_gate_t = 0.0
        self._cam_seen = False
        self._pose_hist = deque(maxlen=280) # (wall_t, x, y) ~14 s @ 20 Hz
        self._progress_radius = p('progress_radius')
        self._progress_window = p('progress_window_s')
        self._progress_mult = p('progress_backup_mult')
        self._recover_phase = None      # None | 'backup' | 'rotate'
        self._phase_until = 0.0
        self._recover_dir = 1.0
        self._backup_s_active = self._gate_backup_s
        self._min_turn_rad = math.radians(p('gate_min_turn_deg'))
        self._rotate_start_yaw = None
        if self._gate:
            self.get_logger().info(
                f'safety gate ON (stop {self._gate_stop} m, '
                f'band {self._gate_band} m, ±{self._gate_half}°, '
                f'recovery after {self._gate_stuck_s} s)')
        self._frames = deque(maxlen=self.cfg['obs_stack'])
        self._tracker = None
        self._done = False
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self,
                                              spin_thread=False)

        self._deadman_s = float(p('sensor_deadman_s'))
        self._lat_f = open(p('latency_log'), 'w') if p('latency_log') else None
        if self._lat_f:
            self._lat_f.write('sim_t,plan_ms,sample_ms,turn_pm,clear,'
                              'scan_age_ms,odom_age_ms\n')

        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(LaserScan, '/scan', self._on_scan, sensor_qos)
        self.create_subscription(LaserScan, '/scan_camera',
                                 self._on_cam_scan, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        if self.behavior in ('wander', 'seek'):
            self.create_subscription(OccupancyGrid, '/map', self._on_map, 1)
        if self.behavior == 'seek':
            self.create_subscription(String, '/color_test/status',
                                     self._on_color_status, 10)
            self._color_status = None
            self._seek_face_since = None
            self._seek_attempts = 0 # failed facing holds on current target
            self._seek_offset = 0.0 # approach-bearing offset fan [rad]
            self._seek_color = None # target the offsets belong to
            self._seek_repositioning = False # drive to the fanned pose
            self._seek_phantom = set() # (color, stamp) estimates given up
            self._coll_seen = None # manager collision count last seen
            self._settle_until = 0.0 # post-collision SLAM settle window
            self._creep_v = -0.05 # wedge-escape creep velocity
            # pinch escape: tight margins can admit the robot to a pocket whose
            # mouth the transit gate then refuses, so the gate relaxes until
            # the robot is clear of the pinch
            self._escape_until = 0.0
            self._escape_from = None
            # regions (2 m map bins) blacklisted after a pinch abandon
            self._region_block = {}
            self._trip_deadline = 0.0 # face-trip commitment leash
            self._trip_best_d = float('inf')
            self._trip_progress_t = 0.0
            self._ring_key = None # routable-ring scan cache
            self._ring_scan_t = 0.0
            self._face_born = {} # face key -> first-seen wall time
            # camera coverage of obstacle faces (lidar-occupied cell edges next
            # to free space). Each view scores frontality x proximity; _face_q
            # keeps the best and faces below SEEK_QMIN attract the explorer.
            # Keys are 0.1 m map positions + a direction, surviving map growth.
            self._face_q = {}
            self._face_gaveup = set()
            # once inspected, a face stays inspected (+/-0.25 m match), so map
            # re-optimization cannot undo walls the camera verified
            self._done_bins = set()
            # abandoned, uninspected faces cool down rather than being retired
            self._face_cool = {} # key -> wall_t expiry
            self._explore_churn = 0 # consecutive stalled/broken trips
            # seek arbiter: one computed mode per tick; guards gate on the
            # modes that have no bounded slowness of their own
            self._seek_mode = 'INIT'
            # per-colour sighting vantages, for retries of queued targets
            self._sight_vantages = {} # color -> (x, y, stamp)
            self._vantage_hist = {} # color -> deque[(x, y, stamp)]
            self._world_hist = deque(maxlen=250) # stagnation watchdog
            self._resight_stamps = set() # (color, stamp) retries consumed
            # recent recovery locations, deferring goals near them; a
            # policy-side signal, so it needs no bumper
            self._recovery_spots = deque(maxlen=20) # (wall_t, wx, wy)
            self._face_target = None # key, map_xy, dir, world, viewpoint
            self._face_view_since = None
            self._cov_pub = self.create_publisher(
                OccupancyGrid, '/seek/coverage', 1)
            self._swatch_pub = self.create_publisher(
                MarkerArray, '/seek/swatches', 1)
            self.create_timer(1.0, self._publish_seek_layers)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._wp_pub = self.create_publisher(PathMsg, '/policy/waypoints', 5)
        self._sub_pub = self.create_publisher(PoseStamped, '/policy/subgoal', 5)

        self.create_timer(1.0 / p('control_rate_hz'), self._control_tick)
        self.create_timer(0.05, self._tracker_tick)

        # wall-clock stall watchdog outside the executor: the rclpy spin can
        # wedge on WSL2, and the real TB3 executes the last cmd_vel forever.
        # Zero the wheels when ticks stall; exit if it persists so the wrapper
        # (run.sh / trial harness) can run its verified stop path.
        self._tick_period = 1.0 / p('control_rate_hz')
        self._last_tick_wall = None # armed after the first completed tick
        threading.Thread(target=self._stall_watchdog, daemon=True).start()


    def _on_scan(self, msg):
        self._scan = msg
        self._scan_rx = time.monotonic() # sensor-age deadman input
        self._ranges_m = resample_scan(msg.ranges, msg.angle_min,
                                       msg.angle_increment)
        # gate ranges: min over valid rays only, remembered ~1.2 s per bin —
        # a sub-range_min contact (invalid return) keeps its last short value
        raw = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(raw)) * msg.angle_increment
        valid = np.isfinite(raw) & (raw >= msg.range_min) & (raw > 0.01)
        bins = np.round(angles / (2 * np.pi) * 360).astype(int) % 360
        cur = np.full(360, np.inf, dtype=np.float32)
        np.minimum.at(cur, bins[valid], raw[valid])
        self._gate_win.append((time.time(), cur))
        stack = np.stack([c for t, c in self._gate_win
                          if time.time() - t <= 1.2])
        mem = stack.min(axis=0)
        mem[~np.isfinite(mem)] = 3.5
        self._gate_ranges = mem


    def _on_cam_scan(self, msg):
        if not self._cam_seen:
            self._cam_seen = True
            self.get_logger().info(
                'camera virtual scan detected — fusing into safety gate')
        raw = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(raw)) * msg.angle_increment
        valid = np.isfinite(raw) & (raw >= msg.range_min) & (raw > 0.01)
        bins = np.round(angles / (2 * np.pi) * 360).astype(int) % 360
        cur = np.full(360, np.inf, dtype=np.float32)
        np.minimum.at(cur, bins[valid], raw[valid])
        self._cam_win.append((time.time(), cur))
        stack = np.stack([c for t, c in self._cam_win
                          if time.time() - t <= 1.2])
        self._cam_gate = stack.min(axis=0)
        self._cam_gate_t = time.time()


    def _on_odom(self, msg):
        self._odom = msg
        self._odom_rx = time.monotonic() # sensor-age deadman input
        # tip-over kill: |roll| or |pitch| beyond ~34° ends the episode with
        # wheels zeroed instead of grinding
        q = msg.pose.pose.orientation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z),
                          1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        if (abs(roll) > 0.6 or abs(pitch) > 0.6) and not self._done:
            self._done = True
            self._tracker = None
            self._cmd_pub.publish(Twist())
            self.get_logger().fatal(
                f'TIP-OVER (roll {math.degrees(roll):.0f}°, pitch '
                f'{math.degrees(pitch):.0f}°) — wheels zeroed, episode over')
        elif (abs(roll) > 0.15 or abs(pitch) > 0.15) and not self._done:
            # a tilt well below the kill threshold still shears the lidar
            # plane and corrupts the map
            self.get_logger().warning(
                f'TILT (roll {math.degrees(roll):.0f}°, pitch '
                f'{math.degrees(pitch):.0f}°) — lidar plane compromised',
                throttle_duration_sec=3.0)


    def _on_map(self, msg):
        self._map = msg
        if self.live_subgoaler is not None:
            self.live_subgoaler.update_map(msg)


    def _on_color_status(self, msg):
        self._color_status = json.loads(msg.data)


def main():
    # short GIL switch interval: the CPU-bound explore worker otherwise starves
    # the 4 Hz control tick
    sys.setswitchinterval(0.002)
    rclpy.init()
    node = DiffusionPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # The real TB3 has NO cmd_vel timeout — a dying node must leave a
        # zero behind or the base keeps executing the last command forever.
        try:
            for _ in range(3):
                node._cmd_pub.publish(Twist())
                time.sleep(0.05)
        except Exception: # noqa: BLE001
            pass
        if node._lat_f:
            node._lat_f.close()
        node.destroy_node()


if __name__ == '__main__':
    main()
