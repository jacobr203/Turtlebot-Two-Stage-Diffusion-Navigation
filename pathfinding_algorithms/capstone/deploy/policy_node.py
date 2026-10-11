#!/usr/bin/env python3
"""diffusion_policy: closed-loop deployment of a trained checkpoint.

Two timers:
  control @ control_rate_hz (2/3/4)  observe -> (two-stage: A* subgoal) ->
                                     DDPM-sample H waypoints -> hand to tracker
  tracker @ 20 Hz                    pure pursuit along the latest waypoints
                                     -> /cmd_vel (smooth between replans)

The conditioning column comes from the CHECKPOINT ('subgoal' -> two-stage,
needs grid_file; 'goal' -> single-stage). Frames follow the expert
convention: /odom re-anchors at the spawn pose, so world = spawn ∘ odom.

Per-inference latency (subgoal planning ms + sampling ms) is appended to
`latency_log` as CSV — the inference-latency metric for evaluation.

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
from diffusion.evaluate_offline import load_policy # noqa: E402
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
                # threads 8->4: a burst stack dump caught the
                # control thread blocked INSIDE the torch UNet forward
                # (4 s tick) — the long-unattributed t_sample spike tail
                # (max ~1.1 s) is torch inference contending with
                # gzserver + Cartographer for 16 cores. A smaller pool
                # trades a little mean latency for a much shorter tail;
                # K=4 batching already amortizes the work.
                ('ddim_steps', 10), ('threads', 4), ('sample_k', 4),
                ('guide_scale', 0.4), # R2 clearance guidance (0 = off)
                ('spawn_x', -2.25), ('spawn_y', 3.0), ('spawn_yaw', 1.5708),
                ('goal_x', -2.25), ('goal_y', 13.0), ('goal_tol', 0.30),
                ('subgoal_dist', 1.2), ('latency_log', ''),
                # sensor-age deadman: hold the wheels when scan or odom is
                # older than this. 0.6 s = 3 missed LDS frames at 5 Hz,
                # sized for the simulator's sensor gaps; the real-robot
                # value is set from measured LDS-02 timing.
                ('sensor_deadman_s', 0.6),
                ('tracker_lookahead', 0.12),
                # EXPERIMENTAL: stretch the denormalized waypoint path by
                # this factor (run.sh --path_scale). The trained horizon is
                # fixed at H=8; scaling extends the same 8 waypoints
                # spatially — longer, faster strides per tick, wider arcs.
                # Out of the training distribution: keep 1.0 for evaluation.
                ('path_scale', 1.0),
                # minimum distance for seek exploration goals while MOVING
                # (run.sh --min_goal_dist). Sub-1 m goals are supported but
                # sit below the conditioning's comfortable regime — expect
                # creeping approaches; the no-progress watchdog is the net.
                # The standstill bound stays 2.5 m (decisions #19 freeze).
                ('seek_min_goal_dist', 0.5),
                # wander goals are direction carrots, never destinations:
                # keep the distance ABOVE goal_clip (3 m) so the conditioning
                # stays in the saturated "cruise toward a far goal" regime the
                # policy was trained in — a nearby goal at zero velocity reads
                # as "arrived, stay put" and the robot freezes.
                ('wander_goal_dist', 3.5), ('wander_interval', 20.0),
                # map-based goal picking (known-free cells only):
                ('wander_min_goal_dist', 2.0), # [m] reject closer cells
                ('wander_goal_clearance', 0.30), # [m] margin from obstacles
                # classical scan-based safety gate between policy and wheels
                # (the proposal's "collision avoidance layer"). Default ON for
                # manual/real runs; the evaluation harness disables it so the
                # scientific comparison measures the bare policies.
                # stop_dist is tuned to still permit BARN's narrowest 0.30 m
                # gaps (body edge ~4 cm off the cylinders at worst).
                ('safety_gate', True),
                ('safety_stop_dist', 0.16), # [m] hard block below this
                ('safety_slow_band', 0.25), # [m] linear slowdown band
                ('safety_sector_deg', 50.0), # total forward cone width
                # gate recovery: braking alone wedges the robot against walls
                # (the policy sees v=0 and predicts timidly; nothing turns it
                # away). After gate_stuck_s of blocked forward motion, rotate
                # toward the most open bearing until the cone clears.
                ('gate_stuck_s', 2.5),
                ('gate_recover_w', 0.8), # [rad/s] recovery spin
                ('gate_resume_clear', 0.30), # [m] cone min to resume
                ('gate_recover_timeout_s', 4.0),
                # recovery backs up before rotating (rotating while pressed
                # against a surface wedges the chassis), guarded by a rear
                # clearance check. The physical-stall trigger catches
                # obstacles the lidar cannot see (couch overhangs scan as
                # open space): commanded forward but odom not moving.
                ('gate_backup_speed', 0.08), # [m/s] reverse speed
                ('gate_backup_s', 1.8), # [s] ~0.14 m of backup
                ('gate_rear_clear', 0.25), # [m] needed to back up
                ('gate_stall_s', 1.5), # cmd>0 vs odom~0 trigger
                # no-progress watchdog: stuck states where the POLICY itself
                # goes timid (near-zero commands) leave every command-based
                # trigger silent. If the robot hasn't left a small circle in
                # progress_window_s while it still has somewhere to be, back
                # up generously and retry.
                # radius sized for the OSCILLATION trap, not just hard pins:
                # a policy dithering at an obstacle shuffles ~0.2-0.3 m and
                # defeats a small circle; cruise covers ~2 m per window
                ('progress_radius', 0.35), # [m]
                ('progress_window_s', 8.0),
                ('progress_backup_mult', 2.5), # generous backup (~0.36 m)
                # mandatory post-backup turn, RANDOM direction: without it
                # the rotate phase exits the moment the cone reads clear
                # (backing up restores clearance) and the robot drives
                # straight back into the same obstacle
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
            # Wandering explores UNKNOWN space (map completion) — there is no
            # grid to plan subgoals on, so only goal-conditioned checkpoints
            # (single_stage) can drive it.
            if self.conditioning != 'goal':
                raise ValueError('behavior=wander needs a goal-conditioned '
                                 '(single_stage) checkpoint, got '
                                 f'{self.conditioning}')
            self.goal = None # picked from the first scan
        if self.behavior == 'seek':
            # 3-color scavenger seeker (color_test_plan.md §5): the two-stage
            # high-level stage plans on the LIVE Cartographer map — frontier
            # exploration until a square is SEEN, approach pose after.
            # Identical in sim and real; no prior grid exists.
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
            # collision ROOT CAUSE (2026-07-23): the 0.16 m stop lets the
            # robot legally park 2-4 cm (hull corners) from a pillar, and
            # every escape primitive then rotates — the sweep clips. Seek
            # parks farther out instead; goto/wander keep the benchmark
            # margins.
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
        # near-field memory for the GATE (not policy features): obstacles
        # closer than the lidar's range_min return INVALID and would read as
        # "clear" — a table leg vanishes from the scan on contact. A bin
        # that recently read short stays short for ~1.2 s.
        self._gate_win = deque(maxlen=8) # (wall_t, per-bin valid mins)
        self._gate_ranges = None
        # camera virtual scan (/scan_camera, BRIO): obstacles below/above the
        # lidar plane, GATE-ONLY — never fed to the policy, whose conditioning
        # must stay in the pure-lidar training distribution. inf fill so bins
        # the camera can't see stay neutral in the min-fusion with lidar.
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
            # pinch-escape mode (2026-08-03, world 768): tight-approach
            # margins let the robot INTO pockets whose mouth the wider
            # transit gate then refuses — it recovered every 8 s for 7
            # straight minutes without ever leaving. While escaping, the
            # transit gate relaxes to the benchmark margins the robot
            # entered under, until it is clear of the pinch.
            self._escape_until = 0.0
            self._escape_from = None
            # regions (2 m map bins) blacklisted after a pinch abandon —
            # without this the picker just walks the face list deeper into
            # the same pocket (9 abandons in a row, world 768)
            self._region_block = {}
            self._trip_deadline = 0.0 # face-trip commitment leash
            self._trip_best_d = float('inf')
            self._trip_progress_t = 0.0
            self._ring_key = None # routable-ring scan cache
            self._ring_scan_t = 0.0
            self._face_born = {} # face key -> first-seen wall time
                                               # before facing again
            # camera-coverage layer over obstacle faces (secondary map): a
            # face is a lidar-occupied cell edge adjacent to free space.
            # Coverage is GRADED, not binary:
            # each observation scores frontality x proximity credit — a
            # head-on view counts even from afar, an oblique view scores
            # low and leaves the face worth re-checking. _face_q keeps the
            # best score per face; faces below SEEK_QMIN attract the
            # explorer. Keys are map-frame positions quantized at 0.1 m +
            # a direction index, so they survive map growth/shift.
            self._face_q = {}
            self._face_gaveup = set()
            # PERSISTENT inspected set: once a face crosses SEEK_QMIN it
            # stays inspected forever, matched with a generous +/-0.25 m
            # radius — map re-optimizations must never "clear out" walls
            # the camera already verified
            self._done_bins = set()
            # abandoned-not-inspected faces sit out a cooldown instead of
            # being retired (2026-08-03, world 768: 43 stale/breaker
            # abandons marked faces "covered" the camera never saw — the
            # third square's pocket faces among them, never revisited)
            self._face_cool = {} # key -> wall_t expiry
            self._explore_churn = 0 # consecutive stalled/broken trips
            # seek-mode arbiter (2026-08-04, notes/seek_arbiter.md): ONE
            # computed mode per tick; guards gate POSITIVELY on the modes
            # that own no bounded slowness, instead of a growing matrix
            # of per-guard exemptions (every observed loop = global guard
            # x deliberate slow state it didn't know about).
            self._seek_mode = 'INIT'
            # per-COLOR sighting vantages (2026-08-04: a queued target
            # sighted mid-approach had no vantage and lost its retry)
            self._sight_vantages = {} # color -> (x, y, stamp)
            self._vantage_hist = {} # color -> deque[(x, y, stamp)]
            self._world_hist = deque(maxlen=250) # stagnation watchdog
            self._resight_stamps = set() # (color, stamp) retries consumed
            # collision-heat memory: recent recovery locations (any cause —
            # gate block, stall, no-progress). Goals near them get deferred
            # so a pocket that just cost us doesn't get retried while
            # cheaper faces remain. Works sim AND real (policy-side signal,
            # no bumper needed).
            self._recovery_spots = deque(maxlen=20) # (wall_t, wx, wy)
            self._face_target = None      # {'key','world','viewpoint'}
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

        # Wall-clock stall watchdog, OUTSIDE the executor: the rclpy spin has
        # been observed to wedge intermittently on WSL2 (ticks stop, process
        # alive, sim running) — and a wedged node leaves the last cmd_vel
        # latched, which the real TB3 executes forever. Zero the wheels as
        # soon as ticks stall; exit if the stall persists so the wrapper
        # (run.sh / trial harness) can run its verified stop path.
        self._tick_period = 1.0 / p('control_rate_hz')
        self._last_tick_wall = None # armed after the first completed tick
        threading.Thread(target=self._stall_watchdog, daemon=True).start()


    # ── inputs / frames ───────────────────────────────────────────────────────
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
        # tip-over kill (2026-07-23: wedge bouncing levered the robot onto
        # its side; wheels kept driving): |roll| or |pitch| beyond ~34°
        # ends the episode with wheels zeroed instead of grinding.
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
            # TILT warning tier (2026-08-04: a cylinder-base climb tilted
            # the robot enough to shear the lidar plane and corrupt the
            # map — unknown count tripled — while staying far below the
            # 34° kill threshold; it, the node didn't)
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
    # tighter GIL switch interval: with the explore-pick
    # worker CPU-bound in pure Python, the default 5 ms interval let
    # it monopolize the interpreter (control-tick starvation storms);
    # 2 ms + the A* cooperative yields keep the 4 Hz tick responsive.
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
