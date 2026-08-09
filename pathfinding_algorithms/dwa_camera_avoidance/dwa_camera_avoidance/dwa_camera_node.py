#!/usr/bin/env python3
"""
DWA obstacle avoidance node fused with monocular camera obstacles.

Identical to dwa_node from dwa_avoidance, extended with:
  - Subscription to /scan_camera (virtual LaserScan from camera_obstacle_node)
  - Merged obstacle list: LiDAR obstacles + camera-detected obstacles
  - camera_obstacle_max_age parameter: camera obstacles older than this are
    discarded to avoid ghost obstacles from a stale image frame

Goal behaviour and all DWA parameters are unchanged from dwa_node.
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

GOAL_INTERVAL_S  = 45.0
GOAL_DIST        = 1.0
MIN_CLEAR_M      = 0.4
SECTOR_DEG       = 30
STUCK_TIMEOUT_S  = 3.0
STUCK_VEL_THRESH = 0.02

# ── Wheel-slip detection ──────────────────────────────────────────────────────
# Slip = odom claims the robot translated, but the LiDAR scan is frozen
# (pushing an obstacle both sensors miss, e.g. a smooth chair base).
SLIP_WINDOW_S        = 1.5    # compare scans this far apart
SLIP_MIN_ODOM_DIST   = 0.08   # m — odom must claim at least this much travel
SLIP_MAX_YAW_CHANGE  = math.radians(10)  # skip check while turning
SLIP_EXPECTED_MIN    = 0.04   # m — a ray must be expected to change this much to count
SLIP_MIN_INFO_RAYS   = 15     # abstain unless enough informative rays (open space!)
SLIP_ACTUAL_RATIO    = 0.35   # slip if median actual change < ratio × median expected

# ── Recovery behaviors (modeled on Nav2's BackUp / Spin / Wait) ──────────────
BUBBLE_STUCK_S     = 4.0    # in-bubble dwell before escalating to recovery

BACKUP_SPEED       = 0.10               # m/s
BACKUP_DIST        = 0.20               # m (level 0)
BACKUP_MAX_DIST    = 0.40               # m cap under escalation
SPIN_ANGLE         = math.radians(90)   # rad (level 0)
SPIN_MAX_ANGLE     = math.radians(180)  # rad cap under escalation
WAIT_S             = 0.5                # settle time after spin
REAR_CLEAR_M       = 0.30               # required clearance to back up
ESCALATE_WINDOW_S  = 30.0               # re-trigger within this → escalate


def _yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class DWACameraNode(Node):
    def __init__(self):
        super().__init__('dwa_camera_node')

        # ── Robot kinematic limits (TurtleBot3 Burger) ───────────────────────
        self.declare_parameter('max_speed',           0.22)
        self.declare_parameter('min_speed',          -0.05)
        self.declare_parameter('max_yaw_rate',        2.84)
        self.declare_parameter('max_accel',           2.50)
        self.declare_parameter('max_delta_yaw_rate',  3.2)
        self.declare_parameter('v_resolution',        0.01)
        self.declare_parameter('yaw_rate_resolution', 0.1)
        self.declare_parameter('dt',                  0.1)
        self.declare_parameter('predict_time',        2.0)
        self.declare_parameter('robot_radius',        0.15)
        self.declare_parameter('obstacle_margin',     0.05)
        self.declare_parameter('w_heading',           0.20)
        self.declare_parameter('w_clearance',         1.00)
        self.declare_parameter('w_speed',             0.50)
        self.declare_parameter('goal_tolerance',      0.25)

        # ── Camera fusion ────────────────────────────────────────────────────
        # Camera obstacles older than this (seconds) are dropped before
        # each DWA iteration so a frozen image can't cause phantom avoidance.
        self.declare_parameter('camera_obstacle_max_age', 0.5)

        # ── State ────────────────────────────────────────────────────────────
        self._state = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._lidar_obstacles:  list[tuple[float, float]] = []
        self._camera_obstacles: list[tuple[float, float]] = []
        self._camera_obs_time: float = 0.0   # wall time of last camera update
        self._scan_msg:   LaserScan | None = None
        self._goal:       tuple[float, float] | None = None
        self._external_goal = False
        self._at_goal       = False
        self._odom_ready    = False

        self._stuck_since:   float | None = None
        self._escape_dir     = 1.0
        self._last_cmd_v     = 0.0
        self._nosafe_since:  float | None = None
        self._dwa_failed     = False
        self._in_violation   = False
        self._bubble_since:  float | None = None

        # Recovery state machine: None | 'backup' | 'spin' | 'wait'
        self._recovery_stage: str | None = None
        self._stage_start    = 0.0
        self._stage_origin   = (0.0, 0.0, 0.0)
        self._spin_accum     = 0.0
        self._last_yaw       = 0.0
        self._backup_target  = BACKUP_DIST
        self._spin_target    = SPIN_ANGLE
        self._recovery_level = 0
        self._last_recovery_end = -math.inf

        # Short histories for slip detection: (t, x, y, yaw) and (t, ranges)
        self._odom_hist: deque[tuple[float, float, float, float]] = deque()
        self._scan_hist: deque[tuple[float, np.ndarray]] = deque()

        # ── ROS I/O ──────────────────────────────────────────────────────────
        self._cmd_pub  = self.create_publisher(Twist,       '/cmd_vel',   10)
        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        be_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan,   '/scan',        self._scan_cb,   be_qos)
        self.create_subscription(LaserScan,   '/scan_camera', self._camera_scan_cb, 10)
        self.create_subscription(Odometry,    '/odom',        self._odom_cb,   10)
        self.create_subscription(PoseStamped, '/goal_pose',   self._goal_cb,   10)

        dt = self.get_parameter('dt').value
        self.create_timer(dt, self._control_loop)
        self.create_timer(GOAL_INTERVAL_S, self._auto_pick_goal)

        self.get_logger().info('DWA-Camera node started — waiting for sensor data…')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        self._scan_msg = msg
        x, y, yaw = self._state[:3]
        obs: list[tuple[float, float]] = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and math.isfinite(r):
                obs.append((r * math.cos(yaw + angle) + x,
                             r * math.sin(yaw + angle) + y))
            angle += msg.angle_increment
        self._lidar_obstacles = obs

        t = self.get_clock().now().nanoseconds * 1e-9
        self._scan_hist.append((t, np.array(msg.ranges, dtype=float)))
        while self._scan_hist and t - self._scan_hist[0][0] > SLIP_WINDOW_S * 2:
            self._scan_hist.popleft()

    def _camera_scan_cb(self, msg: LaserScan):
        x, y, yaw = self._state[:3]
        obs: list[tuple[float, float]] = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and math.isfinite(r):
                obs.append((r * math.cos(yaw + angle) + x,
                             r * math.sin(yaw + angle) + y))
            angle += msg.angle_increment
        self._camera_obstacles = obs
        self._camera_obs_time  = self.get_clock().now().nanoseconds * 1e-9

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._state[0] = p.x
        self._state[1] = p.y
        self._state[2] = _yaw_from_quat(msg.pose.pose.orientation)
        self._state[3] = msg.twist.twist.linear.x
        self._state[4] = msg.twist.twist.angular.z

        t = self.get_clock().now().nanoseconds * 1e-9
        self._odom_hist.append((t, self._state[0], self._state[1], self._state[2]))
        while self._odom_hist and t - self._odom_hist[0][0] > SLIP_WINDOW_S * 2:
            self._odom_hist.popleft()

        if not self._odom_ready:
            self._odom_ready = True
            self._auto_pick_goal()

    def _goal_cb(self, msg: PoseStamped):
        self._goal = (msg.pose.position.x, msg.pose.position.y)
        self._external_goal = True
        self._at_goal = False
        self.get_logger().info(
            f'External goal: ({self._goal[0]:.2f}, {self._goal[1]:.2f})'
        )

    # ── Merged obstacle list ──────────────────────────────────────────────────

    def _merged_obstacles(self) -> list[tuple[float, float]]:
        now = self.get_clock().now().nanoseconds * 1e-9
        max_age = self.get_parameter('camera_obstacle_max_age').value
        if (now - self._camera_obs_time) > max_age:
            return self._lidar_obstacles
        return self._lidar_obstacles + self._camera_obstacles

    # ── Auto goal picker ──────────────────────────────────────────────────────

    def _auto_pick_goal(self):
        if not self._odom_ready or self._scan_msg is None:
            return
        if self._external_goal and not self._at_goal:
            return

        scan   = self._scan_msg
        ranges = np.array(scan.ranges, dtype=float)
        n      = len(ranges)
        if n == 0:
            return

        valid        = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        ranges_clean = np.where(valid, ranges, 0.0)
        sector_size  = max(1, int(math.radians(SECTOR_DEG) / scan.angle_increment))

        best_score  = -1.0
        best_center = 0

        for start in range(0, n, max(1, sector_size // 2)):
            end    = min(n, start + sector_size)
            sector = ranges_clean[start:end]
            if len(sector) == 0:
                continue
            clear_frac = float(np.mean(sector > MIN_CLEAR_M))
            mid_idx    = (start + end) // 2
            mid_angle  = scan.angle_min + mid_idx * scan.angle_increment
            fwd_bias   = 1.0 - abs(mid_angle) / math.pi
            score      = clear_frac * 0.8 + fwd_bias * 0.2
            if clear_frac >= 0.6 and score > best_score:
                best_score  = score
                best_center = (start + end) // 2

        if best_score < 0:
            self.get_logger().warn(
                'Auto goal: no clear sector found.',
                throttle_duration_sec=10.0,
            )
            return

        best_angle  = scan.angle_min + best_center * scan.angle_increment
        x, y, yaw   = self._state[:3]
        world_angle = yaw + best_angle
        self._goal          = (x + GOAL_DIST * math.cos(world_angle),
                               y + GOAL_DIST * math.sin(world_angle))
        self._external_goal = False
        self._at_goal       = False
        self.get_logger().info(
            f'Auto goal: ({self._goal[0]:.2f}, {self._goal[1]:.2f})  '
            f'sector={math.degrees(best_angle):.0f}°  score={best_score:.2f}'
        )

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self):
        obstacles = self._merged_obstacles()
        if not self._odom_ready or not obstacles:
            return

        if self._goal is not None:
            dist = math.hypot(
                self._goal[0] - self._state[0],
                self._goal[1] - self._state[1],
            )
            if dist < self.get_parameter('goal_tolerance').value:
                if not self._at_goal:
                    self.get_logger().info('Goal reached.')
                    self._at_goal = True
                self._publish_stop()
                return

        now   = self.get_clock().now().nanoseconds * 1e-9
        cur_v = abs(self._state[3])

        if self._recovery_stage is not None:
            self._run_recovery(now)
            return

        # Trigger 1: DWA deadlock — forward motion commanded but odom shows none.
        # Gated on the last command so idle/rotating phases don't false-trigger.
        stuck_reason = None
        if cur_v < STUCK_VEL_THRESH and self._last_cmd_v > 0.03:
            if self._stuck_since is None:
                self._stuck_since = now
            elif (now - self._stuck_since) > STUCK_TIMEOUT_S:
                stuck_reason = 'no motion despite commands'
        else:
            self._stuck_since = None

        # Trigger 2: wheel slip — odom claims motion but the scan is frozen
        # (pushing an obstacle below the LiDAR plane / invisible to the camera).
        if stuck_reason is None and self._slip_detected(now):
            stuck_reason = 'wheel slip: odom moving, scan frozen'

        if stuck_reason is not None:
            self._start_recovery(now, stuck_reason)
            return

        best_v, best_w = self._dwa(obstacles)

        # Trigger 3: DWA deadlock — every trajectory rejected for several
        # seconds straight (e.g. wedged against an obstacle inside the
        # safety bubble). Rotation fallback alone never escapes this.
        if self._dwa_failed:
            if self._nosafe_since is None:
                self._nosafe_since = now
            elif (now - self._nosafe_since) > STUCK_TIMEOUT_S:
                self._nosafe_since = None
                self._start_recovery(now, 'no safe trajectory for %.0fs' % STUCK_TIMEOUT_S)
                return
        else:
            self._nosafe_since = None

        # Trigger 4: trapped inside the safety bubble — retreat mode accepts
        # any trajectory that doesn't get closer, which legally includes
        # orbiting an obstacle at constant distance forever. If clearance
        # hasn't been regained after a few seconds, escalate to recovery,
        # which physically drives out and picks a fresh goal.
        if self._in_violation:
            if self._bubble_since is None:
                self._bubble_since = now
            elif (now - self._bubble_since) > BUBBLE_STUCK_S:
                self._bubble_since = None
                self._start_recovery(
                    now, 'trapped in safety bubble for %.0fs' % BUBBLE_STUCK_S)
                return
        else:
            self._bubble_since = None

        cmd = Twist()
        cmd.linear.x  = best_v
        cmd.angular.z = best_w
        self._cmd_pub.publish(cmd)
        self._last_cmd_v = best_v

    # ── DWA core ─────────────────────────────────────────────────────────────

    def _dwa(self, obstacles: list[tuple[float, float]]) -> tuple[float, float]:
        p      = self.get_parameter
        max_v  = p('max_speed').value
        min_v  = p('min_speed').value
        max_w  = p('max_yaw_rate').value
        max_a  = p('max_accel').value
        max_aw = p('max_delta_yaw_rate').value
        v_res  = p('v_resolution').value
        w_res  = p('yaw_rate_resolution').value
        dt     = p('dt').value
        pt     = p('predict_time').value
        r_rad  = p('robot_radius').value
        margin = p('obstacle_margin').value
        w_h    = p('w_heading').value
        w_c    = p('w_clearance').value
        w_s    = p('w_speed').value

        cur_v, cur_w = self._state[3], self._state[4]
        dw_vmin = max(min_v,  cur_v - max_a  * dt)
        dw_vmax = min(max_v,  cur_v + max_a  * dt)
        dw_wmin = max(-max_w, cur_w - max_aw * dt)
        dw_wmax = min( max_w, cur_w + max_aw * dt)

        best_score    = -math.inf
        best_v, best_w = 0.0, 0.0
        threshold     = r_rad + margin

        # If an obstacle is already inside the safety bubble (e.g. wedged
        # against furniture), the normal rule rejects every trajectory —
        # including ones that retreat. In that case accept any trajectory
        # that does not bring us closer than we already are.
        x0, y0 = self._state[0], self._state[1]
        d_now  = min((math.hypot(x0 - ox, y0 - oy) for ox, oy in obstacles),
                     default=math.inf)
        in_violation = d_now < threshold
        self._in_violation = in_violation
        if in_violation:
            self.get_logger().warn(
                f'Obstacle inside safety bubble ({d_now:.2f} m) — '
                f'allowing retreat trajectories.',
                throttle_duration_sec=5.0,
            )

        for v in np.arange(dw_vmin, dw_vmax + v_res, v_res):
            for w in np.arange(dw_wmin, dw_wmax + w_res, w_res):
                traj = self._simulate(v, w, pt, dt)
                if in_violation:
                    clearance = self._retreat_clearance(traj, d_now, obstacles)
                else:
                    clearance = self._min_clearance(traj, threshold, obstacles)
                if clearance is None:
                    continue
                score = (
                    w_h * self._heading_score(traj)
                    + w_c * self._clearance_score(clearance)
                    + w_s * self._speed_score(v, max_v)
                )
                if score > best_score:
                    best_score    = score
                    best_v, best_w = float(v), float(w)

        self._dwa_failed = bool(best_score == -math.inf)
        if self._dwa_failed:
            self.get_logger().warn(
                'No safe trajectory — rotating.',
                throttle_duration_sec=2.0,
            )
            best_v = 0.0
            best_w = self._open_rotation_dir() * max_w * 0.5

        return (float(np.clip(best_v, min_v, max_v)),
                float(np.clip(best_w, -max_w, max_w)))

    # ── Trajectory / scoring ─────────────────────────────────────────────────

    def _simulate(self, v, w, predict_time, dt):
        x, y, yaw = self._state[:3]
        traj = []
        t = 0.0
        while t <= predict_time:
            x   += v * math.cos(yaw) * dt
            y   += v * math.sin(yaw) * dt
            yaw += w * dt
            traj.append((x, y, yaw))
            t += dt
        return traj

    def _min_clearance(self, traj, threshold,
                       obstacles: list[tuple[float, float]]) -> float | None:
        min_d = math.inf
        for (px, py, _) in traj:
            for (ox, oy) in obstacles:
                d = math.hypot(px - ox, py - oy)
                if d < threshold:
                    return None
                if d < min_d:
                    min_d = d
        return min_d

    def _retreat_clearance(self, traj, d_now,
                           obstacles: list[tuple[float, float]]) -> float | None:
        """Clearance rule while already inside the safety bubble: reject only
        trajectories that come closer than we already are (small tolerance),
        so paths that back away from the violating obstacle stay viable."""
        min_d = math.inf
        for (px, py, _) in traj:
            for (ox, oy) in obstacles:
                d = math.hypot(px - ox, py - oy)
                if d < d_now - 0.02:
                    return None
                if d < min_d:
                    min_d = d
        return min_d

    def _clearance_score(self, clearance):
        return min(clearance, 2.0) / 2.0

    def _heading_score(self, traj):
        if not traj:
            return 0.0
        gx, gy = self._effective_goal()
        fx, fy, fyaw = traj[-1]
        angle_to_goal = math.atan2(gy - fy, gx - fx)
        error = abs(math.atan2(math.sin(angle_to_goal - fyaw),
                               math.cos(angle_to_goal - fyaw)))
        return (math.pi - error) / math.pi

    def _speed_score(self, v, max_v):
        return v / max_v if v >= 0 else 0.0

    def _effective_goal(self):
        if self._goal is not None and not self._at_goal:
            return self._goal
        x, y, yaw = self._state[:3]
        return (x + 1.0 * math.cos(yaw), y + 1.0 * math.sin(yaw))

    # ── Slip detection ────────────────────────────────────────────────────────

    def _slip_detected(self, now: float) -> bool:
        """Odom claims translation over the window but the scan barely changed.

        Compares each ray's actual change against the change *expected* from the
        odom displacement (Δr ≈ d·cos(ray − motion direction)). Rays facing open
        space (infinite) or perpendicular to the motion carry no information and
        are excluded; with too few informative rays the check abstains rather
        than false-triggering in open areas.
        """
        if self._last_cmd_v <= 0.03:
            return False  # only meaningful while driving forward
        if len(self._odom_hist) < 2 or len(self._scan_hist) < 2:
            return False

        # oldest odom sample inside the window
        old = next((s for s in self._odom_hist if now - s[0] <= SLIP_WINDOW_S), None)
        if old is None or (now - old[0]) < SLIP_WINDOW_S * 0.6:
            return False
        _, ox, oy, oyaw = old
        _, cx, cy, cyaw = self._odom_hist[-1]

        yaw_change = abs(math.atan2(math.sin(cyaw - oyaw), math.cos(cyaw - oyaw)))
        if yaw_change > SLIP_MAX_YAW_CHANGE:
            return False  # turning legitimately shifts the scan
        disp = math.hypot(cx - ox, cy - oy)
        if disp < SLIP_MIN_ODOM_DIST:
            return False  # odom doesn't claim real travel

        old_scan = next((s for s in self._scan_hist if now - s[0] <= SLIP_WINDOW_S), None)
        scan_msg = self._scan_msg
        if old_scan is None or scan_msg is None:
            return False
        a, b = old_scan[1], self._scan_hist[-1][1]
        if len(a) != len(b) or len(a) != len(scan_msg.ranges):
            return False

        motion_dir = math.atan2(cy - oy, cx - ox)
        ray_angles = (scan_msg.angle_min
                      + np.arange(len(a)) * scan_msg.angle_increment
                      + cyaw)
        expected = np.abs(disp * np.cos(ray_angles - motion_dir))
        informative = np.isfinite(a) & np.isfinite(b) & (expected > SLIP_EXPECTED_MIN)
        if informative.sum() < SLIP_MIN_INFO_RAYS:
            return False  # not enough geometry to judge — abstain
        actual = np.abs(a[informative] - b[informative])
        return float(np.median(actual)) < SLIP_ACTUAL_RATIO * float(np.median(expected[informative]))

    # ── Recovery behaviors (Nav2-style BackUp → Spin → Wait) ─────────────────

    def _start_recovery(self, now: float, reason: str):
        if (now - self._last_recovery_end) < ESCALATE_WINDOW_S:
            self._recovery_level = min(self._recovery_level + 1, 2)
        else:
            self._recovery_level = 0
        f = 1.0 + 0.5 * self._recovery_level
        self._backup_target = min(BACKUP_DIST * f, BACKUP_MAX_DIST)
        self._spin_target   = min(SPIN_ANGLE * f, SPIN_MAX_ANGLE)
        self._escape_dir    = self._open_rotation_dir()

        self._stage_origin = tuple(self._state[:3])
        self._stage_start  = now
        self._spin_accum   = 0.0
        self._last_yaw     = self._state[2]

        if self._rear_clear():
            self._recovery_stage = 'backup'
        elif self._has_clear_forward():
            self._recovery_stage = 'forward'   # rear blocked — escape ahead
        else:
            self._recovery_stage = 'spin'
        self.get_logger().warn(
            f'Stuck ({reason}) — recovery level {self._recovery_level}: '
            f'{self._recovery_stage} first, escape {self._backup_target:.2f} m, '
            f'spin {math.degrees(self._spin_target):.0f}°'
        )

    def _run_recovery(self, now: float):
        cmd = Twist()

        if self._recovery_stage in ('backup', 'forward'):
            forward = self._recovery_stage == 'forward'
            dist    = math.hypot(self._state[0] - self._stage_origin[0],
                                 self._state[1] - self._stage_origin[1])
            timeout = (self._backup_target / BACKUP_SPEED) * 2.0
            if dist >= self._backup_target or (now - self._stage_start) > timeout:
                # after a forward escape the blockage is behind us — no spin
                self._advance_stage(now, 'wait' if forward else 'spin')
            else:
                cmd.linear.x = BACKUP_SPEED if forward else -BACKUP_SPEED
                self._cmd_pub.publish(cmd)
                return

        if self._recovery_stage == 'spin':
            yaw = self._state[2]
            self._spin_accum += abs(math.atan2(math.sin(yaw - self._last_yaw),
                                               math.cos(yaw - self._last_yaw)))
            self._last_yaw = yaw
            max_w   = self.get_parameter('max_yaw_rate').value
            timeout = (self._spin_target / (max_w * 0.5)) * 3.0
            if self._spin_accum >= self._spin_target or (now - self._stage_start) > timeout:
                self._advance_stage(now, 'wait')
            else:
                cmd.angular.z = self._escape_dir * max_w * 0.5
                self._cmd_pub.publish(cmd)
                return

        if self._recovery_stage == 'wait':
            self._publish_stop()
            if (now - self._stage_start) >= WAIT_S:
                self._finish_recovery(now)

    def _advance_stage(self, now: float, stage: str):
        self._recovery_stage = stage
        self._stage_start    = now
        self._stage_origin   = tuple(self._state[:3])
        self._spin_accum     = 0.0
        self._last_yaw       = self._state[2]

    def _finish_recovery(self, now: float):
        self._recovery_stage    = None
        self._last_recovery_end = now
        self._stuck_since       = None
        self._odom_hist.clear()
        self._scan_hist.clear()
        if not self._external_goal:
            self._goal = None      # force a fresh goal away from the blockage
            self._auto_pick_goal()
        self.get_logger().info('Recovery complete — resuming DWA.')

    def _rear_clear(self) -> bool:
        """Median finite range in the rear ±30° sector exceeds REAR_CLEAR_M."""
        scan = self._scan_msg
        if scan is None or not scan.ranges:
            return False
        rear = []
        angle = scan.angle_min
        for r in scan.ranges:
            # angular distance from π (straight back)
            off = math.atan2(math.sin(angle - math.pi), math.cos(angle - math.pi))
            if abs(off) <= math.radians(30) and math.isfinite(r) and r > scan.range_min:
                rear.append(r)
            angle += scan.angle_increment
        return bool(rear) and float(np.median(rear)) > REAR_CLEAR_M

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _open_rotation_dir(self) -> float:
        if self._scan_msg is None:
            return 1.0
        ranges = self._scan_msg.ranges
        n = len(ranges)
        if n == 0:
            return 1.0
        left  = ranges[:n // 2]
        right = ranges[n // 2:]
        avg   = lambda seq: float(np.mean([r for r in seq if math.isfinite(r)] or [0.0]))
        return 1.0 if avg(left) >= avg(right) else -1.0

    def _has_clear_forward(self) -> bool:
        scan = self._scan_msg
        if scan is None:
            return False
        n   = len(scan.ranges)
        arc = max(1, int(math.radians(30) / scan.angle_increment))
        center = n // 2
        sector = scan.ranges[max(0, center - arc): center + arc]
        valid  = [r for r in sector if math.isfinite(r) and r > scan.range_min]
        return bool(valid) and float(np.median(valid)) > MIN_CLEAR_M

    def _publish_stop(self):
        self._cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = DWACameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
