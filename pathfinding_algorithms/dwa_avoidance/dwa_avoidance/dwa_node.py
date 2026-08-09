#!/usr/bin/env python3
"""
Dynamic Window Approach (DWA) obstacle avoidance node for TurtleBot3.

Subscribes to /scan and /odom, publishes /cmd_vel.
Intended to run alongside Cartographer for simultaneous mapping.

Goal behaviour
--------------
An external goal can be set via /goal_pose (geometry_msgs/PoseStamped).
If no external goal is active, the node auto-picks a goal every
GOAL_INTERVAL_S seconds by scanning for the most open direction and
placing a target GOAL_DIST metres away — reachable in ~5 s at full speed.
The robot stops once it reaches the goal and waits for the next pick.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

GOAL_INTERVAL_S = 45.0   # seconds between auto goal picks
GOAL_DIST       = 1.0    # metres — ~4.5 s travel at max speed
MIN_CLEAR_M     = 0.4    # minimum range in goal sector to be considered open
SECTOR_DEG      = 30     # width of each candidate sector in degrees
STUCK_TIMEOUT_S = 3.0    # seconds of near-zero velocity before forced escape
STUCK_VEL_THRESH = 0.02  # m/s  — below this is considered "not moving"


def _yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class DWANode(Node):
    def __init__(self):
        super().__init__('dwa_node')

        # ── Robot kinematic limits (TurtleBot3 Burger) ──────────────────────
        #
        # All values sourced from:
        #   [A] ROBOTIS TurtleBot3 e-Manual "Specifications" table
        #       https://emanual.robotis.com/docs/en/platform/turtlebot3/features/
        #   [B] turtlebot3_navigation2/param/humble/burger.yaml — DWB "FollowPath" block
        #       (~/turtlebot3_ws/src/turtlebot3/turtlebot3_navigation2/param/humble/burger.yaml)
        #   [C] turtlebot3_node/param/burger.yaml — wheel geometry block
        #       (~/turtlebot3_ws/src/turtlebot3/turtlebot3_node/param/burger.yaml)
        #
        # max_speed: 0.22 m/s
        #   [A] "Maximum translational velocity: 0.22 m/s"
        #   [B] max_vel_x: 0.22
        self.declare_parameter('max_speed', 0.22)

        # min_speed: -0.05 m/s
        #   [B] min_vel_x: 0.0 (Nav2 disallows reverse).
        #   We allow a small reverse for DWA recovery when cornered.
        self.declare_parameter('min_speed', -0.05)

        # max_yaw_rate: 2.84 rad/s
        #   [A] "Maximum rotational velocity: 2.84 rad/s"
        #   Derived from wheel geometry [C]: 2 × max_wheel_speed / separation
        #     = 2 × 0.22 / 0.160 ≈ 2.75; ROBOTIS rounds up to 2.84 in the spec.
        self.declare_parameter('max_yaw_rate', 2.84)

        # max_accel: 2.5 m/s²
        #   [B] acc_lim_x: 2.5  (decel_lim_x: -2.5)
        self.declare_parameter('max_accel', 2.50)

        # max_delta_yaw_rate: 3.2 rad/s²
        #   [B] acc_lim_theta: 3.2  (decel_lim_theta: -3.2)
        self.declare_parameter('max_delta_yaw_rate', 3.2)

        # ── DWA sampling resolution ─────────────────────────────────────────
        # [B] linear_granularity: 0.05 m/s, angular_granularity: 0.025 rad/s
        self.declare_parameter('v_resolution',        0.01)
        self.declare_parameter('yaw_rate_resolution', 0.1)

        # ── Prediction horizon ───────────────────────────────────────────────
        # [B] sim_time: 1.5 s — we extend to 2.0 s for more obstacle lookahead.
        self.declare_parameter('dt',           0.1)
        self.declare_parameter('predict_time', 2.0)
        # Control-loop period override [s]. 0 = tick at dt (legacy behavior).
        # Set 0.5 / 0.333 / 0.25 for the capstone's 2/3/4 Hz comparison —
        # the dynamic window widens to match (reachable velocities per tick)
        # while dt stays the trajectory-integration step.
        self.declare_parameter('control_period', 0.0)

        # ── Safety ───────────────────────────────────────────────────────────
        # robot_radius: 0.15 m (body 0.10 m + 0.05 m LiDAR noise buffer)
        # obstacle_margin adds 0.05 m → 0.20 m total minimum clearance.
        self.declare_parameter('robot_radius',    0.15)
        self.declare_parameter('obstacle_margin', 0.05)

        # ── Cost weights ─────────────────────────────────────────────────────
        self.declare_parameter('w_heading',   0.20)
        self.declare_parameter('w_clearance', 1.00)
        self.declare_parameter('w_speed',     0.50)

        # ── Goal ─────────────────────────────────────────────────────────────
        self.declare_parameter('goal_tolerance', 0.25)  # m

        # ── State ────────────────────────────────────────────────────────────
        self._state = [0.0, 0.0, 0.0, 0.0, 0.0]   # x, y, yaw, v, w
        self._obstacles: list[tuple[float, float]] = []
        self._scan_msg: LaserScan | None = None
        self._goal: tuple[float, float] | None = None
        self._external_goal = False   # True when goal came from /goal_pose topic
        self._at_goal = False
        self._odom_ready = False

        # ── Stuck detection ───────────────────────────────────────────────────
        self._stuck_since: float | None = None   # wall time when low-vel started
        self._escaping = False                   # True while forced escape is active
        self._escape_start: float | None = None  # wall time escape began
        self._escape_dir = 1.0                   # +1 or -1 rotation direction

        # ── ROS I/O ──────────────────────────────────────────────────────────
        self._cmd_pub  = self.create_publisher(Twist,       '/cmd_vel',   10)
        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan,   '/scan',      self._scan_cb,  scan_qos)
        self.create_subscription(Odometry,    '/odom',      self._odom_cb,  10)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb,  10)

        dt = self.get_parameter('dt').value
        period = self.get_parameter('control_period').value or dt
        self._tick_period = period
        self.create_timer(period, self._control_loop)

        # Auto goal picker fires every 45 s
        self.create_timer(GOAL_INTERVAL_S, self._auto_pick_goal)

        self.get_logger().info('DWA node started — waiting for sensor data…')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        self._scan_msg = msg
        x, y, yaw = self._state[:3]
        obstacles: list[tuple[float, float]] = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and math.isfinite(r):
                ox = r * math.cos(yaw + angle) + x
                oy = r * math.sin(yaw + angle) + y
                obstacles.append((ox, oy))
            angle += msg.angle_increment
        self._obstacles = obstacles

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._state[0] = p.x
        self._state[1] = p.y
        self._state[2] = _yaw_from_quat(msg.pose.pose.orientation)
        self._state[3] = msg.twist.twist.linear.x
        self._state[4] = msg.twist.twist.angular.z
        if not self._odom_ready:
            self._odom_ready = True
            # Fire the first goal pick as soon as odom arrives
            self._auto_pick_goal()

    def _goal_cb(self, msg: PoseStamped):
        self._goal = (msg.pose.position.x, msg.pose.position.y)
        self._external_goal = True
        self._at_goal = False
        self.get_logger().info(
            f'External goal: ({self._goal[0]:.2f}, {self._goal[1]:.2f})'
        )

    # ── Auto goal picker ─────────────────────────────────────────────────────

    def _auto_pick_goal(self):
        """
        Every GOAL_INTERVAL_S seconds: find the most open direction in the latest
        scan and set a goal GOAL_DIST metres in that direction.

        Sector scoring:
          - Divide the 360° scan into SECTOR_DEG-wide windows.
          - Score each sector by the fraction of rays with range > MIN_CLEAR_M.
          - Bias sectors near the robot's current heading (forward preferred).
          - Pick the highest-scoring sector; give up if no sector is ≥ 60% clear.
        """
        if not self._odom_ready or self._scan_msg is None:
            return

        # External goals from /goal_pose take priority — don't override them
        # unless the robot has already reached that goal.
        if self._external_goal and not self._at_goal:
            return

        scan   = self._scan_msg
        ranges = np.array(scan.ranges, dtype=float)
        n      = len(ranges)

        if n == 0:
            return

        # Replace out-of-range / non-finite values with 0 (treat as close obstacle)
        valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        ranges_clean = np.where(valid, ranges, 0.0)

        sector_size = max(1, int(math.radians(SECTOR_DEG) / scan.angle_increment))

        best_score  = -1.0
        best_center = 0  # ray index

        for start in range(0, n, max(1, sector_size // 2)):   # half-sector stride
            end    = min(n, start + sector_size)
            sector = ranges_clean[start:end]
            if len(sector) == 0:
                continue

            clear_frac = float(np.mean(sector > MIN_CLEAR_M))

            # Heading bias: prefer sectors near robot forward direction (index 0
            # in LDS-02 is directly forward; angle_min = -π, so forward is near
            # the wrap-around point.  Compute the sector's mean angle and penalise
            # large deviations from 0 (forward).
            mid_idx    = (start + end) // 2
            mid_angle  = scan.angle_min + mid_idx * scan.angle_increment
            # Normalise to [-π, π] and compute angular closeness to forward (0)
            fwd_bias   = 1.0 - abs(mid_angle) / math.pi   # 1 = forward, 0 = behind

            score = clear_frac * 0.8 + fwd_bias * 0.2

            if clear_frac >= 0.6 and score > best_score:
                best_score  = score
                best_center = (start + end) // 2

        if best_score < 0:
            self.get_logger().warn(
                'Auto goal: no clear sector found — keeping current goal.',
                throttle_duration_sec=10.0,
            )
            return

        # Convert best ray index → world-frame goal position
        best_angle  = scan.angle_min + best_center * scan.angle_increment
        x, y, yaw   = self._state[:3]
        world_angle = yaw + best_angle
        gx = x + GOAL_DIST * math.cos(world_angle)
        gy = y + GOAL_DIST * math.sin(world_angle)

        self._goal         = (gx, gy)
        self._external_goal = False
        self._at_goal      = False

        self.get_logger().info(
            f'Auto goal picked: ({gx:.2f}, {gy:.2f})  '
            f'sector_angle={math.degrees(best_angle):.0f}°  '
            f'score={best_score:.2f}'
        )

    # ── Main control loop ────────────────────────────────────────────────────

    def _control_loop(self):
        if not self._odom_ready or not self._obstacles:
            return

        if self._goal is not None:
            dist = math.hypot(
                self._goal[0] - self._state[0],
                self._goal[1] - self._state[1],
            )
            if dist < self.get_parameter('goal_tolerance').value:
                if not self._at_goal:
                    self.get_logger().info('Goal reached — waiting for next auto pick.')
                    self._at_goal = True
                self._publish_stop()
                return

        now = self.get_clock().now().nanoseconds * 1e-9

        # ── Stuck detector ────────────────────────────────────────────────────
        # If the robot has been nearly stationary for STUCK_TIMEOUT_S seconds,
        # override DWA with a full-speed rotation until open space is visible.
        cur_v = abs(self._state[3])
        if cur_v < STUCK_VEL_THRESH:
            if self._stuck_since is None:
                self._stuck_since = now
            elif not self._escaping and (now - self._stuck_since) > STUCK_TIMEOUT_S:
                self._escaping = True
                self._escape_start = now
                # Pick whichever rotation direction has more open space
                self._escape_dir = self._open_rotation_dir()
                self.get_logger().warn('Stuck detected — forcing escape rotation.')
        else:
            self._stuck_since = None
            self._escaping = False

        if self._escaping:
            max_w = self.get_parameter('max_yaw_rate').value
            cmd = Twist()
            cmd.angular.z = self._escape_dir * max_w * 0.5
            self._cmd_pub.publish(cmd)
            # Exit escape once we've rotated enough to see a clear path
            if self._scan_msg is not None and self._has_clear_forward():
                self.get_logger().info('Escape complete — resuming DWA.')
                self._escaping = False
                self._stuck_since = None
            return

        best_v, best_w = self._dwa()
        cmd = Twist()
        cmd.linear.x  = best_v
        cmd.angular.z = best_w
        self._cmd_pub.publish(cmd)

    # ── DWA core ─────────────────────────────────────────────────────────────

    def _dwa(self) -> tuple[float, float]:
        p = self.get_parameter
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

        # Reachable-velocity window spans one CONTROL TICK (equals dt unless
        # control_period overrides it for the 2/3/4 Hz comparison).
        tick = self._tick_period
        dw_vmin = max(min_v,  cur_v - max_a  * tick)
        dw_vmax = min(max_v,  cur_v + max_a  * tick)
        dw_wmin = max(-max_w, cur_w - max_aw * tick)
        dw_wmax = min( max_w, cur_w + max_aw * tick)

        best_score = -math.inf
        best_v, best_w = 0.0, 0.0
        clearance_threshold = r_rad + margin

        for v in np.arange(dw_vmin, dw_vmax + v_res, v_res):
            for w in np.arange(dw_wmin, dw_wmax + w_res, w_res):
                traj      = self._simulate(v, w, pt, dt)
                clearance = self._min_clearance(traj, clearance_threshold)
                if clearance is None:
                    continue
                score = (
                    w_h * self._heading_score(traj)
                    + w_c * self._clearance_score(clearance)
                    + w_s * self._speed_score(v, max_v)
                )
                if score > best_score:
                    best_score = score
                    best_v, best_w = float(v), float(w)

        if best_score == -math.inf:
            self.get_logger().warn(
                'No safe trajectory — rotating to escape.',
                throttle_duration_sec=2.0,
            )
            best_v = 0.0
            best_w = self._open_rotation_dir() * max_w * 0.5  # ~1.4 rad/s

        # np.arange can produce values fractionally outside the window —
        # clamp to hard limits before publishing.
        best_v = float(np.clip(best_v, min_v, max_v))
        best_w = float(np.clip(best_w, -max_w, max_w))
        return best_v, best_w

    # ── Trajectory simulation ─────────────────────────────────────────────────

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

    # ── Cost / score functions ────────────────────────────────────────────────

    def _min_clearance(self, traj, threshold):
        # Reject any trajectory whose predicted path comes within `threshold`
        # of any obstacle — absolute check, not relative to start distance.
        # The old relative check let the robot oscillate near walls because
        # micro-moves that didn't get *closer* than the current position were
        # accepted even when the robot was already inside the safe margin.
        min_d = math.inf
        for (px, py, _) in traj:
            for (ox, oy) in self._obstacles:
                d = math.hypot(px - ox, py - oy)
                if d < threshold:
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
        # Fallback: virtual goal straight ahead while waiting for auto pick
        x, y, yaw = self._state[:3]
        return (x + 1.0 * math.cos(yaw), y + 1.0 * math.sin(yaw))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _open_rotation_dir(self) -> float:
        """Return +1.0 or -1.0 for whichever half of the scan has more open space."""
        if self._scan_msg is None:
            return 1.0
        ranges = self._scan_msg.ranges
        n = len(ranges)
        if n == 0:
            return 1.0
        left  = ranges[:n // 2]
        right = ranges[n // 2:]
        avg = lambda seq: float(np.mean([r for r in seq if math.isfinite(r)] or [0.0]))
        return 1.0 if avg(left) >= avg(right) else -1.0

    def _has_clear_forward(self) -> bool:
        """True if the forward arc (±30°) has median clearance > MIN_CLEAR_M."""
        scan = self._scan_msg
        if scan is None:
            return False
        n = len(scan.ranges)
        arc = max(1, int(math.radians(30) / scan.angle_increment))
        center = n // 2
        sector = scan.ranges[max(0, center - arc): center + arc]
        valid = [r for r in sector if math.isfinite(r) and r > scan.range_min]
        if not valid:
            return False
        return float(np.median(valid)) > MIN_CLEAR_M

    def _publish_stop(self):
        self._cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = DWANode()
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
