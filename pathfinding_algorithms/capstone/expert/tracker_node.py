#!/usr/bin/env python3
"""expert_tracker: follows a planned world-frame path via pure pursuit.

Consumes /odom, publishes /cmd_vel at `rate` Hz (sim time) and a JSON status
on /expert/status. The odom frame is anchored at the spawn pose (the
diff-drive plugin re-anchors odom on every respawn), so the world pose is
spawn_pose ∘ odom_pose.

Run with the capstone venv python:
  .venv/bin/python expert/tracker_node.py --ros-args \
      -p use_sim_time:=true -p path_file:=/tmp/path.npy \
      -p spawn_x:=-2.25 -p spawn_y:=3.0 -p spawn_yaw:=1.5708
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.geometry import yaw_from_quaternion  # noqa: E402
from common.pure_pursuit import PurePursuit      # noqa: E402


class ExpertTracker(Node):
    def __init__(self):
        super().__init__('expert_tracker')
        self.declare_parameter('path_file', '')
        self.declare_parameter('spawn_x', -2.25)
        self.declare_parameter('spawn_y', 3.0)
        self.declare_parameter('spawn_yaw', 1.5708)
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('v_max', 0.22)
        self.declare_parameter('lookahead', 0.25)
        self.declare_parameter('goal_tol', 0.15)
        # Noise injection (DART-style, decisions #27a): periodic short yaw-
        # bias bursts drive the robot OFF the expert path so the recorded
        # episode demonstrates drift-and-recover — coverage for exactly the
        # off-distribution states a deployed imitation policy reaches. The
        # centered v2 expert never passes closer than ~0.3 m to obstacles on
        # its own, so without this the policy has no near-obstacle
        # supervision at all. Bursts are suppressed while anything is within
        # perturb_min_clear ahead (a burst into a cylinder fails the episode).
        self.declare_parameter('perturb_every', 0.0)   # mean s between bursts (0 = off)
        self.declare_parameter('perturb_dur', 0.45)    # burst length [s]
        self.declare_parameter('perturb_w', 1.2)       # max |yaw bias| [rad/s]
        self.declare_parameter('perturb_min_clear', 0.35)
        self.declare_parameter('perturb_seed', 0)

        path = np.load(self.get_parameter('path_file').value)
        self.pp = PurePursuit(
            path,
            lookahead=self.get_parameter('lookahead').value,
            v_max=self.get_parameter('v_max').value,
            goal_tol=self.get_parameter('goal_tol').value)

        sx = self.get_parameter('spawn_x').value
        sy = self.get_parameter('spawn_y').value
        syaw = self.get_parameter('spawn_yaw').value
        self._anchor = (sx, sy, syaw)
        self._odom = None
        self._done = False
        self._cross = []

        self._perturb_every = float(self.get_parameter('perturb_every').value)
        self._perturb_dur = float(self.get_parameter('perturb_dur').value)
        self._perturb_w = float(self.get_parameter('perturb_w').value)
        self._perturb_clear = float(
            self.get_parameter('perturb_min_clear').value)
        self._rng = np.random.default_rng(
            int(self.get_parameter('perturb_seed').value))
        self._front_min = 10.0
        self._burst_until = -1.0
        self._burst_w = 0.0
        self._next_burst = None
        if self._perturb_every > 0:
            from sensor_msgs.msg import LaserScan
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            qos = QoSProfile(depth=5)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            self.create_subscription(LaserScan, '/scan', self._on_scan, qos)

        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._status_pub = self.create_publisher(String, '/expert/status', 10)
        period = 1.0 / self.get_parameter('rate').value
        self.create_timer(period, self._tick)
        self._status_i = 0

    def _on_scan(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float32)
        n = len(r)
        # forward +-45 deg cone (scan index 0 = straight ahead on the TB3)
        k = max(1, int(n * 45 / 360))
        front = np.concatenate([r[:k], r[-k:]])
        front = front[np.isfinite(front) & (front > 0.01)]
        self._front_min = float(front.min()) if len(front) else 10.0

    def _on_odom(self, msg):
        self._odom = msg

    def _world_pose(self):
        p = self._odom.pose.pose
        oyaw = yaw_from_quaternion(p.orientation.x, p.orientation.y,
                                   p.orientation.z, p.orientation.w)
        sx, sy, syaw = self._anchor
        c, s = math.cos(syaw), math.sin(syaw)
        return (sx + c * p.position.x - s * p.position.y,
                sy + s * p.position.x + c * p.position.y,
                syaw + oyaw)

    def _perturbation(self, now_s):
        """Current yaw bias [rad/s], scheduling bursts as time passes."""
        if self._perturb_every <= 0:
            return 0.0
        if self._next_burst is None:   # first tick: seed the schedule
            self._next_burst = now_s + self._rng.uniform(
                0.5, self._perturb_every)
        if now_s < self._burst_until:
            return self._burst_w
        if now_s >= self._next_burst:
            self._next_burst = now_s + self._rng.uniform(
                0.6 * self._perturb_every, 1.4 * self._perturb_every)
            # hold off while an obstacle is close ahead — the recovery data
            # comes from veering NEAR clutter, not INTO it
            if self._front_min >= self._perturb_clear:
                self._burst_until = now_s + self._perturb_dur
                self._burst_w = float(self._rng.uniform(0.4, 1.0)
                                      * self._perturb_w
                                      * self._rng.choice([-1.0, 1.0]))
                return self._burst_w
        return 0.0

    def _tick(self):
        if self._odom is None:
            return
        if self._done:
            self._cmd_pub.publish(Twist())   # keep the zero latched
        else:
            x, y, yaw = self._world_pose()
            v, w, done = self.pp.step(x, y, yaw)
            now_s = self.get_clock().now().nanoseconds * 1e-9
            wb = self._perturbation(now_s)
            cmd = Twist()
            cmd.linear.x = v * (0.8 if wb else 1.0)
            cmd.angular.z = w + wb
            self._cmd_pub.publish(cmd)
            self._cross.append(self.pp.cross_track)
            self._done = done

        self._status_i += 1
        if self._status_i % 10 == 0 or self._done:   # ~2 Hz status
            msg = String()
            msg.data = json.dumps({
                'done': self._done,
                'fraction': round(self.pp.done_fraction, 4),
                'cross_track_mean': round(float(np.mean(self._cross)), 4)
                if self._cross else None,
                'cross_track_max': round(float(np.max(self._cross)), 4)
                if self._cross else None,
            })
            self._status_pub.publish(msg)


def main():
    rclpy.init()
    node = ExpertTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
