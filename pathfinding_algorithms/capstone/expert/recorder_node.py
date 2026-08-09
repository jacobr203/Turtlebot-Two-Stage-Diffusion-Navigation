#!/usr/bin/env python3
"""episode_recorder: samples training rows at `rate` Hz (sim time) and writes
one .npz per episode.

Policy-facing quantities (scan, goal/subgoal in the robot frame) are computed
from ODOM-derived pose — exactly what the deployed policy will have. Ground
truth from /gazebo/model_states is recorded separately for metrics only.

Stops (flushes the npz and exits) on /expert/stop or SIGINT.

npz schema (float32 unless noted):
  scans[T,360]   fixed-bin resampled ranges [m]
  odom[T,6]      x, y, yaw (odom frame), v, w, scan_age [s]
  gt_pose[T,3]   ground-truth world x, y, yaw
  goal[T,3]      final goal in robot frame: dx, dy, dist
  subgoal[T,3]   point `subgoal_dist` m along the remaining expert path,
                 robot frame: dx, dy, dist  (two-stage conditioning labels)
  cmd[T,2]       last executed (v, w)
  t[T]           sim time (float64)
  spawn_pose[3], goal_world[2], world_idx[1], path[N,2]   episode metadata
"""

import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from gazebo_msgs.msg import ModelStates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.geometry import world_to_robot, yaw_from_quaternion  # noqa: E402
from common.scan import resample_scan                            # noqa: E402


class EpisodeRecorder(Node):
    def __init__(self):
        super().__init__('episode_recorder')
        self.declare_parameter('out_file', '')
        self.declare_parameter('path_file', '')
        self.declare_parameter('rate', 4.0)
        self.declare_parameter('spawn_x', -2.25)
        self.declare_parameter('spawn_y', 3.0)
        self.declare_parameter('spawn_yaw', 1.5708)
        self.declare_parameter('goal_x', -2.25)
        self.declare_parameter('goal_y', 13.0)
        self.declare_parameter('subgoal_dist', 1.2)
        self.declare_parameter('world_idx', -1)

        self.out_file = self.get_parameter('out_file').value
        self.path = np.load(self.get_parameter('path_file').value)
        seg = np.hypot(*np.diff(self.path.astype(np.float64), axis=0).T)
        self._path_s = np.concatenate([[0.0], np.cumsum(seg)])
        self._path_idx = 0
        self.anchor = (self.get_parameter('spawn_x').value,
                       self.get_parameter('spawn_y').value,
                       self.get_parameter('spawn_yaw').value)
        self.goal_world = np.array([self.get_parameter('goal_x').value,
                                    self.get_parameter('goal_y').value])
        self.subgoal_dist = self.get_parameter('subgoal_dist').value

        self._scan = None
        self._odom = None
        self._gt = None
        self._cmd = (0.0, 0.0)
        self.rows = {k: [] for k in
                     ('scans', 'odom', 'gt_pose', 'goal', 'subgoal', 'cmd', 't')}

        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(LaserScan, '/scan', self._on_scan, sensor_qos)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states',
                                 self._on_gt, 10)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)
        self.create_subscription(Empty, '/expert/stop', self._on_stop, 10)
        self.create_timer(1.0 / self.get_parameter('rate').value, self._tick)

    # ── inputs ────────────────────────────────────────────────────────────────
    def _on_scan(self, msg):
        self._scan = msg

    def _on_odom(self, msg):
        self._odom = msg

    def _on_gt(self, msg):
        try:
            i = msg.name.index('turtlebot3')
        except ValueError:
            return
        p = msg.pose[i]
        self._gt = (p.position.x, p.position.y,
                    yaw_from_quaternion(p.orientation.x, p.orientation.y,
                                        p.orientation.z, p.orientation.w))

    def _on_cmd(self, msg):
        self._cmd = (msg.linear.x, msg.angular.z)

    def _on_stop(self, _msg):
        self._flush()
        raise SystemExit(0)

    # ── sampling ──────────────────────────────────────────────────────────────
    def _world_pose(self):
        p = self._odom.pose.pose
        oyaw = yaw_from_quaternion(p.orientation.x, p.orientation.y,
                                   p.orientation.z, p.orientation.w)
        sx, sy, syaw = self.anchor
        c, s = math.cos(syaw), math.sin(syaw)
        return (sx + c * p.position.x - s * p.position.y,
                sy + s * p.position.x + c * p.position.y,
                syaw + oyaw)

    def _subgoal_world(self, pose):
        """Point subgoal_dist m along the REMAINING path (monotonic index)."""
        window = slice(self._path_idx, min(self._path_idx + 80, len(self.path)))
        d = np.hypot(self.path[window, 0] - pose[0],
                     self.path[window, 1] - pose[1])
        self._path_idx += int(np.argmin(d))
        target_s = self._path_s[self._path_idx] + self.subgoal_dist
        t_idx = min(int(np.searchsorted(self._path_s, target_s)),
                    len(self.path) - 1)
        return self.path[t_idx]

    def _tick(self):
        if self._scan is None or self._odom is None:
            return
        now = self.get_clock().now()
        scan_age = (now.nanoseconds
                    - (self._scan.header.stamp.sec * 10**9
                       + self._scan.header.stamp.nanosec)) * 1e-9

        pose = self._world_pose()
        op = self._odom.pose.pose
        tw = self._odom.twist.twist

        goal_r = world_to_robot(self.goal_world, pose)
        sub_r = world_to_robot(self._subgoal_world(pose), pose)

        self.rows['scans'].append(resample_scan(
            self._scan.ranges, self._scan.angle_min, self._scan.angle_increment))
        self.rows['odom'].append([
            op.position.x, op.position.y,
            yaw_from_quaternion(op.orientation.x, op.orientation.y,
                                op.orientation.z, op.orientation.w),
            tw.linear.x, tw.angular.z, scan_age])
        self.rows['gt_pose'].append(list(self._gt) if self._gt else [0, 0, 0])
        self.rows['goal'].append([goal_r[0], goal_r[1], float(np.hypot(*goal_r))])
        self.rows['subgoal'].append([sub_r[0], sub_r[1], float(np.hypot(*sub_r))])
        self.rows['cmd'].append(list(self._cmd))
        self.rows['t'].append(now.nanoseconds * 1e-9)

    # ── output ────────────────────────────────────────────────────────────────
    def _flush(self):
        if not self.rows['t']:
            self.get_logger().warn('no samples recorded — writing empty file')
        arrays = {k: np.asarray(v, dtype=np.float64 if k == 't' else np.float32)
                  for k, v in self.rows.items()}
        arrays['spawn_pose'] = np.asarray(self.anchor, dtype=np.float32)
        arrays['goal_world'] = self.goal_world.astype(np.float32)
        arrays['world_idx'] = np.asarray(
            [self.get_parameter('world_idx').value], dtype=np.int32)
        arrays['path'] = self.path.astype(np.float32)
        np.savez_compressed(self.out_file, **arrays)
        self.get_logger().info(
            f'wrote {self.out_file} ({len(self.rows["t"])} samples)')


def main():
    rclpy.init()
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            if not Path(node.out_file).exists():
                node._flush()
        except Exception as e:  # noqa: BLE001
            print(f'recorder flush failed: {e}', file=sys.stderr)
        node.destroy_node()


if __name__ == '__main__':
    main()
