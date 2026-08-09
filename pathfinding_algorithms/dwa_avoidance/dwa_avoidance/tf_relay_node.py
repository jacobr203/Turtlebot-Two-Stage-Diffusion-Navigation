#!/usr/bin/env python3
"""
tf_relay_node — wall-clock timestamp relay for /tf.

The TurtleBot3 OpenCR MCU stamps every TF transform (odom→base_footprint,
base_footprint→base_link, etc.) with its own uptime clock (~8000 s from boot).
Cartographer publishes the map→odom transform with wall-clock time (~1.78e9 s).
Because both transforms land in the same TF tree, RViz rejects the MCU-stamped
ones as TF_OLD_DATA — the robot model and laser scan overlay stop updating.

This node subscribes to /tf, detects transforms whose timestamp is more than
STALE_THRESHOLD seconds behind the current wall-clock time, replaces the stamp,
and republishes on /tf.  Freshly republished transforms have near-current
timestamps so they will not be re-processed on the next subscription callback
(no feedback loop).

Static transforms (/tf_static) are not touched — they carry stamp=0 which
means "always valid" and need no correction.
"""

import time

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

STALE_THRESHOLD_S = 60.0   # transforms older than this (relative to wall clock) are fixed


class TFRelayNode(Node):
    def __init__(self, **node_kwargs):
        super().__init__('tf_relay_node', **node_kwargs)
        self._pub = self.create_publisher(TFMessage, '/tf', 100)
        self.create_subscription(TFMessage, '/tf', self._relay, 100)
        self._last_stamp_ns = 0
        self._last_mono_ns  = 0
        self._use_sim_time = bool(self.get_parameter('use_sim_time').value)
        self.get_logger().info(
            f'tf_relay_node started — fixing MCU-uptime TF stamps '
            f'(threshold: {STALE_THRESHOLD_S} s)'
        )

    def _steady_stamp(self) -> TimeMsg:
        """Step-immune wall-clock stamp (see camera_obstacle_node._next_stamp).

        WSL2's wall clock steps backwards ~1 s every ~30 s under time sync;
        follow the wall clock while it is ahead, otherwise advance at the
        monotonic clock's rate so the TF timeline never runs backwards and
        stays on the same envelope as the relayed scan stamps.

        Under sim time, skip the monotonic coast (wall-monotonic elapsed
        diverges from sim time at real-time factor != 1) — the sim clock
        never steps backwards, so a strictly-monotonic guard suffices.
        """
        wall_ns = self.get_clock().now().nanoseconds
        if self._use_sim_time:
            stamp_ns = max(wall_ns, self._last_stamp_ns + 1)
            self._last_stamp_ns = stamp_ns
            return TimeMsg(sec=stamp_ns // 1_000_000_000,
                           nanosec=stamp_ns % 1_000_000_000)
        mono_ns = time.monotonic_ns()
        if self._last_mono_ns == 0:
            stamp_ns = wall_ns
        else:
            elapsed = max(mono_ns - self._last_mono_ns, 1)
            stamp_ns = max(wall_ns, self._last_stamp_ns + elapsed)
        self._last_stamp_ns = stamp_ns
        self._last_mono_ns = mono_ns
        return TimeMsg(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)

    def _relay(self, msg: TFMessage):
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        fixed = []
        stamp = None
        for t in msg.transforms:
            ts = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
            if now_sec - ts > STALE_THRESHOLD_S:
                if stamp is None:
                    stamp = self._steady_stamp()
                t.header.stamp = stamp
                fixed.append(t)

        if fixed:
            out = TFMessage()
            out.transforms = fixed
            self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TFRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
