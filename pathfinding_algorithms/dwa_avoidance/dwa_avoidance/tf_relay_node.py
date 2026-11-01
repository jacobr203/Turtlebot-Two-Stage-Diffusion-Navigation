#!/usr/bin/env python3
"""Republish stale /tf transforms with wall-clock stamps.

The OpenCR stamps TF with its uptime while Cartographer uses wall-clock time, so RViz
rejects the MCU transforms as TF_OLD_DATA. Republished stamps are fresh, so no feedback loop.
"""

import time

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

STALE_THRESHOLD_S = 60.0


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
        """Wall-clock stamp immune to WSL2's backward steps (~1 s every ~30 s).

        Follows the wall clock while ahead, else advances at the monotonic rate.
        Sim time never steps back and diverges from monotonic time, so it only
        needs a strictly-increasing guard.
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
