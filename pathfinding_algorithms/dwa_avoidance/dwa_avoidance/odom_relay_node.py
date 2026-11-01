#!/usr/bin/env python3
"""Relay /odom to /odom_relayed with anchored wall-clock stamps.

The OpenCR stamps /odom with its uptime, which crashes Cartographer's sensor queue.
Anchoring at the first message keeps the MCU's own relative timing.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OdomRelayNode(Node):
    def __init__(self):
        super().__init__('odom_relay_node')
        self._pub = self.create_publisher(Odometry, '/odom_relayed', 10)
        self.create_subscription(Odometry, '/odom', self._relay, 10)
        self._wall_anchor_ns: int | None = None
        self._mcu_anchor_ns:  int | None = None
        self._last_ns: int = 0
        self.get_logger().info('odom_relay_node started — relaying /odom → /odom_relayed (anchored wall-clock)')

    def _relay(self, msg: Odometry):
        mcu_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        if self._wall_anchor_ns is None:
            self._wall_anchor_ns = self.get_clock().now().nanoseconds
            self._mcu_anchor_ns  = mcu_ns
            self.get_logger().info(
                f'Odom anchor set: wall={self._wall_anchor_ns} mcu={self._mcu_anchor_ns}'
            )

        corrected = self._wall_anchor_ns + (mcu_ns - self._mcu_anchor_ns)

        # identical MCU stamps must still increase
        if corrected <= self._last_ns:
            corrected = self._last_ns + 1
        self._last_ns = corrected

        msg.header.stamp.sec     = corrected // 1_000_000_000
        msg.header.stamp.nanosec = corrected  % 1_000_000_000
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
