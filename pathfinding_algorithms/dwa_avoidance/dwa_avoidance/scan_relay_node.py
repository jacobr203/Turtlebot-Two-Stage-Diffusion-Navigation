#!/usr/bin/env python3
"""Relay /scan to /scan_relayed with strictly increasing wall-clock stamps.

OpenCR uptime stamps crash Cartographer's sensor queue. The relayed topic is
RELIABLE so RViz can overlay it, as the LDS-02 publishes BEST_EFFORT.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan


class ScanRelayNode(Node):
    def __init__(self):
        super().__init__('scan_relay_node')

        in_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pub = self.create_publisher(LaserScan, '/scan_relayed', 10)
        self.create_subscription(LaserScan, '/scan', self._relay, in_qos)

        self._last_end_ns: int = 0

        # Gazebo stamps are exact; restamping to arrival time rotates scan
        # geometry during fast spins and shears Cartographer submaps.
        self._passthrough = bool(self.get_parameter('use_sim_time').value)

        mode = 'pass-through (sim stamps)' if self._passthrough \
            else 'wall-clock restamp'
        self.get_logger().info(f'Scan relay started: /scan → /scan_relayed ({mode})')

    def _relay(self, msg: LaserScan):
        if self._passthrough:
            self._pub.publish(msg)
            return
        now_ns = self.get_clock().now().nanoseconds

        # Cartographer subdivides each scan across its sweep and drops data if
        # subdivision times go backwards, so wait for the previous sweep to end.
        if now_ns <= self._last_end_ns:
            now_ns = self._last_end_ns + 1
        sweep_ns = int(msg.time_increment * max(len(msg.ranges) - 1, 0) * 1e9)
        self._last_end_ns = now_ns + sweep_ns

        msg.header.stamp.sec     = now_ns // 1_000_000_000
        msg.header.stamp.nanosec = now_ns  % 1_000_000_000
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
