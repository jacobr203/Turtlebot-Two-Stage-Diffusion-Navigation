#!/usr/bin/env python3
"""
Scan timestamp relay node.

The TurtleBot3 OpenCR board stamps LaserScan messages with the MCU's
monotonic uptime rather than the system wall clock.  When Cartographer
receives those scans it crashes with a non-sorted queue error.

This relay replaces each scan's timestamp with the current ROS wall-clock
time and republishes to /scan_relayed.  Strict monotonicity is enforced so
Cartographer's ordered_multi_queue never sees a backwards timestamp.

The relayed topic also uses RELIABLE QoS, allowing RViz to display the
laser overlay even though the LDS-02 publishes /scan as BEST_EFFORT.
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

        # Sim scans carry perfect Gazebo stamps — restamping them to
        # ARRIVAL time (transport latency + load jitter, up to ~200 ms)
        # rotates scan geometry during fast spins and sheared Cartographer
        # submaps (2026-07-23, world 613 seam). Pass through untouched in
        # sim; restamp only on the real robot (OpenCR uptime stamps).
        self._passthrough = bool(self.get_parameter('use_sim_time').value)

        mode = 'pass-through (sim stamps)' if self._passthrough \
            else 'wall-clock restamp'
        self.get_logger().info(f'Scan relay started: /scan → /scan_relayed ({mode})')

    def _relay(self, msg: LaserScan):
        if self._passthrough:
            self._pub.publish(msg)
            return
        now_ns = self.get_clock().now().nanoseconds

        # Cartographer subdivides each scan across its sweep duration
        # (stamp + i * time_increment), so monotonic stamps are not enough:
        # a new stamp inside the previous scan's sweep window makes subdivision
        # times go backwards and Cartographer drops the data ("Ignored
        # subdivision..."). Never stamp a scan before the previous sweep ended.
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
