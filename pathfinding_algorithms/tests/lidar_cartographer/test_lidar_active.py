"""Tests that the LDS-02 LiDAR produces valid scans; starts bringup over SSH if needed."""

import math
import threading
import time

import pytest
import rclpy
from sensor_msgs.msg import LaserScan

# LDS-02 hardware bounds
RANGE_MIN_M = 0.12
RANGE_MAX_M = 3.50
VALID_RATIO_THRESHOLD = 0.25  # low because open rooms have many inf rays
RECEIVE_TIMEOUT_S     = 10.0


def _wait_for_scan(node, qos, timeout: float) -> LaserScan | None:
    """Spin node until one LaserScan arrives or timeout expires."""
    received: list[LaserScan] = []

    sub = node.create_subscription(
        LaserScan, '/scan',
        lambda msg: received.append(msg),
        qos,
    )
    deadline = time.time() + timeout
    while not received and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_subscription(sub)
    return received[0] if received else None


def test_scan_topic_publishing(ros, bringup, scan_qos):
    """/scan receives a message within RECEIVE_TIMEOUT_S."""
    node = rclpy.create_node('test_lidar_active_publishing')
    try:
        msg = _wait_for_scan(node, scan_qos, RECEIVE_TIMEOUT_S)
        assert msg is not None, (
            f'/scan received no messages within {RECEIVE_TIMEOUT_S} s. '
            'Check that robot bringup is running and LDS-02 is connected.'
        )
    finally:
        node.destroy_node()


def test_scan_data_valid(ros, bringup, scan_qos):
    """At least VALID_RATIO_THRESHOLD of the rays are within LDS-02 bounds."""
    node = rclpy.create_node('test_lidar_active_data_valid')
    try:
        msg = _wait_for_scan(node, scan_qos, RECEIVE_TIMEOUT_S)
        assert msg is not None, f'/scan produced no message within {RECEIVE_TIMEOUT_S} s'

        in_range = [
            r for r in msg.ranges
            if math.isfinite(r) and RANGE_MIN_M <= r <= RANGE_MAX_M
        ]
        ratio = len(in_range) / len(msg.ranges) if msg.ranges else 0.0

        assert len(msg.ranges) > 0, '/scan message has empty ranges array'
        assert ratio >= VALID_RATIO_THRESHOLD, (
            f'Only {ratio:.0%} of scan rays are valid — expected ≥ {VALID_RATIO_THRESHOLD:.0%}. '
            f'LiDAR may be obscured or malfunctioning.'
        )
    finally:
        node.destroy_node()


def test_scan_coverage(ros, bringup, scan_qos):
    """The scan covers a full circle with enough rays for SLAM."""
    node = rclpy.create_node('test_lidar_active_coverage')
    try:
        msg = _wait_for_scan(node, scan_qos, RECEIVE_TIMEOUT_S)
        assert msg is not None, f'/scan produced no message within {RECEIVE_TIMEOUT_S} s'

        arc_deg = math.degrees(msg.angle_max - msg.angle_min)
        assert arc_deg >= 340, (
            f'Scan arc is only {arc_deg:.1f}° — expected ≥ 340° for full coverage. '
            'LDS-02 should produce a 360° scan.'
        )
        assert len(msg.ranges) >= 180, (
            f'Scan has only {len(msg.ranges)} rays — expected ≥ 180 for usable SLAM resolution.'
        )
    finally:
        node.destroy_node()
