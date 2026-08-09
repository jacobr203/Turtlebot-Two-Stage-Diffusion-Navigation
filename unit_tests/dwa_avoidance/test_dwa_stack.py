"""
Test suite: DWA Full Navigation Stack

Verifies that all nodes in the live navigation stack are running and that
the key topics are publishing valid data. This mirrors the exact launch
sequence used during development (dwa_with_cartographer.launch.py).

Node presence tests confirm the full stack is healthy before any topic
or data tests run.

Tests
-----
Node presence (matches live launch order):
  test_scan_relay_running   — scan_relay_node on ROS2 graph
  test_tf_relay_running     — tf_relay_node on ROS2 graph
  test_cartographer_running — cartographer_node on ROS2 graph
  test_dwa_running          — dwa_node on ROS2 graph

Topic / data:
  test_scan_relayed_publishing — /scan_relayed receives messages (relay working)
  test_cmd_vel_publishing      — /cmd_vel receives messages (DWA is commanding)
  test_cmd_vel_within_limits   — velocity commands stay within TurtleBot3 bounds
  test_map_publishing          — /map receives messages (Cartographer mapping)
  test_map_has_known_cells     — map contains non-unknown cells (SLAM is active)
  test_dwa_not_permanently_stuck — DWA produces forward motion within timeout
"""

import math
import time

import pytest
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from lidar_cartographer import helpers

# TurtleBot3 Burger limits (from e-Manual)
MAX_LINEAR_M_S  = 0.22
MAX_ANGULAR_RAD_S = 2.84

TOPIC_TIMEOUT_S  = 15.0
MAP_TIMEOUT_S    = 30.0
MOTION_TIMEOUT_S = 60.0   # DWA should find a forward path within 60 s
KNOWN_CELL_MIN   = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect(node, msg_type, topic, qos, timeout: float, count: int = 1) -> list:
    received = []
    sub = node.create_subscription(msg_type, topic, lambda m: received.append(m), qos)
    deadline = time.time() + timeout
    while len(received) < count and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return received


# ── Node presence ─────────────────────────────────────────────────────────────

def test_scan_relay_running(ros, full_stack):
    """scan_relay_node must be on the ROS2 graph (bug 002 / bug 009 guard)."""
    assert helpers.is_scan_relay_running(), (
        'scan_relay_node is not running — Cartographer will receive '
        'MCU-timestamped scans and crash (see bug logs 002, 009).'
    )


def test_tf_relay_running(ros, full_stack):
    """tf_relay_node must be on the ROS2 graph (robot position in RViz)."""
    assert helpers.is_tf_relay_running(), (
        'tf_relay_node is not running — odom→base_footprint TF will carry '
        'MCU uptime stamps and robot position will not display in RViz.'
    )


def test_cartographer_running(ros, full_stack):
    """cartographer_node must be on the ROS2 graph."""
    assert helpers.is_cartographer_running(), (
        'cartographer_node is not running — no SLAM map will be produced.'
    )


def test_dwa_running(ros, full_stack):
    """dwa_node must be on the ROS2 graph."""
    assert helpers.is_dwa_running(), (
        'dwa_node is not running — robot will not navigate or avoid obstacles.'
    )


# ── Topic / data ──────────────────────────────────────────────────────────────

def test_scan_relayed_publishing(ros, full_stack, scan_qos):
    """
    /scan_relayed must receive messages — confirms scan_relay_node is
    forwarding LiDAR data with corrected wall-clock timestamps.
    """
    node = rclpy.create_node('test_dwa_scan_relayed')
    try:
        msgs = _collect(node, LaserScan, '/scan_relayed', 10, TOPIC_TIMEOUT_S)
        assert msgs, (
            f'/scan_relayed received no messages within {TOPIC_TIMEOUT_S} s. '
            'Check scan_relay_node is running and /scan is active.'
        )
        msg = msgs[0]
        now_s = node.get_clock().now().nanoseconds * 1e-9
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Stamp must be wall-clock (within 5 s of now), not MCU uptime (~11000 s)
        assert abs(now_s - stamp_s) < 5.0, (
            f'/scan_relayed stamp is {stamp_s:.0f} s but wall clock is {now_s:.0f} s. '
            'scan_relay_node is not correcting timestamps (bug 002).'
        )
    finally:
        node.destroy_node()


def test_cmd_vel_publishing(ros, full_stack):
    """
    /cmd_vel must receive messages — confirms dwa_node is running its
    control loop and issuing velocity commands.
    """
    node = rclpy.create_node('test_dwa_cmd_vel_publishing')
    try:
        msgs = _collect(node, Twist, '/cmd_vel', 10, TOPIC_TIMEOUT_S)
        assert msgs, (
            f'/cmd_vel received no messages within {TOPIC_TIMEOUT_S} s. '
            'dwa_node may not have received scan or odom data yet.'
        )
    finally:
        node.destroy_node()


def test_cmd_vel_within_limits(ros, full_stack):
    """
    All /cmd_vel commands must be within TurtleBot3 Burger hardware limits.
    Catches DWA parameter misconfiguration that could damage motors.
    """
    node = rclpy.create_node('test_dwa_cmd_vel_limits')
    try:
        msgs = _collect(node, Twist, '/cmd_vel', 10, TOPIC_TIMEOUT_S, count=10)
        assert msgs, f'/cmd_vel produced no messages within {TOPIC_TIMEOUT_S} s'

        for msg in msgs:
            v = msg.linear.x
            w = msg.angular.z
            assert abs(v) <= MAX_LINEAR_M_S + 0.01, (
                f'linear.x = {v:.3f} m/s exceeds TurtleBot3 limit of {MAX_LINEAR_M_S} m/s'
            )
            assert abs(w) <= MAX_ANGULAR_RAD_S + 0.01, (
                f'angular.z = {w:.3f} rad/s exceeds TurtleBot3 limit of {MAX_ANGULAR_RAD_S} rad/s'
            )
    finally:
        node.destroy_node()


def test_map_publishing(ros, full_stack):
    """
    /map must receive an OccupancyGrid — confirms Cartographer is running
    and the occupancy_grid_node is publishing.
    """
    node = rclpy.create_node('test_dwa_map_publishing')
    try:
        msgs = _collect(node, OccupancyGrid, '/map', 10, MAP_TIMEOUT_S)
        assert msgs, (
            f'/map received no messages within {MAP_TIMEOUT_S} s. '
            'Check cartographer_node and cartographer_occupancy_grid_node.'
        )
    finally:
        node.destroy_node()


def test_map_has_known_cells(ros, full_stack):
    """
    The occupancy grid must contain at least KNOWN_CELL_MIN non-unknown cells,
    confirming Cartographer is actively mapping from LiDAR data.
    """
    node = rclpy.create_node('test_dwa_map_known_cells')
    try:
        msgs = _collect(node, OccupancyGrid, '/map', 10, MAP_TIMEOUT_S)
        assert msgs, f'/map produced no message within {MAP_TIMEOUT_S} s'
        known = [c for c in msgs[0].data if c != -1]
        assert len(known) >= KNOWN_CELL_MIN, (
            f'Map has only {len(known)} known cells — expected ≥ {KNOWN_CELL_MIN}. '
            'Cartographer may not be receiving valid scan data.'
        )
    finally:
        node.destroy_node()


def test_dwa_not_permanently_stuck(ros, full_stack):
    """
    DWA must produce at least one cmd_vel with non-zero linear velocity within
    MOTION_TIMEOUT_S seconds, confirming it can find a safe forward trajectory
    and is not permanently locked in escape-rotation mode (bug 010).

    This test may be skipped if the robot is in a fully enclosed space with
    no forward gap wider than MIN_CLEAR_M = 0.4 m.
    """
    node = rclpy.create_node('test_dwa_not_permanently_stuck')
    forward_seen = False
    try:
        deadline = time.time() + MOTION_TIMEOUT_S
        received = []
        sub = node.create_subscription(Twist, '/cmd_vel',
                                        lambda m: received.append(m), 10)
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            for msg in received:
                if msg.linear.x > 0.01:
                    forward_seen = True
                    break
            received.clear()
            if forward_seen:
                break
        node.destroy_subscription(sub)
    finally:
        node.destroy_node()

    assert forward_seen, (
        f'DWA never produced forward motion (linear.x > 0.01) within '
        f'{MOTION_TIMEOUT_S} s — robot may be permanently stuck in escape mode '
        f'(bug 010). Check that the robot has at least 0.4 m of clearance in '
        f'some direction.'
    )
