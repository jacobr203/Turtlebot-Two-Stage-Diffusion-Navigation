"""
Test suite: Map Building

Verifies that Cartographer receives LiDAR data and actively builds an
occupancy grid. Starts the scan relay + Cartographer stack if not already
running (bringup must already be up — handled by the bringup fixture).

Tests
-----
test_map_topic_publishing  — /map is published within timeout
test_map_has_known_cells   — map contains at least some mapped (non-unknown) cells
test_map_dimensions        — map has reasonable size (robot has moved / scanned area)
test_scan_relay_running    — scan_relay_node is live (prerequisite for Cartographer timestamps)
"""

import time

import pytest
import rclpy
from nav_msgs.msg import OccupancyGrid

from . import helpers

MAP_RECEIVE_TIMEOUT_S  = 30.0   # Cartographer can take ~10 s to produce first submap
KNOWN_CELL_THRESHOLD   = 50     # minimum number of non-unknown cells to pass
MIN_MAP_DIMENSION      = 10     # map must be at least 10 cells wide and tall


def _wait_for_map(node, timeout: float) -> OccupancyGrid | None:
    """Spin node until one OccupancyGrid arrives on /map or timeout expires."""
    received: list[OccupancyGrid] = []

    sub = node.create_subscription(
        OccupancyGrid, '/map',
        lambda msg: received.append(msg),
        10,
    )
    deadline = time.time() + timeout
    while not received and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_subscription(sub)
    return received[0] if received else None


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_scan_relay_running(ros, cartographer):
    """
    scan_relay_node must be present on the ROS2 graph.
    If absent, Cartographer is receiving raw MCU-timestamped scans and will
    crash with a non-sorted queue error (see bug log 002).
    """
    assert helpers.is_scan_relay_running(), (
        'scan_relay_node is not running. '
        'Cartographer will receive incorrectly timestamped scans and crash. '
        'Launch via: ros2 launch dwa_avoidance dwa_with_cartographer.launch.py'
    )


def test_tf_relay_running(ros, cartographer):
    """
    tf_relay_node must be present on the ROS2 graph.
    If absent, the odom→base_footprint TF carries MCU uptime stamps and RViz
    cannot display the robot position (TF_OLD_DATA errors).
    """
    assert helpers.is_tf_relay_running(), (
        'tf_relay_node is not running. '
        'Robot position will not update in RViz — MCU-uptime TF stamps '
        'are incompatible with Cartographer wall-clock map→odom TF. '
        'Launch via: ros2 launch dwa_avoidance dwa_with_cartographer.launch.py'
    )


def test_map_topic_publishing(ros, cartographer):
    """
    /map must receive at least one OccupancyGrid message within
    MAP_RECEIVE_TIMEOUT_S seconds. Fails if cartographer_occupancy_grid_node
    is not running or Cartographer has not yet produced a submap.
    """
    node = rclpy.create_node('test_map_building_publishing')
    try:
        msg = _wait_for_map(node, MAP_RECEIVE_TIMEOUT_S)
        assert msg is not None, (
            f'/map received no messages within {MAP_RECEIVE_TIMEOUT_S} s. '
            'Check that cartographer_node and cartographer_occupancy_grid_node '
            'are both running: ros2 node list | grep cartographer'
        )
    finally:
        node.destroy_node()


def test_map_has_known_cells(ros, cartographer):
    """
    The occupancy grid must contain at least KNOWN_CELL_THRESHOLD cells with
    a non-unknown value (i.e. value ≠ -1). This confirms Cartographer is
    actively mapping rather than publishing an empty placeholder grid.
    """
    node = rclpy.create_node('test_map_building_known_cells')
    try:
        msg = _wait_for_map(node, MAP_RECEIVE_TIMEOUT_S)
        assert msg is not None, f'/map produced no message within {MAP_RECEIVE_TIMEOUT_S} s'

        known = [c for c in msg.data if c != -1]
        assert len(known) >= KNOWN_CELL_THRESHOLD, (
            f'Map contains only {len(known)} known cells — expected ≥ {KNOWN_CELL_THRESHOLD}. '
            'Cartographer may have started but not yet received enough scan data. '
            'Ensure the LiDAR is rotating and the robot has a clear field of view.'
        )
    finally:
        node.destroy_node()


def test_map_dimensions(ros, cartographer):
    """
    The map must be at least MIN_MAP_DIMENSION × MIN_MAP_DIMENSION cells.
    A single-cell or zero-size map indicates the occupancy grid node
    is running but Cartographer has not produced a real submap yet.
    """
    node = rclpy.create_node('test_map_building_dimensions')
    try:
        msg = _wait_for_map(node, MAP_RECEIVE_TIMEOUT_S)
        assert msg is not None, f'/map produced no message within {MAP_RECEIVE_TIMEOUT_S} s'

        w, h = msg.info.width, msg.info.height
        assert w >= MIN_MAP_DIMENSION, (
            f'Map width is {w} cells — expected ≥ {MIN_MAP_DIMENSION}.'
        )
        assert h >= MIN_MAP_DIMENSION, (
            f'Map height is {h} cells — expected ≥ {MIN_MAP_DIMENSION}.'
        )
    finally:
        node.destroy_node()
