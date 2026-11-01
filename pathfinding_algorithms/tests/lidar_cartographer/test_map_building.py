"""Tests that Cartographer builds an occupancy grid from LiDAR data on the real robot."""

import time

import pytest
import rclpy
from nav_msgs.msg import OccupancyGrid

from . import helpers

MAP_RECEIVE_TIMEOUT_S  = 30.0   # the first submap can take ~10 s
KNOWN_CELL_THRESHOLD   = 50
MIN_MAP_DIMENSION      = 10     # cells


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


def test_scan_relay_running(ros, cartographer):
    """scan_relay_node is up; without it MCU-stamped scans crash Cartographer."""
    assert helpers.is_scan_relay_running(), (
        'scan_relay_node is not running. '
        'Cartographer will receive incorrectly timestamped scans and crash. '
        'Launch via: ros2 launch dwa_avoidance dwa_with_cartographer.launch.py'
    )


def test_tf_relay_running(ros, cartographer):
    """tf_relay_node is up; without it RViz cannot show the robot position."""
    assert helpers.is_tf_relay_running(), (
        'tf_relay_node is not running. '
        'Robot position will not update in RViz — MCU-uptime TF stamps '
        'are incompatible with Cartographer wall-clock map→odom TF. '
        'Launch via: ros2 launch dwa_avoidance dwa_with_cartographer.launch.py'
    )


def test_map_topic_publishing(ros, cartographer):
    """/map receives an OccupancyGrid within MAP_RECEIVE_TIMEOUT_S."""
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
    """The grid has KNOWN_CELL_THRESHOLD known cells, so it is not an empty placeholder."""
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
    """The map spans at least MIN_MAP_DIMENSION cells each way."""
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
