"""
Gazebo sim: LiDAR DWA + Cartographer smoke tests.

Verifies the simulated LiDAR publishes valid scans and Cartographer builds
a growing map while the DWA node wanders — the sim equivalent of the
lidar_cartographer suite, with no robot hardware involved.
"""

import time

import pytest
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy)
from sensor_msgs.msg import LaserScan

SCAN_TIMEOUT_S = 15.0
MAP_TIMEOUT_S = 60.0
MAP_GROWTH_WINDOW_S = 20.0

scan_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
map_qos = QoSProfile(depth=1,
                     reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def _collect(msg_type, topic, qos, timeout, count=1):
    node = rclpy.create_node('_lidar_sim_collect')
    received = []
    node.create_subscription(msg_type, topic, received.append, qos)
    deadline = time.time() + timeout
    while len(received) < count and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    return received


def _known_cells(grid: OccupancyGrid) -> int:
    return sum(1 for v in grid.data if v >= 0)


def test_scan_publishing(lidar_sim):
    scans = _collect(LaserScan, '/scan', scan_qos, SCAN_TIMEOUT_S, count=3)
    assert len(scans) >= 3, 'no /scan data from Gazebo'


def test_scan_geometry(lidar_sim):
    scan = _collect(LaserScan, '/scan', scan_qos, SCAN_TIMEOUT_S)[0]
    assert len(scan.ranges) == 360
    assert scan.header.frame_id == 'base_scan'
    assert scan.range_max == pytest.approx(3.5)


def test_map_publishing(lidar_sim):
    maps = _collect(OccupancyGrid, '/map', map_qos, MAP_TIMEOUT_S)
    assert maps, 'Cartographer never published /map'
    assert _known_cells(maps[0]) > 0


def test_map_grows_while_wandering(lidar_sim):
    first = _collect(OccupancyGrid, '/map', map_qos, MAP_TIMEOUT_S)
    assert first, 'no initial map'
    before = _known_cells(first[0])
    time.sleep(MAP_GROWTH_WINDOW_S)
    second = _collect(OccupancyGrid, '/map', map_qos, MAP_TIMEOUT_S)
    assert second, 'no second map sample'
    after = _known_cells(second[-1])
    assert after > before, (
        f'map did not grow: {before} -> {after} known cells '
        f'over {MAP_GROWTH_WINDOW_S}s')
