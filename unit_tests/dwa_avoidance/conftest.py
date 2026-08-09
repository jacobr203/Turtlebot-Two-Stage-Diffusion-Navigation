"""
Session-scoped pytest fixtures for dwa_avoidance tests.

Starts the FULL navigation stack (dwa_with_cartographer.launch.py) so tests
mirror exactly the sequence used during development:
  1. scan_relay_node  — wall-clock scan timestamps
  2. tf_relay_node    — wall-clock TF for robot position
  3. cartographer_node + occupancy_grid_node — SLAM map
  4. dwa_node         — obstacle avoidance + auto goal picking

RViz is suppressed (use_rviz:=false) since tests run headless.
"""

import sys
from pathlib import Path
import time
import pytest
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Share helpers with the lidar_cartographer suite
sys.path.insert(0, str(Path(__file__).parent.parent))
from lidar_cartographer import helpers


@pytest.fixture(scope='session')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope='session')
def bringup(ros):
    helpers.ensure_bringup()
    time.sleep(3)
    yield


@pytest.fixture(scope='session')
def full_stack(bringup):
    """
    Start the complete DWA + Cartographer stack if not already running.
    Tears it down after the test session if this fixture started it.
    """
    started_here = helpers.ensure_full_stack()
    # Allow the stack to initialise and receive first scans/odom
    time.sleep(8)
    yield
    if started_here:
        helpers.kill_started()


@pytest.fixture(scope='session')
def scan_qos():
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
    )
