"""Session-scoped fixtures for the lidar_cartographer tests: rclpy, bringup, Cartographer."""

import time
import pytest
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from . import helpers


@pytest.fixture(scope='session')
def ros():
    """Initialise rclpy once for the session."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope='session')
def bringup(ros):
    """Ensure robot bringup is running; it outlives the session so the robot stays controllable."""
    helpers.ensure_bringup()
    time.sleep(3)
    yield


@pytest.fixture(scope='session')
def cartographer(bringup):
    """Ensure the Cartographer stack is running, tearing it down if this session started it."""
    started_here = helpers.ensure_cartographer()
    time.sleep(5)
    yield
    if started_here:
        helpers.kill_started()


@pytest.fixture(scope='session')
def scan_qos():
    """BEST_EFFORT QoS matching the TurtleBot3 LDS-02 publisher."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
    )
