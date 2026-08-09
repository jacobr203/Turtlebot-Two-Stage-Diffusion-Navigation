"""
Session-scoped pytest fixtures for lidar_cartographer tests.

Fixtures here are shared across both test files in this folder.
They handle rclpy lifecycle, robot bringup, and cartographer startup
so individual test functions stay focused on assertions.
"""

import time
import pytest
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from . import helpers


# ── rclpy ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def ros():
    """Initialise rclpy once for the entire test session."""
    rclpy.init()
    yield
    rclpy.shutdown()


# ── Hardware / stack ──────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def bringup(ros):
    """
    Ensure robot bringup is running before any test uses it.
    Starts it via SSH if needed; does not stop it after tests (bringup
    should outlive the test session so the robot remains controllable).
    """
    helpers.ensure_bringup()
    # Give the LDS-02 a moment to start publishing
    time.sleep(3)
    yield


@pytest.fixture(scope='session')
def cartographer(bringup):
    """
    Ensure the Cartographer stack (scan relay + cartographer_node +
    occupancy_grid_node) is running. Starts it if needed and tears it
    down afterwards if this session was the one that started it.
    """
    started_here = helpers.ensure_cartographer()
    # Allow cartographer to initialise and receive first scans
    time.sleep(5)
    yield
    if started_here:
        helpers.kill_started()


# ── Convenience ───────────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def scan_qos():
    """BEST_EFFORT QoS matching the TurtleBot3 LDS-02 publisher."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
    )
