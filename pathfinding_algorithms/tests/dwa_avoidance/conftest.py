"""Session-scoped fixtures that start the full DWA + Cartographer stack, without RViz."""

import sys
from pathlib import Path
import time
import pytest
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

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
    """Start the DWA + Cartographer stack if needed, tearing it down if started here."""
    started_here = helpers.ensure_full_stack()
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
