"""Fixtures for the capstone_smoke suite.

Verifies the BARN-in-Gazebo layer (worlds load, bumper model detects
collisions) and the expert pipeline (one full episode). Each module gets a
fresh exclusive Gazebo (one gzserver owns the graph at a time), following
the gazebo_sim suite's pattern.
"""

import sys
from pathlib import Path

import pytest

CAPSTONE = Path('/home/turtlebot/pathfinding_algorithms/capstone')
sys.path.insert(0, str(CAPSTONE))

from evaluation.sim_helpers import SimHarness, barn_world  # noqa: E402


@pytest.fixture(scope='session')
def harness():
    h = SimHarness(worker=0, log_dir='/tmp/capstone_smoke')
    yield h
    h.close()


@pytest.fixture(scope='module')
def barn_sim(harness):
    """World 0 (fast variant: identical physics, uncapped RTF) + bumper burger."""
    harness.kill_all()
    harness.start_world(barn_world(0, fast=True), headless=True)
    yield harness
    harness.kill_all()
