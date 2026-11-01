"""Fixtures for the capstone_smoke suite, covering BARN worlds and the expert pipeline.

Each module gets a fresh exclusive Gazebo, as one gzserver owns the graph at a time.
"""

import sys
from pathlib import Path

import pytest

CAPSTONE = Path(__file__).resolve().parents[2] / 'capstone'
sys.path.insert(0, str(CAPSTONE))

from common.sim_harness import SimHarness, barn_world  # noqa: E402


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
