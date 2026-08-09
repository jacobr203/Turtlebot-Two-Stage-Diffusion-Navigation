"""Pytest fixtures for the gazebo_sim suite.

Each test module gets a fresh, exclusive Gazebo instance (only one gzserver
can own the ROS graph at a time), so the world+stack fixtures are
module-scoped and start by killing whatever the previous module left.
"""

import pytest
import rclpy

from . import helpers


@pytest.fixture(scope='session')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope='module')
def lidar_sim(ros):
    helpers.kill_all_sim()
    helpers.start_gazebo_burger()
    helpers.start_lidar_stack()
    yield
    helpers.kill_all_sim()


@pytest.fixture(scope='module')
def camera_sim(ros):
    helpers.kill_all_sim()
    helpers.start_gazebo_brio()
    helpers.start_camera_stack()
    yield
    helpers.kill_all_sim()


@pytest.fixture(scope='module')
def nav2_sim(ros):
    helpers.kill_all_sim()
    helpers.start_gazebo_burger()
    helpers.start_nav2()
    yield
    helpers.kill_all_sim()
