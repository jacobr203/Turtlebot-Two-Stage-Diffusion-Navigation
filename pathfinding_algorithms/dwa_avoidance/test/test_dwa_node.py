"""
Tests that DWANode starts up correctly and its core interfaces are intact.

Run with:
  cd ~/pathfinding_algorithms
  source install/setup.bash
  pytest dwa_avoidance/test/test_dwa_node.py -v
"""

import math
import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from dwa_avoidance.dwa_node import DWANode, _yaw_from_quat


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture()
def node(ros):
    n = DWANode()
    yield n
    n.destroy_node()


# ── Helper ────────────────────────────────────────────────────────────────────

def _spin_for(node, seconds: float):
    """Spin node in a background thread for `seconds`, then stop."""
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline = time.time() + seconds
    t = threading.Thread(target=lambda: executor.spin_until_future_complete(
        rclpy.task.Future(), timeout_sec=seconds), daemon=True)
    t.start()
    t.join(timeout=seconds + 1.0)
    executor.shutdown(timeout_sec=0)


# ── Node lifecycle ────────────────────────────────────────────────────────────

def test_node_starts(node):
    """Node must initialise without raising."""
    assert node is not None


def test_node_name(node):
    assert node.get_name() == 'dwa_node'


# ── Publishers ────────────────────────────────────────────────────────────────

def test_cmd_vel_publisher_exists(node):
    topics = {name for name, _ in node.get_publisher_names_and_types_by_node(
        'dwa_node', '')}
    assert '/cmd_vel' in topics


# ── Subscriptions ─────────────────────────────────────────────────────────────

def test_subscriptions_exist(node):
    topics = {name for name, _ in node.get_subscriber_names_and_types_by_node(
        'dwa_node', '')}
    assert '/scan'      in topics
    assert '/odom'      in topics
    assert '/goal_pose' in topics


# ── Default parameters ────────────────────────────────────────────────────────

@pytest.mark.parametrize('name,expected', [
    # Sourced from e-Manual and turtlebot3_navigation2/param/humble/burger.yaml
    ('max_speed',          0.22),
    ('min_speed',         -0.05),
    ('max_yaw_rate',       2.84),
    ('max_accel',          2.50),
    ('max_delta_yaw_rate', 3.2),
    ('robot_radius',       0.15),
    ('obstacle_margin',    0.05),
    ('dt',                 0.1),
    ('predict_time',       2.0),
    ('w_heading',          0.20),
    ('w_clearance',        1.00),
    ('w_speed',            0.50),
    ('goal_tolerance',     0.25),
])
def test_default_parameter(node, name, expected):
    assert node.get_parameter(name).value == pytest.approx(expected)


# ── Yaw extraction helper ─────────────────────────────────────────────────────

class _Q:
    """Minimal quaternion stub."""
    def __init__(self, x, y, z, w):
        self.x, self.y, self.z, self.w = x, y, z, w


def test_yaw_from_quat_identity():
    yaw = _yaw_from_quat(_Q(0, 0, 0, 1))
    assert yaw == pytest.approx(0.0)


def test_yaw_from_quat_90_degrees():
    # 90° rotation around z: q = (0, 0, sin(π/4), cos(π/4))
    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    yaw = _yaw_from_quat(_Q(0, 0, s, c))
    assert yaw == pytest.approx(math.pi / 2, abs=1e-6)


def test_yaw_from_quat_180_degrees():
    yaw = _yaw_from_quat(_Q(0, 0, 1, 0))
    assert abs(yaw) == pytest.approx(math.pi, abs=1e-6)


# ── Goal callback ─────────────────────────────────────────────────────────────

def test_goal_callback_sets_goal(node):
    msg = PoseStamped()
    msg.pose.position.x = 3.0
    msg.pose.position.y = 4.0
    node._goal_cb(msg)
    assert node._goal == pytest.approx((3.0, 4.0))
    assert not node._at_goal


# ── Odometry callback ─────────────────────────────────────────────────────────

def test_odom_callback_updates_state(node):
    msg = Odometry()
    msg.pose.pose.position.x = 1.0
    msg.pose.pose.position.y = 2.0
    msg.pose.pose.orientation.w = 1.0  # identity → yaw = 0
    msg.twist.twist.linear.x  = 0.1
    msg.twist.twist.angular.z = 0.05

    node._odom_cb(msg)

    assert node._state[0] == pytest.approx(1.0)
    assert node._state[1] == pytest.approx(2.0)
    assert node._state[2] == pytest.approx(0.0)
    assert node._state[3] == pytest.approx(0.1)
    assert node._state[4] == pytest.approx(0.05)
    assert node._odom_ready


# ── Scan callback ─────────────────────────────────────────────────────────────

def test_scan_callback_populates_obstacles(node):
    # Place robot at origin facing 0°
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]

    msg = LaserScan()
    msg.angle_min      = 0.0
    msg.angle_max      = 0.0
    msg.angle_increment = 0.0
    msg.range_min      = 0.1
    msg.range_max      = 5.0
    msg.ranges         = [1.0]   # one obstacle 1 m straight ahead

    node._scan_cb(msg)

    assert len(node._obstacles) == 1
    ox, oy = node._obstacles[0]
    assert ox == pytest.approx(1.0, abs=1e-6)
    assert oy == pytest.approx(0.0, abs=1e-6)


def test_scan_callback_drops_out_of_range(node):
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]

    msg = LaserScan()
    msg.angle_min       = 0.0
    msg.angle_max       = 0.0
    msg.angle_increment = 0.0
    msg.range_min       = 0.1
    msg.range_max       = 3.5
    msg.ranges          = [float('inf'), 0.05, 5.0]  # inf, below min, above max

    node._scan_cb(msg)
    assert node._obstacles == []


# ── DWA scores ────────────────────────────────────────────────────────────────

def test_clearance_score_saturates_at_one(node):
    assert node._clearance_score(2.0) == pytest.approx(1.0)
    assert node._clearance_score(10.0) == pytest.approx(1.0)


def test_clearance_score_zero_at_origin(node):
    assert node._clearance_score(0.0) == pytest.approx(0.0)


def test_speed_score_max_at_max_speed(node):
    max_v = node.get_parameter('max_speed').value
    assert node._speed_score(max_v, max_v) == pytest.approx(1.0)


def test_speed_score_zero_for_reverse(node):
    assert node._speed_score(-0.1, 0.22) == pytest.approx(0.0)


def test_heading_score_on_goal(node):
    """Final trajectory point at the goal itself should score ~1."""
    node._goal = (5.0, 0.0)
    node._at_goal = False
    # Trajectory ending at (5, 0) facing east (yaw=0)
    traj = [(5.0, 0.0, 0.0)]
    assert node._heading_score(traj) == pytest.approx(1.0, abs=1e-6)


def test_heading_score_facing_away(node):
    """Final trajectory point facing directly away from goal should score ~0."""
    node._goal = (5.0, 0.0)
    node._at_goal = False
    # At (0,0) facing west (yaw=π), goal is east — maximum error
    traj = [(0.0, 0.0, math.pi)]
    assert node._heading_score(traj) == pytest.approx(0.0, abs=1e-6)


# ── DWA output stays within limits ───────────────────────────────────────────

def test_dwa_output_within_limits(node):
    """With obstacles present, DWA must return velocities inside kinematic bounds."""
    # Populate a clear environment (obstacles far away)
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]
    node._odom_ready = True
    # Place obstacles in a ring at 3 m — well clear of the robot
    node._obstacles = [
        (3.0 * math.cos(a), 3.0 * math.sin(a))
        for a in [i * math.pi / 8 for i in range(16)]
    ]

    v, w = node._dwa()

    max_v = node.get_parameter('max_speed').value
    min_v = node.get_parameter('min_speed').value
    max_w = node.get_parameter('max_yaw_rate').value

    assert min_v <= v <= max_v
    assert -max_w <= w <= max_w
