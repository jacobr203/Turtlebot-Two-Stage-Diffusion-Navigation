"""
Tests for camera_obstacle_node's step-immune /scan_camera stamping.

Run with:
  cd ~/pathfinding_algorithms
  source install/setup.bash
  python3 -m pytest dwa_camera_avoidance/test/test_camera_stamps.py -v
"""

import time

import pytest
import rclpy
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock

from dwa_camera_avoidance.camera_obstacle_node import CameraObstacleNode, CV_AVAILABLE

pytestmark = pytest.mark.skipif(not CV_AVAILABLE,
                                reason='OpenCV / cv_bridge not available')


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture()
def node(ros):
    # stamp_lag 0: these tests verify the step-immune stamp mechanics; the
    # SLAM-fusion backdating is covered separately below
    n = CameraObstacleNode(parameter_overrides=[
        Parameter('stamp_lag', Parameter.Type.DOUBLE, 0.0)])
    yield n
    n.destroy_node()


def test_stamps_backdated_by_lag(ros):
    n = CameraObstacleNode(parameter_overrides=[
        Parameter('stamp_lag', Parameter.Type.DOUBLE, 0.45)])
    try:
        wall = n.get_clock().now().nanoseconds
        s = _ns(n._next_stamp())
        # stamped ~0.45 s behind the wall clock (within scheduling slop)
        assert wall - 500_000_000 <= s <= wall - 400_000_000
    finally:
        n.destroy_node()


def _ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def test_virtual_scan_no_detection_is_inf(node):
    """Columns with no detection must be inf (no return), not range_max —
    an r_max fill reads as a solid arc of obstacles at 3 m."""
    import math
    scan = node._build_virtual_scan([], node._next_stamp())
    assert all(math.isinf(r) for r in scan.ranges)
    empty = node._empty_scan()
    assert all(math.isinf(r) for r in empty.ranges)


def test_virtual_scan_detection_sets_single_ray(node):
    import math
    scan = node._build_virtual_scan([(1.0, 0.0)], node._next_stamp())
    finite = [r for r in scan.ranges if math.isfinite(r)]
    assert len(finite) == 1
    assert finite[0] == pytest.approx(1.0)


def test_stamps_strictly_increase(node):
    stamps = []
    for _ in range(20):
        stamps.append(_ns(node._next_stamp()))
    assert all(b > a for a, b in zip(stamps, stamps[1:]))


def test_stamps_track_wall_clock_when_ahead(node):
    wall_before = node.get_clock().now().nanoseconds
    s = _ns(node._next_stamp())
    wall_after = node.get_clock().now().nanoseconds
    assert wall_before <= s <= wall_after + 1_000_000


# ── Sim time ──────────────────────────────────────────────────────────────────

def _set_sim_clock(node, clock_pub, t_ns, timeout=5.0):
    """Publish /clock and spin the node until its clock reflects it."""
    msg = Clock()
    msg.clock.sec = t_ns // 1_000_000_000
    msg.clock.nanosec = t_ns % 1_000_000_000
    deadline = time.time() + timeout
    while node.get_clock().now().nanoseconds != t_ns:
        clock_pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
        assert time.time() < deadline, 'node clock never picked up /clock'


@pytest.fixture()
def sim_node(ros):
    n = CameraObstacleNode(
        parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
    helper = rclpy.create_node('clock_pub_helper')
    pub = helper.create_publisher(Clock, '/clock', 10)
    yield n, pub
    helper.destroy_node()
    n.destroy_node()


def test_sim_stamps_track_sim_clock(sim_node):
    node, clock_pub = sim_node
    _set_sim_clock(node, clock_pub, 100_000_000_000)      # t = 100 s
    assert _ns(node._next_stamp()) == 100_000_000_000
    _set_sim_clock(node, clock_pub, 100_500_000_000)      # t = 100.5 s
    assert _ns(node._next_stamp()) == 100_500_000_000


def test_sim_stamps_do_not_coast_ahead_of_sim_time(sim_node):
    """With sim time frozen, stamps must creep by 1 ns per call — never
    advance at the wall/monotonic rate (real-time factor independence)."""
    node, clock_pub = sim_node
    _set_sim_clock(node, clock_pub, 200_000_000_000)
    s1 = _ns(node._next_stamp())
    time.sleep(0.05)                                       # wall time passes...
    s2 = _ns(node._next_stamp())
    time.sleep(0.05)
    s3 = _ns(node._next_stamp())
    assert s1 < s2 < s3                                    # strictly increasing
    assert s3 - s1 <= 10                                   # ...but sim stamps don't


def test_stamps_survive_backward_clock_step(node):
    """Simulate the post-step state: the stamp line sits ahead of the wall
    clock. Subsequent stamps must advance at the monotonic rate, never
    fall back, and never repeat."""
    future_ns = node.get_clock().now().nanoseconds + 2_000_000_000
    node._last_stamp_ns = future_ns
    node._last_mono_ns = time.monotonic_ns()
    time.sleep(0.005)
    s1 = _ns(node._next_stamp())
    time.sleep(0.005)
    s2 = _ns(node._next_stamp())
    assert s1 > future_ns
    assert s2 > s1
    # advanced by roughly the elapsed monotonic time, not pinned +1 ns
    assert s2 - s1 >= 3_000_000
