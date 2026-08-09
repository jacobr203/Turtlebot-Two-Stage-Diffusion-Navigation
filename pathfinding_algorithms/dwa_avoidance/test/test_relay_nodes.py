"""
Tests for the timestamp behavior of scan_relay_node and tf_relay_node.

Both relays restamp robot data with the PC wall clock. The WSL2 wall clock
steps backwards ~1 s every ~30 s under time sync, so the relays must emit
strictly increasing stamps regardless of what the clock does (Cartographer
drops scans whose subdivision times are not increasing).

Run with:
  cd ~/pathfinding_algorithms
  source install/setup.bash
  python3 -m pytest dwa_avoidance/test/test_relay_nodes.py -v
"""

import time

import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

from dwa_avoidance.scan_relay_node import ScanRelayNode
from dwa_avoidance.tf_relay_node import TFRelayNode


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


class _Recorder:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


def _ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _lidar_scan(n=200, time_increment=0.0005):
    msg = LaserScan()
    msg.time_increment = time_increment
    msg.range_min = 0.12
    msg.range_max = 3.5
    msg.ranges = [1.0] * n
    return msg


def _stale_tf():
    msg = TFMessage()
    t = TransformStamped()
    t.header.stamp.sec = 100          # MCU-uptime-style stamp, very stale
    t.header.frame_id = 'odom'
    t.child_frame_id = 'base_footprint'
    msg.transforms = [t]
    return msg


# ── scan_relay_node ───────────────────────────────────────────────────────────

def test_scan_relay_spaces_stamps_by_sweep(ros):
    """Back-to-back scans must be stamped at least one sweep apart so
    Cartographer's per-scan subdivision windows never overlap."""
    node = ScanRelayNode()
    try:
        rec = _Recorder()
        node._pub = rec
        node._relay(_lidar_scan())
        node._relay(_lidar_scan())
        s1, s2 = (_ns(m.header.stamp) for m in rec.msgs)
        sweep_ns = int(0.0005 * 199 * 1e9)
        assert s2 - s1 >= sweep_ns
    finally:
        node.destroy_node()


def test_scan_relay_survives_backward_clock_step(ros):
    """Simulate a post-backward-step state (stamp line ahead of wall clock):
    the next stamp must still move forward."""
    node = ScanRelayNode()
    try:
        rec = _Recorder()
        node._pub = rec
        future_ns = node.get_clock().now().nanoseconds + 2_000_000_000
        node._last_end_ns = future_ns
        node._relay(_lidar_scan())
        assert _ns(rec.msgs[-1].header.stamp) > future_ns
    finally:
        node.destroy_node()


# ── tf_relay_node ─────────────────────────────────────────────────────────────

def test_tf_relay_restamps_stale_transform(ros):
    node = TFRelayNode()
    try:
        rec = _Recorder()
        node._pub = rec
        node._relay(_stale_tf())
        assert len(rec.msgs) == 1
        wall = node.get_clock().now().nanoseconds
        assert abs(wall - _ns(rec.msgs[0].transforms[0].header.stamp)) < 2e9
    finally:
        node.destroy_node()


def test_tf_relay_stamps_strictly_increase(ros):
    node = TFRelayNode()
    try:
        rec = _Recorder()
        node._pub = rec
        for _ in range(5):
            node._relay(_stale_tf())
            time.sleep(0.002)
        stamps = [_ns(m.transforms[0].header.stamp) for m in rec.msgs]
        assert all(b > a for a, b in zip(stamps, stamps[1:]))
    finally:
        node.destroy_node()


def test_tf_relay_survives_backward_clock_step(ros):
    node = TFRelayNode()
    try:
        rec = _Recorder()
        node._pub = rec
        future_ns = node.get_clock().now().nanoseconds + 2_000_000_000
        node._last_stamp_ns = future_ns
        node._last_mono_ns = time.monotonic_ns()
        time.sleep(0.002)
        node._relay(_stale_tf())
        assert _ns(rec.msgs[-1].transforms[0].header.stamp) > future_ns
    finally:
        node.destroy_node()


def test_tf_relay_sim_time_stamps_track_sim_clock(ros):
    """Under use_sim_time, _steady_stamp must follow the sim clock exactly
    (no monotonic coast) and stay strictly increasing when sim time freezes."""
    node = TFRelayNode(
        parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
    helper = rclpy.create_node('tf_clock_pub_helper')
    pub = helper.create_publisher(Clock, '/clock', 10)
    try:
        msg = Clock()
        msg.clock.sec = 500
        deadline = time.time() + 5.0
        while node.get_clock().now().nanoseconds != 500_000_000_000:
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            assert time.time() < deadline, 'node clock never picked up /clock'

        s1 = _ns(node._steady_stamp())
        assert s1 == 500_000_000_000
        time.sleep(0.05)                        # wall time passes, sim frozen
        s2 = _ns(node._steady_stamp())
        assert s1 < s2 and s2 - s1 <= 10        # +1 ns guard, no wall coast
    finally:
        helper.destroy_node()
        node.destroy_node()


def test_tf_relay_leaves_fresh_transforms_alone(ros):
    """Transforms already carrying near-current stamps must not be relayed
    (prevents a feedback loop with its own output)."""
    node = TFRelayNode()
    try:
        rec = _Recorder()
        node._pub = rec
        msg = _stale_tf()
        msg.transforms[0].header.stamp = node.get_clock().now().to_msg()
        node._relay(msg)
        assert rec.msgs == []
    finally:
        node.destroy_node()
