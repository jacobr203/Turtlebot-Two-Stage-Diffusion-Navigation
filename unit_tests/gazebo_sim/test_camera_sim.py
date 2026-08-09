"""
Gazebo sim: camera DWA smoke tests with the turtlebot3_burger_brio model.

Verifies the simulated pinhole camera feeds camera_obstacle_node
(/camera/image_raw/compressed → /scan_camera) and that /scan_camera stamps
respect sim time (regression test for the monotonic-coast fix: at any
real-time factor, stamps must never run ahead of /clock).
"""

import math
import time

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan

be_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)


def _collect_two(topic_a, type_a, qos_a, topic_b, type_b, qos_b,
                 timeout, count_a=1):
    node = rclpy.create_node('_camera_sim_collect')
    a, b = [], []
    node.create_subscription(type_a, topic_a, a.append, qos_a)
    node.create_subscription(type_b, topic_b, b.append, qos_b)
    deadline = time.time() + timeout
    while len(a) < count_a and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    return a, b


def test_camera_info_valid(camera_sim):
    infos, _ = _collect_two('/camera/camera_info', CameraInfo, 10,
                            '/clock', Clock, be_qos, timeout=15)
    assert infos, 'no /camera/camera_info from Gazebo'
    info = infos[0]
    assert info.width == 640 and info.height == 480
    assert info.k[0] > 100, f'bogus fx={info.k[0]} (calibration missing?)'


def test_compressed_images_flowing(camera_sim):
    imgs, _ = _collect_two('/camera/image_raw/compressed', CompressedImage,
                           be_qos, '/clock', Clock, be_qos,
                           timeout=15, count_a=10)
    assert len(imgs) >= 10, 'compressed image stream too slow / missing'


def test_scan_camera_publishing(camera_sim):
    scans, _ = _collect_two('/scan_camera', LaserScan, be_qos,
                            '/clock', Clock, be_qos, timeout=30, count_a=20)
    assert len(scans) >= 20, '/scan_camera not publishing'
    assert scans[0].header.frame_id == 'base_scan'


def test_scan_camera_stamps_respect_sim_time(camera_sim):
    """Stamps must be strictly increasing and never ahead of the sim clock
    by more than a small epsilon — at ANY real-time factor."""
    node = rclpy.create_node('_camera_sim_stamps')
    scans, clocks = [], []
    node.create_subscription(LaserScan, '/scan_camera',
                             scans.append, be_qos)
    node.create_subscription(Clock, '/clock', clocks.append, be_qos)
    deadline = time.time() + 30
    while (len(scans) < 30 or not clocks) and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    assert len(scans) >= 30 and clocks, 'not enough data'

    stamps = [s.header.stamp.sec * 10**9 + s.header.stamp.nanosec
              for s in scans]
    assert all(b > a for a, b in zip(stamps, stamps[1:])), \
        'stamps not strictly increasing'

    latest_clock = clocks[-1].clock.sec * 10**9 + clocks[-1].clock.nanosec
    # allow one sim frame of slack (camera period 33 ms) + delivery skew
    assert stamps[-1] <= latest_clock + 100_000_000, (
        f'scan stamp {stamps[-1]} runs ahead of sim clock {latest_clock} — '
        'monotonic coast leaking into sim time?')


def test_scan_camera_sees_world_geometry(camera_sim):
    """In turtlebot3_world the robot faces pillars/walls within camera range
    somewhere during its wander — expect finite returns within 30 s."""
    node = rclpy.create_node('_camera_sim_hits')
    hits = []

    def cb(m):
        finite = [r for r in m.ranges
                  if math.isfinite(r) and m.range_min <= r <= m.range_max]
        if finite:
            hits.append(len(finite))

    node.create_subscription(LaserScan, '/scan_camera', cb, be_qos)
    deadline = time.time() + 30
    while not hits and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    assert hits, '/scan_camera never produced a finite return in 30 s'
