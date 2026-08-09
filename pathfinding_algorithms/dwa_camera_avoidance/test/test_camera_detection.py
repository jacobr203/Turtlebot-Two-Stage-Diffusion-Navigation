"""
Tests for camera_obstacle_node's detection pipeline behavior:
empty-zone robustness (dynamic params must not crash the node) and the
overhead-channel toggle.

Run with:
  cd ~/pathfinding_algorithms
  source install/setup.bash
  python3 -m pytest dwa_camera_avoidance/test/test_camera_detection.py -v
"""

import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, CompressedImage

from dwa_camera_avoidance.camera_obstacle_node import CameraObstacleNode, CV_AVAILABLE

pytestmark = pytest.mark.skipif(not CV_AVAILABLE,
                                reason='OpenCV / cv_bridge not available')


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_node(**param_overrides):
    overrides = [Parameter(k, value=v) for k, v in param_overrides.items()]
    return CameraObstacleNode(parameter_overrides=overrides)


def _prime_camera_info(node):
    info = CameraInfo()
    info.width, info.height = 640, 480
    info.k = [465.0, 0.0, 320.0, 0.0, 465.0, 240.0, 0.0, 0.0, 1.0]
    node._info_cb(info)


def _jpeg_image(with_edges=True):
    import cv2
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    if with_edges:
        img[200:260, 250:390] = 255       # bright block → strong edges
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    msg = CompressedImage()
    msg.format = 'jpeg'
    msg.data = buf.tobytes()
    return msg


def _finite_points(node, img_msg):
    """Feed an image through the node, return finite ranges it publishes."""
    import math
    published = []
    node._scan_pub = type('P', (), {'publish': lambda self, m: published.append(m),
                                    'get_subscription_count': lambda self: 0})()
    node._image_cb(img_msg)
    assert published, 'node published no scan'
    return [r for r in published[-1].ranges if math.isfinite(r)]


def _image_with(draw):
    import cv2
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    draw(img)
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    msg = CompressedImage()
    msg.format = 'jpeg'
    msg.data = buf.tobytes()
    return msg


# Test geometry pinned explicitly (the node's declared defaults differ from
# the shipped yaml): fx=fy=465, camera 0.12 m high, 8° down. Image row ~312
# projects to the floor ~0.40 m ahead → support window ≈ 34 rows, of which
# the upper half [17, 34] must contain edges for a candidate to pass.
# Blocky noise (8 px cells) survives the 7x7 Gaussian blur and gives dense
# Canny edges like real object texture.

_GEOM = dict(camera_height=0.12, camera_tilt_deg=8.0, camera_x_offset=0.05,
             enable_overhead_detection=False, min_obstacle_columns=1,
             floor_zone_fraction=0.60)


def _noise_block(im, r0, r1, c0=280, c1=360, seed=7):
    import cv2
    rng = np.random.RandomState(seed)
    cells = rng.randint(0, 2, ((r1 - r0) // 8 + 1, (c1 - c0) // 8 + 1),
                        dtype=np.uint8) * 255
    patch = cv2.resize(cells, (c1 - c0, r1 - r0),
                       interpolation=cv2.INTER_NEAREST)
    im[r0:r1, c0:c1] = patch[:, :, None]


def test_floor_obstacle_with_body_above_is_detected(ros):
    """A textured object body rising well above its base edge must survive
    the vertical-continuity filter."""
    node = _make_node(**_GEOM)
    try:
        _prime_camera_info(node)
        msg = _image_with(lambda im: _noise_block(im, 180, 314))  # tall body
        assert len(_finite_points(node, msg)) > 0
    finally:
        node.destroy_node()


def test_isolated_floor_streak_is_rejected(ros):
    """A shallow textured band (carpet pattern / shadow) never reaches the
    upper support window — the continuity filter must reject it."""
    node = _make_node(**_GEOM)
    try:
        _prime_camera_info(node)
        msg = _image_with(lambda im: _noise_block(im, 298, 314))  # flat band
        assert _finite_points(node, msg) == []
    finally:
        node.destroy_node()


def test_continuity_filter_can_be_disabled(ros):
    """floor_support_min_edges: 0 restores pre-filter behavior (band passes)."""
    node = _make_node(floor_support_min_edges=0, **_GEOM)
    try:
        _prime_camera_info(node)
        msg = _image_with(lambda im: _noise_block(im, 298, 314))
        assert len(_finite_points(node, msg)) > 0
    finally:
        node.destroy_node()


def test_empty_zones_do_not_crash(ros):
    """floor_zone_fraction 0 / degenerate overhead zone must be a no-op,
    not an exception (params are dynamic; this crashed the node live)."""
    node = _make_node(floor_zone_fraction=0.0,
                      overhead_zone_top_fraction=0.10,
                      overhead_zone_bottom_fraction=0.10)
    try:
        _prime_camera_info(node)
        node._image_cb(_jpeg_image())       # must not raise
    finally:
        node.destroy_node()


def test_overhead_channel_toggle(ros):
    """With enable_overhead_detection false, only the floor channel runs."""
    node = _make_node(enable_overhead_detection=False)
    try:
        _prime_camera_info(node)
        calls = []
        node._detect_overhead_obstacles = \
            lambda *a, **k: calls.append(1) or []
        node._image_cb(_jpeg_image())
        assert calls == [], 'overhead detector ran despite being disabled'
    finally:
        node.destroy_node()


def test_overhead_channel_on_by_default(ros):
    node = _make_node()
    try:
        _prime_camera_info(node)
        calls = []
        node._detect_overhead_obstacles = \
            lambda *a, **k: calls.append(1) or []
        node._image_cb(_jpeg_image())
        assert calls == [1]
    finally:
        node.destroy_node()
