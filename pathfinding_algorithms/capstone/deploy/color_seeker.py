#!/usr/bin/env python3
"""Color-square detector for the 3-color scavenger test (color_test_plan.md §3).

Watches the BRIO (real) or Gazebo camera (sim) for the three 2"x2" target
squares and streams detections on /color_targets as one JSON object per
frame-with-detections:

    {"stamp": <float>, "detections": [
        {"color": "orange", "bearing_rad": -0.05, "distance_m": 1.42,
         "offnormal_deg": 12.0, "px_height": 18, "aspect": 0.98,
         "odom_x": 1.3, "odom_y": -0.2, "confidence": 0.91}, ...]}

Distance comes from the KNOWN square size via the pixel HEIGHT of the
detection — height is invariant to the viewing angle that compresses width,
which is instead what estimates the off-normal angle (aspect = w/h ≈ cos of
the angle off the face normal). The checkpoint state machine (2-frame
confirmation, found/unfound bookkeeping) lives in the manager, not here.

Subscribe the COMPRESSED image topic in sim too (/camera/image_raw/
compressed): raw 900 KB best-effort frames are silently fragment-dropped
between processes on this box (0 frames delivered while camera_info flowed
— 2026-07-20), and compressed is what the real BRIO path uses anyway.
"""

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.geometry import yaw_from_quaternion # noqa: E402

SQUARE_SIZE_M = 0.0508          # 2" target squares
PROFILES_FILE = Path(__file__).resolve().parent / 'color_profiles.yaml'


def detect_squares(bgr, bands, fx, cx_px, min_area_px=20):
    # min_area 20 (was 60): a 2" square at the ~3.2 m range exploration
    # passes at renders 45-55 px^2 frontal — the 60 floor rejected every
    # sighting of world 405's light_green (sniffer data 2026-07-22); the
    # fill/aspect/HSV gates carry specificity at small areas.
    """Pure detection: BGR frame -> per-color best square candidate.

    Returns a list of dicts (bearing/distance/off-normal per plan §3).
    Aspect is bounded [0.35, 1.3]: real squares only ever compress in width
    (viewing angle), so anything wider-than-tall beyond noise or thinner
    than cos(69°) is not our target.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    out = []
    for color, b in bands.items():
        lo = np.array([b['h'][0], b['s'][0], b['v'][0]], dtype=np.uint8)
        hi = np.array([b['h'][1], b['s'][1], b['v'][1]], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect = w / h
            fill = area / float(w * h)
            # fill<0.6 = ragged blob, not a flat printed square.
            # SMALL-BLOB TIER (search time is
            # dominated by detection RANGE — orange is sighted at ~3.9 m
            # after 14 min of hunting). Below ~7 px of height a 2" square
            # spans 4-6 px, where one pixel of morphological erosion
            # swings fill and aspect wildly — the shape gates, not the
            # area floor, were rejecting genuine far squares. Small blobs
            # keep the AREA and HSV gates (which carry the specificity)
            # with loosened shape bounds; near blobs stay strict, and the
            # 10-consecutive-frame confirmation upstream plus the 6 m
            # plausibility bound below still guard false positives.
            if h <= 7:
                if fill < 0.45 or not (0.25 <= aspect <= 1.6):
                    continue
            elif fill < 0.6 or not (0.35 <= aspect <= 1.3):
                continue
            if best is None or area > best[0]:
                best = (area, x, y, w, h, fill, aspect)
        if best is None:
            continue
        area, x, y, w, h, fill, aspect = best
        u = x + w / 2.0
        bearing = -math.atan2(u - cx_px, fx) # +left in base frame
        distance = fx * SQUARE_SIZE_M / h
        # plausibility: the camera physically can't be closer than ~12 cm to
        # a mounted square, and beyond ~6 m a 2" square is sub-noise — a
        # detection outside this range is a big same-hue surface, not ours
        if not (0.12 <= distance <= 6.0):
            continue
        offnormal = math.degrees(math.acos(max(0.0, min(1.0, aspect))))
        out.append({
            'color': color,
            'bearing_rad': round(bearing, 4),
            'distance_m': round(distance, 3),
            'offnormal_deg': round(offnormal, 1),
            'px_height': h,
            'aspect': round(aspect, 3),
            'confidence': round(fill, 2),
        })
    return out


class ColorSeeker(Node):
    def __init__(self):
        super().__init__('color_seeker')
        p = lambda n: self.get_parameter(n).value # noqa: E731
        for name, default in [
                ('profile', 'real'),
                ('image_topic', '/image_raw/compressed'),
                ('compressed', True),
                ('camera_info_topic', '/camera_info'),
                # fallback intrinsics until the first CameraInfo arrives
                # (BRIO 640x480 ballpark; overwritten by the real values)
                ('fx_fallback', 500.0), ('cx_fallback', 320.0),
                # camera sits slightly ahead of base_link; only affects the
                # odom-frame projection, not distance/bearing themselves
                ('cam_offset_x', 0.03),
                ('min_period_s', 0.15),
                ('publish_debug', False)]:
            self.declare_parameter(name, default)

        profiles = yaml.safe_load(PROFILES_FILE.read_text())
        if p('profile') not in profiles:
            raise SystemExit(f"unknown profile {p('profile')!r} "
                             f"(have {list(profiles)})")
        self._bands = profiles[p('profile')]
        self._fx = p('fx_fallback')
        self._cx = p('cx_fallback')
        self._have_info = False
        self._cam_dx = p('cam_offset_x')
        self._min_period = p('min_period_s')
        self._last = 0.0
        self._odom = None
        self._n_frames = 0
        self._n_dets = 0
        self.create_timer(10.0, self._stats)

        img_qos = QoSProfile(depth=1)
        img_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        if p('compressed'):
            self.create_subscription(CompressedImage, p('image_topic'),
                                     self._on_compressed, img_qos)
        else:
            self.create_subscription(Image, p('image_topic'),
                                     self._on_raw, img_qos)
        self.create_subscription(CameraInfo, p('camera_info_topic'),
                                 self._on_info, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self._pub = self.create_publisher(String, '/color_targets', 10)
        self._debug_pub = (self.create_publisher(Image, '/color_debug', 2)
                           if p('publish_debug') else None)
        self.get_logger().info(
            f"color_seeker up — profile={p('profile')}, "
            f"colors={list(self._bands)}, topic={p('image_topic')}")

    def _stats(self):
        self.get_logger().info(
            f'{self._n_frames} frames, {self._n_dets} detection msgs')

    def _on_info(self, msg):
        if not self._have_info:
            self._fx, self._cx = msg.k[0], msg.k[2]
            self._have_info = True
            self.get_logger().info(f'camera intrinsics: fx={self._fx:.1f} '
                                   f'cx={self._cx:.1f}')

    def _on_odom(self, msg):
        self._odom = msg

    def _on_compressed(self, msg):
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8),
                           cv2.IMREAD_COLOR)
        if bgr is not None:
            self._process(bgr)

    def _on_raw(self, msg):
        if msg.encoding not in ('bgr8', 'rgb8'):
            self.get_logger().warning(f'unsupported encoding {msg.encoding}',
                                      throttle_duration_sec=10.0)
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        self._process(img if msg.encoding == 'bgr8'
                      else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _process(self, bgr):
        self._n_frames += 1
        now = time.time()
        if now - self._last < self._min_period:
            return
        self._last = now
        dets = detect_squares(bgr, self._bands, self._fx, self._cx)
        if not dets:
            return
        if self._odom is not None:
            pq = self._odom.pose.pose
            yaw = yaw_from_quaternion(pq.orientation.x, pq.orientation.y,
                                      pq.orientation.z, pq.orientation.w)
            for d in dets:
                a = yaw + d['bearing_rad']
                r = d['distance_m'] + self._cam_dx
                d['odom_x'] = round(pq.position.x + r * math.cos(a), 3)
                d['odom_y'] = round(pq.position.y + r * math.sin(a), 3)
        self._n_dets += 1
        self._pub.publish(String(data=json.dumps(
            {'stamp': now, 'detections': dets})))


def main():
    rclpy.init()
    node = ColorSeeker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
