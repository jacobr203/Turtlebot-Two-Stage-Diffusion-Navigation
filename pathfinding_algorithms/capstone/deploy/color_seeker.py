#!/usr/bin/env python3
"""Colour-square detector: finds the target squares in the camera image and
streams them on /color_targets as JSON. Distance comes from pixel height, which
viewing angle does not change; width/height estimates the off-normal angle.
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


def detect_squares(bgr, bands, fx, cx_px, min_area_px=20,
                   size_m=SQUARE_SIZE_M):
    """BGR frame -> list of detection dicts, best candidate per colour."""
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
            if area < min_area_px: # a 2" square is only ~50 px^2 at 3.2 m
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect = w / h
            fill = area / float(w * h)
            # fill<0.6 = ragged blob, not a flat printed square. Viewing angle
            # only compresses width, hence the aspect bounds. Below ~7 px of
            # height one pixel of erosion swings fill and aspect wildly, so
            # small blobs get looser shape bounds; area and HSV gates remain.
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
        # height, not width: a target wrapped round a pole keeps its full
        # height while its apparent width shrinks with the curvature
        distance = fx * size_m / h
        # a mounted square can't be nearer than ~12 cm and is sub-noise past
        # ~6 m: anything outside this is a big same-hue surface
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
            'bbox': [int(x), int(y), int(w), int(h)],
        })
    return out


DEBUG_THUMB_FRACTION = 5 # each mask thumbnail is 1/5 of the frame's size
DEBUG_BOX_BGR = (0, 0, 255)
DEBUG_TEXT_BGR = (255, 255, 255)


def draw_debug(bgr, bands, dets):
    """Debug view: boxed detections plus one mask thumbnail per colour."""
    out = bgr.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = out.shape[:2]
    tw, th = w // DEBUG_THUMB_FRACTION, h // DEBUG_THUMB_FRACTION
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, (color, b) in enumerate(bands.items()):
        mask = cv2.inRange(hsv, np.array([b['h'][0], b['s'][0], b['v'][0]]),
                           np.array([b['h'][1], b['s'][1], b['v'][1]]))
        thumb = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        x0, y0 = 4 + k * (tw + 4), h - th - 4
        if x0 + tw > w:
            break
        out[y0:y0 + th, x0:x0 + tw] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(out, (x0, y0), (x0 + tw, y0 + th), DEBUG_TEXT_BGR, 1)
        cv2.putText(out, color, (x0 + 3, y0 + 13), font, 0.4, (0, 0, 0), 3)
        cv2.putText(out, color, (x0 + 3, y0 + 13), font, 0.4,
                    DEBUG_TEXT_BGR, 1)
    for d in dets:
        x, y, bw, bh = d['bbox']
        cv2.rectangle(out, (x, y), (x + bw, y + bh), DEBUG_BOX_BGR, 2)
        label = f"{d['color']} {d['distance_m']:.2f} m"
        ty = y - 6 if y > 18 else y + bh + 16
        cv2.putText(out, label, (x, ty), font, 0.5, (0, 0, 0), 3)
        cv2.putText(out, label, (x, ty), font, 0.5, DEBUG_TEXT_BGR, 1)
    return out


class ColorSeeker(Node):
    def __init__(self):
        super().__init__('color_seeker')
        p = lambda n: self.get_parameter(n).value # noqa: E731
        for name, default in [
                ('profile', 'real'),
                # comma-separated subset of the profile's bands to look for
                ('colors', 'light_green,purple,orange'),
                ('image_topic', '/image_raw/compressed'),
                # raw ~900 KB best-effort frames get fragment-dropped between
                # processes, so use compressed in sim too
                ('compressed', True),
                ('camera_info_topic', '/camera_info'),
                # fallback intrinsics until the first CameraInfo arrives
                ('fx_fallback', 500.0), ('cx_fallback', 320.0),
                # only affects the odom-frame projection
                ('cam_offset_x', 0.03),
                ('min_period_s', 0.15),
                # physical HEIGHT of the targets [m]; 3" sticky notes: 0.0762
                ('square_size_m', SQUARE_SIZE_M),
                ('publish_debug', False)]:
            self.declare_parameter(name, default)

        profiles = yaml.safe_load(PROFILES_FILE.read_text())
        if p('profile') not in profiles:
            raise SystemExit(f"unknown profile {p('profile')!r} "
                             f"(have {list(profiles)})")
        wanted = [c.strip() for c in p('colors').split(',') if c.strip()]
        unknown = [c for c in wanted if c not in profiles[p('profile')]]
        if unknown or not wanted:
            raise SystemExit(f"unknown colors {unknown} for profile "
                             f"{p('profile')!r} "
                             f"(have {list(profiles[p('profile')])})")
        self._bands = {c: profiles[p('profile')][c] for c in wanted}
        self._fx = p('fx_fallback')
        self._cx = p('cx_fallback')
        self._have_info = False
        self._cam_dx = p('cam_offset_x')
        self._min_period = p('min_period_s')
        self._size_m = float(p('square_size_m'))
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
        dets = detect_squares(bgr, self._bands, self._fx, self._cx,
                              size_m=self._size_m)
        if self._debug_pub is not None: # also when nothing is detected
            self._publish_debug(draw_debug(bgr, self._bands, dets))
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


    def _publish_debug(self, img):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = img.shape[:2]
        msg.encoding, msg.step = 'bgr8', img.shape[1] * 3
        msg.data = img.tobytes()
        self._debug_pub.publish(msg)


def main():
    rclpy.init()
    node = ColorSeeker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
