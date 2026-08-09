#!/usr/bin/env python3
"""Live HSV mask visualizer for calibrating the scavenger-test color bands.

Shows the camera feed with detections overlaid plus one mask window per
color; click anywhere in the feed to print the HSV value under the cursor
(that's the number you tune color_profiles.yaml with).

  # real BRIO (default topics)
  ~/pathfinding_algorithms/capstone/.venv/bin/python ~/dev_tools/color_mask_viz.py

  # sim camera
  ... color_mask_viz.py --profile sim --topic /camera/image_raw --raw

Keys: q quits.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

CAPSTONE = Path.home() / 'pathfinding_algorithms' / 'capstone'
sys.path.insert(0, str(CAPSTONE))
from deploy.color_seeker import detect_squares, PROFILES_FILE  # noqa: E402

DRAW = {'light_green': (144, 238, 144), 'purple': (128, 0, 128),
        'orange': (0, 127, 255)}  # BGR


class Viz(Node):
    def __init__(self, args):
        super().__init__('color_mask_viz')
        self.bands = yaml.safe_load(PROFILES_FILE.read_text())[args.profile]
        self.frame = None
        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        if args.raw:
            self.create_subscription(Image, args.topic, self.on_raw, qos)
        else:
            self.create_subscription(CompressedImage, args.topic,
                                     self.on_comp, qos)

    def on_comp(self, msg):
        self.frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8),
                                  cv2.IMREAD_COLOR)

    def on_raw(self, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        self.frame = (img if msg.encoding == 'bgr8'
                      else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='real')
    ap.add_argument('--topic', default='/image_raw/compressed')
    ap.add_argument('--raw', action='store_true',
                    help='subscribe sensor_msgs/Image instead of Compressed')
    args = ap.parse_args()

    rclpy.init()
    node = Viz(args)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and node.frame is not None:
            h, s, v = cv2.cvtColor(node.frame, cv2.COLOR_BGR2HSV)[y, x]
            print(f'({x},{y})  H={h} S={s} V={v}')

    cv2.namedWindow('feed')
    cv2.setMouseCallback('feed', on_mouse)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.frame is None:
            continue
        bgr = node.frame.copy()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        for color, b in node.bands.items():
            mask = cv2.inRange(hsv,
                               np.array([b['h'][0], b['s'][0], b['v'][0]]),
                               np.array([b['h'][1], b['s'][1], b['v'][1]]))
            cv2.imshow(color, mask)
        for d in detect_squares(bgr, node.bands, fx=500.0,
                                cx_px=bgr.shape[1] / 2):
            cv2.putText(bgr, f"{d['color']} {d['distance_m']:.2f}m "
                             f"off{d['offnormal_deg']:.0f}deg",
                        (10, 30 + 25 * list(node.bands).index(d['color'])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        DRAW[d['color']], 2)
        cv2.imshow('feed', bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
