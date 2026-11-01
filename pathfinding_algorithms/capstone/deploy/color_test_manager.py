#!/usr/bin/env python3
"""Checkpoint manager for the colour scavenger test: tracks each colour UNSEEN
-> SEEN -> FOUND from /color_targets, publishes /color_test/status and logs a
results CSV. FOUND = CONSEC_NEEDED consecutive detections within DIST_MAX.
"""

import json
import math
import statistics
import time
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String

DEFAULT_COLORS = 'light_green,purple,orange'
DIST_MAX = 0.6 * 1.15 # +15% tolerance on the outer edge only
# Facing does not gate FOUND: on round pillars an oblique estimate can put an
# approach tens of degrees off the face while the square is in plain view.
OFFNORMAL_MAX = 20.0 # deg, unused
BEARING_MAX = math.radians(25.0) # unused
CONSEC_NEEDED = 10


class ColorTestManager(Node):
    def __init__(self):
        super().__init__('color_test_manager')
        self.declare_parameter('results_csv', '/tmp/color_test_results.csv')
        # must match the seeker's colors parameter
        self.declare_parameter('colors', DEFAULT_COLORS)
        self.colors = tuple(
            c.strip() for c in self.get_parameter('colors').value.split(',')
            if c.strip())
        self.t0 = time.time()
        self._started = time.strftime('%Y-%m-%d_%H:%M:%S')
        self.state = {c: 'UNSEEN' for c in self.colors}
        self.est = {} # color -> fused odom-frame estimate
        # recent detection positions: the estimate is a median, so one bad
        # frame (occluded sliver, quantized distance) can't own it
        self._est_hist = {c: deque(maxlen=7) for c in self.colors}
        self.consec = {c: 0 for c in self.colors}
        self.found_t = {}
        self.path_len = 0.0
        self._last_xy = None
        self._done_logged = False

        self.create_subscription(String, '/color_targets', self._on_dets, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        # sim-only collision count (silent on the real robot); contacts <1 s
        # apart are one episode. Same filter as sim_harness.contact_obstacles.
        self.collisions = 0
        self._last_contact_t = 0.0
        try:
            from gazebo_msgs.msg import ContactsState
            self.create_subscription(ContactsState, '/bumper_states',
                                     self._on_bumper, 10)
        except ImportError:
            pass
        self._status_pub = self.create_publisher(String, '/color_test/status', 10)
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            f'color test armed — find {len(self.colors)} squares '
            f'({", ".join(self.colors)}) '
            f'(<= {DIST_MAX:.2f} m, seen {CONSEC_NEEDED} consecutive frames)')

    def _on_bumper(self, msg):
        hit = None
        for state in msg.states:
            for other in (state.collision1_name, state.collision2_name):
                if 'turtlebot3' in other or 'ground_plane' in other:
                    continue
                hit = other
        if hit is None:
            return
        now = time.time()
        if now - self._last_contact_t > 1.0:
            self.collisions += 1
            self.get_logger().warning(
                f'collision #{self.collisions}: {hit.split("::")[0]}')
        self._last_contact_t = now

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        if self._last_xy is not None:
            self.path_len += math.hypot(p.x - self._last_xy[0],
                                        p.y - self._last_xy[1])
        self._last_xy = (p.x, p.y)

    def _on_dets(self, msg):
        data = json.loads(msg.data)
        seen_this_msg = set()
        for d in data['detections']:
            c = d['color']
            if c not in self.state or self.state[c] == 'FOUND':
                continue
            seen_this_msg.add(c)
            if 'odom_x' in d:
                # position error grows with range, so take the median over
                # the closest half of the history once enough samples exist
                self._est_hist[c].append(
                    (d['odom_x'], d['odom_y'], d['distance_m']))
                hist = list(self._est_hist[c])
                if len(hist) >= 4:
                    hist = sorted(hist, key=lambda p: p[2])
                    hist = hist[:max(2, len(hist) // 2)]
                self.est[c] = {'x': round(statistics.median(p[0] for p in hist), 3),
                               'y': round(statistics.median(p[1] for p in hist), 3),
                               'distance_m': d['distance_m'],
                               'n_dets': len(self._est_hist[c]),
                               'near_m': round(min(p[2] for p in hist), 2),
                               'stamp': data['stamp']}
            if self.state[c] == 'UNSEEN':
                self.state[c] = 'SEEN'
                self.get_logger().info(
                    f'{c} SEEN at {d["distance_m"]:.2f} m '
                    f'(bearing {math.degrees(d["bearing_rad"]):.0f} deg)')
                self._publish_status()
            qualifies = d['distance_m'] <= DIST_MAX
            self.consec[c] = self.consec[c] + 1 if qualifies else 0
            if self.consec[c] >= CONSEC_NEEDED:
                self.state[c] = 'FOUND'
                self.found_t[c] = time.time() - self.t0
                self.get_logger().info(
                    f'*** {c} FOUND ({self.found_t[c]:.1f} s, '
                    f'{d["distance_m"]:.2f} m, '
                    f'off-normal {d["offnormal_deg"]:.0f} deg) — '
                    f'{sum(s == "FOUND" for s in self.state.values())}'
                    f'/{len(self.colors)}')
                self._publish_status()
        # a color not present in this frame loses its consecutive streak
        for c in self.colors:
            if c not in seen_this_msg and self.state[c] != 'FOUND':
                self.consec[c] = 0
        if all(s == 'FOUND' for s in self.state.values()):
            self._finish()

    def _publish_status(self):
        status = {
            'elapsed_s': round(time.time() - self.t0, 1),
            'states': self.state,
            'estimates': self.est,
            'found_times': {k: round(v, 1) for k, v in self.found_t.items()},
            'path_m': round(self.path_len, 2),
            'collisions': self.collisions,
            'done': all(s == 'FOUND' for s in self.state.values()),
        }
        self._status_pub.publish(String(data=json.dumps(status)))
        self._write_progress(status)

    def _write_progress(self, status):
        """Keep <results_csv stem>_progress.json current so a run stopped
        early still leaves its partial result; the CSV row needs completion."""
        csv = Path(self.get_parameter('results_csv').value)
        out = csv.with_name(f'{csv.stem}_progress.json')
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps({'started': self._started, **status}))
        tmp.replace(out) # atomic: a kill mid-write never leaves half a file

    def _finish(self):
        if self._done_logged:
            return
        self._done_logged = True
        total = time.time() - self.t0
        self.get_logger().info(
            f'=== ALL {len(self.colors)} SQUARES FOUND in {total:.1f} s, '
            f'path {self.path_len:.1f} m ===')
        csv = Path(self.get_parameter('results_csv').value)
        header = ('stamp,total_s,path_m,collisions,' +
                  ','.join(f'{c}_s' for c in self.colors))
        # a run with a different color set must not append misaligned
        # columns to an existing CSV — divert to a per-set file instead
        if csv.exists() and csv.read_text().splitlines()[:1] != [header]:
            csv = csv.with_name(
                f'{csv.stem}_{"_".join(self.colors)}{csv.suffix}')
            self.get_logger().warning(
                f'results_csv header differs (other color set) — '
                f'writing {csv}')
        new = not csv.exists()
        with csv.open('a') as f:
            if new:
                f.write(header + '\n')
            f.write(f'{time.strftime("%Y-%m-%d_%H:%M:%S")},{total:.1f},'
                    f'{self.path_len:.1f},{self.collisions},' +
                    ','.join(f'{self.found_t[c]:.1f}'
                             for c in self.colors) + '\n')


def main():
    rclpy.init()
    node = ColorTestManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
