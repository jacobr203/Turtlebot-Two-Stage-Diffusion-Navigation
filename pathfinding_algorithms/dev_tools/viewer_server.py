#!/usr/bin/env python3
"""Live web viewer for the camera-avoidance dev loop.

Serves http://localhost:8765 — a canvas view of Cartographer's inputs:
  /map            occupancy grid (transient-local)
  /scan_relayed   LiDAR scan (Cartographer input 1)
  /scan_camera    camera virtual scan (Cartographer input 2)
  map->base TF    robot pose
plus per-topic rates and node liveness, so the stack can be killed and
relaunched freely while this server keeps running.
"""

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

PORT = 8765
WATCHED_NODES = ['dwa_camera_node', 'camera_obstacle_node', 'cartographer_node',
                 'scan_relay_node', 'tf_relay_node', 'turtlebot3_node']

state_lock = threading.Lock()
state = {
    'map': None, 'pose': None,
    'scan_lidar': None, 'scan_camera': None,
    'stamps': {'map': 0.0, 'scan_relayed': 0.0, 'scan_camera': 0.0},
    'ticks': {'scan_relayed': [], 'scan_camera': [], 'map': []},
    'nodes': {},
}


def _rate(name):
    now = time.time()
    ticks = [t for t in state['ticks'][name] if now - t < 5.0]
    state['ticks'][name] = ticks
    return round(len(ticks) / 5.0, 1)


class Viewer(Node):
    def __init__(self):
        super().__init__('live_viewer')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        scan_qos = QoSProfile(depth=5,
                              reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)
        self.create_subscription(LaserScan, '/scan_relayed',
                                 lambda m: self._scan_cb('scan_relayed', 'scan_lidar', m), scan_qos)
        self.create_subscription(LaserScan, '/scan_camera',
                                 lambda m: self._scan_cb('scan_camera', 'scan_camera', m), scan_qos)
        self.create_timer(2.0, self._poll_graph)

    def _map_cb(self, m):
        with state_lock:
            state['map'] = {
                'width': m.info.width, 'height': m.info.height,
                'resolution': m.info.resolution,
                'origin_x': m.info.origin.position.x,
                'origin_y': m.info.origin.position.y,
                'data': list(m.data),
            }
            state['stamps']['map'] = time.time()
            state['ticks']['map'].append(time.time())

    def _pose(self):
        for base in ('base_footprint', 'base_link'):
            try:
                tf = self.tf_buffer.lookup_transform('map', base, rclpy.time.Time())
                q = tf.transform.rotation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                return {'x': tf.transform.translation.x,
                        'y': tf.transform.translation.y, 'yaw': yaw}
            except Exception:
                continue
        return None

    def _scan_cb(self, tick_name, key, msg):
        pose = self._pose() or {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        pts = []
        a = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                ang = pose['yaw'] + a
                pts.append([round(pose['x'] + r * math.cos(ang), 3),
                            round(pose['y'] + r * math.sin(ang), 3)])
            a += msg.angle_increment
        with state_lock:
            state[key] = pts
            state['pose'] = pose
            state['stamps'][tick_name] = time.time()
            state['ticks'][tick_name].append(time.time())

    def _poll_graph(self):
        names = [n for n, _ in self.get_node_names_and_namespaces()]
        with state_lock:
            state['nodes'] = {n: (n in names) for n in WATCHED_NODES}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/data'):
            now = time.time()
            with state_lock:
                payload = {
                    'map': state['map'],
                    'pose': state['pose'],
                    'scan_lidar': state['scan_lidar'],
                    'scan_camera': state['scan_camera'],
                    'rates': {k: _rate(k) for k in ('scan_relayed', 'scan_camera', 'map')},
                    'ages': {k: (round(now - v, 1) if v else None)
                             for k, v in state['stamps'].items()},
                    'nodes': state['nodes'],
                    'server_time': now,
                }
            self._send(200, 'application/json', json.dumps(payload).encode())
        else:
            page = Path(__file__).with_name('viewer.html')
            self._send(200, 'text/html; charset=utf-8', page.read_bytes())


def main():
    rclpy.init()
    node = Viewer()
    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'[viewer] serving http://localhost:{PORT}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
