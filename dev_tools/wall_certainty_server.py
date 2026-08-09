#!/usr/bin/env python3
"""Live wall-certainty viewer

Runs on http://localhost:8666 : Yields certainty map
(occupancy + per-face inspection certainty, hover readout, pan/zoom) but
fed LIVE from ROS — the page polls /data.json every 2 s and re-renders from input.

  # sim (domain 7)

  ROS_DOMAIN_ID=7 ~/pathfinding_algorithms/capstone/.venv/bin/python \
      ~/dev_tools/wall_certainty_server.py
  # real robot: same, without the domain override
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

PORT = 8666
ARTIFACT_HTML = Path(__file__).resolve().parent / 'wall_certainty_template.html'

state = {'map': None, 'cov': None, 'robot': None}
lock = threading.Lock()


class Collector(Node):
    """Map collector with staleness auto-reconnect: subscriptions created
    before a sim/stack restart can stay silently bound to dead endpoints.
    if no map arrives for 15s, tear the subscriptions down and resubscribe."""

    def __init__(self):
        super().__init__('wall_certainty_server')
        import time
        self._last_rx = time.time()
        self._subs = []
        self._subscribe()
        self.create_timer(10.0, self._check_stale)
        # robot marker (2026-07-23): map-frame pose via TF, polled at 2 Hz
        from tf2_ros import Buffer, TransformListener
        self._tfb = Buffer()
        self._tfl = TransformListener(self._tfb, self)
        self.create_timer(0.5, self._poll_robot)

    def _poll_robot(self):
        import math
        for child in ('base_footprint', 'base_link'):
            try:
                t = self._tfb.lookup_transform('map', child,
                                               rclpy.time.Time())
            except Exception:  # noqa: BLE001 (TF not up yet)
                continue
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            with lock:
                state['robot'] = {'x': t.transform.translation.x,
                                  'y': t.transform.translation.y,
                                  'yaw': yaw}
            return

    def _subscribe(self):
        self._subs = [
            self.create_subscription(OccupancyGrid, '/map', self._m, 1),
            self.create_subscription(OccupancyGrid, '/seek/coverage',
                                     self._c, 1),
        ]

    def _check_stale(self):
        import time
        if time.time() - self._last_rx > 15.0:
            self.get_logger().info('map silent 15 s — resubscribing')
            for s in self._subs:
                self.destroy_subscription(s)
            self._subscribe()
            self._last_rx = time.time()

    def _m(self, msg):
        import time
        self._last_rx = time.time()
        with lock:
            old = state['map']
            if old is not None:
                t_new = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                t_old = old.header.stamp.sec + old.header.stamp.nanosec * 1e-9
                # new-run detection (2026-07-23): layers from the previous
                # run kept serving until overwritten, blending two worlds
                # on the page. A stamp REGRESSION (sim clock restarts near
                # zero on every run) or a fresh tiny map replacing a big
                # one (real-clock cartographer restart) marks a new run —
                # drop both layers so runs never blend.
                area_new = msg.info.width * msg.info.height
                area_old = old.info.width * old.info.height
                if t_new < t_old - 5.0 or area_new < 0.4 * area_old:
                    state['cov'] = None
                    state['robot'] = None
                    # sim time restarted BELOW the buffered transforms —
                    # tf2 then rejects all new data as TF_OLD_DATA and the
                    # robot marker freezes at its pre-restart pose
                    try:
                        self._tfb.clear()
                    except Exception:  # noqa: BLE001
                        pass
                    self.get_logger().info('new run detected — state cleared')
            state['map'] = msg

    def _c(self, msg):
        with lock:
            state['cov'] = msg


def snapshot():
    import time
    with lock:
        m, c, rb = state['map'], state['cov'], state['robot']
    if m is None:
        return None
    return {'w': m.info.width, 'h': m.info.height,
            'res': m.info.resolution,
            'ox': m.info.origin.position.x, 'oy': m.info.origin.position.y,
            'stamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'grid': list(m.data),
            'robot': rb,
            'cov': (list(c.data) if c and c.info.width == m.info.width
                    and c.info.height == m.info.height else None)}


def live_html():
    """The artifact page, adapted: data comes from /data.json on a 2 s poll."""
    import re
    html = ARTIFACT_HTML.read_text()
    html = re.sub(r'const D = \{.*?\};\n', 'let D = null;\n',
                  html, count=1, flags=re.S)
    html = html.replace('renderBase(); fit();', """
async function refresh(first){
  const r = await fetch('/data.json'); if(!r.ok) return;
  D = await r.json();
  if(D.cov){cert.length=0; cert.length=D.w*D.h; cert.fill(null);
    for(let i=0;i<D.cov.length;i++) if(D.cov[i]>=0) cert[i]=100-D.cov[i];}
  renderBase(); if(first) fit(); else layout();
  document.querySelector('.sub').textContent =
    'LIVE · updated ' + D.stamp + ' · hover for certainty';
}
const cert=[];
refresh(true); setInterval(()=>refresh(false), 2000);
""")
    # the original declares cert from D at load; neutralize that line
    html = re.sub(r'const cert = new Array.*?\n(if \(D\.cov\).*?\n)?',
                  '', html, count=1, flags=re.S)
    return html


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == '/data.json':
            snap = snapshot()
            body = json.dumps(snap).encode() if snap else b'null'
            ctype = 'application/json'
        else:
            body = live_html().encode()
            ctype = 'text/html'
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    rclpy.init()
    node = Collector()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    print(f'live certainty map: http://localhost:{PORT}')
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()


if __name__ == '__main__':
    main()
