"""Terminal prompts, live keys, find review and camera viewers."""

import fcntl
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import termios
import time
import tty

import numpy as np

from .records import set_last_field

# Terminal state at startup, restored before every prompt: a killed child can
# leave it non-blocking or in cbreak mode, and input() then reads EOF at once.
_TTY_SANE = None


def _remember_tty():
    global _TTY_SANE
    if sys.stdin.isatty():
        _TTY_SANE = termios.tcgetattr(sys.stdin.fileno())


def ask(prompt):
    """input(), on a terminal put back into the state it started in."""
    if sys.stdin.isatty():
        fd = sys.stdin.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        if flags & os.O_NONBLOCK:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
            print('  (terminal was left non-blocking; repaired)')
        if _TTY_SANE is not None and termios.tcgetattr(fd) != _TTY_SANE:
            termios.tcsetattr(fd, termios.TCSANOW, _TTY_SANE)
            print('  (terminal mode was changed; repaired)')
    return input(prompt)


def found_frames(bag_dir):
    """{colour: (BGR frame with the detector's box, detection)} read from a
    run's bag: the camera frame nearest the detection behind each FOUND."""
    import cv2
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    def reader(topics):
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id='sqlite3'),
               rosbag2_py.ConverterOptions('', ''))
        r.set_filter(rosbag2_py.StorageFilter(topics=topics))
        return r
    found, last = {}, {}
    r = reader(['/color_test/status', '/color_targets'])
    while r.has_next():
        topic, data, t = r.read_next()
        msg = json.loads(deserialize_message(data, String).data)
        if topic == '/color_targets':
            for d in msg.get('detections', []):
                if d['color'] not in found:
                    last[d['color']] = (t, d)
        else:
            for c, state in msg.get('states', {}).items():
                if state == 'FOUND' and c not in found and c in last:
                    found[c] = last[c]
    out = {}
    for c, (t_det, det) in found.items():
        r = reader(['/image_raw/compressed'])
        r.seek(max(0, t_det - 200_000_000))
        best = None
        while r.has_next():
            _, data, t = r.read_next()
            if best is None or abs(t - t_det) < abs(best[0] - t_det):
                best = (t, data)
            if t > t_det + 200_000_000:
                break
        if best is None:
            continue
        img = deserialize_message(best[1], CompressedImage)
        bgr = cv2.imdecode(np.frombuffer(bytes(img.data), np.uint8),
                           cv2.IMREAD_COLOR)
        if 'bbox' in det:
            x, y, w, h = det['bbox']
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 30), (252, 252, 251), -1)
        cv2.putText(bgr, f"{c} FOUND at {det['distance_m']:.2f} m", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (11, 11, 11), 2)
        out[c] = (bgr, det)
    return out


def review_finds(csv_path, row, trial_dir):
    """Show the frame behind every FOUND and record any the operator judges
    false in false_found. A failure here is reported, never fatal."""
    import cv2
    try:
        frames = found_frames(trial_dir / 'bag')
    except Exception as e: # review is a convenience, never fatal
        print(f'  (could not read the finds back from the bag: {e})')
        return
    if not frames:
        print('  no FOUND to review in this run')
        return
    for c, (img, _) in frames.items():
        cv2.imwrite(str(trial_dir / f'found_{c}.png'), img)
    pad = np.full((next(iter(frames.values()))[0].shape[0], 12, 3), 252,
                  np.uint8)
    combined = trial_dir / 'finds.png'
    tiles = []
    for img, _ in frames.values():
        tiles += [img, pad]
    cv2.imwrite(str(combined), np.hstack(tiles[:-1]))
    viewer = subprocess.Popen(['eog', str(combined)], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, start_new_session=True)
    if os.environ.get('WINDOWID'): # keyboard back to this terminal
        time.sleep(1.5)
        subprocess.run(['xdotool', 'windowactivate', os.environ['WINDOWID']],
                       capture_output=True)
    print(f"  finds shown in {combined.name}: {', '.join(frames)}")
    try:
        answer = ask('  any FALSE find? type its colour(s), comma separated, '
                       'or Enter if all are real > ').strip()
    except EOFError:
        answer = ''
    false = [c.strip() for c in answer.split(',') if c.strip() in frames]
    if false:
        set_last_field(csv_path, 'false_found', ','.join(false))
        print(f"  recorded false_found = {','.join(false)}")
    try:
        viewer.terminate()
    except Exception:
        pass


class KeyReader:
    """Single keypresses from the terminal without Enter (cbreak mode)."""

    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def poll(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None

    def __exit__(self, *exc):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)


_VIEWERS = []


VIEWER_W, VIEWER_H = 1000, 800


VIEWER_SLOTS = {'/image_raw/compressed': 0,  # camera: left slot
                '/color_debug': 1}            # detector: right slot


def _viewer_origin():
    """Top-left of the largest monitor, (0, 0) when xrandr is unavailable."""
    try:
        out = subprocess.run(['xrandr', '--listmonitors'], capture_output=True,
                             text=True).stdout
    except FileNotFoundError:
        return 0, 0
    mons = [tuple(map(int, m)) for m in re.findall(
        r'(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)', out)]
    if not mons:
        return 0, 0
    w, h, x, y = max(mons, key=lambda m: m[0] * m[1])
    return x, y


def open_viewers(env):
    """Camera and detector windows, opened once per session and kept out of
    the harness's process list so they survive each run's ROS restart."""
    if any(v.poll() is None for v in _VIEWERS):
        return
    view_env = dict(env, LIBGL_ALWAYS_SOFTWARE='1')
    for topic in VIEWER_SLOTS:
        _VIEWERS.append(subprocess.Popen(
            ['bash', '-c', 'source /opt/ros/humble/setup.bash && exec ros2 '
             f'run rqt_image_view rqt_image_view {topic}'],
            env=view_env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True))
    import threading
    threading.Thread(target=_place_viewers, daemon=True).start()


def _place_viewers(timeout_s=30.0):
    """Put the viewers side by side (both open at the same spot), matching
    windows to topics by process command line. Needs xdotool and xprop."""
    placed, deadline = set(), time.monotonic() + timeout_s
    while len(placed) < len(VIEWER_SLOTS) and time.monotonic() < deadline:
        time.sleep(1.0)
        try:
            wins = subprocess.run(
                ['xdotool', 'search', '--name', 'rqt_image_view__ImageView'],
                capture_output=True, text=True).stdout.split()
        except FileNotFoundError:
            return
        for w in wins:
            pid = subprocess.run(['xprop', '-id', w, '_NET_WM_PID'],
                                 capture_output=True, text=True).stdout.split()
            try:
                cmd = Path(f'/proc/{pid[-1]}/cmdline').read_bytes().decode()
            except (OSError, IndexError):
                continue
            ox, oy = _viewer_origin()
            for topic, slot in VIEWER_SLOTS.items():
                if topic in cmd and topic not in placed:
                    subprocess.run(['xdotool', 'windowsize', w,
                                    str(VIEWER_W), str(VIEWER_H)])
                    subprocess.run(['xdotool', 'windowmove', w,
                                    str(ox + 40 + slot * (VIEWER_W + 20)),
                                    str(oy + 60)])
                    placed.add(topic)
    # operator keys only reach the runner's terminal while it has focus
    if os.environ.get('WINDOWID'):
        subprocess.run(['xdotool', 'windowactivate', os.environ['WINDOWID']],
                       capture_output=True)
