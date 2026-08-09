#!/usr/bin/env python3
"""Watch the camera-avoidance source for edits and restart the stack.

Polls mtimes of *.py / *.yaml / *.lua / launch files in dwa_camera_avoidance
and dwa_avoidance every second; 2 s after the last change settles, runs
restart_camera_stack.sh. Log: /tmp/dwa_watch.log
"""

import subprocess
import sys
import time
from pathlib import Path

WS = Path('/home/turtlebot/pathfinding_algorithms')
WATCH_DIRS = [WS / 'dwa_camera_avoidance', WS / 'dwa_avoidance']
EXTS = {'.py', '.yaml', '.lua', '.xml'}
RESTART = WS / 'dev_tools' / 'restart_camera_stack.sh'
DEBOUNCE_S = 2.0


def snapshot():
    files = {}
    for d in WATCH_DIRS:
        for p in d.rglob('*'):
            if p.suffix in EXTS and '__pycache__' not in p.parts:
                try:
                    files[p] = p.stat().st_mtime
                except OSError:
                    pass
    return files


def log(msg):
    print(f'[watch {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    prev = snapshot()
    log(f'watching {len(prev)} files in dwa_camera_avoidance + dwa_avoidance')
    pending_since = None
    changed_paths = set()

    while True:
        time.sleep(1.0)
        cur = snapshot()
        changed = {p for p in cur if cur.get(p) != prev.get(p)}
        changed |= {p for p in prev if p not in cur}
        if changed:
            changed_paths |= changed
            pending_since = time.time()
            prev = cur
            continue
        prev = cur
        if pending_since and time.time() - pending_since >= DEBOUNCE_S:
            names = ', '.join(sorted(p.name for p in changed_paths)[:5])
            log(f'change detected ({names}) — restarting stack')
            r = subprocess.run(['bash', str(RESTART)],
                               capture_output=True, text=True)
            for line in (r.stdout + r.stderr).splitlines():
                log('  ' + line)
            log('restart ' + ('OK' if r.returncode == 0 else f'FAILED (exit {r.returncode})'))
            pending_since = None
            changed_paths = set()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
