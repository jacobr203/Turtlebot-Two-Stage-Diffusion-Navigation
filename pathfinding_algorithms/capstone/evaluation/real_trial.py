#!/usr/bin/env python3
"""Real-robot benchmark trials: one course, one operator, many trials.

Every trial starts from a full reset: ROS is killed on both machines and
bringup restarted, because odometry only zeroes at bringup and the policy
assumes it is zero at the start mark. A fresh lidar-only Cartographer per
trial gives ground truth; the operator calls collisions (no bumper).

  check      readiness: link, clock, sensor rates and gaps, battery
             (never moves the wheels)

Colour scavenger task in an arena (common/arena.py): two-stage seek vs
single-stage wander, 4 Hz, safety gate on; a run ends when every square
is FOUND or at the time cap.

  map-arena         teleop-map the arena, save it, extract the poles
  colour-schedule   seeded run order for one arena placement
  colour-run        walk it with the operator
  colour-status     counted / pending runs and a per-model summary

Point-to-point benchmark on a course (common/course.py); success is the
SLAM pose entering the 1 m goal circle.

  schedule   seeded run order for one course
  run        walk it with the operator
  status     counted / pending cells for a run

  .venv/bin/python evaluation/real_trial.py schedule \\
      --course evaluation/courses/A.yaml --run-id bench_v1
  .venv/bin/python evaluation/real_trial.py run \\
      --course evaluation/courses/A.yaml --run-id bench_v1
  .venv/bin/python evaluation/real_trial.py run ... --shakedown   # rehearsal

Operator keys in a point-to-point trial: c = collision, a = abort, x =
harness fault, space = stop now and classify after. In a colour run c logs
a collision without ending the run, and w ends a wedged or unsafe run as a
counted failure.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import arena as A
from common import course as C

from evaluation.real.config import (
    BRINGUP_TIMEOUT_S, CAMERA_WAIT_S, CKPTS, CLOCK_OFFSET_FAIL_MS,
    CLOCK_OFFSET_OK_MS, COLOUR_MODELS, COLOUR_ROUNDS, COLOUR_ROW_FIELDS,
    COUNTED, DEADMAN_RAISE_GAP_S, DEFAULT_CAP_MIN, DEFAULT_MIN_BATTERY_V,
    INTERRUPTED, MODELS, RAISED_DEADMAN_S, RESULTS, ROUNDS, ROW_FIELDS,
    SESSION_FIELDS, SLAM_READY_TIMEOUT_S, WS)
from evaluation.real.errors import BatteryLow
from evaluation.real.harness import RealHarness, reset_robot
from evaluation.real.metrics import percentile, ping_rtts
from evaluation.real.operator_io import _remember_tty, ask, review_finds
from evaluation.real.records import (
    add_late_collisions, append_row, git_state, read_rows, set_last_notes)
from evaluation.real.robot import RobotLink, read_robot_env
from evaluation.real.schedule import (
    COLOUR_COUNTED, colour_pending, colour_schedule_path,
    load_colour_schedule, load_schedule, make_colour_schedule,
    make_schedule, next_trial_index, pending_cells, practice_cells,
    run_dir, shakedown_cells)
from evaluation.real.tasks import ColourTask, GotoTask, run_task


def cmd_schedule(args):
    course = C.load(args.course)
    out = run_dir(args.run_id, False) / f'schedule_{course["name"]}.csv'
    if out.exists() and not args.force:
        raise SystemExit(f'{out} exists — the order is fixed once written '
                         '(--force only if no trial has run yet)')
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = make_schedule(course, args.run_id, rounds=args.rounds)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} cells -> {out}')


def cmd_status(args):
    course = C.load(args.course)
    schedule = load_schedule(course, args.run_id)
    done = [r for r in read_rows(run_dir(args.run_id, args.shakedown)
                                 / 'trials_w0.csv')
            if r['course'] == course['name']]
    pending = pending_cells(schedule, done)
    counted = [r for r in done if r['verdict'] in COUNTED]
    print(f"course {course['name']}: {len(counted)} counted, "
          f"{len(done) - len(counted)} re-run attempts, "
          f"{len(pending)}/{len(schedule)} pending")
    for m in MODELS:
        v = [r['verdict'] for r in counted if r['model'] == m]
        if v:
            print(f"  {m:13s} {v.count('success')}/{len(v)} success, "
                  f"{v.count('collision')} collision, "
                  f"{v.count('timeout')} timeout, {v.count('crash')} crash")


def preflight(args, layout_paths):
    """Refuse to start unless the link, code and layout files suit counted
    runs. Returns (sha, dirty)."""
    up = subprocess.run(['ip', 'link', 'show', 'eth0'], capture_output=True,
                        text=True).stdout
    if 'state UP' not in up:
        raise SystemExit('eth0 is down: the CycloneDDS config cannot reach '
                         'the robot. Bring it up (or restart WSL) first.')
    if subprocess.run(['pgrep', '-f', 'run\\.s[h]'],
                      capture_output=True).returncode == 0:
        raise SystemExit('a run.sh session is running — stop it first; two '
                         'stacks would fight over /cmd_vel')
    for m, p in CKPTS.items():
        if not p.exists():
            raise SystemExit(f'missing checkpoint for {m}: {p}')
    sha, dirty = git_state()
    for path in layout_paths:
        tracked = subprocess.run(
            ['git', '-C', str(WS), 'ls-files', '--error-unmatch', str(path)],
            capture_output=True).returncode == 0
        if not tracked:
            dirty = (dirty + '\n' if dirty else '') + \
                f'?? {path} (layout file not committed)'
    if dirty and not (args.shakedown or args.allow_dirty):
        raise SystemExit('uncommitted changes in the code under test — '
                         'counted runs need the frozen commit:\n' + dirty)
    return sha, bool(dirty)


def deadman_for_gap(worst_gap_s):
    """Deadman setting for the worst scan/odom gap seen, or None when even
    the raised value would hold the wheels (fix the link first)."""
    if worst_gap_s <= DEADMAN_RAISE_GAP_S:
        return 0.6
    if worst_gap_s <= 0.9 * RAISED_DEADMAN_S:
        return RAISED_DEADMAN_S
    return None


def cmd_check(args):
    """Read-only readiness report; writes results/real/day0/check_*.txt."""
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import BatteryState, Imu, LaserScan
    lines, verdicts = [], []

    def report(ok, text):
        mark = {True: 'ok  ', False: 'FAIL', None: 'note'}[ok]
        lines.append(f'  {mark}  {text}')
        print(lines[-1])
        if ok is False:
            verdicts.append(text)

    print('PC')
    up = 'state UP' in subprocess.run(['ip', 'link', 'show', 'eth0'],
                                      capture_output=True, text=True).stdout
    report(up, 'eth0 up (CycloneDDS is pinned to it)')
    running = subprocess.run(['pgrep', '-f', 'run\\.s[h]'],
                             capture_output=True).returncode == 0
    report(not running, 'no run.sh session competing for the robot')
    for m, path in CKPTS.items():
        report(path.exists(), f'{m} checkpoint {path.parent.name}')
    sha, dirty = git_state()
    report(None, f'code at {sha}' + (' (UNCOMMITTED changes — fine for '
                                     'Day 0, not for counted trials)'
                                     if dirty else ' (clean)'))

    print('robot link')
    robot = RobotLink(read_robot_env())
    rtts = ping_rtts(subprocess.run(
        ['ping', '-n', '-c', '10', '-i', '0.2', robot.ip],
        capture_output=True, text=True).stdout)
    report(bool(rtts), f'ping {robot.ip}: ' + (
        f'p50 {percentile(rtts, 50):.1f} ms, p95 {percentile(rtts, 95):.1f} ms'
        if rtts else 'no reply'))
    if not rtts:
        return _write_check(lines, verdicts)
    rc, _ = robot.run('true')
    report(rc == 0, 'SSH login')
    offset = robot.clock_offset_ms()
    # Robot stamps are replaced on the PC or timed on arrival, so an offset
    # only matters once large; WSL2's clock alone is often ~0.2 s off NTP.
    report(None if offset is not None and abs(offset) < CLOCK_OFFSET_FAIL_MS
           and abs(offset) >= CLOCK_OFFSET_OK_MS
           else (offset is not None and abs(offset) < CLOCK_OFFSET_OK_MS),
           f'robot clock offset {offset} ms (ok < {CLOCK_OFFSET_OK_MS:.0f}, '
           f'fails at {CLOCK_OFFSET_FAIL_MS:.0f}; run `chronyc tracking` '
           'to see whether the PC or the robot is off)')

    print('ROS')
    h = RealHarness()
    try:
        alive = h.wait_for_node('turtlebot3_node', 5)
        if not alive and not args.no_bringup:
            print('  ...  bringup not running — starting it')
            robot.start_bringup()
            alive = h.wait_for_node('turtlebot3_node', BRINGUP_TIMEOUT_S)
        report(alive, 'turtlebot3_node (bringup) running')
        if not alive:
            return _write_check(lines, verdicts)
        topics = [('/scan', LaserScan), ('/odom', Odometry), ('/imu', Imu),
                  ('/battery_state', BatteryState)]
        if args.camera:
            from sensor_msgs.msg import CompressedImage
            print('  ...  starting the BRIO (colour runs load the Pi with it)')
            robot.start_camera(args.camera_params)
            cam = h.wait_for_message('/image_raw/compressed',
                                     CompressedImage, CAMERA_WAIT_S)
            report(cam is not None, 'BRIO camera publishing')
            topics.append(('/image_raw/compressed', CompressedImage))
        rx = {t: [] for t, _ in topics}
        last = {}
        subs = [h.subscribe(t, ty, (lambda t: lambda m: (
                    rx[t].append(time.monotonic()), last.__setitem__(t, m)))(t))
                for t, ty in topics]
        print(f'  ...  listening {args.seconds:.0f} s — to measure while '
              'driving, teleop the robot now in another terminal')
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            h.spin(0.2)
        for s in subs:
            h.unsubscribe(s)
        worst = 0.0
        for topic, stamps in rx.items():
            if len(stamps) < 2:
                report(False, f'{topic}: {len(stamps)} messages')
                continue
            gaps = [b - a for a, b in zip(stamps, stamps[1:])]
            rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            text = (f'{topic}: {rate:.1f} Hz, gap p99 '
                    f'{percentile(gaps, 99) * 1000:.0f} ms, max '
                    f'{max(gaps) * 1000:.0f} ms')
            report(True, text)
            if topic in ('/scan', '/odom'):
                worst = max(worst, max(gaps))
        batt = last.get('/battery_state')
        if batt is not None:
            report(batt.voltage >= DEFAULT_MIN_BATTERY_V,
                   f'battery {batt.voltage:.2f} V (swap threshold '
                   f'{DEFAULT_MIN_BATTERY_V} V)')
        odom = last.get('/odom')
        if odom is not None:
            p = odom.pose.pose.position
            report(None, f'odom pose ({p.x:.3f}, {p.y:.3f}) — only zero '
                   'right after a bringup restart, which every trial does')
        deadman = deadman_for_gap(worst)
        if deadman is None:
            report(False, f'worst scan/odom gap {worst * 1000:.0f} ms — too '
                   'long for any deadman setting; check the link/robot load')
        else:
            report(None, f'worst scan/odom gap {worst * 1000:.0f} ms -> use '
                   f'--deadman-s {deadman} (measure idle AND driving; the '
                   'larger answer wins, then keep it for every counted '
                   'trial)')
    finally:
        h.close()
    return _write_check(lines, verdicts)


def _write_check(lines, failures):
    out = RESULTS / 'day0'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"check_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt"
    path.write_text('\n'.join(lines) + '\n')
    print(f"\n{'READY' if not failures else f'{len(failures)} FAILED'} "
          f'-> {path}')
    return 1 if failures else 0


def cmd_map_arena(args):
    """Guided mapping: fresh bringup on the start mark, Cartographer while
    the operator teleops, then save the map, resample it into the start
    frame (so arena coordinates are measured from the mark along the robot's
    heading) and extract the poles."""
    import yaml
    from evaluation import course_tools as T
    up = 'state UP' in subprocess.run(['ip', 'link', 'show', 'eth0'],
                                      capture_output=True, text=True).stdout
    if not up:
        raise SystemExit('eth0 is down — bring it up first')
    robot = RobotLink(read_robot_env())
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CompressedImage, LaserScan
    raw = RESULTS / 'arenas' / f'{args.name}_raw'
    raw.parent.mkdir(parents=True, exist_ok=True)
    h = RealHarness()
    try:
        if not args.ready:
            ask('Robot on the start mark, facing +x (into the arena)? '
                  '[Enter] ')
        volts = reset_robot(h, robot, 0.0, print)
        if args.camera:
            print('starting the BRIO (gaps are measured under its load)')
            robot.start_camera(args.camera_params)
            if h.wait_for_message('/image_raw/compressed', CompressedImage,
                                  CAMERA_WAIT_S) is None:
                print('warning: camera did not start')
        print(f'battery {volts} V; starting SLAM')
        h.popen('ros2 launch dwa_avoidance cartographer_only.launch.py '
                'config_basename:=turtlebot3_no_odom.lua',
                raw.parent / f'{args.name}_slam.log')
        deadline = time.monotonic() + SLAM_READY_TIMEOUT_S
        origin = None
        while origin is None and time.monotonic() < deadline:
            h.spin(0.2)
            origin = h.lookup('map', 'base_footprint')
        if origin is None:
            raise SystemExit('SLAM pose never appeared — see '
                             f'{raw.parent / (args.name + "_slam.log")}')
        rx = {'/scan': [], '/odom': []}
        subs = [h.subscribe(t, ty, (lambda t: lambda m: rx[t].append(
                    time.monotonic()))(t))
                for t, ty in (('/scan', LaserScan), ('/odom', Odometry))]
        print('\nSLAM running. In another terminal:\n'
              '    ros2 run turtlebot3_teleop teleop_keyboard\n'
              'Drive slowly past every pole from more than one side and '
              'along the whole boundary.\nWatch the map in RViz if you like '
              '(rviz2, fixed frame "map", add /map).', flush=True)
        if args.done_file:
            done = Path(args.done_file)
            print(f'waiting for {done} to appear...', flush=True)
            while not done.exists():
                h.spin(0.5)
            done.unlink()
        else:
            import threading
            finished = threading.Event()
            threading.Thread(target=lambda: (ask(
                '[Enter] when the map looks complete > '), finished.set()),
                daemon=True).start()
            while not finished.is_set():
                h.spin(0.5)
        for sub in subs:
            h.unsubscribe(sub)
        h.zero_wheels()
        worst = 0.0
        for topic, stamps in rx.items():
            if len(stamps) > 1:
                gaps = [b - a for a, b in zip(stamps, stamps[1:])]
                worst = max(worst, max(gaps))
                print(f'{topic} while driving: '
                      f'{(len(stamps) - 1) / (stamps[-1] - stamps[0]):.1f} Hz,'
                      f' gap p99 {percentile(gaps, 99) * 1000:.0f} ms, max '
                      f'{max(gaps) * 1000:.0f} ms')
        rec = deadman_for_gap(worst)
        print(f'worst gap while driving {worst * 1000:.0f} ms -> '
              + (f'--deadman-s {rec}' if rec else 'too long for any deadman '
                 'setting; check the link'), flush=True)
        saver = h.popen(f'ros2 run nav2_map_server map_saver_cli -f {raw} '
                        '--ros-args -p save_map_timeout:=10.0',
                        raw.parent / f'{args.name}_saver.log')
        saver.wait(timeout=40)
        Path(f'{raw}_start_pose.yaml').write_text(yaml.safe_dump(
            {'x': origin[0], 'y': origin[1], 'yaw': origin[2],
             'note': 'robot pose in the raw map when SLAM started'}))
        print(f'SLAM start pose in the raw map: ({origin[0]:.3f}, '
              f'{origin[1]:.3f}, {origin[2]:.3f} rad) — resampling the map '
              'into the start frame')
        start_map = A.map_to_start_frame(f'{raw}.yaml', origin,
                                         Path(f'{raw}_start.yaml'))
    finally:
        h.close()
    out, poles = T.arena_from_map(start_map, args.name, args.id,
                                  args.pole_diameter)
    png = T.render_arena(A.load(out), 'P1',
                         T.ARENA_RENDER_DIR / f'{args.name}.png')
    print(f'\n{len(poles)} pole candidates -> {out}\nfigure: {png}\n'
          'next: delete any line that is not a course pole, tape-check a '
          'few poles against the figure, then add the squares under '
          'placements and run course_tools.py arena-measure')


def run_session(args, *, label, layout_paths, out, csv_path, fields,
                todo, make_task, prompt, result_line, after_row, review=None):
    """Operator loop shared by both tasks: one run per Enter until nothing
    is owed. Rows are appended per run, so a session can die without loss.

    todo(done_rows) -> cells still owed, in order; make_task(cell) -> task."""
    sha, dirty = preflight(args, layout_paths)
    out.mkdir(parents=True, exist_ok=True)
    robot = RobotLink(read_robot_env())
    session_id = datetime.now().strftime('%Y-%m-%d_%H%M')
    session = {'session_id': session_id, 'run_id': args.run_id,
               'course': label, 'lighting': getattr(args, 'lighting', ''),
               'shakedown': args.shakedown,
               'start': datetime.now().isoformat(timespec='seconds'),
               'git_sha': sha, 'git_dirty': dirty, 'link': args.link,
               'clock_offset_ms': robot.clock_offset_ms()}
    rtts = ping_rtts(subprocess.run(
        ['ping', '-n', '-c', '10', '-i', '0.2', robot.ip],
        capture_output=True, text=True).stdout)
    if rtts:
        session['ping_p50_ms'] = round(percentile(rtts, 50), 1)
        session['ping_p95_ms'] = round(percentile(rtts, 95), 1)
    print(f"session {session_id}: {label}, commit {sha}"
          f"{' (dirty)' if dirty else ''}, clock offset "
          f"{session['clock_offset_ms']} ms, ping p95 "
          f"{session.get('ping_p95_ms')} ms")

    h = RealHarness()
    runs = 0
    try:
        while True:
            pending = todo(read_rows(csv_path))
            if not pending:
                print('nothing left to run here.')
                break
            cell = pending[0]
            print(f"\nnext: #{cell['order_idx']} {cell['model']} @ "
                  f"{float(cell['hz']):g} Hz, round {cell['round']}  "
                  f"({len(pending)} left)")
            try:
                answer = ask(f'  {prompt} [Enter] start · type quit to end '
                               'the session > ').strip().lower()
            except EOFError:
                break
            if answer == 'quit': # a whole word: a stray key cannot end it
                break
            if answer:
                print(f'  ({answer!r} ignored: Enter starts the run, '
                      '"quit" ends the session)')
                continue
            tag = (f"t{next_trial_index(out / 'trials'):03d}_"
                   f"{label.replace('/', '_')}_{cell['model']}_"
                   f"{float(cell['hz']):g}hz_r{cell['round']}")
            t_wall = datetime.now().isoformat(timespec='seconds')
            try:
                row = run_task(h, robot, make_task(cell),
                               out / 'trials' / tag, args)
            except BatteryLow as e:
                print(f'  {e} — swap the battery, then continue')
                continue
            if session.get('battery_v_start') is None:
                session['battery_v_start'] = row.get('battery_v_start')
            row.update({'run_id': args.run_id, 'session_id': session_id,
                        'order_idx': cell['order_idx'], 't_start_wall': t_wall,
                        'ckpt': (CKPTS[cell['model']].parent.name
                                 if cell['model'] in CKPTS else 'dwa'),
                        'git_sha': sha, 'git_dirty': dirty})
            # saved before the prompts: a session dying there keeps the run
            append_row(csv_path, fields, row)
            runs += 1
            if review is not None:
                review(csv_path, row, out / 'trials' / tag)
            try:
                late = (ask('  contacts you could not key in during the run '
                              '(number, Enter for none) > ').strip()
                        if 'collisions' in row else '')
                notes = ask('  notes (Enter to skip) > ').strip()
            except EOFError:
                late, notes = '', ''
            if late.isdigit() and int(late) > 0:
                add_late_collisions(csv_path, int(late))
            if notes:
                set_last_notes(csv_path, notes)
            print(f'  -> {result_line(row)}  (logged to {csv_path.name})')
            after_row(row)
            if row.get('verdict_reason') == INTERRUPTED:
                print('session ended by Ctrl-C')
                break
    except KeyboardInterrupt:
        print('\ninterrupted — stopping the robot')
        h.zero_wheels()
    finally:
        h.close()
        session.update({'end': datetime.now().isoformat(timespec='seconds'),
                        'trials_run': runs})
        append_row(out / 'sessions.csv', SESSION_FIELDS, session)


def cmd_run(args):
    course = C.load(args.course)
    schedule = (shakedown_cells(course) if args.shakedown
                else load_schedule(course, args.run_id))

    def todo(rows):
        done = [r for r in rows if r['course'] == course['name']]
        return (schedule[len(done):] if args.shakedown
                else pending_cells(schedule, done))

    def after(row):
        if row['verdict'] == 'collision':
            print('  put any moved tube back on its mark')

    run_session(
        args, label=course['name'], layout_paths=[course['path']],
        out=run_dir(args.run_id, args.shakedown),
        csv_path=run_dir(args.run_id, args.shakedown) / 'trials_w0.csv',
        fields=ROW_FIELDS, todo=todo,
        make_task=lambda cell: GotoTask(course, cell, args),
        prompt='robot on the start mark, tubes on their marks?',
        result_line=lambda r: (f"{r['verdict']}  {r['time_to_goal_s']} s  "
                               f"path {r['gt_path_len']} m"),
        after_row=after)


def cmd_colour_schedule(args):
    arena = A.load(args.arena)
    colors = A.placement_colors(arena, args.placement)
    if not colors:
        raise SystemExit(f'placement {args.placement} has no squares yet')
    A.lighting_config(arena, args.lighting, args.placement)
    out = colour_schedule_path(arena, args.placement, args.run_id,
                               args.lighting)
    if out.exists() and not args.force:
        raise SystemExit(f'{out} exists — the order is fixed once written')
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = make_colour_schedule(arena, args.placement, args.run_id,
                                args.rounds, args.lighting)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs ({', '.join(colors)}) -> {out}")


def colour_summary(rows):
    """Per-model summary of counted colour runs."""
    out = []
    for m in COLOUR_MODELS:
        g = [r for r in rows if r['model'] == m
             and r['verdict'] in COLOUR_COUNTED]
        if not g:
            continue
        wins = [float(r['total_s']) for r in g if r['verdict'] == 'success']
        out.append({
            'model': m, 'runs': len(g), 'success': len(wins),
            'success_%': round(100 * len(wins) / len(g), 1),
            'median_total_s': (round(percentile(wins, 50), 1)
                               if wins else None),
            'mean_found': round(sum(int(r['found_count']) for r in g)
                                / len(g), 2),
            'collisions_per_run': round(sum(int(r['collisions'] or 0)
                                            for r in g) / len(g), 2),
            'median_path_m': round(percentile(
                [float(r['path_m']) for r in g], 50), 1),
        })
    return out


def cmd_colour_status(args):
    arena = A.load(args.arena)
    rows = [r for r in read_rows(run_dir(args.run_id, args.shakedown)
                                 / 'colour_trials.csv')
            if r['arena'] == arena['name']
            and r['placement'] == args.placement
            and r.get('lighting', '') == args.lighting]
    schedule = load_colour_schedule(arena, args.placement, args.run_id,
                                    args.lighting)
    pending = colour_pending(schedule, rows)
    counted = [r for r in rows if r['verdict'] in COLOUR_COUNTED]
    print(f"arena {arena['name']} {args.placement} ({args.lighting}): "
          f"{len(counted)} counted, "
          f"{len(rows) - len(counted)} re-run attempts, "
          f"{len(pending)}/{len(schedule)} pending")
    for srow in colour_summary(rows):
        print('  ' + '  '.join(f'{k} {v}' for k, v in srow.items()))


def cmd_colour_run(args):
    arena = A.load(args.arena)
    colors = A.placement_colors(arena, args.placement)
    if not colors:
        raise SystemExit(f'placement {args.placement} has no squares yet')
    A.lighting_config(arena, args.lighting, args.placement)
    out = run_dir(args.run_id, args.shakedown)
    csv_path = out / 'colour_trials.csv'
    practice = [0] # runs made in this session
    if args.shakedown:
        schedule = practice_cells(arena, args.placement, args.lighting,
                                  args.models, args.repeat)
    else:
        schedule = load_colour_schedule(arena, args.placement, args.run_id,
                                        args.lighting)

    def todo(rows):
        if args.shakedown:
            return schedule[practice[0]:]
        done = [r for r in rows if r['arena'] == arena['name']
                and r['placement'] == args.placement
                and r.get('lighting', '') == args.lighting]
        return colour_pending(schedule, done)

    def after(row):
        practice[0] += 1
        if int(row['collisions'] or 0):
            print('  put any moved pole back on its mark')

    layout_files = [arena['path']] + ([arena['map'], arena['map'].with_suffix(
        '.pgm')] if arena['map'] else [])
    run_session(
        args, label=f"{arena['name']}/{args.placement}/{args.lighting}",
        layout_paths=layout_files, out=out, csv_path=csv_path,
        fields=COLOUR_ROW_FIELDS, todo=todo,
        make_task=lambda cell: ColourTask(arena, args.placement, cell, args),
        review=review_finds,
        prompt=f"robot on the start mark, squares ({', '.join(colors)}) "
               f'in place, {args.lighting} lighting?',
        result_line=lambda r: (f"{r['verdict']}  {r['found_count']}/"
                               f"{len(colors)} found  {r['total_s']} s  "
                               f"path {r['path_m']} m  "
                               f"{r['collisions']} collisions"),
        after_row=after)


def main():
    _remember_tty()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--seconds', type=float, default=60.0)
    c.add_argument('--no-bringup', action='store_true',
                   help='report instead of starting bringup if it is down')
    c.add_argument('--camera', action='store_true',
                   help='start the BRIO first, so gaps are measured under '
                        'the colour runs\' load')
    c.add_argument('--camera-params', default='')
    for name in ('schedule', 'run', 'status'):
        s = sub.add_parser(name)
        s.add_argument('--course', required=True, help='layout yaml')
        s.add_argument('--run-id', required=True)
        if name != 'schedule':
            s.add_argument('--shakedown', action='store_true',
                           help='uncounted rehearsal trials, separate folder')
    sub.choices['schedule'].add_argument('--rounds', type=int, default=ROUNDS)
    sub.choices['schedule'].add_argument('--force', action='store_true')
    r = sub.choices['run']
    r.add_argument('--min-battery-v', type=float,
                   default=DEFAULT_MIN_BATTERY_V)
    r.add_argument('--deadman-s', type=float, default=0.6,
                   help='policy sensor-age deadman (decide on Day 0)')
    r.add_argument('--link', default='unknown',
                   help='robot network link for the session log '
                        '(e.g. cable, wifi)')
    r.add_argument('--allow-dirty', action='store_true',
                   help='permit counted trials on uncommitted code')
    m = sub.add_parser('map-arena')
    m.add_argument('--name', required=True)
    m.add_argument('--id', type=int, required=True, help='>= 9100')
    m.add_argument('--pole-diameter', type=float, default=0.04445)
    m.add_argument('--ready', action='store_true',
                   help='robot is already on the start mark; skip the prompt')
    m.add_argument('--done-file',
                   help='finish when this file appears instead of on Enter')
    m.add_argument('--camera', action='store_true',
                   help='run the BRIO while mapping (realistic load, and '
                        'lets you watch the feed)')
    m.add_argument('--camera-params', default='')
    for name in ('colour-schedule', 'colour-run', 'colour-status'):
        s = sub.add_parser(name)
        s.add_argument('--arena', required=True, help='arena yaml')
        s.add_argument('--placement', required=True)
        s.add_argument('--run-id', required=True)
        s.add_argument('--lighting', required=True,
                       help="lighting profile in the arena file (e.g. day, "
                            "night); picks the colour bands and camera lock")
        if name != 'colour-schedule':
            s.add_argument('--shakedown', action='store_true',
                           help='one uncounted run per model, own folder')
    cs = sub.choices['colour-schedule']
    cs.add_argument('--rounds', type=int, default=COLOUR_ROUNDS)
    cs.add_argument('--force', action='store_true')
    cr = sub.choices['colour-run']
    cr.add_argument('--cap-min', type=float, default=DEFAULT_CAP_MIN,
                    help='time cap per run [min]')
    cr.add_argument('--models', default=None,
                    help='with --shakedown: which models to practise, comma '
                         'separated (default: both)')
    cr.add_argument('--repeat', type=int, default=1,
                    help='with --shakedown: runs per model')
    cr.add_argument('--show', action='store_true',
                    help='open the camera and detector views (boxes, labels '
                         'and a live mask per colour) for the session')
    cr.add_argument('--camera-params', default='',
                    help='extra -p settings for v4l2_camera_node, e.g. '
                         'exposure / white-balance locks')
    cr.add_argument('--min-battery-v', type=float,
                    default=DEFAULT_MIN_BATTERY_V)
    cr.add_argument('--deadman-s', type=float, default=0.6,
                    help='policy sensor-age deadman (decide on Day 0)')
    cr.add_argument('--link', default='unknown')
    cr.add_argument('--allow-dirty', action='store_true')
    args = ap.parse_args()
    if args.cmd == 'check':
        sys.exit(cmd_check(args))
    if args.cmd == 'map-arena':
        return cmd_map_arena(args)
    {'schedule': cmd_schedule, 'run': cmd_run, 'status': cmd_status,
     'colour-schedule': cmd_colour_schedule, 'colour-run': cmd_colour_run,
     'colour-status': cmd_colour_status}[args.cmd](args)


if __name__ == '__main__':
    main()
