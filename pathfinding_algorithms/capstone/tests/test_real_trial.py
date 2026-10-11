"""Offline tests for the real-robot trial runner and the course tools.

No robot and no ROS graph: covers the parts that decide what a trial
MEANS — schedule, verdict rules, parsers, course geometry — so a protocol
mistake shows up here rather than halfway through a lab session.

  .venv/bin/python tests/test_real_trial.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path

CAPSTONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPSTONE))

import numpy as np                                            # noqa: E402

from barn import world_metrics as W                           # noqa: E402
from common import arena as A                                 # noqa: E402
from common import course as C                                # noqa: E402
from evaluation import real_trial as R                        # noqa: E402
from evaluation import session_report as S                     # noqa: E402

COURSES = CAPSTONE / 'evaluation' / 'courses'


def _course(**over):
    raw = {'name': 'T', 'id': 9999, 'walls': {'back_x': -0.6, 'side_y': 1.35,
                                              'end_x': 4.8},
           'clutter_x': [0.6, 3.0], 'goal': [4.2, 0.0],
           'tubes': [[1.2, 0.3], [1.8, -0.45]]}
    raw.update(over)
    return C.from_dict(raw)


def _raises(fn, exc=C.CourseError):
    try:
        fn()
    except exc:
        return True
    return False


# ── schedule ────────────────────────────────────────────────────────────────
def test_schedule_is_complete_blocks():
    rows = R.make_schedule(_course(), 'bench_v1')
    assert len(rows) == 27
    for rnd in range(3):
        cells = {(r['model'], r['hz']) for r in rows if r['round'] == rnd}
        assert len(cells) == 9, 'every round must run all 9 cells once'
    assert [r['order_idx'] for r in rows] == list(range(27))


def test_schedule_is_reproducible_and_seeded():
    a = R.make_schedule(_course(), 'bench_v1')
    b = R.make_schedule(_course(), 'bench_v1')
    c = R.make_schedule(_course(), 'bench_v2')
    assert a == b
    assert [(r['model'], r['hz']) for r in a] != \
        [(r['model'], r['hz']) for r in c]


def test_pending_orders_fresh_before_retries():
    sched = R.make_schedule(_course(), 'bench_v1')
    done = [dict(model=sched[0]['model'], world_idx=9999,
                 hz=sched[0]['hz'], trial=0, verdict='success'),
            dict(model=sched[1]['model'], world_idx=9999,
                 hz=sched[1]['hz'], trial=0, verdict='harness_error')]
    pend = R.pending_cells(sched, done)
    assert sched[0] not in pend, 'a counted cell is finished'
    assert pend[-1] == sched[1], 'a harness error is retried at the end'
    assert pend[0] == sched[2]
    assert len(pend) == 26
    done[1]['verdict'] = 'collision'
    done.append(dict(done[1], verdict='success'))   # later row wins
    assert len(R.pending_cells(sched, done)) == 25


# ── verdict rules ───────────────────────────────────────────────────────────
def _drive(mon, t0=0.0, speed=0.2, until=None, dt=0.1):
    """Robot driving straight at the goal, 10 Hz scan/SLAM/odom."""
    mon.arm(t0)
    mon.on_cmd(t0 + 0.5)
    t = t0 + 0.5
    x = 0.0
    while mon.check(t) is None and (until is None or t < until):
        t += dt
        x += speed * dt
        mon.on_scan(t)
        mon.on_odom(t, x * 1.02, 0.0, 0.0)   # odom over-reads by 2 %
        mon.on_slam(t, x, 0.0, 0.0)
    return t


def test_success_when_slam_enters_goal_circle():
    mon = R.TrialMonitor((4.2, 0.0), timeout_s=120)
    _drive(mon)
    s = mon.summary()
    assert s['verdict'] == 'success', s
    assert abs(s['gt_path_len'] - 3.2) < 0.05, s
    assert abs(s['time_to_goal_s'] - 16.0) < 0.2, s
    assert s['progress'] > 0.75
    assert 0.03 < s['odom_drift_m'] < 0.1


def test_timeout_is_counted_from_first_command():
    mon = R.TrialMonitor((4.2, 0.0), timeout_s=10)
    _drive(mon, speed=0.01)
    s = mon.summary()
    assert s['verdict'] == 'timeout' and abs(s['time_to_goal_s'] - 10) < 0.2


def test_operator_keys():
    for key, verdict in (('c', 'collision'), ('a', 'aborted'),
                         ('x', 'harness_error')):
        mon = R.TrialMonitor((4.2, 0.0), timeout_s=120)
        mon.arm(0.0)
        mon.on_cmd(1.0)
        mon.on_key(5.0, key)
        assert mon.check(5.1) == verdict
        assert mon.summary()['verdict_reason'] == 'operator'
    mon = R.TrialMonitor((4.2, 0.0), timeout_s=120)
    mon.arm(0.0)
    mon.on_key(1.0, 'z')
    assert mon.check(1.1) is None, 'unknown keys are ignored'


def test_harness_faults_are_not_model_failures():
    mon = R.TrialMonitor((4.2, 0.0), timeout_s=120, start_timeout_s=45)
    mon.arm(0.0)
    assert mon.check(44.0) is None
    assert mon.check(46.0) == 'harness_error'

    mon = R.TrialMonitor((4.2, 0.0), timeout_s=120)
    t = _drive(mon, until=5.0)
    mon.on_slam(t, 1.0, 0.0, 0.0)
    assert mon.check(t + 2.5) == 'harness_error'
    assert 'lidar' in mon.reason

    mon = R.TrialMonitor((4.2, 0.0), timeout_s=120)
    mon.arm(0.0)
    mon.on_controller_exit(3.0)
    assert mon.verdict == 'harness_error', 'died while loading'
    mon = R.TrialMonitor((4.2, 0.0), timeout_s=120)
    mon.arm(0.0)
    mon.on_cmd(1.0)
    mon.on_controller_exit(3.0)
    assert mon.verdict == 'crash', 'died while driving counts as failure'


def test_commands_before_the_goal_do_not_start_the_clock():
    mon = R.TrialMonitor((4.2, 0.0), timeout_s=120)
    mon.on_cmd(0.1)
    mon.on_cmd(0.2)
    mon.arm(1.0)
    mon.on_cmd(1.5)
    assert mon.t0 == 1.5 and mon.summary()['pre_goal_cmds'] == 2


def test_jitter_adds_no_path_length():
    mon = R.TrialMonitor((4.2, 0.0), timeout_s=120)
    mon.arm(0.0)
    mon.on_cmd(0.0)
    rng = np.random.default_rng(0)
    for k in range(600):   # a minute parked, +-5 mm SLAM jitter
        mon.on_slam(k * 0.1, *rng.normal(0, 0.005, 2), 0.0)
    assert mon.summary()['gt_path_len'] < 0.05


def test_first_verdict_wins():
    mon = R.TrialMonitor((4.2, 0.0), timeout_s=120)
    mon.arm(0.0)
    mon.on_cmd(0.5)
    mon.on_key(2.0, 'c')
    mon.on_slam(2.1, 4.2, 0.0, 0.0)
    assert mon.verdict == 'collision'


def test_remote_background_commands_release_the_ssh_channel():
    """`a && b &` keeps the caller's pipes open until b exits, which hung
    every SSH call that started bringup or the camera (found on the robot).
    Run both forms locally with captured pipes, exactly as SSH sees them;
    each runs in its own process group, which is killed afterwards."""
    import os
    import signal
    import subprocess

    def returns_quickly(cmd):
        proc = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                start_new_session=True)
        try:
            proc.communicate(timeout=1.5)
            return True
        except subprocess.TimeoutExpired:
            return False
        finally:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    assert returns_quickly(R.detached('true && exec sleep 30', '/dev/null'))
    assert not returns_quickly('true && nohup sleep 30 > /dev/null 2>&1 '
                               '</dev/null &'), 'old form should hang'
    for cmd in (R.ROBOT_KILL, R.ROBOT_BRINGUP):
        assert cmd.startswith('(') and cmd.endswith('</dev/null &'), cmd


def test_trial_folders_are_never_reused():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert R.next_trial_index(d / 'missing') == 0
        (d / 't000_room1_P1_two_stage_4hz_r0').mkdir()   # interrupted run,
        (d / 't003_room1_P1_single_stage_4hz_r1').mkdir()  # no result row
        (d / 'notes.txt').write_text('x')
        assert R.next_trial_index(d) == 4


def test_clock_offset_uses_the_return_time():
    # remote read its clock at the end of a 1.3 s call; true offset +30 ms
    t_call, setup = 1000.0, 1.3
    remote = t_call + setup + 0.030
    t_return = t_call + setup + 0.001
    assert abs(R.offset_from_return(remote, t_return) - 0.029) < 1e-9


def test_rows_stay_aligned_when_fields_are_added():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'rows.csv'
        R.append_row(path, ['model', 'verdict'], {'model': 'a', 'verdict': 'ok'})
        R.append_row(path, ['model', 'lighting', 'verdict'],   # field added
                     {'model': 'b', 'lighting': 'night', 'verdict': 'timeout'})
        R.append_row(path, ['verdict', 'model'],               # reordered
                     {'model': 'c', 'verdict': 'stuck'})
        rows = R.read_rows(path)
        assert [(r['model'], r['lighting'], r['verdict']) for r in rows] == [
            ('a', '', 'ok'), ('b', 'night', 'timeout'), ('c', '', 'stuck')]


def test_notes_are_added_to_the_saved_row():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'rows.csv'
        R.append_row(path, ['model', 'notes'], {'model': 'a'})
        R.append_row(path, ['model', 'notes'], {'model': 'b'})
        R.set_last_notes(path, 'tether snagged')
        assert [(r['model'], r['notes']) for r in R.read_rows(path)] == [
            ('a', ''), ('b', 'tether snagged')]


def test_a_new_field_is_added_to_the_saved_row():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'rows.csv'
        R.append_row(path, ['model'], {'model': 'a'})
        R.append_row(path, ['model'], {'model': 'b'})
        R.set_last_field(path, 'false_found', 'sky_blue')
        assert [(r['model'], r['false_found']) for r in R.read_rows(path)] == [
            ('a', ''), ('b', 'sky_blue')]


def test_note_sites_cluster_and_rank_by_finds():
    rng = np.random.default_rng(0)
    a = rng.normal((1.2, -0.5), 0.04, (40, 2))
    b = rng.normal((1.2, 1.2), 0.04, (10, 2))
    pts = lambda xy: [dict(color='sky_blue', x=x, y=y) for x, y in xy]  # noqa: E731
    runs = [dict(label='#0', points=pts(b), finds={'sky_blue': (1.2, 1.2)}),
            dict(label='#1', points=pts(a[:20]), finds={'sky_blue': (1.2, -0.5)}),
            dict(label='#2', points=pts(a[20:]), finds={'sky_blue': (1.2, -0.5)})]
    sites = S.note_sites(runs, {'sky_blue': (1.2, 1.2)})
    assert [(s['site'], s['found_by']) for s in sites] == [
        ('sky_blue:A', ['#1', '#2']), ('sky_blue:B', ['#0'])]
    assert sites[0]['from_recorded_m'] > 1.5 and sites[1]['from_recorded_m'] < 0.1


def test_wilson_interval():
    lo, hi = S.wilson(5, 5)
    assert 0.56 < lo < 0.57 and hi == 1.0
    lo, hi = S.wilson(2, 5)
    assert 0.11 < lo < 0.12 and 0.76 < hi < 0.77


def test_late_collisions_add_to_the_count():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'rows.csv'
        R.append_row(path, ['model', 'collisions'], {'model': 'a', 'collisions': 2})
        R.add_late_collisions(path, 3)
        r = R.read_rows(path)[-1]
        assert r['collisions'] == '5' and r['collisions_late'] == '3'


def test_unreachable_robot_does_not_raise():
    link = R.RobotLink({'ROBOT_IP': '203.0.113.1', 'ROBOT_USER': 'nobody'})
    rc, out = link.run('true', timeout=1)             # TEST-NET: never answers
    assert rc != 0 and out == ''


def test_deadman_recommendation():
    assert R.deadman_for_gap(0.2) == 0.6
    assert R.deadman_for_gap(0.5) == 0.6
    assert R.deadman_for_gap(0.7) == 1.0
    assert R.deadman_for_gap(0.95) is None, 'even 1.0 s would hold wheels'


# ── parsers ─────────────────────────────────────────────────────────────────
def test_relative_pose():
    x, y, yaw = R.relative_pose((1.0, 1.0, math.pi / 2), (1.0, 2.0, math.pi))
    assert abs(x - 1.0) < 1e-9 and abs(y) < 1e-9
    assert abs(yaw - math.pi / 2) < 1e-9


def test_percentile_and_ping():
    assert R.percentile([], 95) is None
    assert R.percentile([1, 2, 3, 4, 5], 50) == 3
    text = ('64 bytes from 10.0.0.2: icmp_seq=1 ttl=64 time=1.52 ms\n'
            '64 bytes from 10.0.0.2: icmp_seq=2 ttl=64 time=12.0 ms\n')
    assert R.ping_rtts(text) == [1.52, 12.0]


def test_latency_stats_reads_new_columns():
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
        f.write('sim_t,plan_ms,sample_ms,turn_pm,clear,scan_age_ms,'
                'odom_age_ms\n')
        for k in range(20):
            f.write(f'{k},5.0,40.0,0.1,0.5,{k * 10.0},3.0\n')
    s = R.latency_stats(f.name)
    assert s['lat_sample_ms'] == 40.0 and s['lat_max_ms'] == 45.0
    assert 170 < s['scan_age_p95_ms'] < 190
    assert R.latency_stats('/nonexistent.csv') == {}


def test_latency_log_header_matches_row():
    node = (CAPSTONE / 'deploy' / 'policy_node.py').read_text()
    track = (CAPSTONE / 'deploy' / 'policy' / 'tracking.py').read_text()
    header = node.split("self._lat_f.write('")[1].split("\\n')")[0]
    header = header.replace("'\n", '').replace("'", '').replace(' ', '')
    row = track.split('self._lat_f.write(')[1].split("\\n')")[0]
    assert header.count(',') == row.count("},"), (header, row)


# ── courses ─────────────────────────────────────────────────────────────────
def test_course_validation():
    assert _raises(lambda: _course(tubes=[[1.23, 0.3]])), 'off lattice'
    assert _raises(lambda: _course(tubes=[[0.15, 0.0]])), 'outside field'
    assert _raises(lambda: _course(id=12)), 'BARN index range'
    assert _raises(lambda: _course(start=[0.5, 0.0, 0.0])), 'start not origin'
    assert _raises(lambda: _course(tubes=[[1.2, 0.3], [1.2, 0.3]])), 'dup'
    assert _raises(lambda: _course(tubes=[[1.2, 1.35]])), 'in the wall'
    assert _course(tubes=[[1.205, 0.3]])['tubes'][0][0] == 1.2, 'snaps 5 mm'


def test_layout_path_is_absolute_and_tracked():
    import os
    import subprocess
    cwd = os.getcwd()
    try:
        os.chdir(CAPSTONE)
        c = C.load('evaluation/courses/A.yaml')   # relative, as on the CLI
    finally:
        os.chdir(cwd)
    assert c['path'].is_absolute()
    rc = subprocess.run(['git', '-C', str(R.WS), 'ls-files',
                         '--error-unmatch', str(c['path'])],
                        capture_output=True).returncode
    assert rc == 0, 'preflight must see the committed layout as tracked'


def test_course_grid_and_points():
    c = _course()
    grid = C.arena_grid(c)
    assert grid.shape == (19, 19)   # 17 interior + ring, both axes
    assert grid[1:-1, 1:-1].sum() == 2
    pts = C.obstacle_points(c)
    assert len(pts) == 2 + len(C.wall_points(c))
    assert (np.abs(C.wall_points(c)[:, 1]) <= 1.35 + 1e-9).all()
    assert C.timeout_s(4.67) == 120 and C.timeout_s(5.1) == 130


def test_draft_courses_measure_in_their_target_tiers():
    rows = json.loads(W.OUT.read_text())
    for name in ('A', 'B', 'C'):
        c = C.load(COURSES / f'{name}.yaml')
        _, _, tier = W.place(W.measure_grid(C.arena_grid(c)), rows)
        assert tier == c['tier_target'], (name, tier)
        path, length, _ = C.optimal_route(c)
        assert path is not None and length >= 4.2


def test_place_reproduces_saved_world_scores():
    rows = json.loads(W.OUT.read_text())
    for r in rows[::101]:
        m = W.measure(W.GRIDS / f"grid_{r['world']:03d}.npy")
        score, rank, tier = W.place(m, rows)
        assert abs(score - r['score']) < 1e-9 and rank == r['rank'] \
            and tier == r['tier']


def test_subgoal_planner_reads_a_course_layout():
    from deploy.subgoal_planner import SubgoalPlanner
    c = C.load(COURSES / 'C.yaml')
    sp = SubgoalPlanner(c['path'], c['goal'])
    sub = sp.subgoal((0.0, 0.0))
    assert sub is not None
    assert 0.9 < math.hypot(*sub) < 1.25, sub   # 1.2 m along the route
    route, _, _ = C.optimal_route(c)
    clear = min(np.hypot(*(C.obstacle_points(c) - p).T).min() for p in route)
    assert clear > 0.2, 'the route keeps the planning inflation'


# ── colour task ─────────────────────────────────────────────────────────────
COLS = ['light_green', 'purple', 'orange']


def _status(**states):
    return {'states': {c: states.get(c, 'UNSEEN') for c in COLS}}


def test_colour_success_when_all_found():
    mon = R.ColourMonitor(COLS, cap_s=900)
    mon.arm(0.0)
    mon.on_cmd(2.0)
    mon.on_status(30.0, _status(purple='SEEN'))
    mon.on_status(40.0, _status(purple='FOUND', orange='SEEN'))
    mon.on_status(90.0, _status(purple='FOUND', orange='FOUND'))
    assert mon.check(91.0) is None
    mon.on_status(300.0, _status(purple='FOUND', orange='FOUND',
                                 light_green='FOUND'))
    s = mon.summary()
    assert s['verdict'] == 'success' and s['found_count'] == 3
    assert s['total_s'] == 298.0
    assert s['purple_seen_s'] == 28.0 and s['purple_found_s'] == 38.0
    assert s['light_green_seen_s'] == 298.0, 'FOUND implies SEEN'


def test_colour_timeout_keeps_partial_finds():
    mon = R.ColourMonitor(COLS, cap_s=900)
    mon.arm(0.0)
    mon.on_cmd(1.0)
    for t in range(2, 1000):
        mon.on_scan(float(t))
        mon.on_slam(float(t), 0.0, 0.0, 0.0)
        if t == 100:
            mon.on_status(t, _status(orange='FOUND'))
        if mon.check(float(t)):
            break
    s = mon.summary()
    assert s['verdict'] == 'timeout' and s['found_count'] == 1
    assert s['orange_found_s'] == 99.0 and s['purple_found_s'] == ''


def test_colour_collisions_do_not_end_the_run():
    mon = R.ColourMonitor(COLS, cap_s=900)
    mon.arm(0.0)
    mon.on_cmd(0.0)
    mon.on_key(10.0, 'c')
    mon.on_key(25.0, 'c')
    assert mon.check(26.0) is None
    mon.on_key(40.0, 'w')
    s = mon.summary()
    assert s['verdict'] == 'stuck' and s['collisions'] == 2
    assert s['collision_times_s'] == '10.0 25.0'
    assert 'stuck' in R.COLOUR_COUNTED, 'wedged runs are counted failures'


def test_colour_finds_before_the_clock_count_from_first_command():
    mon = R.ColourMonitor(COLS, cap_s=900)
    mon.arm(0.0)
    mon.on_status(1.0, _status(light_green='FOUND', purple='FOUND',
                               orange='FOUND'))
    assert mon.verdict is None, 'no success before the controller commands'
    mon.on_cmd(3.0)
    mon.on_status(4.0, _status(light_green='FOUND', purple='FOUND',
                               orange='FOUND'))
    assert mon.verdict == 'success'


def test_colour_schedule_blocks():
    arena = {'name': 'room1', 'id': 9101}
    rows = R.make_colour_schedule(arena, 'P1', 'colour_v1', rounds=5,
                                  lighting='day')
    night = R.make_colour_schedule(arena, 'P1', 'colour_v1', rounds=5,
                                   lighting='night')
    assert {r['lighting'] for r in night} == {'night'}
    assert len(rows) == 10
    for rnd in range(5):
        assert sorted(r['model'] for r in rows if r['round'] == rnd) == \
            sorted(R.COLOUR_MODELS)
    assert rows == R.make_colour_schedule(arena, 'P1', 'colour_v1', rounds=5,
                                          lighting='day')
    done = [dict(model=rows[0]['model'], placement='P1', lighting='day',
                 trial=0, verdict='aborted')]
    pend = R.colour_pending(rows, done)
    assert pend[0] == rows[0] and len(pend) == 10, 're-run keeps its place'
    done[0]['verdict'] = 'stuck'
    assert len(R.colour_pending(rows, done)) == 9
    assert len(R.colour_pending(night, done)) == 10, 'day runs never fill night'


def test_practice_runs_are_per_session_and_selectable():
    arena = {'name': 'room1', 'id': 9101}
    both = R.practice_cells(arena, 'P1', 'night')
    assert [c['model'] for c in both] == ['two_stage', 'single_stage']
    two = R.practice_cells(arena, 'P1', 'night', 'two_stage', repeat=3)
    assert [c['model'] for c in two] == ['two_stage'] * 3
    assert [c['round'] for c in two] == [0, 1, 2]
    assert _raises(lambda: R.practice_cells(arena, 'P1', 'night', 'dwa'),
                   SystemExit)


def test_colour_summary():
    rows = [dict(model='two_stage', verdict='success', total_s='300',
                 found_count='3', collisions='1', path_m='40'),
            dict(model='two_stage', verdict='timeout', total_s='900',
                 found_count='2', collisions='0', path_m='80'),
            dict(model='single_stage', verdict='aborted', total_s='50',
                 found_count='0', collisions='0', path_m='5')]
    (ts,) = R.colour_summary(rows)
    assert ts['runs'] == 2 and ts['success_%'] == 50.0
    assert ts['median_total_s'] == 300.0 and ts['mean_found'] == 2.5


def _synthetic_arena_map(folder, poles, pole_r=0.0222):
    """6 x 4 m room: 2-cell walls, a sofa block, thin poles as surface rings."""
    res, ox, oy, w, h = 0.05, -1.0, -2.0, 120, 80
    img = np.full((h, w), 254, dtype=np.uint8)
    img[:2, :] = img[-2:, :] = img[:, :2] = img[:, -2:] = 0
    xs = ox + (np.arange(w) + 0.5) * res
    ys = oy + (h - np.arange(h) - 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    img[(gx > 3.6) & (gx < 4.6) & (gy > 0.8) & (gy < 1.6)] = 0   # sofa
    for px, py in poles:
        d = np.hypot(gx - px, gy - py)
        img[d < pole_r + 0.03] = 0
    pgm = folder / 'syn.pgm'
    pgm.write_bytes(f'P5\n{w} {h}\n255\n'.encode() + img.tobytes())
    (folder / 'syn.yaml').write_text(
        f'image: syn.pgm\nmode: trinary\nresolution: {res}\n'
        f'origin: [{ox}, {oy}, 0]\nnegate: 0\noccupied_thresh: 0.65\n'
        'free_thresh: 0.25\n')
    return folder / 'syn.yaml'


def test_arena_pole_extraction_and_measure():
    from evaluation import course_tools as T
    truth = np.array([[1.02, 0.31], [2.13, -0.52], [3.0, 1.1], [1.6, -1.3]])
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        map_yaml = _synthetic_arena_map(d, truth)
        out, poles = T.arena_from_map(map_yaml, 'syn', 9101, 0.04445,
                                      out_dir=d / 'arenas')
        assert len(poles) == len(truth), f'got {len(poles)} (sofa excluded?)'
        err = max(np.hypot(*(truth - p).T).min() for p in poles)
        assert err < 0.03, f'pole centre error {err:.3f} m'
        arena = A.load(out)
        assert len(arena['poles']) == 4 and arena['placements'] == {'P1': []}
        text = out.read_text().replace(
            'P1: []', 'P1:\n    - {color: purple, x: 1.0, y: 1.8, '
            'facing_deg: -90, mount: wall}\n    - {color: orange, x: 1.02, '
            'y: 0.34, facing_deg: 90, mount: pole, pole: p00}')
        out.write_text(text)
        arena = A.load(out)
        assert A.placement_colors(arena, 'P1') == ['purple', 'orange']
        m = T.measure_arena(arena)
        assert 15 < m['free_area_m2'] < 24 and m['poles'] == 4
        pockets = {c: pk for c, _, pk in m['squares']['P1']}
        # purple looks down x = 1.0 and p00 (1.02, 0.31) blocks it at ~1.45 m
        assert abs(pockets['purple'] - 1.45) < 0.06, pockets
        narrowest = m['gaps'][0][0]
        assert 0.3 < narrowest < 1.0, m['gaps'][:3]


def test_poles_left_unknown_by_cartographer_are_found():
    """Thin poles often end up as UNKNOWN cells (hits and grazing misses
    balance out); they must still be extracted, flagged as weaker."""
    from evaluation import course_tools as T
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        map_yaml = _synthetic_arena_map(d, [[1.0, 0.3], [2.0, -0.5]])
        pgm = d / 'syn.pgm'
        raw = pgm.read_bytes()
        header, img = raw[:raw.index(b'255\n') + 4], bytearray(
            raw[raw.index(b'255\n') + 4:])
        w = 120
        # repaint the second pole's cells as unknown (205)
        for r in range(80):
            for c in range(w):
                x, y = -1.0 + (c + 0.5) * 0.05, -2.0 + (80 - r - 0.5) * 0.05
                if math.hypot(x - 2.0, y + 0.5) < 0.0522:
                    img[r * w + c] = 205
        pgm.write_bytes(header + bytes(img))
        poles, kinds = T.extract_poles(map_yaml, with_evidence=True)
        assert len(poles) == 2, poles
        assert sorted(kinds) == ['occupied', 'unknown'], kinds


def test_unknown_map_cells_are_not_free():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img = np.array([[0, 205, 254]], dtype=np.uint8)
        (d / 'm.pgm').write_bytes(b'P5\n3 1\n255\n' + img.tobytes())
        (d / 'm.yaml').write_text('image: m.pgm\nmode: trinary\n'
                                  'resolution: 0.05\norigin: [0, 0, 0]\n'
                                  'negate: 0\noccupied_thresh: 0.65\n'
                                  'free_thresh: 0.25\n')
        occupied, free, *_ = A.read_map(d / 'm.yaml')
        assert occupied.tolist() == [[True, False, False]]
        assert free.tolist() == [[False, False, True]], 'unknown read as free'


def test_map_resampled_into_the_start_frame():
    """A pole at (1.5, 0.5) in the START frame, drawn into a map whose
    frame is offset and rotated 0.3 rad from the start pose, must come
    back at (1.5, 0.5) after resampling — catches sign errors."""
    from evaluation import course_tools as T
    x0, y0, yaw = 0.4, -0.7, 0.3
    c, s = math.cos(yaw), math.sin(yaw)
    pole_start = np.array([1.5, 0.5])
    pole_map = (x0 + c * pole_start[0] - s * pole_start[1],
                y0 + s * pole_start[0] + c * pole_start[1])
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        map_yaml = _synthetic_arena_map(d, [pole_map])
        start = A.map_to_start_frame(map_yaml, (x0, y0, yaw), d / 'st.yaml')
        poles = T.extract_poles(start)
        assert len(poles) == 1, poles
        assert np.hypot(*(poles[0] - pole_start)) < 0.04, poles


def test_arena_camera_params_become_node_overrides():
    import types
    task = R.ColourTask.__new__(R.ColourTask)
    task.light = {'profile': 'real', 'camera_params': {
        'auto_exposure': 1, 'white_balance_automatic': False, 'gain': 60}}
    task.opts = types.SimpleNamespace(camera_params='-p gain:=80')
    assert task.camera_args() == ('-p auto_exposure:=1 -p '
                                  'white_balance_automatic:=false -p gain:=60 '
                                  '-p gain:=80'), task.camera_args()


def _raycast_scan(map_yaml, pose, n=360, max_r=6.0):
    """A LaserScan-like object seen from pose on a map (occupied cells)."""
    import types
    occupied, _, res, origin, h = A.read_map(map_yaml)
    x0 = pose[0] + math.cos(pose[2]) * -0.032      # lidar behind base
    y0 = pose[1] + math.sin(pose[2]) * -0.032
    ranges = []
    for k in range(n):
        a = pose[2] - math.pi + 2 * math.pi * k / n
        r = 0.12
        while r < max_r:
            c = int((x0 + r * math.cos(a) - origin[0]) / res)
            rr = h - 1 - int((y0 + r * math.sin(a) - origin[1]) / res)
            if not (0 <= rr < occupied.shape[0] and 0 <= c < occupied.shape[1]) \
                    or occupied[rr, c]:
                break
            r += 0.01
        ranges.append(r if r < max_r else float('inf'))
    return types.SimpleNamespace(ranges=ranges, angle_min=-math.pi,
                                 angle_increment=2 * math.pi / n,
                                 range_min=0.12)


def test_start_pose_is_recovered_from_a_scan():
    from evaluation import course_tools as T
    from evaluation import real_paths as P
    truth = (0.62, -0.18, 0.35)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        map_yaml = _synthetic_arena_map(d, [[1.0, 0.3], [2.0, -0.5],
                                            [2.6, 0.9]])
        out, _ = T.arena_from_map(map_yaml, 'syn', 9101, 0.04445,
                                  out_dir=d / 'arenas')
        arena = A.load(out)
        pts = P.scan_points([_raycast_scan(arena['map'], truth)])
        near = P.localize(arena, pts, prior=(0.5, 0.0))
        far = P.localize(arena, pts, prior=(-0.6, 1.2), radius=0.2)
        for got in (near, far):             # far prior -> global fallback
            assert math.hypot(got['x'] - truth[0], got['y'] - truth[1]) < 0.04, got
            assert abs(math.remainder(got['yaw'] - truth[2], 2 * math.pi)) \
                < math.radians(2.5), got
            assert got['match'] > 0.9 and got['rival'] < got['match'] - 0.2, got


def test_paths_compose_into_the_arena_frame():
    from evaluation import real_paths as P
    rel = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, math.pi / 2]]
    out = P.compose((2.0, 1.0, math.pi / 2), rel)
    assert np.allclose(out[1, :2], [2.0, 2.0]) and np.allclose(out[2, :2], [1.0, 2.0])
    assert abs(math.remainder(out[2, 2] - math.pi, 2 * math.pi)) < 1e-9
    assert P.position_at(0.5, np.array([0.0, 1.0]),
                         np.array([[0.0, 0.0], [2.0, 4.0]])) == (1.0, 2.0)


def test_run_paths_are_saved_and_drawn():
    from evaluation import course_tools as T
    from evaluation import real_paths as P
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        map_yaml = _synthetic_arena_map(d, [[1.0, 0.3], [2.0, -0.5]])
        out, _ = T.arena_from_map(map_yaml, 'syn', 9101, 0.04445,
                                  out_dir=d / 'arenas')
        out.write_text(out.read_text().replace('P1: []', (
            'P1:\n    - {color: sky_blue, x: 1.02, y: 0.32, facing_deg: 180, '
            'mount: pole, pole: p00}\n    - {color: lime_green, x: 3.5, '
            'y: 1.2, facing_deg: 180, mount: wall}')))
        arena = A.load(out)
        mon = R.ColourMonitor(['sky_blue', 'lime_green'], cap_s=900)
        mon.arm(0.0)
        mon.on_slam(0.3, 0.0, 0.0, 0.0)        # parked, before the first
        mon.on_slam(0.6, 0.0, 0.0, 0.0)        # command: negative times
        mon.on_cmd(1.0)
        for k in range(1, 200):
            t = 1.0 + k * 0.1
            mon.on_slam(t, 0.015 * k, 0.004 * k, 0.1)
        mon.seen_t['sky_blue'], mon.found_t['sky_blue'] = 3.0, 6.0
        mon.collision_t.append(9.0)
        runs = []
        for model in ('two_stage', 'single_stage'):
            trial = d / f't00{len(runs)}_syn_P1_{model}_4hz_r0'
            trial.mkdir()
            start = {'x': 0.1, 'y': -0.05, 'yaw': 0.05, 'match': 0.98,
                     'rival': 0.6}
            assert P.save_run(trial, start, mon, {
                'model': model, 'round': 0, 'verdict': 'timeout',
                'total_s': 900, 'path_m': 3.0, 'found_count': 1})
            times, xy, ev = P.load_run(trial)
            assert xy.shape[1] == 3 and times[0] <= 0 < times[-1]
            assert ev['found']['sky_blue']['t'] == 5.0
            assert len(ev['collisions']) == 1
            png = P.render_run(arena, 'P1', trial)
            assert png.exists() and png.with_suffix('.pdf').exists()
            runs.append((model, times, xy, ev))
        fig = P.render_placement(arena, 'P1', runs, d / 'figs' / 'paths')
        assert fig.exists() and fig.with_suffix('.pdf').exists()


def test_detector_debug_view_boxes_detections_and_shows_masks():
    import cv2
    from deploy.color_seeker import detect_squares, draw_debug
    frame = np.full((480, 640, 3), 235, np.uint8)
    frame[200:260, 300:360] = (205, 180, 45)           # a sky-blue-ish square
    hsv_px = cv2.cvtColor(frame[230:231, 330:331], cv2.COLOR_BGR2HSV)[0, 0]
    bands = {'sky_blue': {'h': [int(hsv_px[0]) - 6, int(hsv_px[0]) + 6],
                          's': [100, 255], 'v': [110, 255]},
             'lime_green': {'h': [33, 53], 's': [55, 255], 'v': [90, 255]}}
    dets = detect_squares(frame, bands, fx=464.7, cx_px=320.0, size_m=0.0915)
    assert len(dets) == 1 and dets[0]['bbox'] == [300, 200, 60, 60], dets
    out = draw_debug(frame, bands, dets)
    assert out.shape == frame.shape and not np.array_equal(out, frame)
    assert (out[200, 300:360] == (0, 0, 255)).all(axis=1).any(), 'box drawn'
    thumb = out[480 - 96 - 4:480 - 4, 4:4 + 128]         # first mask thumbnail
    assert (thumb > 250).all(axis=2).sum() > 100, 'mask shows the square'
    empty = draw_debug(frame, bands, [])                # nothing detected:
    assert empty.shape == frame.shape                   # still a view


def test_lighting_profiles_and_legacy_camera_params():
    room = A.load(COURSES.parent / 'arenas' / 'room1.yaml')
    assert set(room['lighting']) == {'day', 'night'}
    night = A.lighting_config(room, 'night', 'P1')
    assert night['profile'] == 'real_night'
    assert night['camera_params']['white_balance_temperature'] == 3200
    assert _raises(lambda: A.lighting_config(room, 'dusk'), A.ArenaError)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'a.yaml'
        base = {'name': 'a', 'id': 9101, 'pole_diameter_m': 0.044,
                'poles': [], 'placements': {'P1': [
                    {'color': 'purple', 'x': 1, 'y': 0, 'facing_deg': 0,
                     'mount': 'wall'}]}}
        f.write_text(json.dumps({**base, 'camera_params': {'gain': 60}}))
        legacy = A.load(f)           # first format: one 'day' profile
        assert legacy['lighting'] == {'day': {'profile': 'real',
                                              'camera_params': {'gain': 60}}}
        f.write_text(json.dumps({**base, 'lighting': {
            'night': {'profile': 'real_night'}}}))
        # purple was never calibrated at night: refuse rather than run blind
        assert _raises(lambda: A.lighting_config(A.load(f), 'night', 'P1'),
                       A.ArenaError)


def test_arena_validation():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        def arena_with(squares):
            f = d / 'a.yaml'
            f.write_text(json.dumps({
                'name': 'a', 'id': 9101, 'pole_diameter_m': 0.044,
                'poles': [[1.0, 0.0]], 'placements': {'P1': squares}}))
            return lambda: A.load(f)
        ok = {'color': 'purple', 'x': 2, 'y': 0, 'facing_deg': 0,
              'mount': 'wall'}
        assert A.placement_colors(arena_with([ok])(), 'P1') == ['purple']
        bad = [
            [dict(ok, color='teal')],
            [dict(ok, mount='pole')],                       # no pole id
            [dict(ok, mount='pole', pole='p00')],           # not on p00
            [ok, dict(ok)],                                 # duplicate colour
            [dict(ok, color='light_green'), dict(ok, color='lime_green')],
            [{k: v for k, v in ok.items() if k != 'facing_deg'}],
        ]
        for squares in bad:
            assert _raises(arena_with(squares), A.ArenaError), squares


def main():
    tests = [(n, f) for n, f in globals().items()
             if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'  ok    {name}')
        except Exception as e:   # report every failure, not just the first
            failed += 1
            print(f'  FAIL  {name}: {type(e).__name__}: {e}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
