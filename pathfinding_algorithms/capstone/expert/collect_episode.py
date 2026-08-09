#!/usr/bin/env python3
"""Run ONE expert episode in an already-running BARN world.

Flow (harness must have the world up):
  respawn robot at jittered start (paused) -> plan A* on the world's grid ->
  launch tracker + recorder (venv python, sim time) -> unpause -> monitor
  ground truth for success / bumper for collision / sim clock for timeout ->
  pause, stop recorder (flushes npz), kill nodes -> manifest row.

Also usable standalone for a single-episode check:
  .venv/bin/python expert/collect_episode.py --world-idx 0 --episode 0
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import grid as G                                  # noqa: E402
from evaluation.sim_helpers import (SimHarness, VENV_PY,      # noqa: E402
                                    barn_grid, barn_world,
                                    contact_obstacles, stop_proc)

HERE = Path(__file__).parent

DEFAULTS = dict(
    inflate_radius=G.DEFAULT_INFLATE_RADIUS,
    start_jitter_xy=0.15, start_jitter_yaw=0.15, goal_jitter_xy=0.30,
    subgoal_dist=1.2, rate_hz=4.0, tracker_rate_hz=20.0,
    timeout_sim_s=240.0, settle_s=1.0,
)


def _sample_multi_route(cylinders, rng, cfg):
    """v3 multi-start (diffusion_v3_plan.md §1): a start/goal pair sampled
    anywhere in free space whose route length falls in route_len_range —
    short, varied segments instead of only full-course runs. Start yaw
    faces along the route. None after max tries (caller falls back to the
    canonical start/goal)."""
    lo, hi = cfg.get('route_len_range', (2.0, 6.0))
    planner = G.Planner(cylinders, inflate_radius=G.WIDE_INFLATE_RADIUS)
    # sample from actually-free planning cells — uniform-rectangle sampling
    # mostly hits the fill in dense/fork worlds (~25% acceptance observed)
    fi, fj = np.nonzero(~planner.blocked)
    if len(fi) < 10:
        return None
    free_xy = np.stack([planner.x0 + fi * planner.res,
                        planner.y0 + fj * planner.res], axis=1)
    for _ in range(25):
        s = tuple(free_xy[rng.integers(len(free_xy))])
        g = tuple(free_xy[rng.integers(len(free_xy))])
        if planner.clearance_at(s) < 0.35 or planner.clearance_at(g) < 0.35:
            continue
        d = np.hypot(g[0] - s[0], g[1] - s[1])
        if not (lo * 0.6 <= d <= hi):
            continue
        path = planner.plan_centered(s, g)
        if path is None:
            continue
        length = G.path_length(path)
        if not (lo <= length <= hi):
            continue
        k = min(len(path) - 1, 6)
        yaw = float(np.arctan2(path[k][1] - s[1], path[k][0] - s[0])) \
            + rng.uniform(-cfg['start_jitter_yaw'], cfg['start_jitter_yaw'])
        return (float(s[0]), float(s[1]), yaw), (float(g[0]), float(g[1]))
    return None


def run_episode(h, world_idx, episode, out_file, rng, cfg=None):
    """Returns a manifest-row dict. The world must already be running."""
    cfg = {**DEFAULTS, **(cfg or {})}
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cylinders = G.arena_grid_to_cylinders(np.load(barn_grid(world_idx)))
    start = None
    legs = cfg.get('pocket_legs')
    if legs:
        # pocket worlds (2026-08-03 corridor-pocket top-up): episodes cycle
        # the yaml's verified enter/exit legs; only the start jitters — the
        # goal is the 0.55 m stand-off pose and must stay off the inflation
        leg = legs[episode % len(legs)]
        jx = cfg['start_jitter_xy'] * 0.7
        start = (leg['sx'] + rng.uniform(-jx, jx),
                 leg['sy'] + rng.uniform(-jx, jx),
                 leg['syaw'] + rng.uniform(-cfg['start_jitter_yaw'],
                                           cfg['start_jitter_yaw']))
        goal = (leg['gx'], leg['gy'])
    if start is None and episode >= cfg.get('multi_start_from_episode', 10**9):
        pick = _sample_multi_route(cylinders, rng, cfg)
        if pick is not None:
            start, goal = pick
    if start is None:
        jx = cfg['start_jitter_xy']
        start = (G.START_POSE[0] + rng.uniform(-jx, jx),
                 G.START_POSE[1] + rng.uniform(-jx, jx),
                 G.START_POSE[2] + rng.uniform(-cfg['start_jitter_yaw'],
                                               cfg['start_jitter_yaw']))
        goal = None
        if cfg.get('interior_goal'):
            # multipath worlds: even outside-start episodes end INSIDE the
            # clutter — canonical goal only as the
            # sampling fallback
            planner = G.Planner(cylinders,
                                inflate_radius=G.WIDE_INFLATE_RADIUS)
            fi, fj = np.nonzero(~planner.blocked)
            free_xy = np.stack([planner.x0 + fi * planner.res,
                                planner.y0 + fj * planner.res], axis=1)
            for _ in range(20):
                g = tuple(free_xy[rng.integers(len(free_xy))])
                if planner.clearance_at(g) < 0.35:
                    continue
                if np.hypot(g[0] - start[0], g[1] - start[1]) < 2.5:
                    continue
                if planner.plan_centered(start[:2], g) is not None:
                    goal = (float(g[0]), float(g[1]))
                    break
        if goal is None:
            jg = cfg['goal_jitter_xy']
            goal = (G.GOAL_XY[0] + rng.uniform(-jg, jg),
                    G.GOAL_XY[1] + rng.uniform(-jg, jg))

    row = dict(world_idx=world_idx, episode=episode, file=out_file.name,
               start_x=round(start[0], 4), start_y=round(start[1], 4),
               start_yaw=round(start[2], 4),
               goal_x=round(goal[0], 4), goal_y=round(goal[1], 4))

    # ── plan (v2 route convention: wide-corridor preference + centering) ──────
    planner, path, _ = G.plan_route(
        cylinders, start[:2], goal,
        inflates=(G.WIDE_INFLATE_RADIUS, cfg['inflate_radius']))
    if path is None:
        row.update(verdict='no_path', samples=0)
        return row
    path_file = out_file.with_suffix('.path.npy')
    np.save(path_file, path)

    # ── place robot + start nodes ─────────────────────────────────────────────
    h.respawn_robot(*start)
    common_params = (f'-p use_sim_time:=true -p path_file:={path_file} '
                     f'-p spawn_x:={start[0]} -p spawn_y:={start[1]} '
                     f'-p spawn_yaw:={start[2]}')
    perturb = ''
    _pl = cfg.get('perturb_episodes')          # explicit list (v3) beats the
    noisy = (episode in _pl) if _pl else \
        (episode >= cfg.get('perturb_from_episode', 10**9))  # v2 threshold
    if noisy:
        # noise-injected episode (decisions #27a): the tracker veers off the
        # path in short bursts and pure pursuit demonstrates the recovery
        perturb = (f' -p perturb_every:={cfg.get("perturb_every", 2.5)}'
                   f' -p perturb_dur:={cfg.get("perturb_dur", 0.45)}'
                   f' -p perturb_w:={cfg.get("perturb_w", 1.2)}'
                   f' -p perturb_min_clear:={cfg.get("perturb_min_clear", 0.35)}'
                   f' -p perturb_seed:={int(rng.integers(2 ** 31))}')
    tracker = h.popen(
        f'{VENV_PY} {HERE}/tracker_node.py --ros-args {common_params} '
        f'-p rate:={cfg["tracker_rate_hz"]}{perturb}',
        f'tracker_w{h.worker}.log')
    recorder = h.popen(
        f'{VENV_PY} {HERE}/recorder_node.py --ros-args {common_params} '
        f'-p rate:={cfg["rate_hz"]} -p out_file:={out_file} '
        f'-p goal_x:={goal[0]} -p goal_y:={goal[1]} '
        f'-p subgoal_dist:={cfg["subgoal_dist"]} -p world_idx:={world_idx}',
        f'recorder_w{h.worker}.log')
    if not (h.wait_for_node('expert_tracker', 20)
            and h.wait_for_node('episode_recorder', 20)):
        _stop_nodes(h, tracker, recorder)
        row.update(verdict='node_start_failed', samples=0)
        return row

    # ── run + monitor ─────────────────────────────────────────────────────────
    from gazebo_msgs.msg import ContactsState, ModelStates
    from std_msgs.msg import String
    hits = set()
    bumper_sub = h.subscribe('/bumper_states', ContactsState,
                             lambda m: hits.update(contact_obstacles(m)))
    gt = h.track('/gazebo/model_states', ModelStates, best_effort=False)
    status = h.track('/expert/status', String, best_effort=False)

    try:
        h.unpause()
    except RuntimeError:
        _stop_nodes(h, tracker, recorder)
        raise   # gzserver is gone — the caller's watchdog restarts the world
    t0 = h.sim_time()
    verdict, collided_with = 'timeout', ''
    tracker_done = False
    last_sim_t, last_advance = t0, time.time()
    try:
        while True:
            h.spin(0.3)
            sim_t = h.sim_time()
            # a dead gzserver freezes /clock, so the sim timeout alone would
            # never fire — treat frozen sim time as a harness failure
            if sim_t > last_sim_t:
                last_sim_t, last_advance = sim_t, time.time()
            elif time.time() - last_advance > 20.0:
                verdict = 'sim_stalled'
                break
            if hits:
                verdict, collided_with = 'collision', sorted(hits)[0]
                break
            if status.msg is not None:
                tracker_done = json.loads(status.msg.data).get('done', False)
            if gt.msg is not None and 'turtlebot3' in gt.msg.name:
                p = gt.msg.pose[gt.msg.name.index('turtlebot3')].position
                if math.hypot(p.x - goal[0], p.y - goal[1]) <= G.SUCCESS_RADIUS \
                        and tracker_done:
                    verdict = 'success'
                    break
            if sim_t - t0 > cfg['timeout_sim_s']:
                verdict = 'timeout'
                break
    finally:
        try:
            h.pause()
        except RuntimeError:
            pass   # gzserver dead; verdict already reflects it
        h._node.destroy_subscription(bumper_sub)
        for holder in (gt, status):
            h.untrack(holder)

    row['duration_sim_s'] = round(h.sim_time() - t0, 2)
    if status.msg is not None:
        st = json.loads(status.msg.data)
        row['cross_track_mean'] = st.get('cross_track_mean')
        row['cross_track_max'] = st.get('cross_track_max')

    # ── stop recorder (flushes npz) + tracker ─────────────────────────────────
    from std_msgs.msg import Empty as EmptyMsg
    h.publish('/expert/stop', EmptyMsg(), n=3, interval=0.2)
    _wait_exit(recorder, 10) or _stop_nodes(h, recorder)
    _stop_nodes(h, tracker)
    try:
        h.unpause()   # leave the world running for the next episode's respawn
    except RuntimeError:
        pass   # gzserver dead; the caller's watchdog restarts it anyway

    # ── stats from the written file ───────────────────────────────────────────
    row.update(verdict=verdict, collided_with=collided_with, samples=0)
    if out_file.exists():
        with np.load(out_file) as ep:
            row['samples'] = int(len(ep['t']))
            if row['samples'] > 1:
                row['gt_path_len'] = round(G.path_length(ep['gt_pose'][:, :2]), 3)
                row['planned_len'] = round(G.path_length(path), 3)
    path_file.unlink(missing_ok=True)
    return row


def _wait_exit(proc, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.2)
    return False


def _stop_nodes(h, *procs):
    for p in procs:
        stop_proc(p)   # group-kill: wrapper-only signals orphan the node
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world-idx', type=int, required=True)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--out', type=Path,
                    default=None, help='default: expert/data/single/...')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gui', action='store_true',
                    help='run with gzclient for visual debugging')
    ap.add_argument('--eval-world', action='store_true',
                    help='use the RTF-capped eval world instead of worlds_fast')
    args = ap.parse_args()

    out = args.out or (HERE / 'data' / 'single'
                       / f'ep_{args.world_idx:03d}_{args.episode}.npz')
    rng = np.random.default_rng(
        args.seed * 100003 + args.world_idx * 101 + args.episode)

    h = SimHarness(worker=0)
    try:
        h.kill_all()
        h.start_world(barn_world(args.world_idx, fast=not args.eval_world),
                      headless=not args.gui)
        row = run_episode(h, args.world_idx, args.episode, out, rng)
        print(json.dumps(row, indent=2))
        sys.exit(0 if row['verdict'] == 'success' else 1)
    finally:
        h.close()


if __name__ == '__main__':
    main()
