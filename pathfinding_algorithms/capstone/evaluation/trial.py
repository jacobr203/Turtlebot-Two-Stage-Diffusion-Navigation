#!/usr/bin/env python3
"""One closed-loop navigation trial in a BARN world.

The seed of the Phase E matrix runner: given (model, world, hz), spawn the
robot at the official start, run the model's stack, and monitor ground truth
until success (within 1 m of goal) / collision (bumper) / timeout.

Models:
  two_stage | single_stage   diffusion policy node (checkpoint from config)
  dwa                        existing dwa_node in goal mode at control_period
                             1/hz (goal published in the odom frame)

Standalone:
  .venv/bin/python evaluation/trial.py --model two_stage --world-idx 18 --hz 4
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import grid as G                                   # noqa: E402
from common.geometry import world_to_robot                     # noqa: E402
from evaluation.sim_helpers import (SimHarness, VENV_PY,       # noqa: E402
                                    barn_grid, barn_world,
                                    contact_obstacles, stop_proc)

CAPSTONE = Path(__file__).resolve().parents[1]

# env overrides support checkpoint A/B tests (e.g. v1-vs-v2 controls)
CKPTS = {
    'two_stage': Path(os.environ.get(
        'TWO_STAGE_CKPT', CAPSTONE / 'diffusion/runs/two_stage_v2/best.pt')),
    'single_stage': Path(os.environ.get(
        'SINGLE_STAGE_CKPT',
        CAPSTONE / 'diffusion/runs/single_stage_v2/best.pt')),
}

# threads: torch inference threads for the policy node. 8 is right for a
# lone worker on the idle box; concurrent shard workers MUST size it to
# their core share (full_v1 ran 3 workers x 8 threads on 16 cores and
# inflated DDPM sampling from ~40 ms to 250-650 ms — over the 4 Hz budget).
DEFAULTS = dict(timeout_sim_s=240.0, success_radius=G.SUCCESS_RADIUS,
                threads=8)


def _start_policy(h, model, world_idx, hz, start, goal, latency_log, denoiser,
                  threads, safety_gate=False):
    # safety_gate OFF by default: evaluation measures the bare learned
    # policies — the classical shield is the deployment configuration
    # (run.sh policy) and an explicit ablation axis (--safety-gate).
    args = (f'--ros-args -p use_sim_time:=true '
            f'-p checkpoint:={CKPTS[model]} '
            f'-p control_rate_hz:={hz} -p denoiser:={denoiser} '
            f'-p safety_gate:={str(bool(safety_gate)).lower()} '
            f'-p threads:={int(threads)} '
            f'-p spawn_x:={start[0]} -p spawn_y:={start[1]} '
            f'-p spawn_yaw:={start[2]} '
            f'-p goal_x:={goal[0]} -p goal_y:={goal[1]} '
            f'-p latency_log:={latency_log}')
    if model == 'two_stage':
        args += f' -p grid_file:={barn_grid(world_idx)}'
    proc = h.popen(f'{VENV_PY} {CAPSTONE}/deploy/policy_node.py {args}',
                   f'policy_w{h.worker}.log')
    if not h.wait_for_node('diffusion_policy', 30):
        raise RuntimeError('diffusion_policy node failed to start — see log')
    return proc


def _start_dwa(h, hz, start, goal):
    proc = h.popen(
        'ros2 run dwa_avoidance dwa_node --ros-args '
        '-p use_sim_time:=true '
        f'-p control_period:={1.0 / hz}',
        f'dwa_w{h.worker}.log')
    if not h.wait_for_node('dwa_node', 30):
        raise RuntimeError('dwa_node failed to start — see log')
    # dwa_node's goal frame is odom: transform the world goal through the
    # spawn anchor (odom origin sits at the spawn pose)
    gx, gy = world_to_robot(goal, start)
    from geometry_msgs.msg import PoseStamped
    msg = PoseStamped()
    msg.header.frame_id = 'odom'
    msg.pose.position.x = float(gx)
    msg.pose.position.y = float(gy)
    msg.pose.orientation.w = 1.0
    h.publish('/goal_pose', msg, n=3, interval=0.3)
    return proc


def run_trial(h, model, world_idx, hz, out_dir, trial_id=0, cfg=None,
              denoiser='ddpm', safety_gate=False):
    """World must already be running (eval variant). Returns a result row."""
    cfg = {**DEFAULTS, **(cfg or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, goal = G.START_POSE, G.GOAL_XY
    latency_log = out_dir / f'lat_{model}_{world_idx:03d}_{hz}hz_{trial_id}.csv'

    h.respawn_robot(*start)
    if model in CKPTS:
        proc = _start_policy(h, model, world_idx, hz, start, goal,
                             latency_log, denoiser, cfg['threads'],
                             safety_gate=safety_gate)
    elif model == 'dwa':
        proc = _start_dwa(h, hz, start, goal)
    else:
        raise ValueError(model)

    from gazebo_msgs.msg import ContactsState, ModelStates
    hits = set()
    bumper_sub = h.subscribe('/bumper_states', ContactsState,
                             lambda m: hits.update(contact_obstacles(m)))
    gt = h.track('/gazebo/model_states', ModelStates, best_effort=False)

    h.unpause()
    t0 = h.sim_time()
    verdict = 'timeout'
    gt_trace = []
    last_sim_t, last_advance = t0, time.time()
    try:
        while True:
            h.spin(0.3)
            sim_t = h.sim_time()
            if sim_t > last_sim_t:
                last_sim_t, last_advance = sim_t, time.time()
            elif time.time() - last_advance > 20.0:
                verdict = 'sim_stalled'
                break
            if hits:
                verdict = 'collision'
                break
            if gt.msg is not None and 'turtlebot3' in gt.msg.name:
                p = gt.msg.pose[gt.msg.name.index('turtlebot3')].position
                gt_trace.append((sim_t, p.x, p.y))
                if math.hypot(p.x - goal[0], p.y - goal[1]) \
                        <= cfg['success_radius']:
                    verdict = 'success'
                    break
            if sim_t - t0 > cfg['timeout_sim_s']:
                break
    finally:
        try:
            h.pause()
        except RuntimeError:
            pass
        h._node.destroy_subscription(bumper_sub)
        h.untrack(gt)
        stop_proc(proc)
        try:
            h.unpause()
        except RuntimeError:
            pass

    path_len = 0.0
    for a, b in zip(gt_trace, gt_trace[1:]):
        path_len += math.hypot(b[1] - a[1], b[2] - a[2])
    row = dict(model=model, world_idx=world_idx, hz=hz, trial=trial_id,
               verdict=verdict, time_to_goal_s=round(last_sim_t - t0, 2),
               gt_path_len=round(path_len, 3),
               collided_with=sorted(hits)[0] if hits else '')
    if latency_log.exists() and model in CKPTS:
        import csv
        with open(latency_log) as f:
            rows = list(csv.DictReader(f))
        if rows:
            samp = [float(r['sample_ms']) for r in rows]
            plan = [float(r['plan_ms']) for r in rows]
            row['lat_sample_ms'] = round(sum(samp) / len(samp), 1)
            row['lat_plan_ms'] = round(sum(plan) / len(plan), 1)
            row['lat_max_ms'] = round(max(s + p for s, p in zip(samp, plan)), 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['two_stage', 'single_stage', 'dwa'])
    ap.add_argument('--world-idx', type=int, required=True)
    ap.add_argument('--hz', type=float, default=4.0)
    ap.add_argument('--denoiser', default='ddpm', choices=['ddpm', 'ddim'])
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--safety-gate', action='store_true',
                    help='run the policy WITH the classical scan gate (the '
                         'deployment configuration); default measures the '
                         'bare policy')
    ap.add_argument('--out', type=Path, default=CAPSTONE / 'results/single')
    args = ap.parse_args()

    h = SimHarness(worker=0)
    try:
        h.kill_all()
        h.start_world(barn_world(args.world_idx), headless=not args.gui)
        row = run_trial(h, args.model, args.world_idx, args.hz, args.out,
                        safety_gate=args.safety_gate,
                        denoiser=args.denoiser)
        print(json.dumps(row, indent=2))
        sys.exit(0 if row['verdict'] == 'success' else 1)
    finally:
        h.close()


if __name__ == '__main__':
    main()
