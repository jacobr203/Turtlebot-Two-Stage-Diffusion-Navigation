#!/usr/bin/env python3
"""Collect the full expert dataset: train worlds x episodes, resumable.

One gzserver per world (bounds WSL2 Gazebo decay), delete+respawn per
episode. Failed episodes (collision/timeout/no_path) are retried once with
fresh jitter; persistent failures are recorded in the manifest and excluded
from training by dataset.py.

Resumable: (world, episode) pairs whose npz already exists and whose manifest
row says success are skipped, so the sweep can be re-run after interruption.

Sharding for parallel workers (separate terminals):
  .venv/bin/python expert/collect_dataset.py --shard 0/3 --worker 0
  .venv/bin/python expert/collect_dataset.py --shard 1/3 --worker 1
  .venv/bin/python expert/collect_dataset.py --shard 2/3 --worker 2
Each worker uses its own ROS_DOMAIN_ID + GAZEBO_MASTER_URI and its own
manifest file (manifest_w<k>.csv).

Usage: .venv/bin/python expert/collect_dataset.py [--config expert/configs/collect.yaml]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.sim_helpers import SimHarness, barn_world   # noqa: E402
from expert.collect_episode import run_episode              # noqa: E402

HERE = Path(__file__).parent

FIELDS = ['world_idx', 'episode', 'file', 'verdict', 'samples',
          'duration_sim_s', 'gt_path_len', 'planned_len',
          'cross_track_mean', 'cross_track_max', 'collided_with',
          'start_x', 'start_y', 'start_yaw', 'goal_x', 'goal_y', 'attempt']


def load_done(manifest):
    done = {}
    if manifest.exists():
        with open(manifest) as f:
            for r in csv.DictReader(f):
                done[(int(r['world_idx']), int(r['episode']))] = r['verdict']
    return done


def _restart_world(h, world_idx, tries=2):
    """Fresh gzserver for this world; one retry if the bringup itself fails."""
    for k in range(tries):
        h.kill_all()
        try:
            h.start_world(barn_world(world_idx, fast=True), headless=True)
            return
        except RuntimeError as e:
            if k == tries - 1:
                raise
            print(f'[collect] world {world_idx}: bringup failed ({e}) — '
                  f'retrying', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=HERE / 'configs/collect.yaml')
    ap.add_argument('--shard', default='0/1', help='k/n — this worker takes '
                    'train worlds with index %% n == k')
    ap.add_argument('--worker', type=int, default=0,
                    help='ROS domain / Gazebo port offset for this process')
    ap.add_argument('--limit-worlds', type=int, default=0,
                    help='stop after N worlds (smoke runs)')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    meta = yaml.safe_load(open(HERE.parent / 'barn' / 'metadata.yaml'))
    k, n = map(int, args.shard.split('/'))

    # standard BARN train worlds (metadata.yaml, benchmark 260/40 split) +
    # stress-world train splits (dense_worlds.yaml: dense/winding/scatter;
    # stress worlds without a split key are excluded)
    split = {int(i): w['split'] for i, w in meta['worlds'].items()}
    stress_file = HERE.parent / 'barn' / 'dense_worlds.yaml'
    n_stress = 0
    if stress_file.exists():
        stress = yaml.safe_load(open(stress_file)).get('worlds', {})
        split.update({int(i): w.get('split', 'holdout')
                      for i, w in stress.items()})
        n_stress = sum(1 for i, w in stress.items()
                       if w.get('split') == 'train')
    # fork worlds (diffusion v3, barn/make_fork_worlds.py) — opt-in per
    # config so v2 recollections stay reproducible
    fork_file = HERE.parent / 'barn' / 'fork_worlds.yaml'
    if cfg.get('include_fork_worlds') and fork_file.exists():
        fork = yaml.safe_load(open(fork_file))
        split.update({int(i): 'train' for i in fork.get('train', [])})
        split.update({int(i): 'holdout' for i in fork.get('holdout', [])})
    # v3 world redesign (2026-07-21): train EXCLUSIVELY on the generated
    # multipath pool (fork 48 + trainpool 500+); original BARN/stress
    # worlds keep their files and remain the closed-loop eval benchmark
    if cfg.get('generated_pool_only'):
        split = {i: s for i, s in split.items() if i >= 400}
        pool_file = HERE.parent / 'barn' / 'trainpool_v3.yaml'
        if pool_file.exists():
            pool = yaml.safe_load(open(pool_file))
            split.update({int(i): 'train' for i in pool.get('train', [])})
    # pocket worlds (2026-08-03 corridor-pocket top-up,
    # barn/make_pocket_worlds.py) — opt-in per config; pocket_only limits
    # the sweep to just them (the top-up collection run)
    pocket_file = HERE.parent / 'barn' / 'pocket_worlds.yaml'
    pocket_legs = {}
    if cfg.get('include_pocket_worlds') and pocket_file.exists():
        pw = yaml.safe_load(open(pocket_file))['worlds']
        split.update({int(i): w.get('split', 'train') for i, w in pw.items()})
        pocket_legs = {int(i): w['legs'] for i, w in pw.items()}
        if cfg.get('pocket_only'):
            split = {i: s for i, s in split.items() if i in pocket_legs}
    train_worlds = [i for i, s in sorted(split.items())
                    if s == 'train' and i % n == k]
    print(f'[collect] train pool: {len([i for i, s in split.items() if s == "train"])} '
          f'worlds ({n_stress} stress)')
    if args.limit_worlds:
        train_worlds = train_worlds[:args.limit_worlds]
    episodes = cfg['episodes_per_world']

    out_dir = HERE / 'data' / cfg['dataset_id']
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f'manifest_w{args.worker}.csv'
    done = load_done(manifest)
    write_header = not manifest.exists()

    print(f'[collect] shard {k}/{n}: {len(train_worlds)} worlds x {episodes} '
          f'episodes -> {out_dir} (done already: {len(done)})')

    h = SimHarness(worker=args.worker, log_dir=out_dir / 'logs')
    t_start = time.time()
    n_run = n_ok = 0
    try:
        with open(manifest, 'a', newline='') as mf:
            writer = csv.DictWriter(mf, fieldnames=FIELDS, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            for wi, world_idx in enumerate(train_worlds):
                todo = [e for e in range(episodes)
                        if done.get((world_idx, e)) != 'success']
                if not todo:
                    continue
                _restart_world(h, world_idx)
                for ep in todo:
                    row = None
                    for attempt in (1, 2, 3):
                        rng = np.random.default_rng(
                            cfg['seed'] * 100003 + world_idx * 101
                            + ep * 13 + attempt)
                        out = out_dir / f'ep_{world_idx:03d}_{ep}.npz'
                        # Watchdog: gzserver segfaults every ~15 delete+respawn
                        # cycles on WSL2 (exit -11) — restart the world and
                        # burn an attempt instead of dying.
                        try:
                            # multipath (fork) worlds start INSIDE the
                            # clutter for every episode — except every 6th
                            # fork world, which keeps the standard scheme
                            # ("a few environments also start on the
                            # outside",                             cfg_w = cfg
                            if world_idx >= 400:
                                cfg_w = {**cfg, 'interior_goal': True}
                                if cfg.get('fork_all_multi_start') \
                                        and world_idx % 6 != 0:
                                    cfg_w['multi_start_from_episode'] = 0
                            if world_idx in pocket_legs:
                                cfg_w = {**cfg, 'pocket_legs':
                                         pocket_legs[world_idx]}
                            row = run_episode(h, world_idx, ep, out, rng,
                                              cfg_w)
                        except Exception as e:  # noqa: BLE001
                            print(f'[collect] world {world_idx} ep {ep}: '
                                  f'harness error ({e}) — restarting world',
                                  flush=True)
                            row = dict(world_idx=world_idx, episode=ep,
                                       file=out.name, verdict='harness_error',
                                       samples=0)
                            _restart_world(h, world_idx)
                        row['attempt'] = attempt
                        n_run += 1
                        if row['verdict'] == 'success':
                            n_ok += 1
                            break
                        if row['verdict'] in ('sim_stalled', 'node_start_failed'):
                            print(f'[collect] world {world_idx} ep {ep}: '
                                  f'{row["verdict"]} — restarting world',
                                  flush=True)
                            _restart_world(h, world_idx)
                        if attempt < 3:
                            out.unlink(missing_ok=True)
                    writer.writerow(row)
                    mf.flush()
                rate = n_ok / max(n_run, 1)
                elapsed = (time.time() - t_start) / 60
                print(f'[collect] world {world_idx:3d} done '
                      f'({wi + 1}/{len(train_worlds)}), success {n_ok}/{n_run} '
                      f'({rate:.0%}), {elapsed:.0f} min elapsed', flush=True)
    finally:
        h.close()

    print(f'[collect] finished: {n_ok}/{n_run} successful episodes '
          f'({(time.time() - t_start) / 3600:.1f} h)')


if __name__ == '__main__':
    main()
