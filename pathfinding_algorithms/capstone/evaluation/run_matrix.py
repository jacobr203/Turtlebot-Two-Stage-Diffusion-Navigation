#!/usr/bin/env python3
"""The evaluation matrix: (model x holdout world x control rate x trial).

Resumable (per-key skip against existing CSVs), shardable across workers
(--shard k/n + --worker, same isolation as expert collection), and hardened
with the collector's watchdog: gzserver segfaults / sim stalls restart the
world and retry the trial; genuine verdicts (success/collision/timeout) are
data and are never retried.

A fresh gzserver per (world, model) keeps respawn cycles per server at
hz_list x trials (<= 12) — under the ~15-cycle WSL2 segfault horizon.

  .venv/bin/python evaluation/run_matrix.py --config evaluation/configs/eval_poster.yaml
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.sim_helpers import SimHarness, barn_world   # noqa: E402
from evaluation.trial import run_trial                      # noqa: E402

HERE = Path(__file__).parent
CAPSTONE = HERE.parent

FIELDS = ['model', 'world_idx', 'hz', 'trial', 'verdict', 'time_to_goal_s',
          'gt_path_len', 'collided_with', 'lat_sample_ms', 'lat_plan_ms',
          'lat_max_ms', 'attempt']

HARNESS_VERDICTS = ('sim_stalled', 'node_start_failed')


def load_done(out_dir):
    done = set()
    for f in out_dir.glob('trials_w*.csv'):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r['verdict'] not in HARNESS_VERDICTS:
                    done.add((r['model'], int(r['world_idx']),
                              float(r['hz']), int(r['trial'])))
    return done


def resolve_worlds(cfg):
    if cfg['worlds'] == 'holdout':
        meta = yaml.safe_load(open(CAPSTONE / 'barn' / 'metadata.yaml'))
        return meta['holdout']
    return list(cfg['worlds'])


RESTART_TRIES = 3


def restart_world(h, world_idx, tries=RESTART_TRIES):
    for k in range(tries):
        h.kill_all()
        try:
            h.start_world(barn_world(world_idx), headless=True)
            return
        except RuntimeError as e:
            if k == tries - 1:
                raise
            print(f'[matrix] world {world_idx}: bringup failed ({e}) — retry',
                  flush=True)


def safe_restart(h, world_idx):
    """restart_world that reports failure instead of killing the worker:
    a persistently unbringable world abandons its block, and the per-key
    resume skip picks the gap up on a later rerun."""
    try:
        restart_world(h, world_idx)
        return True
    except RuntimeError as e:
        print(f'[matrix] world {world_idx}: bringup failed {RESTART_TRIES}x '
              f'({e}) — abandoning block for resume', flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path,
                    default=HERE / 'configs/eval_poster.yaml')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--worker', type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    k, n = map(int, args.shard.split('/'))

    worlds = [w for i, w in enumerate(sorted(resolve_worlds(cfg)))
              if i % n == k]
    out_dir = CAPSTONE / 'results' / cfg['run_id']
    out_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(out_dir)
    total = len(worlds) * len(cfg['models']) * len(cfg['hz_list']) * cfg['trials']
    print(f"[matrix] {cfg['run_id']} shard {k}/{n}: {len(worlds)} worlds x "
          f"{cfg['models']} x {cfg['hz_list']} Hz x {cfg['trials']} trials "
          f"= {total} ({len(done)} already done)", flush=True)

    trial_cfg = dict(timeout_sim_s=cfg.get('timeout_sim_s', 240.0))
    if 'threads' in cfg:
        trial_cfg['threads'] = cfg['threads']
    h = SimHarness(worker=args.worker, log_dir=out_dir / 'logs')

    def revive(h, world_idx):
        """restart_world with a harness-recycle escalation. After ~100
        gzserver generations a long-lived harness's DDS graph can go
        stale: consecutive fresh gzservers come up healthy (per their
        logs) yet stay invisible to the harness node, so every bringup
        'fails' forever (worker 2 hit this at world 104, ~3 h in, while
        the other workers ran on). A fresh Context forces rediscovery."""
        if safe_restart(h, world_idx):
            return True, h
        print(f'[matrix] world {world_idx}: bringups failing on a healthy '
              f'sim — recycling harness (stale DDS graph)', flush=True)
        try:
            h.close()
        except Exception:  # noqa: BLE001
            pass
        h = SimHarness(worker=args.worker, log_dir=out_dir / 'logs')
        return safe_restart(h, world_idx), h
    manifest = out_dir / f'trials_w{args.worker}.csv'
    write_header = not manifest.exists()
    t_start, n_run = time.time(), 0
    try:
        with open(manifest, 'a', newline='') as mf:
            writer = csv.DictWriter(mf, fieldnames=FIELDS, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            for world_idx in worlds:
                for model in cfg['models']:
                    todo = [(hz, tr) for hz in cfg['hz_list']
                            for tr in range(cfg['trials'])
                            if (model, world_idx, float(hz), tr) not in done]
                    if not todo:
                        continue
                    ok, h = revive(h, world_idx)
                    if not ok:
                        continue
                    abandoned = False
                    for hz, tr in todo:
                        row = None
                        for attempt in (1, 2, 3):
                            try:
                                row = run_trial(h, model, world_idx, hz,
                                                out_dir / 'latency', tr,
                                                cfg=trial_cfg,
                                                denoiser=cfg.get('denoiser',
                                                                 'ddpm'))
                            except Exception as e:  # noqa: BLE001
                                print(f'[matrix] {model} w{world_idx} '
                                      f'{hz}Hz t{tr}: harness error ({e})',
                                      flush=True)
                                row = dict(model=model, world_idx=world_idx,
                                           hz=hz, trial=tr,
                                           verdict='node_start_failed')
                            row['attempt'] = attempt
                            if row['verdict'] not in HARNESS_VERDICTS:
                                break              # genuine verdict = data
                            if attempt < 3:
                                ok, h = revive(h, world_idx)
                                if not ok:
                                    abandoned = True
                                    break
                        writer.writerow(row)
                        mf.flush()
                        n_run += 1
                        if abandoned:
                            break
                        if row['verdict'] in HARNESS_VERDICTS:
                            # 3 failed attempts — revive the sim before the
                            # next trial key rather than failing it too
                            ok, h = revive(h, world_idx)
                            if not ok:
                                break
                    el = (time.time() - t_start) / 60
                    print(f'[matrix] {model} w{world_idx} done '
                          f'({n_run} trials, {el:.0f} min)', flush=True)
    finally:
        h.close()
    print(f'[matrix] finished: {n_run} trials in '
          f'{(time.time() - t_start) / 3600:.1f} h', flush=True)


if __name__ == '__main__':
    main()
