#!/usr/bin/env python3
"""Train a diffusion policy (two-stage or single-stage conditioning).

  .venv/bin/python -m diffusion.train --config diffusion/configs/two_stage.yaml

Checkpoints (runs/<run_name>/, gitignored): last.pt every epoch, best.pt on
best validation ADE. Each checkpoint bundles model weights (EMA), the
normalization stats, the full config, and the git SHA — everything the
deployment node needs to reproduce inference exactly.

Validation = held-out episode `val_episode` of every training world:
DDPM-sampled waypoints vs expert, ADE/FDE in meters.
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import ExpertDataset
from .ddpm import DDPM, EMA
from .model import DiffusionPolicy

HERE = Path(__file__).parent


def git_sha():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True,
                              cwd=HERE).stdout.strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


@torch.no_grad()
def eval_ade_fde(policy, ddpm, ds_val, cfg, n_max=768):
    """Sampled-waypoint error vs expert on a fixed validation subset [m]."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds_val), min(n_max, len(ds_val)), replace=False)
    ades, fdes = [], []
    for lo in range(0, len(idx), cfg['batch_size']):
        batch = [ds_val[i] for i in idx[lo:lo + cfg['batch_size']]]
        scans = torch.stack([b[0] for b in batch])
        vec = torch.stack([b[1] for b in batch])
        wp_true = torch.stack([b[2] for b in batch])
        wp_pred = ddpm.sample(policy, scans, vec)
        p = ds_val.denormalize_wp(wp_pred.permute(0, 2, 1).numpy())
        t = ds_val.denormalize_wp(wp_true.permute(0, 2, 1).numpy())
        err = np.linalg.norm(p - t, axis=-1)          # [B, H]
        ades.append(err.mean(axis=1))
        fdes.append(err[:, -1])
    return float(np.concatenate(ades).mean()), float(np.concatenate(fdes).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--max-steps', type=int, default=0,
                    help='stop after N optimizer steps (profiling/smoke)')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.set_num_threads(cfg.get('threads', 12))
    device = cfg.get('device', 'cpu')

    common = dict(data_dir=cfg['data_dir'], conditioning=cfg['conditioning'],
                  horizon=cfg['horizon'], obs_stack=cfg['obs_stack'],
                  goal_clip=cfg['goal_clip'], val_episode=cfg['val_episode'],
                  scan_arc_deg=cfg.get('scan_arc_deg', 360))
    ds_train = ExpertDataset(**common, val=False,
                             corrupt_frac=cfg.get('corrupt_frac', 0.0),
                             segment_balance=cfg.get('segment_balance', False),
                             seg_len_m=cfg.get('seg_len_m', 1.5),
                             balance_seed=cfg['seed'])
    # val: clean scans, natural class mix — corruption and balancing are
    # training-time treatments, never evaluation-time
    ds_val = ExpertDataset(**common, val=True, stats=ds_train.stats)
    if getattr(ds_train, 'class_counts', None):
        print(f'[train] segment balance over {ds_train.class_counts} '
              f'-> {len(ds_train)} samples')
    dl = DataLoader(ds_train, batch_size=cfg['batch_size'], shuffle=True,
                    num_workers=cfg.get('num_workers', 4), drop_last=True,
                    persistent_workers=cfg.get('num_workers', 4) > 0)

    cfg['vec_dim'] = ds_train.vec_dim   # into the checkpoint for deployment
    cfg['scan_padding'] = ('circular' if cfg.get('scan_arc_deg', 360) >= 360
                           else 'replicate')   # an arc must not wrap
    policy = DiffusionPolicy(obs_stack=cfg['obs_stack'], horizon=cfg['horizon'],
                             vec_dim=ds_train.vec_dim,
                             channels=tuple(cfg['channels']),
                             scan_padding=cfg['scan_padding']).to(device)
    ddpm = DDPM(T=cfg['T'], device=device)
    ema = EMA(policy, decay=cfg['ema_decay'])
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['epochs'] * (len(ds_train) // cfg['batch_size']))

    run_dir = HERE / 'runs' / cfg['run_name']
    run_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(run_dir)
    sha = git_sha()
    print(f"[train] {cfg['run_name']}: {policy.n_params()/1e6:.2f} M params, "
          f"{len(ds_train):,} train / {len(ds_val):,} val samples, "
          f"git {sha}", flush=True)

    def save(name, epoch, ade, fde):
        torch.save({'model': ema.shadow.state_dict(), 'stats': ds_train.stats,
                    'config': cfg, 'git': sha, 'epoch': epoch,
                    'val_ade': ade, 'val_fde': fde}, run_dir / name)

    best_ade, step = float('inf'), 0
    t0 = time.time()
    for epoch in range(cfg['epochs']):
        policy.train()
        losses = []
        for scans, vec, wp in dl:
            loss = ddpm.loss(policy, wp.to(device), scans.to(device),
                             vec.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ema.update(policy)
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                rate = step / (time.time() - t0)
                k = max(10, args.max_steps // 5)
                print(f'[train] profile: {step} steps in '
                      f'{time.time()-t0:.0f}s = {rate:.2f} it/s -> '
                      f'{len(ds_train)/cfg["batch_size"]/rate/60:.1f} '
                      f'min/epoch | loss first {k}: '
                      f'{np.mean(losses[:k]):.3f} -> last {k}: '
                      f'{np.mean(losses[-k:]):.3f}', flush=True)
                return
        tb.add_scalar('loss/train', float(np.mean(losses)), epoch)

        if (epoch + 1) % cfg['eval_every'] == 0 or epoch == cfg['epochs'] - 1:
            ema.shadow.eval()
            ade, fde = eval_ade_fde(ema.shadow, ddpm, ds_val, cfg)
            tb.add_scalar('val/ade_m', ade, epoch)
            tb.add_scalar('val/fde_m', fde, epoch)
            save('last.pt', epoch, ade, fde)
            if ade < best_ade:
                best_ade = ade
                save('best.pt', epoch, ade, fde)
            print(f'[train] epoch {epoch + 1}/{cfg["epochs"]} '
                  f'loss {np.mean(losses):.4f} val ADE {ade*100:.2f} cm '
                  f'FDE {fde*100:.2f} cm ({(time.time()-t0)/60:.0f} min)',
                  flush=True)
        else:
            print(f'[train] epoch {epoch + 1}/{cfg["epochs"]} '
                  f'loss {np.mean(losses):.4f}', flush=True)

    print(f'[train] done: best val ADE {best_ade*100:.2f} cm '
          f'({(time.time()-t0)/3600:.1f} h)', flush=True)


if __name__ == '__main__':
    main()
