#!/usr/bin/env python3
"""Offline checkpoint evaluation: waypoint accuracy + CPU sampling latency.

  .venv/bin/python -m diffusion.evaluate_offline --ckpt diffusion/runs/two_stage_v2/best.pt

Reports ADE/FDE on the full validation split (held-out episode of every
training world) and single-sample inference latency for DDPM-T and DDIM-10 —
the numbers that gate the 2/3/4 Hz control-loop budgets (500/333/250 ms).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import ExpertDataset
from .ddpm import DDPM
from .model import DiffusionPolicy


def load_policy(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    policy = DiffusionPolicy(obs_stack=cfg['obs_stack'],
                             horizon=cfg['horizon'],
                             vec_dim=cfg.get('vec_dim', 6),
                             channels=tuple(cfg['channels']),
                             scan_padding=cfg.get('scan_padding',
                                                  'circular')).to(device)
    policy.load_state_dict(ckpt['model'])
    policy.eval()
    return policy, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--latency-reps', type=int, default=30)
    args = ap.parse_args()

    policy, ckpt = load_policy(args.ckpt)
    cfg = ckpt['config']
    torch.set_num_threads(cfg.get('threads', 12))
    ddpm = DDPM(T=cfg['T'])

    ds = ExpertDataset(data_dir=cfg['data_dir'],
                       conditioning=cfg['conditioning'],
                       horizon=cfg['horizon'], obs_stack=cfg['obs_stack'],
                       goal_clip=cfg['goal_clip'], val=True,
                       val_episode=cfg['val_episode'], stats=ckpt['stats'])
    print(f"[eval] {args.ckpt} (epoch {ckpt['epoch']}, git {ckpt['git']}) "
          f"on {len(ds):,} val samples")

    # ── accuracy (full val split, DDPM-T) ────────────────────────────────────
    errs = []
    with torch.no_grad():
        for lo in range(0, len(ds), args.batch):
            batch = [ds[i] for i in range(lo, min(lo + args.batch, len(ds)))]
            scans = torch.stack([b[0] for b in batch])
            vec = torch.stack([b[1] for b in batch])
            wp_true = torch.stack([b[2] for b in batch])
            wp_pred = ddpm.sample(policy, scans, vec)
            p = ds.denormalize_wp(wp_pred.permute(0, 2, 1).numpy())
            t = ds.denormalize_wp(wp_true.permute(0, 2, 1).numpy())
            errs.append(np.linalg.norm(p - t, axis=-1))
    err = np.concatenate(errs)                       # [N, H]
    print(f'[eval] ADE {err.mean()*100:.2f} cm   FDE {err[:, -1].mean()*100:.2f} cm'
          f'   per-step cm: {(err.mean(axis=0)*100).round(1).tolist()}')

    # ── latency (batch 1, the deployment case) ───────────────────────────────
    scans1, vec1 = ds[0][0][None], ds[0][1][None]
    for name, fn in ((f'DDPM-{cfg["T"]}',
                      lambda: ddpm.sample(policy, scans1, vec1)),
                     ('DDIM-10',
                      lambda: ddpm.sample_ddim(policy, scans1, vec1, steps=10))):
        fn()   # warm-up
        times = []
        for _ in range(args.latency_reps):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        t = np.array(times)
        print(f'[eval] {name}: mean {t.mean():.0f} ms  p95 {np.percentile(t, 95):.0f} ms '
              f'(budgets: 4 Hz=250, 3 Hz=333, 2 Hz=500)')


if __name__ == '__main__':
    main()
