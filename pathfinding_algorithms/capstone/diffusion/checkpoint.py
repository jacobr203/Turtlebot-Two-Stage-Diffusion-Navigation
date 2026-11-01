"""Rebuild a trained policy from its checkpoint alone."""

import torch

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
