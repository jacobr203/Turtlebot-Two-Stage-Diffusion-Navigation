"""Observation encoder + conditional temporal U-Net denoiser.

Chi-et-al-style diffusion policy sized for CPU training (~2 M params):
the encoder embeds (lidar stack, goal/vel vector) into a context vector;
the U-Net denoises the [B, 2, H] waypoint sequence conditioned on that
context plus a sinusoidal diffusion-timestep embedding via FiLM.
"""

import math

import torch
import torch.nn as nn


""" Takes the stack of lidar scans and the goal/velocity vector and encodes them into a context vector."""
class ObsEncoder(nn.Module):
    def __init__(self, obs_stack=2, vec_dim=6, out_dim=112,
                 scan_padding='circular'):
        # circular padding is only correct for full-360 scans; the v3
        # forward-arc crop (diffusion_v3_plan.md §3) must NOT wrap the
        # arc's edges onto each other -> 'replicate' there. Width itself
        # is free (AdaptiveAvgPool), so 160 bins need no other change.
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(obs_stack, 32, 5, stride=2, padding=2,
                      padding_mode=scan_padding),
            nn.Mish(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2,
                      padding_mode=scan_padding),
            nn.Mish(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1,
                      padding_mode=scan_padding),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.scan_head = nn.Linear(64, 64)
        self.vec_mlp = nn.Sequential(
            nn.Linear(obs_stack * vec_dim, 64), nn.Mish(),
            nn.Linear(64, out_dim - 64),
        )

    def forward(self, scans, vec):
        z = self.conv(scans).squeeze(-1)
        return torch.cat([self.scan_head(z), self.vec_mlp(vec)], dim=-1)


class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(half, device=t.device) / (half - 1))
        ang = t.float()[:, None] * freqs[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class FiLMResBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out), nn.Mish())
        self.block2 = nn.Sequential(
            nn.Conv1d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out), nn.Mish())
        self.film = nn.Linear(cond_dim, 2 * c_out)
        self.skip = (nn.Conv1d(c_in, c_out, 1)
                     if c_in != c_out else nn.Identity())

    def forward(self, x, cond):
        h = self.block1(x)
        scale, bias = self.film(cond)[..., None].chunk(2, dim=1)
        h = h * (1.0 + scale) + bias
        h = self.block2(h)
        return h + self.skip(x)


class ConditionalUnet1D(nn.Module):
    """Denoises [B, 2, H] action sequences; H=8 -> two downsamplings."""
    """This is: Batchsize, meters ahead/to the size, horizon length of the action sequence. """

    def __init__(self, action_dim=2, obs_dim=112, time_dim=64,
                 channels=(64, 128, 256)):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmb(time_dim), nn.Linear(time_dim, 128), nn.Mish(),
            nn.Linear(128, 128))
        cond_dim = 128 + obs_dim
        c1, c2, c3 = channels

        self.down1 = FiLMResBlock(action_dim, c1, cond_dim)
        self.down2 = FiLMResBlock(c1, c2, cond_dim)
        self.pool = nn.AvgPool1d(2)
        self.mid = FiLMResBlock(c2, c3, cond_dim)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = FiLMResBlock(c3 + c2, c2, cond_dim)
        self.up1 = FiLMResBlock(c2 + c1, c1, cond_dim)
        self.head = nn.Conv1d(c1, action_dim, 1)

    def forward(self, x, t, obs_emb):
        cond = torch.cat([self.time_mlp(t), obs_emb], dim=-1)
        d1 = self.down1(x, cond)                     # [B, c1, H]
        d2 = self.down2(self.pool(d1), cond)         # [B, c2, H/2]
        m = self.mid(self.pool(d2), cond)            # [B, c3, H/4]
        u2 = self.up2(torch.cat([self.up(m), d2], dim=1), cond)
        u1 = self.up1(torch.cat([self.up(u2), d1], dim=1), cond)
        return self.head(u1)


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_stack=2, vec_dim=6, horizon=8, obs_dim=112,
                 channels=(64, 128, 256), scan_padding='circular'):
        super().__init__()
        self.horizon = horizon
        self.encoder = ObsEncoder(obs_stack, vec_dim, obs_dim,
                                  scan_padding=scan_padding)
        self.unet = ConditionalUnet1D(2, obs_dim, channels=channels)

    def eps(self, x_noisy, t, scans, vec):
        return self.unet(x_noisy, t, self.encoder(scans, vec))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
