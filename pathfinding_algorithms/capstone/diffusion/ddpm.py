"""DDPM machinery: squared-cosine schedule, training loss, samplers, EMA.

Plain torch — no diffusers dependency (keeps the venv lean; this is ~150
lines of schedule math we control and can cite line-by-line).

T=35 denoising steps is the proposal's number and the default for both
training and inference; DDIM with fewer steps is the risk-table fallback
("reduce diffusion steps from 35").
"""

import copy
import torch


def cosine_alphas_cumprod(T, s=0.008):
    """Nichol & Dhariwal squared-cosine noise schedule."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * torch.pi / 2) ** 2
    ac = (f / f[0]).clamp(1e-5, 1.0)
    return ac[1:].float() # alpha-bar for t = 0..T-1


class DDPM:
    def __init__(self, T=35, device='cpu'):
        self.T = T
        ac = cosine_alphas_cumprod(T).to(device)
        self.ac = ac
        self.sqrt_ac = ac.sqrt()
        self.sqrt_1mac = (1 - ac).sqrt()
        ac_prev = torch.cat([torch.ones(1, device=device), ac[:-1]])
        self.alphas = ac / ac_prev
        self.betas = 1 - self.alphas
        self.post_var = self.betas * (1 - ac_prev) / (1 - ac)
        # q(x_{t-1} | x_t, x0) mean = c0 * x0 + c1 * x_t
        self.post_c0 = ac_prev.sqrt() * self.betas / (1 - ac)
        self.post_c1 = self.alphas.sqrt() * (1 - ac_prev) / (1 - ac)

    # ── training ──────────────────────────────────────────────────────────────
    def loss(self, policy, x0, scans, vec):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = (self.sqrt_ac[t, None, None] * x0
               + self.sqrt_1mac[t, None, None] * noise)
        eps_hat = policy.eps(x_t, t, scans, vec)
        return torch.nn.functional.mse_loss(eps_hat, noise)

    # ── sampling ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, policy, scans, vec, shape=None, guide=None,
               guide_steps=10):
        """Full-T ancestral DDPM sampling. Returns normalized actions.

        guide (optional): callable x0_hat -> x0_hat applied on the last
        `guide_steps` denoise steps — cost-guided sampling in the
        Motion-Planning-Diffusion / NaviDiffusor pattern (expert R2).
        The model remains the author: guidance only nudges the clean
        estimate before the posterior step. guide=None (the default,
        and always for the goto benchmark) is byte-identical to the
        unguided sampler."""
        B = scans.shape[0]
        shape = shape or (B, 2, policy.horizon)
        x = torch.randn(shape, device=scans.device)
        obs_emb = policy.encoder(scans, vec)
        for ti in reversed(range(self.T)):
            t = torch.full((B,), ti, device=x.device, dtype=torch.long)
            eps = policy.unet(x, t, obs_emb)
            x0_hat = ((x - self.sqrt_1mac[ti] * eps) / self.sqrt_ac[ti]) \
                .clamp(-1.5, 1.5)
            if guide is not None and ti < guide_steps:
                x0_hat = guide(x0_hat).clamp(-1.5, 1.5)
            if ti == 0:
                x = x0_hat
            else:
                mean = self.post_c0[ti] * x0_hat + self.post_c1[ti] * x
                x = mean + self.post_var[ti].sqrt() * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample_ddim(self, policy, scans, vec, steps=10, shape=None,
                    guide=None, guide_steps=4):
        """Deterministic DDIM (eta=0) on a T-step-strided sub-schedule.
        guide: same contract as sample() — applied on the last
        guide_steps of the strided schedule."""
        B = scans.shape[0]
        shape = shape or (B, 2, policy.horizon)
        x = torch.randn(shape, device=scans.device)
        obs_emb = policy.encoder(scans, vec)
        seq = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        for i, ti in enumerate(seq):
            t = torch.full((B,), ti, device=x.device, dtype=torch.long)
            eps = policy.unet(x, t, obs_emb)
            x0_hat = ((x - self.sqrt_1mac[ti] * eps) / self.sqrt_ac[ti]) \
                .clamp(-1.5, 1.5)
            if guide is not None and i >= steps - guide_steps:
                x0_hat = guide(x0_hat).clamp(-1.5, 1.5)
            ac_next = self.ac[seq[i + 1]] if i + 1 < len(seq) \
                else torch.ones((), device=x.device)
            x = ac_next.sqrt() * x0_hat + (1 - ac_next).sqrt() * eps
        return x


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p, 1.0 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)
