"""Expert-episode dataset for diffusion-policy training.

Each sample is (observation, action):

  observation  scans[To, 360]  fixed-bin ranges normalized to [0, 1]
               vec[To * 6]     per frame: goal bearing sin/cos, clipped
                               distance, v, w, scan age (To=2: frames t-1, t)
  action       wp[2, H]        H=8 future robot-frame positions at
                               0.25 s spacing, min/max-normalized to [-1, 1]

Waypoint labels are built from the episode's ODOM pose stream (what the
deployed policy will integrate against), not ground truth. Past the episode
end the terminal pose is repeated — the expert stops at the goal, so the
padding teaches exactly that.

Conditioning: 'subgoal' (two-stage v1 — 1.2 m along the remaining expert
path, recorded by the collector), 'goal' (single-stage — final goal), or
'subgoal_goal' (two-stage v2 — subgoal AND final goal explicitly, 9 vec
features: the sampled waypoints are conditioned on the known target goal
directly, not only through the planner's subgoal). All columns live in
every npz, so one dataset serves every variant.

Validation split: episode index `val_episode` of EVERY training world is
held out (unseen trajectories, seen worlds). The 40 holdout WORLDS have no
episodes at all — they are reserved for closed-loop sim evaluation.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

N_SCAN = 360
VEC_DIM = 6            # 'subgoal' / 'goal' conditioning
VEC_DIM_SUB_GOAL = 9   # 'subgoal_goal' conditioning


def vec_dim(conditioning):
    return VEC_DIM_SUB_GOAL if conditioning == 'subgoal_goal' else VEC_DIM


def obs_vec(cond_dx, cond_dy, cond_dist, v, w, scan_age, goal_clip,
            goal_dx=None, goal_dy=None, goal_dist=None):
    """The observation vector — single source of truth, used by training
    (ExpertDataset) and deployment (policy_node) identically.

    6 features: conditioning-point bearing sin/cos + clipped distance, v, w,
    scan age. With the optional goal triple (subgoal_goal conditioning), the
    final goal's bearing sin/cos + clipped distance are inserted after the
    conditioning-point features (9 total)."""
    bearing = math.atan2(cond_dy, cond_dx)
    feats = [math.sin(bearing), math.cos(bearing),
             min(cond_dist, goal_clip) / goal_clip]
    if goal_dx is not None:
        gb = math.atan2(goal_dy, goal_dx)
        feats += [math.sin(gb), math.cos(gb),
                  min(goal_dist, goal_clip) / goal_clip]
    feats += [v / 0.22, w / 2.84, min(scan_age, 0.5) / 0.5]
    return np.array(feats, dtype=np.float32)


def _episode_files(data_dir, val, val_episode):
    files = sorted(Path(data_dir).glob('ep_*.npz'))
    keep = []
    for f in files:
        ep = int(f.stem.split('_')[2])
        if (ep == val_episode) == val:
            keep.append(f)
    if not keep:
        raise FileNotFoundError(f'no episodes in {data_dir} (val={val})')
    return keep


def crop_arc(scans, arc_deg):
    """Forward-centered arc crop of 360-bin scans (bin 0 = straight ahead):
    [-arc/2, +arc/2) as a CONTIGUOUS arc_deg-wide array. Single source of
    truth — training and policy_node deployment must crop identically."""
    if arc_deg >= 360:
        return scans
    half = int(arc_deg) // 2
    return np.concatenate([scans[..., -half:],
                           scans[..., :int(arc_deg) - half]], axis=-1)


def _motion_class(pose, t, seg_len_m):
    """Curvature class of the ~seg_len_m window starting at t: what kind of
    motion is the label teaching here? (v3 segment balancing, plan §1)."""
    xy = pose[t:, :2]
    if len(xy) < 3:
        return 'straight'
    d = np.hypot(*np.diff(xy, axis=0).T)
    s = np.cumsum(d)
    end = int(np.searchsorted(s, seg_len_m)) + 1
    w = pose[t:t + max(end, 3)]
    if len(w) < 3:
        return 'straight'
    heads = np.arctan2(np.diff(w[:, 1]), np.diff(w[:, 0]))
    dh = np.arctan2(np.sin(np.diff(heads)), np.cos(np.diff(heads)))
    net = math.degrees(dh.sum())
    total = math.degrees(np.abs(dh).sum())
    if total > 60 and abs(net) < 30:
        return 's_curve'
    if abs(net) >= 45:
        return 'hard_left' if net > 0 else 'hard_right'
    if abs(net) >= 15:
        return 'gentle_left' if net > 0 else 'gentle_right'
    return 'straight'


class ExpertDataset(Dataset):
    def __init__(self, data_dir, conditioning='subgoal', horizon=8,
                 obs_stack=2, goal_clip=3.0, val=False, val_episode=3,
                 stats=None, scan_arc_deg=360, corrupt_frac=0.0,
                 segment_balance=False, seg_len_m=1.5, balance_seed=0):
        assert conditioning in ('subgoal', 'goal', 'subgoal_goal')
        self.conditioning = conditioning
        self.vec_dim = vec_dim(conditioning)
        self.horizon = horizon
        self.obs_stack = obs_stack
        self.goal_clip = goal_clip
        self.scan_arc_deg = scan_arc_deg
        self.corrupt_frac = corrupt_frac
        self._balance = (segment_balance, seg_len_m, balance_seed)

        cond_col = 'subgoal' if conditioning == 'subgoal_goal' else conditioning
        self.episodes = []
        self.index = []
        for f in _episode_files(data_dir, val, val_episode):
            with np.load(f) as z:
                ep = {
                    'scans': (z['scans'] / 3.5).astype(np.float32),
                    'pose': z['odom'][:, :3].astype(np.float64),   # odom frame
                    'vel': z['odom'][:, 3:5].astype(np.float32),
                    'scan_age': z['odom'][:, 5].astype(np.float32),
                    'cond': z[cond_col].astype(np.float32),
                }
                if conditioning == 'subgoal_goal':
                    ep['goal'] = z['goal'].astype(np.float32)
            e = len(self.episodes)
            self.episodes.append(ep)
            self.index.extend((e, t) for t in range(len(ep['pose'])))

        if self._balance[0]:
            self._balance_index()
        self.stats = stats if stats is not None else self._compute_stats()

    def _balance_index(self):
        """v3 motion-class balancing (plan §1): classify every timestep by
        the ~1.5 m window it labels, then OVERSAMPLE minority classes to
        parity — no straight-dominated batches, no train-loop changes."""
        _, seg_len, seed = self._balance
        classes = [
            _motion_class(self.episodes[e]['pose'], t, seg_len)
            for e, t in self.index]
        from collections import Counter
        counts = Counter(classes)
        target = max(counts.values())
        rng = np.random.default_rng(seed)
        extra = []
        for cls, cnt in counts.items():
            if cnt >= target:
                continue
            members = [i for i, c in enumerate(classes) if c == cls]
            extra.extend(self.index[i] for i in
                         rng.choice(members, size=target - cnt))
        self.index.extend(extra)
        self.class_counts = dict(counts)

    # ── normalization ─────────────────────────────────────────────────────────
    def _compute_stats(self, sample_every=7):
        pts = []
        for e, t in self.index[::sample_every]:
            pts.append(self._waypoints(e, t))
        pts = np.concatenate(pts, axis=0)
        lo = pts.min(axis=0) - 1e-3
        hi = pts.max(axis=0) + 1e-3
        return {'wp_lo': lo.astype(np.float32), 'wp_hi': hi.astype(np.float32)}

    def normalize_wp(self, wp):
        lo, hi = self.stats['wp_lo'], self.stats['wp_hi']
        return (2.0 * (wp - lo) / (hi - lo) - 1.0).astype(np.float32)

    def denormalize_wp(self, wp_n):
        lo, hi = self.stats['wp_lo'], self.stats['wp_hi']
        return (wp_n + 1.0) / 2.0 * (hi - lo) + lo

    # ── label / feature assembly ──────────────────────────────────────────────
    def _waypoints(self, e, t):
        """H future positions in the robot frame at time t (odom stream)."""
        pose = self.episodes[e]['pose']
        T = len(pose)
        x, y, yaw = pose[t]
        c, s = math.cos(yaw), math.sin(yaw)
        idx = np.minimum(np.arange(t + 1, t + 1 + self.horizon), T - 1)
        d = pose[idx, :2] - (x, y)
        return np.stack([c * d[:, 0] + s * d[:, 1],
                         -s * d[:, 0] + c * d[:, 1]], axis=1)  # [H, 2]

    def _vec(self, e, t):
        ep = self.episodes[e]
        cx, cy, cd = ep['cond'][t]
        goal = {}
        if self.conditioning == 'subgoal_goal':
            gx, gy, gd = ep['goal'][t]
            goal = dict(goal_dx=gx, goal_dy=gy, goal_dist=gd)
        return obs_vec(cx, cy, cd, ep['vel'][t, 0], ep['vel'][t, 1],
                       ep['scan_age'][t], self.goal_clip, **goal)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        frames = [max(0, t - k) for k in range(self.obs_stack - 1, -1, -1)]
        scans = crop_arc(self.episodes[e]['scans'][frames],
                         self.scan_arc_deg)                      # [To, arc]
        if self.corrupt_frac > 0.0:
            # v3 augmentation (plan §4): a fraction of bins get Gaussian
            # noise or dropout-to-max-range — real-LDS variance + occlusion
            scans = scans.copy()
            mask = np.random.random(scans.shape) < self.corrupt_frac
            drop = mask & (np.random.random(scans.shape) < 0.5)
            noise = mask & ~drop
            scans[drop] = 1.0
            scans[noise] = np.clip(
                scans[noise] + np.random.normal(0, 0.05, noise.sum()),
                0.0, 1.0)
        vec = np.stack([self._vec(e, f) for f in frames])        # [To, 6]
        wp = self.normalize_wp(self._waypoints(e, t))            # [H, 2]
        return (torch.from_numpy(np.ascontiguousarray(scans)),
                torch.from_numpy(vec.reshape(-1)),
                torch.from_numpy(wp.T.copy()))                   # [2, H]
