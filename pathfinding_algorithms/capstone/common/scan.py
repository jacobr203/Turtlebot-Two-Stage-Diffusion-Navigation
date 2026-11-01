"""Lidar preprocessing shared by the recorder, the policy, and evaluation.

Scans go onto a fixed 360-bin grid (bin b is b degrees CCW from +x) so models
trained on sim scans consume the real LDS-02's scans unchanged.
"""

import numpy as np

N_BINS = 360
R_MIN = 0.12
R_MAX = 3.5


def resample_scan(ranges, angle_min, angle_increment,
                  n_bins=N_BINS, r_min=R_MIN, r_max=R_MAX):
    """Map a LaserScan's ranges onto n_bins fixed angular bins.

    Invalid returns (nan/inf/<r_min) and empty bins are r_max; bins hit by
    several rays keep the nearest.
    """
    r = np.asarray(ranges, dtype=np.float32)
    angles = angle_min + np.arange(len(r)) * angle_increment
    valid = np.isfinite(r) & (r >= r_min)
    r = np.where(valid, np.clip(r, r_min, r_max), r_max)

    bins = np.round(angles / (2.0 * np.pi) * n_bins).astype(int) % n_bins
    out = np.full(n_bins, r_max, dtype=np.float32)
    np.minimum.at(out, bins, r)
    return out


def normalize(scan, r_max=R_MAX):
    """[r_min, r_max] -> [0, 1] for the policy's encoder."""
    return np.asarray(scan, dtype=np.float32) / r_max
