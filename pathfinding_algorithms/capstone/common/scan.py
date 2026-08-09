"""Lidar preprocessing shared by the recorder, the policy, and evaluation.

Scans are resampled onto a fixed 360-bin angular grid in the robot frame
(bin b covers angles around b degrees, counter-clockwise from +x). This
absorbs differences between the sim lidar (exactly 360 rays at angle_min 0)
and the real LDS-02 (whose beam count/offset differ), so models trained on
sim scans consume real scans unchanged.
"""

import numpy as np

N_BINS = 360
R_MIN = 0.12
R_MAX = 3.5


def resample_scan(ranges, angle_min, angle_increment,
                  n_bins=N_BINS, r_min=R_MIN, r_max=R_MAX):
    """Map a LaserScan's ranges onto n_bins fixed angular bins.

    Invalid returns (nan/inf/<r_min) become r_max ("no obstacle seen").
    Bins hit by several rays keep the MINIMUM (nearest obstacle wins);
    bins hit by none are r_max.
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
