"""Small 2D frame/angle helpers shared by expert, deployment, and evaluation."""

import math

import numpy as np


def yaw_from_quaternion(x, y, z, w):
    """Z-axis rotation of a (x, y, z, w) quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def world_to_robot(pts, pose):
    """Transform world-frame points into the robot frame.

    pts: (2,) or (N, 2); pose: (x, y, yaw). Robot frame: +x forward, +y left.
    """
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.atleast_2d(np.asarray(pts, dtype=np.float64)) - (x, y)
    out = np.stack([c * p[:, 0] + s * p[:, 1],
                    -s * p[:, 0] + c * p[:, 1]], axis=1)
    return out[0] if np.ndim(pts) == 1 else out


def robot_to_world(pts, pose):
    """Inverse of world_to_robot."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    out = np.stack([c * p[:, 0] - s * p[:, 1] + x,
                    s * p[:, 0] + c * p[:, 1] + y], axis=1)
    return out[0] if np.ndim(pts) == 1 else out
