"""
Tests for DWACameraNode's stuck handling: wheel-slip detection, safety-bubble
retreat trajectories, and the Nav2-style recovery state machine
(BackUp / Forward / Spin / Wait with escalation).

Run with:
  cd ~/pathfinding_algorithms
  source install/setup.bash
  python3 -m pytest dwa_camera_avoidance/test/ -v
"""

import math
import time

import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import LaserScan

from dwa_camera_avoidance.dwa_camera_node import (
    DWACameraNode,
    BACKUP_DIST,
    BUBBLE_STUCK_S,
    SPIN_ANGLE,
    WAIT_S,
    SLIP_WINDOW_S,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture()
def node(ros):
    n = DWACameraNode()
    yield n
    n.destroy_node()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan(n_rays=60, ranges=None, angle_min=0.0, range_min=0.05, range_max=3.5):
    msg = LaserScan()
    msg.angle_min = angle_min
    msg.angle_increment = 2.0 * math.pi / n_rays
    msg.range_min = range_min
    msg.range_max = range_max
    msg.ranges = list(ranges) if ranges is not None else [1.0] * n_rays
    return msg


def _prime_slip_history(node, now, disp_x, new_ranges, old_ranges=None,
                        yaw_new=0.0):
    """Fill odom/scan history simulating 1 s of claimed +x travel."""
    n = len(new_ranges)
    old = np.array(old_ranges if old_ranges is not None else [1.0] * n,
                   dtype=float)
    node._scan_msg = _scan(n_rays=n, ranges=new_ranges)
    node._odom_hist.clear()
    node._scan_hist.clear()
    node._odom_hist.append((now - 1.0, 0.0, 0.0, 0.0))
    node._odom_hist.append((now, disp_x, 0.0, yaw_new))
    node._scan_hist.append((now - 1.0, old))
    node._scan_hist.append((now, np.array(new_ranges, dtype=float)))


# ── Wheel-slip detection ──────────────────────────────────────────────────────

def test_slip_detected_when_scan_frozen(node):
    """Odom claims 15 cm of travel, scan identical → slip."""
    node._last_cmd_v = 0.1
    _prime_slip_history(node, 1000.0, disp_x=0.15, new_ranges=[1.0] * 60)
    assert node._slip_detected(1000.0) is True


def test_no_slip_when_scan_tracks_motion(node):
    """Scan ranges change consistently with the claimed motion → no slip."""
    node._last_cmd_v = 0.1
    angles = np.arange(60) * (2.0 * math.pi / 60)
    moved = list(1.0 - 0.15 * np.cos(angles))   # walls approach as expected
    _prime_slip_history(node, 1000.0, disp_x=0.15, new_ranges=moved)
    assert node._slip_detected(1000.0) is False


def test_slip_abstains_in_open_space(node):
    """Nearly all rays infinite → too few informative rays → abstain."""
    node._last_cmd_v = 0.1
    ranges = [float('inf')] * 60
    for i in range(5):
        ranges[i] = 1.0
    _prime_slip_history(node, 1000.0, disp_x=0.15,
                        new_ranges=ranges, old_ranges=ranges)
    assert node._slip_detected(1000.0) is False


def test_slip_skipped_while_turning(node):
    """Yaw changed > threshold over the window → scan shift is legitimate."""
    node._last_cmd_v = 0.1
    _prime_slip_history(node, 1000.0, disp_x=0.15, new_ranges=[1.0] * 60,
                        yaw_new=0.5)
    assert node._slip_detected(1000.0) is False


def test_slip_gated_on_commanded_motion(node):
    """No forward command → slip check must not fire."""
    node._last_cmd_v = 0.0
    _prime_slip_history(node, 1000.0, disp_x=0.15, new_ranges=[1.0] * 60)
    assert node._slip_detected(1000.0) is False


def test_slip_requires_odom_displacement(node):
    """Odom claims almost no travel → not slip (just slow/stopped)."""
    node._last_cmd_v = 0.1
    _prime_slip_history(node, 1000.0, disp_x=0.01, new_ranges=[1.0] * 60)
    assert node._slip_detected(1000.0) is False


# ── Safety-bubble retreat ─────────────────────────────────────────────────────

def test_retreat_clearance_accepts_retreating_path(node):
    obstacles = [(0.15, 0.0)]                      # inside bubble, ahead
    traj = [(-0.05, 0.0, math.pi)]                 # backing away
    assert node._retreat_clearance(traj, 0.15, obstacles) == pytest.approx(0.20)


def test_retreat_clearance_rejects_approaching_path(node):
    obstacles = [(0.15, 0.0)]
    traj = [(0.05, 0.0, 0.0)]                      # moving closer
    assert node._retreat_clearance(traj, 0.15, obstacles) is None


def test_dwa_escapes_bubble_instead_of_deadlocking(node):
    """Obstacle already inside the bubble behind the robot: DWA must still
    find a trajectory (the pre-fix behavior rejected everything)."""
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]
    ring = [(3.0 * math.cos(a), 3.0 * math.sin(a))
            for a in np.arange(0, 2 * math.pi, math.pi / 8)]
    obstacles = ring + [(-0.16, 0.0)]              # touching the rear shell
    v, w = node._dwa(obstacles)
    assert node._dwa_failed is False
    assert v >= 0.0                                # retreat here = drive away


# ── Recovery state machine ────────────────────────────────────────────────────

def test_recovery_prefers_backup_when_rear_clear(node, monkeypatch):
    monkeypatch.setattr(node, '_rear_clear', lambda: True)
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]
    node._start_recovery(1000.0, 'test')
    assert node._recovery_stage == 'backup'
    assert node._backup_target == pytest.approx(BACKUP_DIST)
    assert node._spin_target == pytest.approx(SPIN_ANGLE)


def test_recovery_escapes_forward_when_rear_blocked(node, monkeypatch):
    monkeypatch.setattr(node, '_rear_clear', lambda: False)
    monkeypatch.setattr(node, '_has_clear_forward', lambda: True)
    node._start_recovery(1000.0, 'test')
    assert node._recovery_stage == 'forward'


def test_recovery_spins_when_boxed_in(node, monkeypatch):
    monkeypatch.setattr(node, '_rear_clear', lambda: False)
    monkeypatch.setattr(node, '_has_clear_forward', lambda: False)
    node._start_recovery(1000.0, 'test')
    assert node._recovery_stage == 'spin'


def test_recovery_full_sequence_backup_spin_wait(node, monkeypatch):
    monkeypatch.setattr(node, '_rear_clear', lambda: True)
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]
    node._odom_ready = False                       # keeps _auto_pick_goal inert
    node._start_recovery(1000.0, 'test')
    assert node._recovery_stage == 'backup'

    node._run_recovery(1001.0)                     # not far enough yet
    assert node._recovery_stage == 'backup'

    node._state[0] = -(BACKUP_DIST + 0.05)         # backed up past target
    node._run_recovery(1002.0)
    assert node._recovery_stage == 'spin'

    node._state[2] = 1.0                           # rotated 1.0 rad
    node._run_recovery(1003.0)
    assert node._recovery_stage == 'spin'

    node._state[2] = 2.0                           # cumulative 2.0 > 90°
    node._run_recovery(1004.0)
    assert node._recovery_stage == 'wait'

    node._run_recovery(1004.0 + WAIT_S / 2)        # still waiting
    assert node._recovery_stage == 'wait'

    node._run_recovery(1004.0 + WAIT_S + 0.1)
    assert node._recovery_stage is None            # done


def test_recovery_escalates_on_rapid_retrigger(node, monkeypatch):
    monkeypatch.setattr(node, '_rear_clear', lambda: True)
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]
    node._start_recovery(1000.0, 'first')
    node._finish_recovery(1005.0)
    node._start_recovery(1010.0, 'second')         # within escalate window
    assert node._recovery_level == 1
    assert node._backup_target > BACKUP_DIST
    assert node._spin_target > SPIN_ANGLE


def test_forward_escape_skips_spin(node, monkeypatch):
    monkeypatch.setattr(node, '_rear_clear', lambda: False)
    monkeypatch.setattr(node, '_has_clear_forward', lambda: True)
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]
    node._odom_ready = False
    node._start_recovery(1000.0, 'test')
    node._state[0] = node._backup_target + 0.05    # escaped forward
    node._run_recovery(1001.0)
    assert node._recovery_stage == 'wait'          # no spin after forward


# ── Rear clearance helper ─────────────────────────────────────────────────────

def test_rear_clear_true_with_open_rear(node):
    node._scan_msg = _scan(n_rays=36, ranges=[1.0] * 36)
    assert node._rear_clear() is True


def test_rear_clear_false_with_blocked_rear(node):
    ranges = [1.0] * 36
    for i in range(14, 23):                        # ±30° around 180°
        ranges[i] = 0.1
    node._scan_msg = _scan(n_rays=36, ranges=ranges)
    assert node._rear_clear() is False


def test_rear_clear_handles_negative_angle_min(node):
    """Same geometry expressed as a -π..π scan (rear at the array edges)."""
    ranges = [1.0] * 36
    for i in list(range(0, 3)) + list(range(33, 36)):
        ranges[i] = 0.1
    node._scan_msg = _scan(n_rays=36, ranges=ranges, angle_min=-math.pi)
    assert node._rear_clear() is False


# ── Bubble dwell escalation ──────────────────────────────────────────────────

def test_bubble_dwell_triggers_recovery(node, monkeypatch):
    """Retreat mode legally allows orbiting an obstacle at constant distance
    forever; staying inside the bubble past BUBBLE_STUCK_S must escalate to
    the recovery machine (this trapped the sim robot at a pillar, bug #017)."""
    monkeypatch.setattr(node, '_rear_clear', lambda: True)
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]
    node._odom_ready = True
    node._scan_msg = _scan()
    node._goal = (5.0, 5.0)
    # obstacle glued to the rear shell keeps _dwa in violation every tick
    ring = [(3.0 * math.cos(a), 3.0 * math.sin(a))
            for a in np.arange(0, 2 * math.pi, math.pi / 8)]
    node._lidar_obstacles = ring + [(-0.16, 0.0)]
    node._camera_obstacles = []

    real_clock = node.get_clock().now
    t = {'now': 1000.0}
    monkeypatch.setattr(node, 'get_clock', lambda: type(
        'C', (), {'now': staticmethod(lambda: type(
            'T', (), {'nanoseconds': int(t['now'] * 1e9)})())})())

    node._control_loop()                      # enters bubble, timer starts
    assert node._in_violation is True
    assert node._recovery_stage is None

    t['now'] = 1000.0 + BUBBLE_STUCK_S + 0.5  # dwell exceeded
    node._control_loop()
    assert node._recovery_stage is not None   # escalated to recovery


def test_bubble_timer_resets_when_clear(node):
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]
    node._odom_ready = True
    node._scan_msg = _scan()
    node._goal = (5.0, 5.0)
    node._lidar_obstacles = [(3.0, 0.0)]      # far away — no violation
    node._camera_obstacles = []
    node._bubble_since = 123.0                # stale timer from a past bubble
    node._control_loop()
    assert node._in_violation is False
    assert node._bubble_since is None


# ── Caged behavior ────────────────────────────────────────────────────────────

def test_caged_robot_does_not_ram_the_cage(node):
    """Fully surrounded at 5 cm: rotation in place stays legal under the
    retreat rule (distance never shrinks), but forward motion must not be
    chosen at any meaningful speed."""
    node._state = [0.0, 0.0, 0.0, 0.0, 0.0]
    cage = [(0.05 * math.cos(a), 0.05 * math.sin(a))
            for a in np.arange(0, 2 * math.pi, math.pi / 16)]
    v, _ = node._dwa(cage)
    assert node._dwa_failed is False
    assert v <= 0.05
