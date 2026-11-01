"""Last check between the policy and the wheels: a classical speed gate,
recovery behaviours (back up, rotate, escape) and the stall watchdog. Separate
from the policy so the evaluation harness can switch it off.
"""

import math
import os
import time

import numpy as np

from geometry_msgs.msg import Twist

from common.geometry import yaw_from_quaternion


class SafetyMixin:
    """Safety gate, recovery, and the stall watchdog."""


    def _fused_gate_ranges(self):
        """Lidar gate ranges min-fused with the camera virtual scan (if fresh).
        Camera bins with no data are inf, so fusion only ever ADDS obstacles."""
        r = self._gate_ranges if self._gate_ranges is not None else self._ranges_m
        if (self._cam_gate is not None
                and time.time() - self._cam_gate_t <= 1.5):
            r = np.minimum(r, self._cam_gate)
        return r


    def _forward_clearance(self, center_deg=0):
        r = self._fused_gate_ranges()
        h = self._gate_half
        if center_deg:
            r = np.roll(r, -int(center_deg))
        return min(r[:h + 1].min(), r[-h:].min())


    def _gate_forward_speed(self, v, w=0.0):
        """Scale forward speed by clearance; 0 blocks translation but leaves
        rotation free. The cone leans into the commanded turn, where side clips
        happen. Margins relax to the benchmark 0.16 m stop while a seek goal is
        active or during an escape: the 0.21 m stop walls off ~0.35 m gaps."""
        if not self._gate or v <= 0.0 or self._ranges_m is None:
            return v
        lean = max(-35, min(35, int(math.degrees(w * 0.4))))
        ahead = self._forward_clearance(lean)
        stop, band = self._gate_stop, self._gate_band
        if getattr(self, '_tight_ok', False) or self._escaping():
            stop, band = min(stop, 0.16), min(band, 0.25)
        scale = (ahead - stop) / band
        scale = max(0.0, min(1.0, scale))
        if scale < 1.0:
            self.get_logger().warning(
                f'safety gate: {ahead:.2f} m ahead — speed x{scale:.2f}',
                throttle_duration_sec=2.0)
        return v * scale


    def _rear_clearance(self):
        # wide ±35° sector: a narrow cone misses pillars in the rear quarter
        r = self._fused_gate_ranges()
        return r[145:216].min()


    def _spin_safe(self):
        """True when an in-place rotation cannot clip anything: hull corners
        sweep ~0.14 m. Tight mode relaxes the threshold to 0.15 m so facing
        turns at pocket mouths aren't vetoed."""
        r = self._fused_gate_ranges()
        tight = getattr(self, '_tight_ok', False) or self._escaping()
        thr = 0.15 if tight else 0.17
        return r is None or float(np.min(r)) > thr


    def _escaping(self):
        """Pinch escape active: gate margins stay relaxed until 1.2 m from the
        pinch or the 30 s leash."""
        if self.behavior != 'seek' or self._escape_from is None:
            return False
        if time.time() >= self._escape_until:
            self._escape_from = None
            return False
        p = self._world_pose()
        if math.hypot(p[0] - self._escape_from[0],
                      p[1] - self._escape_from[1]) > 1.2:
            self._escape_from = None
            self.get_logger().info('pinch escape complete — normal margins')
            return False
        return True


    def _finish_recovery(self):
        self._recover_phase = None
        self._gate_blocked_since = None
        self._stall_since = None
        self._pose_hist.clear() # fresh window
        if self.behavior in ('wander', 'seek'):
            self._wander_next = 0.0 # fresh goal now
        self.get_logger().info('gate recovery done — resuming')


    def _enter_recovery(self, reason, backup_s=None):
        now = time.time()
        if self.behavior == 'seek' and self._odom is not None:
            p = self._world_pose()
            self._recovery_spots.append((now, p[0], p[1]))
            near = sum(1 for t, hx, hy in self._recovery_spots
                       if now - t < 90.0
                       and math.hypot(p[0] - hx, p[1] - hy) < 1.0)
            if near >= 3:
                # recovery-loop breaker: without it the robot ping-pongs into
                # the same pinch. Consume the strikes, escape on relaxed
                # margins and blacklist the pocket's region. Explore faces get
                # abandoned; target approaches shift bearing.
                self._recovery_spots.clear()
                self._escape_until = now + 30.0
                self._escape_from = (p[0], p[1])
                mp = self._map_pose(p)
                if mp is not None:
                    b = (int(math.floor(mp[0] / 2.0)),
                         int(math.floor(mp[1] / 2.0)))
                    self._region_block[b] = now + 120.0
                # colour branch first: a stale _face_target must not shadow a
                # live target
                if self._seek_color is None and self._face_target is not None:
                    t = self._face_target
                    fb = (int(math.floor(t['map_xy'][0] / 2.0)),
                          int(math.floor(t['map_xy'][1] / 2.0)))
                    self._region_block[fb] = now + 120.0
                    # cooled, not retired: the face was never inspected.
                    # Outlasts the 120 s region block, so the region reopens first.
                    self._face_cool[t['key']] = now + 180.0
                    if t.get('map_xy'):
                        self._cool_spots = getattr(self, '_cool_spots', [])
                        self._cool_spots.append(
                            (tuple(t['map_xy']), now + 180.0))
                    self._face_target = None
                    # escape owns its exit goal, or the next pick seizes it
                    self.goal = None
                    if self._ranges_m is not None:
                        wm_b = np.min(np.stack(
                            [np.roll(self._ranges_m, k)
                             for k in range(-15, 16)]), axis=0)
                        b_b = int(np.argmax(wm_b))
                        yaw_b = self._world_pose()[2]
                        ang_b = yaw_b + math.radians(b_b)
                        self.goal = (p[0] + 1.5 * math.cos(ang_b),
                                     p[1] + 1.5 * math.sin(ang_b))
                    self._explore_churn += 1 # frontier-push trigger
                    self.get_logger().warning(
                        'recovery loop at a pinch — abandoning that face, '
                        'blacklisting its region, escaping on relaxed '
                        'margins')
                elif self._seek_color is not None:
                    # recoveries vary the approach side but never burn the
                    # give-up budget; the progress-based patience timer does
                    self._seek_attempts = min(self._seek_attempts + 1, 7)
                    self._seek_repositioning = False
                    self.get_logger().warning(
                        'recovery loop at a pinch — next routable '
                        'approach side')
        # alternate sides: random turns can undo each other
        self._recover_dir = -getattr(self, '_recover_dir', -1.0)
        # close colour approaches keep recoveries short, or each one undoes the
        # approach; the wedge escalation below still outranks this cap
        if self.behavior == 'seek' and backup_s is None \
                and self._seek_color is not None:
            backup_s = min(self._gate_backup_s, 2.5)
        # wedge escalation: consecutive stall or contact recoveries without
        # 0.15 m of displacement back up longer and alternate turn sides. Past
        # 4, log loudly: a dead run must read as dead, not exploration churn.
        if reason.startswith(('physical stall', 'post-collision')) \
                and self._odom is not None:
            p = self._world_pose()
            lp = self._stall_recov_pose
            if lp is not None and \
                    math.hypot(p[0] - lp[0], p[1] - lp[1]) < 0.15:
                self._stall_streak += 1
            else:
                self._stall_streak = 0
            self._stall_recov_pose = (p[0], p[1])
            if self._stall_streak >= 2:
                backup_s = max(backup_s or self._gate_backup_s, 4.0)
                self._recover_dir = 1.0 if self._stall_streak % 2 else -1.0
            if self._stall_streak >= 4:
                self.get_logger().error(
                    f'TERMINAL WEDGE suspected — {self._stall_streak} '
                    'stall recoveries without displacement')
        self._backup_s_active = backup_s or self._gate_backup_s
        self._rotate_start_yaw = None # captured when rotation begins
        if self._rear_clearance() >= self._gate_rear_clear:
            self._recover_phase = 'backup'
            self._phase_until = now + self._backup_s_active
        elif not self._spin_safe():
            # rear blocked and too tight to spin: creep along the freer axis
            # only if it has room (creep bypasses the gate), else rotate
            rear_c, fwd_c = self._rear_clearance(), self._forward_clearance()
            if max(rear_c, fwd_c) > 0.18:
                self._creep_v = -0.05 if rear_c >= fwd_c else 0.05
                self._recover_phase = 'creep'
                self._phase_until = now + 2.0
            else:
                self._recover_phase = 'rotate'
                self._phase_until = now + self._gate_recover_timeout
        else:
            self._recover_phase = 'rotate' # rear blocked
            self._phase_until = now + self._gate_recover_timeout
        self._gate_blocked_since = None
        self._stall_since = None
        self.get_logger().warning(
            f'gate recovery ({reason}): {self._recover_phase} '
            f'({self._backup_s_active:.1f} s), then a >='
            f'{math.degrees(self._min_turn_rad):.0f}° turn '
            f'{"left" if self._recover_dir > 0 else "right"} (alternating)')


    def _gate_recovery(self, v_wanted, v_gated):
        """Stuck-against-obstacle escape: back up, rotate to open space.
        Returns a Twist to publish instead of the tracker's command, or None
        for normal operation."""
        now = time.time()

        if self._recover_phase == 'creep':
            # wedged too tight to spin: translate along the freer axis until
            # rotation is safe, re-checking clearance as the axis can close
            if now < self._phase_until and not self._spin_safe():
                ahead_c = self._forward_clearance() if self._creep_v > 0 \
                    else self._rear_clearance()
                if ahead_c > 0.14:
                    cmd = Twist()
                    cmd.linear.x = self._creep_v
                    return cmd
            self._recover_phase = 'rotate'
            self._phase_until = now + self._gate_recover_timeout

        if self._recover_phase == 'backup':
            if now < self._phase_until \
                    and self._rear_clearance() >= self._gate_rear_clear * 0.8:
                cmd = Twist()
                cmd.linear.x = -self._gate_backup_v
                return cmd
            if not self._spin_safe():
                self._creep_v = -0.05 if self._rear_clearance() >= \
                    self._forward_clearance() else 0.05
                self._recover_phase = 'creep'
                self._phase_until = now + 2.0
                return self._gate_recovery(v_wanted, v_gated)
            self._recover_phase = 'rotate'
            self._phase_until = now + self._gate_recover_timeout
        if self._recover_phase == 'rotate':
            q = self._odom.pose.pose.orientation
            yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
            if self._rotate_start_yaw is None:
                self._rotate_start_yaw = yaw
            turned = abs(math.atan2(math.sin(yaw - self._rotate_start_yaw),
                                    math.cos(yaw - self._rotate_start_yaw)))
            # always complete the minimum turn; keep going past it only
            # while the forward cone is still blocked
            if now < self._phase_until and (
                    turned < self._min_turn_rad
                    or self._forward_clearance() < self._gate_resume):
                cmd = Twist()
                cmd.angular.z = self._recover_dir * self._gate_recover_w
                return cmd
            self._finish_recovery() # turned + clear, or timed out

        # trigger 1: scan-blocked (gate holding the robot at zero)
        blocked = v_wanted > 0.03 and v_gated < 0.01
        if blocked:
            if self._gate_blocked_since is None:
                self._gate_blocked_since = now
            elif now - self._gate_blocked_since >= self._gate_stuck_s:
                self._enter_recovery(f'scan-blocked {self._gate_stuck_s:.0f} s')
                return self._gate_recovery(v_wanted, v_gated)
        else:
            self._gate_blocked_since = None

        # trigger 2: physical stall, commanded forward but not moving. Catches
        # obstacles the lidar can't see (couch overhangs). Signs: odom ~0
        # (motors stalled) or a static scan (wheels slipping, so odom lies);
        # the scan check abstains without informative rays, as open space
        # looks static too.
        odom_v = abs(self._odom.twist.twist.linear.x)
        front = self._ranges_m[np.r_[0:90, 270:360]]
        self._scan_hist.append((now, front.copy()))
        scan_static = False
        scan_disp = float('nan')
        for t_old, old in self._scan_hist:
            if now - t_old >= 1.2:
                informative = (front < 3.0) & (old < 3.0)
                if informative.sum() >= 15:
                    # median: a robot wiggling against a thin obstacle moves
                    # a few far bins a lot — the bulk staying put is the tell
                    scan_disp = float(np.median(
                        np.abs(front[informative] - old[informative])))
                    scan_static = scan_disp < 0.02
                break
        # at crawl speeds 2 cm/1.2 s can't be told from real slow motion, so
        # the scan-static path needs a commanded speed the scan could show
        stalled_now = (self._last_cmd_v > 0.05 and odom_v < 0.015) \
            or (self._last_cmd_v >= 0.10 and scan_static)
        if stalled_now:
            if self._stall_since is None:
                self._stall_since = now
            elif now - self._stall_since >= self._gate_stall_s:
                self._enter_recovery(
                    f'physical stall (cmd {self._last_cmd_v:.2f} m/s, '
                    f'odom {odom_v:.3f} m/s, scan_static={scan_static}, '
                    f'scan_disp {scan_disp:.3f} m)')
                return self._gate_recovery(v_wanted, v_gated)
        else:
            self._stall_since = None

        # trigger 3: no progress, for when the policy goes timid and the
        # command-based triggers stay silent. Backs up generously.
        op = self._odom.pose.pose.position
        self._pose_hist.append((now, op.x, op.y))
        if self.behavior == 'seek' \
                and (self._escaping()
                     or self._seek_mode not in ('APPROACH', 'FACE_SWEEP',
                                                'WANDER')):
            # only for modes with no bounded slowness of their own: arc dwells,
            # frontier crawls and escape transits are deliberate. Triggers 1-2
            # stay armed in every mode.
            return None
        for t_old, ox, oy in self._pose_hist:
            if now - t_old >= self._progress_window:
                if math.hypot(op.x - ox, op.y - oy) < self._progress_radius:
                    self._enter_recovery(
                        f'no progress ({self._progress_radius:.2f} m circle '
                        f'for {self._progress_window:.0f} s)',
                        backup_s=self._gate_backup_s * self._progress_mult)
                    return self._gate_recovery(v_wanted, v_gated)
                break
        return None


    def _stall_watchdog(self):
        while True:
            time.sleep(0.1)
            if self._done or self._last_tick_wall is None:
                continue
            # monotonic: WSL2's wall clock steps backwards ~1 s every ~32 s
            age = time.monotonic() - self._last_tick_wall
            if age > max(3.0 * self._tick_period, 3.0):
                # spam zeros so they win any race with a latched twist, and
                # drop the tracker so a resuming tick can't re-drive its path
                self._tracker = None
                self._cmd_pub.publish(Twist())
                if int(age) != getattr(self, '_stall_logged_s', -1):
                    self._stall_logged_s = int(age)
                    self.get_logger().error(
                        f'control ticks stalled for {age:.0f} s — '
                        'wheels zeroed')
                # one all-threads stack dump per burst, at the 4 s mark
                if age >= 4.0 and not getattr(self, '_stall_dumped',
                                              False):
                    self._stall_dumped = True
                    try:
                        import faulthandler
                        with open('/tmp/policy_stall_stacks.txt',
                                  'a') as fh:
                            fh.write(f'\n=== burst dump wall='
                                     f'{time.time():.1f} age='
                                     f'{age:.1f} ===\n')
                            faulthandler.dump_traceback(
                                file=fh, all_threads=True)
                    except Exception:
                        pass
            else:
                self._stall_logged_s = -1
                self._stall_dumped = False
            if age > 20.0:
                self.get_logger().fatal(
                    'stall persists — exiting so the wrapper stops the robot')
                os._exit(2)
