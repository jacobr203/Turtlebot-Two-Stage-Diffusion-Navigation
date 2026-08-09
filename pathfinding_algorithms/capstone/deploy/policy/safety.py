"""The last check between the policy's output and the wheels.

A classical, non-learned layer: it scales speed down as the forward
cone closes, refuses motion that would hit something, and runs the
recovery behaviours (back up, rotate to a clear bearing, escape) when
the robot is wedged or has stopped making progress. Also holds the
watchdog that notices the control loop has gone quiet.

Kept deliberately simple and separate from the policy so the evaluation
harness can switch it off and measure the bare learned controller.

Mixin composed into DiffusionPolicyNode; `self` is the node.
"""

import math
import os
import time

import numpy as np

from geometry_msgs.msg import Twist

from common.geometry import yaw_from_quaternion


class SafetyMixin:
    """Safety gate, recovery, and the stall watchdog."""


    # ── safety gate: classical scan check between policy and wheels ──────────
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
        """Scale forward speed by clearance in the forward cone; 0 blocks
        translation but leaves rotation free (turning away stays possible).
        The cone LEANS INTO the commanded turn (2026-07-23): the fixed
        forward cone missed the side clips that caused every collision —
        while turning, the swept side is where contact actually happens.
        Context margins (2026-07-24): the seek's 0.21 m transit stop walls
        off every ~0.35 m gap — pockets became unreachable and faces
        churned. Inside the final 1.5 m to the current goal the stop drops
        to the benchmark-proven 0.16 (speed there is already creep-capped)."""
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
        # ±35° rear sector (was ±15°): collision forensics 2026-07-23 —
        # the robot reversed into a pillar sitting in its rear-QUARTER
        # while the narrow cone read clear. The wider cone both aborts
        # backups earlier and makes recovery prefer creep in tight spots.
        r = self._fused_gate_ranges()
        return r[145:216].min()


    def _spin_safe(self):
        """True when an in-place rotation cannot clip anything: the hull
        corners sweep ~0.14 m, so any 360° return below 0.17 m means a
        rotate-in-place will contact. Root-cause guard (2026-07-23): the
        wedge collisions were rotations next to pillars, not drive-ins.
        In tight-approach mode the threshold relaxes to 0.15 (still above
        the 0.14 sweep) so pocket-mouth facing turns aren't vetoed."""
        r = self._fused_gate_ranges()
        tight = getattr(self, '_tight_ok', False) or self._escaping()
        thr = 0.15 if tight else 0.17
        return r is None or float(np.min(r)) > thr


    def _escaping(self):
        """Pinch-escape active: gate margins relax to the benchmark values
        the robot entered the pocket under (2026-08-03, world 768). Ends on
        1.2 m displacement from the pinch or the 30 s leash."""
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
                # recovery-loop breaker (2026-07-23: 16 recoveries in one
                # run — the robot ping-ponged into the same pinch because
                # nothing ever cancelled the goal behind it). Explore
                # faces get abandoned; target approaches shift bearing.
                # 2026-08-03 (world 768): three strikes must be CONSUMED —
                # the old counter never reset, so every later recovery
                # abandoned a fresh face instantly (9 in a row, none ever
                # visited). And abandoning alone doesn't free a robot the
                # transit gate has walled into a pocket: enter escape mode
                # (benchmark gate margins until 1.2 m from here) and
                # blacklist the pocket's region so the next goal leaves it.
                self._recovery_spots.clear()
                self._escape_until = now + 30.0
                self._escape_from = (p[0], p[1])
                mp = self._map_pose(p)
                if mp is not None:
                    b = (int(math.floor(mp[0] / 2.0)),
                         int(math.floor(mp[1] / 2.0)))
                    self._region_block[b] = now + 120.0
                # color branch FIRST (2026-08-03 run 4, +175 s): a stale
                # _face_target from before the sighting shadowed the
                # color branch — the breaker "abandoned" a face the robot
                # wasn't even driving to and blacklisted a region near
                # the live target instead of switching approach side.
                if self._seek_color is None and self._face_target is not None:
                    t = self._face_target
                    fb = (int(math.floor(t['map_xy'][0] / 2.0)),
                          int(math.floor(t['map_xy'][1] / 2.0)))
                    self._region_block[fb] = now + 120.0
                    # cooled, NOT retired (2026-08-03, world 768): the
                    # breaker abandons UNINSPECTED faces; retiring them
                    # forever lied to coverage — outlives the 120 s
                    # region block so the region reopens before the face
                    self._face_cool[t['key']] = now + 180.0
                    if t.get('map_xy'):
                        self._cool_spots = getattr(self, '_cool_spots', [])
                        self._cool_spots.append(
                            (tuple(t['map_xy']), now + 180.0))
                    self._face_target = None
                    # escape OWNS its exit goal here too (the
                    # 240 ms escape death was THIS path — goal=None let
                    # the next tick's pick seize it with a doomed face)
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
                    # recoveries shift the approach side for VARIETY but
                    # never burn the give-up budget — that is the job of
                    # the progress-based patience timer (2026-07-24)
                    self._seek_attempts = min(self._seek_attempts + 1, 7)
                    self._seek_repositioning = False
                    self.get_logger().warning(
                        'recovery loop at a pinch — next routable '
                        'approach side')
        # DETERMINISTIC alternation (2026-08-04 random turns undid each other —
        # left then right, the robot staring at the same pillar from the same
        # angle 16 s apart). Consecutive recoveries now sweep opposite sides by
        # design.
        self._recover_dir = -getattr(self, '_recover_dir', -1.0)
        # close color approaches keep recoveries SHORT (2026-08-03 run 5):
        # each 4.5 s backup + random turn undid the approach progress it
        # followed — 200 s of net-zero circling at a 0.99 m sighting
        # until the patience timer phantomed a REAL square. The wedge
        # escalation below still outranks this cap.
        if self.behavior == 'seek' and backup_s is None \
                and self._seek_color is not None:
            backup_s = min(self._gate_backup_s, 2.5)
        # wedge escalation (2026-08-03, world 768 run 3): 18 identical
        # 1.8 s backups never freed a chassis wedged between two
        # cylinders (odom matched cmd, scan static, zero displacement
        # for 3+ min — wheels spinning). Consecutive stall recoveries
        # without 0.15 m displacement rock harder: double-length backup,
        # deterministically ALTERNATING turn side (random re-rolled the
        # same side back-to-back). Past 4, say so loudly — a dead run
        # must read as dead in the log, not as exploration churn.
        # reason-agnostic since run 6: contact grinds re-enter via
        # 'post-collision backup', not 'physical stall', and were never
        # escalating (3 contacts on one cylinder before a lucky exit)
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
            # rear blocked AND too tight to spin: creep along the freer
            # axis — but ONLY if that axis actually has room. Creep
            # bypasses the gate's zero-speed floor by design, and at
            # 0.12-0.15 m it drove into pillars (both agents' top
            # finding, 2026-08-04: 3 collisions). Below 0.18 m on both
            # axes, slow-rotate the hull instead of translating.
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
            # wedged so tight a spin would clip: translate along the freer
            # axis until rotation is safe (root-cause fix 2026-07-23).
            # testing (BOTH cross-cycle collisions were creep/
            # turn primitives below 0.25 m): re-check clearance along
            # the creep direction DURING the motion, not just at entry —
            # the axis that had room can close mid-creep.
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

        # trigger 2: physical stall — commanded forward but not actually
        # moving. Catches obstacles the lidar can't see (couch overhangs).
        # Two independent signs of "not moving":
        # odom velocity ~0 (motors stalled under load), OR
        # the scan is static (wheels slipping: odom LIES about motion —
        # the dwa_camera slip lesson, bug #014). Scan check abstains
        # without enough informative rays (open space looks static too).
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
        # crawl calibration (28-fire storm at cmd
        # 0.06-0.20 m/s destroyed both approaches; at gate-crawl speeds
        # the 2 cm/1.2 s threshold is indistinguishable from legitimate
        # slow motion). The scan-static path needs a commanded speed the
        # scan could actually SHOW: >= 0.10 m/s. The odom_v path (motors
        # stalled under load) keeps the old floor.
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

        # trigger 3: no progress — hasn't left a small circle in
        # progress_window_s while it still has somewhere to be. Catches
        # stuck states where the POLICY goes timid (near-zero commands, so
        # the command-based triggers above stay silent). Backs up generously.
        op = self._odom.pose.pose.position
        self._pose_hist.append((now, op.x, op.y))
        if self.behavior == 'seek' \
                and (self._escaping()
                     or self._seek_mode not in ('APPROACH', 'FACE_SWEEP',
                                                'WANDER')):
            # ARBITER gating (2026-08-04, notes/seek_arbiter.md): the
            # no-progress metronome fires only in modes with no bounded
            # slowness of their own. ARC dwells, FRONTIER crawls and
            # ESCAPE transits are deliberate — every historical loop was
            # this trigger (or a sibling guard) undoing one of them.
            # Triggers 1-2 (hard block, physical stall) stay armed in
            # every mode.
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
            # monotonic: this box's wall
            # clock steps backwards ~1 s/32 s (WSL2) — wall stamps can
            # spuriously age or refresh the liveness check
            age = time.monotonic() - self._last_tick_wall
            if age > max(3.0 * self._tick_period, 3.0):
                # 10 Hz zeroing + tracker clear: during the
                # testing storm the camera showed a LATCHED twist still
                # executing — and colliding — through seven consecutive
                # 1 Hz "wheels zeroed" publishes. Spam zeros so they win
                # any race, and drop the tracker so a resuming tick
                # cannot re-drive the stale path it froze on.
                self._tracker = None
                self._cmd_pub.publish(Twist())
                if int(age) != getattr(self, '_stall_logged_s', -1):
                    self._stall_logged_s = int(age)
                    self.get_logger().error(
                        f'control ticks stalled for {age:.0f} s — '
                        'wheels zeroed')
                # burst stack capture two candidate mechanisms with different
                # fixes — unyielding picker GIL phases vs a planner lock — one
                # stack trace settles it; do not fix blind). One all-threads
                # dump per burst, at the 4 s mark.
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
