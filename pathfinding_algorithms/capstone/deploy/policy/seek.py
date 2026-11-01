"""Colour-search behaviour: target bookkeeping plus one mode per control tick
(APPROACH, ARC, FRONTIER, FACE_SWEEP, WANDER), with RECOVER and ESCAPE overlaid.
Effort per target is bounded: a resisting target is shelved on a cooldown.
"""

from collections import deque
import math
import time

import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

from common.geometry import robot_to_world, world_to_robot


class SeekMixin:
    """Decides what the robot is hunting and how it approaches it."""

    # The goal sits almost on the square: the policy is trained to stop at its
    # goal, so aiming at a standoff makes it halt outside the checkpoint radius.
    SEEK_NEAR = 0.15           # [m] goal offset in front of the square
    SEEK_FACE_D = 0.60         # [m] range at which facing starts
    SEEK_FACE_TOL = 0.15       # [rad] facing dead-band
    SEEK_FACE_W = 0.7          # [rad/s] facing rotation speed
    SEEK_HOLD_TIMEOUT = 8.0    # [s] facing hold before re-approach

    SEEK_VIEW_DIST = 0.9       # [m] face-inspection standoff
    SEEK_COVER_RANGE = 3.0     # [m] max range that earns credit
    SEEK_COVER_HALF = 0.50     # [rad] half-FOV credited
    SEEK_R_FULL = 1.5          # [m] full proximity credit inside this

    TRIP_ARRIVED_M = 0.35      # [m] close enough to count as arrived
    TRIP_PROGRESS_M = 0.15     # [m] improvement that counts as progress
    TRIP_TIMEOUT_S = 45.0      # [s] give up and blame the route
    TRIP_FAIL_COOL_S = 120.0   # [s] cooldown on a standoff that failed
    TRIP_MIN_WORTH_M = 0.4     # [m] closer than this is not worth a trip

    SEEK_QMIN = 0.45 # quality >= this counts as inspected

    _DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1)) # map-frame face normals


    def _publish_seek_layers(self):
        """RViz layers: /seek/coverage (uncovered faces = 100, covered = 0
        over unknown) and /seek/swatches (SEEN = sphere, FOUND = cube)."""
        m = self._map
        if m is None or self._done:
            return
        fpos, dix = self._all_faces(m)
        H, W = m.info.height, m.info.width
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        data = np.full(H * W, -1, dtype=np.int8)
        if len(fpos):
            # log-scaled urgency heatmap: 100 = never seen, fading to 0 as
            # observation quality reaches SEEK_QMIN (or the face is given up)
            vals = np.empty(len(fpos), dtype=np.int8)
            for n, ((fx, fy), d) in enumerate(zip(fpos, dix)):
                if self._face_done_at(fx, fy, d):
                    vals[n] = 0
                else:
                    qf = min(max(self._face_q_at(fx, fy, d), 0.01),
                             self.SEEK_QMIN)
                    urg = math.log(qf / self.SEEK_QMIN) \
                        / math.log(0.01 / self.SEEK_QMIN)
                    vals[n] = max(1, int(round(100 * urg)))
            xs = np.clip(((fpos[:, 0] - ox) / res).astype(int), 0, W - 1)
            ys = np.clip(((fpos[:, 1] - oy) / res).astype(int), 0, H - 1)
            data[ys * W + xs] = vals
        out = OccupancyGrid()
        out.header = m.header
        out.info = m.info
        out.data = data.tolist()
        self._cov_pub.publish(out)

        st = self._color_status
        if not st:
            return
        rgb = {'light_green': (0.56, 0.93, 0.56), 'purple': (0.5, 0.0, 0.5),
               'orange': (1.0, 0.5, 0.0), 'lime_green': (0.2, 0.8, 0.2),
               'sky_blue': (0.42, 0.65, 0.8)}
        ma = MarkerArray()
        for i, (color, state) in enumerate(st['states'].items()):
            est = st['estimates'].get(color)
            if state == 'UNSEEN' or not est:
                continue
            mk = Marker()
            mk.header.frame_id = 'odom'
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns, mk.id = 'swatches', i
            mk.type = Marker.CUBE if state == 'FOUND' else Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = est['x']
            mk.pose.position.y = est['y']
            mk.pose.position.z = 0.12
            mk.scale.x = mk.scale.y = mk.scale.z = \
                0.15 if state == 'FOUND' else 0.10
            r, g, b = rgb[color]
            mk.color.r, mk.color.g, mk.color.b = r, g, b
            mk.color.a = 1.0
            ma.markers.append(mk)
        # the current goal, as a yellow sphere
        if self.behavior == 'seek' and self.goal is not None:
            g2 = self._to_odom(np.array([list(self.goal)]))[0]
            mk = Marker()
            mk.header.frame_id = 'odom'
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns, mk.id = 'goal', 99
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = float(g2[0])
            mk.pose.position.y = float(g2[1])
            mk.pose.position.z = 0.05
            mk.scale.x = mk.scale.y = mk.scale.z = 0.18
            mk.color.r = 1.0
            mk.color.g = 1.0
            mk.color.b = 0.0
            mk.color.a = 0.9
            ma.markers.append(mk)
        self._swatch_pub.publish(ma)


    def _seek_targets(self):
        """SEEN-not-FOUND squares as (color, world_x, world_y), from the
        manager's camera-only odom-frame estimates."""
        st = self._color_status
        if not st:
            return []
        out = []
        for color, state in st['states'].items():
            est = st['estimates'].get(color)
            if state != 'SEEN' or not est:
                if state == 'FOUND':
                    # engagement over: clear the cap count and side latch so a
                    # re-SEEN colour or a rerun starts fresh
                    if getattr(self, '_cap_count', None):
                        self._cap_count.pop(color, None)
                    if getattr(self, '_side_flip', None):
                        self._side_flip.pop(color, None)
                continue
            # phantom: an estimate that survived many failed attempts without
            # a fresh sighting; any new detection (new stamp) revives it
            if (color, est.get('stamp')) in self._seek_phantom:
                continue
            w = robot_to_world(np.array([est['x'], est['y']]), self.anchor)
            out.append((color, float(w[0]), float(w[1])))
        return out


    def _seek_normal(self, tx, ty):
        """World-frame outward normal of the surface at the estimate (centroid
        of nearby occupied cells -> estimate), or None if the map can't say.
        Gives the right approach side even after an oblique first sighting."""
        m = self._map
        tf_mo = self._map_T_odom()
        if m is None or tf_mo is None:
            return None
        sx, sy, syaw = self.anchor
        est_map = robot_to_world(
            world_to_robot(np.array([tx, ty]), (sx, sy, syaw)), tf_mo)
        grid = np.asarray(m.data, dtype=np.int8).reshape(
            m.info.height, m.info.width)
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        ci = int((est_map[0] - ox) / res)
        cj = int((est_map[1] - oy) / res)
        r = int(0.25 / res)
        j0, j1 = max(0, cj - r), min(m.info.height, cj + r + 1)
        i0, i1 = max(0, ci - r), min(m.info.width, ci + r + 1)
        win = grid[j0:j1, i0:i1]
        ys, xs = np.nonzero(win > 50)
        if len(ys) < 2:
            return None
        cxm = ox + (i0 + xs.mean() + 0.5) * res
        cym = oy + (j0 + ys.mean() + 0.5) * res
        vx, vy = est_map[0] - cxm, est_map[1] - cym
        if math.hypot(vx, vy) < 1e-3:
            return None
        # direction: rotate map-frame angle into the world frame
        return math.atan2(vy, vx) - tf_mo[2] + syaw


    def _seek_tick(self, pose, ranges_m, now):
        """Seek goal management. True = tick fully handled (done, or the
        rotate-to-face override published the command)."""
        st = self._color_status
        if st and st.get('done'):
            if not self._done:
                self._done = True
                self._tracker = None
                self._cmd_pub.publish(Twist())
                self.get_logger().info('color test complete — stopping')
            return True
        # post-collision settle: an impact jerk between 5 Hz scans smears the
        # scan-match-only SLAM map, so hold still 1 s while it re-converges
        if st is not None:
            c = st.get('collisions', 0)
            if self._coll_seen is None:
                self._coll_seen = c
            elif c > self._coll_seen:
                self._coll_seen = c
                now_w = time.time()
                # same-wedge test is displacement, not time: a robot that has
                # not moved is still wedged
                sp = getattr(self, '_settle_pose', None)
                moved = sp is None or \
                    math.hypot(pose[0] - sp[0], pose[1] - sp[1]) >= 0.35
                if moved:
                    self._settle_until = now_w + 1.0
                    self._settle_pose = (pose[0], pose[1])
                    self.get_logger().warning(
                        'collision — settling 1 s so SLAM re-converges')
                else:
                    # same contact episode: settling again only re-registers
                    # contact, so force the backup and drop the goal behind it
                    self._settle_until = 0.0
                    self._enter_recovery('post-collision backup')
                    if self._face_target is not None:
                        self._done_bins.add(self._face_target['key'])
                        self._face_target = None
                        self.goal = None
                    elif self._seek_color is not None:
                        self._seek_offset += math.radians(50.0)
                        self._seek_repositioning = False
                    self.get_logger().warning(
                        'collision persists — backing off and dropping the '
                        'goal behind the wedge')
        if time.time() < self._settle_until:
            self._tracker = None
            self._last_cmd_v = 0.0
            self._cmd_pub.publish(Twist())
            return True
        self._update_coverage(pose)
        targets = self._seek_targets()
        # a chosen vantage owns the goal until reached, timed out or found;
        # otherwise the next tick's ring geometry overwrites it. Arrival is
        # judged on measured distance, so a stalled trip reports as a failure.
        trip = getattr(self, '_vantage_trip', None)
        if trip is not None:
            still_hunting = any(t[0] == trip['color'] for t in targets)
            range_to_standoff = math.hypot(trip['goal'][0] - pose[0],
                                           trip['goal'][1] - pose[1])
            elapsed = time.time() - trip['set_t']
            if range_to_standoff < trip['best_range'] - self.TRIP_PROGRESS_M:
                trip['best_range'] = range_to_standoff

            arrived = range_to_standoff < self.TRIP_ARRIVED_M
            timed_out = elapsed > self.TRIP_TIMEOUT_S
            if not still_hunting or arrived or timed_out:
                if arrived:
                    self.get_logger().info(
                        f"{trip['color']}: reached scored vantage "
                        f'({elapsed:.0f} s) — resuming approach')
                elif timed_out:
                    standoff_map = self._world_to_map_xy(*trip['goal'])
                    robot_map = self._map_pose(pose)
                    corridor_recorded = False
                    if standoff_map is not None:
                        self._vantage_cooldowns[
                            (round(standoff_map[0], 1),
                             round(standoff_map[1], 1))] = \
                            time.time() + self.TRIP_FAIL_COOL_S
                        # blame the route, measured from where the robot
                        # stalled, so standoffs behind the same blockage fail
                        if robot_map is not None:
                            corridor_recorded = self._corridor_kill(
                                tuple(robot_map[:2]), standoff_map)
                    self.get_logger().warning(
                        f"{trip['color']}: visibility re-aim FAILED — "
                        f'{range_to_standoff:.2f} m from the standoff '
                        f'after {elapsed:.0f} s; vantage cooled'
                        f"{', approach corridor marked dead' if corridor_recorded else ''}"
                        ', falling back to the arc')
                self._vantage_trip = None
            else:
                self.goal = trip['goal']       # the trip owns the goal
                self._publish_subgoal(self.goal, now)
                return False
        # vantage bookkeeping for every colour with a fresh detection: the
        # robot has line of sight now, and queued targets need it later
        now_v = time.time()
        # instrumentation: log the first clear sightline to each estimate, to
        # tell a range-limited detector from a target with no sightline yet
        st_los = self._color_status or {}
        if self._map is not None and (st_los.get('estimates') or {}):
            mp_l = self._map_pose(pose)
            for c, e in (st_los.get('estimates') or {}).items():
                if not e or c in getattr(self, '_los_first', {}):
                    continue
                if mp_l is None:
                    break
                tm_l = self._world_to_map_xy(
                    *robot_to_world(np.array([e['x'], e['y']]),
                                    self.anchor))
                if tm_l is None:
                    break
                if self._los_clear(self._map, mp_l[0], mp_l[1],
                                   tm_l[0], tm_l[1]):
                    d = getattr(self, '_los_first', {})
                    d[c] = now_v
                    self._los_first = d
                    self.get_logger().info(
                        f'{c}: first clear sightline available '
                        f'(range {math.hypot(tm_l[0] - mp_l[0], tm_l[1] - mp_l[1]):.2f} m)')
        for c, e in ((self._color_status or {})
                     .get('estimates', {}) or {}).items():
            if e and now_v - e.get('stamp', 0.0) < 3.0:
                self._sight_vantages[c] = (pose[0], pose[1], e['stamp'])
                # one entry per distinct detection stamp; the estimate-quality
                # gate judges triangulation from the vantage spread
                h = self._vantage_hist.setdefault(c, deque(maxlen=6))
                if not h or h[-1][2] != e['stamp']:
                    h.append((pose[0], pose[1], e['stamp']))
        # arbiter: derive the one active mode from the same sorted target and
        # staleness predicate the behaviours use
        now_md = time.time()
        # only estimates detected within 30 s can be adopted; older ones stay
        # SEEN and revive on a fresh sighting. The engaged colour is exempt,
        # since ARC exists to re-sight stale estimates.
        if targets:
            ests_c3 = (self._color_status or {}).get('estimates', {})
            targets = [t for t in targets
                       if t[0] == self._seek_color
                       or now_md - ((ests_c3.get(t[0], {}) or {})
                                    .get('stamp', 0.0)) < 30.0]
        # a capped colour is barred from re-adoption for a while, so the tour
        # moves on before returning with a fresh angle
        if targets:
            cool_c = getattr(self, '_color_cool_until', {})
            targets = [t for t in targets
                       if cool_c.get(t[0], 0.0) <= now_md]
        if targets:
            targets.sort(key=lambda t: math.hypot(t[1] - pose[0],
                                                  t[2] - pose[1]))
            # sticky nearest: keep the active colour unless a rival is 0.5 m
            # nearer, as flapping restarts the per-target timers
            if self._seek_color is not None \
                    and targets[0][0] != self._seek_color:
                for i, t in enumerate(targets):
                    if t[0] == self._seek_color:
                        d_cur = math.hypot(t[1] - pose[0], t[2] - pose[1])
                        d_new = math.hypot(targets[0][1] - pose[0],
                                           targets[0][2] - pose[1])
                        if d_cur <= d_new + 0.5:
                            targets.insert(0, targets.pop(i))
                        break
            c0, tx0, ty0 = targets[0]
            est_md = ((self._color_status or {}).get('estimates', {})
                      .get(c0, {}) or {}).get('stamp', 0.0)
            # per-target wall-clock engagement cap, immune to the per-mode
            # timers; a new sighting restarts it
            bud = getattr(self, '_tgt_budget', None)
            if bud is None:
                bud = self._tgt_budget = {}
            bud[c0] = bud.get(c0, 0.0) + 0.26 # ~one 4 Hz tick
            # progress credit: an advancing engagement earns its budget back,
            # so the cap ends blind pursuits, not working ones
            ebd = getattr(self, '_eng_best', None)
            if ebd is None:
                ebd = self._eng_best = {}
            d_b0 = math.hypot(tx0 - pose[0], ty0 - pose[1])
            best_d, best_arc = ebd.get(c0, (float('inf'), -1))
            arc_i = getattr(self, '_arc_idx', 0) \
                if c0 == self._seek_color else -1
            if d_b0 < best_d - 0.3 or arc_i > best_arc:
                ebd[c0] = (min(best_d, d_b0), max(best_arc, arc_i))
                if bud[c0] > 30.0:
                    self.get_logger().info(
                        f'{c0}: engagement advancing — budget credited '
                        f'(was {bud[c0]:.0f} s)')
                bud[c0] = 0.0
                # credits reset the budget but not the cap count, a per-colour
                # futility ledger cleared only when the colour is FOUND
            if bud[c0] > 150.0:
                # escalating cooldown (60 s per cap, up to 240 s): per-cycle
                # guards alone remember nothing across cycles
                cc = getattr(self, '_cap_count', {})
                cc[c0] = cc.get(c0, 0) + 1
                self._cap_count = cc
                cool_s = min(60.0 * cc[c0], 240.0)
                self.get_logger().warning(
                    f'{c0}: 150 s engagement without progress — hard '
                    f'cap #{cc[c0]}, phantoming + {cool_s:.0f} s cooldown')
                self._seek_phantom.add((c0, est_md))
                del bud[c0]
                ebd.pop(c0, None)
                cool_c = getattr(self, '_color_cool_until', {})
                cool_c[c0] = now_md + cool_s
                self._color_cool_until = cool_c
                # latch the arc side flip; cleared only when the colour is FOUND
                sf = getattr(self, '_side_flip', {})
                sf[c0] = not sf.get(c0, False)
                self._side_flip = sf
                targets = [t for t in targets if t[0] != c0]
        if targets:
            c0, tx0, ty0 = targets[0]
            est_md = ((self._color_status or {}).get('estimates', {})
                      .get(c0, {}) or {}).get('stamp', 0.0)
            d0 = math.hypot(tx0 - pose[0], ty0 - pose[1])
            v0 = self._sight_vantages.get(c0)
            sr0 = d0 > 1.5 and self._est_single_ray(c0, tx0, ty0)
            self._arc_single_ray = sr0
            arc_now = v0 is not None and d0 > 0.9 \
                and now_md - est_md > (3.0 if sr0 else 6.0)
            self._set_mode('ARC' if arc_now else 'APPROACH',
                           f'target {c0}')
        elif (self._face_target or {}).get('key') == ('frontier',):
            self._set_mode('FRONTIER', 'push active')
        elif self._face_target is not None:
            self._set_mode('FACE_SWEEP', 'face trip')
        else:
            self._set_mode('WANDER', 'no faces/targets')
        # stagnation timeout for the two modes with no bounded slowness of
        # their own: FACE_SWEEP picks can churn and WANDER can orbit
        self._world_hist.append((now_md, pose[0], pose[1]))
        while self._world_hist and now_md - self._world_hist[0][0] > 50.0:
            self._world_hist.popleft()
        if self._seek_mode in ('FACE_SWEEP', 'WANDER'):
            ref = None
            for t_h, x_h, y_h in self._world_hist:
                if now_md - t_h >= 45.0:
                    ref = (x_h, y_h)
                else:
                    break
            if ref is not None and math.hypot(pose[0] - ref[0],
                                              pose[1] - ref[1]) < 0.5:
                self.get_logger().warning(
                    f'{self._seek_mode} stagnant (<0.5 m in 45 s) — '
                    f'{"cooling face, " if self._face_target else ""}'
                    'escaping toward open space')
                # cool what we abandon, or the same face is re-picked at once
                if self._face_target is not None:
                    self._face_cool[self._face_target['key']] = \
                        now_md + 90.0
                    if self._face_target.get('map_xy'):
                        self._cool_spots = getattr(self, '_cool_spots', [])
                        self._cool_spots.append(
                            (tuple(self._face_target['map_xy']),
                             now_md + 90.0))
                    self._register_trip_fail(
                        self._face_target.get('map_xy'))
                self._face_target = None
                # the escape owns a goal toward the most open scan direction;
                # the repick is suppressed while escaping with a goal set
                wm_e = np.min(np.stack([np.roll(ranges_m, k)
                                        for k in range(-15, 16)]), axis=0)
                b_e = int(np.argmax(wm_e))
                ang_e = pose[2] + math.radians(b_e)
                self.goal = (pose[0] + 1.5 * math.cos(ang_e),
                             pose[1] + 1.5 * math.sin(ang_e))
                self._explore_churn = 2 # frontier push after escape
                self._escape_until = now_md + 30.0
                self._escape_from = (pose[0], pose[1])
                self._world_hist.clear()
        # map-divergence alarm, independent of the mode
        if self._map is not None \
                and now_md - getattr(self, '_div_chk_t', 0.0) > 10.0:
            self._div_chk_t = now_md
            n_unk = int((np.asarray(self._map.data, dtype=np.int8)
                         == -1).sum())
            if n_unk > 9000:
                self.get_logger().error(
                    f'map divergence suspected: {n_unk} unknown cells — '
                    'run results untrustworthy from here',
                    throttle_duration_sec=30.0)
        if not targets:
            if self._seek_color is not None:
                # the target retired with the robot point-blank in its pocket:
                # park its arc progress and exit on relaxed margins
                saved_a = getattr(self, '_arc_saved', {})
                saved_a[self._seek_color] = (getattr(self, '_arc_idx', 0),
                                             time.time())
                self._arc_saved = saved_a
                self._seek_color = None
                self._escape_until = time.time() + 30.0
                self._escape_from = (pose[0], pose[1])
                # replace the retired target's goal with an exit to open space
                wm_r = np.min(np.stack([np.roll(ranges_m, k)
                                        for k in range(-15, 16)]), axis=0)
                ang_r = pose[2] + math.radians(int(np.argmax(wm_r)))
                self.goal = (pose[0] + 1.5 * math.cos(ang_r),
                             pose[1] + 1.5 * math.sin(ang_r))
                self.get_logger().info(
                    'target retired — escaping toward open space on '
                    'relaxed margins')
            # nothing sighted: visit obstacle faces the camera hasn't inspected,
            # with frontier wander when none are usable
            self._seek_face_since = None
            self._seek_offset = 0.0
            self._seek_v_cap = None
            if self._face_target is not None:
                t = self._face_target
                q_now = self._face_q_at(t['map_xy'][0], t['map_xy'][1],
                                        t['dir'])
                if self._face_done_at(t['map_xy'][0], t['map_xy'][1],
                                      t['dir']) \
                        or q_now >= 0.8 * self.SEEK_QMIN:
                    # inspected, or covered while driving here. Clear the goal
                    # too, or trip commitment drives it until the leash.
                    self._face_target = None
                    self.goal = None
                elif math.hypot(pose[0] - self.goal[0],
                                pose[1] - self.goal[1]) < 0.40:
                    return self._view_face(pose)
            now_s = now.nanoseconds * 1e-9
            if self._face_target is not None and self.goal is not None:
                # committed to a face trip: only a stalled trip is cut loose,
                # and its face is cooled, not retired, as it was never inspected
                now_w = time.time()
                d_trip = math.hypot(pose[0] - self.goal[0],
                                    pose[1] - self.goal[1])
                if d_trip < self._trip_best_d - 0.3:
                    self._trip_best_d = d_trip
                    self._trip_progress_t = now_w
                stalled = now_w - self._trip_progress_t > 25.0
                if stalled or now_w >= self._trip_deadline:
                    self._face_cool[self._face_target['key']] = now_w + 90.0
                    if self._face_target.get('map_xy'):
                        self._cool_spots = getattr(self, '_cool_spots', [])
                        self._cool_spots.append(
                            (tuple(self._face_target['map_xy']),
                             now_w + 90.0))
                    self._register_trip_fail(
                        self._face_target.get('map_xy'))
                    self._face_target = None
                    self.goal = None
                    self._explore_churn += 1 # frontier-push trigger
                    self.get_logger().info(
                        f'face trip {"stalled" if stalled else "over leash"}'
                        f' ({d_trip:.1f} m short) — cooling face 90 s, next')
                    self._pick_seek_explore_goal(pose, ranges_m)
            elif (self.goal is None or now_s >= self._wander_next) \
                    and not (self._escaping() and self.goal is not None):
                # no repick may seize an escape's exit goal
                self._pick_seek_explore_goal(pose, ranges_m)
                if self._face_target is None:
                    # no faces yet: retry sooner than the wander leash, but not
                    # every tick or the heading flaps
                    self._wander_next = min(self._wander_next, now_s + 6.0)
            return False
        # already sorted by the arbiter; re-sorting would undo the stickiness
        color, tx, ty = targets[0]
        if color != self._seek_color: # new target — reset the offset fan
            prev_color = self._seek_color
            if self._seek_color is not None:
                # switching straight off a retired target: the robot is still
                # inside its pocket
                self._escape_until = time.time() + 30.0
                self._escape_from = (pose[0], pose[1])
                self.get_logger().info(
                    'target retired — escaping the pocket on relaxed '
                    'margins before the next approach')
            # the colour approach owns the goal; a leftover face trip would
            # go stale and shadow the breaker's colour branch
            self._face_target = None
            self._seek_color = color
            self._seek_attempts = 0
            self._seek_offset = 0.0
            self._seek_repositioning = False
            self._seek_best_d = float('inf')
            self._seek_progress_t = time.time()
            # reset per-target state. Vantage histories persist, and so does
            # arc progress per colour, or each glimpse after a brief drop
            # restarts the walk-around at stop 1.
            saved = getattr(self, '_arc_saved', {})
            if prev_color is not None:
                saved[prev_color] = (getattr(self, '_arc_idx', 0),
                                     time.time())
                self._arc_saved = saved
            st_a = saved.get(color)
            # the window must outlast the longest cap cooldown (240 s)
            if st_a is not None and time.time() - st_a[1] < 300.0:
                self._arc_idx = st_a[0]
                self.get_logger().info(
                    f'{color}: resuming arc at stop {st_a[0] + 1}'
                    ' (progress persisted across re-adoption)')
            else:
                self._arc_idx = 0
            self._arc_hard_t = None
            self._arc_arr_t = None
            self._arc_bear_ref = None # fresh orbit reference
            self._carrot_side = 0.0
        # approach along the map-derived surface normal; the robot->square ray
        # is a young-map fallback. The offset fans out on failed facing holds.
        n_ang = self._seek_normal(tx, ty)
        base = n_ang if n_ang is not None \
            else math.atan2(pose[1] - ty, pose[0] - tx)
        # routable approach selection: check candidate bearings against the
        # live A*, so failures jump to a verified-different side
        cand_deg = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
        if n_ang is not None:
            # with a known normal, only use bearings the camera can work from:
            # a routable pose behind the pillar can never re-sight the face
            cand_deg = (0.0, 45.0, -45.0, 80.0, -80.0)
        tf_mo = self._map_T_odom()
        d_est = math.hypot(tx - pose[0], ty - pose[1])
        # ring selection only in the 0.9-1.8 m band: farther out the estimate
        # is too rough to ring (drive toward it); nearer in, ring poses sit in
        # the map's inflation, so just face and confirm
        in_ring_band = (0.9 < d_est <= 1.8 and tf_mo is not None
                        and self.live_subgoaler.planners)
        rk = (color, self._seek_attempts)
        if in_ring_band and self._ring_key == rk \
                and time.time() - self._ring_scan_t < 1.5:
            pass # cached: keep the current verdict (a scan costs up to 0.35 s)
        elif in_ring_band:
            self._ring_key = rk
            self._ring_scan_t = time.time()
            sxa, sya, syawa = self.anchor
            rm = robot_to_world(world_to_robot(
                np.array(pose[:2]), (sxa, sya, syawa)), tf_mo)
            t0r = time.time()
            # attempts counts failed routable sides; unroutable ones are free
            routable_seen = 0
            chosen = None
            scanned_all = True
            for cd in cand_deg:
                a = base + math.radians(cd)
                cw = np.array([tx + 0.55 * math.cos(a),
                               ty + 0.55 * math.sin(a)])
                cm = robot_to_world(world_to_robot(cw, (sxa, sya, syawa)),
                                    tf_mo)
                if self.live_subgoaler.route_min_clearance(
                        tuple(rm), tuple(cm)) is not None:
                    if routable_seen == self._seek_attempts:
                        chosen = cd
                        break
                    routable_seen += 1
                if time.time() - t0r > 0.35:
                    scanned_all = False
                    break
            if chosen is not None:
                self._seek_offset = math.radians(chosen)
                self._ring_blocked = False
            elif scanned_all:
                # no ring pose is routable, but the checkpoint only needs the
                # camera within 0.69 m and tight margins can thread gaps the
                # planner refuses: demote to a direct approach
                self._ring_blocked = True
                self._seek_offset = 0.0
                self.get_logger().info(
                    f'{color}: no routable ring pose — direct approach',
                    throttle_duration_sec=10.0)
                # this proves the fixed ring useless, so try a scored viewpoint
                # now; throttled to one scan per 20 s per colour
                lastv = getattr(self, '_vp_scan_t', {})
                if time.time() - lastv.get(color, 0.0) > 20.0:
                    lastv[color] = time.time()
                    self._vp_scan_t = lastv
                    mp_r = self._map_pose(pose)
                    tm_r = self._world_to_map_xy(tx, ty)
                    if mp_r is not None and tm_r is not None:
                        vp_r = self._visible_viewpoint(mp_r, tm_r[0],
                                                       tm_r[1])
                        vw_r = self._map_to_world(vp_r) if vp_r else None
                        if vw_r is not None:
                            self.goal = vw_r
                            self.get_logger().info(
                                f'{color}: ring blocked — routing to a '
                                'scored viewpoint instead')
                            return False
            # timeout mid-scan: keep the current offset this tick
        else:
            self._ring_blocked = False
        ang = base + self._seek_offset
        # facing only helps from the face's side of the pillar. After a failed
        # hold, the reposition latch blocks re-facing until the robot reaches
        # the fanned pose, or the fan would never move it.
        robot_ang = math.atan2(pose[1] - ty, pose[0] - tx)
        on_face_side = n_ang is None or math.cos(robot_ang - n_ang) > 0.0
        d_now = math.hypot(tx - pose[0], ty - pose[1])
        # point-blank: the map normal under the robot's nose is junk and the
        # robot arrived via the face, so always face and hold
        if d_now <= 0.35:
            on_face_side = True
            self._seek_repositioning = False
        if on_face_side and not self._seek_repositioning and \
                d_now <= self.SEEK_FACE_D:
            if self._seek_face(pose, color, tx, ty):
                return True
            # fan stepped — recompute the offset pose and drive to it
            self._seek_repositioning = True
            ang = base + self._seek_offset
        # staged stand-off: from afar aim 0.55 m along the fanned normal,
        # outside the inflation so A* can plan to it; inside the final metre
        # tighten to SEEK_NEAR so the trained stop lands within the checkpoint
        d_sq = math.hypot(tx - pose[0], ty - pose[1])
        # progress-based patience: closing distance or a fresh detection counts;
        # only 120 s without either gives the target up
        now_pt = time.time()
        if d_sq < getattr(self, '_seek_best_d', float('inf')) - 0.2:
            self._seek_best_d = d_sq
            self._seek_progress_t = now_pt
        est_t = ((self._color_status or {}).get('estimates', {})
                 .get(color, {}) or {}).get('stamp', 0.0)
        if now_pt - est_t < 3.0:
            self._seek_progress_t = now_pt
        if now_pt - getattr(self, '_seek_progress_t', now_pt) > 120.0:
            rk = (color, est_t)
            if self._sight_vantages.get(color) is not None \
                    and rk not in self._resight_stamps:
                # one bonus 60 s window, if a vantage exists to work from
                self._resight_stamps.add(rk)
                self._seek_progress_t = now_pt - 60.0
                self._seek_best_d = float('inf')
                self.get_logger().warning(
                    f'{color}: no progress — 60 s arc re-sight window '
                    f'before phantoming')
            else:
                self._seek_phantom.add((color, est_t))
                self._seek_attempts = 0
                self._seek_offset = 0.0
                self._seek_best_d = float('inf')
                self.get_logger().warning(
                    f'{color}: 120 s without progress or fresh sighting — '
                    f'marking phantom, resuming exploration')
                return False
        if getattr(self, '_ring_blocked', False):
            # unroutable ring: aim at the estimate itself — snap + carrot
            # + tight margins carry the final stretch
            self.goal = (tx, ty)
        elif d_sq > 1.8:
            # far target: aim at the estimate and let A*'s free-cell snap find
            # the reachable spot beside it; a ring pose off a rough estimate
            # often lands inside obstacles
            self.goal = (tx, ty)
        else:
            r_off = self.SEEK_NEAR if d_sq < 1.0 \
                else max(self.SEEK_FACE_D - 0.05, 0.35)
            self.goal = (tx + r_off * math.cos(ang),
                         ty + r_off * math.sin(ang))
        # ARC re-sight: a stale estimate mid-approach means the camera can't
        # see the face from here. Sweep stops around the estimate from the
        # sighting vantage outward, aiming the camera at each; a fresh
        # detection resumes the approach. ARC entry is the arbiter's call.
        v = self._sight_vantages.get(color)
        single_ray = getattr(self, '_arc_single_ray', False)
        if self._seek_mode == 'ARC':
            # (bearing offset, ring radius) stops. Radius-shrink stops walk in
            # along the sighting ray, as a pocket face is only visible through
            # its mouth. 0.6 m is inside confirm range (0.69).
            offs = ((0.0, 1.6), (0.5, 1.6), (-0.5, 1.6), (0.0, 1.15),
                    (1.0, 1.6), (-1.0, 1.6), (0.0, 0.9), (0.0, 0.6))
            va = math.atan2(v[1] - ty, v[0] - tx)
            # side switch after a hard cap, as sieges end when the robot reaches
            # the open side: a latched per-colour flag toggled at cap fire
            if getattr(self, '_side_flip', {}).get(color, False):
                va += math.pi
            # advance on arrival (4 s dwell) with a 20 s travel budget: a fixed
            # timer skips exactly the deep stops that take longest to reach
            idx = getattr(self, '_arc_idx', 0)
            if getattr(self, '_arc_hard_t', None) is None:
                self._arc_hard_t = now_pt + 20.0
            a_off, r_arc = offs[idx % len(offs)]
            stop = (tx + r_arc * math.cos(va + a_off),
                    ty + r_arc * math.sin(va + a_off))
            # orbit gate: budget expiry advances only if the robot actually
            # orbited >=15 deg, so escalation follows geometry, not a timer
            bear_now = math.atan2(pose[1] - ty, pose[0] - tx)
            b_ref = getattr(self, '_arc_bear_ref', None)
            if b_ref is None:
                b_ref = self._arc_bear_ref = bear_now
            d_orb = abs(math.atan2(math.sin(bear_now - b_ref),
                                   math.cos(bear_now - b_ref)))
            adv = now_pt >= self._arc_hard_t and d_orb >= math.radians(15.0)
            if now_pt >= self._arc_hard_t and not adv:
                self.get_logger().info(
                    f'{color}: arc stop held — only '
                    f'{math.degrees(d_orb):.0f}° of orbit since last '
                    'advance (no physical progress)',
                    throttle_duration_sec=15.0)
                self._arc_hard_t = now_pt + 20.0
                # a held stop means the ring walk buys no sightlines, often
                # because it is centred on a wrong estimate: ask which routable
                # standoff can see the estimate. One query per held stop.
                mp_v = self._map_pose(pose)
                if mp_v is not None:
                    tm = self._world_to_map_xy(tx, ty)
                    vp = self._visible_viewpoint(mp_v, tm[0], tm[1]) \
                        if tm is not None else None
                    if vp is not None:
                        vw = self._map_to_world(vp)
                        if vw is not None and math.hypot(
                                vw[0] - pose[0],
                                vw[1] - pose[1]) > self.TRIP_MIN_WORTH_M:
                            # commit: the next tick's geometry would otherwise
                            # recompute the goal and the robot would never go
                            range_to_standoff = math.hypot(
                                vw[0] - pose[0], vw[1] - pose[1])
                            self._vantage_trip = {
                                'goal': vw, 'color': color,
                                'set_t': now_pt,
                                'best_range': range_to_standoff}
                            self.goal = vw
                            self._arc_bear_ref = bear_now
                            self.get_logger().info(
                                f'{color}: visibility re-aim — driving to '
                                f'standoff map({vp[0]:.1f},{vp[1]:.1f}) '
                                f'(LOS clear, {range_to_standoff:.1f}'
                                ' m away)')
                            return False
            if math.hypot(pose[0] - stop[0], pose[1] - stop[1]) < 0.45:
                if getattr(self, '_arc_arr_t', None) is None:
                    self._arc_arr_t = now_pt
                elif now_pt - self._arc_arr_t > 4.0:
                    adv = True
            if adv:
                idx = (idx + 1) % len(offs)
                self._arc_idx = idx
                self._arc_arr_t = None
                self._arc_hard_t = now_pt + 20.0
                self._arc_bear_ref = bear_now # orbit measured per-stop
                a_off, r_arc = offs[idx]
                stop = (tx + r_arc * math.cos(va + a_off),
                        ty + r_arc * math.sin(va + a_off))
            self.goal = stop
            why = 'estimate stale (single-ray, 3 s trigger)' \
                if single_ray else 'estimate stale'
            self.get_logger().info(
                f'{color}: {why} — arc re-sight stop '
                f'{getattr(self, "_arc_idx", 0) + 1}/{len(offs)}',
                throttle_duration_sec=8.0)
            # at the stop: AIM the camera at the estimate (the tracker
            # leaves the robot pointing along its final path segment)
            if math.hypot(pose[0] - self.goal[0],
                          pose[1] - self.goal[1]) < 0.45:
                bear = math.atan2(ty - pose[1], tx - pose[0]) - pose[2]
                bear = math.atan2(math.sin(bear), math.cos(bear))
                # spin-safe gate: rotation is gate-exempt, and the hull sweep
                # contacts what the translation gate never sees
                if abs(bear) > self.SEEK_FACE_TOL and self._spin_safe():
                    cmd = Twist()
                    cmd.angular.z = math.copysign(self.SEEK_FACE_W, bear)
                    self._tracker = None
                    self._last_cmd_v = 0.0
                    self._cmd_pub.publish(cmd)
                    return True
        # creep the last stretch: full speed overshoots the facing pose, and
        # 0.10 m/s keeps detections streaming for the estimate fusion
        self._seek_v_cap = 0.10 if d_sq < 1.2 else None
        if self._seek_repositioning and \
                math.hypot(pose[0] - self.goal[0],
                           pose[1] - self.goal[1]) < 0.25:
            self._seek_repositioning = False
        self._seek_face_since = None
        return False


    def _set_mode(self, mode, reason=''):
        """Arbiter transition point — the only writer of _seek_mode."""
        if mode != self._seek_mode:
            self.get_logger().info(
                f'mode {self._seek_mode} -> {mode}'
                f'{" (" + reason + ")" if reason else ""}')
            if mode in ('FACE_SWEEP', 'WANDER'):
                # fresh stagnation window on entry
                self._world_hist.clear()
            # likewise the no-progress evidence, or stillness accrued in the
            # previous mode fires it
            self._pose_hist.clear()
            self._seek_mode = mode


    def _est_single_ray(self, color, tx, ty):
        """True when every sighting came from one narrow bearing fan (<25 deg)
        and none within 1.5 m: a lone far ray fixes range poorly."""
        h = self._vantage_hist.get(color)
        if not h:
            return False # nothing to judge
        bearings = []
        for vx, vy, _ in h:
            if math.hypot(tx - vx, ty - vy) < 1.5:
                return False
            bearings.append(math.atan2(ty - vy, tx - vx))
        if len(bearings) < 2:
            return True
        spread = 0.0
        for i in range(len(bearings)):
            for j in range(i + 1, len(bearings)):
                d = bearings[i] - bearings[j]
                d = abs(math.atan2(math.sin(d), math.cos(d)))
                spread = max(spread, d)
        return spread < math.radians(25.0)


    def _seek_face(self, pose, color, tx, ty):
        """At the approach pose: stop, rotate to face the square and hold for
        the manager's confirmation. On timeout, try the next approach side."""
        bearing = math.atan2(ty - pose[1], tx - pose[0]) - pose[2]
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        cmd = Twist()
        if abs(bearing) > self.SEEK_FACE_TOL and not self._spin_safe():
            # rotating to face would clip — ease off the pillar first
            cmd.linear.x = -0.05
            self._tracker = None
            self._last_cmd_v = cmd.linear.x
            self._cmd_pub.publish(cmd)
            return True
        if abs(bearing) > self.SEEK_FACE_TOL:
            cmd.angular.z = math.copysign(self.SEEK_FACE_W, bearing)
            self._seek_face_since = None
        elif self._seek_face_since is None:
            self._seek_face_since = time.time()
            self.get_logger().info(
                f'facing {color} — holding for checkpoint confirmation')
        elif time.time() - self._seek_face_since > self.SEEK_HOLD_TIMEOUT:
            # advance past the failed side; the routable selector picks the next
            self._seek_attempts += 1
            if self._seek_attempts >= 8:
                # all sides exhausted: the estimate is phantom, so give it up;
                # a fresh detection stamp revives it
                st = self._color_status or {}
                est = (st.get('estimates') or {}).get(color) or {}
                self._seek_phantom.add((color, est.get('stamp')))
                self._seek_attempts = 0
                self._seek_offset = 0.0
                self._seek_face_since = None
                self.get_logger().warning(
                    f'{color} estimate gave no confirmation from any side '
                    f'— marking phantom, resuming exploration')
                return False
            self._seek_face_since = None
            self.get_logger().warning(
                f'{color} not confirmed after facing hold — next routable '
                f'approach side')
            return False # next tick plans to the shifted approach pose
        self._tracker = None
        self._last_cmd_v = 0.0
        self._cmd_pub.publish(cmd)
        return True


    def _seek_subgoal(self, pose):
        """World-frame subgoal: live-map A* toward self.goal, straight-line
        carrot fallback while the map/TF is still warming up."""
        if self.goal is None:
            # a retire path cleared the goal this tick: hold one tick (zero
            # offset), the next tick repicks
            return (pose[0], pose[1])
        # a nested goal would broadcast into garbage in the frame transforms
        if np.ndim(self.goal) != 1:
            self.get_logger().error(
                f'malformed seek goal {self.goal!r} — flattening')
            g = np.ravel(np.asarray(self.goal, dtype=np.float64))
            self.goal = (float(g[0]), float(g[1]))
        tf_mo = self._map_T_odom()
        sub_w = None
        if tf_mo is not None and self.live_subgoaler is not None:
            sx, sy, syaw = self.anchor
            robot_map = robot_to_world(
                world_to_robot(np.array(pose[:2]), (sx, sy, syaw)), tf_mo)
            goal_map = robot_to_world(
                world_to_robot(np.array(self.goal), (sx, sy, syaw)), tf_mo)
            try:
                sub_map = self.live_subgoaler.subgoal(tuple(robot_map),
                                                      tuple(goal_map))
            except TypeError:
                # survive on the carrot fallback, logging the full live state
                self.get_logger().error(
                    'subgoal TypeError — goal=%r (nd %s) goal_map=%r '
                    'robot_map=%r tf_mo=%r anchor=%r planner_x0s=%r' % (
                        self.goal, np.ndim(self.goal), goal_map, robot_map,
                        tf_mo, self.anchor,
                        [getattr(p, 'x0', None)
                         for p in self.live_subgoaler.planners]))
                sub_map = None
            if sub_map is not None:
                sub_w = robot_to_world(
                    world_to_robot(np.array(sub_map), tf_mo), (sx, sy, syaw))
                sub_w = (float(sub_w[0]), float(sub_w[1]))
        # degenerate plan guard: toward an unreachable goal A* can return the
        # robot's own cell, and a zero-offset subgoal makes the stop-at-goal
        # policy hold forever. Use the straight-line carrot instead.
        if sub_w is not None and \
                math.hypot(sub_w[0] - pose[0], sub_w[1] - pose[1]) < 0.45 and \
                math.hypot(self.goal[0] - pose[0],
                           self.goal[1] - pose[1]) > 0.5:
            sub_w = None
        if sub_w is None:
            d = math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1])
            f = min(1.0, self.get_parameter('subgoal_dist').value / max(d, 1e-6))
            sub_w = (pose[0] + f * (self.goal[0] - pose[0]),
                     pose[1] + f * (self.goal[1] - pose[1]))
            # carrot side hysteresis: a carrot ray blocked by a pillar conditions
            # the policy left/right-symmetrically. Offset the conditioning
            # point to a remembered side; cleared when the ray is free.
            if self._ranges_m is not None and d > 0.6:
                bear = math.atan2(sub_w[1] - pose[1],
                                  sub_w[0] - pose[0]) - pose[2]
                bi = int(round(math.degrees(bear))) % 360
                sect = np.arange(bi - 8, bi + 9) % 360
                rr = self._ranges_m[sect]
                rr = rr[np.isfinite(rr)]
                if len(rr) and float(np.min(rr)) < min(1.2, d):
                    if getattr(self, '_carrot_side', 0.0) == 0.0:
                        tw_cs = self._odom.twist.twist
                        self._carrot_side = 1.0 \
                            if tw_cs.angular.z >= 0.0 else -1.0
                        self.get_logger().info(
                            f'carrot ray blocked — conditioning offset '
                            f'{"left" if self._carrot_side > 0 else "right"}',
                            throttle_duration_sec=5.0)
                    ux = (self.goal[0] - pose[0]) / max(d, 1e-6)
                    uy = (self.goal[1] - pose[1]) / max(d, 1e-6)
                    sub_w = (sub_w[0] - uy * 0.55 * self._carrot_side,
                             sub_w[1] + ux * 0.55 * self._carrot_side)
                else:
                    self._carrot_side = 0.0
        return sub_w
