"""Exploration when nothing is being chased: frontier goals, wander goals and a
systematic tour of obstacle faces the camera has not inspected, ranked by
expected information gain against travel cost.
"""

import math
import threading
import time

import numpy as np

from geometry_msgs.msg import Twist

from common.geometry import robot_to_world, world_to_robot


class ExploreMixin:
    """Exploration: wander goals, frontier push, face coverage."""


    def _map_free_goal(self, pose, v_now):
        """Best known-free cell to head for, as (world_x, world_y, at_frontier,
        distance_m), or None without a map, TF or acceptable cell."""
        m = self._map
        tf_mo = self._map_T_odom()
        if m is None or tf_mo is None:
            return None
        from scipy.ndimage import binary_dilation, distance_transform_edt

        H, W = m.info.height, m.info.width
        grid = np.asarray(m.data, dtype=np.int8).reshape(H, W)
        free = grid == 0
        occupied = grid > 50
        unknown = grid == -1
        res = m.info.resolution

        clear_m = distance_transform_edt(~occupied) * res
        p = self.get_parameter
        safe = free & (clear_m >= p('wander_goal_clearance').value)
        if not safe.any():
            return None
        frontier = safe & binary_dilation(unknown, iterations=2)

        # robot + candidate positions in the MAP frame
        sx, sy, syaw = self.anchor
        odom_xy = world_to_robot(np.array(pose[:2]), (sx, sy, syaw))
        robot_map = robot_to_world(odom_xy, tf_mo)
        ys, xs = np.nonzero(safe)
        cx = m.info.origin.position.x + (xs + 0.5) * res
        cy = m.info.origin.position.y + (ys + 0.5) * res
        d = np.hypot(cx - robot_map[0], cy - robot_map[1])

        # near-standstill starts need a far goal to stay in the cruise regime
        min_d = 3.0 if v_now < 0.05 else p('wander_min_goal_dist').value
        # the target may sit far beyond wander_goal_dist: the subgoal planner
        # turns it into in-distribution 1.2 m carrots
        ok = (d >= min_d) & (d <= max(self.wander_goal_dist, 12.0))
        if not ok.any():
            return None
        # frontier + novelty - travel cost - revisit count + heading momentum
        score = 1.5 * frontier[ys, xs] - 0.05 * d
        # visibility gain: frontier adjacency is only a proxy, so ray-cast the
        # top candidates and up-weight by the unknown area each can see
        try:
            n_vis = min(40, int(ok.sum()))
            if n_vis > 0:
                pre = np.where(ok, score, -1e3)
                sub = np.argpartition(pre, -n_vis)[-n_vis:]
                gains = self._visibility_gain(
                    grid, m.info, np.stack([cx[sub], cy[sub]], axis=1))
                gmax = float(gains.max())
                if gmax > 0:
                    score[sub] = score[sub] + 2.0 * (gains / gmax)
        except Exception as e: # never let scoring kill a tick
            self.get_logger().warning(
                f'visibility-gain scoring skipped: {e}',
                throttle_duration_sec=30.0)
        # novelty bias: up-weight candidates up to 2 m from visited cells
        if self._visited_cells:
            from scipy.spatial import cKDTree
            va = (np.array(sorted(self._visited_cells), dtype=np.float64)
                  + 0.5) * 0.5
            dv = cKDTree(va).query(np.stack([cx, cy], axis=1))[0]
            score = score + 0.8 * np.minimum(dv, 2.0) / 2.0
        # coarse visit-count penalty (1.5 m grid)
        vg = getattr(self, '_visit_grid', None)
        if vg:
            gx = np.floor(cx / 1.5).astype(int)
            gy = np.floor(cy / 1.5).astype(int)
            vc = np.array([vg.get((int(a), int(b)), 0)
                           for a, b in zip(gx, gy)], dtype=np.float64)
            score = score - 0.5 * np.minimum(vc, 8.0) / 8.0
        # heading persistence: reversals read as figure-eights
        if v_now > 0.05:
            hd_map = pose[2] - syaw + tf_mo[2]
            bear_c = np.arctan2(cy - robot_map[1], cx - robot_map[0])
            dh = bear_c - hd_map
            score = score + 0.35 * np.cos(np.arctan2(np.sin(dh),
                                                     np.cos(dh)))
        score[~ok] = -1e3
        # random among the top candidates: a deterministic argmax re-picks
        # the SAME goal after every recovery, steering the robot back into
        # the obstacle it just escaped
        k = min(8, int(ok.sum()))
        top = np.argpartition(score, -k)[-k:]
        i = int(np.random.choice(top))

        goal_odom = world_to_robot(np.array([cx[i], cy[i]]), tf_mo)
        goal_world = robot_to_world(goal_odom, (sx, sy, syaw))
        return (float(goal_world[0]), float(goal_world[1]),
                bool(frontier[ys[i], xs[i]]), float(d[i]))


    def _pick_wander_goal(self, pose, ranges_m):
        # wander-goal commitment: keep the active goal while it is being closed,
        # or fresh goals every few seconds confine the robot to a strip. Same
        # leash as face trips: alive while closing >=0.25 m per 20 s, time cap,
        # dropped on arrival (<0.5 m) or when another writer takes the goal.
        now_w = time.time()
        meta = self._wander_meta
        if (self.behavior == 'seek' and meta is not None
                and self.goal is not None and self.goal == meta['goal']):
            d_go = math.hypot(pose[0] - self.goal[0],
                              pose[1] - self.goal[1])
            if d_go < meta['best_d'] - 0.25:
                meta['best_d'] = d_go
                meta['progress_t'] = now_w
            if (d_go > 0.5 and now_w - meta['set_t']
                    < meta.get('cap_s', 60.0)
                    and now_w - meta['progress_t'] < 20.0):
                self.get_logger().info(
                    f'wander goal persists ({d_go:.1f} m to go)',
                    throttle_duration_sec=10.0)
                self._wander_next = self.get_clock().now().nanoseconds \
                    * 1e-9 + self.wander_interval
                return
        v_now = abs(self._odom.twist.twist.linear.x)
        picked = self._map_free_goal(pose, v_now)
        if picked is not None:
            gx, gy, at_frontier, d = picked
            self.goal = (gx, gy)
            self.get_logger().info(
                f'wander goal (map): {d:.1f} m, known-free'
                f'{", frontier" if at_frontier else ""}')
        else:
            # no usable map yet: carrot toward the most open scan direction
            # (±15° windowed minimum), far enough to stay in the cruise regime
            wm = np.min(np.stack([np.roll(ranges_m, k)
                                  for k in range(-15, 16)]), axis=0)
            b = int(np.argmax(wm))
            ang = pose[2] + math.radians(b)
            self.goal = (pose[0] + self.wander_goal_dist * math.cos(ang),
                         pose[1] + self.wander_goal_dist * math.sin(ang))
            self.get_logger().info(
                f'wander goal (scan fallback): {self.wander_goal_dist:.1f} m '
                f'at {b}° (clearance {wm[b]:.1f} m)')
        # the commitment cap scales with distance: a long leg at crawl pace is
        # legitimate while the progress leash holds
        d_leg = math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1])
        self._wander_meta = {'goal': self.goal, 'set_t': now_w,
                             'best_d': float('inf'), 'progress_t': now_w,
                             'cap_s': min(240.0, max(60.0, 25.0 * d_leg))}
        self._wander_next = self.get_clock().now().nanoseconds * 1e-9 \
            + self.wander_interval


    @staticmethod
    def _face_key(fx, fy, d):
        return (int(round(fx * 10)), int(round(fy * 10)), d)


    def _all_faces(self, m):
        """Obstacle faces of the current map: occupied cells with a free
        4-neighbor. Returns (pos[N,2] map-frame face centers, dir_idx[N])."""
        H, W = m.info.height, m.info.width
        grid = np.asarray(m.data, dtype=np.int8).reshape(H, W)
        occ = grid > 50
        free = grid == 0
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        pos, dix = [], []
        for d, (dx, dy) in enumerate(self._DIRS):
            nb = np.zeros_like(free)
            if dx == 1:
                nb[:, :-1] = free[:, 1:]
            elif dx == -1:
                nb[:, 1:] = free[:, :-1]
            elif dy == 1:
                nb[:-1, :] = free[1:, :]
            else:
                nb[1:, :] = free[:-1, :]
            ys, xs = np.nonzero(occ & nb)
            if len(ys) == 0:
                continue
            pos.append(np.stack([ox + (xs + 0.5 + 0.5 * dx) * res,
                                 oy + (ys + 0.5 + 0.5 * dy) * res], axis=1))
            dix.append(np.full(len(ys), d))
        if not pos:
            return np.empty((0, 2)), np.empty(0, dtype=int)
        return np.concatenate(pos), np.concatenate(dix)


    def _face_q_at(self, fx, fy, d):
        """Best recorded quality near this face; a 3x3 neighbourhood, so SLAM
        corrections shifting a wall a few cm don't orphan its score."""
        bx, by = int(round(fx * 10)), int(round(fy * 10))
        best = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                k = (bx + di, by + dj, d)
                if k in self._face_gaveup:
                    return 1.0
                q = self._face_q.get(k, 0.0)
                if q > best:
                    best = q
        return best


    def _face_done_at(self, fx, fy, d):
        bx, by = int(round(fx * 10)), int(round(fy * 10))
        for di in (-2, -1, 0, 1, 2):
            for dj in (-2, -1, 0, 1, 2):
                if (bx + di, by + dj, d) in self._done_bins:
                    return True
        return self._face_q_at(fx, fy, d) >= self.SEEK_QMIN


    def _update_coverage(self, pose):
        """Score faces in view (frontality x proximity), keeping each best."""
        m = self._map
        mp = self._map_pose(pose)
        if m is None or mp is None:
            return
        rx, ry, ryaw = mp
        fpos, dix = self._all_faces(m)
        if not len(fpos):
            return
        rel = fpos - (rx, ry)
        dist = np.hypot(rel[:, 0], rel[:, 1])
        near = (dist <= self.SEEK_COVER_RANGE) & (dist > 0.15)
        if not near.any():
            return
        rel, dist, fpos, dix = rel[near], dist[near], fpos[near], dix[near]
        normals = np.array(self._DIRS)[dix]
        frontality = (-rel * normals).sum(axis=1) / dist # cos(incidence)
        bearing = np.arctan2(rel[:, 1], rel[:, 0]) - ryaw
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        seen = (frontality > 0.10) & (np.abs(bearing) <= self.SEEK_COVER_HALF)
        # occlusion gate: no credit through walls, as a hidden face marked done
        # leaves the sweep forever
        if seen.any():
            Hv, Wv = m.info.height, m.info.width
            grid_v = np.asarray(m.data, dtype=np.int8).reshape(Hv, Wv)
            res_v = m.info.resolution
            ox_v = m.info.origin.position.x
            oy_v = m.info.origin.position.y
            for k in np.nonzero(seen)[0]:
                fx, fy = fpos[k]
                lim = dist[k] - 1.6 * res_v # face sits ON a wall cell
                if lim <= 0:
                    continue
                n = max(2, int(lim / (0.6 * res_v)))
                ts = np.linspace(0.0, lim / dist[k], n)[1:]
                ix = ((rx + (fx - rx) * ts - ox_v) / res_v).astype(int)
                iy = ((ry + (fy - ry) * ts - oy_v) / res_v).astype(int)
                okb = (ix >= 0) & (ix < Wv) & (iy >= 0) & (iy < Hv)
                if (grid_v[iy[okb], ix[okb]] > 50).any():
                    seen[k] = False
        # frontality squared punishes oblique views; sqrt proximity keeps
        # head-on credit at range
        prox = np.minimum(1.0, self.SEEK_R_FULL / dist)
        q = frontality ** 2 * np.sqrt(prox)
        # detector-reach rule-out: the detector fires on a 2" square out to
        # ~4 m, so one unoccluded view fully inspects a face. Frontality floor
        # 0.42 mirrors the detector's aspect gate (~65° worst case).
        q = np.where((dist <= 4.0) & (frontality > 0.42), 1.0, q)
        for (fx, fy), d, qi in zip(fpos[seen], dix[seen], q[seen]):
            k = self._face_key(fx, fy, d)
            if qi > self._face_q.get(k, 0.0):
                self._face_q[k] = float(qi)
            if qi >= self.SEEK_QMIN:
                self._done_bins.add(k)


    def _register_trip_fail(self, map_xy):
        """Count a failed trip here. Three strikes within the rolling window
        retire the spot, or the novelty bias revives it after every cooldown."""
        if not map_xy:
            return
        b = (round(map_xy[0] * 2) / 2.0, round(map_xy[1] * 2) / 2.0)
        sf = getattr(self, '_seam_fail', {})
        cnt, _ = sf.get(b, (0, 0.0))
        sf[b] = (cnt + 1, time.time() + 180.0)
        self._seam_fail = sf


    def _frontier_push_goal(self, m, mp):
        """Nearest routable free cell on the boundary of the best reachable
        unknown blob (map frame), or None: undiscovered squares are in the
        unknown mass, whatever is left to sweep in mapped clutter."""
        from scipy.ndimage import (label,
                                   distance_transform_edt)
        H, W = m.info.height, m.info.width
        grid = np.asarray(m.data, dtype=np.int8).reshape(H, W)
        unknown = grid == -1
        free = grid == 0
        res = m.info.resolution
        clear_m = distance_transform_edt(grid <= 50) * res
        lbl, n = label(unknown)
        if n == 0:
            return None
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        # blobs rank by size / travel cost; a blob that just got a push cools
        # for 180 s (not a blacklist), so smaller regions get their turn
        now_bc = time.time()
        self._blob_cool = {k: v for k, v in
                           getattr(self, '_blob_cool', {}).items()
                           if v > now_bc}
        blob_rank = []
        for bi in np.argsort(sizes)[::-1][:6]:
            if sizes[bi] < 300: # ~0.75 m^2 at 5 cm — noise blobs
                break
            ys_b, xs_b = np.nonzero(lbl == bi)
            cbx = m.info.origin.position.x + (float(xs_b.mean()) + 0.5) \
                * res
            cby = m.info.origin.position.y + (float(ys_b.mean()) + 0.5) \
                * res
            bkey = (round(cbx), round(cby))
            cooled = self._blob_cool.get(bkey, 0.0) > now_bc
            cost = max(math.hypot(cbx - mp[0], cby - mp[1]), 1.0)
            blob_rank.append((cooled, -float(sizes[bi]) / cost, bi, bkey))
        blob_rank.sort() # un-cooled first, best gain/cost
        # two passes: seek a routable candidate across all blobs, and only then
        # fall back to carrot probing, so an unroutable blob can't outrank a
        # reachable one
        for probe_pass in (False, True):
            if probe_pass:
                self.get_logger().info(
                    'frontier: no routable candidate in any blob — '
                    'falling back to carrot probing',
                    throttle_duration_sec=15.0)
            got = self._frontier_scan_blobs(
                m, mp, lbl, sizes, free, clear_m, res, blob_rank,
                now_bc, probe_pass)
            if got is not None:
                return got
        self.get_logger().info(
            'frontier: no admissible candidate (cooled/unroutable/'
            'saturated seams)', throttle_duration_sec=10.0)
        return None


    def _frontier_scan_blobs(self, m, mp, lbl, sizes, free, clear_m,
                             res, blob_rank, now_bc, probe_pass):
        """One pass over ranked blobs. probe_pass=False accepts only
        A*-routable candidates; True allows the seam-rotation carrot."""
        from scipy.ndimage import binary_dilation
        for _, _, bi, bkey in blob_rank:
            # 0.22 m clearance + 3-cell reach: the robot only stands here
            edge = binary_dilation(lbl == bi, iterations=3) & free \
                & (clear_m >= 0.22)
            ys, xs = np.nonzero(edge)
            if not len(ys):
                continue
            cx = m.info.origin.position.x + (xs + 0.5) * res
            cy = m.info.origin.position.y + (ys + 0.5) * res
            d = np.hypot(cx - mp[0], cy - mp[1])
            # novelty bias: discount candidates by distance from visited
            # ground, so seams in untouched regions win ties
            if self._visited_cells:
                from scipy.spatial import cKDTree
                va = (np.array(sorted(self._visited_cells),
                               dtype=np.float64) + 0.5) * 0.5
                dv = np.minimum(
                    cKDTree(va).query(np.stack([cx, cy], axis=1))[0], 2.0)
                d_rank = d - 1.5 * dv
            else:
                d_rank = d
            # visibility-gain re-rank: among close, novel seams prefer the one
            # that reveals the most area
            order = np.argsort(np.where(d >= 0.8, d_rank, np.inf))[:12]
            order = order[np.isfinite(d_rank[order])]
            if len(order) > 1:
                try:
                    gains = self._visibility_gain(
                        np.asarray(m.data, dtype=np.int8).reshape(
                            m.info.height, m.info.width), m.info,
                        np.stack([cx[order], cy[order]], axis=1))
                    gmax = float(gains.max())
                    if gmax > 0:
                        # blend: keep travel cost meaningful, let a
                        # 2x-visibility seam outrank a slightly nearer one
                        blended = d_rank[order] - 2.0 * (gains / gmax)
                        order = order[np.argsort(blended)]
                except Exception as e:
                    self.get_logger().warning(
                        f'frontier visibility-gain skipped: {e}',
                        throttle_duration_sec=30.0)
            # frontier goals respect cool spots too
            now_fs = time.time()
            spots_f = [(sx, sy) for (sx, sy), tt
                       in getattr(self, '_cool_spots', []) if tt > now_fs]
            # 3-strike seam-fail skip
            self._seam_fail = {k: v for k, v in
                               getattr(self, '_seam_fail', {}).items()
                               if v[1] > now_fs}
            checked = 0
            t_fp0 = time.time() # wall budget for the candidate loop
            for j in order:
                if not np.isfinite(d[j]):
                    break
                if checked >= 5 or time.time() - t_fp0 > 1.2:
                    break
                cand = (float(cx[j]), float(cy[j]))
                if any(math.hypot(cand[0] - sx, cand[1] - sy) < 0.45
                       for sx, sy in spots_f):
                    continue
                bsf = (round(cand[0] * 2) / 2.0, round(cand[1] * 2) / 2.0)
                if self._seam_fail.get(bsf, (0, 0.0))[0] >= 3:
                    continue
                # a seam behind a corridor that just failed is the same trap
                if self._corridor_dead(tuple(mp), cand):
                    continue
                checked += 1
                mc = self.live_subgoaler.route_min_clearance(
                    tuple(mp), cand)
                # any routable candidate qualifies: reaching a frontier in
                # dense clutter necessarily threads a squeeze
                if mc is not None:
                    self.get_logger().info(
                        f'frontier push: {int(sizes[bi])} unknown cells — '
                        f'goal map({cand[0]:.1f},{cand[1]:.1f}) '
                        f'(route clearance {mc:.2f} m)')
                    self._blob_cool[bkey] = now_bc + 180.0
                    return cand
                n_unroutable = getattr(self, '_unroutable_n', 0) + 1
                self._unroutable_n = n_unroutable
                self.get_logger().info(
                    f'frontier candidate map({cand[0]:.1f},'
                    f'{cand[1]:.1f}) unroutable at every inflation '
                    f'(total {n_unroutable}) — expert discriminator: '
                    'all-fail => live-map sealing, mixed => ranking',
                    throttle_duration_sec=20.0)
            # carrot fallback with seam rotation, reached only when no blob
            # offered a routable candidate. Each 0.5 m goal bin gets 3 probes
            # per 3 min, so another part of the perimeter gets tried, and the
            # gate + policy probe the real corridor as the map opens.
            if not probe_pass:
                continue
            now_sf = time.time()
            self._seam_fail = {k: v for k, v in
                               getattr(self, '_seam_fail', {}).items()
                               if v[1] > now_sf}
            for j in order:
                if not np.isfinite(d[j]):
                    break
                cand = (float(cx[j]), float(cy[j]))
                if any(math.hypot(cand[0] - sx, cand[1] - sy) < 0.45
                       for sx, sy in spots_f):
                    continue
                b = (round(cand[0] * 2) / 2.0, round(cand[1] * 2) / 2.0)
                cnt, _ = self._seam_fail.get(b, (0, 0.0))
                if cnt >= 3:
                    continue
                self._seam_fail[b] = (cnt + 1, now_sf + 180.0)
                self.get_logger().info(
                    f'frontier push (carrot — no live route, seam try '
                    f'{cnt + 1}/3): {int(sizes[bi])} unknown cells — goal '
                    f'map({cand[0]:.1f},{cand[1]:.1f})')
                self._blob_cool[bkey] = now_bc + 180.0
                return cand
        return None


    def _pick_seek_explore_goal(self, pose, ranges_m):
        """Run the exploration pick on a worker thread: its route queries can
        take seconds, which would block the control tick while the last
        velocity command stays latched. One pick runs at a time."""
        if getattr(self, '_pick_busy', False):
            return
        self._pick_busy = True
        rm = None if ranges_m is None else np.array(ranges_m, copy=True)

        def _work():
            try:
                self._pick_seek_explore_goal_impl(pose, rm)
            except Exception as e:
                self.get_logger().error(f'explore pick failed: {e!r}')
            finally:
                self._pick_busy = False
        threading.Thread(target=_work, daemon=True).start()


    def _pick_seek_explore_goal_impl(self, pose, ranges_m):
        """Choose where to explore: a frontier push after repeated churn, else
        a viewpoint on an uninspected face, else a wander goal. Runs on a
        worker thread, so it gives up if a target or escape claims the goal."""
        if self._escaping() and self.goal is not None:
            return
        if self._seek_color is not None:
            # a target was adopted while this pick computed: don't stomp its
            # goal or plant a stale _face_target
            return
        m = self._map
        mp = self._map_pose(pose)
        self._face_target = None
        if m is not None and mp is not None \
                and getattr(self, '_explore_churn', 0) >= 2:
            fg = self._frontier_push_goal(m, mp)
            if fg is not None:
                vp_w = self._map_to_world(fg)
                if vp_w is not None:
                    self._explore_churn = 0
                    # fresh slate: strikes from the pre-push churn at this
                    # pinch must not kill the trip
                    self._recovery_spots.clear()
                    self.goal = vp_w
                    # pseudo face target: reuses the trip leash and the
                    # arrival panoramic — a spin at a fresh frontier is
                    # exactly where an unseen square gets sighted
                    self._face_target = {'key': ('frontier',),
                                         'map_xy': fg, 'dir': -1,
                                         'world': vp_w, 'viewpoint': vp_w}
                    self._face_view_since = None
                    now_s = self.get_clock().now().nanoseconds * 1e-9
                    self._wander_next = now_s + self.wander_interval
                    now_w = time.time()
                    self._trip_deadline = now_w + 75.0
                    self._trip_best_d = float('inf')
                    self._trip_progress_t = now_w
                    return
        if m is not None and mp is not None:
            fpos, dix = self._all_faces(m)
            if len(fpos):
                keys = [self._face_key(fx, fy, d)
                        for (fx, fy), d in zip(fpos, dix)]
                now_cool = time.time()
                self._face_cool = {k: t for k, t
                                   in self._face_cool.items()
                                   if t > now_cool}
                # abandons cool a 0.45 m disc, not one key: a neighbouring face
                # on the same wall would stall the same way
                spots = [(sx, sy) for (sx, sy), t
                         in getattr(self, '_cool_spots', [])
                         if t > now_cool]
                self._cool_spots = [((sx, sy), t) for (sx, sy), t
                                    in getattr(self, '_cool_spots', [])
                                    if t > now_cool]
                todo = np.array([k not in self._face_cool
                                 and not self._face_done_at(fx, fy, d)
                                 and all(math.hypot(fx - sx, fy - sy)
                                         >= 0.45 for sx, sy in spots)
                                 for k, (fx, fy), d
                                 in zip(keys, fpos, dix)])
                # regions blacklisted by the recovery-loop breaker sit out their
                # expiry; the next-nearest face is otherwise in the same pocket
                now_rb = time.time()
                self._region_block = {k: t for k, t
                                      in self._region_block.items()
                                      if t > now_rb}
                if self._region_block and todo.any():
                    bins = np.floor(fpos / 2.0).astype(int)
                    open_r = np.array(
                        [(int(bx), int(by)) not in self._region_block
                         for bx, by in bins])
                    if (todo & open_r).any(): # never empty the face list
                        todo = todo & open_r
                if todo.any():
                    fpos, dix = fpos[todo], dix[todo]
                    keys = [k for k, t in zip(keys, todo) if t]
                    qs = np.array([self._face_q_at(fx, fy, d)
                                   for (fx, fy), d in zip(fpos, dix)])
                    normals = np.array(self._DIRS)[dix]
                    vps = fpos + self.SEEK_VIEW_DIST * normals
                    d_r = np.hypot(vps[:, 0] - mp[0], vps[:, 1] - mp[1])
                    # near-standstill needs far goals
                    v_now = abs(self._odom.twist.twist.linear.x)
                    lo = 2.5 if v_now < 0.05 \
                        else self.get_parameter('seek_min_goal_dist').value
                    ok = (d_r >= lo) & (d_r <= self.wander_goal_dist)
                    if not ok.any() and (d_r >= lo).any():
                        ok = d_r >= lo
                    if ok.any():
                        idx = np.nonzero(ok)[0]
                        # log-scaled urgency: 1 at q<=0.01, 0 at q=SEEK_QMIN
                        qf = np.clip(qs[idx], 0.01, self.SEEK_QMIN)
                        urg = np.log(qf / self.SEEK_QMIN) \
                            / math.log(0.01 / self.SEEK_QMIN)
                        # fresh cells are usually map churn, so a face's weight
                        # is a sigmoid of its age (50% at 12 s, ~95% by 24 s)
                        now_born = time.time()
                        for k in keys:
                            self._face_born.setdefault(k, now_born)
                        ages = np.array([now_born - self._face_born[keys[j]]
                                         for j in idx])
                        w_age = 1.0 / (1.0 + np.exp(-(ages - 12.0) / 4.0))
                        urg = urg * w_age
                        # effective cost: distance discounted by urgency, so
                        # a far never-seen face can outrank a near half-seen;
                        # immature faces additionally pay up to 2x
                        s = d_r[idx] * (1.0 - 0.6 * urg) * (2.0 - w_age)
                        # coarse region tour: favor the richest 2 m region, as
                        # pure nearest-first drains one pocket at a time
                        rbin = np.floor(fpos[idx] / 2.0).astype(int)
                        rkey = rbin[:, 0] * 10007 + rbin[:, 1]
                        best_rk, best_score = None, -1.0
                        for rk in np.unique(rkey):
                            sel = rkey == rk
                            cnt = int(sel.sum())
                            dmin_r = float(d_r[idx][sel].min())
                            score = cnt / math.sqrt(max(dmin_r, 0.5))
                            if score > best_score:
                                best_score, best_rk = score, rk
                        s = np.where(rkey == best_rk, s, s * 4.0)
                        # refine the cheapest few: tight corridors and recently
                        # punishing areas sink in the queue
                        order = idx[np.argsort(s)[:8]]
                        s_by = dict(zip(idx.tolist(), s.tolist()))
                        now_w = time.time()
                        hot = [(hx, hy) for t, hx, hy in self._recovery_spots
                               if now_w - t < 120.0]
                        rescored = []
                        t_budget0 = time.time()
                        for j in order:
                            cost = s_by[j]
                            # each refinement is a live-map A*: past the 0.3 s
                            # budget, take raw costs
                            if time.time() - t_budget0 > 0.3:
                                rescored.append((cost, j))
                                continue
                            mc = self.live_subgoaler.route_min_clearance(
                                (mp[0], mp[1]), tuple(vps[j]))
                            if mc is None:
                                cost *= 3.0 # no route yet — last resort
                            elif mc < 0.26:
                                # route squeezes below what the transit gate
                                # will drive (stop 0.21 + margin)
                                cost *= 6.0
                            else:
                                cost *= 1.0 + 0.8 * max(
                                    0.0, 1.0 - mc / 0.40)
                            vw = self._map_to_world(vps[j])
                            if vw is not None and hot:
                                n_hot = sum(1 for hx, hy in hot
                                            if math.hypot(vw[0] - hx,
                                                          vw[1] - hy) < 1.2)
                                cost *= 1.0 + 0.5 * n_hot
                            rescored.append((cost, j))
                        rescored.sort(key=lambda t: t[0])
                        # first routable of the rescored set wins; if none, the
                        # best face becomes a best-effort trip, bounded by the
                        # progress leash and face cooling
                        i = None
                        t0v = time.time()
                        for _, j in rescored:
                            if time.time() - t0v > 0.45:
                                break
                            if self.live_subgoaler.route_min_clearance(
                                    tuple(mp), tuple(vps[j])) is not None:
                                i = j
                                break
                        best_effort_goal = None
                        if i is None and rescored:
                            i = rescored[0][1]
                            # aim at the farthest reachable point toward the
                            # face: the arrival spin there rules out faces
                            # beyond the blockage and grows the map
                            ap = self.live_subgoaler.reachable_toward(
                                tuple(mp), tuple(vps[i]))
                            if ap is not None and math.hypot(
                                    ap[0] - mp[0], ap[1] - mp[1]) > 0.6:
                                best_effort_goal = self._map_to_world(ap)
                            self.get_logger().info(
                                'no routable face in rescored set — '
                                'best-effort trip '
                                f'({"approach point" if best_effort_goal else "viewpoint"}'
                                ', leash armed)',
                                throttle_duration_sec=5.0)
                        if i is None:
                            self.get_logger().info(
                                'no face candidates — wander fallback',
                                throttle_duration_sec=5.0)
                            self._pick_wander_goal(pose, ranges_m)
                            return
                        vp_w = self._map_to_world(vps[i])
                        face_w = self._map_to_world(fpos[i])
                        # re-check: a target or escape may have claimed the
                        # goal during this compute
                        if self._seek_color is not None or \
                                (self._escaping()
                                 and self.goal is not None):
                            return
                        if vp_w is not None:
                            self.goal = best_effort_goal or vp_w
                            self._face_target = {'key': keys[i],
                                                 'map_xy': tuple(fpos[i]),
                                                 'dir': int(dix[i]),
                                                 'world': face_w,
                                                 'viewpoint': vp_w}
                            self._face_view_since = None
                            n_todo = int(todo.sum())
                            self.get_logger().info(
                                f'seek explore: face {d_r[i]:.1f} m away '
                                f'at map({fpos[i][0]:.1f},{fpos[i][1]:.1f}) '
                                f'({n_todo} faces unchecked)')
                            now_s = self.get_clock().now().nanoseconds * 1e-9
                            self._wander_next = now_s + self.wander_interval
                            # commitment leash: alive while closing >=0.3 m per
                            # 25 s, 75 s cap; never cut by the interval repick
                            now_w = time.time()
                            self._trip_deadline = now_w + 75.0
                            self._trip_best_d = float('inf')
                            self._trip_progress_t = now_w
                            return
        if self._seek_color is not None or \
                (self._escaping() and self.goal is not None):
            return # apply-time re-check (same race, wander branch)
        self._pick_wander_goal(pose, ranges_m) # frontier fallback


    def _view_face(self, pose):
        """Inspect from a face viewpoint: a full spin when several faces remain
        nearby (the spin credits them all), else a glance at the target face.
        Requires trustworthy scan stamps: skewed ones tear the SLAM map."""
        t = self._face_target
        cmd = Twist()
        if not self._spin_safe():
            # too close to something for a sweep — ease off first
            cmd.linear.x = -0.05
            self._tracker = None
            self._last_cmd_v = cmd.linear.x
            self._cmd_pub.publish(cmd)
            return True
        if self._face_view_since is None:
            self._face_view_since = time.time()
            # spin only when several unresolved faces sit in rule-out range
            n_todo = 0
            m = self._map
            mp = self._map_pose(pose)
            if m is not None and mp is not None:
                fpos, dix = self._all_faces(m)
                if len(fpos):
                    dd = np.hypot(fpos[:, 0] - mp[0], fpos[:, 1] - mp[1])
                    for (fx, fy), d1, di in zip(fpos, dd, dix):
                        if d1 <= 4.0 and not self._face_done_at(fx, fy, di):
                            n_todo += 1
                            if n_todo >= 3:
                                break
            t['spin'] = n_todo >= 3
        if t.get('spin'):
            if time.time() - self._face_view_since \
                    > 2.0 * math.pi / self.SEEK_FACE_W + 0.5:
                self._done_bins.add(t['key']) # the spin covered all in view
                self._face_target = None
                self.goal = None # repick now (commitment)
                self._explore_churn = 0 # productive trip
                return False
            cmd.angular.z = self.SEEK_FACE_W
        else:
            # sparse surroundings: just glance at the target face
            bearing = math.atan2(t['world'][1] - pose[1],
                                 t['world'][0] - pose[0]) - pose[2]
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if abs(bearing) > self.SEEK_FACE_TOL:
                cmd.angular.z = math.copysign(self.SEEK_FACE_W, bearing)
            elif time.time() - self._face_view_since > 1.5:
                self._done_bins.add(t['key'])
                self._face_target = None
                self.goal = None # repick now (commitment)
                self._explore_churn = 0 # productive trip
                return False
        self._tracker = None
        self._last_cmd_v = 0.0
        self._cmd_pub.publish(cmd)
        return True
