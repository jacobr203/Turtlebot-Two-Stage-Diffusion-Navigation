"""Deciding where to go when nothing is being chased.

Covers frontier exploration, wander goal selection, and the systematic
tour of obstacle faces the camera has not yet inspected. Goals are
ranked by expected information gain against travel cost, so a small
unknown region nearby can outrank a large one across the arena.

References:
    Yamauchi, B. (1997), "A Frontier-Based Approach for Autonomous
    Exploration", IEEE International Symposium on Computational
    Intelligence in Robotics and Automation, pp. 146-151. Frontier
    detection on the boundary between free and unknown space.

    Gonzalez-Banos, H. H. and Latombe, J.-C. (2002), "Navigation
    Strategies for Exploring Indoor Environments", International
    Journal of Robotics Research 21(10-11):829-848. Scoring candidate
    goals by expected gain divided by cost to reach.

    Hoerner, J. (2016), `explore_lite` (ROS package). Practical
    gain-per-cost frontier ranking with cooldowns on served frontiers.

Mixin composed into DiffusionPolicyNode; `self` is the node.
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
        """Best known-free cell to head for: safely clear of obstacles,
        far (up to wander_goal_dist), frontier-adjacent preferred. Returns a
        world-frame goal or None (no map / no TF / no acceptable cell)."""
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

        # near-standstill starts must stay in the saturated cruise regime
        # (decisions #19): demand a far goal, else fall back to the carrot
        min_d = 3.0 if v_now < 0.05 else p('wander_min_goal_dist').value
        # ARENA-SCALE coverage targets): the target may sit far beyond
        # wander_goal_dist — the subgoal planner turns it into successive in-
        # distribution 1.2 m carrots (the same two-stage pattern seek uses),
        # and goal features saturate at goal_clip. The 3.5 m number was a PICK
        # radius, and at arena scale it produced a mean-reverting random walk
        # (operator's looping-figure-eight evidence).
        ok = (d >= min_d) & (d <= max(self.wander_goal_dist, 12.0))
        if not ok.any():
            return None
        # gain-flavored scoring: frontier + novelty −
        # mild travel cost − coarse revisit count + heading momentum.
        # The pure distance REWARD (d / wander_goal_dist) is gone — it
        # was the mean-reversion engine.
        score = 1.5 * frontier[ys, xs] - 0.05 * d
        # VISIBILITY-GAIN ranking (frontier-adjacency is a PROXY for "unknown
        # area is near"; what actually converts a target is unknown area this
        # candidate can SEE. Ray-cast the shortlist and up-weight by visible-
        # unknown count. Only the top candidates by the cheap score are cast
        # (the raycast is vectorized but not free, and this runs inside a
        # control tick).
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
        # novelty bias: up-weight candidates far from every
        # visited 0.5 m cell, capped at 2 m of separation
        if self._visited_cells:
            from scipy.spatial import cKDTree
            va = (np.array(sorted(self._visited_cells), dtype=np.float64)
                  + 0.5) * 0.5
            dv = cKDTree(va).query(np.stack([cx, cy], axis=1))[0]
            score = score + 0.8 * np.minimum(dv, 2.0) / 2.0
        # coarse visit-count penalty (1.5 m grid — "the 0.5 m memory is
        # the right idea at the wrong scale")
        vg = getattr(self, '_visit_grid', None)
        if vg:
            gx = np.floor(cx / 1.5).astype(int)
            gy = np.floor(cy / 1.5).astype(int)
            vc = np.array([vg.get((int(a), int(b)), 0)
                           for a, b in zip(gx, gy)], dtype=np.float64)
            score = score - 0.5 * np.minimum(vc, 8.0) / 8.0
        # heading persistence: figure-eights are direction reversals —
        # favor goals near the current heading unless outscored
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
        # wander-goal commitment: the seek 6 s
        # face-retry loop lands here on every failed reachability pass,
        # and minting a fresh goal each time gave goals ~6-13 s
        # lifetimes — 72 goals in 924 s, each abandoned under ~15 %
        # complete, confining 32 m of driving to a ~4.75 m² strip
        #. Persist the active goal while it is
        # actually being closed — same leash shape as face trips:
        # alive while closing >=0.25 m per 20 s, 60 s cap, dropped on
        # arrival (<0.5 m) or when another writer has taken the goal.
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
            # no usable map yet: carrot toward the most open scan direction.
            # Circular windowed minimum (±15°) picks the DIRECTION only; the
            # goal sits wander_goal_dist out to stay in the cruise regime.
            wm = np.min(np.stack([np.roll(ranges_m, k)
                                  for k in range(-15, 16)]), axis=0)
            b = int(np.argmax(wm))
            ang = pose[2] + math.radians(b)
            self.goal = (pose[0] + self.wander_goal_dist * math.cos(ang),
                         pose[1] + self.wander_goal_dist * math.sin(ang))
            self.get_logger().info(
                f'wander goal (scan fallback): {self.wander_goal_dist:.1f} m '
                f'at {b}° (clearance {wm[b]:.1f} m)')
        # distance-scaled commitment cap:
        # arena-scale coverage legs need more than the old 60 s — a
        # 12 m leg at gate-crawl pace is legitimate for minutes as
        # long as the progress leash keeps being satisfied
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
        """Best recorded quality near this face — 3x3 quantization
        neighborhood, so Cartographer's pose-graph corrections shifting a
        wall a few cm don't orphan its certainty (observed: 'green walls
        clearing out' between snapshots after a big map re-optimization)."""
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
        """Score obstacle faces the camera happens to be looking at.

        Quality is how square-on the view is, multiplied by how close it
        is. A head-on view keeps full credit out to the credit range,
        while a glancing view scores low and leaves the face worth
        revisiting. Each face keeps its best score so far.
        """
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
        # occlusion gate: frontality x
        # proximity alone granted rule-out credit THROUGH walls — fatal
        # once approach-point trips park the camera at a blocking band
        # and spin (a hidden face marked done is erased from the sweep
        # forever). Walk the ray robot->face on the occupancy grid and
        # drop credit if any occupied cell sits strictly before the
        # face's own wall.
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
        # angle-dominant rebalance (2026-07-20): frontality squared punishes
        # oblique views hard; sqrt proximity keeps head-on credit at range —
        # "head-on but far = certain, sideways = go look again"
        prox = np.minimum(1.0, self.SEEK_R_FULL / dist)
        q = frontality ** 2 * np.sqrt(prox)
        # detector-reach rule-out: the color
        # detector fires on a 2" square out to ~4 m at moderate obliquity
        # a face SEEN under those conditions cannot be hiding a swatch,
        # so a single glance fully inspects it (one 360° spin rules out
        # every unoccluded pillar in view). Frontality floor 0.42 mirrors
        # the detector's own aspect gate (~65° worst case). Edge-on or
        # beyond-range glimpses keep the graded partial score — the
        # "partly under cover" tier that still earns a visit.
        q = np.where((dist <= 4.0) & (frontality > 0.42), 1.0, q)
        for (fx, fy), d, qi in zip(fpos[seen], dix[seen], q[seen]):
            k = self._face_key(fx, fy, d)
            if qi > self._face_q.get(k, 0.0):
                self._face_q[k] = float(qi)
            if qi >= self.SEEK_QMIN:
                self._done_bins.add(k)


    def _register_trip_fail(self, map_xy):
        """Record that a trip to this spot ended badly.

        Three strikes within a rolling window retire the spot. Without
        this the novelty bias revives the same unreachable candidate
        every time its cooldown lapses, and the robot re-attempts it
        indefinitely.
        """
        if not map_xy:
            return
        b = (round(map_xy[0] * 2) / 2.0, round(map_xy[1] * 2) / 2.0)
        sf = getattr(self, '_seam_fail', {})
        cnt, _ = sf.get(b, (0, 0.0))
        sf[b] = (cnt + 1, time.time() + 180.0)
        self._seam_fail = sf


    def _frontier_push_goal(self, m, mp):
        """Nearest routable free cell on the boundary of the largest
        reachable unknown blob (map frame), or None. Frontier push
        (2026-08-04, world 768): the face sweep's region tour ground the
        mapped clutter for 15+ min while a >5k-cell unknown mass — and
        the third square — sat unexplored in the north; undiscovered
        squares are BY DEFINITION in the unknown mass."""
        from scipy.ndimage import (label, binary_dilation,
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
        # gain-per-cost blob ranking + served-blob cooldown explore_lite
        # pattern). Size-only top-3 let the biggest unknown mass monopolize
        # every push — testing-convicted SW-hole mechanism. Blobs now rank by
        # size / travel-cost; a blob that just received a push COOLS for 180 s
        # (a cooldown, NOT a blacklist — sealed-region detection à la world 733
        # must survive), so smaller quadrants get their turn while the big mass
        # rests.
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
        # TWO-PASS structure: pass 1 seeks a
        # ROUTABLE candidate across ALL ranked blobs; only if none
        # exists anywhere does pass 2 fall back to carrot probing. The
        # old single pass probed from the FIRST blob that failed
        # routing, so a routable seam in a lower-ranked blob was never
        # even tried — testing logged 4/4 pushes as carrot fallbacks
        # ("no live route") while the robot searched 14 minutes for
        # orange. Reachability is definitional in frontier exploration
        # (Yamauchi 1997; gain-per-cost needs a real path cost), so an
        # unroutable blob must not out-rank a reachable one.
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
            # 0.22 m standing clearance + 3-cell reach (2026-08-04 live:
            # 0.30/2 cells left ZERO candidates on the evolved map — the
            # robot only needs to STAND here, half-diagonal 0.113)
            edge = binary_dilation(lbl == bi, iterations=3) & free \
                & (clear_m >= 0.22)
            ys, xs = np.nonzero(edge)
            if not len(ys):
                continue
            cx = m.info.origin.position.x + (xs + 0.5) * res
            cy = m.info.origin.position.y + (ys + 0.5) * res
            d = np.hypot(cx - mp[0], cy - mp[1])
            # novelty bias: nearest-first alone kept recycling
            # seams at the explored strip's edge — discount candidates by
            # their distance from all visited ground so the untouched
            # quadrant's seams win ties
            if self._visited_cells:
                from scipy.spatial import cKDTree
                va = (np.array(sorted(self._visited_cells),
                               dtype=np.float64) + 0.5) * 0.5
                dv = np.minimum(
                    cKDTree(va).query(np.stack([cx, cy], axis=1))[0], 2.0)
                d_rank = d - 1.5 * dv
            else:
                d_rank = d
            # VISIBILITY-GAIN re-rank (d_rank picks the seam that is CLOSE and
            # novel; among those, prefer the one that actually reveals area. A
            # seam facing into a wall pocket and a seam opening onto the
            # unexplored north score the same under cell-count ranking — they
            # do not here.
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
            # frontier goals respect cool spots too (
            # the same unreachable seam goal was pushed four times)
            now_fs = time.time()
            spots_f = [(sx, sy) for (sx, sy), tt
                       in getattr(self, '_cool_spots', []) if tt > now_fs]
            # 3-strike seam-fail skip in the MAIN loop too:
            # only the carrot fallback consulted it, so the novelty bias
            # kept resurrecting the same failed candidate (6x one seam)
            self._seam_fail = {k: v for k, v in
                               getattr(self, '_seam_fail', {}).items()
                               if v[1] > now_fs}
            checked = 0
            t_fp0 = time.time() # wall budget: this loop had
            # a COUNT limit but no time budget —
            # ~5 s of the 10-11 s storm bursts
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
                # a seam behind a corridor a re-aim just failed on is the same
                # trap wearing a different hat (
                if self._corridor_dead(tuple(mp), cand):
                    continue
                checked += 1
                mc = self.live_subgoaler.route_min_clearance(
                    tuple(mp), cand)
                # any ROUTABLE candidate qualifies — demanding a wide
                # corridor (0.26, live 2026-08-04) rejected every path to
                # the north blob, because reaching a frontier in this
                # density NECESSARILY threads the squeeze; that is what
                # the v5 policy + benchmark margins are for
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
            # carrot fallback WITH SEAM ROTATION (2026-08-04, world 733
            # take 2: the carrot probed the SAME sealed nearest seam every
            # 45 s for 20 min). Each 0.5 m goal bin gets 3 probes per
            # 3 min; saturated bins defer to the next-nearest candidate so
            # a different part of the blob perimeter gets tried. Push
            # toward the seam: subgoal fails onto the straight carrot,
            # the gate + policy probe the real corridor, and the map opens
            # en route (color targets' "direct approach" pattern).
            # SECOND-PRIORITY only — reached solely when no
            # blob anywhere offered a routable candidate.
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
        """Start the exploration pick on a worker thread.

        The route queries this pick makes can take seconds, which would
        block the control tick and leave the last velocity command
        latched — the robot keeps driving blind while nothing is
        watching. The control tick therefore only launches the worker;
        results are applied by the worker and picked up on a later tick,
        with the same tolerance for a second or two of goal staleness
        that the off-thread planner rebuild already assumes.

        Only one pick runs at a time, and a result that lands during an
        escape manoeuvre is discarded.
        """
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
        """Choose where to explore next.

        Preference order:
          1. a viewpoint on an obstacle face the camera has not yet
             inspected (the systematic face sweep);
          2. a frontier goal, when no face is usable — either the map is
             too young or every known face is already covered, so the
             right move is to grow the map;
          3. a frontier push toward the largest unknown region, which
             takes priority once several trips in a row have churned.

        Runs on a worker thread, so it must not assume state stays still
        underneath it and gives up if an escape claims the goal midway.
        """
        if self._escaping() and self.goal is not None:
            return
        if self._seek_color is not None:
            # a color target was adopted while this pick computed —
            # applying the pick now would stomp the approach goal and
            # plant a stale _face_target that shadows the breaker's
            # color branch (the run-4 bug, async edition; observed
            # 2026-08-05: worker landed 0.1 s after purple adoption)
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
                    # fresh slate for the push (2026-08-04 run 8): the
                    # breaker was killing frontier trips 10-20 s in with
                    # strikes accrued during the PRE-push churn at the
                    # same pinch — the trip never got its 3 own strikes
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
                # RADIUS cooling (2026-08-04 a stalled face was replaced by its
                # 30 cm neighbor four cycles in a row — same wall, same stall).
                # Abandons cool a 0.45 m disc, not just one key.
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
                # pocket regions blacklisted by the recovery-loop breaker
                # sit out for their expiry (2026-08-03, world 768) — the
                # next-nearest face is otherwise in the same pocket
                now_rb = time.time()
                self._region_block = {k: t for k, t
                                      in self._region_block.items()
                                      if t > now_rb}
                if self._region_block and todo.any():
                    bins = np.floor(fpos / 2.0).astype(int)
                    open_r = np.array(
                        [(int(bx), int(by)) not in self._region_block
                         for bx, by in bins])
                    if (todo & open_r).any(): # never blacklist into
                        todo = todo & open_r # an empty face list
                if todo.any():
                    fpos, dix = fpos[todo], dix[todo]
                    keys = [k for k, t in zip(keys, todo) if t]
                    qs = np.array([self._face_q_at(fx, fy, d)
                                   for (fx, fy), d in zip(fpos, dix)])
                    normals = np.array(self._DIRS)[dix]
                    vps = fpos + self.SEEK_VIEW_DIST * normals
                    d_r = np.hypot(vps[:, 0] - mp[0], vps[:, 1] - mp[1])
                    # near-standstill needs far goals (decisions #19)
                    v_now = abs(self._odom.twist.twist.linear.x)
                    lo = 2.5 if v_now < 0.05 \
                        else self.get_parameter('seek_min_goal_dist').value
                    ok = (d_r >= lo) & (d_r <= self.wander_goal_dist)
                    if not ok.any() and (d_r >= lo).any():
                        ok = d_r >= lo
                    if ok.any():
                        idx = np.nonzero(ok)[0]
                        # log-scaled urgency: near-zero
                        # quality is far more urgent than a mediocre view —
                        # urgency 1 at q<=0.01, 0 at q=SEEK_QMIN
                        qf = np.clip(qs[idx], 0.01, self.SEEK_QMIN)
                        urg = np.log(qf / self.SEEK_QMIN) \
                            / math.log(0.01 / self.SEEK_QMIN)
                        # curiosity maturity:
                        # freshly-appeared cells are usually map churn or
                        # half-observed edges — chasing them wastes trips.
                        # A face's exploration weight follows a sigmoid of
                        # its AGE (50% at 12 s, ~95% by 24 s); swatch
                        # detections bypass this entirely (targets branch).
                        # Churned keys read as young and self-discount.
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
                        # coarse region tour (2026-07-23, 3rd greedy
                        # complaint): pure nearest-first drains one pocket
                        # completely before ranging out — the robot orbits a
                        # quadrant for tens of minutes. Cluster faces into 2 m
                        # bins and favor the RICHEST region (unchecked count,
                        # mildly distance-discounted): faces inside the chosen
                        # region keep their local cost, everything else is
                        # pushed back.
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
                        # refine the cheapest few with corridor width and
                        # collision heat (heuristic cleanup, plan STATUS):
                        # tight A* corridors and recently-punishing areas
                        # sink in the queue instead of being retried
                        order = idx[np.argsort(s)[:8]]
                        s_by = dict(zip(idx.tolist(), s.tolist()))
                        now_w = time.time()
                        hot = [(hx, hy) for t, hx, hy in self._recovery_spots
                               if now_w - t < 120.0]
                        rescored = []
                        t_budget0 = time.time()
                        for j in order:
                            cost = s_by[j]
                            # clearance refinement is a live-map A* per
                            # candidate — on a grown late-run map 8 of them
                            # blew past the control period and tripped the
                            # tick-stall watchdog (observed 2026-07-23,
                            # world 613, right after orange FOUND). Budget:
                            # refine while under 0.3 s, take raw costs after.
                            if time.time() - t_budget0 > 0.3:
                                rescored.append((cost, j))
                                continue
                            mc = self.live_subgoaler.route_min_clearance(
                                (mp[0], mp[1]), tuple(vps[j]))
                            if mc is None:
                                cost *= 3.0 # no route yet — last resort
                            elif mc < 0.26:
                                # route squeezes below what the transit gate
                                # will drive (stop 0.21 + margin): sending
                                # the robot there IS the 8 s recovery
                                # metronome (2026-08-03, world 768). Near-
                                # prohibitive, not a mild 1.8x.
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
                        # REACHABILITY gate on the winner (9
                        # consecutive trips, 100% stall rate, all across
                        # an impassable field — cooling was doing the
                        # work rejection should). First routable of the
                        # rescored set wins. the top-4-or-wander
                        # version starved exploration outright (68
                        # rejections/15 min, exploration confined to the
                        # cleared strip, orange never sighted) — so none
                        # routable now takes the BEST face as a
                        # best-effort trip instead of falling to wander.
                        # Safe post-gate era: the progress leash (0.3 m
                        # per 25 s, 75 s cap) plus 90 s face cooling
                        # bound a doomed trip, vs. the unbounded 9-trip
                        # storms the gate was built against.
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
                            # aim the trip at the farthest
                            # REACHABLE point toward the face, not the
                            # unroutable viewpoint itself — 82 % of
                            # testing best-effort trips stalled the leash
                            # short of viewpoints across the impassable
                            # band. Arrival (<0.40 m of self.goal) then
                            # runs the normal inspection spin right at
                            # the band, where the camera can rule out
                            # faces beyond it and the map grows.
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
                        # apply-time re-check the entry guard narrows but does
                        # not close the race — a color adopted during this
                        # compute must not be stomped at apply time)
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
                            # face-trip commitment leash (2026-08-03): the
                            # trip is abandoned only by inspection, the
                            # breaker, or the leash — never by the interval
                            # repick (goal churn read as wander). Progress-
                            # based (world 768: EVERY flat-45 s trip in the
                            # gated clutter expired short, zero arrivals):
                            # alive while closing >=0.3 m per 25 s, 75 s cap.
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
        """Do one full panoramic spin at a face viewpoint.

        Coverage credits every face the camera sweeps past, not only the
        one that prompted the trip: turning to each face individually
        kept re-visiting the same region. A fixed spin duration also
        means a poorly chosen viewpoint cannot lock the robot up.

        Requires trustworthy scan timestamps — a full spin with skewed
        stamps tears the SLAM map.
        """
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
            # spin only when it PAYS (2026-07-23: pointless pivoting in
            # already-cleared regions): with rule-out coverage, driving here
            # cleared most nearby faces — a full 360° harvest is only worth it
            # when several unresolved faces sit in rule-out range.
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
                self._done_bins.add(t['key']) # target retired; the spin
                self._face_target = None # marked everything visible
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
