"""Running the policy and following what it predicts.

Two loops:

    control tick    observe -> plan a subgoal -> sample trajectories
                    from the diffusion model -> pick one
    tracker (20 Hz) pure pursuit along the chosen trajectory, so the
                    robot keeps moving smoothly between predictions

Several trajectories are drawn per tick and scored on progress,
smoothness and clearance rather than taking a single sample. Optional
clearance guidance nudges the denoiser away from obstacles while it
generates. Both are inference-time only, so the trained weights and
the evaluation comparison stay untouched.

References:
    Ho, J., Jain, A. and Abbeel, P. (2020), "Denoising Diffusion
    Probabilistic Models", NeurIPS 33:6840-6851.

    Chi, C. et al. (2023), "Diffusion Policy: Visuomotor Policy
    Learning via Action Diffusion", Robotics: Science and Systems.
    Receding-horizon execution of sampled action chunks.

    Sridhar, A. et al. (2024), "NoMaD: Goal Masked Diffusion Policies
    for Navigation and Exploration", ICRA. Sampling a batch of
    candidate trajectories at deployment and selecting among them.

    Carvalho, J. et al. (2023), "Motion Planning Diffusion: Learning
    and Planning of Robot Motions with Diffusion Models", IROS.
    Guiding the reverse diffusion process with a cost gradient.

Mixin composed into DiffusionPolicyNode; `self` is the node.
"""

import math
import time

import numpy as np
import torch

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as PathMsg

from common.geometry import robot_to_world, world_to_robot
from common.pure_pursuit import PurePursuit
from diffusion.dataset import crop_arc, obs_vec


class TrackingMixin:
    """Control loop, pure-pursuit tracker, and debug publishers."""


    # ── control: observe -> sample -> retarget tracker ───────────────────────
    def _make_clearance_guide(self, pts):
        """R2 (expert; MPD / NaviDiffusor pattern): clearance-cost
        guidance applied to x0_hat on the last denoise steps. Analytic
        gradient of 0.5*max(0, r_safe - d_min)^2 per waypoint against
        the scanned obstacle points, pushed back through the fixed
        denormalization. The model authors every sample — guidance
        nudges the clean estimate by a clipped normalized step, so
        over-guidance (NaviDiffusor's documented failure) is bounded
        by construction. Returns None (guidance off) for the goto
        benchmark path, an empty scan, or guide_scale 0."""
        lam = float(self.get_parameter('guide_scale').value)
        if lam <= 0.0 or not len(pts) or self.behavior == 'goto':
            return None
        lo = torch.as_tensor(self.stats['wp_lo'], dtype=torch.float32)
        hi = torch.as_tensor(self.stats['wp_hi'], dtype=torch.float32)
        scale = float(self.get_parameter('path_scale').value)
        span = (hi - lo) * scale # [2]
        P = torch.as_tensor(pts, dtype=torch.float32) # [N, 2]
        r_safe = 0.20

        def guide(x0):
            # x0: [K, 2, H] normalized -> wp [K, H, 2] robot frame [m]
            wp = ((x0.permute(0, 2, 1) + 1.0) / 2.0 * (hi - lo) + lo) \
                * scale
            diff = wp[:, :, None, :] - P[None, None, :, :]
            d = diff.norm(dim=-1) # [K, H, N]
            dmin, idx = d.min(dim=-1) # [K, H]
            viol = (r_safe - dmin).clamp(min=0.0)
            if not bool((viol > 0).any()):
                return x0
            nearest = P[idx] # [K, H, 2]
            away = (wp - nearest) / dmin.clamp(min=1e-6)[..., None]
            step_wp = lam * viol[..., None] * away # [m], away from pts
            dx = (step_wp * 2.0 / span).clamp(-0.06, 0.06)
            return x0 + dx.permute(0, 2, 1)
        return guide


    def _control_tick(self):
        # liveness stamp on EVERY exit (testing — the stall-storm saga's
        # true ending): only 2 of the tick's many return paths stamped
        # _last_tick_wall, so goal-None holds (async pick computing) and
        # seek-handled paths (aim rotations, facing holds, 9 s panoramic
        # spins — the "uniform 10-11 s burst ceiling") read as stalls.
        # The watchdog then fought the robot's own rotation commands
        # with 10 Hz zeros + tracker clears (the camera's pirouette-
        # during-"wheels zeroed" evidence). The faulthandler burst dump
        # settled it: executor IDLE mid-"stall" — nothing ever wedged.
        # The stamp now lives in a finally so the watchdog only fires
        # on its true target: a wedged executor that runs no ticks.
        ok = False
        try:
            self._control_tick_impl()
            self._tick_exc = 0
            ok = True
        except Exception:
            # exception-loop counter: a tick that
            # raises every call must NOT read as live — after 3
            # consecutive failures the stamp freezes and the watchdog
            # takes over (zeroing + eventual exit)
            self._tick_exc = getattr(self, '_tick_exc', 0) + 1
            ok = self._tick_exc <= 3
            raise
        finally:
            if ok:
                self._last_tick_wall = time.monotonic()
            self._tick_end_mono = time.monotonic()


    def _control_tick_impl(self):
        if self._scan is None or self._odom is None or self._done:
            return
        now = self.get_clock().now()
        pose = self._world_pose()
        # tight-approach mode: final 1.5 m to the current seek goal — the
        # gate/spin margins relax to benchmark values there (context
        # margins, 2026-07-24)
        # benchmark margins for ALL seek driving (2026-08-03, world 768
        # runs 2-4): at conservative transit margins the gate cannot cross
        # this density at all — every explore trip in three runs ended
        # zero-progress (stalled full-distance short; ~2 cycles of gate
        # stop -> 8 s metronome -> 4.5 s recovery per 25 s trip), and
        # color approaches only converged where benchmark margins applied.
        # The policy is validated threading BARN gaps at these margins;
        # the veto, breaker, pinch escape and wedge escalation are the
        # safety net. Conservative margins remain for wander/goto.
        self._tight_ok = (self.behavior == 'seek' and self.goal is not None)

        ranges_m = self._ranges_m

        if self.behavior == 'wander':
            now_s = now.nanoseconds * 1e-9
            if (self.goal is None or now_s >= self._wander_next
                    or math.hypot(pose[0] - self.goal[0],
                                  pose[1] - self.goal[1]) < 1.5):
                self._pick_wander_goal(pose, ranges_m)
                self._publish_subgoal(self.goal, now)
        elif self.behavior == 'seek':
            if self._seek_tick(pose, ranges_m, now):
                return
            if self.goal is None:
                # a retire path cleared the goal THIS tick (view_face
                # completion returns False to keep driving) — every
                # goal consumer below (subgoal plan, goal features)
                # NaN-crashes on None (two node deaths 2026-08-04,
                # second one via goal_feats after the first guard sat
                # too deep). Hold one tick; the next tick repicks.
                self._cmd_pub.publish(Twist())
                return
        else:
            d_goal = math.hypot(pose[0] - self.goal[0],
                                pose[1] - self.goal[1])
            if d_goal < self.goal_tol:
                self._done = True
                self._tracker = None
                self._cmd_pub.publish(Twist())
                self.get_logger().info('goal reached — stopping')
                return
            if d_goal < 1.0 and not self._near_goal_logged:
                # the policy imitates the literature's stopping behavior and may
                # hold slightly outside goal_tol — that still counts as
                # success by the BARN criterion (1 m radius)
                self._near_goal_logged = True
                self.get_logger().info(
                    f'within success radius ({d_goal:.2f} m of goal)')

        t0 = time.perf_counter()
        if self.behavior == 'seek':
            sub_w = self._seek_subgoal(pose)
            cond = world_to_robot(np.array(sub_w), pose)
            self._publish_subgoal(sub_w, now)
        elif self.subgoaler is not None:
            sub_w = self.subgoaler.subgoal(pose[:2])
            if sub_w is None:
                return # never planned successfully yet
            cond = world_to_robot(np.array(sub_w), pose)
            self._publish_subgoal(sub_w, now)
        else:
            cond = world_to_robot(np.array(self.goal), pose)
        t_plan = (time.perf_counter() - t0) * 1000

        # v3 checkpoints see only the forward arc (cfg scan_arc_deg); the
        # crop matches training exactly (single source: dataset.crop_arc).
        # The gate/recovery keep the full 360° — only the policy's view
        # narrows.
        scan = crop_arc(ranges_m / 3.5, self.cfg.get('scan_arc_deg', 360))
        scan_age = (now.nanoseconds
                    - (self._scan.header.stamp.sec * 10**9
                       + self._scan.header.stamp.nanosec)) * 1e-9
        tw = self._odom.twist.twist
        goal_feats = {}
        if self.conditioning == 'subgoal_goal':
            goal_r = world_to_robot(np.array(self.goal), pose)
            goal_feats = dict(goal_dx=goal_r[0], goal_dy=goal_r[1],
                              goal_dist=float(np.hypot(*goal_r)))
        vec = obs_vec(cond[0], cond[1], float(np.hypot(*cond)),
                      tw.linear.x, tw.angular.z, scan_age,
                      self.cfg['goal_clip'], **goal_feats)
        self._frames.append((scan.astype(np.float32), vec))
        while len(self._frames) < self._frames.maxlen:
            self._frames.appendleft(self._frames[0])

        scans = torch.from_numpy(np.stack([f[0] for f in self._frames]))[None]
        vecs = torch.from_numpy(
            np.concatenate([f[1] for f in self._frames]))[None]

        # K-sample generation + selection (expert R1 — the NoMaD deployment
        # pattern: 8 samples per tick through one batched denoise, pick by
        # score; NavDP selects by). Single-draw inference discards the
        # multimodality that is the POINT of a diffusion policy — the pocket-
        # mouth veto clusters and knife-edge coin-flips are the textbook
        # symptom. The sampler itself is untouched: selection, not perturbation
        # — the model remains the sole author of every candidate.
        K = max(1, int(self.get_parameter('sample_k').value))
        # obstacle point set — computed BEFORE sampling now:
        # it feeds both the R2 clearance guidance and the veto/selection
        gr = self._fused_gate_ranges()
        ang = np.radians(np.arange(360.0))
        near = np.isfinite(gr) & (gr < 2.5) & (gr > 0.17)
        pts = np.stack([gr[near] * np.cos(ang[near]),
                        gr[near] * np.sin(ang[near])], axis=1)
        guide = self._make_clearance_guide(pts)
        t1 = time.perf_counter()
        with torch.no_grad():
            if self.denoiser == 'ddim':
                sk = scans.repeat(K, 1, 1)
                vk = vecs.repeat(K, 1)
                out = self.ddpm.sample_ddim(self.policy, sk, vk,
                                            steps=self.ddim_steps,
                                            guide=guide)
            elif K > 1:
                # testing: candidate 0 is deterministic
                # DDIM eta=0 — mode-seeking, empirically smoother; the
                # other K-1 stay stochastic DDPM for threading diversity
                out_det = self.ddpm.sample_ddim(self.policy, scans,
                                                vecs,
                                                steps=self.ddim_steps,
                                                guide=guide)
                sk = scans.repeat(K - 1, 1, 1)
                vk = vecs.repeat(K - 1, 1)
                out_sto = self.ddpm.sample(self.policy, sk, vk,
                                           guide=guide)
                out = torch.cat([out_det, out_sto], dim=0)
            else:
                out = self.ddpm.sample(self.policy, scans, vecs,
                                       guide=guide)
        t_sample = (time.perf_counter() - t1) * 1000

        cands = (out.permute(0, 2, 1).numpy() + 1.0) / 2.0 \
            * (self.stats['wp_hi'] - self.stats['wp_lo']) + self.stats['wp_lo']
        cands = cands * self.get_parameter('path_scale').value
        # Waypoint clearance veto — collision PREVENTION, not recovery
        #: truncate the chunk at the first
        # waypoint whose bearing's scanned obstacle is nearer than the
        # waypoint + hull margin. The policy still navigates; we just
        # refuse the provably-unsafe tail instead of colliding and
        # backing out. Uses the fused (lidar+camera) gate ranges.
        # (v2 of the veto, 2026-07-23: the first bearing-ray version vetoed
        # paths passing BESIDE pillars — 28 recoveries and a meander in one
        # run. This is the true corridor test: perpendicular distance from
        # each path SEGMENT to each scanned obstacle POINT.)
        # the SAME test now scores every candidate — pick by
        # (safe length, then min clearance); veto only when ALL K are
        # unsafe. (pts computed above, pre-sampling, shared with R2
        # guidance; the >0.17 m floor excludes the robot's CURRENT
        # standoff — the gate's domain, not the path's — which once
        # vetoed every chunk near anything: 269 vetoes, 2026-07-23.)
        # selection key: (viable, PROGRESS toward the conditioned
        # carrot, clearance). Progress must dominate clearance — the
        # first key attempt (safe-length, clearance) had a degenerate
        # trap: a stand-still sample has no segments to test, scored
        # max clearance at full "safe length", and beat every real
        # path (live 2026-08-05: robot micro-rotating at 0.001 m/s
        # while staring at purple).
        cvec = np.asarray(cond[:2], dtype=float)
        cnorm = float(np.hypot(*cvec))
        cu = cvec / cnorm if cnorm > 1e-6 else None
        best_key, best_wp, best_n, worst_safe = None, None, 0, None
        for kk in range(K):
            wp_k = cands[kk]
            n_safe = len(wp_k)
            min_clear = 1e3
            if len(pts):
                prev = np.zeros(2)
                for i, wp in enumerate(wp_k):
                    seg = wp - prev
                    seg_l2 = float(seg @ seg)
                    if seg_l2 > 1e-8:
                        t = np.clip((pts - prev) @ seg / seg_l2, 0.0, 1.0)
                        proj = prev + t[:, None] * seg
                        dmin = float(np.min(np.hypot(*(pts - proj).T)))
                        if dmin < 0.13:
                            n_safe = i
                            break
                        min_clear = min(min_clear, dmin)
                    prev = wp
            last = wp_k[n_safe - 1] if n_safe > 0 else np.zeros(2)
            prog = float(last @ cu) if cu is not None \
                else float(np.hypot(*last))
            # R9 start-anchor: a chunk
            # whose FIRST waypoint teleports away from the robot tracks
            # a path the robot isn't on — penalize the excess
            anchor_pen = max(0.0, float(np.hypot(*wp_k[0])) - 0.30)
            # testing:
            # straightness + mode-consistency as a tier BETWEEN progress
            # and clearance. Straightness (|Δheading| per meter) is
            # CLEARANCE-GATED — full weight only above 0.45 m, zero by
            # 0.30 m: the literature data itself curves below that, and
            # straightening a thread is a collision generator.
            # Consistency penalizes initial headings that break from the
            # current motion direction (the inter-tick left/right
            # flip-flop that reads as weaving).
            segs = np.diff(np.vstack([[0.0, 0.0], wp_k[:n_safe]]), axis=0)
            seg_l = np.hypot(segs[:, 0], segs[:, 1])
            good = seg_l > 0.03
            turn_pm = 0.0
            if int(good.sum()) >= 2:
                hd = np.arctan2(segs[good][:, 1], segs[good][:, 0])
                dh = np.abs(np.arctan2(np.sin(np.diff(hd)),
                                       np.cos(np.diff(hd))))
                turn_pm = float(dh.sum() / max(float(seg_l[good].sum()),
                                               1e-6))
            mc_eff = min(min_clear, 1.0)
            w_s = min(max((mc_eff - 0.30) / 0.15, 0.0), 1.0)
            k2i = min(2, max(n_safe - 1, 0))
            init_dev = abs(math.atan2(wp_k[k2i][1], wp_k[k2i][0])) \
                if abs(self._odom.twist.twist.linear.x) > 0.05 else 0.0
            smooth_pen = w_s * 0.15 * turn_pm + 0.15 * init_dev
            key = (n_safe >= 2, round(prog - anchor_pen, 1),
                   round(-smooth_pen, 2), min_clear)
            if best_key is None or key > best_key:
                best_key, best_wp, best_n = key, wp_k[:n_safe], n_safe
                self._sel_diag = (turn_pm, mc_eff) # receipt columns
            if worst_safe is None or n_safe < worst_safe:
                worst_safe = n_safe
        n_safe = best_n
        wp_r = best_wp
        if n_safe < 2:
            self.get_logger().warning(
                f'waypoint veto: all {K} sampled paths run into an '
                'obstacle — holding for replan', throttle_duration_sec=2.0)
            self._last_tick_wall = time.monotonic()
            return
        if K > 1 and worst_safe < 2 <= n_safe:
            # audit signature: the selection just prevented a veto-hold
            self.get_logger().info(
                f'sample selection avoided a veto (best {n_safe}/'
                f'{len(cands[0])} waypoints safe, K={K})',
                throttle_duration_sec=5.0)
        wp_w = robot_to_world(wp_r, pose) # [H, 2] world
        # Short lookahead: the predicted path is only ~0.4 m long —
        # the literature's 0.25 m would aim near its end and cut the curve.
        self._tracker = PurePursuit(
            wp_w, goal_tol=0.03,
            lookahead=self.get_parameter('tracker_lookahead').value)
        self._publish_waypoints(wp_w, now)

        if self._lat_f:
            dg = getattr(self, '_sel_diag', (0.0, 0.0))
            self._lat_f.write(f'{now.nanoseconds*1e-9:.3f},'
                              f'{t_plan:.1f},{t_sample:.1f},'
                              f'{dg[0]:.2f},{dg[1]:.2f}\n')
            self._lat_f.flush()
        self._last_tick_wall = time.monotonic()


    # ── tracker: 20 Hz pure pursuit along the latest prediction ──────────────
    def _tracker_tick(self):
        if self._odom is None or self._done:
            return
        # sensor-age deadman: the tick's
        # early returns on missing sensors now read "live" to the
        # watchdog, so INPUT liveness needs its own check — driving on
        # a stale scan/odom is driving blind. 0.6 s at 5 Hz scans =
        # 3 missed frames.
        now_dm = time.monotonic()
        # executor-occupancy exemption deadman fires were self-inflicted — a
        # long denoise tick queues the sensor callbacks behind itself on the
        # single-threaded executor, then the deadman punishes the node for its
        # own compute). The exemption suppresses only the DEADMAN JUDGMENT
        # right after a tick ends — v1 of this returned from the WHOLE tracker
        # tick, and since ticks end every 0.25 s the robot never received a
        # command at all (live 2026-08-06, two dead launches: parked at spawn,
        # zero sightings).
        recent_tick = now_dm - getattr(self, '_tick_end_mono', 0.0) < 0.20
        if not recent_tick and (
                now_dm - getattr(self, '_scan_rx', now_dm) > 0.6 or
                now_dm - getattr(self, '_odom_rx', now_dm) > 0.6):
            self._cmd_pub.publish(Twist())
            self.get_logger().warning(
                'sensor deadman: scan/odom stale >0.6 s — wheels held',
                throttle_duration_sec=2.0)
            return
        if self._tracker is None:
            return
        x, y, yaw = self._world_pose()
        v, w, done = self._tracker.step(x, y, yaw)
        if done:
            self._cmd_pub.publish(Twist())
            return
        cap = getattr(self, '_seek_v_cap', None)
        if cap is not None:
            v = min(v, cap)
        # pocket exits crawl (2026-08-03 run 5): the escape's relaxed
        # margins let the gate DRIVE out of a pocket, but full tracker
        # speed clipped a cylinder 1.5 s after the escape engaged (2
        # contacts). Slow exit keeps the relaxed-margin pass usable
        # without the contact risk.
        if self._escaping():
            v = min(v, 0.08)
        v_gated = self._gate_forward_speed(v, w)
        if self._gate and self._ranges_m is not None:
            recovery_cmd = self._gate_recovery(v, v_gated)
            if recovery_cmd is not None:
                self._last_cmd_v = recovery_cmd.linear.x
                self._cmd_pub.publish(recovery_cmd)
                return
        cmd = Twist()
        cmd.linear.x = v_gated
        cmd.angular.z = w
        self._last_cmd_v = v_gated
        self._cmd_pub.publish(cmd)


    def _publish_waypoints(self, wp_w, now):
        msg = PathMsg()
        msg.header.frame_id = 'odom'
        msg.header.stamp = now.to_msg()
        for x, y in self._to_odom(wp_w):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._wp_pub.publish(msg)


    def _publish_subgoal(self, sub_w, now):
        x, y = self._to_odom(np.array(sub_w))
        ps = PoseStamped()
        ps.header.frame_id = 'odom'
        ps.header.stamp = now.to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.w = 1.0
        self._sub_pub.publish(ps)
