"""Where to stand to actually SEE a target.

Instead of walking a fixed ring around a colour estimate, candidate
standoff poses are scored by how much they reveal, divided by what they
cost to reach. Also holds the corridor memory (which failed approaches
to avoid repeating) and the visibility raycast that ranks exploration
targets.

Reference:
    Gonzalez-Banos, H. H. and Latombe, J.-C. (2002), "Navigation
    Strategies for Exploring Indoor Environments", International
    Journal of Robotics Research 21(10-11):829-848. Next-best-view
    scoring by expected information gain against travel cost.

Mixin composed into DiffusionPolicyNode; `self` is the node.
"""

import math
import time

import numpy as np

# nav_msgs/OccupancyGrid encoding
UNKNOWN_CELL = -1
FREE_CELL = 0


class ViewpointMixin:
    """Scores standoff poses by what they reveal, not by ring geometry."""

    # --- visibility raycast -------------------------------------------
    VIS_RANGE_M = 3.5          # credited sensing range (LDS-02 usable)
    VIS_RAYS = 24              # rays per candidate; coarse on purpose
    VIS_STEP_FRACTION = 2.0    # ray step = resolution / this
    VIS_MIN_STEPS = 4          # floor on samples per ray

    # --- vantage admission --------------------------------------------
    VANTAGE_MIN_MOVE_M = 0.6   # a vantage must relocate the camera
    VANTAGE_RING_RADII_M = (0.8, 1.2, 1.6)   # standoff distances tried
    VANTAGE_RING_STEP_DEG = 30                # bearing spacing on a ring
    VANTAGE_SCAN_BUDGET_S = 0.6               # scan deadline per tick
    VANTAGE_COOL_S = 90.0                     # re-pick cooldown
    OCCUPIED_THRESHOLD = 50    # occupancy grid value above = obstacle

    # --- corridor memory ----------------------------------------------
    CORRIDOR_MIN_LEN_M = 1.0   # shorter failures are local wedges
    CORRIDOR_COOL_S = 150.0    # how long a failed corridor stays dead
    CORRIDOR_MATCH_RADIUS_M = 0.9    # midpoint proximity for a match
    CORRIDOR_MATCH_ANGLE_RAD = 0.7   # bearing tolerance for a match

    def _visibility_gain(self, grid, info, candidates):
        """Count how much unseen area each candidate pose would reveal.

        Fires a fan of rays from each candidate and counts the unknown
        cells reached before any obstacle blocks the way. Standing next
        to a large unknown area is worth nothing if a wall hides it;
        this measures what would actually come into view.

        candidates: (N, 2) array of map-frame x, y.
        Returns an (N,) array of visible-unknown cell counts.

        Vectorized over candidates and rays at once, so a 40-candidate
        scan costs well under a millisecond and is safe inside a control
        tick.
        """
        resolution = info.resolution
        height, width = info.height, info.width
        steps_per_ray = max(self.VIS_MIN_STEPS,
                            int(self.VIS_RANGE_M
                                / (self.VIS_STEP_FRACTION * resolution)))
        bearings = np.linspace(0.0, 2.0 * math.pi, self.VIS_RAYS,
                               endpoint=False)
        distances = np.linspace(0.0, self.VIS_RANGE_M,
                                steps_per_ray + 1)[1:]

        # (rays, steps) sample offsets, broadcast against (N, 1, 1) origins
        offset_x = np.cos(bearings)[:, None] * distances[None, :]
        offset_y = np.sin(bearings)[:, None] * distances[None, :]
        sample_x = candidates[:, 0][:, None, None] + offset_x[None, :, :]
        sample_y = candidates[:, 1][:, None, None] + offset_y[None, :, :]

        col = ((sample_x - info.origin.position.x)
               / resolution).astype(np.int32)
        row = ((sample_y - info.origin.position.y)
               / resolution).astype(np.int32)
        in_bounds = ((col >= 0) & (col < width)
                     & (row >= 0) & (row < height))
        np.clip(col, 0, width - 1, out=col)
        np.clip(row, 0, height - 1, out=row)

        cells = grid[row, col]
        blocked = (cells > self.OCCUPIED_THRESHOLD) | ~in_bounds
        # A sample counts only if nothing blocks the ray before it, so
        # the running total of blockers along each ray must still be 0.
        unobstructed = np.cumsum(blocked, axis=2) == 0
        is_unknown = cells == UNKNOWN_CELL
        return (is_unknown & unobstructed).sum(axis=(1, 2)).astype(np.float64)


    def _corridor_kill(self, start_xy, goal_xy, cool_s=None):
        """Remember that this route failed, so we stop retrying it.

        Cooling only the destination is not enough: the next scan just
        picks a neighbouring pose behind the same blocked gap and drives
        the identical route again. Recording the whole approach as a
        (midpoint, bearing) sector rejects those neighbours too.

        Very short failures are ignored — they mean the robot is wedged
        against something right here, not that a route is impassable,
        and blacklisting a sector centred on the robot would reject
        every direction at once.

        Returns True if a corridor was recorded.
        """
        length = math.hypot(goal_xy[0] - start_xy[0],
                            goal_xy[1] - start_xy[1])
        if length < self.CORRIDOR_MIN_LEN_M:
            return False

        cool_s = self.CORRIDOR_COOL_S if cool_s is None else cool_s
        midpoint_x = (start_xy[0] + goal_xy[0]) * 0.5
        midpoint_y = (start_xy[1] + goal_xy[1]) * 0.5
        bearing = math.atan2(goal_xy[1] - start_xy[1],
                             goal_xy[0] - start_xy[0])

        now = time.time()
        self._dead_corridors = [c for c in self._live_corridors()
                                if c[3] > now]
        self._dead_corridors.append(
            (midpoint_x, midpoint_y, bearing, now + cool_s))
        return True

    def _live_corridors(self):
        """Dead corridors that have not expired yet."""
        now = time.time()
        return [c for c in getattr(self, '_dead_corridors', [])
                if c[3] > now]

    def _corridor_dead(self, start_xy, goal_xy):
        """True if this route repeats one that recently failed.

        Both the midpoint and the heading must match: a pose reached
        from the far side of the same obstacle is a genuinely different
        route and stays available.
        """
        dead = self._live_corridors()
        self._dead_corridors = dead
        if not dead:
            return False

        midpoint_x = (start_xy[0] + goal_xy[0]) * 0.5
        midpoint_y = (start_xy[1] + goal_xy[1]) * 0.5
        bearing = math.atan2(goal_xy[1] - start_xy[1],
                             goal_xy[0] - start_xy[0])

        for dead_x, dead_y, dead_bearing, _ in dead:
            if (math.hypot(midpoint_x - dead_x, midpoint_y - dead_y)
                    > self.CORRIDOR_MATCH_RADIUS_M):
                continue
            offset = abs(math.atan2(math.sin(bearing - dead_bearing),
                                    math.cos(bearing - dead_bearing)))
            if offset <= self.CORRIDOR_MATCH_ANGLE_RAD:
                return True
        return False


    def _visible_viewpoint(self, robot_map_xy, target_x, target_y):
        """Pick the best pose to stand in to see a target.

        Rings the target estimate with candidate standoffs and keeps the
        cheapest one that is free space, has line of sight to the
        target, is reachable, and is far enough away to be worth
        driving to.

        Ranking is by A* PATH length, not straight-line distance: a
        ruler makes poses on the far side of an obstacle look cheap and
        keeps re-picking the side that just failed.

        Known-free cells outrank unknown ones, but unknown is still
        allowed. Demanding known-free biases the search toward already
        explored - and therefore already unsuccessful - ground, and
        returns nothing precisely when a new vantage is most needed.

        Returns (x, y) in the map frame, or None if nothing qualifies.
        """
        grid_msg = self._map
        if grid_msg is None:
            return None

        resolution = grid_msg.info.resolution
        height, width = grid_msg.info.height, grid_msg.info.width
        grid = np.asarray(grid_msg.data, dtype=np.int8).reshape(height, width)
        origin_x = grid_msg.info.origin.position.x
        origin_y = grid_msg.info.origin.position.y

        now = time.time()
        self._vantage_cooldowns = {
            pose: expiry for pose, expiry
            in getattr(self, '_vantage_cooldowns', {}).items()
            if expiry > now}

        rejected = {'out_of_bounds': 0, 'occupied': 0, 'cooling': 0,
                    'no_line_of_sight': 0, 'unreachable': 0,
                    'failed_route': 0, 'too_close': 0}
        best = None                     # (free_rank, path_len, x, y)
        scan_started = time.time()

        for radius in self.VANTAGE_RING_RADII_M:
            for degrees in range(0, 360, self.VANTAGE_RING_STEP_DEG):
                if (time.time() - scan_started
                        > self.VANTAGE_SCAN_BUDGET_S):
                    break               # out of time; use the best so far
                bearing = math.radians(degrees)
                cand_x = target_x + radius * math.cos(bearing)
                cand_y = target_y + radius * math.sin(bearing)

                col = int((cand_x - origin_x) / resolution)
                row = int((cand_y - origin_y) / resolution)
                if not (0 <= col < width and 0 <= row < height):
                    rejected['out_of_bounds'] += 1
                    continue

                cell = grid[row, col]
                if cell > self.OCCUPIED_THRESHOLD:
                    rejected['occupied'] += 1
                    continue
                free_rank = 0 if cell == FREE_CELL else 1

                cooldown_key = (round(cand_x, 1), round(cand_y, 1))
                if self._vantage_cooldowns.get(cooldown_key, 0.0) > now:
                    rejected['cooling'] += 1
                    continue

                if not self._los_clear(grid_msg, cand_x, cand_y,
                                       target_x, target_y):
                    rejected['no_line_of_sight'] += 1
                    continue

                if self._corridor_dead(tuple(robot_map_xy),
                                       (cand_x, cand_y)):
                    rejected['failed_route'] += 1
                    continue

                clearance, path_len = self.live_subgoaler.route_cost(
                    tuple(robot_map_xy), (cand_x, cand_y))
                if clearance is None:
                    rejected['unreachable'] += 1
                    continue

                if path_len is None:
                    path_len = math.hypot(cand_x - robot_map_xy[0],
                                          cand_y - robot_map_xy[1])
                # A pose we are already standing on offers no new angle,
                # so driving to it would only stall the approach.
                if path_len < self.VANTAGE_MIN_MOVE_M:
                    rejected['too_close'] += 1
                    continue

                score = (free_rank, path_len)
                if best is None or score < best[:2]:
                    best = (free_rank, path_len, cand_x, cand_y)

            if time.time() - scan_started > self.VANTAGE_SCAN_BUDGET_S:
                break
        if best is None:
            self.get_logger().info(
                'viewpoint scan found none — rejects: '
                f"{rejected['occupied']} occupied, "
                f"{rejected['no_line_of_sight']} occluded, "
                f"{rejected['unreachable']} unroutable, "
                f"{rejected['cooling']} cooled, "
                f"{rejected['failed_route']} dead-corridor, "
                f"{rejected['too_close']} too-close, "
                f"{rejected['out_of_bounds']} out-of-bounds",
                throttle_duration_sec=15.0)
            return None

        free_rank, path_len, best_x, best_y = best
        # Cool the chosen pose so a fruitless visit is not repeated
        # forever. A cooldown rather than a permanent ban: mapping can
        # open up a pose that looks useless now.
        self._vantage_cooldowns[(round(best_x, 1), round(best_y, 1))] = \
            now + self.VANTAGE_COOL_S
        self.get_logger().info(
            f'viewpoint scored: map({best_x:.1f},{best_y:.1f}) '
            f"{'free' if free_rank == 0 else 'unknown'} cell, path "
            f'{path_len:.1f} m, LOS clear '
            f"(rejected {rejected['no_line_of_sight']} occluded, "
            f"{rejected['unreachable']} unroutable, "
            f"{rejected['failed_route']} dead-corridor)")
        return (best_x, best_y)
