"""Frame conversions and line-of-sight tests.

Three frames are in play:

    world   fixed at the spawn pose; what the policy is conditioned in
    odom    wheel odometry, drifts slowly, re-anchored at spawn
    map     the SLAM frame the occupancy grid lives in

World and odom differ by the spawn anchor; odom and map differ by the
SLAM correction, which is only available once TF publishes it. Anything
returning None here means that transform is not up yet.

Also stamps the coverage memory: every map pose the robot computes
marks where it has been, which the exploration scorers use to prefer
unvisited ground.

Mixin composed into DiffusionPolicyNode; `self` is the node.
"""

import math
import time

import numpy as np
import rclpy

from common.geometry import robot_to_world, world_to_robot, yaw_from_quaternion

VISITED_CELL_M = 0.5       # fine grid marking ground already stood on
VISIT_COUNT_CELL_M = 1.5   # coarse grid counting arena-scale revisits
VISIT_COUNT_MIN_GAP_S = 1.0    # do not count the same cell every tick
OCCUPIED_THRESHOLD = 50    # occupancy grid value above this = obstacle
RAY_STEP_FRACTION = 0.6    # ray sample spacing, in grid cells
RAY_MIN_SAMPLES = 2


class GeometryMixin:
    """Converts between frames and answers "can I see from here to there"."""

    def _map_T_odom(self):
        """Pose of the odom frame within the map frame.

        Returns (x, y, yaw), or None if SLAM has not published the
        transform yet.
        """
        try:
            transform = self._tf_buffer.lookup_transform(
                'map', 'odom', rclpy.time.Time())
        except Exception:      # noqa: BLE001 — no map TF yet
            return None
        rotation = transform.transform.rotation
        return (transform.transform.translation.x,
                transform.transform.translation.y,
                yaw_from_quaternion(rotation.x, rotation.y,
                                    rotation.z, rotation.w))

    def _world_pose(self):
        """Current robot pose in the world frame as (x, y, yaw)."""
        odom_pose = self._odom.pose.pose
        odom_yaw = yaw_from_quaternion(odom_pose.orientation.x,
                                       odom_pose.orientation.y,
                                       odom_pose.orientation.z,
                                       odom_pose.orientation.w)
        anchor_x, anchor_y, anchor_yaw = self.anchor
        cos_a, sin_a = math.cos(anchor_yaw), math.sin(anchor_yaw)
        return (anchor_x + cos_a * odom_pose.position.x
                - sin_a * odom_pose.position.y,
                anchor_y + sin_a * odom_pose.position.x
                + cos_a * odom_pose.position.y,
                anchor_yaw + odom_yaw)

    def _world_to_map_xy(self, world_x, world_y):
        """World point to map frame. None if the map transform is absent."""
        odom_in_map = self._map_T_odom()
        if odom_in_map is None:
            return None
        anchor = self.anchor
        in_odom = world_to_robot(np.array([world_x, world_y]), anchor)
        map_x, map_y = robot_to_world(in_odom, odom_in_map)
        return (float(map_x), float(map_y))

    def _map_to_world(self, map_xy):
        """Map point back to the world frame. None without the transform."""
        odom_in_map = self._map_T_odom()
        if odom_in_map is None:
            return None
        in_odom = world_to_robot(np.array(map_xy), odom_in_map)
        world = robot_to_world(in_odom, self.anchor)
        return (float(world[0]), float(world[1]))

    def _los_clear(self, grid_msg, from_x, from_y, to_x, to_y,
                   stop_short_cells=1.6):
        """True if nothing blocks the straight line between two map points.

        Stops slightly short of the destination because a target sits on
        its own wall, which would otherwise always read as blocked.
        """
        span = math.hypot(to_x - from_x, to_y - from_y)
        resolution = grid_msg.info.resolution
        walk_len = span - stop_short_cells * resolution
        if walk_len <= 0:
            return True

        height, width = grid_msg.info.height, grid_msg.info.width
        grid = np.asarray(grid_msg.data, dtype=np.int8).reshape(height, width)
        samples = max(RAY_MIN_SAMPLES,
                      int(walk_len / (RAY_STEP_FRACTION * resolution)))
        fractions = np.linspace(0.0, walk_len / span, samples)[1:]

        col = ((from_x + (to_x - from_x) * fractions
                - grid_msg.info.origin.position.x) / resolution).astype(int)
        row = ((from_y + (to_y - from_y) * fractions
                - grid_msg.info.origin.position.y) / resolution).astype(int)
        in_bounds = ((col >= 0) & (col < width)
                     & (row >= 0) & (row < height))
        return not bool((grid[row[in_bounds], col[in_bounds]]
                         > OCCUPIED_THRESHOLD).any())

    def _map_pose(self, pose):
        """Robot pose in the map frame, recording where it has been.

        Returns (x, y, yaw), or None without the map transform. Also
        stamps two coverage grids: a fine one marking ground already
        stood on, and a coarse one counting revisits. Exploration
        scoring uses both to prefer unfamiliar ground, which stops goal
        selection from recycling the same explored strip.
        """
        odom_in_map = self._map_T_odom()
        if odom_in_map is None:
            return None
        anchor_x, anchor_y, anchor_yaw = self.anchor
        in_odom = world_to_robot(np.array(pose[:2]),
                                 (anchor_x, anchor_y, anchor_yaw))
        map_x, map_y = robot_to_world(in_odom, odom_in_map)

        self._visited_cells.add((int(math.floor(map_x / VISITED_CELL_M)),
                                 int(math.floor(map_y / VISITED_CELL_M))))

        now = time.monotonic()
        if now - getattr(self, '_visit_t', 0.0) > VISIT_COUNT_MIN_GAP_S:
            self._visit_t = now
            visit_counts = getattr(self, '_visit_grid', None)
            if visit_counts is None:
                visit_counts = self._visit_grid = {}
            cell = (int(math.floor(map_x / VISIT_COUNT_CELL_M)),
                    int(math.floor(map_y / VISIT_COUNT_CELL_M)))
            visit_counts[cell] = visit_counts.get(cell, 0) + 1

        return (float(map_x), float(map_y),
                pose[2] - anchor_yaw + odom_in_map[2])

    def _to_odom(self, points):
        """World points to odom frame, for RViz debug topics."""
        return world_to_robot(points, self.anchor)
