-- Cartographer config for TurtleBot3 Burger — LiDAR + BRIO camera FUSED SLAM.
--
-- Same as turtlebot3_camera_slam.lua but with num_laser_scans = 2:
--   scan_1 -> /scan_relayed  (LiDAR, step-immune wall-clock stamps)
--   scan_2 -> /scan_camera   (floor-contact virtual scan from the BRIO)
--
-- The bug #013 blockers are resolved as of 2026-07-08:
--   * real intrinsics (fx=fy=464.7 from ~/.ros/camera_info on the robot,
--     was a fx=500 placeholder),
--   * level mount (tilt 0, was 8° down) with lidar-referee height
--     calibration (see camera_obstacle_params.yaml),
--   * edge-free columns publish inf, which the cartographer sensor bridge
--     drops entirely (no free-space carving through walls).
--
-- The camera contributes obstacles the LiDAR plane cannot see (couch
-- overhangs, low clutter), so the occupancy grid stops reporting the space
-- under furniture as known-free.
--
-- num_accumulated_range_data = 2 is load-bearing: with two scan topics and
-- the default of 1, every sparse camera scan (often 1-11 points) becomes
-- its own trajectory node, and the constraint builder matches those few
-- points anywhere with ~90% scores — bogus loop closures translated up to
-- ~10 m tore the 2026-07-12 live map into islands. Accumulating 2 range
-- datas per node pairs each camera scan with a ~200-point LiDAR scan, so
-- no camera-only node ever reaches scan matching or loop closure.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_link",
  published_frame = "odom",
  odom_frame = "odom",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = true,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 2,              -- LiDAR + camera virtual scan
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_2d = true

TRAJECTORY_BUILDER_2D.min_range = 0.05           -- camera can detect closer than LiDAR
TRAJECTORY_BUILDER_2D.max_range = 3.5
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 2   -- 1 LiDAR + 1 camera scan per node (see header)
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3.
TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.1)

POSE_GRAPH.constraint_builder.min_score = 0.65
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.7

return options
