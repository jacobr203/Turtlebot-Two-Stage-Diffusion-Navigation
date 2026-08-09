# DWA Camera Avoidance

Extends `dwa_avoidance` with monocular-camera obstacle detection for the
TurtleBot3 Burger. A Logitech BRIO on the robot streams JPEG-compressed
images; `camera_obstacle_node` converts them into a virtual `LaserScan`
(`/scan_camera`), and `dwa_camera_node` fuses those obstacles with the LiDAR
for Dynamic Window Approach avoidance — catching low obstacles under the
LiDAR scan plane (~0.17 m) and overhanging ones above it.

## Nodes

### `camera_obstacle_node`

Subscribes `/image_raw/compressed` + `/camera_info`, runs Canny edge
detection in a floor zone and an overhead zone, back-projects edges to
obstacle points, and publishes `/scan_camera` (~30 Hz, frame `base_scan`,
69° FOV). Parameters in `config/camera_obstacle_params.yaml` (camera mount
geometry, detection zones, thresholds).

Stamps are **step-immune**: the WSL2 wall clock steps backwards ~1 s every
~30 s under time sync (bug #015), so `_next_stamp()` follows the wall clock
while it is ahead and otherwise advances at the monotonic clock's rate.

### `dwa_camera_node`

The DWA controller from `dwa_avoidance` plus:

- **Merged obstacles** — LiDAR points + camera points (camera entries expire
  after `camera_obstacle_max_age` = 0.5 s so a frozen frame can't leave ghosts).
- **Wheel-slip detection** — if odometry claims ≥ 8 cm of travel over 1.5 s
  while the LiDAR scan stays frozen (per-ray change far below what the
  claimed motion predicts), the robot is physically stuck — e.g. pushing a
  smooth chair base both sensors miss (bug #014). The check compares actual
  vs *expected* per-ray change and abstains in open space rather than guess.
- **Safety-bubble retreat** — when an obstacle is already inside the
  collision bubble (robot_radius + margin), trajectories that *increase*
  clearance stay viable instead of everything being rejected (the pre-fix
  behavior deadlocked into permanent rotation against couches/walls).
- **Nav2-style recovery behaviors** — stuck triggers run BackUp (rear
  checked; Forward escape if the rear is blocked) → Spin toward the open
  side → Wait → fresh auto-goal. Re-triggering within 30 s escalates
  distance and angle (0.20→0.40 m, 90°→180°).

## SLAM integration

`launch/dwa_camera_with_cartographer.launch.py` runs the full stack: relays,
camera node, Cartographer, occupancy grid, DWA (and RViz unless
`use_rviz:=false`).

Cartographer is **LiDAR-only** (`config/turtlebot3_camera_slam.lua`,
`num_laser_scans = 1`). Feeding the camera's virtual scan to SLAM made the
pose diverge (bug #013): approximate intrinsics bias the ranges, 30 Hz
camera vs 9 Hz LiDAR outvotes the real sensor, and edge-free columns carve
free space through real walls. The camera scan is used for *avoidance only*
until the BRIO has a real calibration.

## Running

The simplest way is the unified runner (handles robot bringup / Gazebo,
cleanup, and the right launch args):

```bash
~/pathfinding_algorithms/run.sh camera            # real robot
~/pathfinding_algorithms/run.sh camera --sim      # Gazebo (turtlebot3_world)
```

Manually, on the real robot:

```bash
# On the robot: bringup + camera
ros2 launch turtlebot3_bringup robot.launch.py
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:=/dev/video0 -p image_size:=[640,480]

# On the PC
cd ~/pathfinding_algorithms && source install/setup.bash
ros2 launch dwa_camera_avoidance dwa_camera_with_cartographer.launch.py
```

## Gazebo simulation

The package ships `models/turtlebot3_burger_brio` — a standard Burger plus
a **pinhole** camera at the real BRIO mount (x 0.05 m, z 0.12 m, 8° down,
69° HFOV, 640×480 @ 30 Hz), so `camera_obstacle_params.yaml` geometry holds
in sim and the sim intrinsics (fx≈465.6) nearly match the real BRIO
(fx≈464.7). The upstream `turtlebot3_burger_cam` model was unusable here:
its camera is a 182° fisheye, incompatible with pinhole back-projection.

```bash
# world + robot (gzclient GUI unless headless:=true)
ros2 launch dwa_camera_avoidance sim_world_brio.launch.py [world:=turtlebot3_house] [headless:=true]

# algorithm stack on top — note the sim camera topics
ros2 launch dwa_camera_avoidance dwa_camera_with_cartographer.launch.py \
  use_sim_time:=true \
  image_topic:=/camera/image_raw/compressed \
  camera_info_topic:=/camera/camera_info
```

Sim-time note: the step-immune stamping (bug #015) coasts on the monotonic
clock, which is wrong under sim time — with `use_sim_time` the nodes switch
to the node clock with a strictly-monotonic guard automatically.

For the edit → rebuild → relaunch dev loop and the live browser viewer, see
[`../dev_tools/README.md`](../dev_tools/README.md).

## Tests

```bash
cd ~/pathfinding_algorithms
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 -m pytest dwa_camera_avoidance/test/ -v
```

Tests cover the slip detector (detection, motion-consistency,
open-space abstention, turn/command gating), retreat clearance, the
recovery state machine (stage selection, full sequence, escalation,
forward-escape), rear-clearance geometry in both scan conventions, and
stamping (step-immune wall-clock behavior plus sim-time tracking).
Gazebo integration smoke tests live in `~/unit_tests/gazebo_sim/`.
