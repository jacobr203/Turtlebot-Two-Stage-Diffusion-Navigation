# DWA Avoidance

A ROS 2 (Humble) implementation of the **Dynamic Window Approach (DWA)** obstacle avoidance algorithm for the TurtleBot3 Burger. The node reads LiDAR data from `/scan`, computes safe velocity commands, and publishes them to `/cmd_vel`. It is designed to run alongside **Cartographer SLAM** so the robot autonomously explores and builds a map while avoiding obstacles.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Building](#building)
- [Running](#running)
  - [Real Robot](#real-robot)
  - [Gazebo Simulation](#gazebo-simulation)
- [Sending a Goal](#sending-a-goal)
- [Parameters](#parameters)
- [ROS Topics](#ros-topics)
- [Tests](#tests)
- [Package Layout](#package-layout)

---

## How It Works

At each control tick (100 ms) the node:

1. **Builds a Dynamic Window** — the set of `(v, ω)` velocity pairs reachable within one timestep given the robot's current speed and its acceleration limits.
2. **Simulates trajectories** — for every sampled `(v, ω)` pair, the robot's path is simulated 2 seconds into the future.
3. **Scores each trajectory** using three weighted terms from the original DWA paper (Fox et al., 1997):
   - **Clearance** (weight 1.0) — minimum distance to any obstacle along the path. Trajectories that collide are discarded entirely.
   - **Speed** (weight 0.5) — rewards faster forward motion to keep the robot moving.
   - **Heading** (weight 0.2) — rewards trajectories that point toward the goal.
4. **Publishes the best `(v, ω)`** to `/cmd_vel`.

**Wander mode** (no goal set): a virtual goal is placed 2 m ahead of the robot so it keeps exploring forward while still avoiding obstacles — useful for building a Cartographer map without a pre-planned route.

**Goal mode**: publish a `geometry_msgs/PoseStamped` to `/goal_pose` and the robot navigates there, stopping within 0.25 m.

---

## Prerequisites

All of the following must be installed on the machine running the ROS 2 stack (your laptop/desktop, not the robot):

| Requirement | Version | Notes |
|---|---|---|
| Ubuntu | 22.04 LTS | Other versions untested |
| ROS 2 | Humble Hawksbill | [Install guide](https://docs.ros.org/en/humble/Installation.html) |
| TurtleBot3 packages | Humble | See installation step below |
| Cartographer ROS | Humble | Pulled in by TurtleBot3 packages |
| Python | 3.10+ | Ships with Ubuntu 22.04 |
| NumPy | any | `sudo apt install python3-numpy` |

For the **real robot**, the TurtleBot3 Burger must already be running its bringup (see [Real Robot](#real-robot)).

---

## Installation

Open a terminal and run these commands from top to bottom. Each block only needs to be run once.

### 1. Source ROS 2

```bash
source /opt/ros/humble/setup.bash
```

Add this to your `~/.bashrc` so you don't need to repeat it:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### 2. Install TurtleBot3 packages

```bash
sudo apt update
sudo apt install -y \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-simulations \
    ros-humble-cartographer-ros
```

### 3. Set the TurtleBot3 model

```bash
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
```

### 4. Clone this repo

```bash
cd ~
git clone <this-repo-url> pathfinding_algorithms
```

The repo root is itself a colcon workspace — no separate `~/colcon_ws` is needed.

### 5. Install ROS dependencies

```bash
cd ~/pathfinding_algorithms
rosdep install -r --from-paths dwa_avoidance --ignore-src -y --rosdistro humble
```

---

## Building

```bash
cd ~/pathfinding_algorithms
source /opt/ros/humble/setup.bash
colcon build --packages-select dwa_avoidance --symlink-install
source install/setup.bash
```

> **Shortcut:** `~/pathfinding_algorithms/run.sh lidar [--sim]` handles building,
> sourcing, robot/world bringup, and cleanup automatically — the manual steps
> below are only needed if you want to run the pieces individually.

`--symlink-install` means edits to Python source files take effect immediately without a rebuild.

---

## Running

### Real Robot

You need **three terminals**. Source your workspace in each one before running any command:

```bash
source /opt/ros/humble/setup.bash && source ~/pathfinding_algorithms/install/setup.bash
```

**Terminal 1 — Robot bringup** (SSH into the TurtleBot3 and run this on the robot):

```bash
ros2 launch turtlebot3_bringup robot.launch.py
```

**Terminal 2 — DWA + Cartographer** (run on your machine):

```bash
source /opt/ros/humble/setup.bash && source ~/pathfinding_algorithms/install/setup.bash
ros2 launch dwa_avoidance dwa_with_cartographer.launch.py
```

This starts Cartographer SLAM and the DWA avoidance node together. RViz2 opens automatically showing the live map.

**Terminal 3 — Save the map when done** (optional):

```bash
source /opt/ros/humble/setup.bash
ros2 run nav2_map_server map_saver_cli -f ~/my_map
```

---

### Gazebo Simulation

You need **two terminals**.

**Terminal 1 — Gazebo world**:

```bash
source /opt/ros/humble/setup.bash && source ~/pathfinding_algorithms/install/setup.bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Wait for the simulation to fully load before starting Terminal 2.

**Terminal 2 — DWA + Cartographer**:

```bash
source /opt/ros/humble/setup.bash && source ~/pathfinding_algorithms/install/setup.bash
ros2 launch dwa_avoidance dwa_with_cartographer.launch.py use_sim_time:=true
```

The robot will begin wandering and building a map immediately. You can watch the map grow in the RViz2 window that opens automatically.

---

## Sending a Goal

While the node is running, publish a goal from a third terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 1.5, y: 0.5, z: 0.0}, orientation: {w: 1.0}}}"
```

The robot will navigate toward that point while avoiding obstacles. Once within 0.25 m it stops. Send a new goal to continue.

Remove the goal and return to wander mode by restarting the DWA node.

---

## Parameters

All parameters can be overridden at launch via the command line or by editing `config/dwa_params.yaml`.

| Parameter | Default | Unit | Source |
|---|---|---|---|
| `max_speed` | 0.22 | m/s | TurtleBot3 e-Manual — "Maximum translational velocity"; confirmed in `turtlebot3_navigation2/param/humble/burger.yaml` → `max_vel_x` |
| `min_speed` | -0.05 | m/s | Nav2 DWB sets `min_vel_x: 0.0`; small reverse added here for recovery manoeuvres |
| `max_yaw_rate` | 2.84 | rad/s | e-Manual — "Maximum rotational velocity"; derived from wheel geometry: `2 × 0.22 / 0.160 m ≈ 2.75`, ROBOTIS rounds to 2.84 |
| `max_accel` | 2.5 | m/s² | `burger.yaml` DWB block → `acc_lim_x: 2.5` |
| `max_delta_yaw_rate` | 3.2 | rad/s² | `burger.yaml` DWB block → `acc_lim_theta: 3.2`. The ROS 1 DWA planner used 17.45 rad/s² (raw XL430 torque figure); Nav2 reduced it to 3.2 for smoother motion |
| `v_resolution` | 0.01 | m/s | Sampling step for linear velocity |
| `yaw_rate_resolution` | 0.1 | rad/s | Sampling step for angular velocity |
| `dt` | 0.1 | s | Control loop period and simulation step size |
| `predict_time` | 2.0 | s | Lookahead horizon per trajectory. Nav2 DWB uses `sim_time: 1.5` |
| `robot_radius` | 0.15 | m | Nav2 costmap sets `robot_radius: 0.1 m` (actual body). Bumped to 0.15 m to buffer LiDAR noise and odometry drift during SLAM |
| `obstacle_margin` | 0.05 | m | Extra clearance on top of `robot_radius` → 0.20 m total minimum gap |
| `w_heading` | 0.20 | — | Cost weight: alignment with goal direction |
| `w_clearance` | 1.00 | — | Cost weight: minimum obstacle distance (highest to prioritise safety) |
| `w_speed` | 0.50 | — | Cost weight: forward speed |
| `goal_tolerance` | 0.25 | m | Stop when within this distance. Matches Nav2 `xy_goal_tolerance: 0.25` |
| `wander_lookahead` | 2.0 | m | Distance of virtual goal in wander mode |

Override a parameter at launch:

```bash
ros2 launch dwa_avoidance dwa_with_cartographer.launch.py \
  --ros-args -p max_speed:=0.15
```

---

## ROS Topics

| Topic | Type | Direction | Purpose |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | Subscribed | LiDAR data used to detect obstacles |
| `/odom` | `nav_msgs/Odometry` | Subscribed | Robot pose and current velocity |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Subscribed | Optional navigation goal |
| `/cmd_vel` | `geometry_msgs/Twist` | Published | Velocity commands sent to the robot |

---

## Relay Nodes

Two support nodes restamp robot data with the PC wall clock (the OpenCR MCU
stamps everything with its own uptime, which Cartographer and RViz reject):

- **`scan_relay_node`** — republishes `/scan` as `/scan_relayed` (RELIABLE QoS).
  Consecutive stamps are spaced at least one sweep duration apart so
  Cartographer's per-scan subdivision windows never overlap.
- **`tf_relay_node`** — replaces MCU-uptime stamps on `/tf` transforms.
- **`odom_relay_node`** — same treatment for `/odom`.

Both stampers are **step-immune**: the WSL2 wall clock is stepped backwards
~1 s every ~30 s by time sync (bug #015), so stamps follow the wall clock
while it is ahead but otherwise advance at the monotonic clock's rate —
`max(wall_now, last_stamp + monotonic_elapsed)` — and never run backwards.

---

## Gazebo simulation

`launch/sim_world.launch.py` brings up Gazebo Classic with a TurtleBot3:
gzserver + optional gzclient (`headless:=true`), `robot_state_publisher`,
and a spawned model (defaults to the standard Burger; `model_sdf` /
`urdf_file` args let other packages substitute their own — see
`dwa_camera_avoidance`'s `sim_world_brio.launch.py`). Worlds:
`turtlebot3_world` (default), `turtlebot3_house`, `empty_world`.

```bash
ros2 launch dwa_avoidance sim_world.launch.py
ros2 launch dwa_avoidance dwa_with_cartographer.launch.py use_sim_time:=true
```

Or in one step with the unified runner: `~/pathfinding_algorithms/run.sh lidar --sim`.

---

## Tests

`test_dwa_node.py` verifies node startup, all ROS interfaces, every default
parameter, the yaw quaternion helper, all callbacks, each scoring function,
and that the DWA output always stays within kinematic limits.
`test_relay_nodes.py` verifies the relays' stamp spacing, stale-TF detection,
and that stamps keep increasing across simulated backward clock steps.

```bash
cd ~/pathfinding_algorithms
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 -m pytest dwa_avoidance/test/ -v
```

Expected output: **37 passed**.

---

## Package Layout

```
dwa_avoidance/
├── dwa_avoidance/
│   ├── __init__.py
│   ├── dwa_node.py          # DWA algorithm and ROS 2 node
│   ├── scan_relay_node.py   # /scan → /scan_relayed wall-clock restamping
│   ├── tf_relay_node.py     # fixes MCU-uptime TF stamps
│   └── odom_relay_node.py   # fixes MCU-uptime odometry stamps
├── launch/
│   ├── dwa_with_cartographer.launch.py
│   ├── cartographer_only.launch.py   # mapping stack without the DWA node
│   └── sim_world.launch.py           # Gazebo world + robot (sim bringup)
├── config/
│   └── dwa_params.yaml      # All tunable parameters with source citations
├── rviz/
│   └── dwa_cartographer.rviz
├── test/
│   ├── test_dwa_node.py     # DWA node unit tests
│   └── test_relay_nodes.py  # relay stamping tests (incl. clock-step cases)
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```
