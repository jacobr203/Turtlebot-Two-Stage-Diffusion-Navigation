# TurtleBot3 Project

Robot IP:   <robot-ip>  (alias: tbot)
Robot user: <robot-user>
This PC IP: <workstation-ip>
ROS:        Humble Hawksbill (Ubuntu 22.04 WSL2)
RMW:        CycloneDDS  (cyclone_config.xml — eth0 multicast)


## Quick Start

Always kill all ROS2 processes before starting or restarting anything:

    bash ~/kill_all.sh

Kill the stale daemon on every startup:

    pkill -f "ros2 daemon"


## Running the Algorithms (run.sh)

One command runs any of the three navigation algorithms, on the real robot
(default) or in Gazebo (`--sim`). **Ctrl-C stops any run and cleans up
everything** — Gazebo included, and in real mode the robot's wheels are
stopped.

    ~/pathfinding_algorithms/run.sh <lidar|camera|nav2> [--sim [world]] [--headless] [--no-rviz] [--map <yaml>]

### Simulation (no robot needed)

    # 1. LiDAR DWA + Cartographer — Gazebo + RViz open; robot wanders and maps the pillar world
    ~/pathfinding_algorithms/run.sh lidar --sim

    # 2. Camera DWA + Cartographer — Burger with the simulated BRIO camera
    ~/pathfinding_algorithms/run.sh camera --sim

    # 3. Nav2 — localizes automatically at the spawn pose; use "Nav2 Goal" in RViz
    ~/pathfinding_algorithms/run.sh nav2 --sim

    # 4. Diffusion policy (capstone) — wanders toward open space and maps the world
    ~/pathfinding_algorithms/run.sh policy --sim
    #    goal-directed instead: in a BARN world it auto-uses the two-stage model
    ~/pathfinding_algorithms/run.sh policy --sim \
        ~/pathfinding_algorithms/capstone/barn/worlds/world_018.world --goto -2.25 13

Success looks like: for 1 and 2 the robot drives itself around the pillars
while the map crystallizes in RViz; for 3 the robot sits localized until you
click **Nav2 Goal**, then drives there and stops.

Every lidar/camera run auto-saves its occupancy map (refreshed every 30 s)
to `pathfinding_algorithms/maps/<sim|real>/<lidar|camera>/<timestamp>_<algo>.{pgm,yaml}`
— see `pathfinding_algorithms/maps/README.md`. Navigate on one later with
`run.sh nav2 --map <that yaml>`.

### Real robot (power it on; bringup and camera auto-start over SSH)

    # 1. LiDAR DWA — wanders and maps the room. Give it floor space first!
    ~/pathfinding_algorithms/run.sh lidar

    # 2. Camera DWA — also auto-starts the BRIO camera on the robot
    ~/pathfinding_algorithms/run.sh camera

    # 3. Nav2 on the saved room map — one manual step:
    #    in RViz, click "2D Pose Estimate" where the robot actually is
    ~/pathfinding_algorithms/run.sh nav2

    # 4. Diffusion policy — wanders + maps the room (goals are relative to
    #    where the robot started; Ctrl-C stops the wheels, verified)
    ~/pathfinding_algorithms/run.sh policy
    ~/pathfinding_algorithms/run.sh policy --goto 2.0 0.0 --map ~/project_/maps_test/my_map.yaml

### Variants

    ~/pathfinding_algorithms/run.sh lidar --sim house        # bigger house world
    ~/pathfinding_algorithms/run.sh camera --sim --headless  # no Gazebo GUI (RViz still opens)
    ~/pathfinding_algorithms/run.sh lidar --no-rviz          # no RViz (doesn't apply to nav2)
    ~/pathfinding_algorithms/run.sh nav2 --map ~/some/other_map.yaml

### Automated checks

    python3 ~/unit_tests/run_tests.py -gazebo_sim            # all 3 algorithms in sim, ~1 min, no robot
    python3 ~/unit_tests/run_tests.py -lidar_cartographer    # real robot: lidar + mapping health
    python3 ~/unit_tests/run_tests.py -dwa_avoidance         # real robot: full stack, robot will drive

If a sim run refuses to start, a leftover gzserver is the usual suspect —
run the same command again; the script kills strays before launching.


## MSc Capstone (diffusion policy vs DWA on BARN)

`pathfinding_algorithms/capstone/` contains the dissertation artefact:
*"Imitation Training a Two-Stage Short Horizon Diffusion Policy for
Cluttered Indoor Robotics Navigation."* The three `run.sh` algorithms above
are the surrounding infrastructure; the capstone adds the 300 BARN benchmark
worlds, an A* expert-data pipeline, two trained diffusion policies
(two-stage = subgoal-conditioned, single-stage = goal-conditioned), and the
evaluation matrix that compares them against the custom DWA baseline.

Status and headline numbers live in `pathfinding_algorithms/capstone/README.md`
(quickstart + per-phase results).

    # drive any algorithm in a BARN world (absolute path works everywhere)
    ~/pathfinding_algorithms/run.sh lidar --sim \
        ~/pathfinding_algorithms/capstone/barn/worlds/world_000.world

    # one closed-loop capstone trial (policy or dwa)
    cd ~/pathfinding_algorithms/capstone
    .venv/bin/python evaluation/trial.py --model two_stage --world-idx 18 --hz 4

    # capstone smoke tests (no robot)
    python3 ~/unit_tests/run_tests.py -capstone_smoke


## Run the Full Navigation Stack Manually (DWA + SLAM)

> Superseded by `run.sh lidar` above, kept for reference / debugging.

    # 1. Start robot bringup on the TurtleBot (SSH, blocks until Ctrl-C)
    bash ~/scripts/bringup_robot.sh

    # 2. Launch DWA obstacle avoidance + Cartographer SLAM + RViz on this PC
    source /opt/ros/humble/setup.bash
    source ~/pathfinding_algorithms/install/setup.bash
    ros2 launch dwa_avoidance dwa_with_cartographer.launch.py

    # Suppress RViz (headless / test mode):
    ros2 launch dwa_avoidance dwa_with_cartographer.launch.py use_rviz:=false

This starts (in order): scan_relay → tf_relay → cartographer → occupancy_grid → rviz2 → dwa_node


## Run Cartographer SLAM Only (no DWA, robot stays still)

    source /opt/ros/humble/setup.bash
    source ~/pathfinding_algorithms/install/setup.bash
    ros2 launch dwa_avoidance cartographer_only.launch.py

Starts: scan_relay → tf_relay → cartographer → occupancy_grid


## Teleoperation

    bash ~/scripts/teleop.sh

Arrow keys to drive; robot stops when you quit.


## Camera (bringup + image viewer)

    bash ~/scripts/start_robot.sh

Starts bringup and camera on the robot, opens rqt_image_view locally on /camera/image_raw.


## Timeshift (system restore)

    # Create a restore point before risky experiments
    sudo timeshift --create --comments "Before ROS experiment"

    # Roll back to a previous snapshot
    sudo timeshift --restore


---

## Folder Reference

### pathfinding_algorithms/
Colcon workspace: the three navigation stacks + the capstone layer.

    pathfinding_algorithms/
    ├── run.sh                    — unified runner (real robot / Gazebo; see above)
    ├── dwa_avoidance/            — LiDAR DWA + Cartographer (ROS package)
    │   ├── dwa_avoidance/
    │   │   ├── dwa_node.py          — DWA; wander + goal modes; control_period param (2/3/4 Hz)
    │   │   ├── scan_relay_node.py   — re-stamps /scan with wall clock → /scan_relayed
    │   │   ├── tf_relay_node.py     — fixes MCU-uptime odom→base_footprint TF stamps
    │   │   └── odom_relay_node.py   — odom relay (kept but not used in launch)
    │   ├── launch/                  — full stack, SLAM-only, and Gazebo world bringup
    │   │                              (sim_world.launch.py takes names or absolute .world paths)
    │   ├── config/                  — dwa_params.yaml + Cartographer .lua configs
    │   └── rviz/
    ├── dwa_camera_avoidance/     — camera-fused DWA variant + burger_brio Gazebo model
    ├── capstone/                 — MSc capstone: BARN worlds, expert pipeline, diffusion
    │                               policies, evaluation matrix (see capstone/README.md)
    ├── dev_tools/                — live map/scan viewer + auto-restart dev loop
    └── maps/                     — per-run occupancy maps (sim|real / lidar|camera)

Build after any source change:

    cd ~/pathfinding_algorithms
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install
    source ./install/setup.bash


### unit_tests/
Integration test suites that mirror the live launch sequence.

    unit_tests/
    ├── run_tests.py                         — test runner (pass suite flags)
    ├── lidar_cartographer/
    │   ├── test_lidar_active.py             — LiDAR is publishing on /scan
    │   └── test_map_building.py             — Cartographer builds a valid map
    ├── dwa_avoidance/
    │   └── test_dwa_stack.py                — full stack: nodes, topics, velocity
    │                                          limits, map quality, forward motion
    ├── gazebo_sim/                          — all 3 algorithms in headless Gazebo
    │   ├── test_lidar_sim.py                  (robot-free; see unit_tests/README.md)
    │   ├── test_camera_sim.py
    │   └── test_nav2_sim.py
    └── capstone_smoke/                      — BARN world + bumper + expert episode
        ├── test_barn_world_loads.py           (robot-free; needs generated worlds)
        ├── test_bumper_model.py
        └── test_expert_episode.py

Run all suites (robot must be on and bringup running):

    python3 ~/unit_tests/run_tests.py -lidar_cartographer -dwa_avoidance

Run a single suite:

    cd ~/unit_tests
    python3 -m pytest lidar_cartographer/ -v
    python3 -m pytest dwa_avoidance/ -v

Tests start the required launch files automatically and tear them down after.



### scripts/
Shell helpers for common robot operations.

    scripts/
    ├── bringup_robot.sh   — SSH into robot and start turtlebot3_bringup (blocking)
    ├── teleop.sh          — keyboard teleoperation on this PC
    ├── start_robot.sh     — bringup + camera on robot, image viewer on PC
    ├── enterBot.sh        — interactive SSH shell into the robot
    └── camera.sh          — camera launch only


### maps_test/
Saved LiDAR maps from previous sessions.

    maps_test/
    ├── my_map.yaml          — saved occupancy grid map
    └── upstairs_lidar.rviz  — RViz config used during mapping session

Load a saved map with Nav2:

    ros2 launch turtlebot3_navigation2 navigation2.launch.py \
        use_sim_time:=False map:=~/maps_test/my_map.yaml



### turtlebot3_ws/   (excluded from git)
Standard TurtleBot3 ROS2 stack (bringup, navigation2, cartographer, Gazebo).
Not modified — treated as read-only dependency.

### colcon_ws/       (excluded from git)
Contains ros2_orb_slam3 (ORB-SLAM3 monocular example). See CLAUDE.md for usage.


---

## Class Execution Order by Scenario

Reference map of which classes (and scripts) run, in order, for each common
scenario. Paths are relative to `pathfinding_algorithms/` unless noted.
Where a stage is plain bash/scripts, there are no classes and that is said.

### 1) Running the model (either variant)

`run.sh policy [--sim]` (bash: cleanup, bringup, checkpoint pick) launches
`capstone/deploy/policy_node.py`; construction order:

1. `DiffusionPolicyNode` — `capstone/deploy/policy_node.py` (the ROS node)
2. `load_policy()` (`capstone/diffusion/evaluate_offline.py`) builds
   `DiffusionPolicy` — `capstone/diffusion/model.py` — which constructs
   `ObsEncoder` and `ConditionalUnet1D`, which builds `SinusoidalEmb` and
   stacks of `FiLMResBlock` (all in `model.py`)
3. `DDPM` — `capstone/diffusion/ddpm.py` (denoising sampler)
4. *Two-stage only:* `SubgoalPlanner` (static BARN grid) or
   `LiveSubgoalPlanner` (live map) — `capstone/deploy/subgoal_planner.py`,
   planning via `Planner` in `capstone/common/grid.py`. Single-stage skips
   this stage entirely.
5. Per control tick: scan prep (`resample_scan` in `capstone/common/scan.py`;
   `crop_arc`/`obs_vec` in `capstone/diffusion/dataset.py`) → DDPM sampling →
   `PurePursuit` (`capstone/common/pure_pursuit.py`) converts the predicted
   waypoints to (v, ω)

Color-scavenger runs additionally start `ColorSeeker`
(`capstone/deploy/color_seeker.py`) and `ColorTestManager`
(`capstone/deploy/color_test_manager.py`) alongside the policy node.

### 2) Training the model

`capstone/diffusion/train.py` (functions only, no class of its own):

1. `ExpertDataset` — `capstone/diffusion/dataset.py` (train + val instances)
2. `DiffusionPolicy` → `ObsEncoder` → `ConditionalUnet1D` →
   `SinusoidalEmb` + `FiLMResBlock` (same chain as scenario 1)
3. `DDPM` (noise schedule + loss) and `EMA` (shadow weights) —
   `capstone/diffusion/ddpm.py`
4. Epoch loop saves `last.pt` / `best.pt` into `diffusion/runs/<run_name>/`

### 3) Generating BARN worlds

All in `capstone/barn/`, all plain scripts — **no classes**. Canonical order:

1. `download_barn.sh` — clone the official 300 worlds into `raw/` (pinned SHA)
2. `fixup_worlds.py` — physics-fixed `worlds/` (eval) + `worlds_fast/` variants
3. `world_to_grid.py` — cylinder poses from each `.world` → `grids/*.npy`
4. `build_metadata.py` — `metadata.yaml`: per-world facts + seeded
   train/holdout split
5. *Optional generators:* `make_dense_worlds.py` (dense clutter, idx ≥ 300),
   `make_fork_worlds.py` (multipath v3/v4 worlds, idx 400+) — each followed
   by the same grid/metadata steps for their outputs
6. `grids_to_maps.py` — rasterize grids → `maps/` for Nav2 (DWA baseline)
7. `place_color_targets.py` — stamp the color squares → `worlds_color/` +
   `targets/`

### 4) Testing and evaluating BARN models

1. `capstone/evaluation/run_matrix.py` (entry script) iterates the sweep config
2. `SimHarness` (+ nested `_Latest` topic cache) —
   `capstone/evaluation/sim_helpers.py`: boots gzserver with the BARN world,
   watches for segfaults
3. `run_trial()` — `capstone/evaluation/trial.py`: resolves the checkpoint via
   the `CKPTS` map (env-overridable), spawns the policy node subprocess
   (scenario 1's chain runs inside it), or Nav2 + `capstone/barn/nav2_route_goal.py`
   for the DWA baseline; success checks via `capstone/common/grid.py`
4. Afterwards: `capstone/evaluation/aggregate.py` (metrics) and
   `capstone/evaluation/plots.py` (figures); offline metrics via
   `capstone/diffusion/evaluate_offline.py`

Smoke tests: `python3 ~/unit_tests/run_tests.py -capstone_smoke`

### 5) Configuration for nodes and relay with the bot

Config layer (no classes): `~/cyclone_config.xml` (CycloneDDS, eth0
multicast) + env vars in `~/.bashrc` (`RMW_IMPLEMENTATION`, `CYCLONEDDS_URI`,
`TURTLEBOT3_MODEL`, `LDS_MODEL`), read by every node at startup; `run.sh`
then SSHes bringup onto the robot. Relay classes (DWA stack, run on the PC to
restamp robot topics against the WSL2 clock-step problem), started together
by `dwa_avoidance/launch/*.launch.py` before any consumer node:

1. `ScanRelayNode` — `dwa_avoidance/dwa_avoidance/scan_relay_node.py`
2. `TFRelayNode` — `dwa_avoidance/dwa_avoidance/tf_relay_node.py`
3. `OdomRelayNode` — `dwa_avoidance/dwa_avoidance/odom_relay_node.py`

### 6) Simulation-specific classes

- `SimHarness` + `_Latest` — `capstone/evaluation/sim_helpers.py`; everything
  Gazebo goes through them (world boot, respawn, pause/unpause, bumper contacts)
- Expert-data nodes only ever run in sim: `ExpertTracker`
  (`capstone/expert/tracker_node.py`) and `EpisodeRecorder`
  (`capstone/expert/recorder_node.py`)
- Sim assets (not classes): `capstone/models/turtlebot3_burger_bumper[_brio]`
  (SDF robots with bumper plugin), `capstone/barn/worlds*`; `run.sh --sim`
  wires `use_sim_time` + camera topic remaps

### 7) Real-world-specific classes

Deliberately almost none — the same `DiffusionPolicyNode`, `ColorSeeker`,
`ColorTestManager` run on hardware, launched by `run.sh` real mode (bringup
over SSH, `use_sim_time:=false`). Real-only pieces:

- The relay trio from scenario 5 (they exist because of the real robot +
  WSL2 clock skew; sim does not use them)
- `capstone/deploy/color_profiles.yaml` — real-camera HSV calibration for
  `ColorSeeker` (sim uses the sim profile in the same file)
- `run.sh` `stop_robot()` and the SSH bringup paths

### 8) Scripts that shut the robot off

All bash — **no classes**:

1. `run.sh` `cleanup()` (pkill -9 of every PC-side node) + `stop_robot()`
   (publishes zero cmd_vel on a matched subscription), both wired into the
   `on_exit` trap so Ctrl-C tears everything down, incl. `pkill -9 -f ros2`
   on the robot over SSH
2. `~/kill_all.sh` — all ROS2 on PC *and* robot (SSH), incl. the ROS2 daemon
3. `~/kill_robot.sh` — robot only: SSH heredoc Python that pkills
   `turtlebot3_node`, lidar, camera, then kills survivors by PID

### 9) Full diffusion pipeline; two-stage and single-stage

**Phase A — expert data** (`capstone/expert/`): `collect_dataset.py` →
`SimHarness` → `collect_episode.py::run_episode` per episode → A* expert path
from `Planner` (`capstone/common/grid.py`) → `ExpertTracker`
(`tracker_node.py`, drives the path with `PurePursuit`) in parallel with
`EpisodeRecorder` (`recorder_node.py`, logs scan/odom/actions to npz).

**Phase B — training** (`capstone/diffusion/`): `ExpertDataset` →
`DiffusionPolicy` (`ObsEncoder` → `ConditionalUnet1D` →
`SinusoidalEmb`/`FiLMResBlock`) → `DDPM` + `EMA` → checkpoints; sanity-checked
by `evaluate_offline.py::load_policy`.

**Phase C — deployment** (`capstone/deploy/`): `DiffusionPolicyNode` hosting
the loaded `DiffusionPolicy` + `DDPM` + `PurePursuit`.

Variant difference is conditioning only: **single-stage** feeds the (relative)
final goal straight into `ObsEncoder` — no planner class runs. **Two-stage**
inserts `SubgoalPlanner` (precomputed grid) or `LiveSubgoalPlanner` (live
occupancy) between goal and network — stage 1 plans an A* corridor and emits
short-horizon subgoals; stage 2 is the *same* diffusion network conditioned
on the current subgoal. Both variants share every network class; only the
checkpoint's `conditioning` field and the planner stage differ
(`policy_node.py` enforces the pairing).

### 10) Virtual-environment configuration

`capstone/setup_venv.sh` (bash, no classes): creates `capstone/.venv` with
`--system-site-packages` (apt-installed `rclpy` stays importable next to pip
CPU torch), falls back to pip-installed `virtualenv` (box lacks
`python3.10-venv`), installs torch from the CPU wheel index +
`requirements.txt`, then sanity-checks `numpy<2` + torch + rclpy in one
interpreter. Downstream, `VENV_PY` in `capstone/evaluation/sim_helpers.py`
is how every harness launches nodes with that interpreter.

### 11) Hidden files that had to be configured

- `~/.bashrc` — the load-bearing one: exports `TURTLEBOT3_MODEL`,
  `LDS_MODEL`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
  `CYCLONEDDS_URI=file:///home/turtlebot/cyclone_config.xml`, sources ROS
  (its target `cyclone_config.xml` sits unhidden in `~`)
- `~/.ssh/` — host keys for SSH to the robot (`tbot` alias; scripts also
  pass `-o StrictHostKeyChecking=no`)
- `~/.gitignore` / `.gitignore` exceptions — the home directory is itself a
  git repo; exceptions track e.g. `maps_test/my_map.pgm`, while
  `capstone/barn/raw/` and `capstone/diffusion/runs/` stay ignored
- `~/.gazebo` / `~/.ignition` — Gazebo model cache; the custom bumper models
  under `capstone/models/` must be resolvable via the model path
- `~/.rviz2` plus the checked-in `capstone/deploy/policy_debug.rviz` /
  `policy_mapping.rviz` — RViz layouts run.sh loads
- `~/.claude/` — assistant project memory; not part of the robot stack

Note: nothing here runs "v1" checkpoints anymore — the oldest surviving
checkpoints under `capstone/diffusion/runs/` are v2.
