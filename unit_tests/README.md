# Unit Tests

Integration tests for the TurtleBot3 ROS2 stack. Each test folder is self-contained — it handles starting any components it needs (bringup, Cartographer, etc.) if they are not already running, so tests can be run against a cold or partially-started system.

---

## Folder Structure

```
unit_tests/
├── run_tests.py                  ← test runner
├── README.md
├── lidar_cartographer/           ← one test suite per folder
│   ├── conftest.py               ← shared pytest fixtures (rclpy, bringup, cartographer)
│   ├── helpers.py                ← start/check bringup and cartographer stack
│   ├── test_lidar_active.py      ← 3 tests: scan publishing, data validity, coverage
│   └── test_map_building.py      ← 5 tests: relays, map publishing, known cells, dimensions
├── dwa_avoidance/                ← full navigation stack (DWA + Cartographer)
│   ├── conftest.py
│   └── test_dwa_stack.py         ← 10 tests: node presence, topic data, velocity limits
├── gazebo_sim/                   ← all 3 algorithms in headless Gazebo (no robot!)
│   ├── conftest.py               ← per-module Gazebo instances (burger / burger_brio)
│   ├── helpers.py                ← local launches only — no SSH
│   ├── test_lidar_sim.py         ← 4 tests: scan validity, map builds & grows
│   ├── test_camera_sim.py        ← 5 tests: camera topics, /scan_camera, sim-time stamps
│   └── test_nav2_sim.py          ← 2 tests: lifecycle nodes active, map served
└── capstone_smoke/               ← BARN worlds + expert pipeline in headless Gazebo (no robot!)
    ├── conftest.py               ← SimHarness fixtures (capstone/evaluation/sim_helpers.py)
    ├── test_barn_world_loads.py  ← 3 tests: lidar in BARN world 0, GT state plugin, bumper quiet
    ├── test_bumper_model.py      ← 1 test: deliberate wall crash fires /bumper_states
    └── test_expert_episode.py    ← 5 tests: full A*+pure-pursuit episode, npz schema, tracking
```

Package-level unit tests (pure logic, no robot needed) live inside the
packages themselves — see `~/pathfinding_algorithms/dwa_avoidance/test/` and
`~/pathfinding_algorithms/dwa_camera_avoidance/test/`; run them with
`python3 -m pytest <package>/test/ -v` from a sourced workspace.

---

## Running Tests

### List available test folders
```bash
python3 ~/unit_tests/run_tests.py --list
```

### Run one folder
```bash
python3 ~/unit_tests/run_tests.py -lidar_cartographer
```

### Run multiple folders
```bash
python3 ~/unit_tests/run_tests.py -lidar_cartographer & -another_folder & -third_folder
```

The `&` between folder names is treated as a separator — it does not run tests in parallel.

### Useful flags
```bash
python3 ~/unit_tests/run_tests.py -lidar_cartographer -v     # verbose output
python3 ~/unit_tests/run_tests.py -lidar_cartographer -x     # stop on first failure
```

---

## Test Suites

### `lidar_cartographer/`

Tests that the LiDAR is running and Cartographer is actively building a map.

**Prerequisites:** Robot must be powered on and reachable at `<robot-ip>`.
Everything else (bringup, scan relay, Cartographer) is started automatically if not already running.

| Test file | Tests | What it checks |
|---|---|---|
| `test_lidar_active.py` | `test_scan_topic_publishing` | `/scan` receives messages within 10 s |
| | `test_scan_data_valid` | ≥ 50% of range values are finite and within 0.12–3.50 m |
| | `test_scan_coverage` | scan arc ≥ 340° with ≥ 180 rays |
| `test_map_building.py` | `test_scan_relay_running` | `scan_relay_node` is live on the ROS2 graph |
| | `test_tf_relay_running` | `tf_relay_node` is live on the ROS2 graph |
| | `test_map_topic_publishing` | `/map` receives an `OccupancyGrid` within 30 s |
| | `test_map_has_known_cells` | map contains ≥ 50 non-unknown cells |
| | `test_map_dimensions` | map is at least 10 × 10 cells |

### `dwa_avoidance/`

Runs against the live full navigation stack (`dwa_with_cartographer.launch.py`,
started automatically if not running). Verifies all four stack nodes are on
the graph, `/scan_relayed`, `/cmd_vel`, and `/map` publish valid data,
velocity commands stay within TurtleBot3 Burger limits, and the DWA produces
forward motion within 60 s (not permanently stuck).

### `gazebo_sim/`

**No robot required.** Starts headless
Gazebo Classic per test module (plain burger for lidar/nav2, the
`turtlebot3_burger_brio` camera model for the camera tests) and launches
each algorithm with `use_sim_time:=true`. Covers: LiDAR scan validity and
Cartographer map growth, camera topics → `/scan_camera` with stamps that
respect sim time, and Nav2 lifecycle nodes reaching `active`. Runtime is a
few minutes (three Gazebo bring-ups).

### `capstone_smoke/`

**No robot required.** Smoke tests for the MSc capstone layer
(`~/pathfinding_algorithms/capstone/`): BARN world 0 loads in Gazebo Classic
with the `turtlebot3_burger_bumper` model at the official spawn (lidar alive,
`/gazebo/model_states` ground truth, `/bumper_states` quiet at rest), a
deliberate wall crash fires the bumper, and one full expert episode
(A* plan → pure-pursuit tracking → 4 Hz recorder → npz) succeeds with sane
tracking error. Prerequisite: the BARN worlds must be generated first — see
`~/pathfinding_algorithms/capstone/README.md` (download + fixup + grids).

`lidar_cartographer/` and `dwa_avoidance/` still require the physical robot
at `<robot-ip>`.

---

## Adding a New Test Suite

1. Create a new folder under `~/unit_tests/`:
   ```bash
   mkdir ~/unit_tests/my_new_suite
   ```

2. Add the required files:
   ```
   my_new_suite/
   ├── __init__.py
   ├── conftest.py     ← pytest fixtures (copy and adapt from lidar_cartographer)
   ├── helpers.py      ← any setup/teardown helpers
   └── test_*.py       ← one or more test files
   ```

3. Run it:
   ```bash
   python3 ~/unit_tests/run_tests.py -my_new_suite
   ```

The runner automatically discovers all `test_*.py` files in the folder — no registration needed.
