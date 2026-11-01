# Architecture

Which classes and scripts run, in order, for each scenario. Paths are relative
to `pathfinding_algorithms/`.

## Running a trained policy

`run.sh policy` handles cleanup, bringup and checkpoint choice, then launches
`capstone/deploy/policy_node.py`.

1. `DiffusionPolicyNode` (`capstone/deploy/policy_node.py`) is the ROS node. Its
   behaviour comes from the mixins in `capstone/deploy/policy/`.
2. `load_policy()` (`capstone/diffusion/checkpoint.py`) builds `DiffusionPolicy`
   (`capstone/diffusion/model.py`): `ObsEncoder` plus `ConditionalUnet1D`, which
   is made of `SinusoidalEmb` and `FiLMResBlock`.
3. `DDPM` (`capstone/diffusion/ddpm.py`) is the sampler.
4. Two-stage only: `SubgoalPlanner` (known grid) or `LiveSubgoalPlanner` (live
   map) in `capstone/deploy/subgoal_planner.py`, using `Planner` from
   `capstone/common/grid.py`.
5. On each control tick: `resample_scan` (`capstone/common/scan.py`), then DDPM
   sampling, then candidate selection. `PurePursuit`
   (`capstone/common/pure_pursuit.py`) follows the chosen path at 20 Hz, and the
   safety gate has the last word before `/cmd_vel`.

The colour task also starts `ColorSeeker` (`capstone/deploy/color_seeker.py`)
and `ColorTestManager` (`capstone/deploy/color_test_manager.py`).

The two variants share every network class. Only the checkpoint's
`conditioning` field and the planner stage differ.

## Generating worlds

Plain scripts in `capstone/barn/`, in order:

1. `download_barn.sh` clones the official 300 worlds at a pinned commit.
2. `fixup_worlds.py` writes `worlds/` (evaluation) and `worlds_fast/`
   (collection).
3. `world_to_grid.py` writes `grids/*.npy`.
4. `build_metadata.py` writes `metadata.yaml` with the train and holdout split.
5. Optional generators: `make_dense_worlds.py`, `make_fork_worlds.py` and
   `make_pocket_worlds.py`.
6. `grids_to_maps.py` writes maps for Nav2.
7. `place_color_targets.py` writes `worlds_color/` and `targets/`.

## Collecting expert data

`capstone/expert/collect_dataset.py` starts a `SimHarness`
(`capstone/common/sim_harness.py`) and calls `run_episode`
(`collect_episode.py`) per episode. `Planner` produces the A* path,
`ExpertTracker` (`tracker_node.py`) drives it with `PurePursuit`, and
`EpisodeRecorder` (`recorder_node.py`) logs each sample at 4 Hz.

## Training

`capstone/diffusion/train.py` uses `ExpertDataset` (`dataset.py`), the
`DiffusionPolicy` chain above, and `DDPM` and `EMA` (`ddpm.py`). It saves
`last.pt` and `best.pt` to `diffusion/runs/<run_name>/`. Each checkpoint holds
everything needed to rebuild the policy.

## Evaluating in simulation

1. `capstone/evaluation/run_matrix.py` walks a sweep config, shardable across
   workers.
2. `SimHarness` boots Gazebo per world and recovers from gzserver crashes.
3. `capstone/evaluation/trial.py` runs one closed-loop trial of a policy or DWA.
4. `capstone/evaluation/aggregate.py` writes the summaries in `results/summary/`.

## Evaluating on the real robot

`capstone/evaluation/real_trial.py` is the command-line entry point, and
`capstone/evaluation/real/` holds the parts:

| Module | Responsibility |
|---|---|
| `config.py` | Paths, timeouts, topics, CSV schemas |
| `schedule.py` | Seeded run orders and pending runs |
| `robot.py` | SSH access to the robot |
| `harness.py` | Starts, resets and stops the ROS stack around a trial |
| `tasks.py` | The point-to-point and colour trials, and the run loop |
| `monitors.py` | Per-trial state machines fed from ROS callbacks |
| `metrics.py` | Per-trial calculations |
| `records.py` | Reading and amending the results CSV |
| `operator_io.py` | Prompts, live keys, find review, camera viewers |
| `paths.py` | Saves and draws run paths on the arena map |
| `errors.py` | Harness exceptions |

`session_report.py` summarises a finished session, and `course_tools.py`
measures arenas and courses.

## Talking to the robot

Everything uses CycloneDDS with `config/cyclone_config.xml`. `run.sh` starts
bringup over SSH using `robot.env`. Three relays in `dwa_avoidance/` restamp
robot data on the PC, because the WSL2 clock steps backwards:
`ScanRelayNode`, `TFRelayNode` and `OdomRelayNode`.

## Shutting down

- `run.sh` traps Ctrl-C, stops every node and verifies the wheels have stopped.
- `tools/kill_all.sh` stops all ROS processes on the PC and the robot.
- `tools/kill_robot.sh` stops only the robot's processes.
