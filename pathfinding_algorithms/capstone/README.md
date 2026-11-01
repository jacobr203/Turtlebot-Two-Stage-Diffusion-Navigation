# Two-Stage Short-Horizon Diffusion Policy

BARN environments, an A* expert-data pipeline, diffusion policies and the
evaluation framework. The rest of `pathfinding_algorithms/` provides the DWA
baseline and the Gazebo and real-robot infrastructure.

`COLCON_IGNORE` keeps colcon out of this directory. Everything here is plain
Python run by `capstone/.venv`, which holds CPU torch alongside the system
`rclpy` (see `setup_venv.sh`).

## How the two-stage system runs

The diffusion model authors every trajectory the robot follows. Nothing
around it writes motion; the surrounding layers decide **where to aim**
the model and refuse commands that would collide.

```
                    ┌─────────────── control tick, 4 Hz ───────────────┐
  LiDAR ─┐          │                                                  │
  odom  ─┼─ observe │  1. pick a goal      seek / explore layers        │
  map   ─┘          │  2. plan a subgoal   A* on the live map, ~1.2 m   │
                    │  3. sample K paths   diffusion model, H=8 waypts  │
                    │  4. choose one       progress, smoothness, clear  │
                    └──────────────────────┬───────────────────────────┘
                                           │ chosen path
                    ┌──────────────────────▼───────────────────────────┐
   camera ─────────►│  tracker, 20 Hz      pure pursuit along the path  │
   (colour task)    │  safety gate         scale or refuse the command  │
                    └──────────────────────┬───────────────────────────┘
                                           ▼
                                        /cmd_vel
```

**Stage 1 — where to go.** A* runs on the map SLAM is building and
emits a subgoal about 1.2 m ahead along a clearance-centred route. This
is the planning stage the capstone is testing: it keeps the learned
policy inside the short-horizon regime it was trained on, instead of
asking it to reason about a goal ten metres away behind three walls.

**Stage 2 — how to get there.** The diffusion model takes a two-frame
LiDAR stack plus goal and subgoal features and denoises a chunk of 8
waypoints (roughly 0.4 m of travel). Several chunks are drawn per tick
and the best is selected on measured progress, smoothness and
clearance; the rest are discarded.

**Following it.** Pure pursuit at 20 Hz keeps the wheels moving
smoothly between predictions, so a 4 Hz replan does not produce 4 Hz
jerks. The safety gate is the last word before `/cmd_vel`.

The extra deployment layers — colour seeking, exploration, viewpoint
selection, recovery — only ever choose the goal handed to stage 1. That
separation is deliberate: it keeps the learned component's contribution
measurable, and it is why the evaluation harness can switch the safety
gate off and still get a meaningful comparison.

## The two policy variants

Both share the same network (1.1 M-parameter FiLM temporal U-Net, DDPM T=35),
training pipeline and dataset. They differ only in conditioning, so any
performance gap isolates the value of the planning stage.

| Variant | Conditioning |
|---|---|
| **two_stage** | Live A* subgoal (1.2 m along a clearance-centred route), final goal, velocity, scan age, two LiDAR frames |
| **single_stage** | Final goal, velocity, scan age, two LiDAR frames |

Checkpoints live at `diffusion/runs/<variant>_v5/best.pt`. Each one bundles its
config and normalisation statistics, so `deploy/policy_node.py` can rebuild
inference from the checkpoint alone.

## Data generations

| Generation | Training worlds | Change |
|---|---|---|
| v1 | 260 BARN | Baseline, taut expert |
| v2 | 318 (adds 58 stress worlds) | Clearance-centred expert, noise-injected episodes |
| v3 | 340 multipath only | Route diversity, 160 degree scan arc, segment balancing |
| v4 | 658 (v2 and v3 pools combined) | Same treatments on the combined pool |
| v5 | 728 (adds 70 pocket worlds) | Entering and leaving narrow recesses; four candidates per tick |

Config files named `two_stage.yaml` and `single_stage.yaml` are generation v2.

## Layout

| Directory | Contents |
|---|---|
| `barn/` | BARN download, world fix-up, grids, metadata, world generators, difficulty ranking |
| `models/` | Gazebo Burger models with a contact sensor and encoder odometry |
| `common/` | Grid A*, geometry, scan resampling, pure pursuit, arena and course files, `SimHarness` |
| `expert/` | A* expert tracker, 4 Hz recorder, dataset collectors |
| `diffusion/` | Dataset, model, DDPM and DDIM, training, checkpoint loading, offline evaluation |
| `deploy/` | `policy_node.py`, subgoal planner, colour detector, `policy/` behaviour layers |
| `evaluation/` | Simulation sweeps and real-robot trials (`real/` holds the harness) |
| `tests/` | Unit tests that run without a simulator |
| `results/summary/` | Per-model and per-world tables; raw trials are not tracked |

### Deployment layers (`deploy/policy/`)

Each module is a mixin composed into the node, so `self` is the node
and no layer owns state privately.

| Module | Responsibility |
|---|---|
| `geometry.py` | frame conversions, line of sight, coverage memory |
| `viewpoint.py` | where to stand to see a target; corridor memory |
| `explore.py` | frontier exploration, wander goals, face coverage |
| `seek.py` | what to chase and how, during the colour search |
| `safety.py` | speed gate, recovery behaviours, watchdog |
| `tracking.py` | control tick, trajectory sampling, pure pursuit |

## Reproducing the pipeline

Full reference: **[COMMANDS.md](COMMANDS.md)**. Run everything from this
directory.

```bash
# One-time setup
./setup_venv.sh
source /opt/ros/humble/setup.bash

# Prepare the 300 BARN worlds
bash barn/download_barn.sh
.venv/bin/python barn/fixup_worlds.py
.venv/bin/python barn/world_to_grid.py
.venv/bin/python barn/build_metadata.py

# Collect expert data (resumable; shard with --shard and --worker)
.venv/bin/python expert/collect_dataset.py --config expert/configs/collect_v3.yaml
.venv/bin/python expert/collect_dataset.py --config expert/configs/collect_pocket.yaml

# Train both variants on the same data
.venv/bin/python -m diffusion.train --config diffusion/configs/two_stage_v5.yaml
.venv/bin/python -m diffusion.train --config diffusion/configs/single_stage_v5.yaml

# Offline metrics, then one closed-loop trial, then a full sweep
.venv/bin/python -m diffusion.evaluate_offline --ckpt diffusion/runs/two_stage_v5/best.pt
.venv/bin/python evaluation/trial.py --model two_stage --world-idx 18 --hz 4
.venv/bin/python evaluation/run_matrix.py --config evaluation/configs/eval_full_v5_two_stage.yaml
.venv/bin/python evaluation/aggregate.py --run-id full_v5_two_stage

# Watch the shortest useful run
../run.sh policy-two-stage --3_color_test --sim 20
```

`data/v5` is a merged directory of `data/v3` and `data/pocket_v1`. Create it by
linking both sets of episodes into one folder before training v5.

## Metrics conventions

- **Success**: ground truth within **1.0 m** of the goal (BARN parity);
  timeout 240 s sim; collisions from the bumper contact sensor (ground
  contacts filtered).
- **Path efficiency** = `gt_path_len / (optimal_len_ours − 1.0 m)`: success
  fires at the goal circle's edge, so the optimal path is truncated at the
  same boundary — comparing against the raw optimal length would flatter
  every model by ~1 m.
- **Inference latency** is wall-clock per control tick, split into subgoal
  planning (`lat_plan_ms`, two-stage only) and diffusion sampling
  (`lat_sample_ms`). Budgets: 500/333/250 ms for 2/3/4 Hz.
- Evaluation runs at **RTF 1.0** (latency is wall-clock while control is
  sim-time; faster-than-realtime would silently penalize slow models).
  Expert collection uses the unbounded-RTF world variants.

## Known behavior

- **Marginal 3-cell gaps** (~0.30 m surface) flip between success and
  collision run to run — imitation-policy variance off the expert manifold.
  This is a measured phenomenon, not a bug; the 4-trials-per-world axis
  quantifies it.
- **The winding/scatter stress tier is beyond bare policies of every
  generation** (v1, v2, v2+noise all fail its opening clutter; so did stock
  Nav2 before route pinning). Its corridors leave ~3.7 cm of total
  closed-loop error budget at expert speed — less than the stack's combined
  prediction + tracking error. The tier is the brittleness exhibit by
  design; the `--safety-gate` deployment configuration converts crashes
  into safe timeouts/partial progress.
- **Path efficiency vs the taut optimum**: `optimal_len_ours` is the taut
  string-pulled A* length, but expert paths from v2 onwards are clearance-centered and
  ~3–8 % longer by design — the policies inherit that offset. Compare against the manifests' `planned_len`.
- gzserver **segfaults every ~15 delete+respawn cycles** on WSL2; the
  collectors and matrix runner restart the world and retry automatically
  (harness errors are never recorded as model failures).

## BARN geometry (canonical, from the challenge repo's `run.py`)

- Every world: 30×64 cell walled arena, 0.15 m lattice, origin (−4.425, 0.075);
  solid side/back walls, open top; cylinders r = 0.075 m
- Spawn (−2.25, 3, yaw 1.57), goal (−2.25, 13), success radius 1.0 m
- Optimal path lengths 10.0–10.7 m across all 300 worlds
- `path_files/path_%d.npy` = BARN's reference path (its own frame; used only
  for the reference length in `metadata.yaml`)
