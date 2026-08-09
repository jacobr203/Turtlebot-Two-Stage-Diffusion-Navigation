# Capstone — Two-Stage Short-Horizon Diffusion Policy

Artefact for the MSc capstone *"Imitation Training a Two-Stage Short Horizon
Diffusion Policy for Cluttered Indoor Robotics Navigation"* (Univ. of
Liverpool CSCK700). The rest of `pathfinding_algorithms/` provides the
benchmark baseline (custom DWA) and the Gazebo/real-robot infrastructure;
this directory adds the BARN environments, the A* expert-data pipeline, the
diffusion policies, and the evaluation framework.

`COLCON_IGNORE` keeps colcon out of this subtree — everything here is plain
Python executed by `capstone/.venv` (torch + rclpy in one interpreter, see
`setup_venv.sh`; no sudo needed), not ament packages.

## Status (2026-07-12)

| Phase | Deliverable | Result |
|---|---|---|
| 0 | venv + skeleton | torch 2.12 CPU + rclpy in one interpreter |
| A | 300 BARN worlds in Gazebo Classic | all load; grids derived; **all 300 plannable** (0.21 m inflation); bumper model gives GT collisions |
| A+ | 66 stress worlds (dense/winding/scatter) | generated + plannability-gated; train/holdout splits in `barn/dense_worlds.yaml` |
| B | Expert dataset **v2** | **1,908 episodes** (318 worlds × 4 clean + 2 noise-injected), 500,951 samples @ 4 Hz, zero collisions in final data; centered wide-preferring routes (`common.grid.plan_route`) |
| C | Diffusion policies (1.1 M params) | v2 retrain ~55 min each on CPU; val ADE 1.83 cm (two-stage, subgoal+goal conditioning) / 2.09 cm (single-stage); DDPM-35 sampling **34 ms** (DDIM-10: 10 ms) vs 250 ms @ 4 Hz budget |
| D | Closed-loop deployment | standard tier at v1 parity (w18 ✓ bare); stress tier = documented negative result for bare policies (documented negative result) — safety gate converts winding crashes into safe timeouts |
| E | Evaluation matrix | `--safety-gate` ablation axis added to trial.py; v1 poster subset numbers retained for reference; full matrix rerun pending on v2 |
| F–G | Real robot + analysis | pending (Sep–Oct) |

Dated results log and method-decision rationale are kept as local working
notes (`notes/`, not tracked).

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

Commands: [COMMANDS.md](COMMANDS.md). World difficulty:
[barn/WORLDS.md](barn/WORLDS.md).

## The two policy variants

Both share the same network (1.1 M-param FiLM temporal U-Net, DDPM T=35),
training pipeline, and dataset — they differ only in conditioning, so any
performance gap isolates the value of the high-level planning stage:

| Variant | Conditioning | Used by |
|---|---|---|
| **two_stage** | live A* subgoal (1.2 m along the clearance-centered route toward the goal) + final-goal features + lidar stack — the main capstone artefact | `run.sh policy --goto` (grid available), evaluation matrix |
| **single_stage** | final-goal features + lidar stack only (no planner stage) — the ablation baseline | `run.sh policy` wander/mapping mode (no grid exists to plan subgoals on), evaluation matrix |

The two are always trained back-to-back on the same dataset version: the
architecture comparison in the evaluation matrix is only meaningful when the
data is held constant, and wander mode should never silently run on a
checkpoint from an older expert. Checkpoints live at
`diffusion/runs/<variant>_v2/best.pt`; each bundles its config (including the
conditioning type and vec_dim), normalization stats, and git SHA, and
`deploy/policy_node.py` reconstructs inference exactly from the checkpoint
alone.

## Layout

| Dir | Contents |
|---|---|
| `barn/` | BARN download (pinned SHA), world fix-up (eval + fast variants), world→grid derivation, per-world metadata + seeded 260/40 train/holdout split |
| `models/` | `turtlebot3_burger_bumper` — apt Burger + contact sensor → `/bumper_states` + encoder-sourced odom (re-anchors at zero per respawn, like the real robot) |
| `common/` | Grid A* (0.05 m supersampled, Burger-inflated) + string-pull, frame helpers, fixed-360-bin scan resampling, pure-pursuit tracker — shared by expert, deployment, and evaluation |
| `expert/` | A* expert: tracker + 4 Hz recorder nodes, per-episode and dataset collectors (resumable, gzserver-segfault watchdog) |
| `diffusion/` | Dataset loader (H=8 waypoint labels), 1D-CNN encoder + FiLM temporal U-Net, DDPM T=35 (+DDIM fallback), training loop (TensorBoard), offline ADE/FDE + latency eval |
| `deploy/` | `policy_node.py` (node setup and ROS callbacks) plus `deploy/policy/` — the behaviour layers, one concern per module; live A* `subgoal_planner.py` is the high-level stage |
| `evaluation/` | `SimHarness` (worker isolation, gazebo lifecycle), `trial.py` (one closed-loop trial: policies + DWA), `run_matrix.py` (resumable sharded sweep), `aggregate.py`, configs |
| `tests/` | Helper and equivalence tests that run without a simulator |
| `results/` | Trial CSVs + latency logs (gitignored except `results/summary/`) |

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

## Quickstart

Full command reference, including which world to run and why:
**[COMMANDS.md](COMMANDS.md)**. The shortest useful run is the colour
task on an open world:

```bash
cd ~/pathfinding_algorithms/capstone
../run.sh policy-two-stage --3_color_test --sim 20
```

Setup and the rest of the pipeline:

```bash
# One-time: venv with CPU torch + ROS interop (no sudo needed)
./setup_venv.sh

# One-time: fetch + prepare the 300 BARN environments
bash barn/download_barn.sh          # sparse clone, pinned commit, verifies 300+300
.venv/bin/python barn/fixup_worlds.py     # → barn/worlds/ (eval) + barn/worlds_fast/ (collection)
.venv/bin/python barn/world_to_grid.py    # → barn/grids/grid_%03d.npy (committed)
.venv/bin/python barn/build_metadata.py   # → barn/metadata.yaml (start/goal/optimal lengths/splits)

# Drive the existing DWA stack in a BARN world
~/pathfinding_algorithms/run.sh lidar --sim \
    ~/pathfinding_algorithms/capstone/barn/worlds/world_000.world

# Collect the expert dataset (headless, resumable, overnight-friendly;
# v2 = clean episodes 0-3 + noise-injected 4-5, shard with --shard/--worker)
.venv/bin/python expert/collect_dataset.py --config expert/configs/collect_v2.yaml

# Train the diffusion policy (both conditioning variants, ~25 min each on CPU)
.venv/bin/python -m diffusion.train --config diffusion/configs/two_stage.yaml
.venv/bin/python -m diffusion.train --config diffusion/configs/single_stage.yaml
tensorboard --logdir diffusion/runs          # training curves

# Offline checkpoint metrics (ADE/FDE + CPU sampling latency vs Hz budgets)
.venv/bin/python -m diffusion.evaluate_offline --ckpt diffusion/runs/two_stage_v1/best.pt

# One closed-loop trial (add --gui to watch it in Gazebo)
.venv/bin/python evaluation/trial.py --model two_stage --world-idx 18 --hz 4

# Manual driving, ONE terminal (Gazebo/bringup + Cartographer + RViz + policy;
# Ctrl-C tears everything down, real-robot wheels stopped and verified):
~/pathfinding_algorithms/run.sh policy --sim                  # wander + map (single_stage)
~/pathfinding_algorithms/run.sh policy --sim \
    ~/pathfinding_algorithms/capstone/barn/worlds/world_018.world --goto -2.25 13
~/pathfinding_algorithms/run.sh policy                        # real robot: wander + map the room
~/pathfinding_algorithms/run.sh policy --goto 2.0 0.0 --map ~/maps_test/my_map.yaml

# Evaluation sweeps (resumable; shard with --shard k/n --worker k)
.venv/bin/python evaluation/run_matrix.py --config evaluation/configs/eval_smoke.yaml
.venv/bin/python evaluation/run_matrix.py --config evaluation/configs/eval_poster.yaml
.venv/bin/python evaluation/aggregate.py --run-id poster_v1
```

Smoke tests live in `~/unit_tests/capstone_smoke/`:
`python3 ~/unit_tests/run_tests.py -capstone_smoke`.

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
  into safe timeouts/partial progress (decisions.md #27).
- **Path efficiency vs the taut optimum**: `optimal_len_ours` is the taut
  string-pulled A* length, but v2 expert paths are clearance-centered and
  ~3–8 % longer by design — v2 policies inherit that offset. Compare within
  v2, or against the manifests' `planned_len`.
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
