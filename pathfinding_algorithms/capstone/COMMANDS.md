# Command reference

Everything here runs from the capstone directory with its own virtual
environment:

```bash
cd ~/pathfinding_algorithms/capstone
source /opt/ros/humble/setup.bash
```

The venv holds a CPU build of torch and borrows system `rclpy`. Create
it once with `./setup_venv.sh` (no sudo needed), then always invoke
Python as `.venv/bin/python` — the system interpreter cannot import the
model code.

Simulation runs go through `../run.sh`, which brings up Gazebo, SLAM,
the policy node and RViz together, and tears the whole set down on
Ctrl-C. Pick worlds deliberately: see [`barn/WORLDS.md`](barn/WORLDS.md)
for difficulty tiers.

---

## 1. Colour scavenger task

Find three coloured swatches, camera only, without collisions.

```bash
# open world — quick check that a change works at all (EASY, ~10 min)
../run.sh policy-two-stage --3_color_test --sim 20

# main development world (HARD, expect 25-30 min)
../run.sh policy-two-stage --3_color_test --sim 768

# hardest world in the set (EXTREME)
../run.sh policy-two-stage --3_color_test --sim 305

# no GUI — faster, for unattended or batch runs
../run.sh policy-two-stage --3_color_test --sim 20 --headless
```

Progress is reported on two streams:

| stream | contents |
|---|---|
| `/tmp/color_test.log` | swatch SEEN / FOUND events, collisions, final time and path length |
| the launching terminal | mode changes, viewpoint scoring, safety gate, recovery |

Watch just the task outcome:

```bash
tail -f /tmp/color_test.log | grep -E "SEEN|FOUND|collision"
```

Results append to `/tmp/color_test_results.csv` as
`stamp, total_s, path_m, collisions, <per-colour times>`.

---

## 2. Point-to-point navigation

Drive to a coordinate — the benchmark configuration, and the one used
for the DWA comparison.

```bash
# two-stage policy: A* proposes subgoals, the model drives
../run.sh policy-two-stage --goto 0 6 --sim 20

# single-stage policy: goal-conditioned, no planner
../run.sh policy --goto 0 6 --sim 20

# real robot (brings up the TurtleBot over SSH)
../run.sh policy-two-stage --goto 0 6
```

Useful flags:

| flag | effect |
|---|---|
| `--hz N` | control rate, default 4 Hz |
| `--ckpt PATH` | use a specific checkpoint |
| `--min_goal_dist M` | minimum exploration goal distance |
| `--path_scale F` | stretch predicted paths (experimental; leave at 1.0 for evaluation) |
| `--no-rviz` | skip visualization |
| `--map FILE.yaml` | start from a saved map instead of mapping live |

---

## 3. Wander and map

No goal — explore and build a map.

```bash
../run.sh policy --sim 20            # simulation
../run.sh policy                     # real robot
```

---

## 4. Evaluation sweeps

A sweep runs many worlds and models unattended and writes a results
tree per run id.

```bash
# smoke config first — a handful of trials, confirms the harness works
.venv/bin/python evaluation/run_matrix.py \
    --config evaluation/configs/eval_smoke.yaml

# full comparison across models and worlds
.venv/bin/python evaluation/run_matrix.py \
    --config evaluation/configs/eval_full_v4_policies.yaml

# summarize a completed run
.venv/bin/python evaluation/aggregate.py --run-id <id>

# plots for the report
.venv/bin/python evaluation/plots.py --run-id <id>
```

Single trial, useful when debugging one world:

```bash
.venv/bin/python evaluation/trial.py --model two_stage  --world-idx 768
.venv/bin/python evaluation/trial.py --model single_stage --world-idx 768
.venv/bin/python evaluation/trial.py --model dwa          --world-idx 768
```

Evaluation disables the safety gate on purpose, so the comparison
measures the policies themselves rather than the classical layer
sitting behind them.

---

## 5. Data collection and training

```bash
# expert demonstrations from the A* planner
.venv/bin/python expert/collect_dataset.py

# train (config selects single- or two-stage and the data pool)
.venv/bin/python -m diffusion.train \
    --config diffusion/configs/two_stage_v5.yaml

# offline metrics for a checkpoint, no simulator needed
.venv/bin/python -m diffusion.evaluate_offline \
    --checkpoint diffusion/runs/two_stage_v5/best.pt
```

---

## 6. World generation

```bash
.venv/bin/python barn/make_dense_worlds.py      # indoor-style clutter
.venv/bin/python barn/make_fork_worlds.py       # branching corridors
.venv/bin/python barn/place_color_targets.py    # add the three swatches
```

---

## 7. Tests

```bash
# no simulator needed
.venv/bin/python tests/test_viewpoint_helpers.py
.venv/bin/python tests/test_refactor_equivalence.py

# simulation smoke tests
python3 ~/unit_tests/run_tests.py -capstone_smoke
```

---

## 8. Cleanup

Simulation processes outlive a killed terminal and a stale one silently
breaks the next run, so clear them before relaunching:

```bash
pkill -9 -f "gzserver|gzclient|policy_node|color_test|rviz"
pkill -f "ros2 daemon"
```

Confirm nothing survived:

```bash
ps aux | grep -E "gzserver|policy_node" | grep -v grep
```

---

## Environment

Set in `~/.bashrc` and needed by every command above:

```bash
TURTLEBOT3_MODEL=burger
LDS_MODEL=LDS-02
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
CYCLONEDDS_URI=file:///home/turtlebot/cyclone_config.xml
ROS_DOMAIN_ID=0
```

A tool that cannot see the robot's topics is usually missing
`CYCLONEDDS_URI` rather than genuinely disconnected.
