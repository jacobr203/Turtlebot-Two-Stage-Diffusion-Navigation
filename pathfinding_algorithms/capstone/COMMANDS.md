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

# alternative colour set (lime green + slightly-darker sky blue); the sim
# world must be regenerated with the same set first:
.venv/bin/python barn/place_color_targets.py --idx 20 --colors lime_green,sky_blue
../run.sh policy-two-stage --3_color_test --colors lime_green,sky_blue --sim 20
```

Known colours: `light_green`, `purple`, `orange`, `lime_green`, `sky_blue`
(HSV bands in `deploy/color_profiles.yaml`; default set is the classic
three). Regenerating a world with `--colors` overwrites its
`worlds_color`/`targets` files — rerun the default command to restore them
(placement is seeded, so the restore is byte-identical). On the real robot
the `real` bands for the new colours still need calibrating against the
printed squares; note `lime_green` overlaps the wide `real` `light_green`
band, so don't enable both in one run until then.

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

## 5. Real-robot trials

Operator-driven runs on the TurtleBot3 (protocol:
`notes/real_robot_data_plan.md`). The real-robot protocol is the
**colour task** in an arena; the point-to-point benchmark tools further
down are kept but not part of it.

### Colour task in an arena

```bash
# Day-0 readiness (read-only; never drives): link, clock offset, sensor
# rates and worst gaps -> the --deadman-s value, battery. Run it idle,
# then again while driving with teleop in another terminal.
.venv/bin/python evaluation/real_trial.py check --camera --seconds 300

# map the arena from the start mark (you teleop), extract the poles
.venv/bin/python evaluation/real_trial.py map-arena --name room1 --id 9101

# after pruning the pole list and adding squares under placements:
.venv/bin/python evaluation/course_tools.py arena-measure evaluation/arenas/room1.yaml

# fix the run order for a placement, before any run
.venv/bin/python evaluation/real_trial.py colour-schedule \
    --arena evaluation/arenas/room1.yaml --placement P1 --run-id colour_v1 --lighting day

# rehearsal (one uncounted run per model), then counted runs
.venv/bin/python evaluation/real_trial.py colour-run \
    --arena evaluation/arenas/room1.yaml --placement P1 --run-id colour_v1 --lighting day \
    --shakedown --deadman-s <D5> --link cable
.venv/bin/python evaluation/real_trial.py colour-run \
    --arena evaluation/arenas/room1.yaml --placement P1 --run-id colour_v1 --lighting day \
    --deadman-s <D5> --link cable --camera-params "-p <name>:=<value>"

.venv/bin/python evaluation/real_trial.py colour-status \
    --arena evaluation/arenas/room1.yaml --placement P1 --run-id colour_v1 --lighting day

# paths for the write-up: every counted run of a placement, side by side by
# model, on the arena map (PNG + vector PDF); re-renders each run's own
# figure too (trials/<tag>/path.png/.pdf)
.venv/bin/python evaluation/real_paths.py --run-id colour_v1 \
    --arena evaluation/arenas/room1.yaml --placement P1 --lighting day

# after a session: results per model, where the notes really were (from the
# detections) and path maps with the notes there, plus data checks
# -> results/real/colour_v1/reports/room1_P1_day/ (report.md + figures)
source /opt/ros/humble/setup.bash
.venv/bin/python evaluation/session_report.py --run-id colour_v1 \
    --arena evaluation/arenas/room1.yaml --placement P1 --lighting day
```

Every colour run locates itself on the arena map from its first lidar
scans (before it moves), so its path is saved in the arena frame:
`path_arena.csv` (t, x, y, yaw; t from the first command) and
`path_events.json` (start pose and match, where each note was first seen
and found, where collisions were logged), plus `path.png`/`.pdf`.

`--lighting` (day or night, as defined in the arena file) picks the
calibrated colour bands and the locked camera settings for that light,
and every row records it; each lighting has its own run order. Add
`--show` to watch the camera and the detector (boxes plus a live mask per
colour) during the session.

Each run fully resets both machines and restarts bringup, starts the
camera, `camera_obstacle_node`, Cartographer, the detector and the
checkpoint manager, then the policy (two-stage `seek`, single-stage
`wander`; 4 Hz, gate on). It ends when every square is FOUND or at the
cap (`--cap-min`, default 15). Keys during a run: `c` log a collision
(the run continues), `w` wedged/unsafe (ends as a counted failure), `a`
abort, `x` harness fault, `space` stop now and classify after.
Results: `results/real/<run-id>/colour_trials.csv`.

### Point-to-point benchmark (courses)

Courses are BARN-style arenas of 15 cm cardboard tubes; each layout in
`evaluation/courses/` lists tube centres in the robot's start frame and
doubles as the two-stage policy's prior map.

```bash
# course difficulty on the BARN tier scale, optimal route, timeout
.venv/bin/python evaluation/course_tools.py measure evaluation/courses/*.yaml

# build sheet: figures + every tape mark, one printable file
# (-> results/real/courses/build_sheet.md and A.png, B.png, C.png)
.venv/bin/python evaluation/course_tools.py sheet evaluation/courses/*.yaml

# after building: map it (teleop + Cartographer from the start mark),
# then check every tube is where the layout says
.venv/bin/python evaluation/course_tools.py check-map \
    evaluation/courses/A.yaml ~/pathfinding_algorithms/maps/real/<...>.yaml

# new layout for a target tier
.venv/bin/python evaluation/course_tools.py design --tier MODERATE \
    --tubes 16 --name B --id 9002 --seed 2
```

Trials, one course per session:

```bash
# fix the run order BEFORE the session (refuses to overwrite)
.venv/bin/python evaluation/real_trial.py schedule \
    --course evaluation/courses/A.yaml --run-id bench_v1

# rehearsal: 2 trials per model at 4 Hz, never counted
.venv/bin/python evaluation/real_trial.py run \
    --course evaluation/courses/A.yaml --run-id bench_v1 --shakedown

# counted trials (refuses uncommitted code or an uncommitted layout)
.venv/bin/python evaluation/real_trial.py run \
    --course evaluation/courses/A.yaml --run-id bench_v1 --link cable

.venv/bin/python evaluation/real_trial.py status \
    --course evaluation/courses/A.yaml --run-id bench_v1

# summary tables, same format as the sim matrix
.venv/bin/python evaluation/aggregate.py --run-id real/bench_v1 \
    --courses evaluation/courses
```

Every trial fully resets both machines and restarts bringup (odometry
only zeroes there), starts a fresh Cartographer for ground truth, and
records a bag. During a trial: `c` collision, `a` abort (outside
interference), `x` harness fault, `space` stop now and classify after.
Aborted and harness-fault cells are re-run at the end of the session.
Results land in `results/real/<run-id>/`.

Useful `run` flags: `--deadman-s` (policy sensor deadman, default 0.6),
`--min-battery-v` (swap threshold, default 11.1), `--link`.

---

## 6. Data collection and training

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

## 7. World generation

```bash
.venv/bin/python barn/make_dense_worlds.py      # indoor-style clutter
.venv/bin/python barn/make_fork_worlds.py       # branching corridors
.venv/bin/python barn/place_color_targets.py    # add the three swatches
```

---

## 8. Tests

```bash
# no simulator needed
.venv/bin/python tests/test_viewpoint_helpers.py
.venv/bin/python tests/test_refactor_equivalence.py
.venv/bin/python tests/test_real_trial.py

# simulation smoke tests
python3 ~/unit_tests/run_tests.py -capstone_smoke
```

---

## 9. Cleanup

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
