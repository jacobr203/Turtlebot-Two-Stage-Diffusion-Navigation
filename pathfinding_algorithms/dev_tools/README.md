# dev_tools — camera-avoidance dev loop

> Works against Gazebo too: start a sim with `../run.sh camera --sim` (or the
> sim_world launch files) and the viewer shows the sim map/scans unchanged —
> only the `turtlebot3_node` liveness chip stays red (it doesn't exist in sim).

Tooling for iterating on `dwa_camera_avoidance`: a live browser view of the
SLAM inputs, and an edit → rebuild → relaunch loop that fires automatically
on every source change.

## Components

| File | What it does |
|---|---|
| `viewer_server.py` | Standalone live viewer. Subscribes to `/map`, `/scan_relayed`, `/scan_camera`, and the `map→base` TF, and serves a canvas view at **http://localhost:8765** with per-topic rates and node up/down chips. Runs independently of the stack — it stays up across restarts and the chips go red/green as nodes come and go. |
| `viewer.html` | The page `viewer_server.py` serves. Drag to pan, scroll to zoom, double-click to reset. Follows the robot until you pan manually. |
| `restart_camera_stack.sh` | Kills the PC-side stack (never the robot's bringup/camera, the viewer, or the watcher), sends a zero `cmd_vel` so the robot doesn't coast, rebuilds `dwa_avoidance` + `dwa_camera_avoidance`, and relaunches `dwa_camera_with_cartographer.launch.py` headless. If the build fails the stack stays down and the error lands in `/tmp/dwa_build.log`. |
| `watch_and_restart.py` | Polls every `*.py / *.yaml / *.lua / *.xml` in both packages once per second; 2 s after a change settles it runs `restart_camera_stack.sh`. Log: `/tmp/dwa_watch.log`. |

## Launching the viewer (standalone)

```bash
source /opt/ros/humble/setup.bash
source ~/pathfinding_algorithms/install/setup.bash
python3 ~/pathfinding_algorithms/dev_tools/viewer_server.py
# then open http://localhost:8765
```

Background variant (as used in the dev loop):

```bash
nohup python3 ~/pathfinding_algorithms/dev_tools/viewer_server.py \
  > /tmp/dwa_viewer.log 2>&1 &
```

## Full dev loop

```bash
# 1. Robot side (bringup + BRIO camera) must be running.
# 2. Start the viewer (above), then:
bash ~/pathfinding_algorithms/dev_tools/restart_camera_stack.sh   # first launch
nohup python3 ~/pathfinding_algorithms/dev_tools/watch_and_restart.py \
  > /tmp/dwa_watch.log 2>&1 &

# From here, saving any watched file rebuilds and relaunches the stack.
```

## Logs

| Path | Contents |
|---|---|
| `/tmp/dwa_camera_stack.log` | Current stack launch output (truncated on each relaunch) |
| `/tmp/dwa_build.log` | Last `colcon build` output |
| `/tmp/dwa_watch.log` | Watcher activity + restart results |
| `/tmp/dwa_viewer.log` | Viewer server output |
