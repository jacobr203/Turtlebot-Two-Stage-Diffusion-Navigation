#!/usr/bin/env bash
# Start (or restart) the camera_obstacle_node detached, with the shipped yaml.
# Usage: start_camera_obstacle.sh [logfile]
LOG=${1:-/tmp/camera_obstacle.log}
pkill -9 -f 'camera_obstacle_nod[e]' 2>/dev/null
sleep 1
source /opt/ros/humble/setup.bash
source /home/turtlebot/pathfinding_algorithms/install/setup.bash
setsid nohup ros2 run dwa_camera_avoidance camera_obstacle_node --ros-args \
  --params-file /home/turtlebot/pathfinding_algorithms/dwa_camera_avoidance/config/camera_obstacle_params.yaml \
  -p enable_overhead_detection:=false > "$LOG" 2>&1 < /dev/null &
disown
for i in $(seq 1 15); do
  sleep 1
  grep -q "Camera info received" "$LOG" 2>/dev/null && { grep "Camera info" "$LOG"; exit 0; }
done
echo "node did not report camera info in 15 s — log:"
tail -5 "$LOG" 2>/dev/null
exit 1
