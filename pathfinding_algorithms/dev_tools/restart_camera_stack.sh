#!/bin/bash
# Restart the PC-side camera-avoidance stack after an algorithm change.
# Leaves alone: robot bringup, robot camera node, viewer_server, watcher.

WS=/home/turtlebot/pathfinding_algorithms
LOG=/tmp/dwa_camera_stack.log

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/turtlebot/cyclone_config.xml
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=0

source /opt/ros/humble/setup.bash

echo "[restart $(date +%T)] killing PC-side stack..."
pkill -9 -f dwa_camera_node
pkill -9 -f camera_obstacle_node
pkill -9 -f cartographer_node
pkill -9 -f cartographer_occupancy_grid_node
pkill -9 -f scan_relay_node
pkill -9 -f tf_relay_node
pkill -9 -f "dwa_camera_with_cartographer"
sleep 1

# DWA may have died mid-motion — command the robot to stop.
timeout 8 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1 \
  && echo "[restart $(date +%T)] robot stopped (zero cmd_vel sent)" \
  || echo "[restart $(date +%T)] WARNING: could not publish zero cmd_vel"

echo "[restart $(date +%T)] rebuilding dwa_camera_avoidance + dwa_avoidance..."
cd "$WS"
if ! colcon build --symlink-install --packages-select dwa_avoidance dwa_camera_avoidance > /tmp/dwa_build.log 2>&1; then
  echo "[restart $(date +%T)] BUILD FAILED — stack NOT relaunched. See /tmp/dwa_build.log:"
  tail -20 /tmp/dwa_build.log
  exit 1
fi

source "$WS/install/setup.bash"

echo "[restart $(date +%T)] launching dwa_camera_with_cartographer (headless)..."
nohup ros2 launch dwa_camera_avoidance dwa_camera_with_cartographer.launch.py use_rviz:=false \
  > "$LOG" 2>&1 &

sleep 5
if pgrep -f dwa_camera_node >/dev/null && pgrep -f cartographer_node >/dev/null; then
  echo "[restart $(date +%T)] stack up. Log: $LOG"
else
  echo "[restart $(date +%T)] WARNING: stack may not have started — tail of $LOG:"
  tail -15 "$LOG"
  exit 1
fi
