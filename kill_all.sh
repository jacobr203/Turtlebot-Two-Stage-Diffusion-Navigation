#!/usr/bin/env bash
# kill_all.sh — nuke all ROS2 processes on both PC and robot, including daemons

# Robot connection details (gitignored; see robot.env.example)
[ -f "$HOME/pathfinding_algorithms/robot.env" ] && . "$HOME/pathfinding_algorithms/robot.env"

# sourced from robot.env above (gitignored)

echo "=== Killing local ROS2 processes ==="
pkill -9 -f "cartographer"    2>/dev/null || true
pkill -9 -f "scan_relay"      2>/dev/null || true
pkill -9 -f "tf_relay"        2>/dev/null || true
pkill -9 -f "dwa_node"        2>/dev/null || true
pkill -9 -f "dwa_camera_node" 2>/dev/null || true
pkill -9 -f "camera_obstacle_node" 2>/dev/null || true
pkill -9 -f "policy_node"     2>/dev/null || true
pkill -9 -f "color_seeker"    2>/dev/null || true
pkill -9 -f "color_test_manager" 2>/dev/null || true
pkill -9 -f "rviz2"           2>/dev/null || true
pkill -9 -f "ros2"            2>/dev/null || true
pkill -9 -f "component_container" 2>/dev/null || true
pkill -9 -f "robot_state_pub" 2>/dev/null || true
pkill -9 -f "nav2_container"  2>/dev/null || true
pkill -9 -f "nav2_route_goal" 2>/dev/null || true
pkill -9 -f "gzserver"        2>/dev/null || true
pkill -9 -f "gzclient"        2>/dev/null || true
pkill -9 -f "spawn_entity"    2>/dev/null || true
pkill -9 -f "map_saver_cli"   2>/dev/null || true

echo "  Stopping ROS2 daemon..."
pkill -9 -f "ros2_daemon"     2>/dev/null || true
pkill -9 -f "_ros2_daemon"    2>/dev/null || true
pkill -9 -f "daemon/__main__" 2>/dev/null || true

echo "  Local done."

echo ""
echo "=== Killing robot ROS2 processes ==="
sshpass -p "$ROBOT_PASS" ssh \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=10 \
    "$ROBOT_USER@$ROBOT_IP" \
    'pgrep -af "ros2|turtlebot3|ld08|component_container|v4l2_camera" | grep -v "pgrep\|sshd\|bash" | awk "{print \$1}" | xargs -r kill -9 2>/dev/null; \
     sleep 1; \
     pgrep -a "ros" | grep -v "pgrep\|sshd\|bash" | awk "{print \$1}" | xargs -r kill -9 2>/dev/null; \
     source /opt/ros/humble/setup.bash 2>/dev/null; ros2 daemon stop 2>/dev/null || true; \
     echo "  Robot done."' 2>&1

echo ""
echo "=== All clear ==="
