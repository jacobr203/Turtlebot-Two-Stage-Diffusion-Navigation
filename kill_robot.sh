#!/bin/bash
# Kill all ROS2 and robot processes on the TurtleBot3 (<robot-ip>)

# Robot connection details (gitignored; see robot.env.example)
[ -f "$HOME/pathfinding_algorithms/robot.env" ] && . "$HOME/pathfinding_algorithms/robot.env"

sshpass -p "$ROBOT_PASS" ssh -o StrictHostKeyChecking=no "$ROBOT_USER@$ROBOT_IP" python3 << 'EOF'
import subprocess

targets = [
    ['pkill', '-9', 'ros2'],
    ['pkill', '-9', 'turtlebot3_node'],
    ['pkill', '-9', 'turtlebot3_ros'],
    ['pkill', '-9', 'lds_node'],
    ['pkill', '-9', 'hlds_laser'],
    ['pkill', '-9', 'v4l2_camera_node'],
    ['pkill', '-9', 'robot_state_pub'],
]
for cmd in targets:
    subprocess.run(cmd, capture_output=True)

import time; time.sleep(1)

remaining = subprocess.run(['pgrep', '-a', 'ros'], capture_output=True, text=True)
if remaining.stdout.strip():
    # Kill any survivors by PID
    for line in remaining.stdout.strip().splitlines():
        pid = line.split()[0]
        subprocess.run(['kill', '-9', pid], capture_output=True)
    time.sleep(0.5)

final = subprocess.run(['pgrep', '-a', 'ros'], capture_output=True, text=True)
print("Remaining:", final.stdout.strip() or "NONE — robot fully stopped")
EOF
