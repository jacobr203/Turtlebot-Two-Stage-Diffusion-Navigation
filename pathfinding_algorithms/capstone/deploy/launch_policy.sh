#!/usr/bin/env bash
# Run the diffusion policy node with the capstone venv python (torch + rclpy).
# All arguments pass through to the node, e.g.:
#   deploy/launch_policy.sh --ros-args -p use_sim_time:=true \
#     -p checkpoint:=diffusion/runs/two_stage_v2/best.pt \
#     -p grid_file:=barn/grids/grid_000.npy -p control_rate_hz:=4.0
set -eo pipefail
CAPSTONE="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$CAPSTONE/../install/setup.bash"
exec "$CAPSTONE/.venv/bin/python" "$CAPSTONE/deploy/policy_node.py" "$@"
