#!/bin/bash
# Unified runner for the project's navigation algorithms, on the real robot
# (default) or in Gazebo simulation (--sim).
#
#   ./run.sh <lidar|camera|nav2|policy> [--sim [world]] [--headless] [--no-rviz] [--map <yaml>]
#
#   lidar    DWA obstacle avoidance from LiDAR + Cartographer SLAM
#   camera   DWA with camera-fused obstacles + Cartographer SLAM
#   nav2     Nav2 (NavFn global + RPP local) on a saved map; on BARN worlds
#            the goal is sent as a through-poses route along the expert A*
#            corridor (NAV2_SINGLE_GOAL=1 for the stock single-goal baseline)
#   policy   capstone diffusion policy + Cartographer SLAM. Default behavior
#            is WANDER (explores toward open space to complete the map,
#            single_stage checkpoint). Extra flags:
#              --goto X Y    drive to (X, Y) instead of wandering — uses the
#                            two_stage checkpoint when a grid is available
#                            (BARN sim world, or --map <yaml> on the robot)
#              --hz N        control rate (default 4)
#              --ckpt <pt>   checkpoint override
#              --path_scale N  EXPERIMENTAL: stretch the diffused waypoint
#                            path spatially (e.g. 2 = double). The trained
#                            horizon stays H=8; this lengthens the strides —
#                            faster, wider arcs, out of training
#                            distribution. Keep 1.0 for evaluation runs.
#   policy-single-stage / policy-two-stage
#            policy with the variant forced. With --3_color_test runs the
#            scavenger test (capstone/notes/color_test_plan.md): find the 3
#            colored squares via the camera; single-stage wanders, two-stage
#            seeks with the live-map A* planner. Sim world may be a BARN
#            index (--sim 306 -> barn/worlds_color/world_306.world) or
#            "random" (--sim random -> random world from the pool). No
#            prior map is used in either variant.
#
#   --sim [world]   run in Gazebo instead of on the robot. Optional world:
#                   turtlebot3_world (default) | house | empty | absolute
#                   .world path (e.g. a BARN world under capstone/barn/worlds)
#   --headless      no Gazebo GUI (sim only)
#   --no-rviz       don't open RViz
#   --map <yaml>    map for nav2, or the obstacle map for 'policy --goto'
#                   (defaults: real = ~/maps_test/my_map.yaml,
#                   sim = turtlebot3_navigation2's map, which matches turtlebot3_world)
#
# Real robot: ensures bringup (and the BRIO camera for 'camera') over SSH
# before launching. Ctrl-C tears everything down; in real mode the robot's
# wheels are stopped and verified via /odom.

set -o pipefail

# Robot connection details live in robot.env, which is gitignored — never
# commit an address or a credential. Copy robot.env.example to robot.env and
# fill it in, or export these in your shell. Simulation runs do not need them.
ROBOT_IP="${ROBOT_IP:-}"
ROBOT_USER="${ROBOT_USER:-}"
ROBOT_PASS="${ROBOT_PASS:-}"
WS=/home/turtlebot/pathfinding_algorithms
[ -f "$WS/robot.env" ] && . "$WS/robot.env"
SIM_LOG=/tmp/sim_world.log
MAPS_ROOT=$WS/maps   # per-run occupancy maps: maps/<sim|real>/<algo>/<stamp>_<algo>

usage() { grep '^#' "$0" | sed -n '2,30p' | sed 's/^# \{0,1\}//'; exit 1; }

# ── Args ──────────────────────────────────────────────────────────────────────
ALGO=${1:-}; shift || true
POLICY_VARIANT=""            # forced by the policy-<variant> aliases
case "$ALGO" in
  policy-single-stage) ALGO=policy; POLICY_VARIANT=single_stage ;;
  policy-two-stage)    ALGO=policy; POLICY_VARIANT=two_stage ;;
esac
[[ "$ALGO" =~ ^(lidar|camera|nav2|policy)$ ]] || usage

SIM=false; WORLD=turtlebot3_world; HEADLESS=false; RVIZ=true; MAP=""
GOTO_X=""; GOTO_Y=""; HZ=4.0; CKPT=""; COLOR_TEST=false; PATH_SCALE=1.0
MIN_GOAL_DIST=0.5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sim)
      SIM=true
      if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
        case "$2" in
          house) WORLD=turtlebot3_house ;;
          empty) WORLD=empty_world ;;
          *)     WORLD=$2 ;;
        esac
        shift
      fi ;;
    --headless) HEADLESS=true ;;
    --no-rviz)  RVIZ=false ;;
    --map)      MAP=$2; shift ;;
    --goto)     GOTO_X=$2; GOTO_Y=$3; shift 2 ;;
    --hz)       HZ=$2; shift ;;
    --ckpt)     CKPT=$2; shift ;;
    --path_scale) PATH_SCALE=$2; shift ;;
    --min_goal_dist) MIN_GOAL_DIST=$2; shift ;;
    --3_color_test) COLOR_TEST=true ;;
    *) echo "unknown arg: $1"; usage ;;
  esac
  shift
done

if $COLOR_TEST; then
  [ "$ALGO" = policy ] || { echo "--3_color_test needs a policy-* algo"; exit 1; }
  [ -n "$GOTO_X" ] && { echo "--3_color_test and --goto are exclusive"; exit 1; }
  # single-stage wanders; two-stage runs the live-map A* seeker (plan §5/§6)
  POLICY_VARIANT=${POLICY_VARIANT:-single_stage}
elif [ "$POLICY_VARIANT" = two_stage ] && [ -z "$GOTO_X" ]; then
  echo "policy-two-stage needs --goto X Y (grid-conditioned) or --3_color_test:"
  echo "plain wander has no subgoal source for the two_stage conditioning."
  exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
if ip link show eth0 2>/dev/null | grep -q "state UP"; then
  export CYCLONEDDS_URI=file:///home/turtlebot/cyclone_config.xml
else
  # eth0 down (WSL2 drops it after host sleep, 2026-08-04): the pinned
  # eth0-multicast config then fails ALL local DDS init ("rmw handle is
  # invalid"), sim included. Default config falls back to loopback —
  # sim is fully local; real-robot mode needs eth0 up regardless.
  export CYCLONEDDS_URI=
  echo "[run] eth0 DOWN — default CycloneDDS config (loopback only)"
fi
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=0   # real robot's domain (sim overrides below)
export GAZEBO_MODEL_DATABASE_URI=""   # prevent online model-DB stall on WSL2
# turtlebot3 meshes (model://turtlebot3_common/...) are not on any default
# path — without this the robot is INVISIBLE in gzclient (physics still
# works: collisions are inline geometry, so headless runs never noticed)
export GAZEBO_MODEL_PATH="/opt/ros/humble/share/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH:-}"

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
  # bracketed patterns so pkill never matches this script's own cmdline
  pkill -9 -f 'dwa_camera_nod[e]';  pkill -9 -f 'dwa_nod[e]'
  pkill -9 -f 'camera_obstacle_nod[e]'
  pkill -9 -f 'cartographer_nod[e]'; pkill -9 -f 'cartographer_occupancy_grid_nod[e]'
  pkill -9 -f 'scan_relay_nod[e]';  pkill -9 -f 'tf_relay_nod[e]'
  pkill -9 -f 'policy_nod[e]'
  pkill -9 -f 'color_seeke[r]'; pkill -9 -f 'color_test_manage[r]'
  pkill -9 -f 'with_cartographer.launch.p[y]'
  pkill -9 -f 'cartographer_only.launch.p[y]'
  pkill -9 -f 'navigation2.launch.p[y]'
  # nav2 composes everything (amcl/planner/controller/bt) into one container;
  # pkill -9 of the launch orphans it and a survivor fights the next session
  # over /tf and /cmd_vel
  pkill -9 -f 'nav2_containe[r]'
  pkill -9 -f 'nav2_route_goa[l]'
  pkill -9 -f 'rviz[2]'
  pkill -9 -f 'rqt_image_vie[w]'
  pkill -9 -f 'sim_world.*launch.p[y]'
  pkill -9 -f 'gzserve[r]'; pkill -9 -f 'gzclien[t]'
  pkill -9 -f 'spawn_entit[y]'; pkill -9 -f 'robot_state_publishe[r]'
  pkill -9 -f 'map_saver_cl[i]'
  return 0
}

stop_robot() {  # publish zeros on a MATCHED subscription (pub --once can miss)
  timeout 15 python3 - <<'EOF' 2>/dev/null
import time
import rclpy
from geometry_msgs.msg import Twist
rclpy.init()
n = rclpy.create_node('_run_sh_stop')
pub = n.create_publisher(Twist, '/cmd_vel', 10)
deadline = time.time() + 8
while pub.get_subscription_count() == 0 and time.time() < deadline:
    rclpy.spin_once(n, timeout_sec=0.2)
for _ in range(10):
    pub.publish(Twist())
    time.sleep(0.1)
EOF
}

verify_stopped() {  # /odom must report ~zero velocity — catches a wedged OpenCR
  timeout 20 python3 - <<'EOF' 2>/dev/null
import sys, time
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
rclpy.init()
n = rclpy.create_node('_run_sh_verifystop')
S = {}
qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
n.create_subscription(Odometry, '/odom', lambda m: S.__setitem__(
    'v', (abs(m.twist.twist.linear.x), abs(m.twist.twist.angular.z))), qos)
t0 = time.time()
while time.time() - t0 < 15:
    rclpy.spin_once(n, timeout_sec=0.2)
    v = S.get('v')
    if v and v[0] < 0.01 and v[1] < 0.05:
        sys.exit(0)
sys.exit(1)
EOF
}

reset_robot_motors() {  # last resort: restart bringup to re-init the OpenCR
  echo "[run] WARNING: robot still reports motion — restarting bringup to reset the motor controller..."
  ssh_robot_bg 'pkill -9 -f ros2; pkill -9 -f turtlebot3; sleep 2;
    source /opt/ros/humble/setup.bash &&
    source /home/$ROBOT_USER/turtlebot3_ws/install/setup.bash &&
    export TURTLEBOT3_MODEL=burger LDS_MODEL=LDS-02 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp &&
    nohup ros2 launch turtlebot3_bringup robot.launch.py > /tmp/tbot_bringup.log 2>&1 &'
  sleep 12
  stop_robot
}

MAP_SAVER_PID=""
MAP_BASE=""

start_map_saver() {  # start_map_saver <sim|real> <algo> — save /map every 30s
  local dir="$MAPS_ROOT/$1/$2"
  mkdir -p "$dir"
  MAP_BASE="$dir/$(date +%Y-%m-%d_%H-%M-%S)_${ALGO}"
  echo "[run] live map -> ${MAP_BASE}.pgm/.yaml (refreshed every 30 s)"
  (
    sleep 25   # give Cartographer time to publish a first map
    while true; do
      timeout 25 ros2 run nav2_map_server map_saver_cli -f "$MAP_BASE" \
        --ros-args -p save_map_timeout:=15.0 >/dev/null 2>&1
      sleep 30
    done
  ) &
  MAP_SAVER_PID=$!
}

_cleaned=false
on_exit() {
  $_cleaned && return; _cleaned=true
  echo; echo "[run] shutting down..."
  if [ -n "$MAP_SAVER_PID" ]; then
    kill "$MAP_SAVER_PID" 2>/dev/null
    # final snapshot while the stack is still up
    timeout 15 ros2 run nav2_map_server map_saver_cli -f "$MAP_BASE" \
      --ros-args -p save_map_timeout:=8.0 >/dev/null 2>&1 \
      && echo "[run] final map saved: ${MAP_BASE}.pgm"
  fi
  # Kill ALL nodes first, THEN stop the wheels — in the other order a
  # dying DWA node can publish one last command after the zero and the
  # OpenCR executes it forever (no cmd_vel timeout on the TB3).
  cleanup
  if ! $SIM; then
    stop_robot
    if verify_stopped; then
      echo "[run] robot stopped (verified via /odom)."
    else
      reset_robot_motors
      if verify_stopped; then
        echo "[run] robot stopped after motor-controller reset."
      else
        echo "[run] !!! ROBOT MAY STILL BE MOVING — hit its power switch !!!"
      fi
    fi
  fi
}
trap on_exit EXIT INT TERM

echo "[run] cleaning up leftover processes..."
cleanup
sleep 1

# ── Helpers ───────────────────────────────────────────────────────────────────
node_alive() {  # node_alive <node_name> — rclpy graph check (CLI daemon is flaky)
  timeout 15 python3 - "$1" <<'EOF'
import sys
import rclpy
rclpy.init()
n = rclpy.create_node('_run_sh_check')
for _ in range(15):
    rclpy.spin_once(n, timeout_sec=0.2)
sys.exit(0 if sys.argv[1] in [name for name, _ in n.get_node_names_and_namespaces()] else 1)
EOF
}

topic_exists() {  # topic_exists <topic>
  timeout 15 python3 - "$1" <<'EOF'
import sys
import rclpy
rclpy.init()
n = rclpy.create_node('_run_sh_check')
for _ in range(15):
    rclpy.spin_once(n, timeout_sec=0.2)
sys.exit(0 if sys.argv[1] in {t for t, _ in n.get_topic_names_and_types()} else 1)
EOF
}

wait_for_topic() {  # wait_for_topic <topic> <timeout_s>
  local deadline=$(( $(date +%s) + $2 ))
  while ! topic_exists "$1"; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[run] TIMEOUT waiting for $1 — last sim log lines:"; tail -10 "$SIM_LOG"
      exit 1
    fi
    echo "[run]   waiting for $1 ..."
  done
  echo "[run] $1 is up."
}

require_robot_creds() {
  [ -n "$ROBOT_IP" ] && [ -n "$ROBOT_USER" ] && return 0
  echo "[run] real-robot mode needs ROBOT_IP and ROBOT_USER."
  echo "[run] copy robot.env.example to robot.env and fill it in"
  echo "[run] (robot.env is gitignored), or export them in your shell."
  exit 1
}

ssh_robot_bg() {  # fire-and-forget remote command
  require_robot_creds
  sshpass -p "$ROBOT_PASS" ssh -o StrictHostKeyChecking=no \
    "$ROBOT_USER@$ROBOT_IP" "$1" >/dev/null 2>&1 &
}

suppress_rviz() {  # turtlebot3_navigation2's launch has no rviz switch —
  ( for _ in $(seq 1 60); do   # honor --no-rviz by killing it once it appears
      pgrep -f 'rviz[2]' >/dev/null && { pkill -9 -f 'rviz[2]'; break; }
      sleep 1
    done ) &
}

# ── Real-robot sensor bringup ─────────────────────────────────────────────────
ensure_bringup() {
  if node_alive turtlebot3_node; then
    echo "[run] bringup already running."
    return
  fi
  echo "[run] starting robot bringup via SSH..."
  ssh_robot_bg 'source /opt/ros/humble/setup.bash &&
    source /home/$ROBOT_USER/turtlebot3_ws/install/setup.bash &&
    export TURTLEBOT3_MODEL=burger LDS_MODEL=LDS-02 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp &&
    nohup ros2 launch turtlebot3_bringup robot.launch.py > /tmp/tbot_bringup.log 2>&1 &'
  for i in $(seq 1 20); do
    sleep 1
    node_alive turtlebot3_node && { echo "[run] bringup up after ${i}s."; return; }
  done
  echo "[run] ERROR: robot bringup did not come up in 20s."; exit 1
}

ensure_camera() {
  if topic_exists /image_raw/compressed; then
    echo "[run] robot camera already publishing."
    return
  fi
  echo "[run] starting BRIO camera via SSH..."
  ssh_robot_bg 'source /opt/ros/humble/setup.bash &&
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp &&
    nohup ros2 run v4l2_camera v4l2_camera_node --ros-args \
      -p video_device:=/dev/video0 -p image_size:=[640,480] \
      > /tmp/tbot_camera.log 2>&1 &'
  for i in $(seq 1 15); do
    sleep 1
    topic_exists /image_raw/compressed && { echo "[run] camera up after ${i}s."; return; }
  done
  echo "[run] ERROR: camera did not come up in 15s."; exit 1
}

auto_initial_pose() {  # background: wait for AMCL, then publish the spawn pose
  timeout 90 python3 - <<'EOF'
import time
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped

rclpy.init()
n = rclpy.create_node('_run_sh_initpose')
pub = n.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
deadline = time.time() + 80
while pub.get_subscription_count() == 0 and time.time() < deadline:
    rclpy.spin_once(n, timeout_sec=0.2)
import math
import os
msg = PoseWithCovarianceStamped()
msg.header.frame_id = 'map'
msg.pose.pose.position.x = float(os.environ.get('SPAWN_X', '-2.0'))
msg.pose.pose.position.y = float(os.environ.get('SPAWN_Y', '-0.5'))
yaw = float(os.environ.get('SPAWN_YAW', '0.0'))
msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
msg.pose.covariance[0] = msg.pose.covariance[7] = 0.25
msg.pose.covariance[35] = 0.07
for _ in range(3):
    pub.publish(msg)
    rclpy.spin_once(n, timeout_sec=0.3)
    time.sleep(0.5)
print(f'[run] initial pose published at spawn '
      f'({msg.pose.pose.position.x}, {msg.pose.pose.position.y})')
EOF
}

# ── Sim world bringup ─────────────────────────────────────────────────────────
auto_barn_goal() {  # background: wait for bt_navigator, send the BARN goal
  if [ -z "${NAV2_SINGLE_GOAL:-}" ] && [ -n "$GRID" ] && [ -f "$GRID" ]; then
    # Exact grid available: pin Nav2's global route to the expert A* corridor
    # via NavigateThroughPoses. Single-goal NavFn routing corner-hugs into
    # scatter pockets RPP refuses on the winding worlds (decisions #24/#24a).
    # NAV2_SINGLE_GOAL=1 restores the stock single-goal baseline behavior.
    timeout 900 "$CAPSTONE/.venv/bin/python" "$CAPSTONE/barn/nav2_route_goal.py" \
      --grid "$GRID"
    if [ $? -eq 3 ] && [ -n "${WEDGE_FLAG:-}" ]; then
      # gazebo physics wedge (decisions #26): flag the sim nav2 loop to
      # rebuild the whole stack, and kill the launch to unblock its wait
      touch "$WEDGE_FLAG"
      pkill -9 -f 'navigation2.launch.p[y]'
      pkill -9 -f 'nav2_containe[r]'
    fi
    return
  fi
  timeout 180 python3 - <<'PYGOAL'
import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose

rclpy.init()
n = Node('_run_sh_barn_goal')
ac = ActionClient(n, NavigateToPose, '/navigate_to_pose')
if not ac.wait_for_server(timeout_sec=150):
    raise SystemExit(1)
time.sleep(8.0)   # let AMCL take the auto initial pose first
g = NavigateToPose.Goal()
g.pose.header.frame_id = 'map'
g.pose.pose.position.x = -2.25
g.pose.pose.position.y = 13.0
g.pose.pose.orientation.w = 1.0
fut = ac.send_goal_async(g)
deadline = time.time() + 15
while not fut.done() and time.time() < deadline:
    rclpy.spin_once(n, timeout_sec=0.2)
print('[run] BARN goal (-2.25, 13.0) sent:',
      bool(fut.done() and fut.result() and fut.result().accepted), flush=True)
rclpy.shutdown()
PYGOAL
}

start_sim_world() {
  local pkg=$1 file=$2; shift 2   # remaining args pass through to the launch
  echo "[run] starting Gazebo ($WORLD, headless=$HEADLESS) — log: $SIM_LOG"
  nohup ros2 launch "$pkg" "$file" world:="$WORLD" headless:="$HEADLESS" "$@" \
    > "$SIM_LOG" 2>&1 &
  wait_for_topic /scan 60
}

# ── Capstone policy runner ────────────────────────────────────────────────────
CAPSTONE=$WS/capstone
SPAWN_X=-2.0; SPAWN_Y=-0.5; SPAWN_YAW=0.0; GRID=""

start_policy_slam() {  # start_policy_slam <use_sim_time>
  if ! $SIM && [ "${POLICY_FUSED_SLAM:-0}" = "1" ]; then
    # EXPERIMENTAL opt-in: FUSED LiDAR + BRIO SLAM (couch-height obstacles
    # in the map). Two live tests DIVERGED (map islands). Root cause found
    # 2026-07-12: sparse camera-only trajectory nodes (1-11 points) matched
    # anywhere at ~90% score, injecting bogus multi-meter loop closures —
    # fixed with num_accumulated_range_data = 2 in the fused lua (not yet
    # re-validated live). Stamps were fine (stamp_lag + relays held up).
    # Enable with:
    #   POLICY_FUSED_SLAM=1 ./run.sh policy
    ensure_camera
    echo "[run] starting FUSED Cartographer (LiDAR + camera, EXPERIMENTAL) — log: /tmp/policy_carto.log"
    nohup ros2 launch dwa_camera_avoidance camera_slam_only.launch.py \
      use_sim_time:=$1 > /tmp/policy_carto.log 2>&1 &
  else
    if ! $SIM; then
      # BRIO → /scan_camera for the policy SAFETY GATE only: below/above-plane
      # obstacles the lidar can't see. Not fed to Cartographer (fused SLAM
      # remains the POLICY_FUSED_SLAM experiment) or the policy's conditioning.
      ensure_camera
      echo "[run] starting camera_obstacle_node (BRIO → policy safety gate) — log: /tmp/policy_camera.log"
      nohup ros2 run dwa_camera_avoidance camera_obstacle_node --ros-args \
        --params-file "$WS/dwa_camera_avoidance/config/camera_obstacle_params.yaml" \
        > /tmp/policy_camera.log 2>&1 &
    fi
    echo "[run] starting Cartographer (LiDAR-only map completion) — log: /tmp/policy_carto.log"
    # sim: odometry-aided SLAM (Gazebo odom stamps are consistent) — pure
    # scan matching diverged on fast seek rotations, twice on world 613.
    # real: no_odom stays (OpenCR board-uptime stamps, lua header).
    local carto_lua=turtlebot3_no_odom.lua
    [ "$1" = true ] && carto_lua=turtlebot3_with_odom.lua
    nohup ros2 launch dwa_avoidance cartographer_only.launch.py \
      use_sim_time:=$1 config_basename:=$carto_lua \
      > /tmp/policy_carto.log 2>&1 &
  fi
  if $RVIZ; then
    nohup rviz2 -d "$CAPSTONE/deploy/policy_mapping.rviz" >/dev/null 2>&1 &
  fi
}

start_color_test() {  # start_color_test <sim|real> — seeker + checkpoint manager
  local profile=$1 img_topic=/image_raw/compressed info_topic=/camera_info
  if [ "$profile" = sim ]; then
    img_topic=/camera/image_raw/compressed; info_topic=/camera/camera_info
  fi
  echo "[run] 3-color test: seeker + manager (profile $profile) — logs /tmp/color_{seeker,test}.log"
  nohup "$CAPSTONE/.venv/bin/python" "$CAPSTONE/deploy/color_seeker.py" \
    --ros-args -p profile:=$profile -p image_topic:=$img_topic \
    -p compressed:=true -p camera_info_topic:=$info_topic \
    > /tmp/color_seeker.log 2>&1 &
  nohup "$CAPSTONE/.venv/bin/python" "$CAPSTONE/deploy/color_test_manager.py" \
    --ros-args -p results_csv:=/tmp/color_test_results.csv \
    > /tmp/color_test.log 2>&1 &
  # camera POV window (what the color detector sees) — suppressed with --no-rviz
  $RVIZ && nohup env LIBGL_ALWAYS_SOFTWARE=1 \
    ros2 run rqt_image_view rqt_image_view "$img_topic" \
    > /tmp/rqt_image.log 2>&1 &
}

run_policy() {  # run_policy <use_sim_time> — foreground; Ctrl-C lands here
  local simtime=$1 extra=""
  local behavior=wander
  # policy-<variant> aliases pin the checkpoint; in the color test the
  # two-stage variant runs the live-map A* seeker (no prior grid, sim+real)
  [ -n "$POLICY_VARIANT" ] && \
    CKPT=${CKPT:-$CAPSTONE/diffusion/runs/${POLICY_VARIANT}_v4/best.pt}
  $COLOR_TEST && [ "$POLICY_VARIANT" = two_stage ] && behavior=seek
  if [ -n "$GOTO_X" ]; then
    behavior=goto
    [ -z "$GRID" ] && [ -n "$MAP" ] && GRID=$MAP
    if [ -n "$GRID" ]; then
      CKPT=${CKPT:-$CAPSTONE/diffusion/runs/two_stage_v4/best.pt}
      extra="-p grid_file:=$GRID"
    else
      echo "[run] no grid/map available — goto falls back to single_stage"
      CKPT=${CKPT:-$CAPSTONE/diffusion/runs/single_stage_v4/best.pt}
    fi
    extra="$extra -p goal_x:=$GOTO_X -p goal_y:=$GOTO_Y"
  else
    # wandering explores unknown space: goal-conditioned checkpoint
    CKPT=${CKPT:-$CAPSTONE/diffusion/runs/single_stage_v4/best.pt}
  fi
  echo "[run] policy: behavior=$behavior  hz=$HZ  ckpt=$CKPT"
  echo "[run] Ctrl-C stops everything (robot wheels included)."
  "$CAPSTONE/deploy/launch_policy.sh" --ros-args \
    -p use_sim_time:=$simtime -p checkpoint:="$CKPT" -p behavior:=$behavior \
    -p control_rate_hz:=$HZ -p path_scale:=$PATH_SCALE \
    -p seek_min_goal_dist:=$MIN_GOAL_DIST \
    -p spawn_x:=$SPAWN_X -p spawn_y:=$SPAWN_Y -p spawn_yaw:=$SPAWN_YAW \
    -p latency_log:=/tmp/policy_lat_$($SIM && echo sim || echo real)_$$.csv $extra
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
if $SIM; then
  # Sim lives on its own ROS domain: a sim session and a real-robot session
  # running at the same time on domain 0 MERGE their topics — each policy
  # node reads a blend of both robots' /scan and /odom (observed: the real
  # robot drove blind on the sim's clear scan). Manual introspection of a
  # sim run needs: export ROS_DOMAIN_ID=7
  export ROS_DOMAIN_ID=7
  echo "[run] sim uses ROS_DOMAIN_ID=7 (real robot stays on 0 — concurrent sessions can't collide)"
  export TURTLEBOT3_MODEL=burger   # nav2/cartographer param selection
  # color test: resolve the world token to the color-target variant
  # (numeric index or a plain barn/worlds path both map into worlds_color).
  # Generation can fail for a world (place_color_targets constraints) —
  # refuse to fall back to the plain world rather than run a 0-target test.
  # --sim random: pick a random generated world (color pool for the color
  # test, plain barn pool otherwise); prints the pick so the run can be
  # reproduced with an explicit index
  if [[ "$WORLD" == random ]]; then
    pool=$CAPSTONE/barn/worlds
    $COLOR_TEST && pool=$CAPSTONE/barn/worlds_color
    WORLD=$(ls "$pool"/world_*.world | shuf -n1)
    echo "[run] --sim random -> $(basename "$WORLD")"
  fi
  if $COLOR_TEST; then
    if [[ "$WORLD" =~ ^[0-9]+$ ]]; then
      WORLD=$CAPSTONE/barn/worlds_color/world_$(printf '%03d' "$WORLD").world
    elif [[ "$WORLD" == */barn/worlds/* ]]; then
      WORLD=${WORLD/\/barn\/worlds\//\/barn\/worlds_color\/}
    fi
    [[ "$WORLD" == */worlds_color/* && -f "$WORLD" ]] || {
      echo "[run] --3_color_test needs a generated color world (got: $WORLD)"
      echo "[run] generate with: capstone/.venv/bin/python capstone/barn/place_color_targets.py --idx N"
      exit 1; }
  fi
  # BARN worlds (any mode): spawn at the official start; note the index for
  # the world's grid (policy two-stage) and generated map (nav2)
  BARN_IDX=""
  if [[ "$WORLD" == */capstone/barn/worlds*/world_*.world ]]; then
    SPAWN_X=-2.25; SPAWN_Y=3.0; SPAWN_YAW=1.5708
    BARN_IDX=$(basename "$WORLD" .world); BARN_IDX=${BARN_IDX#world_}
    GRID=$CAPSTONE/barn/grids/grid_${BARN_IDX}.npy
  fi
  export SPAWN_X SPAWN_Y SPAWN_YAW
  case "$ALGO" in
    lidar)
      start_sim_world dwa_avoidance sim_world.launch.py
      start_map_saver sim lidar
      echo "[run] launching LiDAR DWA + Cartographer (sim time)..."
      ros2 launch dwa_avoidance dwa_with_cartographer.launch.py \
        use_sim_time:=true use_rviz:=$RVIZ
      ;;
    camera)
      start_sim_world dwa_camera_avoidance sim_world_brio.launch.py
      wait_for_topic /camera/camera_info 30
      start_map_saver sim camera
      echo "[run] launching camera DWA + Cartographer (sim time)..."
      ros2 launch dwa_camera_avoidance dwa_camera_with_cartographer.launch.py \
        use_sim_time:=true use_rviz:=$RVIZ \
        image_topic:=/camera/image_raw/compressed \
        camera_info_topic:=/camera/camera_info \
        enable_overhead_detection:=false
      ;;
    nav2)
      NAV2_ARGS=()
      if [ -z "$MAP" ] && [ -n "$BARN_IDX" ]; then
        MAP=$CAPSTONE/barn/maps/map_${BARN_IDX}.yaml
        [ -f "$MAP" ] || "$CAPSTONE/.venv/bin/python" \
          "$CAPSTONE/barn/grids_to_maps.py" --idx "$BARN_IDX"
        # stock TB3 inflation (and the stale values some installs resolve)
        # swamps BARN-width openings — pass the tuned params explicitly
        NAV2_ARGS+=(params_file:=$CAPSTONE/barn/nav2_barn_params.yaml)
      fi
      MAP=${MAP:-/opt/ros/humble/share/turtlebot3_navigation2/map/map.yaml}
      # Retry loop: gazebo on WSL2 can enter a contact degeneracy where the
      # robot ignores motion commands (decisions #26). The route sender
      # detects it (exit 3 -> WEDGE_FLAG via auto_barn_goal); only a fresh
      # gzserver clears it, and a gazebo-only restart would reset sim time
      # to zero under a live Nav2 — so rebuild the whole stack.
      export WEDGE_FLAG=/tmp/run_sh_wedge_$$
      for _wedge_attempt in 1 2 3; do
        rm -f "$WEDGE_FLAG"
        start_sim_world dwa_avoidance sim_world.launch.py \
          x_pose:=$SPAWN_X y_pose:=$SPAWN_Y yaw_pose:=$SPAWN_YAW
        echo "[run] launching Nav2 (sim time, map: $MAP)..."
        echo "[run] initial pose will be set automatically at the spawn pose."
        $RVIZ || suppress_rviz
        auto_initial_pose &
        if [ -n "$BARN_IDX" ]; then
          echo "[run] BARN world: the official goal (-2.25, 13.0) will be sent"
          echo "[run] automatically once Nav2 is active (new goals: RViz 'Nav2 Goal')."
          [ -f "$GRID" ] && echo "[run] routing via the expert A* corridor (NavigateThroughPoses)."
          auto_barn_goal &
        fi
        # background + wait (not foreground): a second Ctrl-C during launch's
        # shutdown derails its kill escalation and the nav2 container wedges —
        # bash then defers the trap until launch exits, hanging the terminal.
        # wait is interruptible, so the trap fires at once and cleanup reaps
        # the whole stack (launch + container patterns are in cleanup()).
        ros2 launch turtlebot3_navigation2 navigation2.launch.py \
          use_sim_time:=true map:="$MAP" "${NAV2_ARGS[@]}" &
        wait $!
        $_cleaned && break                  # Ctrl-C already tore us down
        [ -f "$WEDGE_FLAG" ] || break       # normal end — no wedge
        [ "$_wedge_attempt" -lt 3 ] || break
        echo "[run] SIM WEDGE detected (known WSL2 flake, decisions #26): robot"
        echo "[run] ignored motion commands. Restarting the full sim stack"
        echo "[run] (restart $_wedge_attempt of 2)..."
        cleanup
        # let WSL2's multicast/discovery settle — restarting too fast can
        # re-trigger the DDS-collapse variant of the wedge on the new stack
        sleep 12
      done
      rm -f "$WEDGE_FLAG"
      ;;
    policy)
      # bumper-model burger: encoder odom (re-anchors at spawn, like the
      # real robot) — required by the policy's frame convention. The color
      # test uses the bumper+BRIO splice (same odom + contact, adds camera).
      MODEL_SDF=$CAPSTONE/models/turtlebot3_burger_bumper/model.sdf
      $COLOR_TEST && MODEL_SDF=$CAPSTONE/models/turtlebot3_burger_bumper_brio/model.sdf
      start_sim_world dwa_avoidance sim_world.launch.py \
        model_sdf:="$MODEL_SDF" \
        x_pose:=$SPAWN_X y_pose:=$SPAWN_Y yaw_pose:=$SPAWN_YAW
      if $COLOR_TEST; then
        wait_for_topic /camera/image_raw/compressed 30
        start_color_test sim
      fi
      start_policy_slam true
      start_map_saver sim policy
      run_policy true
      ;;
  esac
else
  case "$ALGO" in
    lidar)
      ensure_bringup
      start_map_saver real lidar
      echo "[run] launching LiDAR DWA + Cartographer (real robot)..."
      ros2 launch dwa_avoidance dwa_with_cartographer.launch.py use_rviz:=$RVIZ
      ;;
    camera)
      ensure_bringup
      ensure_camera
      start_map_saver real camera
      echo "[run] launching camera DWA + Cartographer (real robot)..."
      ros2 launch dwa_camera_avoidance dwa_camera_with_cartographer.launch.py \
        use_rviz:=$RVIZ
      ;;
    nav2)
      ensure_bringup
      MAP=${MAP:-$HOME/maps_test/my_map.yaml}
      echo "[run] launching Nav2 (real robot, map: $MAP)..."
      $RVIZ || suppress_rviz
      # background + wait: see the sim nav2 branch (double-Ctrl-C wedge)
      ros2 launch turtlebot3_navigation2 navigation2.launch.py map:="$MAP" &
      wait $!
      ;;
    policy)
      ensure_bringup
      # the robot's odom zeroed at bringup: its start pose IS the origin.
      # --goto coordinates (and any --map for two-stage) are in that frame.
      SPAWN_X=0.0; SPAWN_Y=0.0; SPAWN_YAW=0.0
      start_policy_slam false
      $COLOR_TEST && start_color_test real
      start_map_saver real policy
      run_policy false
      ;;
  esac
fi
