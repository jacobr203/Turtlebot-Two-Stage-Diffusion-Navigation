"""Harness constants: paths, models, timeouts, topics, CSV schemas."""

from pathlib import Path

from common import arena as A

CAPSTONE = Path(__file__).resolve().parents[2]


WS = CAPSTONE.parent


WS_SETUP = WS / 'install' / 'setup.bash'


VENV_PY = CAPSTONE / '.venv' / 'bin' / 'python'


RESULTS = CAPSTONE / 'results' / 'real'


ROBOT_ENV = WS / 'robot.env'


CKPTS = {
    'two_stage': CAPSTONE / 'diffusion/runs/two_stage_v5/best.pt',
    'single_stage': CAPSTONE / 'diffusion/runs/single_stage_v5/best.pt',
}


MODELS = ('two_stage', 'single_stage', 'dwa')


HZ_LIST = (2.0, 3.0, 4.0)


ROUNDS = 3


COUNTED = ('success', 'collision', 'timeout', 'crash')


RERUN = ('aborted', 'harness_error')


POLICY_THREADS = 4 # matches the sim matrix (eval_full_v5_*.yaml)


# Colour task: two-stage vs single-stage, 4 Hz, gate on
COLOUR_MODELS = ('two_stage', 'single_stage')


COLOUR_BEHAVIOR = {'two_stage': 'seek', 'single_stage': 'wander'}


COLOUR_HZ = 4.0


COLOUR_ROUNDS = 5


DEFAULT_CAP_MIN = 15.0 # sim colour runs finished in 5-14 min


CAMERA_WAIT_S = 20.0


LOCALIZE_SCANS = 8 # stationary scans stacked for the start-pose match


START_OFF_M = 0.15 # counted runs must start this close to the mark...


START_OFF_DEG = 10.0 # ...and this close to its heading


START_MATCH_WARN = 0.85


SEEK_MIN_GOAL_DIST = 0.5 # run.sh's --min_goal_dist default


INTERRUPTED = 'Ctrl-C during the run'


COLOUR_BAG_TOPICS = ['/image_raw/compressed', '/camera_info', '/scan_camera',
                     '/color_targets', '/color_test/status']


START_TIMEOUT_S = 45.0 # controller ready -> first command


SCAN_GAP_LIMIT_S = 2.0 # longer lidar silence = harness fault, not the model


SLAM_LOST_LIMIT_S = 3.0


# Path length: bucket-average the trace, then count only steps of at least
# PATH_MIN_STEP_M, so a parked robot's pose jitter adds nothing.
PATH_BUCKET_S = 0.25


PATH_MIN_STEP_M = 0.02


MONITOR_PERIOD_S = 0.1


BRINGUP_TIMEOUT_S = 30.0


SENSOR_WAIT_S = 15.0


NODE_TIMEOUT_S = 60.0 # the policy loads torch + a checkpoint


SLAM_READY_TIMEOUT_S = 30.0


ODOM_ZERO_TOL_M = 0.02


ODOM_ZERO_TOL_RAD = 0.03


STOP_V_TOL = 0.01


STOP_W_TOL = 0.05


BAG_CLOSE_GRACE_S = 15.0


BAG_START_CHECK_S = 2.0 # a recorder that fails (e.g. folder exists) exits fast


DEFAULT_MIN_BATTERY_V = 11.1


# Raise the policy deadman if lidar or odometry ever goes this quiet: the
# 0.6 s default would hold the wheels mid-trial.
DEADMAN_RAISE_GAP_S = 0.5


RAISED_DEADMAN_S = 1.0


CLOCK_OFFSET_OK_MS = 50.0


CLOCK_OFFSET_FAIL_MS = 1000.0


BAG_TOPICS = ['/scan', '/odom', '/tf', '/tf_static', '/cmd_vel', '/imu',
              '/battery_state', '/map', '/policy/waypoints',
              '/policy/subgoal', '/goal_pose', '/rosout']


# Bracketed patterns never match the pkill command line that carries them.
LOCAL_KILL_PATTERNS = [
    'policy_nod[e]', 'dwa_nod[e]', 'cartographer_nod[e]',
    'cartographer_occupancy_grid_nod[e]', 'scan_relay_nod[e]',
    'tf_relay_nod[e]', 'cartographer_only.launch.p[y]', 'bag recor[d]',
    'color_seeke[r]', 'color_test_manage[r]', 'camera_obstacle_nod[e]',
]


def detached(cmd, log):
    """A remote command that outlives the SSH call without holding it open:
    redirecting the whole group releases the channel, which `a && b &`
    would keep, so SSH would wait forever on bringup."""
    return f'({cmd}) > {log} 2>&1 </dev/null &'


ROBOT_KILL = detached("pkill -9 -f '/opt/ro[s]/'; "
                      "pkill -9 -f 'turtlebot[3]'; pkill -9 -f 'ros[2]'",
                      '/tmp/real_trial_kill.log')


ROBOT_ALIVE = "pgrep -af '/opt/ro[s]/|turtlebot[3]|ros[2] ' || true"


ROBOT_BRINGUP = detached(
    'source /opt/ros/humble/setup.bash && '
    'source $HOME/turtlebot3_ws/install/setup.bash && '
    'export TURTLEBOT3_MODEL=burger LDS_MODEL=LDS-02 '
    'RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && '
    'exec ros2 launch turtlebot3_bringup robot.launch.py',
    '/tmp/tbot_bringup.log')


ROW_FIELDS = [
    # sim-compatible (trial.py / aggregate.py)
    'model', 'world_idx', 'hz', 'trial', 'verdict', 'time_to_goal_s',
    'gt_path_len', 'collided_with', 'lat_sample_ms', 'lat_plan_ms',
    'lat_max_ms',
    # real-robot additions
    'run_id', 'course', 'session_id', 'order_idx', 't_start_wall', 'ckpt',
    'git_sha', 'git_dirty', 'odom_path_len', 'min_goal_dist_m',
    'final_goal_dist_m', 'progress', 'collision_source', 'verdict_reason',
    'scan_age_p95_ms', 'scan_gap_max_ms', 'deadman_lines', 'recovery_lines',
    'pre_goal_cmds', 'odom_drift_m', 'battery_v_start', 'battery_v_end',
    'ping_p95_ms', 'stop_verified', 'notes',
]


COLOUR_ROW_FIELDS = [
    'model', 'arena', 'placement', 'lighting', 'hz', 'trial', 'verdict',
    'total_s',
    'found_count', 'colors', 'path_m', 'odom_path_m', 'collisions',
    'collision_times_s',
    *[f'{c}_found_s' for c in A.KNOWN_COLORS],
    *[f'{c}_seen_s' for c in A.KNOWN_COLORS],
    'lat_sample_ms', 'lat_plan_ms', 'lat_max_ms', 'scan_age_p95_ms',
    'scan_gap_max_ms', 'deadman_lines', 'recovery_lines', 'gate_lines',
    'odom_drift_m', 'battery_v_start', 'battery_v_end', 'ping_p95_ms',
    'stop_verified', 'cap_s', 'false_found', 'start_x', 'start_y',
    'start_yaw_deg', 'start_match', 'run_id', 'world_idx',
    'session_id', 'order_idx', 't_start_wall', 'ckpt', 'git_sha',
    'git_dirty', 'verdict_reason', 'pre_goal_cmds', 'collisions_late',
    'notes',
]


SESSION_FIELDS = [
    'session_id', 'run_id', 'course', 'lighting', 'shakedown', 'start',
    'end',
    'git_sha', 'git_dirty', 'link', 'ping_p50_ms', 'ping_p95_ms',
    'clock_offset_ms', 'battery_v_start', 'trials_run', 'notes',
]
