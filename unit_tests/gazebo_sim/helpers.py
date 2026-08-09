"""
Shared helpers for gazebo_sim tests.

Unlike the real-robot suites, nothing here touches SSH or the physical
robot: Gazebo Classic runs headless on this machine and the algorithm
stacks are launched locally with use_sim_time:=true.
"""

import os
import subprocess
import time

import rclpy
import rclpy.node

SIM_LOG = '/tmp/gazebo_sim_test.log'
STACK_LOG = '/tmp/gazebo_sim_stack.log'

_started: list[subprocess.Popen] = []


# ── Environment ───────────────────────────────────────────────────────────────

def ros2_env() -> dict:
    env = os.environ.copy()
    # run_tests.py prepends ~/unit_tests to PYTHONPATH for pytest. Launched
    # ROS nodes must NOT inherit it: unit_tests/dwa_avoidance/ (a test
    # folder) shadows the dwa_avoidance ROS package and its entry points
    # crash with ModuleNotFoundError.
    env['PYTHONPATH'] = ':'.join(
        p for p in env.get('PYTHONPATH', '').split(':')
        if p and not p.rstrip('/').endswith('unit_tests'))
    env['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
    env['CYCLONEDDS_URI'] = 'file:///home/turtlebot/cyclone_config.xml'
    env['TURTLEBOT3_MODEL'] = 'burger'
    env['ROS_DOMAIN_ID'] = '0'
    env['GAZEBO_MODEL_DATABASE_URI'] = ''   # no online model DB stall on WSL2
    return env


def _launch_cmd(ros2_launch_args: str) -> str:
    return (
        'source /opt/ros/humble/setup.bash && '
        'source /home/turtlebot/pathfinding_algorithms/install/setup.bash && '
        + ros2_launch_args
    )


def _popen(ros2_launch_args: str, log: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        _launch_cmd(ros2_launch_args),
        shell=True, executable='/bin/bash',
        stdout=open(log, 'w'), stderr=subprocess.STDOUT,
        env=ros2_env(),
    )
    _started.append(proc)
    return proc


# ── Graph discovery via rclpy ─────────────────────────────────────────────────

def _spin_probe(seconds: float = 1.0):
    tmp = rclpy.create_node('_gazebo_sim_probe')
    try:
        deadline = time.time() + seconds
        while time.time() < deadline:
            rclpy.spin_once(tmp, timeout_sec=0.2)
        return ([n for n, _ in tmp.get_node_names_and_namespaces()],
                {t for t, _ in tmp.get_topic_names_and_types()})
    finally:
        tmp.destroy_node()


def node_alive(name: str) -> bool:
    nodes, _ = _spin_probe()
    return name in nodes


def topic_exists(topic: str) -> bool:
    _, topics = _spin_probe()
    return topic in topics


def wait_for_topic(topic: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if topic_exists(topic):
            return True
    return False


# ── Component starters ────────────────────────────────────────────────────────

def start_gazebo_burger(world: str = 'turtlebot3_world'):
    """Headless Gazebo with the plain burger. Returns once /scan exists."""
    print(f'  [helpers] starting Gazebo ({world}, burger, headless)...')
    _popen(f'ros2 launch dwa_avoidance sim_world.launch.py '
           f'world:={world} headless:=true', SIM_LOG)
    assert wait_for_topic('/scan', 60), \
        f'Gazebo /scan never appeared — see {SIM_LOG}'


def start_gazebo_brio(world: str = 'turtlebot3_world'):
    """Headless Gazebo with the burger_brio (camera) model."""
    print(f'  [helpers] starting Gazebo ({world}, burger_brio, headless)...')
    _popen(f'ros2 launch dwa_camera_avoidance sim_world_brio.launch.py '
           f'world:={world} headless:=true', SIM_LOG)
    assert wait_for_topic('/scan', 60), \
        f'Gazebo /scan never appeared — see {SIM_LOG}'
    assert wait_for_topic('/camera/camera_info', 30), \
        f'Gazebo camera never appeared — see {SIM_LOG}'


def start_lidar_stack():
    print('  [helpers] starting LiDAR DWA + Cartographer (sim time)...')
    _popen('ros2 launch dwa_avoidance dwa_with_cartographer.launch.py '
           'use_sim_time:=true use_rviz:=false', '/tmp/gazebo_sim_lidar.log')
    _wait_node('cartographer_node')


def start_camera_stack():
    print('  [helpers] starting camera DWA + Cartographer (sim time)...')
    _popen('ros2 launch dwa_camera_avoidance dwa_camera_with_cartographer.launch.py '
           'use_sim_time:=true use_rviz:=false '
           'image_topic:=/camera/image_raw/compressed '
           'camera_info_topic:=/camera/camera_info '
           'enable_overhead_detection:=false', '/tmp/gazebo_sim_camera.log')
    _wait_node('cartographer_node')


def start_nav2():
    print('  [helpers] starting Nav2 (sim time, default map)...')
    # navigation2.launch.py has no use_rviz arg — RViz always opens;
    # kill it once the stack is up since this suite runs headless.
    _popen('ros2 launch turtlebot3_navigation2 navigation2.launch.py '
           'use_sim_time:=true '
           'map:=/opt/ros/humble/share/turtlebot3_navigation2/map/map.yaml',
           '/tmp/gazebo_sim_nav2.log')
    _wait_node('bt_navigator')
    subprocess.run(['pkill', '-9', '-f', 'rviz[2]'], capture_output=True)
    publish_initial_pose()


def publish_initial_pose(x: float = -2.0, y: float = -0.5):
    """AMCL never publishes map->odom until it gets an initial pose, which
    blocks the costmaps' TF and stalls lifecycle activation — publish the
    spawn pose so Nav2 can finish coming up."""
    print('  [helpers] publishing /initialpose at spawn...')
    node = rclpy.create_node('_gazebo_sim_initpose')
    try:
        from geometry_msgs.msg import PoseWithCovarianceStamped
        pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        deadline = time.time() + 30
        while pub.get_subscription_count() == 0 and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.07
        for _ in range(3):
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.3)
            time.sleep(0.5)
    finally:
        node.destroy_node()


def _wait_node(name: str, timeout_s: float = 30):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if node_alive(name):
            print(f'  [helpers] {name} up.')
            return
    raise RuntimeError(f'Timed out waiting for {name} ({timeout_s} s) — '
                       f'see {STACK_LOG}')


# ── Teardown ──────────────────────────────────────────────────────────────────

def kill_all_sim():
    """Kill every sim + stack process. Bracketed patterns avoid self-match."""
    patterns = [
        'dwa_camera_nod[e]', 'dwa_nod[e]', 'camera_obstacle_nod[e]',
        'cartographer_nod[e]', 'cartographer_occupancy_grid_nod[e]',
        'scan_relay_nod[e]', 'tf_relay_nod[e]',
        'with_cartographer.launch.p[y]', 'navigation2.launch.p[y]',
        'sim_world.*launch.p[y]', 'component_containe[r]',
        'lifecycle_manage[r]', 'map_serve[r]', 'amc[l]', 'bt_navigato[r]',
        'controller_serve[r]', 'planner_serve[r]', 'behavior_serve[r]',
        'waypoint_followe[r]', 'velocity_smoothe[r]',
        'gzserve[r]', 'gzclien[t]', 'spawn_entit[y]',
        'robot_state_publishe[r]',
    ]
    for p in patterns:
        subprocess.run(['pkill', '-9', '-f', p], capture_output=True)
    for proc in _started:
        try:
            proc.kill()
        except Exception:
            pass
    _started.clear()
    time.sleep(4)   # let gzserver's sockets/locks clear before a restart
