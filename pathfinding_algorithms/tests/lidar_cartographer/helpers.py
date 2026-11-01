"""Shared helpers for the real-robot test suites: discovery via rclpy, launches via subprocess."""

import os
import subprocess
import time
from pathlib import Path

import rclpy
import rclpy.node

# Robot connection details come from the environment; see robot.env.example.
ROBOT_IP   = os.environ.get('ROBOT_IP', '')
ROBOT_USER = os.environ.get('ROBOT_USER', '')
ROBOT_PASS = os.environ.get('ROBOT_PASS', '')

_started: list[subprocess.Popen] = []


WS = Path(__file__).resolve().parents[2]


def ros2_env() -> dict:
    env = os.environ.copy()
    # run_tests.py prepends tests/ to PYTHONPATH, where tests/dwa_avoidance/
    # would shadow the ROS package and crash launched nodes.
    env['PYTHONPATH'] = ':'.join(
        p for p in env.get('PYTHONPATH', '').split(':')
        if p and not p.rstrip('/').endswith('/tests'))
    env['RMW_IMPLEMENTATION']  = 'rmw_cyclonedds_cpp'
    env['CYCLONEDDS_URI']      = f'file://{WS}/config/cyclone_config.xml'
    env['TURTLEBOT3_MODEL']    = 'burger'
    env['ROS_DOMAIN_ID']       = '0'
    env['DISPLAY']             = ':0'
    return env


def _launch_cmd(ros2_launch_args: str) -> str:
    return (
        'source /opt/ros/humble/setup.bash && '
        f'source {WS}/install/setup.bash && '
        + ros2_launch_args
    )


def _ssh(remote_cmd: str, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['sshpass', '-p', ROBOT_PASS,
         'ssh', '-o', 'StrictHostKeyChecking=no',
         f'{ROBOT_USER}@{ROBOT_IP}', remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _ssh_background(remote_cmd: str) -> None:
    """Fire-and-forget SSH: the remote command runs detached."""
    subprocess.Popen(
        ['sshpass', '-p', ROBOT_PASS,
         'ssh', '-o', 'StrictHostKeyChecking=no',
         f'{ROBOT_USER}@{ROBOT_IP}', remote_cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _node_names() -> list[str]:
    """Return all node names currently on the ROS2 graph."""
    tmp = rclpy.create_node('_helper_discovery')
    try:
        rclpy.spin_once(tmp, timeout_sec=1.0)
        return [name for name, _ in tmp.get_node_names_and_namespaces()]
    finally:
        tmp.destroy_node()


def is_bringup_running() -> bool:
    return 'turtlebot3_node' in _node_names()


def is_cartographer_running() -> bool:
    return 'cartographer_node' in _node_names()


def is_scan_relay_running() -> bool:
    return 'scan_relay_node' in _node_names()


def is_tf_relay_running() -> bool:
    return 'tf_relay_node' in _node_names()


def is_dwa_running() -> bool:
    return 'dwa_node' in _node_names()


def ensure_bringup() -> bool:
    """Start robot bringup via SSH if needed; True if this call started it."""
    if is_bringup_running():
        print('  [helpers] bringup already running — skipping start')
        return False

    print('  [helpers] starting robot bringup via SSH (background)...')
    _ssh_background(
        'source /opt/ros/humble/setup.bash && '
        'source $HOME/turtlebot3_ws/install/setup.bash && '
        'export TURTLEBOT3_MODEL=burger && '
        'export LDS_MODEL=LDS-02 && '
        'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && '
        'nohup ros2 launch turtlebot3_bringup robot.launch.py '
        '> /tmp/tbot_bringup.log 2>&1 &'
    )

    for i in range(20):
        time.sleep(1)
        if is_bringup_running():
            print(f'  [helpers] bringup up after {i+1}s')
            return True

    raise RuntimeError('Timed out waiting for robot bringup (20 s).')


def ensure_cartographer() -> bool:
    """Start Cartographer without DWA, so the robot stays still; True if this call started it."""
    if is_cartographer_running():
        print('  [helpers] cartographer already running — skipping start')
        return False

    print('  [helpers] starting cartographer stack (no DWA)...')
    proc = subprocess.Popen(
        _launch_cmd(
            'ros2 launch dwa_avoidance cartographer_only.launch.py'
        ),
        shell=True, executable='/bin/bash',
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=ros2_env(),
    )
    _started.append(proc)

    for i in range(25):
        time.sleep(1)
        if is_cartographer_running():
            print(f'  [helpers] cartographer up after {i+1}s')
            return True

    raise RuntimeError('Timed out waiting for cartographer_node (25 s).')


def ensure_full_stack() -> bool:
    """Start the DWA + Cartographer stack if needed; True if this call started it."""
    if is_cartographer_running() and is_dwa_running():
        print('  [helpers] full stack already running — skipping start')
        return False

    print('  [helpers] starting full DWA + Cartographer stack...')
    proc = subprocess.Popen(
        _launch_cmd(
            'ros2 launch dwa_avoidance dwa_with_cartographer.launch.py use_rviz:=false'
        ),
        shell=True, executable='/bin/bash',
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=ros2_env(),
    )
    _started.append(proc)

    for i in range(30):
        time.sleep(1)
        if is_cartographer_running() and is_dwa_running():
            print(f'  [helpers] full stack up after {i+1}s')
            return True

    raise RuntimeError('Timed out waiting for full stack (30 s).')


def kill_started() -> None:
    """Kill only the processes this session started."""
    for proc in _started:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    _started.clear()


def kill_all_ros2() -> None:
    """Kill every ROS2 process on both machines."""
    subprocess.run(
        'pkill -9 -f "ros2|cartographer|dwa_node|rviz2|scan_relay" 2>/dev/null || true',
        shell=True,
    )
    _ssh('pkill -9 -f ros2; pkill -9 -f turtlebot3; pkill -9 -f ld08')
    _started.clear()
