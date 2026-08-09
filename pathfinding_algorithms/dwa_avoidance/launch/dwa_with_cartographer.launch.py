"""
Launch: DWA obstacle avoidance + Cartographer SLAM + RViz map display

Node start order:
  1. scan_relay_node  — re-stamps /scan with anchored wall-clock time (bug 002 fix).
  2. tf_relay_node    — fixes MCU-uptime TF stamps to wall-clock time so RViz
                        can display the robot position correctly.
  3. cartographer_node — scan-matching only (no odometry); subscribes to /scan_relayed.
                         Publishes map→odom TF continuously as map builds.
  4. cartographer_occupancy_grid_node — publishes /map OccupancyGrid.
  5. rviz2            — displays the map as it is built (uses /scan_relayed so
                        laser overlay works despite LDS-02 BEST_EFFORT QoS).
  6. dwa_node         — reads /scan (BEST_EFFORT) and /odom, publishes /cmd_vel.
                        Auto-picks a new goal pose every 45 s from open scan space.

Odometry is intentionally excluded from Cartographer.  Feeding odom requires a
shared wall-clock anchor across both relay nodes; independent anchors produce a
~3.6 ms offset that causes pose_extrapolator.cc:229 to fire (bug 008).
Cartographer's scan-to-scan matching tracks the robot accurately without odom,
and the DWA node reads /odom directly for its own velocity model.

Usage (real robot — bringup must already be running on the robot):
  ros2 launch dwa_avoidance dwa_with_cartographer.launch.py

Usage (Gazebo simulation):
  ros2 launch dwa_avoidance dwa_with_cartographer.launch.py use_sim_time:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    use_rviz     = LaunchConfiguration('use_rviz',     default='true')

    cartographer_config_dir = os.path.join(
        get_package_share_directory('dwa_avoidance'), 'config'
    )
    rviz_config = os.path.join(
        get_package_share_directory('dwa_avoidance'),
        'rviz', 'dwa_cartographer.rviz'
    )
    dwa_params = os.path.join(
        get_package_share_directory('dwa_avoidance'), 'config', 'dwa_params.yaml'
    )

    # ── 1. Scan relay ─────────────────────────────────────────────────────────
    scan_relay = Node(
        package='dwa_avoidance',
        executable='scan_relay_node',
        name='scan_relay_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ── 2. TF relay ───────────────────────────────────────────────────────────
    # Robot TF (odom→base_footprint) uses MCU uptime; Cartographer's map→odom
    # uses wall-clock time.  This relay detects stale-stamped TF transforms and
    # republishes them with current wall-clock time so RViz shows robot position.
    tf_relay = Node(
        package='dwa_avoidance',
        executable='tf_relay_node',
        name='tf_relay_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ── 3. Cartographer SLAM ──────────────────────────────────────────────────
    # Scan-matching only (no odometry input) — avoids the anchor-offset crash.
    # Remapped to /scan_relayed so Cartographer sees wall-clock-stamped scans.
    cartographer = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', cartographer_config_dir,
            '-configuration_basename', 'turtlebot3_no_odom.lua',
        ],
        remappings=[
            ('scan', 'scan_relayed'),
        ],
    )

    # ── 4. Occupancy grid publisher ───────────────────────────────────────────
    occupancy_grid = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
    )

    # ── 5. RViz2 ─────────────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
    )

    # ── 6. DWA avoidance node ─────────────────────────────────────────────────
    dwa = Node(
        package='dwa_avoidance',
        executable='dwa_node',
        name='dwa_node',
        output='screen',
        parameters=[
            dwa_params,
            {'use_sim_time': use_sim_time},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use Gazebo simulation clock',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Open RViz2 to display the map as it is built',
        ),
        scan_relay,
        tf_relay,
        cartographer,
        occupancy_grid,
        rviz,
        dwa,
    ])
