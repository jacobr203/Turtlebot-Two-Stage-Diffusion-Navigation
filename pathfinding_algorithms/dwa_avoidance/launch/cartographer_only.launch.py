"""
Launch: Cartographer SLAM only (no DWA, no RViz)

Used by unit tests that need to verify map building without the
avoidance node driving the robot.

Starts:
  1. scan_relay_node         — wall-clock timestamps for Cartographer
  2. tf_relay_node           — fixes MCU-uptime TF stamps (mirrors live stack)
  3. cartographer_node       — SLAM (subscribed to /scan_relayed)
  4. cartographer_occupancy_grid_node — publishes /map
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    # no_odom is the REAL-robot default (OpenCR board-uptime odom stamps,
    # see the lua's header). Sim odometry is Gazebo-stamped and consistent,
    # and scan-match-only SLAM diverged twice on fast seek rotations
    # (2026-07-23, world 613) — run.sh passes with_odom for sim runs.
    config_basename = LaunchConfiguration(
        'config_basename', default='turtlebot3_no_odom.lua')

    cartographer_config_dir = os.path.join(
        get_package_share_directory('dwa_avoidance'), 'config'
    )

    scan_relay = Node(
        package='dwa_avoidance',
        executable='scan_relay_node',
        name='scan_relay_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    tf_relay = Node(
        package='dwa_avoidance',
        executable='tf_relay_node',
        name='tf_relay_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    cartographer = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', cartographer_config_dir,
            '-configuration_basename', config_basename,
        ],
        remappings=[('scan', 'scan_relayed')],
    )

    occupancy_grid = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use Gazebo simulation clock',
        ),
        DeclareLaunchArgument(
            'config_basename',
            default_value='turtlebot3_no_odom.lua',
            description='Cartographer lua (with_odom for sim runs)',
        ),
        scan_relay,
        tf_relay,
        cartographer,
        occupancy_grid,
    ])
