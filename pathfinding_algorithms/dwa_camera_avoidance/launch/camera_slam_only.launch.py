"""Launch: LiDAR + camera FUSED Cartographer SLAM (no avoidance node, no RViz).

The camera-into-SLAM counterpart of dwa_avoidance/cartographer_only.launch.py:
  1. scan_relay_node      — step-immune wall-clock stamps for the LiDAR
  2. tf_relay_node        — fixes MCU-uptime TF stamps
  3. camera_obstacle_node — BRIO floor-contact detections → /scan_camera
  4. cartographer_node    — turtlebot3_camera_slam_fused.lua (2 laser scans)
  5. cartographer_occupancy_grid_node → /map

The camera (and its /camera_info with real intrinsics) must already be
publishing — run.sh handles that over SSH in real mode.

Usage:
  ros2 launch dwa_camera_avoidance camera_slam_only.launch.py
  ros2 launch dwa_camera_avoidance camera_slam_only.launch.py use_sim_time:=true \
      image_topic:=/camera/image_raw/compressed camera_info_topic:=/camera/camera_info
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    image_topic = LaunchConfiguration('image_topic',
                                      default='/image_raw/compressed')
    camera_info_topic = LaunchConfiguration('camera_info_topic',
                                            default='/camera_info')

    this_pkg = get_package_share_directory('dwa_camera_avoidance')
    config_dir = os.path.join(this_pkg, 'config')
    camera_obstacle_params = os.path.join(config_dir,
                                          'camera_obstacle_params.yaml')

    scan_relay = Node(
        package='dwa_avoidance', executable='scan_relay_node',
        name='scan_relay_node', output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )
    tf_relay = Node(
        package='dwa_avoidance', executable='tf_relay_node',
        name='tf_relay_node', output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )
    camera_obstacle = Node(
        package='dwa_camera_avoidance', executable='camera_obstacle_node',
        name='camera_obstacle_node', output='screen',
        parameters=[
            camera_obstacle_params,
            {'use_sim_time': use_sim_time,
             'image_topic': image_topic,
             'camera_info_topic': camera_info_topic,
             'enable_overhead_detection': False},
        ],
    )
    cartographer = Node(
        package='cartographer_ros', executable='cartographer_node',
        name='cartographer_node', output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', config_dir,
            '-configuration_basename', 'turtlebot3_camera_slam_fused.lua',
        ],
        remappings=[
            ('scan_1', 'scan_relayed'),
            ('scan_2', 'scan_camera'),
        ],
    )
    occupancy_grid = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node', output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('image_topic',
                              default_value='/image_raw/compressed'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera_info'),
        scan_relay, tf_relay, camera_obstacle, cartographer, occupancy_grid,
    ])
