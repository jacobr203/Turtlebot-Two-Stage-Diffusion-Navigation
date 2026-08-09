"""
Launch: camera-fused DWA + Cartographer SLAM + RViz

Node start order:
  1. scan_relay_node      (from dwa_avoidance) — re-stamps /scan to wall clock.
  2. tf_relay_node        (from dwa_avoidance) — fixes MCU-uptime TF stamps.
  3. camera_obstacle_node — converts /image_raw → /scan_camera virtual scan.
  4. cartographer_node    — 2-scan config: /scan_relayed (LiDAR) + /scan_camera.
  5. cartographer_occupancy_grid_node.
  6. rviz2.
  7. dwa_camera_node      — DWA with merged LiDAR + camera obstacle list.

Camera prerequisites
--------------------
The Logitech BRIO must be publishing before this launch is run.  On the robot:
  ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:=/dev/video0
Or use usb_cam:
  ros2 run usb_cam usb_cam_node_exe

The camera node must publish:
  /image_raw   (sensor_msgs/Image)
  /camera_info (sensor_msgs/CameraInfo)

Usage (real robot — bringup must already be running):
  ros2 launch dwa_camera_avoidance dwa_camera_with_cartographer.launch.py

Usage (Gazebo):
  ros2 launch dwa_camera_avoidance dwa_camera_with_cartographer.launch.py use_sim_time:=true
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
    image_topic       = LaunchConfiguration('image_topic',
                                            default='/image_raw/compressed')
    camera_info_topic = LaunchConfiguration('camera_info_topic',
                                            default='/camera_info')
    enable_overhead   = LaunchConfiguration('enable_overhead_detection',
                                            default='false')

    this_pkg = get_package_share_directory('dwa_camera_avoidance')
    dwa_pkg  = get_package_share_directory('dwa_avoidance')

    cartographer_config_dir = os.path.join(this_pkg, 'config')
    dwa_camera_params       = os.path.join(this_pkg, 'config', 'dwa_camera_params.yaml')
    camera_obstacle_params  = os.path.join(this_pkg, 'config', 'camera_obstacle_params.yaml')
    rviz_config             = os.path.join(dwa_pkg,  'rviz',   'dwa_cartographer.rviz')

    # ── 1. Scan relay (re-stamps LiDAR /scan to wall clock for Cartographer) ─
    scan_relay = Node(
        package='dwa_avoidance',
        executable='scan_relay_node',
        name='scan_relay_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ── 2. TF relay (fixes MCU-uptime TF stamps) ──────────────────────────────
    tf_relay = Node(
        package='dwa_avoidance',
        executable='tf_relay_node',
        name='tf_relay_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ── 3. Camera obstacle node ───────────────────────────────────────────────
    # Converts /image_raw → /scan_camera (virtual LaserScan).
    # Camera must already be running and publishing /image_raw + /camera_info.
    camera_obstacle = Node(
        package='dwa_camera_avoidance',
        executable='camera_obstacle_node',
        name='camera_obstacle_node',
        output='screen',
        parameters=[
            camera_obstacle_params,
            {'use_sim_time': use_sim_time,
             'image_topic': image_topic,
             'camera_info_topic': camera_info_topic,
             'enable_overhead_detection': enable_overhead},
        ],
    )

    # ── 4. Cartographer SLAM (LiDAR-only — see turtlebot3_camera_slam.lua) ───
    # scan / scan_1 → /scan_relayed  (LiDAR, step-immune wall-clock stamps)
    # scan_2        → /scan_camera   (only used if num_laser_scans = 2)
    cartographer = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', cartographer_config_dir,
            '-configuration_basename', 'turtlebot3_camera_slam.lua',
        ],
        remappings=[
            ('scan',   'scan_relayed'),   # used when num_laser_scans = 1
            ('scan_1', 'scan_relayed'),   # used when num_laser_scans >= 2
            ('scan_2', 'scan_camera'),
        ],
    )

    # ── 5. Occupancy grid ─────────────────────────────────────────────────────
    occupancy_grid = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
    )

    # ── 6. RViz2 ──────────────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
    )

    # ── 7. DWA-Camera avoidance node ─────────────────────────────────────────
    dwa_camera = Node(
        package='dwa_camera_avoidance',
        executable='dwa_camera_node',
        name='dwa_camera_node',
        output='screen',
        parameters=[
            dwa_camera_params,
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
            description='Open RViz2',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/image_raw/compressed',
            description='CompressedImage input (sim: /camera/image_raw/compressed)',
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/camera_info',
            description='CameraInfo input (sim: /camera/camera_info)',
        ),
        DeclareLaunchArgument(
            'enable_overhead_detection',
            default_value='false',
            description='Overhead obstacle channel. Default OFF: its '
                        'fixed-height back-projection measured 46-77% '
                        'phantoms in sim AND on the real floor (bug #017); '
                        'pass true to re-enable once redesigned/calibrated.',
        ),
        scan_relay,
        tf_relay,
        camera_obstacle,
        cartographer,
        occupancy_grid,
        rviz,
        dwa_camera,
    ])
