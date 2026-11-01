"""Camera-fused DWA + LiDAR-only Cartographer SLAM + RViz.

Robot bringup and the BRIO must already be running (run.sh starts both).

Usage: ros2 launch dwa_camera_avoidance dwa_camera_with_cartographer.launch.py [use_sim_time:=true]
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
            ('scan_2', 'scan_camera'),    # used when num_laser_scans = 2
        ],
    )

    occupancy_grid = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
    )

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
                        'phantoms in sim AND on the real floor; '
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
