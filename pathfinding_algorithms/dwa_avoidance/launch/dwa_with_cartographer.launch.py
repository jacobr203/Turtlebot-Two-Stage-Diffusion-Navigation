"""DWA obstacle avoidance + Cartographer SLAM + RViz map display.

Odometry stays out of Cartographer: feeding it needs a wall-clock anchor shared
by both relays, and independent anchors crash the pose extrapolator.

Usage: ros2 launch dwa_avoidance dwa_with_cartographer.launch.py [use_sim_time:=true]
(on the real robot, bringup must already be running)
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

    scan_relay = Node(
        package='dwa_avoidance',
        executable='scan_relay_node',
        name='scan_relay_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Robot TF uses MCU uptime and Cartographer wall-clock time; without the
    # relay RViz cannot show the robot's position.
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
            '-configuration_basename', 'turtlebot3_no_odom.lua',
        ],
        remappings=[
            ('scan', 'scan_relayed'),
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
