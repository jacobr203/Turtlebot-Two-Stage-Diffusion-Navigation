"""
Launch: Gazebo world with the turtlebot3_burger_brio model (Burger + pinhole
camera at the real BRIO mount). Thin wrapper over dwa_avoidance's
sim_world.launch.py — forwards world/headless/pose args.

Camera topics in sim: /camera/image_raw[/compressed], /camera/camera_info.

Usage:
  ros2 launch dwa_camera_avoidance sim_world_brio.launch.py
  ros2 launch dwa_camera_avoidance sim_world_brio.launch.py world:=turtlebot3_house headless:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

THIS_SHARE = get_package_share_directory('dwa_camera_avoidance')
DWA_SHARE  = get_package_share_directory('dwa_avoidance')


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='turtlebot3_world'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('x_pose', default_value='-2.0'),
        DeclareLaunchArgument('y_pose', default_value='-0.5'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(DWA_SHARE, 'launch', 'sim_world.launch.py')),
            launch_arguments={
                'world':     LaunchConfiguration('world'),
                'headless':  LaunchConfiguration('headless'),
                'x_pose':    LaunchConfiguration('x_pose'),
                'y_pose':    LaunchConfiguration('y_pose'),
                'model_sdf': os.path.join(THIS_SHARE, 'models',
                                          'turtlebot3_burger_brio', 'model.sdf'),
                'urdf_file': os.path.join(THIS_SHARE, 'urdf',
                                          'turtlebot3_burger_brio.urdf'),
            }.items(),
        ),
    ])
