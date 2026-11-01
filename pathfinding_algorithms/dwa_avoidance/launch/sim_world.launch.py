"""Gazebo Classic world + TurtleBot3, standing in for the real robot's bringup.

Usage: ros2 launch dwa_avoidance sim_world.launch.py [world:=turtlebot3_house] [headless:=true]
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

TB3_GAZEBO_SHARE = get_package_share_directory('turtlebot3_gazebo')
GAZEBO_ROS_SHARE = get_package_share_directory('gazebo_ros')

DEFAULT_MODEL_SDF = os.path.join(TB3_GAZEBO_SHARE, 'models',
                                 'turtlebot3_burger', 'model.sdf')
DEFAULT_URDF = os.path.join(TB3_GAZEBO_SHARE, 'urdf', 'turtlebot3_burger.urdf')


def _setup(context, *args, **kwargs):
    world_name = LaunchConfiguration('world').perform(context)
    model_sdf  = LaunchConfiguration('model_sdf').perform(context)
    urdf_file  = LaunchConfiguration('urdf_file').perform(context)
    x_pose     = LaunchConfiguration('x_pose').perform(context)
    y_pose     = LaunchConfiguration('y_pose').perform(context)
    yaw_pose   = LaunchConfiguration('yaw_pose').perform(context)

    if not world_name.endswith('.world'):
        world_name += '.world'
    # bare names resolve in the apt turtlebot3_gazebo share
    world_path = world_name if os.path.isabs(world_name) \
        else os.path.join(TB3_GAZEBO_SHARE, 'worlds', world_name)

    with open(urdf_file) as f:
        robot_description = f.read()

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(GAZEBO_ROS_SHARE, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world_path}.items(),
    )

    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(GAZEBO_ROS_SHARE, 'launch', 'gzclient.launch.py')),
        condition=UnlessCondition(LaunchConfiguration('headless')),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_description,
        }],
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'turtlebot3',
                   '-file', model_sdf,
                   '-x', x_pose, '-y', y_pose, '-z', '0.01',
                   '-Y', yaw_pose],
        output='screen',
    )

    headless = LaunchConfiguration('headless').perform(context).lower() == 'true'
    if not headless:
        # a model inserted while gzclient is still initializing its scene is
        # intermittently never rendered, so hold the spawn back
        spawn = TimerAction(period=10.0, actions=[spawn])

    return [gzserver, gzclient, robot_state_publisher, spawn]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='turtlebot3_world',
                              description='World name in turtlebot3_gazebo/worlds, '
                                          'or an absolute .world path'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='true skips gzclient (no GUI)'),
        DeclareLaunchArgument('x_pose', default_value='-2.0'),
        DeclareLaunchArgument('y_pose', default_value='-0.5'),
        DeclareLaunchArgument('yaw_pose', default_value='0.0'),
        DeclareLaunchArgument('model_sdf', default_value=DEFAULT_MODEL_SDF,
                              description='Absolute path to robot model.sdf'),
        DeclareLaunchArgument('urdf_file', default_value=DEFAULT_URDF,
                              description='Absolute path to robot URDF'),
        OpaqueFunction(function=_setup),
    ])
