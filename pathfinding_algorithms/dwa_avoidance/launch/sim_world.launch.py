"""
Launch: Gazebo Classic world + TurtleBot3 for simulation runs.

Replaces the real robot's bringup: gzserver (+ gzclient unless headless),
robot_state_publisher, and a spawned TurtleBot3 model. The algorithm launch
files (dwa_with_cartographer, dwa_camera_with_cartographer, nav2) run on top
with use_sim_time:=true, exactly as they would against the real robot.

Args:
  world      turtlebot3_world (default) | turtlebot3_house | empty_world,
             or an absolute path to a .world file (e.g. a BARN world under
             capstone/barn/worlds/)
  headless   'true' skips gzclient (default 'false')
  x_pose / y_pose / yaw_pose   spawn pose (defaults -2.0 / -0.5 / 0.0)
  model_sdf  absolute path to the robot SDF (default: apt burger model)
  urdf_file  absolute path to the URDF for robot_state_publisher

Usage:
  ros2 launch dwa_avoidance sim_world.launch.py
  ros2 launch dwa_avoidance sim_world.launch.py world:=turtlebot3_house headless:=true
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
    # Absolute paths (e.g. BARN worlds under capstone/barn/worlds/) are used
    # as-is; bare names resolve in the apt turtlebot3_gazebo share as before.
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
        # gzclient initializes its scene slowly (seconds under WSLg); a model
        # inserted DURING that window is intermittently never rendered — the
        # robot exists in gzserver (physics, topics) but is invisible in the
        # GUI. Insertions into a ready client render reliably, so hold the
        # spawn back when a GUI was requested.
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
