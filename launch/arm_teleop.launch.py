#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated ROS 2 Launch File for Single OpenArm 7-DoF RealSense Teleoperation.
Launches:
  1. RealSense Camera & MediaPipe IK Bridge Node (openarm_realsense_teleop_node.py)
  2. Robot State Publisher with custom OpenArm URDF
  3. MoveIt2 Servo / OpenArm Controller Node
  4. RViz2 3D Motion & Joint State Visualization
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # --------------------------------------------------------------------------
    # 1. Launch Arguments & Paths
    # --------------------------------------------------------------------------
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    urdf_package = LaunchConfiguration('urdf_package', default='openarm_description')
    config_file = LaunchConfiguration('config_file', default=os.path.join(
        os.path.dirname(__file__), '..', 'config', 'openarm_teleop_config.yaml'
    ))

    # Declare Arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz2 visualization'
    )
    declare_urdf_package = DeclareLaunchArgument(
        'urdf_package', default_value='openarm_description',
        description='ROS 2 package containing the custom OpenArm URDF'
    )

    # --------------------------------------------------------------------------
    # 2. Robot Description (URDF Load)
    # --------------------------------------------------------------------------
    # Locate URDF (searches share directory or falls back to local workspace)
    try:
        pkg_share = FindPackageShare('openarm_description').find('openarm_description')
        urdf_path = os.path.join(pkg_share, 'urdf', 'openarm.urdf')
    except Exception:
        # Fallback to local workspace relative path
        urdf_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'robot_control-jazzy', 'ros_ws', 'src', 'openarm_description', 'urdf', 'openarm.urdf'
        ))

    if os.path.exists(urdf_path):
        with open(urdf_path, 'r') as f:
            robot_desc = f.read()
    else:
        robot_desc = '<?xml version="1.0"?><robot name="openarm_7dof"><link name="base_link"/></robot>'

    # --------------------------------------------------------------------------
    # 3. ROS 2 Nodes
    # --------------------------------------------------------------------------
    # Node 1: Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': use_sim_time
        }]
    )

    # Node 2: RealSense & MediaPipe IK Teleop Bridge Node
    teleop_bridge_node = Node(
        package='openarm_teleop',
        executable='openarm_realsense_teleop_node',
        name='openarm_realsense_teleop_node',
        output='screen',
        parameters=[config_file, {'use_sim_time': use_sim_time}]
    )

    # Node 3: OpenArm Controller / MoveIt2 Servo Container Bridge
    moveit_servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        name='servo_node',
        output='screen',
        parameters=[config_file, {'robot_description': robot_desc}],
        condition=IfCondition('false') # Enable when MoveIt Servo C++ binary is built
    )

    # Node 4: RViz2 Visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(os.path.dirname(__file__), '..', 'config', 'openarm_teleop.rviz')],
        condition=IfCondition(use_rviz)
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_use_rviz,
        declare_urdf_package,
        LogInfo(msg=['[OpenArm Teleop] Launching RealSense Vision Teleoperation System...']),
        robot_state_publisher_node,
        teleop_bridge_node,
        rviz_node,
    ])
