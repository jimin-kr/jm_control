#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone OpenArm RViz Launch Script (No colcon build required).
Loads OpenArm URDF file and starts robot_state_publisher + rviz2 directly.

Usage:
    python3 launch_rviz.py --urdf ../openarm_description/assets/robot/openarm_v2.0/urdf/example/v2.urdf
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Standalone OpenArm RViz Launcher")
    parser.add_argument(
        "--urdf",
        type=str,
        default="../openarm_description/assets/robot/openarm_v2.0/urdf/example/v2.urdf",
        help="Path to openarm URDF file"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.abspath(os.path.join(script_dir, args.urdf))

    if not os.path.exists(urdf_path):
        print(f"[ERROR] URDF file not found at: {urdf_path}")
        sys.exit(1)

    print("=================================================================")
    print("      OpenArm Standalone RViz Launcher (No Build Required)       ")
    print(f"      URDF File : {urdf_path}")
    print("=================================================================")

    # Create temporary launch python file
    temp_launch = os.path.join(script_dir, ".temp_rviz_launch.py")

    launch_content = f"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    urdf_file = "{urdf_path}"
    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{{'robot_description': robot_desc}}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '{os.path.join(script_dir, "../openarm_description/rviz/display_openarm.rviz")}'] if os.path.exists('{os.path.join(script_dir, "../openarm_description/rviz/display_openarm.rviz")}') else []
        )
    ])
"""
    with open(temp_launch, "w") as f:
        f.write(launch_content)

    install_setup = os.path.abspath(os.path.join(script_dir, "../openarm_description/install/setup.bash"))
    try:
        if os.path.exists(install_setup):
            cmd = f"bash -c 'source {install_setup} && ros2 launch {temp_launch}'"
        else:
            cmd = f"ros2 launch {temp_launch}"
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        print("\n[System] RViz launcher stopped.")
    finally:
        if os.path.exists(temp_launch):
            os.remove(temp_launch)

if __name__ == "__main__":
    main()
