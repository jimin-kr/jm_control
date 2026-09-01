#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenArm RViz / MoveIt Goal Joint Angle Extractor.
Extracts current joint angles (in both Radians and Degrees) from RViz / MoveIt / JointStates.

Usage:
    python3 extract_rviz_pose.py --arm left
"""

import sys
import argparse
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from moveit_msgs.msg import DisplayTrajectory
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class PoseExtractorNode(Node):
    def __init__(self, target_arm="left"):
        super().__init__("open_arm_pose_extractor")
        self.target_arm = target_arm
        self.prefix = f"openarm_{target_arm}_joint"
        self.latest_joint_state = {}

        # Subscriptions
        self.sub_js = self.create_subscription(JointState, "/joint_states", self.js_callback, 10)
        self.sub_traj = self.create_subscription(DisplayTrajectory, "/display_planned_path", self.traj_callback, 10)

    def js_callback(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.latest_joint_state[name] = pos

    def traj_callback(self, msg: DisplayTrajectory):
        if not msg.trajectory or not msg.trajectory[0].joint_trajectory.points:
            return
        
        last_point = msg.trajectory[0].joint_trajectory.points[-1]
        joint_names = msg.trajectory[0].joint_trajectory.joint_names
        
        print(f"\n=================================================================")
        print(f"  🎯 [MoveIt Plan Captured] Arm: {self.target_arm.upper()}")
        print(f"=================================================================")
        
        arm_joints = []
        for i in range(1, 8):
            jname = f"{self.prefix}{i}"
            if jname in joint_names:
                idx = joint_names.index(jname)
                arm_joints.append((jname, last_point.positions[idx]))

        if arm_joints:
            rad_vals = [val for _, val in arm_joints]
            deg_vals = [np.degrees(val) for val in rad_vals]

            print(" [Radian Array (for Code / Ready Pose)]")
            print(f" ready_pose_rad = ({', '.join([f'{r:.4f}' for r in rad_vals])})")
            print("\n [Degree Array]")
            print(f" ready_pose_deg = ({', '.join([f'{d:+6.1f}°' for d in deg_vals])})")
            print("=================================================================\n")

    def print_current_js(self):
        print(f"\n=================================================================")
        print(f"  📍 [Current /joint_states] Arm: {self.target_arm.upper()}")
        print(f"=================================================================")
        
        arm_joints = []
        for i in range(1, 8):
            jname = f"{self.prefix}{i}"
            val = self.latest_joint_state.get(jname, 0.0)
            arm_joints.append((jname, val))

        rad_vals = [val for _, val in arm_joints]
        deg_vals = [np.degrees(val) for val in rad_vals]

        print(" [Radian Array]")
        print(f" ({', '.join([f'{r:.4f}' for r in rad_vals])})")
        print("\n [Degree Array]")
        print(f" ({', '.join([f'{d:+6.1f}°' for d in deg_vals])})")
        print("=================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Extract Joint Angles from RViz / MoveIt Pose")
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right"], help="Target arm side")
    args = parser.parse_args()

    if not ROS2_AVAILABLE:
        print("[ERROR] rclpy is not available in current environment.")
        sys.exit(1)

    rclpy.init()
    node = PoseExtractorNode(target_arm=args.arm)

    print("=================================================================")
    print("      OpenArm RViz / MoveIt Joint Angle Extractor               ")
    print(f"      Target Arm : {args.arm.upper()}")
    print("=================================================================")
    print(" 💡 Tip 1: In RViz, drag the Interactive Marker and click 'Plan'")
    print("           -> The planned goal joint angles will print here automatically!\n")
    print(" 💡 Tip 2: Press ENTER in this terminal anytime")
    print("           -> Prints current RViz /joint_states angles immediately!\n")

    import threading
    def spin_thread():
        try:
            rclpy.spin(node)
        except Exception:
            pass

    t = threading.Thread(target=spin_thread, daemon=True)
    t.start()

    try:
        while True:
            input("\n👉 [Press ENTER to capture current RViz pose]...")
            node.print_current_js()
    except (KeyboardInterrupt, EOFError):
        print("\n[System] Extractor Node stopped.")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
