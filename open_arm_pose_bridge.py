#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenArm 3D End-Effector Pose ROS 2 Teleoperation Bridge Node (Option B).
Receives real-time 3D wrist/end-effector Cartesian pose from arm_tracking.py via UDP (Port 9870)
and publishes geometry_msgs/msg/PoseStamped to ROS 2 (MoveIt / Servo / IK Stack).

Usage:
    python3 open_arm_pose_bridge.py --arm left --port 9870 --topic /target_pose
"""

import sys
import os
import json
import socket
import argparse
import time
import math
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class OpenArmPoseBridgeNode(Node):
    def __init__(self, target_arm="left", topic_name="/target_pose", js_topic="/joint_states"):
        super().__init__("open_arm_pose_bridge")
        self.target_arm = target_arm
        self.publisher = self.create_publisher(PoseStamped, topic_name, 10)
        self.js_publisher = self.create_publisher(JointState, js_topic, 10)
        self.get_logger().info(f"OpenArm Pose Bridge B active -> Arm: {target_arm.upper()} | Pose: {topic_name} | JointStates: {js_topic}")

    def publish_pose(self, pos_3d: list, quat_wxyz: list):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        # Position (meters)
        msg.pose.position.x = float(pos_3d[0])
        msg.pose.position.y = float(pos_3d[1])
        msg.pose.position.z = float(pos_3d[2])

        # Quaternion (w, x, y, z) -> ROS geometry_msgs (x, y, z, w)
        if len(quat_wxyz) >= 4:
            msg.pose.orientation.w = float(quat_wxyz[0])
            msg.pose.orientation.x = float(quat_wxyz[1])
            msg.pose.orientation.y = float(quat_wxyz[2])
            msg.pose.orientation.z = float(quat_wxyz[3])
        else:
            msg.pose.orientation.w = 1.0

        self.publisher.publish(msg)

    def publish_joint_states(self, joint_names: list, q_safe_deg: list):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        joint_dict = {name: 0.0 for name in joint_names}
        prefix = f"openarm_{self.target_arm}_"
        for i in range(1, 8):
            joint_dict[f"{prefix}joint{i}"] = math.radians(q_safe_deg[i-1])

        msg.name = list(joint_dict.keys())
        msg.position = list(joint_dict.values())
        self.js_publisher.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="OpenArm 3D End-Effector Pose ROS 2 Teleop Bridge (Option B)")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="UDP Listening IP")
    parser.add_argument("--port", type=int, default=9870, help="UDP Listening Port")
    parser.add_argument("--topic", type=str, default="/target_pose", help="ROS 2 PoseStamped topic name")
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right"], help="Arm side (left or right)")
    args = parser.parse_args()

    if not ROS2_AVAILABLE:
        print("[ERROR] rclpy / geometry_msgs is not installed in current Python environment.")
        sys.exit(1)

    try:
        from open_arm_ros2_bridge import GeometricFabricSafetyFilter
    except ImportError:
        class GeometricFabricSafetyFilter:
            def __init__(self, arm_side="right"): pass
            def filter(self, q): return q

    # Initialize ROS 2
    rclpy.init()
    node = OpenArmPoseBridgeNode(target_arm=args.arm, topic_name=args.topic)

    safety_filter = GeometricFabricSafetyFilter(arm_side=args.arm)
    all_joint_names = [
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7",
        "openarm_left_finger_joint1", "openarm_left_finger_joint2",
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7",
        "openarm_right_finger_joint1", "openarm_right_finger_joint2"
    ]

    # Open UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))
    sock.setblocking(False)

    print("=================================================================")
    print("      OpenArm 3D End-Effector Pose ROS 2 Teleop Bridge (Option B) ")
    print(f"      UDP Listening : {args.ip}:{args.port}")
    print(f"      Target Arm    : {args.arm.upper()}")
    print(f"      ROS 2 Topic   : {args.topic} (geometry_msgs/msg/PoseStamped)")
    print("=================================================================")
    print("Waiting for vision tracking packets from arm_tracking.py...\n")

    last_print = time.time()
    packet_count = 0

    try:
        while rclpy.ok():
            try:
                # Socket Drain: Keep ONLY freshest frame
                latest_data = None
                while True:
                    try:
                        chunk, _ = sock.recvfrom(8192)
                        latest_data = chunk
                    except socket.error:
                        break

                if latest_data is None:
                    time.sleep(0.001)
                    continue

                packet_count += 1
                payload = json.loads(latest_data.decode("utf-8"))

                # Extract 3D wrist pose & 3D position correctly from telemetry dictionary
                wrist_6dof = payload.get("wrist_6dof") if isinstance(payload, dict) else None
                joints_3d = payload.get("joints_3d_meters") if isinstance(payload, dict) else None

                pos_3d = [0.0, 0.0, 0.0]
                quat_wxyz = [1.0, 0.0, 0.0, 0.0]

                if isinstance(joints_3d, dict):
                    pos_3d = joints_3d.get("wrist", [0.0, 0.0, 0.0])

                if isinstance(wrist_6dof, dict):
                    quat_wxyz = wrist_6dof.get("orientation_quat", [1.0, 0.0, 0.0, 0.0])
                    if np.all(np.abs(pos_3d) < 1e-4):
                        pos_3d = wrist_6dof.get("position", [0.0, 0.0, 0.0])

                # Extract raw 7-DoF joint angles for instant RViz 3D joint states driving
                open_arm_dict = payload.get("open_arm_7dof") if isinstance(payload, dict) else {}
                if not isinstance(open_arm_dict, dict):
                    open_arm_dict = {}
                q_raw_deg = open_arm_dict.get("joint_angles_deg", [0.0]*7)
                if not isinstance(q_raw_deg, (list, tuple)) or len(q_raw_deg) < 7:
                    q_raw_deg = [0.0] * 7

                q_safe_deg = safety_filter.filter(np.array(q_raw_deg[:7], dtype=np.float64))

                # Publish 3D Pose (Option B) AND JointStates (Instant RViz 3D Drive)
                node.publish_pose(pos_3d, quat_wxyz)
                node.publish_joint_states(all_joint_names, q_safe_deg)

                now = time.time()
                if now - last_print >= 0.2: # 5Hz Console Feedback
                    last_print = now
                    print(f"\r[Pose Bridge B] Packets: {packet_count:6d} | 3D Wrist (X,Y,Z): [{pos_3d[0]:+5.2f}m, {pos_3d[1]:+5.2f}m, {pos_3d[2]:+5.2f}m]", end="", flush=True)

            except socket.error:
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[System] OpenArm Pose Bridge Node stopped.")
    finally:
        sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
