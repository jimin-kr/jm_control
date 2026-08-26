#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenArm 7-DoF ROS 2 & RViz Teleoperation Bridge Node.
Receives real-time joint angle packets over UDP (Port 9870) from arm_tracking.py,
applies Geometric Fabric / Soft Barrier Safety Filtering, and publishes sensor_msgs/msg/JointState to ROS 2.

Usage:
    python3 open_arm_ros2_bridge.py --port 9870
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
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class GeometricFabricSafetyFilter:
    """
    Geometric Fabric / Soft Barrier Reactive Controller for 7-DoF Joint & Self-Collision Protection.
    Prevents joint limit saturation, singularity spikes, excessive acceleration,
    and Self-Collision (arm colliding into OpenArm body/torso).
    """
    def __init__(self, num_joints=7, limits_deg=None, arm_side="right"):
        self.num_joints = num_joints
        self.arm_side = arm_side
        # Default OpenArm 7-DoF limits
        self.limits_deg = limits_deg or [
            (-160.0, 160.0), # J1: Shoulder Yaw
            (-110.0, 110.0), # J2: Shoulder Pitch
            (-170.0, 170.0), # J3: Shoulder Roll
            (0.0, 150.0),    # J4: Elbow Pitch
            (-170.0, 170.0), # J5: Wrist Roll
            (-90.0, 90.0),   # J6: Wrist Pitch
            (-90.0, 90.0),   # J7: Wrist Yaw
        ]
        self.q_filtered_deg = np.zeros(num_joints, dtype=np.float64)
        self.dq_filtered_deg = np.zeros(num_joints, dtype=np.float64)
        self.alpha = 0.35  # Smoothing factor
        self.max_deg_per_sec = 250.0  # Max velocity limit
        self.last_time = time.time()

    def filter(self, target_q_deg: np.ndarray) -> np.ndarray:
        now = time.time()
        dt = max(1e-3, min(0.1, now - self.last_time))
        self.last_time = now

        target = np.array(target_q_deg, dtype=np.float64, copy=True)

        # -------------------------------------------------------------
        # 1. Geometric Fabric Self-Collision Avoidance Potential Barrier
        # -------------------------------------------------------------
        # Prevents arm from colliding into the OpenArm body/chest cylinder
        j1, j2, j3, j4 = target[0], target[1], target[2], target[3]

        if self.arm_side == "left":
            # Left arm body collision zones:
            # - Excessively swinging inward (J1 approaching inward angle)
            # - Folding elbow (J4 > 90°) while shoulder pitch/roll pulls inward
            if j1 > 75.0: # Heading towards right/center body
                target[0] = 75.0 - 5.0 * math.tanh((j1 - 75.0) / 5.0)

            # Prevent elbow flexion (J4) from smashing wrist into chest
            if j4 > 95.0 and j2 < -15.0:
                # Push J2 (shoulder pitch) slightly forward/outward to prevent chest crash
                repulsion = (j4 - 95.0) * 0.35
                target[1] += repulsion
                target[3] = 95.0 + 35.0 * math.tanh((j4 - 95.0) / 35.0)

        elif self.arm_side == "right":
            # Right arm body collision zones
            if j1 < -75.0: # Heading towards left/center body
                target[0] = -75.0 + 5.0 * math.tanh((-75.0 - j1) / 5.0)

            if j4 > 95.0 and j2 < -15.0:
                repulsion = (j4 - 95.0) * 0.35
                target[1] += repulsion
                target[3] = 95.0 + 35.0 * math.tanh((j4 - 95.0) / 35.0)

        safe_q = np.zeros(self.num_joints, dtype=np.float64)

        for i in range(self.num_joints):
            min_lim, max_lim = self.limits_deg[i]
            val = target[i]

            # 2. Soft Barrier Repulsive Potential (Joint Limit Barrier)
            margin = 8.0
            if val > max_lim - margin:
                excess = val - (max_lim - margin)
                val = (max_lim - margin) + margin * (1.0 - math.exp(-excess / margin))
            elif val < min_lim + margin:
                excess = (min_lim + margin) - val
                val = (min_lim + margin) - margin * (1.0 - math.exp(-excess / margin))

            val = float(np.clip(val, min_lim, max_lim))

            # 3. Velocity Limit Filter (Rate Limiter)
            max_delta = self.max_deg_per_sec * dt
            delta = val - self.q_filtered_deg[i]
            delta = float(np.clip(delta, -max_delta, max_delta))
            val = self.q_filtered_deg[i] + delta

            # 4. EMA Low-pass Smoothing
            self.q_filtered_deg[i] = self.alpha * val + (1.0 - self.alpha) * self.q_filtered_deg[i]
            safe_q[i] = self.q_filtered_deg[i]

        return safe_q


def main():
    parser = argparse.ArgumentParser(description="OpenArm ROS 2 & RViz Teleoperation Bridge")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="UDP Listening IP")
    parser.add_argument("--port", type=int, default=9870, help="UDP Listening Port")
    parser.add_argument("--topic", type=str, default="/joint_states", help="ROS 2 JointState topic name")
    parser.add_argument("--arm", type=str, default="right", choices=["right", "left"], help="Arm side (right or left)")
    args = parser.parse_args()

    if not ROS2_AVAILABLE:
        print("[ERROR] rclpy / sensor_msgs is not installed in current Python environment.")
        sys.exit(1)

    # Initialize ROS 2
    rclpy.init()
    node = Node("open_arm_rviz_bridge")
    publisher = node.create_publisher(JointState, args.topic, 10)

    # Open UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))
    sock.setblocking(False)

    safety_filter = GeometricFabricSafetyFilter(arm_side=args.arm)

    # Official OpenArm URDF Full Joint Names (Left Arm + Right Arm + Fingers)
    all_joint_names = [
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7",
        "openarm_left_finger_joint1", "openarm_left_finger_joint2",
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7",
        "openarm_right_finger_joint1", "openarm_right_finger_joint2"
    ]

    print("=================================================================")
    print("      OpenArm 7-DoF ROS 2 / RViz Teleoperation Bridge           ")
    print(f"      UDP Listening : {args.ip}:{args.port}")
    print(f"      Target Arm    : {args.arm.upper()}")
    print(f"      ROS 2 Topic   : {args.topic}")
    print("=================================================================")
    print("Waiting for vision tracking packets from arm_tracking.py...\n")

    last_print = time.time()
    packet_count = 0

    try:
        while rclpy.ok():
            try:
                data, _ = sock.recvfrom(8192)
                packet_count += 1
                payload = json.loads(data.decode("utf-8"))

                # Extract raw 7-DoF joint angles in degrees safely
                open_arm_dict = payload.get("open_arm_7dof") if isinstance(payload, dict) else None
                if not isinstance(open_arm_dict, dict):
                    open_arm_dict = {}
                q_raw_deg = open_arm_dict.get("joint_angles_deg", [0.0]*7)
                if not isinstance(q_raw_deg, (list, tuple)) or len(q_raw_deg) < 7:
                    q_raw_deg = [0.0] * 7

                # Apply Geometric Fabric Safety & Smoothing Filter
                q_safe_deg = safety_filter.filter(q_raw_deg)

                # Map active arm joints and set default 0.0 for all other URDF joints
                joint_dict = {name: 0.0 for name in all_joint_names}
                active_prefix = f"openarm_{args.arm}_"
                for i in range(1, 8):
                    joint_dict[f"{active_prefix}joint{i}"] = math.radians(q_safe_deg[i-1])

                # Publish ROS 2 JointState message with full TF coverage
                msg = JointState()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.name = list(joint_dict.keys())
                msg.position = list(joint_dict.values())
                publisher.publish(msg)

                now = time.time()
                if now - last_print >= 0.2: # 5Hz Console Feedback
                    last_print = now
                    formatted_q = ", ".join([f"{q:+6.1f}°" for q in q_safe_deg])
                    print(f"\r[ROS 2 Bridge] Packets: {packet_count:6d} | Joint Angles: [{formatted_q}]", end="", flush=True)

            except socket.error:
                time.sleep(0.005) # Sleep briefly when no packet in queue

    except KeyboardInterrupt:
        print("\n[System] ROS 2 Bridge Node stopped.")
    finally:
        sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
