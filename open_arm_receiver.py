#!/usr/bin/env python3
"""
OpenArm 7-DoF Robot Control Receiver Demo.
Receives real-time 7-DoF Joint Angles and 6-DoF Wrist Pose via UDP socket (Port 9870).
Ready to be forwarded to OpenArm CAN Bus, ROS2 /joint_states topic, or PyBullet simulation.

Usage:
    python3 open_arm_receiver.py --port 9870
"""

import sys
import os
import json
import socket
import argparse
import time
import numpy as np


def render_joint_bar(name: str, angle_deg: float, min_deg: float, max_deg: float, width: int = 24) -> str:
    """Renders a clean ASCII gauge bar for joint angle monitoring."""
    ratio = (angle_deg - min_deg) / (max_deg - min_deg + 1e-6)
    ratio = max(0.0, min(1.0, ratio))
    pos = int(ratio * width)
    bar = ["-"] * width
    center = int((-min_deg) / (max_deg - min_deg + 1e-6) * width)
    if 0 <= center < width:
        bar[center] = "|"
    if 0 <= pos < width:
        bar[pos] = "O"
    return f"{name:2s} [{''.join(bar)}] {angle_deg:+6.1f}° (min:{min_deg:+4.0f}°, max:{max_deg:+4.0f}°)"


def main():
    parser = argparse.ArgumentParser(description="OpenArm 7-DoF UDP Joint Angle Receiver")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Listening IP address")
    parser.add_argument("--port", type=int, default=9870, help="UDP listening port")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))
    sock.setblocking(False)

    print("=================================================================")
    print("      OpenArm 7-DoF Real-time Joint Receiver & Teleoperation     ")
    print(f"      Listening on UDP: {args.ip}:{args.port}                    ")
    print("=================================================================")
    print("Waiting for packets from arm_tracking.py...\n")

    joint_names = ["J1 (Shoulder Yaw)", "J2 (Shoulder Pitch)", "J3 (Shoulder Roll)",
                   "J4 (Elbow Pitch)", "J5 (Wrist Roll)", "J6 (Wrist Pitch)", "J7 (Wrist Yaw)"]
    joint_limits = [(-160, 160), (-110, 110), (-170, 170), (0, 150), (-170, 170), (-90, 90), (-90, 90)]

    packet_count = 0
    last_print_time = time.time()

    try:
        while True:
            try:
                data, _ = sock.recvfrom(8192)
                packet_count += 1
                now = time.time()

                if now - last_print_time >= 0.066: # ~15 FPS console refresh
                    last_print_time = now
                    payload = json.loads(data.decode("utf-8"))

                    q_deg = payload.get("open_arm_7dof", {}).get("joint_angles_deg", [0.0]*7)
                    swivel = payload.get("swivel_angle_deg", 0.0)
                    mode = payload.get("tracking_mode", "UNKNOWN")
                    pos = payload.get("wrist_pose", {}).get("position", {})

                    # Clear screen (ANSI terminal)
                    sys.stdout.write("\033[H\033[J")
                    print("=================================================================")
                    print(f"   OpenArm 7-DoF Teleop Active | Packets: {packet_count} | Mode: {mode}")
                    print("=================================================================")
                    print(f" End-Effector 3D : X={pos.get('x',0):+.3f}m, Y={pos.get('y',0):+.3f}m, Z={pos.get('z',0):+.3f}m")
                    print(f" Swivel Redundancy: {swivel:+6.1f}° (Elbow Orbit Angle)")
                    print("-----------------------------------------------------------------")
                    print(" [OpenArm 7-DoF Joint Command Angles]")
                    for i in range(7):
                        name = f"J{i+1}"
                        val = q_deg[i] if i < len(q_deg) else 0.0
                        l_min, l_max = joint_limits[i]
                        print("  " + render_joint_bar(name, val, l_min, l_max) + f"  ({joint_names[i]})")
                    print("=================================================================")
                    print("Press Ctrl+C to stop.")

            except BlockingIOError:
                time.sleep(0.005)
            except Exception as e:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nReceiver stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
