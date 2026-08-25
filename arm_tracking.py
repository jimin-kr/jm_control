#!/usr/bin/env python3
"""
Human Arm Motion Tracking System for 7-DoF OpenArm Robot Teleoperation.
Uses Intel RealSense RGB-D Camera, ArUco Marker (Wrist 6-DoF), MediaPipe Pose,
and Kinematic Dead-Reckoning Fusion to maintain continuous tracking directly on the user's arm.

Author: OpenArm Vision Team
"""

import sys
import os
import time
import math
import json
import socket
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

from realsense_camera import RealSenseCamera, MockRealSenseCamera, PYREALSENSE_AVAILABLE
from aruco_tracker import ArUcoTracker, WristPose
from pose_tracker import MediaPipeArmTracker, ArmJointsData
from filters import OneEuroFilter, ExponentialMovingAverage, QuaternionFilter, AngleContinuityFilter
from open_arm_kinematics import OpenArm7DoFSolver, OpenArm7DoFState


@dataclass
class ArmTrackingResult:
    """
    Complete structured telemetry packet produced for every video frame.
    """
    timestamp: float = 0.0
    frame_index: int = 0
    arm_side: str = "right"
    tracking_mode: str = "SEARCHING"
    is_tracking_valid: bool = False
    is_fused_estimate: bool = False
    
    # 6-DoF Wrist Pose
    wrist_pose: Optional[WristPose] = None
    
    # 3D Joint Positions (Shoulder, Elbow, Wrist)
    shoulder_pos_3d: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    elbow_pos_3d: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    wrist_pos_3d: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    
    # 7-DoF Redundancy
    arm_swivel_angle_deg: float = 0.0
    arm_swivel_angle_rad: float = 0.0
    elbow_joint_angle_deg: float = 0.0
    upper_arm_length_m: float = 0.0
    forearm_length_m: float = 0.0
    is_singularity: bool = False

    # OpenArm 7-DoF Joint Angles State
    open_arm_7dof: Optional[OpenArm7DoFState] = None
    
    # Filtering status
    is_filtered: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "arm_side": self.arm_side,
            "tracking_mode": self.tracking_mode,
            "is_tracking_valid": self.is_tracking_valid,
            "is_fused_estimate": self.is_fused_estimate,
            "is_singularity": self.is_singularity,
            "is_filtered": self.is_filtered,
            "wrist_6dof": self.wrist_pose.to_dict() if self.wrist_pose else None,
            "joints_3d_meters": {
                "shoulder": self.shoulder_pos_3d.tolist(),
                "elbow": self.elbow_pos_3d.tolist(),
                "wrist": self.wrist_pos_3d.tolist()
            },
            "link_lengths_m": {
                "upper_arm": float(self.upper_arm_length_m),
                "forearm": float(self.forearm_length_m)
            },
            "redundancy": {
                "swivel_angle_deg": float(self.arm_swivel_angle_deg),
                "swivel_angle_rad": float(self.arm_swivel_angle_rad),
                "elbow_joint_angle_deg": float(self.elbow_joint_angle_deg)
            },
            "open_arm_7dof": self.open_arm_7dof.to_dict() if self.open_arm_7dof else None
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ArmTrackingPipeline:
    """
    Integrated Arm Motion Tracking Pipeline with Anti-Occlusion Kinematic Fusion.
    """
    def __init__(
        self,
        camera_width: int = 1280,
        camera_height: int = 720,
        fps: int = 30,
        marker_id: int = 0,
        marker_size_m: float = 0.05,
        dict_name: str = "DICT_4X4_50",
        arm_side: str = "right",
        mirror: bool = False,
        use_mock: bool = False,
        video_source: Optional[str] = None,
        enable_filtering: bool = True,
        filter_type: str = "one_euro",
        udp_port: Optional[int] = None,
        log_file: Optional[str] = None
    ):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fps = fps
        self.arm_side = arm_side.lower()
        self.mirror = mirror
        self.enable_filtering = enable_filtering
        self.filter_type = filter_type
        self.udp_port = udp_port
        self.log_file = log_file
        self.log_fp = None

        if self.log_file:
            self.log_fp = open(self.log_file, "a", encoding="utf-8")

        self.udp_socket = None
        if self.udp_port is not None:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[Telemetry] Streaming JSON telemetry over UDP to 127.0.0.1:{self.udp_port}")

        # 1. Initialize Camera
        if use_mock or not PYREALSENSE_AVAILABLE:
            if not PYREALSENSE_AVAILABLE and not use_mock:
                print("[Camera] pyrealsense2 not detected. Falling back to Mock/Webcam mode.")
            self.camera = MockRealSenseCamera(
                width=camera_width,
                height=camera_height,
                fps=fps,
                source=video_source
            )
        else:
            try:
                self.camera = RealSenseCamera(
                    width=camera_width,
                    height=camera_height,
                    fps=fps,
                    enable_filters=True
                )
            except Exception as e:
                print(f"[Camera] RealSense initialization failed ({e}). Falling back to Mock mode.")
                self.camera = MockRealSenseCamera(
                    width=camera_width,
                    height=camera_height,
                    fps=fps,
                    source=video_source
                )

        # 2. Initialize Trackers
        self.aruco_tracker = ArUcoTracker(
            target_marker_id=marker_id,
            marker_size_m=marker_size_m,
            dict_name=dict_name,
            axis_length_m=marker_size_m * 0.85
        )
        self.pose_tracker = MediaPipeArmTracker(
            arm_side=self.arm_side,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 3. Initialize Filters
        self._init_filters()

        # 4. Initialize OpenArm 7-DoF Inverse Kinematics Solver
        self.open_arm_solver = OpenArm7DoFSolver()

        self.frame_count = 0
        self.fps_display = 0.0
        self.last_fps_time = time.time()
        self.fps_frame_counter = 0

    def _init_filters(self):
        if self.filter_type == "one_euro":
            self.filter_wrist_pos = OneEuroFilter(min_cutoff=0.6, beta=0.03, d_cutoff=1.0)
            self.filter_shoulder_pos = OneEuroFilter(min_cutoff=0.5, beta=0.01, d_cutoff=1.0)
            self.filter_elbow_pos = OneEuroFilter(min_cutoff=0.7, beta=0.02, d_cutoff=1.0)
        else:
            self.filter_wrist_pos = ExponentialMovingAverage(alpha=0.45)
            self.filter_shoulder_pos = ExponentialMovingAverage(alpha=0.35)
            self.filter_elbow_pos = ExponentialMovingAverage(alpha=0.4)

        self.filter_wrist_quat = QuaternionFilter(alpha=0.35)
        self.filter_swivel_angle = AngleContinuityFilter(in_degrees=True, smoothing_alpha=0.3)

    def reset_filters(self):
        self._init_filters()
        print("[Filter] All filters reset.")

    def run_frame(self) -> Tuple[bool, Optional[ArmTrackingResult], Optional[np.ndarray]]:
        now = time.time()
        self.frame_count += 1
        self.fps_frame_counter += 1

        if now - self.last_fps_time >= 1.0:
            self.fps_display = self.fps_frame_counter / (now - self.last_fps_time)
            self.fps_frame_counter = 0
            self.last_fps_time = now

        # 1. Grab Camera Frames
        success, color_img, depth_img, raw_depth = self.camera.get_frames()
        if not success or color_img is None:
            return False, None, None

        # Mirror mode if requested
        if self.mirror:
            color_img = cv2.flip(color_img, 1)
            if depth_img is not None and isinstance(depth_img, np.ndarray):
                depth_img = cv2.flip(depth_img, 1)

        vis_img = color_img.copy()

        # 2. ArUco 6-DoF Wrist Detection (Instant full-frame detection)
        raw_wrist_pose, vis_img = self.aruco_tracker.detect(vis_img, self.camera.intrinsics)

        # 3. MediaPipe / OpenCV Pose & Kinematic Fusion Directly on User's Body
        joints_data, final_wrist_pose, vis_img = self.pose_tracker.process(
            vis_img, raw_depth if not self.mirror else depth_img, self.camera, aruco_pose=raw_wrist_pose
        )

        # 4. Construct Telemetry Packet
        result = ArmTrackingResult(
            timestamp=now,
            frame_index=self.frame_count,
            arm_side=self.arm_side,
            tracking_mode=joints_data.tracking_mode,
            is_tracking_valid=joints_data.is_tracking_valid,
            is_fused_estimate=final_wrist_pose.is_fused_estimate,
            wrist_pose=final_wrist_pose,
            shoulder_pos_3d=joints_data.shoulder.point_3d.copy(),
            elbow_pos_3d=joints_data.elbow.point_3d.copy(),
            wrist_pos_3d=final_wrist_pose.position.copy(),
            upper_arm_length_m=joints_data.upper_arm_length_m,
            forearm_length_m=joints_data.forearm_length_m,
            arm_swivel_angle_deg=joints_data.swivel_angle_deg,
            arm_swivel_angle_rad=joints_data.swivel_angle_rad,
            elbow_joint_angle_deg=joints_data.elbow_angle_deg,
            is_singularity=joints_data.is_singularity,
            is_filtered=self.enable_filtering
        )

        # 5. Apply Anti-Jitter Filters
        if self.enable_filtering and result.is_tracking_valid:
            if isinstance(self.filter_wrist_pos, OneEuroFilter):
                filtered_wrist = self.filter_wrist_pos.filter(result.wrist_pos_3d, now)
                result.shoulder_pos_3d = self.filter_shoulder_pos.filter(result.shoulder_pos_3d, now)
                result.elbow_pos_3d = self.filter_elbow_pos.filter(result.elbow_pos_3d, now)
            else:
                filtered_wrist = self.filter_wrist_pos.update(result.wrist_pos_3d)
                result.shoulder_pos_3d = self.filter_shoulder_pos.update(result.shoulder_pos_3d)
                result.elbow_pos_3d = self.filter_elbow_pos.update(result.elbow_pos_3d)

            result.wrist_pos_3d = filtered_wrist
            if result.wrist_pose:
                result.wrist_pose.position = filtered_wrist
                filtered_quat = self.filter_wrist_quat.update(result.wrist_pose.quaternion)
                result.wrist_pose.quaternion = filtered_quat

            if not result.is_singularity:
                smoothed_deg = self.filter_swivel_angle.update(result.arm_swivel_angle_deg)
                result.arm_swivel_angle_deg = smoothed_deg
                result.arm_swivel_angle_rad = math.radians(smoothed_deg)

        # 6. Solve OpenArm 7-DoF Joint Command Angles (q1 ~ q7)
        if result.is_tracking_valid and result.wrist_pose:
            open_arm_state = self.open_arm_solver.solve_from_joints_and_orientation(
                p_shoulder=result.shoulder_pos_3d,
                p_elbow=result.elbow_pos_3d,
                p_wrist=result.wrist_pos_3d,
                r_target_wrist=result.wrist_pose.rotation_matrix,
                swivel_angle_deg=result.arm_swivel_angle_deg,
                arm_side=self.arm_side
            )
            result.open_arm_7dof = open_arm_state

        # 7. Render HUD Overlay
        self._render_hud_overlay(vis_img, result)

        # 8. Telemetry output
        if self.udp_socket is not None:
            try:
                msg = result.to_json().encode("utf-8")
                self.udp_socket.sendto(msg, ("127.0.0.1", self.udp_port))
            except Exception:
                pass

        if self.log_fp is not None:
            self.log_fp.write(result.to_json() + "\n")
            self.log_fp.flush()

        return True, result, vis_img

    def _render_hud_overlay(self, img: np.ndarray, res: ArmTrackingResult):
        h, w = img.shape[:2]

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 145), (15, 15, 15), -1)
        cv2.rectangle(overlay, (w - 300, 150), (w - 10, 275), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Header info
        title = f"OpenArm 7-DoF Tracking [{res.arm_side.upper()} ARM]"
        cv2.putText(img, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        fps_text = f"FPS: {self.fps_display:.1f} | Frame: {res.frame_index}"
        cv2.putText(img, fps_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Tracking Mode Badge
        if res.wrist_pose and res.wrist_pose.detected and not res.is_fused_estimate:
            mode_text = f"[MODE: {res.wrist_pose.engine_name} 6-DOF]"
            mode_color = (0, 255, 0)
        elif res.is_fused_estimate:
            mode_text = "[MODE: KINEMATIC FUSED (HAND TRACKING)]"
            mode_color = (0, 200, 255)
        else:
            mode_text = "[MODE: SEARCHING TARGET...]"
            mode_color = (0, 0, 255)
        cv2.putText(img, mode_text, (260, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)

        filter_str = f"Filter: {'ON (' + self.filter_type + ')' if res.is_filtered else 'OFF [F]'}"
        cv2.putText(img, filter_str, (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0) if res.is_filtered else (0, 0, 255), 1, cv2.LINE_AA)

        # Arm Lengths Info
        arm_info = f"Arm Links: Upper={res.upper_arm_length_m*100:.1f}cm, Forearm={res.forearm_length_m*100:.1f}cm"
        cv2.putText(img, arm_info, (600, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 255), 1, cv2.LINE_AA)

        # Wrist 6-DoF Display
        wp = res.wrist_pose
        if wp and wp.detected:
            p = res.wrist_pos_3d
            rpy = wp.rpy_deg
            q = wp.quaternion
            source_tag = wp.engine_name if not res.is_fused_estimate else "Fused Estimate"
            wrist_txt1 = f"Wrist Pos ({source_tag}): X={p[0]:+.3f}m  Y={p[1]:+.3f}m  Z={p[2]:+.3f}m"
            wrist_txt2 = f"Wrist RPY: R={rpy[0]:+5.1f} P={rpy[1]:+5.1f} Y={rpy[2]:+5.1f} deg | Quat: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]"
            cv2.putText(img, wrist_txt1, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)
            cv2.putText(img, wrist_txt2, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 255, 150), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Wrist: Searching for human arm & marker...", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 7-DoF Redundancy Panel
        gauge_x = w - 290
        cv2.putText(img, "7-DoF REDUNDANCY", (gauge_x, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        swivel_txt = f"Swivel: {res.arm_swivel_angle_deg:+6.1f} deg"
        elbow_txt = f"Elbow : {res.elbow_joint_angle_deg:5.1f} deg"
        cv2.putText(img, swivel_txt, (gauge_x, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, elbow_txt, (gauge_x, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Swivel visual bar
        bar_w = 240
        bar_x = gauge_x
        bar_y = 250
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (70, 70, 70), -1)
        cv2.line(img, (bar_x + bar_w // 2, bar_y - 2), (bar_x + bar_w // 2, bar_y + 12), (255, 255, 255), 2)
        clamped_deg = np.clip(res.arm_swivel_angle_deg, -180.0, 180.0)
        marker_px = int(bar_x + (clamped_deg + 180.0) / 360.0 * bar_w)
        cv2.circle(img, (marker_px, bar_y + 5), 6, (0, 255, 255), -1)

        # OpenArm 7-DoF Joint Command Angles Panel (Bottom Banner)
        if res.open_arm_7dof:
            q = res.open_arm_7dof.joint_angles_deg
            cv2.rectangle(img, (0, h - 70), (w, h), (15, 15, 15), -1)
            cv2.line(img, (0, h - 70), (w, h - 70), (0, 200, 255), 2)
            cv2.putText(img, "OPENARM 7-DoF JOINT ANGLES (REAL-TIME COMMAND)", (20, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

            q_txt = (f"J1(Yaw):{q[0]:+5.1f} | J2(Pitch):{q[1]:+5.1f} | J3(Roll):{q[2]:+5.1f} | "
                     f"J4(Elbow):{q[3]:5.1f} | J5(W-Roll):{q[4]:+5.1f} | J6(W-Pitch):{q[5]:+5.1f} | J7(W-Yaw):{q[6]:+5.1f} deg")
            cv2.putText(img, q_txt, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 255, 100), 2, cv2.LINE_AA)

        # Bottom keybindings info
        cv2.putText(img, "[Q/ESC] Quit | [F] Filter | [M] Mirror | [R] Reset | [S] Snap",
                    (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    def start(self):
        self.camera.start()

    def close(self):
        self.camera.stop()
        self.pose_tracker.close()
        if self.log_fp is not None:
            self.log_fp.close()
        if self.udp_socket is not None:
            self.udp_socket.close()
        print("[System] Tracking pipeline closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Intel RealSense 7-DoF Human Arm Motion Tracking for OpenArm Robot"
    )
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Framerate (default: 30)")
    parser.add_argument("--arm", type=str, default="right", choices=["right", "left"],
                        help="Target human arm to track (default: right)")
    parser.add_argument("--mirror", action="store_true", help="Mirror video horizontally like a reflection")
    parser.add_argument("--marker-id", type=int, default=0, help="Wrist ArUco Marker ID (default: 0)")
    parser.add_argument("--marker-size", type=float, default=0.05,
                        help="Physical marker side length in meters (default: 0.05m = 50mm)")
    parser.add_argument("--dict", type=str, default="DICT_4X4_50", help="ArUco dictionary (default: DICT_4X4_50)")
    parser.add_argument("--mock", action="store_true", help="Run in mock/simulation mode without RealSense camera")
    parser.add_argument("--video", type=str, default=None, help="Video file or webcam index for offline testing")
    parser.add_argument("--no-filter", action="store_true", help="Disable anti-jitter smoothing filter")
    parser.add_argument("--filter-type", type=str, default="one_euro", choices=["one_euro", "ema"],
                        help="Filter type (default: one_euro)")
    parser.add_argument("--udp-port", type=int, default=None,
                        help="UDP port to stream JSON telemetry packets (e.g. 9870)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to save telemetry JSONL logs")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV GUI window")

    args = parser.parse_args()

    pipeline = ArmTrackingPipeline(
        camera_width=args.width,
        camera_height=args.height,
        fps=args.fps,
        marker_id=args.marker_id,
        marker_size_m=args.marker_size,
        dict_name=args.dict,
        arm_side=args.arm,
        mirror=args.mirror,
        use_mock=args.mock,
        video_source=args.video,
        enable_filtering=not args.no_filter,
        filter_type=args.filter_type,
        udp_port=args.udp_port,
        log_file=args.log_file
    )

    print("=================================================================")
    print("   OpenArm 7-DoF Motion Tracking with Real-time Body Lock        ")
    print("=================================================================")
    print(f" Target Arm     : {args.arm.upper()}")
    print(f" Mirror Mode    : {'ON' if args.mirror else 'OFF [Press M to toggle]'}")
    print(f" ArUco Marker   : ID {args.marker_id} ({args.dict}, {args.marker_size*1000:.0f}mm)")
    print(f" Filter Mode    : {'OFF' if args.no_filter else args.filter_type.upper()}")
    if args.udp_port:
        print(f" UDP Stream     : 127.0.0.1:{args.udp_port}")
    if args.log_file:
        print(f" Log File       : {args.log_file}")
    print("=================================================================")

    try:
        pipeline.start()
        window_name = "OpenArm Human Motion Tracking (RealSense)"

        while True:
            success, result, vis_img = pipeline.run_frame()
            if not success or vis_img is None:
                continue

            if not args.headless:
                cv2.imshow(window_name, vis_img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                elif key in (ord('f'), ord('F')):
                    pipeline.enable_filtering = not pipeline.enable_filtering
                    print(f"[UI] Filter toggled: {'ON' if pipeline.enable_filtering else 'OFF'}")
                elif key in (ord('m'), ord('M')):
                    pipeline.mirror = not pipeline.mirror
                    print(f"[UI] Mirror mode toggled: {'ON' if pipeline.mirror else 'OFF'}")
                elif key in (ord('r'), ord('R')):
                    pipeline.reset_filters()
                elif key in (ord('s'), ord('S')):
                    snap_name = f"snapshot_{int(time.time())}.png"
                    cv2.imwrite(snap_name, vis_img)
                    print(f"[UI] Snapshot saved to {snap_name}")
            else:
                time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\n[System] Interrupted by user.")
    finally:
        pipeline.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
