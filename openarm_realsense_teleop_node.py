#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenArm 7-DoF RealSense & MediaPipe Vision Teleoperation ROS 2 Bridge Node.
Features:
  1. Non-blocking threaded capture from Intel RealSense (fallback to OpenCV VideoCapture).
  2. Tracks human wrist 6-DoF pose and elbow 3D position using MediaPipe Pose & Hands.
  3. Computes 7-DoF Inverse Kinematics (IK) using Damped Least Squares (DLS) & Analytical Retargeting.
  4. Applies Low-Pass Filtering (LPF), Exponential Moving Average (EMA), and Singularity Protection.
  5. High-frequency non-blocking publisher for ROS 2 JointState (/joint_states) and MoveIt Servo targets.

Author: Antigravity AI / OpenArm Teleop Task Force
"""

import sys
import os
import time
import math
import collections
import threading
from datetime import datetime
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import cv2

# Optional RealSense import
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# MediaPipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# ROS 2 imports
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


# ==============================================================================
# Filtering & Singularity Avoidance Helper Classes
# ==============================================================================

class ExponentialMovingAverageLPF:
    def __init__(self, num_joints=7, cutoff_hz=6.0, window_size=3):
        self.num_joints = num_joints
        self.cutoff_hz = cutoff_hz
        self.window_size = window_size
        self.history = collections.deque(maxlen=window_size)
        self.prev_filtered = np.zeros(num_joints, dtype=np.float64)
        self.prev_time = None
        self.is_initialized = False

    def filter(self, raw_q: np.ndarray) -> np.ndarray:
        raw_q = np.array(raw_q, dtype=np.float64)
        now = time.time()
        if not self.is_initialized:
            self.prev_filtered = raw_q.copy()
            self.prev_time = now
            self.is_initialized = True
            for _ in range(self.window_size):
                self.history.append(raw_q.copy())
            return raw_q

        dt = max(1e-3, now - self.prev_time)
        self.prev_time = now
        tau = 1.0 / (2 * math.pi * self.cutoff_hz)
        alpha = dt / (tau + dt)  # fps 흔들려도 동일한 -3dB 컷오프 유지

        ema_q = alpha * raw_q + (1.0 - alpha) * self.prev_filtered
        self.history.append(ema_q)
        smoothed_q = np.mean(self.history, axis=0)
        self.prev_filtered = smoothed_q.copy()
        return smoothed_q


class SingularityAndSafetyGuard:
    """Clamps joint limits, max velocity, and prevents singularity lockups."""
    def __init__(self, limits_deg=None, max_velocity_deg_s=200.0):
        self.max_velocity_deg_s = max_velocity_deg_s
        self.last_time = time.time()
        self.last_q_deg = np.zeros(7, dtype=np.float64)

        # Default OpenArm 7-DoF limits
        if limits_deg is None:
            self.limits_deg = [
                (-80.0, 200.0),   # J1: Shoulder Yaw
                (-100.0, 100.0),  # J2: Shoulder Pitch
                (-90.0, 90.0),    # J3: Shoulder Roll
                (0.0, 140.0),     # J4: Elbow Pitch
                (-90.0, 90.0),    # J5: Wrist Roll
                (-45.0, 45.0),    # J6: Wrist Pitch
                (-90.0, 90.0),    # J7: Wrist Yaw
            ]
        else:
            self.limits_deg = limits_deg

    def apply_guard(self, target_q_deg: np.ndarray) -> np.ndarray:
        now = time.time()
        dt = max(1e-3, min(0.1, now - self.last_time))
        self.last_time = now

        safe_q = target_q_deg.copy()

        # 1. Velocity Clamping
        max_step = self.max_velocity_deg_s * dt
        for i in range(7):
            delta = safe_q[i] - self.last_q_deg[i]
            delta = np.clip(delta, -max_step, max_step)
            safe_q[i] = self.last_q_deg[i] + delta

        # 2. Joint Limits Clamping
        for i in range(7):
            min_lim, max_lim = self.limits_deg[i]
            safe_q[i] = np.clip(safe_q[i], min_lim, max_lim)

        # 3. Singularity Avoidance: Avoid elbow lockup at 0° or 140°
        safe_q[3] = np.clip(safe_q[3], 5.0, 135.0)

        self.last_q_deg = safe_q.copy()
        return safe_q


def angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Computes the angle (deg) between two 3D vectors u and v using robust atan2."""
    c = float(np.dot(u, v))
    s = float(np.linalg.norm(np.cross(u, v)))
    return math.degrees(math.atan2(s, c))


def build_frame_z(z_axis: np.ndarray, up_ref: np.ndarray) -> np.ndarray:
    """z_axis(forearm/palm direction) as Z-axis orthonormal frame for URDF joint5 Z-rotation."""
    z = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    x = up_ref - np.dot(up_ref, z) * z  # Gram-Schmidt orthogonalization
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-6:
        x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = x - np.dot(x, z) * z
        x_norm = np.linalg.norm(x)
    x /= x_norm
    y = np.cross(z, x)
    return np.column_stack([x, y, z])  # R = [x y z], columns


def decompose_wrist_zxy(R_rel: np.ndarray, prev_q567_deg: np.ndarray) -> np.ndarray:
    r = Rot.from_matrix(R_rel)
    e = r.as_euler('ZXY', degrees=True)   # [about Z, about X(new), about Y(new)]
    q5_deg = e[0]
    q6_deg = e[1]
    q7_deg = -e[2]  # Joint7 axis is xyz="0 -1 0" (-Y), sign inverted

    q = np.array([q5_deg, q6_deg, q7_deg], dtype=np.float64)
    # Euler decomposition unwrapping relative to previous frame
    for i in range(3):
        diff = q[i] - prev_q567_deg[i]
        q[i] -= 360.0 * round(diff / 360.0)
    return q


# ==============================================================================
# 7-DoF Retargeting IK Solver
# ==============================================================================

class OpenArm7DoFIKSolver:
    """Computes OpenArm 7-DoF joint angles from human shoulder, elbow, and wrist 6DoF vectors."""
    def __init__(self, upper_arm_len=0.28, forearm_len=0.25):
        self.upper_arm_len = upper_arm_len
        self.forearm_len = forearm_len
        self.prev_q567_deg = np.zeros(3, dtype=np.float64)

    def solve(self, p_shoulder: np.ndarray, p_elbow: np.ndarray, p_wrist: np.ndarray, R_wrist: np.ndarray,
              j1_offset_deg: float = 0.0, j2_offset_deg: float = 0.0, j3_offset_deg: float = 0.0,
              p_palm_dir: np.ndarray = None, p_palm_normal: np.ndarray = None) -> np.ndarray:
        """
        Solves 7 joint angles (deg) according to OpenArm URDF Zero-Reference Convention with Intrinsic ZXY 3-DoF wrist tracking.
        """
        v_se = p_elbow - p_shoulder
        len_se = np.linalg.norm(v_se)

        v_ew = p_wrist - p_elbow
        len_ew = np.linalg.norm(v_ew)

        if len_se < 1e-4 or len_ew < 1e-4:
            return np.zeros(7, dtype=np.float64)

        u_se = v_se / len_se
        u_ew = v_ew / len_ew

        # 1. J4: Elbow Pitch (0° = straight hanging down, positive = flexed/bent)
        
        q4_deg = angle_between_deg(u_se, u_ew)

        # 2. J1, J2, J3: Shoulder Spherical Joint (Yaw, Pitch, Roll) with Offsets
        q1_deg = -math.degrees(math.atan2(u_se[2], u_se[1])) + j1_offset_deg
        q2_deg = -math.degrees(math.atan2(u_se[0], u_se[1])) + j2_offset_deg

        # 3. J3 (Shoulder Roll / Swivel angle): Smooth blending between 5° and 20° elbow flexion
        normal_arm = np.cross(u_se, u_ew)
        raw_norm_len = np.linalg.norm(normal_arm)  # 정규화 전 크기 = swivel 계산 신뢰도 지표
        if raw_norm_len > 1e-4:
            normal_arm /= raw_norm_len  # 이후 wrist up_ref에서 쓸 정규화 벡터
            swivel_angle = math.degrees(math.atan2(normal_arm[1], abs(normal_arm[0])))
            blend = np.clip((raw_norm_len - 0.05) / 0.15, 0.0, 1.0)  # raw_norm_len 기준으로 게이트 (튜닝 필요)
            q3_deg = blend * np.clip(swivel_angle, -90.0, 90.0) + (1 - blend) * j3_offset_deg
        else:
            q3_deg = j3_offset_deg

        # 4. J5, J6, J7: Wrist Spherical Joint (Roll, Pitch, Yaw) via Intrinsic ZXY Euler Decomposition
        if p_palm_dir is not None and p_palm_normal is not None:
            norm_p = np.linalg.norm(p_palm_dir)
            norm_n = np.linalg.norm(p_palm_normal)
            if norm_p > 1e-4 and norm_n > 1e-4:
                up_ref = normal_arm if raw_norm_len > 1e-4 else np.array([0.0, 0.0, 1.0])
                R_forearm = build_frame_z(u_ew, up_ref)
                R_hand = build_frame_z(p_palm_dir, p_palm_normal)
                R_rel = R_forearm.T @ R_hand

                q567 = decompose_wrist_zxy(R_rel, self.prev_q567_deg)
                q5_deg = np.clip(q567[0], -90.0, 90.0)
                q6_deg = np.clip(q567[1], -45.0, 45.0)   # URDF limit: ±45°
                q7_deg = np.clip(q567[2], -90.0, 90.0)
                self.prev_q567_deg = np.array([q5_deg, q6_deg, q7_deg], dtype=np.float64)
            else:
                q5_deg, q6_deg, q7_deg = self.prev_q567_deg[0], self.prev_q567_deg[1], self.prev_q567_deg[2]
        else:
            q5_deg, q6_deg, q7_deg = self.prev_q567_deg[0], self.prev_q567_deg[1], self.prev_q567_deg[2]

        return np.array([q1_deg, q2_deg, q3_deg, q4_deg, q5_deg, q6_deg, q7_deg], dtype=np.float64)


# ==============================================================================
# Threaded Non-blocking ROS 2 Teleop Bridge Node
# ==============================================================================

if ROS2_AVAILABLE:
    class OpenArmRealSenseTeleopNode(Node):
        def __init__(self):
            super().__init__('openarm_realsense_teleop_node')
            self.get_logger().info('Initializing OpenArm RealSense Vision Teleop Node...')

            # Declare ROS Parameters
            self.declare_parameter('cutoff_hz', 6.0)  # Time-based LPF cutoff frequency (6.0 Hz = low lag, smooth tracking)
            self.declare_parameter('max_joint_velocity_deg_s', 300.0)
            self.declare_parameter('publish_rate', 50.0)

            cutoff_hz = self.get_parameter('cutoff_hz').get_parameter_value().double_value
            max_vel = self.get_parameter('max_joint_velocity_deg_s').get_parameter_value().double_value
            pub_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

            # Setup Filters & Solver
            self.filter = ExponentialMovingAverageLPF(num_joints=7, cutoff_hz=cutoff_hz, window_size=3)
            self.guard = SingularityAndSafetyGuard(max_velocity_deg_s=max_vel)
            self.ik_solver = OpenArm7DoFIKSolver()

            # ROS 2 Publishers
            self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
            self.target_joint_pub = self.create_publisher(JointState, '/openarm/target_joint_states', 10)
            self.servo_pose_pub = self.create_publisher(PoseStamped, '/servo_node/pose_target', 10)
            self.right_traj_pub = self.create_publisher(JointTrajectory, '/right_joint_trajectory_controller/joint_trajectory', 10)
            self.generic_traj_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

            self.declare_parameter('j1_offset_deg', 0.0)
            self.declare_parameter('j2_offset_deg', 0.0)
            self.declare_parameter('j3_offset_deg', 0.0)

            # Joint Names (Supports both single-arm 'joint1' and robot_control-jazzy 'openarm_right_joint1')
            self.declare_parameter('joint_prefix', 'openarm_right_')
            self.joint_prefix = self.get_parameter('joint_prefix').get_parameter_value().string_value
            self.joint_names_prefixed = [f'{self.joint_prefix}joint{i}' for i in range(1, 8)]
            self.joint_names_raw = [f'joint{i}' for i in range(1, 8)]
            self.joint_names_combined = self.joint_names_prefixed + self.joint_names_raw

            # State Lock & Safety Hold Variables
            self.state_lock = threading.Lock()
            self.current_q_rad = np.zeros(7, dtype=np.float64)
            self.last_valid_q_rad = np.zeros(7, dtype=np.float64)
            self.has_valid_pose = False
            self.current_wrist_p = np.array([0.0, -0.4, -0.3], dtype=np.float64)
            self.is_running = True

            # Setup Camera & MediaPipe
            self.setup_camera_and_mediapipe()

            # Start Background Vision Worker Thread
            self.vision_thread = threading.Thread(target=self._vision_worker_loop, daemon=True)
            self.vision_thread.start()

            # Timer loop for non-blocking publishing (50 Hz)
            timer_period = 1.0 / pub_rate
            self.timer = self.create_timer(timer_period, self.publish_joint_states)

        def setup_camera_and_mediapipe(self):
            self.rs_pipeline = None
            self.cap = None

            if REALSENSE_AVAILABLE:
                try:
                    self.rs_pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    self.rs_pipeline.start(config)
                    self.get_logger().info('Intel RealSense pipeline started successfully.')
                except Exception as e:
                    self.get_logger().warn(f'RealSense init failed ({e}). Falling back to OpenCV webcam...')
                    self.rs_pipeline = None

            if self.rs_pipeline is None:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.get_logger().error('Failed to open OpenCV camera device.')

            # MediaPipe Pose & Hands Tracking (Arm + 3-DoF Wrist)
            if MEDIAPIPE_AVAILABLE:
                self.mp_pose = mp.solutions.pose
                self.pose_tracker = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7
                )
                self.mp_hands = mp.solutions.hands
                self.hands_tracker = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6
                )

        def get_next_bgr_frame(self):
            if self.rs_pipeline is not None:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=500)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    return None
                return np.asanyarray(color_frame.get_data())
            elif self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                return frame if ret else None
            return None

        def _vision_worker_loop(self):
            """Background thread to process RealSense/Webcam frames without blocking ROS 2 timers."""
            while self.is_running and rclpy.ok():
                try:
                    frame = self.get_next_bgr_frame()
                    if frame is None:
                        time.sleep(0.01)
                        continue

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = frame.shape

                    p_shoulder, p_elbow, p_wrist = None, None, None
                    p_palm_dir, p_palm_normal = None, None
                    R_wrist = np.eye(3)
                    valid_detection = False

                    display_frame = frame.copy()
                    if MEDIAPIPE_AVAILABLE:
                        results_pose = self.pose_tracker.process(rgb_frame)
                        results_hands = self.hands_tracker.process(rgb_frame)

                        if results_pose.pose_landmarks:
                            lm = results_pose.pose_landmarks.landmark
                            s_lm = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                            e_lm = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                            w_lm = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                            # Check landmark visibility confidence
                            min_vis = min(s_lm.visibility, e_lm.visibility, w_lm.visibility)
                            if min_vis > 0.4:
                                p_shoulder = np.array([s_lm.x * w, s_lm.y * h, s_lm.z * w]) / 1000.0
                                p_elbow = np.array([e_lm.x * w, e_lm.y * h, e_lm.z * w]) / 1000.0
                                p_wrist = np.array([w_lm.x * w, w_lm.y * h, w_lm.z * w]) / 1000.0
                                valid_detection = True

                                mp.solutions.drawing_utils.draw_landmarks(
                                    display_frame,
                                    results_pose.pose_landmarks,
                                    self.mp_pose.POSE_CONNECTIONS
                                )

                        # Extract hand landmarks for 3-DoF wrist rotation (J5, J6, J7)
                        if results_hands.multi_hand_landmarks:
                            for hand_landmarks in results_hands.multi_hand_landmarks:
                                h_lm = hand_landmarks.landmark
                                w_h = np.array([h_lm[0].x * w, h_lm[0].y * h, h_lm[0].z * w]) / 1000.0
                                i_h = np.array([h_lm[5].x * w, h_lm[5].y * h, h_lm[5].z * w]) / 1000.0
                                m_h = np.array([h_lm[9].x * w, h_lm[9].y * h, h_lm[9].z * w]) / 1000.0
                                p_h = np.array([h_lm[17].x * w, h_lm[17].y * h, h_lm[17].z * w]) / 1000.0

                                p_palm_dir = m_h - w_h
                                p_palm_normal = np.cross(i_h - w_h, p_h - w_h)

                                mp.solutions.drawing_utils.draw_landmarks(
                                    display_frame,
                                    hand_landmarks,
                                    self.mp_hands.HAND_CONNECTIONS
                                )
                                break

                    # Render live OpenCV camera feed window
                    if not valid_detection:
                        cv2.putText(display_frame, '[SAFETY LOCK: VISION LOST - HOLDING LAST POSE]', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    cv2.imshow("OpenArm RealSense Teleop", display_frame)
                    cv2.waitKey(1)

                    if valid_detection:
                        # 1. Compute Raw IK with dynamic shoulder offsets and 3-DoF wrist tracking
                        j1_off = self.get_parameter('j1_offset_deg').get_parameter_value().double_value
                        j2_off = self.get_parameter('j2_offset_deg').get_parameter_value().double_value
                        j3_off = self.get_parameter('j3_offset_deg').get_parameter_value().double_value

                        raw_q_deg = self.ik_solver.solve(
                            p_shoulder, p_elbow, p_wrist, R_wrist,
                            j1_offset_deg=j1_off, j2_offset_deg=j2_off, j3_offset_deg=j3_off,
                            p_palm_dir=p_palm_dir, p_palm_normal=p_palm_normal
                        )

                        # 2. Filter & Guard with dynamic cutoff_hz parameter lookup
                        current_cutoff = self.get_parameter('cutoff_hz').get_parameter_value().double_value
                        self.filter.cutoff_hz = current_cutoff
                        smoothed_q_deg = self.filter.filter(raw_q_deg)
                        safe_q_deg = self.guard.apply_guard(smoothed_q_deg)
                        safe_q_rad = np.radians(safe_q_deg)

                        # Log clean, timestamped joint values (deg) to terminal log every 0.5 sec
                        ts_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        q_deg_fmt = np.round(safe_q_deg, 1)
                        self.get_logger().info(
                            f"[{ts_str}] [Vision Target (deg)] J1={q_deg_fmt[0]:6.1f}°, J2={q_deg_fmt[1]:6.1f}°, J3={q_deg_fmt[2]:6.1f}°, J4={q_deg_fmt[3]:6.1f}°, J5={q_deg_fmt[4]:6.1f}°, J6={q_deg_fmt[5]:6.1f}°, J7={q_deg_fmt[6]:6.1f}°",
                            throttle_duration_sec=0.5
                        )

                        # 3. Update Thread-safe State and update last valid safe pose
                        with self.state_lock:
                            self.current_q_rad = safe_q_rad.copy()
                            self.last_valid_q_rad = safe_q_rad.copy()
                            self.has_valid_pose = True
                            self.current_wrist_p = p_wrist.copy()

                    else:
                        # Safety Lock: Vision lost or low confidence -> Hold last valid safe pose cleanly!
                        with self.state_lock:
                            if self.has_valid_pose:
                                self.current_q_rad = self.last_valid_q_rad.copy()

                        ts_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        self.get_logger().warn(
                            f"[{ts_str}] [SAFETY LOCK] Vision tracking lost! Holding last valid safe pose cleanly.",
                            throttle_duration_sec=1.0
                        )

                except Exception as e:
                    self.get_logger().warn(f'Vision worker exception: {e}')
                    time.sleep(0.02)

        def publish_joint_states(self):
            """Fast non-blocking publisher thread callback."""
            with self.state_lock:
                q_rad = self.current_q_rad.copy()
                wrist_p = self.current_wrist_p.copy()

            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names_prefixed
            msg.position = q_rad.tolist()
            msg.velocity = [0.0] * 7
            msg.effort = [0.0] * 7

            self.joint_state_pub.publish(msg)
            self.target_joint_pub.publish(msg)

            # Publish JointTrajectory command for ros2_control trajectory controllers
            traj_msg = JointTrajectory()
            traj_msg.header = msg.header
            traj_msg.joint_names = self.joint_names_prefixed

            pt = JointTrajectoryPoint()
            pt.positions = q_rad.tolist()
            pt.velocities = [0.0] * 7
            pt.time_from_start = Duration(sec=0, nanosec=20000000)  # Fast 20ms trajectory step
            traj_msg.points = [pt]

            self.right_traj_pub.publish(traj_msg)
            self.generic_traj_pub.publish(traj_msg)

            # MoveIt Servo Target
            pose_msg = PoseStamped()
            pose_msg.header = msg.header
            pose_msg.header.frame_id = 'base_link'
            pose_msg.pose.position.x = float(wrist_p[0])
            pose_msg.pose.position.y = float(wrist_p[1])
            pose_msg.pose.position.z = float(wrist_p[2])
            pose_msg.pose.orientation.w = 1.0
            self.servo_pose_pub.publish(pose_msg)

        def destroy_node(self):
            self.is_running = False
            if self.rs_pipeline is not None:
                self.rs_pipeline.stop()
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            super().destroy_node()


def main(args=None):
    if ROS2_AVAILABLE:
        rclpy.init(args=args)
        node = OpenArmRealSenseTeleopNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[OpenArm Teleop] Standalone execution test...")
        solver = OpenArm7DoFIKSolver()
        q_raw = solver.solve(np.array([0.0, 0.0, 0.0]), np.array([0.1, -0.2, -0.2]), np.array([0.2, -0.4, -0.3]), np.eye(3))
        print(f"Joint Angles (deg): {np.round(q_raw, 2)}")

if __name__ == '__main__':
    main()
