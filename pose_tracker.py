"""
MediaPipe Pose Joint Tracker & 7-DoF Arm Kinematic Fusion Engine.
Features:
- Google MediaPipe Neural Pose Estimation (33 full-body anatomical 3D landmarks)
- Multi-hypothesis Joint Tracking (Shoulder, Elbow, Wrist) with Depth & Kinematic Ray Optimization
- Dual Shoulder Tracking: Visualizes both left & right shoulders so user clearly sees body lock-on
- Dynamic Arm Calibration: Locks upper arm and forearm link lengths from initial high-confidence frames
- Anti-Occlusion Kinematic Solver: Recovers 3D joint positions and 6-DoF wrist pose even when ArUco or depth is occluded
- 7-DoF Swivel Angle (Elbow plane redundancy) computation with singularity damping
- High-visibility Large Target Markers directly on the user's actual anatomical joints
"""

import sys
import os
import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

# Ensure ~/.local/lib/python3.10/site-packages is in sys.path
local_site = os.path.expanduser('~/.local/lib/python3.10/site-packages')
if local_site not in sys.path:
    sys.path.insert(0, local_site)

from realsense_camera import CameraIntrinsics
from aruco_tracker import (
    WristPose,
    rotation_matrix_to_euler_rpy,
    rotation_matrix_to_quaternion,
    euler_rpy_to_rotation_matrix
)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False


@dataclass
class Joint3D:
    """3D Joint data with pixel coords and confidence visibility."""
    name: str
    pixel_u: int = 0
    pixel_v: int = 0
    depth_m: float = 0.0
    point_3d: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    visibility: float = 0.0
    is_valid: bool = False
    is_reconstructed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pixel": [self.pixel_u, self.pixel_v],
            "depth_m": float(self.depth_m),
            "position_3d": {
                "x_m": float(self.point_3d[0]),
                "y_m": float(self.point_3d[1]),
                "z_m": float(self.point_3d[2])
            },
            "visibility": float(self.visibility),
            "is_valid": self.is_valid,
            "is_reconstructed": self.is_reconstructed
        }


@dataclass
class ArmJointsData:
    """Holds 3D positions of the human arm joints and 7-DoF redundancy metrics."""
    arm_side: str = "right"
    shoulder: Joint3D = field(default_factory=lambda: Joint3D("shoulder"))
    elbow: Joint3D = field(default_factory=lambda: Joint3D("elbow"))
    wrist: Joint3D = field(default_factory=lambda: Joint3D("wrist"))
    opposite_shoulder: Joint3D = field(default_factory=lambda: Joint3D("opp_shoulder"))
    
    swivel_angle_rad: float = 0.0
    swivel_angle_deg: float = 0.0
    elbow_angle_deg: float = 0.0
    upper_arm_length_m: float = 0.0
    forearm_length_m: float = 0.0
    is_singularity: bool = False
    is_tracking_valid: bool = False
    tracking_mode: str = "SEARCHING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_side": self.arm_side,
            "tracking_mode": self.tracking_mode,
            "shoulder": self.shoulder.to_dict(),
            "elbow": self.elbow.to_dict(),
            "wrist": self.wrist.to_dict(),
            "redundancy_swivel_angle": {
                "rad": float(self.swivel_angle_rad),
                "deg": float(self.swivel_angle_deg)
            },
            "elbow_angle_deg": float(self.elbow_angle_deg),
            "link_lengths_m": {
                "upper_arm": float(self.upper_arm_length_m),
                "forearm": float(self.forearm_length_m)
            },
            "is_singularity": self.is_singularity,
            "is_tracking_valid": self.is_tracking_valid
        }


def compute_swivel_angle(
    shoulder_3d: np.ndarray,
    elbow_3d: np.ndarray,
    wrist_3d: np.ndarray,
    ref_axis: Optional[np.ndarray] = None
) -> Tuple[float, float, float, bool]:
    """Computes the 7-DoF Swivel Angle (Arm Angle psi)."""
    p_s = np.asarray(shoulder_3d, dtype=np.float64)
    p_e = np.asarray(elbow_3d, dtype=np.float64)
    p_w = np.asarray(wrist_3d, dtype=np.float64)

    v_se = p_e - p_s
    v_ew = p_w - p_e
    v_sw = p_w - p_s

    len_se = np.linalg.norm(v_se)
    len_ew = np.linalg.norm(v_ew)
    len_sw = np.linalg.norm(v_sw)

    if len_se < 1e-3 or len_ew < 1e-3 or len_sw < 1e-3:
        return 0.0, 0.0, 0.0, True

    u_sw = v_sw / len_sw

    cos_elbow = np.dot(-v_se, v_ew) / (len_se * len_ew)
    cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
    elbow_angle_deg = float(np.degrees(math.acos(cos_elbow)))

    n_arm = np.cross(v_se, v_sw)
    n_arm_norm = np.linalg.norm(n_arm)

    is_singularity = (elbow_angle_deg > 165.0) or (elbow_angle_deg < 15.0) or (n_arm_norm < 1e-3)

    if is_singularity or n_arm_norm < 1e-4:
        return 0.0, 0.0, elbow_angle_deg, True

    u_n_arm = n_arm / n_arm_norm

    if ref_axis is None:
        ref_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    dot_ref = np.dot(u_sw, ref_axis)
    if abs(dot_ref) > 0.92:
        ref_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    n_ref = np.cross(ref_axis, u_sw)
    n_ref_norm = np.linalg.norm(n_ref)
    if n_ref_norm < 1e-4:
        return 0.0, 0.0, elbow_angle_deg, True
    u_n_ref = n_ref / n_ref_norm

    cos_psi = np.dot(u_n_ref, u_n_arm)
    sin_psi = np.dot(np.cross(u_n_ref, u_n_arm), u_sw)

    swivel_rad = float(math.atan2(sin_psi, cos_psi))
    swivel_deg = float(np.degrees(swivel_rad))

    return swivel_rad, swivel_deg, elbow_angle_deg, False


class ArmKinematicFusionEngine:
    """Maintains calibrated human arm link lengths and performs dead-reckoning."""
    def __init__(self):
        self.calibrated = False
        self.upper_arm_len = 0.28
        self.forearm_len = 0.25
        self.wrist_offset = np.zeros(3, dtype=np.float64)

        self.last_valid_shoulder = np.array([0.0, -0.15, 1.2], dtype=np.float64)
        self.last_valid_elbow = np.array([0.15, 0.05, 1.15], dtype=np.float64)
        self.last_valid_wrist = np.array([0.25, -0.05, 1.0], dtype=np.float64)
        self.last_valid_rpy_deg = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.last_valid_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.last_valid_rot_mat = np.eye(3, dtype=np.float64)
        self.last_valid_wrist_px: Optional[Tuple[int, int]] = None
        self.initialized_wrist = False
        self.consecutive_outlier_count = 0

        self.calib_samples = 0
        self.occlusion_frames = 0

    def filter_wrist_jump(
        self,
        p_candidate: np.ndarray,
        px_candidate: Tuple[int, int],
        p_elbow: np.ndarray,
        is_aruco_direct: bool = False
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Suppresses sudden jumps (e.g. wrist jumping across the screen due to
        left/right landmark confusion).
        """
        if not self.initialized_wrist or self.last_valid_wrist_px is None:
            self.last_valid_wrist = p_candidate.copy()
            self.last_valid_wrist_px = px_candidate
            self.initialized_wrist = True
            return p_candidate, px_candidate

        # Distance from elbow sanity check
        dist_from_elbow = np.linalg.norm(p_candidate - p_elbow)
        if dist_from_elbow > 0.55:
            # Kinematically impossible forearm length: hold previous
            return self.last_valid_wrist.copy(), self.last_valid_wrist_px

        # If direct ArUco detection, trust it unless physically impossible
        if is_aruco_direct and dist_from_elbow < 0.50:
            self.last_valid_wrist = p_candidate.copy()
            self.last_valid_wrist_px = px_candidate
            self.consecutive_outlier_count = 0
            return p_candidate, px_candidate

        # Displacement check from last known wrist
        delta_3d = np.linalg.norm(p_candidate - self.last_valid_wrist)
        delta_px = math.hypot(px_candidate[0] - self.last_valid_wrist_px[0],
                              px_candidate[1] - self.last_valid_wrist_px[1])

        # Large jump threshold
        if delta_3d > 0.35 or delta_px > 250:
            if self.consecutive_outlier_count < 4:
                self.consecutive_outlier_count += 1
                return self.last_valid_wrist.copy(), self.last_valid_wrist_px
            else:
                self.consecutive_outlier_count = 0
        else:
            self.consecutive_outlier_count = 0

        self.last_valid_wrist = p_candidate.copy()
        self.last_valid_wrist_px = px_candidate
        return p_candidate, px_candidate

    def update_calibration(self, p_s: np.ndarray, p_e: np.ndarray, p_w: np.ndarray, p_aruco: Optional[np.ndarray]):
        l_se = np.linalg.norm(p_e - p_s)
        l_ew = np.linalg.norm(p_w - p_e)

        if 0.18 < l_se < 0.45 and 0.16 < l_ew < 0.42:
            alpha = 0.05 if self.calibrated else 0.2
            self.upper_arm_len = (1.0 - alpha) * self.upper_arm_len + alpha * l_se
            self.forearm_len = (1.0 - alpha) * self.forearm_len + alpha * l_ew

            if p_aruco is not None:
                offset = p_aruco - p_w
                if np.linalg.norm(offset) < 0.25:
                    self.wrist_offset = (1.0 - alpha) * self.wrist_offset + alpha * offset

            self.calib_samples += 1
            if self.calib_samples > 10:
                self.calibrated = True

    def solve_elbow_on_ray(self, ray_e: np.ndarray, p_s: np.ndarray, p_w: np.ndarray) -> np.ndarray:
        z_base = 0.5 * (p_s[2] + p_w[2]) if p_w[2] > 0 else p_s[2]
        best_d = z_base
        min_err = float('inf')

        for d in np.linspace(max(0.4, z_base - 0.4), min(2.5, z_base + 0.4), 25):
            candidate_e = ray_e * d
            err = (abs(np.linalg.norm(candidate_e - p_s) - self.upper_arm_len) +
                   abs(np.linalg.norm(p_w - candidate_e) - self.forearm_len))
            if err < min_err:
                min_err = err
                best_d = d

        return ray_e * best_d

    def solve_occluded_wrist(
        self,
        ray_w: np.ndarray,
        p_e: np.ndarray,
        raw_depth_w: float
    ) -> np.ndarray:
        """
        Robustly resolves wrist 3D coordinates when ArUco is occluded (e.g. pointing to ground).
        Prevents depth rays from hitting the floor/background behind the hand.
        """
        L = self.forearm_len  # Calibrated forearm length (~0.26m)

        # 1. If raw depth is valid and produces a physically plausible forearm length:
        if raw_depth_w > 0.3:
            candidate_p = ray_w * raw_depth_w
            dist_to_elbow = np.linalg.norm(candidate_p - p_e)
            if 0.18 <= dist_to_elbow <= 0.36:
                return candidate_p

        # 2. Otherwise (depth missed hand and sampled floor/background),
        # Intersect the wrist pixel ray with the sphere of radius L centered at Elbow p_e!
        u_ray = ray_w / np.linalg.norm(ray_w)
        b = -2.0 * np.dot(u_ray, p_e)
        c = np.dot(p_e, p_e) - (L ** 2)
        disc = b * b - 4.0 * c

        if disc >= 0:
            d1 = (-b + math.sqrt(disc)) / 2.0
            d2 = (-b - math.sqrt(disc)) / 2.0
            valid_d = [d for d in (d1, d2) if d > 0.2]
            if valid_d:
                best_d = min(valid_d)
                return u_ray * best_d

        # 3. Geometric projection along ray
        d_proj = max(0.3, float(np.dot(u_ray, p_e)))
        p_proj = u_ray * d_proj
        v_ew = p_proj - p_e
        dist_ew = np.linalg.norm(v_ew)
        if dist_ew > 1e-3:
            return p_e + (v_ew / dist_ew) * L
        return p_e + np.array([0.0, L, 0.0], dtype=np.float64)

    def solve_wrist_on_ray(self, ray_w: np.ndarray, p_e: np.ndarray) -> np.ndarray:
        """Compatibility wrapper for ray-sphere wrist solver."""
        return self.solve_occluded_wrist(ray_w, p_e, raw_depth_w=0.0)

    def estimate_wrist_orientation_from_forearm(
        self,
        p_elbow: np.ndarray,
        p_wrist: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v_ew = p_wrist - p_elbow
        len_ew = np.linalg.norm(v_ew)
        if len_ew < 1e-3:
            return self.last_valid_rpy_deg, self.last_valid_quat, self.last_valid_rot_mat

        u_forearm = v_ew / len_ew
        yaw_rad = math.atan2(u_forearm[0], u_forearm[2])
        pitch_rad = math.asin(np.clip(-u_forearm[1], -1.0, 1.0))
        roll_rad = math.radians(self.last_valid_rpy_deg[0])

        R = euler_rpy_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
        rpy_rad, rpy_deg = rotation_matrix_to_euler_rpy(R)
        quat = rotation_matrix_to_quaternion(R)

        return rpy_deg, quat, R


class MediaPipeArmTracker:
    """
    State-of-the-Art MediaPipe Pose Neural Tracker with Robust Multi-Arm Support.
    """
    def __init__(
        self,
        arm_side: str = "right",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        self.arm_side = arm_side.lower()
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        if self.arm_side == "right":
            self.idx_shoulder = 12       # RIGHT_SHOULDER
            self.idx_elbow = 14          # RIGHT_ELBOW
            self.idx_wrist = 16          # RIGHT_WRIST
            self.idx_opp_shoulder = 11   # LEFT_SHOULDER
        else:
            self.idx_shoulder = 11       # LEFT_SHOULDER
            self.idx_elbow = 13          # LEFT_ELBOW
            self.idx_wrist = 15          # LEFT_WRIST
            self.idx_opp_shoulder = 12   # RIGHT_SHOULDER

        self.pose = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            print(f"[MediaPipe] Neural Pose active: Tracking '{self.arm_side.upper()}' arm.")
        else:
            print("[MediaPipe] Warning: mediapipe not found.")

        self.fusion = ArmKinematicFusionEngine()
        self.last_valid_swivel_deg = 0.0

    def process(
        self,
        color_image: np.ndarray,
        depth_source: Any,
        camera_ctrl: Any,
        aruco_pose: Optional[WristPose] = None
    ) -> Tuple[ArmJointsData, WristPose, np.ndarray]:
        vis_image = color_image.copy()
        h, w = color_image.shape[:2]
        intr = camera_ctrl.intrinsics

        data = ArmJointsData(arm_side=self.arm_side)
        fused_wrist = aruco_pose if (aruco_pose and aruco_pose.detected) else WristPose()

        if self.pose is None:
            return self._fallback_simulated_joints(vis_image, camera_ctrl, fused_wrist)

        # 1. Run MediaPipe Pose Neural Network
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.pose.process(rgb_image)

        if not results.pose_landmarks:
            banner_w, banner_h = 560, 50
            bx = (w - banner_w) // 2
            by = h // 2 - 25
            cv2.rectangle(vis_image, (bx, by), (bx + banner_w, by + banner_h), (20, 20, 20), -1)
            cv2.rectangle(vis_image, (bx, by), (bx + banner_w, by + banner_h), (0, 140, 255), 2)
            cv2.putText(vis_image, "SEARCHING BODY: PLEASE SHOW SHOULDER & CHEST", (bx + 15, by + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)

            if aruco_pose and aruco_pose.detected:
                data.tracking_mode = "ARUCO_ONLY"
                data.wrist.point_3d = aruco_pose.position.copy()
                data.wrist.is_valid = True
                data.is_tracking_valid = True
            return data, fused_wrist, vis_image

        landmarks = results.pose_landmarks.landmark
        lm_s = landmarks[self.idx_shoulder]
        lm_e = landmarks[self.idx_elbow]
        lm_w = landmarks[self.idx_wrist]
        lm_opp = landmarks[self.idx_opp_shoulder]

        u_s, v_s = int(np.clip(lm_s.x * w, 0, w - 1)), int(np.clip(lm_s.y * h, 0, h - 1))
        u_e, v_e = int(np.clip(lm_e.x * w, 0, w - 1)), int(np.clip(lm_e.y * h, 0, h - 1))
        u_w, v_w = int(np.clip(lm_w.x * w, 0, w - 1)), int(np.clip(lm_w.y * h, 0, h - 1))
        u_opp, v_opp = int(np.clip(lm_opp.x * w, 0, w - 1)), int(np.clip(lm_opp.y * h, 0, h - 1))

        # 2. Extract Depth
        d_s = camera_ctrl.get_depth_at_pixel(depth_source, u_s, v_s, patch_radius=6)
        d_e = camera_ctrl.get_depth_at_pixel(depth_source, u_e, v_e, patch_radius=6)
        d_w = camera_ctrl.get_depth_at_pixel(depth_source, u_w, v_w, patch_radius=6)
        d_opp = camera_ctrl.get_depth_at_pixel(depth_source, u_opp, v_opp, patch_radius=6)

        ray_s = np.array([(u_s - intr.cx) / intr.fx, (v_s - intr.cy) / intr.fy, 1.0], dtype=np.float64)
        ray_e = np.array([(u_e - intr.cx) / intr.fx, (v_e - intr.cy) / intr.fy, 1.0], dtype=np.float64)
        ray_w = np.array([(u_w - intr.cx) / intr.fx, (v_w - intr.cy) / intr.fy, 1.0], dtype=np.float64)

        # 2. Resolve Shoulder 3D Position
        if d_s > 0.3:
            p_s = ray_s * d_s
            self.fusion.last_valid_shoulder = p_s.copy()
        elif d_opp > 0.3:
            p_s = ray_s * d_opp
            self.fusion.last_valid_shoulder = p_s.copy()
        else:
            p_s = ray_s * self.fusion.last_valid_shoulder[2]

        p_opp = np.array([(u_opp - intr.cx) / intr.fx, (v_opp - intr.cy) / intr.fy, 1.0]) * (d_opp if d_opp > 0.3 else p_s[2])

        # 3. Resolve Elbow 3D Position (Before wrist in kinematic chain)
        if d_e > 0.3 and abs(np.linalg.norm(ray_e * d_e - p_s) - self.fusion.upper_arm_len) < 0.18:
            p_e = ray_e * d_e
            data.elbow.is_reconstructed = False
        elif d_e > 0.3:
            p_e = ray_e * d_e
            data.elbow.is_reconstructed = False
        else:
            p_e = ray_e * self.fusion.last_valid_elbow[2]
            data.elbow.is_reconstructed = True

        # 4. Resolve Wrist 3D Position
        aruco_detected = (aruco_pose is not None and aruco_pose.detected)

        if aruco_detected:
            # 100% Priority Master: ArUco Marker center & RealSense Depth
            u_w, v_w = aruco_pose.center_2d
            d_aruco = camera_ctrl.get_depth_at_pixel(depth_source, u_w, v_w, patch_radius=6)

            if d_aruco > 0.3:
                ray_aruco = np.array([(u_w - intr.cx) / intr.fx, (v_w - intr.cy) / intr.fy, 1.0], dtype=np.float64)
                p_w = ray_aruco * d_aruco
            else:
                p_w = aruco_pose.position.copy()

            self.fusion.last_valid_wrist = p_w.copy()
            self.fusion.last_valid_wrist_px = (u_w, v_w)
            self.fusion.initialized_wrist = True
            self.fusion.occlusion_frames = 0

            fused_wrist = aruco_pose
            fused_wrist.position = p_w.copy()
            fused_wrist.center_2d = (u_w, v_w)
            fused_wrist.is_fused_estimate = False
            fused_wrist.confidence = 1.0
            data.tracking_mode = "ARUCO_MASTER"

            self.fusion.update_calibration(p_s, p_e, p_w, p_w)
        else:
            # Seamless Fallback: Continuously track hand in real-time without freezing
            self.fusion.occlusion_frames += 1
            data.tracking_mode = "HAND_TRACKING"

            # Use robust occluded wrist ray-sphere solver (prevents floor/background depth spikes)
            p_w = self.fusion.solve_occluded_wrist(ray_w, p_e, d_w)

            self.fusion.last_valid_wrist = p_w.copy()
            self.fusion.last_valid_wrist_px = (u_w, v_w)

            fused_wrist.detected = True
            fused_wrist.is_fused_estimate = True
            fused_wrist.confidence = max(0.4, 0.9 - self.fusion.occlusion_frames * 0.01)
            fused_wrist.position = p_w.copy()
            fused_wrist.center_2d = (u_w, v_w)

            rpy_deg, quat, rot_mat = self.fusion.estimate_wrist_orientation_from_forearm(
                p_e, p_w
            )
            fused_wrist.rpy_deg = rpy_deg
            fused_wrist.rpy_rad = np.radians(rpy_deg)
            fused_wrist.quaternion = quat
            fused_wrist.rotation_matrix = rot_mat

        # 6. Populate Joint Data
        data.shoulder = Joint3D("shoulder", u_s, v_s, float(p_s[2]), p_s, float(lm_s.visibility), True)
        data.elbow = Joint3D("elbow", u_e, v_e, float(p_e[2]), p_e, float(lm_e.visibility), True, data.elbow.is_reconstructed)
        data.wrist = Joint3D("wrist", u_w, v_w, float(p_w[2]), p_w, float(lm_w.visibility), True, not aruco_detected)
        data.opposite_shoulder = Joint3D("opp_shoulder", u_opp, v_opp, float(p_opp[2]), p_opp, float(lm_opp.visibility), True)

        data.upper_arm_length_m = float(np.linalg.norm(p_e - p_s))
        data.forearm_length_m = float(np.linalg.norm(p_w - p_e))

        # 6. Compute 7-DoF Swivel Angle
        swivel_rad, swivel_deg, elbow_deg, is_sing = compute_swivel_angle(p_s, p_e, p_w)
        data.elbow_angle_deg = elbow_deg
        data.is_singularity = is_sing

        if not is_sing:
            data.swivel_angle_rad = swivel_rad
            data.swivel_angle_deg = swivel_deg
            self.last_valid_swivel_deg = swivel_deg
        else:
            data.swivel_angle_deg = self.last_valid_swivel_deg
            data.swivel_angle_rad = math.radians(self.last_valid_swivel_deg)

        data.is_tracking_valid = True

        # 7. Render Skeleton & Big Target Markers on Screen
        self._draw_arm_skeleton(vis_image, data, fused_wrist, results.pose_landmarks)

        return data, fused_wrist, vis_image

    def _fallback_simulated_joints(
        self,
        vis_image: np.ndarray,
        camera_ctrl: Any,
        fused_wrist: WristPose
    ) -> Tuple[ArmJointsData, WristPose, np.ndarray]:
        h, w = vis_image.shape[:2]
        data = ArmJointsData(arm_side=self.arm_side, tracking_mode="MOCK_SIMULATION")

        u_s, v_s = int(w * 0.5), int(h * 0.35)
        u_e, v_e = int(w * 0.55), int(h * 0.55)
        u_w, v_w = int(w * 0.6), int(h * 0.75)

        p_s = camera_ctrl.deproject_pixel_to_point(u_s, v_s, 1.2)
        p_e = camera_ctrl.deproject_pixel_to_point(u_e, v_e, 1.15)
        p_w = fused_wrist.position.copy() if fused_wrist.detected else camera_ctrl.deproject_pixel_to_point(u_w, v_w, 1.1)

        data.shoulder = Joint3D("shoulder", u_s, v_s, 1.2, p_s, 1.0, True)
        data.elbow = Joint3D("elbow", u_e, v_e, 1.15, p_e, 1.0, True)
        data.wrist = Joint3D("wrist", u_w, v_w, 1.1, p_w, 1.0, True)

        sw_rad, sw_deg, el_deg, is_sing = compute_swivel_angle(p_s, p_e, p_w)
        data.swivel_angle_rad = sw_rad
        data.swivel_angle_deg = sw_deg
        data.elbow_angle_deg = el_deg
        data.is_singularity = is_sing
        data.is_tracking_valid = True

        self._draw_arm_skeleton(vis_image, data, fused_wrist, None)
        return data, fused_wrist, vis_image

    def _draw_joint_marker(self, img: np.ndarray, x: int, y: int, color_bgr: Tuple[int, int, int], label: str, sub_label: str):
        """Draws a large prominent glowing target marker with high visibility directly on physical joint."""
        # 1. Outer glowing pulse ring
        cv2.circle(img, (x, y), 24, color_bgr, 3, cv2.LINE_AA)
        cv2.circle(img, (x, y), 30, color_bgr, 1, cv2.LINE_AA)
        # 2. Solid core
        cv2.circle(img, (x, y), 10, color_bgr, -1, cv2.LINE_AA)
        # 3. Bright white center dot
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Crosshairs
        tick = 10
        cv2.line(img, (x - 30, y), (x - 30 + tick, y), color_bgr, 2, cv2.LINE_AA)
        cv2.line(img, (x + 30 - tick, y), (x + 30, y), color_bgr, 2, cv2.LINE_AA)
        cv2.line(img, (x, y - 30), (x, y - 30 + tick), color_bgr, 2, cv2.LINE_AA)
        cv2.line(img, (x, y + 30 - tick), (x, y + 30), color_bgr, 2, cv2.LINE_AA)

        # Text Badge
        text = f"{label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bx, by = x + 34, y - 10
        cv2.rectangle(img, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (20, 20, 20), -1)
        cv2.rectangle(img, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), color_bgr, 1)
        cv2.putText(img, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)

        if sub_label:
            (stw, sth), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (bx - 4, by + 6), (bx + stw + 4, by + sth + 10), (20, 20, 20), -1)
            cv2.putText(img, sub_label, (bx, by + sth + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_arm_skeleton(self, img: np.ndarray, data: ArmJointsData, wrist: WristPose, raw_landmarks: Any):
        h, w = img.shape[:2]
        s, e, wr = data.shoulder, data.elbow, data.wrist
        opp_s = data.opposite_shoulder

        if raw_landmarks is not None:
            lm = raw_landmarks.landmark
            l_sh = lm[11]
            r_sh = lm[12]
            l_hip = lm[23]
            r_hip = lm[24]
            p_l_sh = (int(l_sh.x * w), int(l_sh.y * h))
            p_r_sh = (int(r_sh.x * w), int(r_sh.y * h))
            p_l_hip = (int(l_hip.x * w), int(l_hip.y * h))
            p_r_hip = (int(r_hip.x * w), int(r_hip.y * h))

            # Chest & Torso line connecting both shoulders
            cv2.line(img, p_l_sh, p_r_sh, (0, 200, 255), 3, cv2.LINE_AA)
            cv2.line(img, p_l_sh, p_l_hip, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.line(img, p_r_sh, p_r_hip, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.line(img, p_l_hip, p_r_hip, (80, 80, 80), 1, cv2.LINE_AA)

            # Draw small marker on opposite shoulder for reference
            cv2.circle(img, (opp_s.pixel_u, opp_s.pixel_v), 10, (120, 120, 120), -1, cv2.LINE_AA)
            cv2.putText(img, "Opposite Shoulder", (opp_s.pixel_u + 12, opp_s.pixel_v - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

        # Thick Glowing Arm Bones directly between joints
        cv2.line(img, (s.pixel_u, s.pixel_v), (e.pixel_u, e.pixel_v), (255, 160, 0), 6, cv2.LINE_AA)
        cv2.line(img, (s.pixel_u, s.pixel_v), (e.pixel_u, e.pixel_v), (255, 230, 100), 2, cv2.LINE_AA)

        forearm_color_glow = (50, 255, 50) if not wrist.is_fused_estimate else (0, 180, 255)
        forearm_color_core = (180, 255, 180) if not wrist.is_fused_estimate else (150, 230, 255)
        cv2.line(img, (e.pixel_u, e.pixel_v), (wr.pixel_u, wr.pixel_v), forearm_color_glow, 6, cv2.LINE_AA)
        cv2.line(img, (e.pixel_u, e.pixel_v), (wr.pixel_u, wr.pixel_v), forearm_color_core, 2, cv2.LINE_AA)

        # 1. Main Target Shoulder Marker (Neon Orange)
        sh_color = (0, 140, 255)
        sh_sub = f"[{s.point_3d[0]:+.2f}, {s.point_3d[1]:+.2f}, {s.point_3d[2]:.2f}]m"
        self._draw_joint_marker(img, s.pixel_u, s.pixel_v, sh_color, f"SHOULDER ({data.arm_side.upper()})", sh_sub)

        # 2. Elbow Marker (Neon Yellow)
        el_color = (0, 255, 255) if not e.is_reconstructed else (0, 165, 255)
        el_tag = "Depth" if not e.is_reconstructed else "Kinematic"
        el_sub = f"[{e.point_3d[0]:+.2f}, {e.point_3d[1]:+.2f}, {e.point_3d[2]:.2f}]m ({el_tag})"
        self._draw_joint_marker(img, e.pixel_u, e.pixel_v, el_color, "ELBOW", el_sub)

        # 3. Wrist Marker (Neon Green if ArUco / Neon Amber if Fused)
        wr_color = (50, 255, 50) if not wrist.is_fused_estimate else (0, 215, 255)
        wr_tag = "ArUco 6DoF" if not wrist.is_fused_estimate else "Fused Estimate"
        wr_sub = f"[{wr.point_3d[0]:+.2f}, {wr.point_3d[1]:+.2f}, {wr.point_3d[2]:.2f}]m ({wr_tag})"
        self._draw_joint_marker(img, wr.pixel_u, wr.pixel_v, wr_color, "WRIST", wr_sub)

        status_box_x = w // 2 - 160
        status_box_y = 155
        cv2.rectangle(img, (status_box_x, status_box_y), (status_box_x + 320, status_box_y + 35), (20, 20, 20), -1)
        cv2.rectangle(img, (status_box_x, status_box_y), (status_box_x + 320, status_box_y + 35), (0, 255, 0), 2)
        cv2.putText(img, "HUMAN ARM LOCKED & ACTIVE", (status_box_x + 15, status_box_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        if data.is_singularity:
            cv2.putText(img, "[SINGULARITY: ARM STRAIGHT]", (e.pixel_u - 60, e.pixel_v + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    def close(self):
        if self.pose is not None:
            self.pose.close()
