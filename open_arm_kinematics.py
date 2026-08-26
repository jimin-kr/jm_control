"""
OpenArm 7-DoF Robot Manipulator Analytical Kinematics & Inverse Kinematics (IK) Solver.
Structure: Anthropomorphic 7-DoF Arm (Spherical Shoulder 3-DoF + Elbow 1-DoF + Spherical Wrist 3-DoF)
Joint Mapping:
  - J1 (q1): Shoulder Yaw / Base Pan
  - J2 (q2): Shoulder Pitch / Elevation
  - J3 (q3): Shoulder Roll / Humeral Rotation (Controlled by Swivel Angle psi)
  - J4 (q4): Elbow Pitch / Flexion-Extension
  - J5 (q5): Wrist Roll / Forearm Pronation-Supination
  - J6 (q6): Wrist Pitch / Flexion-Extension
  - J7 (q7): Wrist Yaw / Radial-Ulnar Deviation
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

from aruco_tracker import (
    rotation_matrix_to_euler_rpy,
    rotation_matrix_to_quaternion,
    euler_rpy_to_rotation_matrix
)


@dataclass
class JointLimits:
    """Joint limits in degrees."""
    min_deg: float
    max_deg: float

    def clamp(self, val_deg: float) -> float:
        return float(np.clip(val_deg, self.min_deg, self.max_deg))


DEFAULT_JOINT_LIMITS = [
    JointLimits(-160.0, 160.0), # J1: Shoulder Yaw
    JointLimits(-110.0, 110.0), # J2: Shoulder Pitch
    JointLimits(-170.0, 170.0), # J3: Shoulder Roll
    JointLimits(0.0, 150.0),    # J4: Elbow Pitch (0 = straight, 150 = fully bent)
    JointLimits(-170.0, 170.0), # J5: Wrist Roll
    JointLimits(-90.0, 90.0),   # J6: Wrist Pitch
    JointLimits(-90.0, 90.0),   # J7: Wrist Yaw
]


@dataclass
class OpenArm7DoFState:
    """Holds 7 joint angles in degrees and radians with joint velocities and limits."""
    joint_angles_deg: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    joint_angles_rad: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    is_valid: bool = True
    is_singularity: bool = False
    swivel_angle_deg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_angles_deg": [round(float(q), 2) for q in self.joint_angles_deg],
            "joint_angles_rad": [round(float(q), 4) for q in self.joint_angles_rad],
            "swivel_angle_deg": round(float(self.swivel_angle_deg), 2),
            "is_valid": self.is_valid,
            "is_singularity": self.is_singularity
        }


class OpenArm7DoFSolver:
    """
    Closed-Form Analytical Inverse Kinematics Solver for 7-DoF Anthropomorphic Robot Arm.
    Directly computes (q1, q2, q3, q4, q5, q6, q7) from Human Tracking data:
      - Shoulder position P_s (3D)
      - Elbow position P_e (3D)
      - Wrist position P_w (3D)
      - Wrist 6-DoF Orientation Matrix R_target (3x3)
      - Swivel Angle psi (deg)
    """
    def __init__(
        self,
        upper_arm_len_m: float = 0.28,
        forearm_len_m: float = 0.25,
        joint_limits: Optional[List[JointLimits]] = None
    ):
        self.upper_arm_len = upper_arm_len_m
        self.forearm_len = forearm_len_m
        self.joint_limits = joint_limits or DEFAULT_JOINT_LIMITS
        self.last_valid_q_deg = np.zeros(7, dtype=np.float64)

    def solve_from_joints_and_orientation(
        self,
        p_shoulder: np.ndarray,
        p_elbow: np.ndarray,
        p_wrist: np.ndarray,
        r_target_wrist: np.ndarray,
        swivel_angle_deg: float,
        arm_side: str = "right"
    ) -> OpenArm7DoFState:
        """
        Solves analytical IK for all 7 joints.
        """
        p_s = np.asarray(p_shoulder, dtype=np.float64)
        p_e = np.asarray(p_elbow, dtype=np.float64)
        p_w = np.asarray(p_wrist, dtype=np.float64)

        v_se = p_e - p_s
        len_se = np.linalg.norm(v_se)
        v_ew = p_w - p_e
        len_ew = np.linalg.norm(v_ew)

        state = OpenArm7DoFState(swivel_angle_deg=swivel_angle_deg)

        if len_se < 1e-3 or len_ew < 1e-3:
            state.is_valid = False
            state.joint_angles_deg = self.last_valid_q_deg.copy()
            state.joint_angles_rad = np.radians(self.last_valid_q_deg)
            return state

        u_se = v_se / len_se
        u_ew = v_ew / len_ew

        # -------------------------------------------------------------
        # 1. J4: Elbow Pitch Angle (Flexion)
        # -------------------------------------------------------------
        cos_elbow = np.dot(u_se, u_ew)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        # q4 = 0 when straight, positive when bent
        q4_rad = math.acos(cos_elbow)
        q4_deg = math.degrees(q4_rad)

        is_singularity = (q4_deg < 5.0) or (q4_deg > 165.0)
        state.is_singularity = is_singularity

        # -------------------------------------------------------------
        # 2. J1, J2, J3: Shoulder Spherical Joint (Yaw, Pitch, Roll)
        # -------------------------------------------------------------
        # In Camera Coordinate Frame: X=right, Y=down, Z=forward
        # When facing camera:
        # q1 (Shoulder Yaw): azimuth angle of upper arm
        q1_rad = math.atan2(u_se[0], u_se[2])
        # q2 (Shoulder Pitch): elevation angle (Y axis is downward)
        q2_rad = math.asin(np.clip(-u_se[1], -1.0, 1.0))
        # q3 (Shoulder Roll): controlled directly by the Swivel Angle psi!
        swivel_wrapped = (swivel_angle_deg + 180.0) % 360.0 - 180.0
        q3_deg_raw = swivel_wrapped
        if arm_side == "left":
            q3_deg_raw = -q3_deg_raw

        q1_deg = math.degrees(q1_rad)
        q2_deg = math.degrees(q2_rad)
        q3_deg = q3_deg_raw
        q3_rad = math.radians(q3_deg)

        # -------------------------------------------------------------
        # 3. Shoulder-Elbow Base Rotation Matrix R_se
        # -------------------------------------------------------------
        # Compute forward orientation of upper arm & forearm
        R_shoulder = euler_rpy_to_rotation_matrix(q3_rad, q2_rad, q1_rad)
        # Elbow pitch rotation around local pitch axis
        R_elbow_pitch = np.array([
            [ math.cos(q4_rad), 0, math.sin(q4_rad)],
            [ 0,                1, 0               ],
            [-math.sin(q4_rad), 0, math.cos(q4_rad)]
        ], dtype=np.float64)

        R_arm_base = R_shoulder @ R_elbow_pitch

        # -------------------------------------------------------------
        # 4. J5, J6, J7: Wrist Spherical Joint (Roll, Pitch, Yaw)
        # -------------------------------------------------------------
        # R_target = R_arm_base @ R_wrist -> R_wrist = R_arm_base.T @ R_target
        R_wrist = R_arm_base.T @ r_target_wrist

        # Decompose R_wrist into Roll-Pitch-Yaw Euler angles (Z-Y-X convention)
        r_wrist_rad, r_wrist_deg = rotation_matrix_to_euler_rpy(R_wrist)

        q5_deg = r_wrist_deg[0] # Wrist Roll
        q6_deg = r_wrist_deg[1] # Wrist Pitch
        q7_deg = r_wrist_deg[2] # Wrist Yaw

        # -------------------------------------------------------------
        # 5. Joint Limits Clamping & Smoothing
        # -------------------------------------------------------------
        raw_q_deg = np.array([q1_deg, q2_deg, q3_deg, q4_deg, q5_deg, q6_deg, q7_deg], dtype=np.float64)
        clamped_q_deg = np.zeros(7, dtype=np.float64)

        for i in range(7):
            clamped_q_deg[i] = self.joint_limits[i].clamp(raw_q_deg[i])

        self.last_valid_q_deg = clamped_q_deg.copy()
        state.joint_angles_deg = clamped_q_deg
        state.joint_angles_rad = np.radians(clamped_q_deg)
        state.is_valid = True

        return state
