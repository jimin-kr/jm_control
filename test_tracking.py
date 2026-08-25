"""
Comprehensive test suite for OpenArm Tracking Pipeline:
- Filters (EMA, One-Euro, Quaternion SLERP, Angle Continuity)
- ArUco 6-DoF Tracker & Rotation Conversions
- MediaPipe Joint 3D De-projection & 7-DoF Swivel Angle
- Anti-Occlusion Kinematic Fusion Engine (Dead-Reckoning & Ray Solving)
- Full Pipeline Integration
"""

import unittest
import math
import numpy as np
import cv2

from filters import OneEuroFilter, ExponentialMovingAverage, QuaternionFilter, AngleContinuityFilter
from realsense_camera import CameraIntrinsics
from aruco_tracker import ArUcoTracker, rotation_matrix_to_euler_rpy, rotation_matrix_to_quaternion
from pose_tracker import compute_swivel_angle, ArmKinematicFusionEngine
from arm_tracking import ArmTrackingPipeline


class TestFilters(unittest.TestCase):
    def test_ema_filter(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        v1 = ema.update(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(v1, [1.0, 2.0, 3.0])
        v2 = ema.update(np.array([2.0, 4.0, 6.0]))
        np.testing.assert_allclose(v2, [1.5, 3.0, 4.5])

    def test_one_euro_filter(self):
        filter_3d = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        p0 = np.array([0.0, 0.0, 1.0])
        f0 = filter_3d.filter(p0, timestamp=0.0)
        np.testing.assert_allclose(f0, p0)
        p1 = np.array([0.01, 0.0, 1.0])
        f1 = filter_3d.filter(p1, timestamp=0.033)
        self.assertTrue(0.0 <= f1[0] <= 0.01)

    def test_quaternion_slerp(self):
        q_ident = np.array([1.0, 0.0, 0.0, 0.0])
        q_z90 = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
        q_half = QuaternionFilter.slerp(q_ident, q_z90, 0.5)
        q_expected = np.array([math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)])
        np.testing.assert_allclose(q_half, q_expected, atol=1e-4)

    def test_angle_continuity_filter(self):
        ang_filter = AngleContinuityFilter(in_degrees=True, smoothing_alpha=1.0)
        a1 = ang_filter.update(175.0)
        self.assertAlmostEqual(a1, 175.0, places=3)
        a2 = ang_filter.update(-175.0)
        self.assertAlmostEqual(a2, 185.0, places=3)


class TestArucoTracker(unittest.TestCase):
    def test_rotation_conversions(self):
        R_ident = np.eye(3)
        rad, deg = rotation_matrix_to_euler_rpy(R_ident)
        np.testing.assert_allclose(deg, [0.0, 0.0, 0.0], atol=1e-4)
        q_ident = rotation_matrix_to_quaternion(R_ident)
        np.testing.assert_allclose(q_ident, [1.0, 0.0, 0.0, 0.0], atol=1e-4)

    def test_marker_detection_on_image(self):
        img = cv2.imread('marker_0.png')
        self.assertIsNotNone(img, "marker_0.png must exist for test")
        intrinsics = CameraIntrinsics(width=img.shape[1], height=img.shape[0], fx=600, fy=600, cx=img.shape[1]/2, cy=img.shape[0]/2)
        tracker = ArUcoTracker(target_marker_id=0, marker_size_m=0.05)
        pose, _ = tracker.detect(img, intrinsics)
        self.assertTrue(pose.detected)
        self.assertEqual(pose.marker_id, 0)
        self.assertGreater(pose.position[2], 0.0)


class TestPoseRedundancy(unittest.TestCase):
    def test_swivel_angle_horizontal(self):
        shoulder = np.array([0.0, 0.0, 1.0])
        elbow = np.array([0.25, 0.0, 1.0])
        wrist = np.array([0.25, 0.25, 1.0])
        rad, deg, el_deg, sing = compute_swivel_angle(shoulder, elbow, wrist)
        self.assertFalse(sing)
        self.assertAlmostEqual(el_deg, 90.0, delta=1.0)

    def test_singularity_detection(self):
        shoulder = np.array([0.0, 0.0, 1.0])
        elbow = np.array([0.25, 0.0, 1.0])
        wrist = np.array([0.50, 0.0, 1.0])
        rad, deg, el_deg, sing = compute_swivel_angle(shoulder, elbow, wrist)
        self.assertTrue(sing)
        self.assertAlmostEqual(el_deg, 180.0, delta=1.0)


class TestKinematicFusion(unittest.TestCase):
    def test_occlusion_wrist_recovery(self):
        fusion = ArmKinematicFusionEngine()
        fusion.upper_arm_len = 0.30
        fusion.forearm_len = 0.25
        p_shoulder = np.array([0.0, -0.1, 1.2])
        p_elbow = np.array([0.2, 0.0, 1.1])
        p_wrist_true = np.array([0.35, -0.05, 0.95])
        ray_w = p_wrist_true / p_wrist_true[2]
        p_wrist_est = fusion.solve_wrist_on_ray(ray_w, p_elbow)
        err = np.linalg.norm(p_wrist_est - p_wrist_true)
        self.assertLess(err, 0.1, f"Wrist recovery error too high: {err}")

    def test_occlusion_elbow_recovery(self):
        fusion = ArmKinematicFusionEngine()
        fusion.upper_arm_len = 0.30
        fusion.forearm_len = 0.25
        p_shoulder = np.array([0.0, -0.1, 1.2])
        p_elbow_true = np.array([0.2, 0.0, 1.1])
        p_wrist = np.array([0.35, -0.05, 0.95])
        ray_e = p_elbow_true / p_elbow_true[2]
        p_elbow_est = fusion.solve_elbow_on_ray(ray_e, p_shoulder, p_wrist)
        err = np.linalg.norm(p_elbow_est - p_elbow_true)
        self.assertLess(err, 0.05, f"Elbow recovery error too high: {err}")


class TestFullPipeline(unittest.TestCase):
    def test_pipeline_mock_run(self):
        pipeline = ArmTrackingPipeline(
            camera_width=640,
            camera_height=480,
            fps=30,
            marker_id=0,
            use_mock=True,
            enable_filtering=True
        )
        pipeline.start()
        success, result, vis_img = pipeline.run_frame()
        pipeline.close()

        self.assertTrue(success)
        self.assertIsNotNone(result)
        self.assertIsNotNone(vis_img)
        self.assertEqual(result.arm_side, "right")
        self.assertIn("wrist_6dof", result.to_dict())
        self.assertIn("redundancy", result.to_dict())
        self.assertIn("open_arm_7dof", result.to_dict())


class TestOpenArmKinematics(unittest.TestCase):
    def test_7dof_ik_solution(self):
        from open_arm_kinematics import OpenArm7DoFSolver
        solver = OpenArm7DoFSolver()
        p_shoulder = np.array([0.0, -0.15, 1.2])
        p_elbow = np.array([0.15, 0.05, 1.15])
        p_wrist = np.array([0.25, -0.05, 1.0])
        R_target = np.eye(3)
        swivel = 30.0

        state = solver.solve_from_joints_and_orientation(
            p_shoulder, p_elbow, p_wrist, R_target, swivel_angle_deg=swivel, arm_side="right"
        )
        self.assertTrue(state.is_valid)
        self.assertEqual(len(state.joint_angles_deg), 7)
        self.assertAlmostEqual(state.swivel_angle_deg, 30.0, places=1)
        self.assertGreater(state.joint_angles_deg[3], 0.0)


if __name__ == "__main__":
    unittest.main()
