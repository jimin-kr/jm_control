"""
Intel RealSense Depth Camera Pipeline & Wrapper module.
Handles RGB + Depth streaming, Depth-to-Color Alignment, Camera Intrinsics,
Robust Depth Patch Sampling, and Synthetic/Webcam Fallbacks.
"""

import sys
import time
import math
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    PYREALSENSE_AVAILABLE = False


class CameraIntrinsics:
    """
    Holds camera intrinsic parameters and projection utilities.
    """
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fx: float = 920.0,
        fy: float = 920.0,
        cx: float = 640.0,
        cy: float = 360.0,
        coeffs: Optional[np.ndarray] = None
    ):
        self.width = width
        self.height = height
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.coeffs = coeffs if coeffs is not None else np.zeros((5,), dtype=np.float64)

    @property
    def matrix(self) -> np.ndarray:
        """Returns the 3x3 camera matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def deproject_pixel_to_point(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """
        De-projects a 2D pixel (u, v) and depth in meters to a 3D point (X, Y, Z) in camera frame.
        Coordinate system: X right, Y down, Z forward.
        """
        if depth_m <= 0.0 or math.isnan(depth_m):
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return np.array([x, y, z], dtype=np.float64)

    def project_point_to_pixel(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Projects a 3D point (X, Y, Z) in camera frame to a 2D pixel (u, v).
        """
        x, y, z = point_3d[0], point_3d[1], point_3d[2]
        if z <= 0.0:
            return 0, 0
        u = int(round((x * self.fx / z) + self.cx))
        v = int(round((y * self.fy / z) + self.cy))
        return u, v


class RealSenseCamera:
    """
    RealSense RGB-D Camera Controller.
    """
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        enable_filters: bool = True,
        min_depth: float = 0.15,
        max_depth: float = 3.5
    ):
        if not PYREALSENSE_AVAILABLE:
            raise RuntimeError(
                "pyrealsense2 is not installed. Please install via 'pip install pyrealsense2' "
                "or run with mock mode."
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.enable_filters = enable_filters
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.rs_intrinsics = None
        self.depth_scale = 0.001  # Default 1mm per unit

        # Post-processing filters for depth stabilization
        if self.enable_filters:
            self.spatial_filter = rs.spatial_filter()
            self.temporal_filter = rs.temporal_filter()
            self.hole_filling_filter = rs.hole_filling_filter()

        self._is_running = False

    def start(self):
        """Starts the RealSense pipeline and depth alignment."""
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        profile = self.pipeline.start(self.config)

        # Depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Alignment object to align depth frames to color frames
        self.align = rs.align(rs.stream.color)

        # Extract color stream intrinsics
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.rs_intrinsics = color_stream.get_intrinsics()

        self.intrinsics = CameraIntrinsics(
            width=self.rs_intrinsics.width,
            height=self.rs_intrinsics.height,
            fx=self.rs_intrinsics.fx,
            fy=self.rs_intrinsics.fy,
            cx=self.rs_intrinsics.ppx,
            cy=self.rs_intrinsics.ppy,
            coeffs=np.array(self.rs_intrinsics.coeffs, dtype=np.float64)
        )

        self._is_running = True
        print(f"[RealSense] Pipeline started: {self.width}x{self.height} @ {self.fps}fps")
        print(f"[RealSense] Intrinsics: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
              f"cx={self.intrinsics.cx:.2f}, cy={self.intrinsics.cy:.2f}")

    def get_frames(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[Any]]:
        """
        Polls and aligns frames.
        Returns:
            success (bool), color_image (bgr), depth_image (meters or mm uint16), raw_depth_frame
        """
        if not self._is_running:
            return False, None, None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=50)
        except Exception:
            return False, None, None, None
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return False, None, None, None

        # Apply depth filters
        if self.enable_filters:
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.hole_filling_filter.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return True, color_image, depth_image, depth_frame

    def get_depth_at_pixel(self, depth_frame_or_img, u: int, v: int, patch_radius: int = 3) -> float:
        """
        Samples depth around (u, v) using a square neighborhood patch and calculates
        the median depth in meters, rejecting 0/NaN/out-of-range depths.
        """
        u = int(round(u))
        v = int(round(v))

        # Check if depth_frame has get_distance method
        if hasattr(depth_frame_or_img, "get_distance"):
            depth_func = depth_frame_or_img.get_distance
            depth_samples = []
            for du in range(-patch_radius, patch_radius + 1):
                for dv in range(-patch_radius, patch_radius + 1):
                    pu, pv = u + du, v + dv
                    if 0 <= pu < self.width and 0 <= pv < self.height:
                        d = depth_func(pu, pv)
                        if self.min_depth <= d <= self.max_depth:
                            depth_samples.append(d)
            if len(depth_samples) > 0:
                return float(np.median(depth_samples))
            return 0.0

        # Fallback if numpy depth image (uint16 in mm or float32 in m)
        if isinstance(depth_frame_or_img, np.ndarray):
            h, w = depth_frame_or_img.shape[:2]
            u_min = max(0, u - patch_radius)
            u_max = min(w, u + patch_radius + 1)
            v_min = max(0, v - patch_radius)
            v_max = min(h, v + patch_radius + 1)

            patch = depth_frame_or_img[v_min:v_max, u_min:u_max]
            if depth_frame_or_img.dtype == np.uint16:
                patch_m = patch.astype(np.float64) * self.depth_scale
            else:
                patch_m = patch.astype(np.float64)

            valid = patch_m[(patch_m >= self.min_depth) & (patch_m <= self.max_depth)]
            if len(valid) > 0:
                return float(np.median(valid))
            return 0.0

        return 0.0

    def deproject_pixel_to_point(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """
        De-projects pixel (u, v) and depth to 3D point (X, Y, Z) in meters.
        """
        if self.rs_intrinsics is not None and rs is not None:
            point = rs.rs2_deproject_pixel_to_point(self.rs_intrinsics, [float(u), float(v)], float(depth_m))
            return np.array(point, dtype=np.float64)
        elif self.intrinsics is not None:
            return self.intrinsics.deproject_pixel_to_point(u, v, depth_m)
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def stop(self):
        """Stops the RealSense camera pipeline."""
        if self._is_running:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self._is_running = False
            print("[RealSense] Pipeline stopped.")


class MockRealSenseCamera:
    """
    Mock RealSense camera for simulation and testing without physical hardware.
    Can use standard webcam, a synthetic moving arm, or video file.
    """
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        source: Optional[str] = None
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.source = source
        self.cap = None
        self.intrinsics = CameraIntrinsics(
            width=width, height=height,
            fx=width * 0.75, fy=width * 0.75,
            cx=width / 2.0, cy=height / 2.0
        )
        self.depth_scale = 0.001
        self._is_running = False
        self._start_time = time.time()

    def start(self):
        if self.source is not None:
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._is_running = True
        self._start_time = time.time()
        print(f"[MockRealSense] Simulation started: {self.width}x{self.height} @ {self.fps}fps")

    def get_frames(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[Any]]:
        if not self._is_running:
            return False, None, None, None

        if self.cap is not None and self.cap.isOpened():
            ret, color_image = self.cap.read()
            if not ret:
                return False, None, None, None
            color_image = cv2.resize(color_image, (self.width, self.height))
            # Synthetic depth map: flat plane at ~1.0m with radial gradient
            depth_image = np.ones((self.height, self.width), dtype=np.uint16) * 1000  # 1.0m
            return True, color_image, depth_image, depth_image

        # Generate synthetic test frame with ArUco marker and simulated user
        t = time.time() - self._start_time
        color_image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 40

        # Draw a synthetic torso/arm for demonstration
        shoulder_px = (int(self.width * 0.5), int(self.height * 0.35))
        elbow_px = (
            int(self.width * 0.5 + 160 * math.cos(t * 1.2)),
            int(self.height * 0.35 + 160 * math.sin(t * 1.2))
        )
        wrist_px = (
            int(elbow_px[0] + 140 * math.cos(t * 1.5 + 0.5)),
            int(elbow_px[1] + 140 * math.sin(t * 1.5 + 0.5))
        )

        # Draw lines
        cv2.line(color_image, shoulder_px, elbow_px, (0, 200, 255), 8)
        cv2.line(color_image, elbow_px, wrist_px, (0, 255, 100), 8)
        cv2.circle(color_image, shoulder_px, 12, (255, 0, 0), -1)
        cv2.circle(color_image, elbow_px, 12, (0, 255, 255), -1)
        cv2.circle(color_image, wrist_px, 12, (0, 255, 0), -1)

        # Render simulated ArUco marker on wrist
        try:
            dict_obj = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) if hasattr(cv2.aruco, "getPredefinedDictionary") else cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            if hasattr(cv2.aruco, "generateImageMarker"):
                m_img = cv2.aruco.generateImageMarker(dict_obj, 0, 80)
            else:
                m_img = np.zeros((80, 80), dtype=np.uint8)
                cv2.aruco.drawMarker(dict_obj, 0, 80, m_img)
            m_bgr = cv2.cvtColor(m_img, cv2.COLOR_GRAY2BGR)
            
            # Place marker near wrist
            my, mx = max(0, min(self.height - 80, wrist_px[1] - 40)), max(0, min(self.width - 80, wrist_px[0] - 40))
            color_image[my:my+80, mx:mx+80] = m_bgr
        except Exception:
            pass

        depth_image = np.ones((self.height, self.width), dtype=np.uint16) * 1100  # 1.1m
        time.sleep(1.0 / self.fps)
        return True, color_image, depth_image, depth_image

    def get_depth_at_pixel(self, depth_frame_or_img, u: int, v: int, patch_radius: int = 3) -> float:
        return 1.1  # 1.1 meters constant in mock mode

    def deproject_pixel_to_point(self, u: float, v: float, depth_m: float) -> np.ndarray:
        return self.intrinsics.deproject_pixel_to_point(u, v, depth_m)

    def stop(self):
        if self.cap is not None:
            self.cap.release()
        self._is_running = False
        print("[MockRealSense] Simulation stopped.")
