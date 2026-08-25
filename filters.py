"""
Signal filtering and smoothing module for Vision & Pose tracking.
Includes:
- Exponential Moving Average (EMA)
- One-Euro Filter (Adaptive low-pass filter for position and angles)
- Quaternion SLERP Smoothing
- Angle Continuity / Phase Unwrap Filter
"""

import math
import time
import numpy as np
from typing import Optional, Tuple


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) filter for 1D, 2D, or 3D numpy arrays.
    S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
    """
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Smoothing factor between 0.0 (max smoothing/delay) and 1.0 (no smoothing).
        """
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.state: Optional[np.ndarray] = None

    def update(self, measurement: np.ndarray) -> np.ndarray:
        if measurement is None:
            return self.state
        
        measurement = np.asarray(measurement, dtype=np.float64)
        if self.state is None:
            self.state = measurement.copy()
        else:
            self.state = self.alpha * measurement + (1.0 - self.alpha) * self.state
        return self.state.copy()

    def reset(self):
        self.state = None


class LowPassFilter:
    """
    First-order low-pass filter used in One-Euro Filter.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.hat_x_prev: Optional[np.ndarray] = None

    def __call__(self, x: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = alpha
        
        x = np.asarray(x, dtype=np.float64)
        if self.hat_x_prev is None:
            hat_x = x.copy()
        else:
            hat_x = self.alpha * x + (1.0 - self.alpha) * self.hat_x_prev
        self.hat_x_prev = hat_x.copy()
        return hat_x

    def reset(self):
        self.hat_x_prev = None


class OneEuroFilter:
    """
    One-Euro Filter: An adaptive low-pass filter designed for noisy human motion tracking.
    Reference: Casiez et al., "1 € Filter: A Simple Speed-based Low-pass Filter for Noisy Input in HCI" (CHI 2012)
    
    Features:
    - High smoothing at low speed (eliminates jitter)
    - Low smoothing (low latency) at high speed (eliminates lag)
    """
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        """
        Args:
            min_cutoff: Minimum cutoff frequency in Hz (decrease to reduce slow jitter).
            beta: Speed coefficient (increase to reduce lag during fast movements).
            d_cutoff: Cutoff frequency for derivative calculation in Hz.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.t_prev: Optional[float] = None
        self.x_prev: Optional[np.ndarray] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Filter a scalar or numpy vector measurement.
        """
        if x is None:
            return self.x_prev
            
        x = np.asarray(x, dtype=np.float64)
        if timestamp is None:
            timestamp = time.time()

        if self.t_prev is None:
            self.t_prev = timestamp
            self.x_prev = x.copy()
            self.dx_filter.hat_x_prev = np.zeros_like(x)
            return self.x_filter(x, alpha=1.0)

        dt = timestamp - self.t_prev
        if dt <= 1e-6:
            dt = 1e-3  # Avoid division by zero

        # Estimate derivative
        dx = (x - self.x_prev) / dt
        edx = self.dx_filter(dx, self._alpha(self.d_cutoff, dt))

        # Use derivative magnitude to compute dynamic cutoff frequency
        speed = np.linalg.norm(edx) if edx.ndim > 0 else abs(edx)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter the signal
        hat_x = self.x_filter(x, self._alpha(cutoff, dt))

        self.t_prev = timestamp
        self.x_prev = hat_x.copy()
        return hat_x

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.t_prev = None
        self.x_prev = None


class QuaternionFilter:
    """
    Quaternion filter using Spherical Linear Interpolation (SLERP)
    and sign alignment to avoid shortest path discontinuity (q and -q represent same rotation).
    Quaternion format: [w, x, y, z]
    """
    def __init__(self, alpha: float = 0.3):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.state: Optional[np.ndarray] = None

    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / norm

    @staticmethod
    def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between two unit quaternions q0 and q1.
        """
        q0 = QuaternionFilter.normalize(q0)
        q1 = QuaternionFilter.normalize(q1)

        dot = np.dot(q0, q1)

        # Ensure shortest path
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        # If quaternions are very close, use linear interpolation
        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            result = q0 + t * (q1 - q0)
            return QuaternionFilter.normalize(result)

        # SLERP formula
        theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = math.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = math.sin(theta_t)

        s0 = math.cos(theta_t) - dot * sin_theta_t / sin_theta_0
        s1 = sin_theta_t / sin_theta_0

        return QuaternionFilter.normalize(s0 * q0 + s1 * q1)

    def update(self, q: np.ndarray) -> np.ndarray:
        """
        q: [w, x, y, z]
        """
        if q is None:
            return self.state
            
        q = np.asarray(q, dtype=np.float64)
        q = self.normalize(q)

        if self.state is None:
            self.state = q.copy()
        else:
            self.state = self.slerp(self.state, q, self.alpha)
        return self.state.copy()

    def reset(self):
        self.state = None


class AngleContinuityFilter:
    """
    Prevents -pi to +pi (or -180 to +180) wrap-around jumps in angle measurements (e.g. Swivel Angle).
    Maintains a continuous unwrapped angle and filters it with EMA or One-Euro.
    """
    def __init__(self, in_degrees: bool = True, smoothing_alpha: float = 0.4):
        self.in_degrees = in_degrees
        self.mod = 360.0 if in_degrees else (2.0 * math.pi)
        self.half_mod = self.mod / 2.0
        self.last_raw_angle: Optional[float] = None
        self.unwrapped_angle: Optional[float] = None
        self.ema = ExponentialMovingAverage(alpha=smoothing_alpha)

    def update(self, angle: float) -> float:
        if angle is None or math.isnan(angle):
            return self.unwrapped_angle

        if self.last_raw_angle is None:
            self.last_raw_angle = angle
            self.unwrapped_angle = angle
            self.ema.update(np.array([angle]))
            return angle

        # Calculate smallest delta taking into account wrap-around
        delta = angle - self.last_raw_angle
        while delta > self.half_mod:
            delta -= self.mod
        while delta < -self.half_mod:
            delta += self.mod

        self.unwrapped_angle += delta
        self.last_raw_angle = angle

        smoothed = self.ema.update(np.array([self.unwrapped_angle]))[0]
        return float(smoothed)

    def reset(self):
        self.last_raw_angle = None
        self.unwrapped_angle = None
        self.ema.reset()
