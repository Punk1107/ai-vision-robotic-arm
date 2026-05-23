"""
motion.py — Motion / Tracking Filters (Category 6)
===================================================
Filters for temporal analysis and motion estimation in video streams:

  6.1  KalmanTracker2D      — smooth 2D object position tracking
  6.2  OpticalFlowSmoother  — smooth dense/sparse optical flow vectors

These work in concert with the existing CentroidTracker in tracker.py.
While tracker.py implements the full multi-object tracking pipeline,
these classes provide lower-level building blocks and standalone utilities.

Mathematical basis
------------------
  Kalman 2D (position only):
    State: x = [px, py, vx, vy]ᵀ
    Measurement: z = [px, py]ᵀ
    F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]  (constant velocity)
    H = [[1,0,0,0],[0,1,0,0]]

  Optical Flow (Lucas-Kanade):
    Minimises Σ (I(x+u,y+v,t+1) − I(x,y,t))²  over (u,v) in window
    ≈ [Ix Iy; ...] · [u; v] = −[It; ...]   (brightness constancy)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("filters.motion")


# =============================================================================
# 6.1  Kalman 2D Tracker (standalone, single object)
# =============================================================================

class KalmanTracker2D:
    """
    Single-object 2D Kalman filter tracker.

    State vector: x = [px, py, vx, vy]
    Tracks position (px, py) and estimates velocity (vx, vy).

    This is the per-object Kalman filter used inside tracker.py's Track class
    (which also does 3D world-coordinate Kalman via _F_3D).

    This standalone version is useful when you only have one object to track
    (e.g., the currently selected pick target) and don't need the full
    multi-object management of CentroidTracker.

    Applications in this project:
        - Smooth the target centroid during visual servoing approach
        - Predict target position between YOLO frames (frame_skip)
        - Motion-based object validation (velocity check)

    Args:
        process_noise_pos:  Q for position states (lower = trust dynamics more).
        process_noise_vel:  Q for velocity states.
        measurement_noise:  R for position measurements.
        initial_position:   Starting [px, py] in pixels.
    """

    def __init__(
        self,
        process_noise_pos:  float = 0.03,
        process_noise_vel:  float = 0.10,
        measurement_noise:  float = 2.0,
        initial_position:   Optional[Tuple[float, float]] = None,
    ) -> None:
        # State transition matrix F (constant velocity model)
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix H (observe [px, py])
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise Q
        self._Q = np.diag([
            process_noise_pos, process_noise_pos,
            process_noise_vel, process_noise_vel,
        ]).astype(np.float32)

        # Measurement noise R
        self._R = np.eye(2, dtype=np.float32) * measurement_noise

        # State and covariance
        self._x = np.zeros(4, dtype=np.float32)
        self._P = np.eye(4, dtype=np.float32) * 100.0  # high initial uncertainty

        if initial_position is not None:
            self._x[0] = initial_position[0]
            self._x[1] = initial_position[1]

        self._I4 = np.eye(4, dtype=np.float32)
        log.debug(
            f"KalmanTracker2D | Q_pos={process_noise_pos} | "
            f"Q_vel={process_noise_vel} | R={measurement_noise}"
        )

    def predict(self) -> np.ndarray:
        """
        Prediction step — advance state by one time step.

        Returns:
            Predicted [px, py] position.
        """
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x[:2].copy()

    def update(self, measurement: Tuple[float, float]) -> np.ndarray:
        """
        Measurement update step.

        Args:
            measurement: (px, py) observation in pixels.

        Returns:
            Corrected [px, py] position estimate.
        """
        z = np.array(measurement, dtype=np.float32)
        S = self._H @ self._P @ self._H.T + self._R  # 2×2 innovation covariance

        # Closed-form 2×2 inverse (avoids LAPACK for embedded performance)
        det = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
        if abs(det) < 1e-9:
            det = 1e-9
        S_inv = np.array([
            [ S[1, 1], -S[0, 1]],
            [-S[1, 0],  S[0, 0]],
        ], dtype=np.float32) / det

        K       = self._P @ self._H.T @ S_inv
        innov   = z - self._H @ self._x
        self._x = self._x + K @ innov
        self._P = (self._I4 - K @ self._H) @ self._P
        return self._x[:2].copy()

    def step(self, measurement: Tuple[float, float]) -> np.ndarray:
        """Convenience: predict + update in one call."""
        self.predict()
        return self.update(measurement)

    @property
    def position(self) -> Tuple[float, float]:
        """Current estimated position (px, py)."""
        return float(self._x[0]), float(self._x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current estimated velocity (vx, vy) in pixels/frame."""
        return float(self._x[2]), float(self._x[3])

    def reset(self, position: Optional[Tuple[float, float]] = None) -> None:
        self._x = np.zeros(4, dtype=np.float32)
        self._P = np.eye(4, dtype=np.float32) * 100.0
        if position is not None:
            self._x[0] = position[0]
            self._x[1] = position[1]


# =============================================================================
# 6.2  Optical Flow Smoother
# =============================================================================

class OpticalFlowSmoother:
    """
    Optical Flow Estimator with temporal smoothing.

    Supports two backends:
        "sparse" — Lucas-Kanade (LK) sparse tracking of keypoints
        "dense"  — Farneback dense flow field (all pixels)

    Optical Flow basics:
        Given two frames I_t and I_{t+1}, optical flow (u, v) satisfies
        the brightness constancy assumption:
            I(x, y, t) = I(x+u, y+v, t+1)

        Which leads to the optical flow equation:
            Ix·u + Iy·v + It = 0
        Solved via LK (local window) or variational methods (Farneback).

    Temporal smoothing:
        Raw flow vectors are noisy frame-to-frame.  An EMA filter
        (alpha=0.7 default) smooths the flow while preserving responsiveness.

    Applications in this project:
        - Autonomous arm navigation (avoid moving obstacles)
        - Motion estimation for scene analysis
        - Conveyor belt velocity measurement
        - Camera ego-motion compensation

    Args:
        mode:       "sparse" (LK) or "dense" (Farneback).
        alpha:      EMA smoothing coefficient for temporal flow averaging.
                    0.5 = heavy smoothing, 0.9 = fast response, 0.7 = default.
        max_corners: (Sparse) Max ShiTomasi corners to track.
        quality:    (Sparse) Corner quality threshold.
        min_dist:   (Sparse) Min distance between corners.
    """

    def __init__(
        self,
        mode:        str   = "sparse",
        alpha:       float = 0.7,
        max_corners: int   = 100,
        quality:     float = 0.3,
        min_dist:    float = 7.0,
    ) -> None:
        if mode not in ("sparse", "dense"):
            raise ValueError("mode must be 'sparse' or 'dense'")
        self._mode     = mode
        self._alpha    = alpha
        self._max_cor  = max_corners
        self._quality  = quality
        self._min_dist = min_dist

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts:  Optional[np.ndarray] = None

        # Smoothed flow: sparse → (N, 2) array, dense → (H, W, 2) array
        self._smooth_flow: Optional[np.ndarray] = None

        # LK params
        self._lk_params = dict(
            winSize   = (15, 15),
            maxLevel  = 3,
            criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        log.info(f"OpticalFlowSmoother | mode={mode} | α={alpha:.2f}")

    # ── Sparse (Lucas-Kanade) ─────────────────────────────────────────────────

    def update_sparse(
        self,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute sparse optical flow using Lucas-Kanade.

        Returns:
            (N, 2) array of smoothed flow vectors (one per tracked point),
            or None if this is the first frame.
        """
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) == 0:
            # First frame or no corners — detect features
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners   = self._max_cor,
                qualityLevel = self._quality,
                minDistance  = self._min_dist,
            )
            self._prev_gray = gray
            return None

        # Compute LK flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray,
            self._prev_pts, None,
            **self._lk_params,
        )

        if next_pts is None or status is None:
            self._prev_gray = gray
            self._prev_pts  = None
            return None

        # Keep only successfully tracked points
        good_prev = self._prev_pts[status.ravel() == 1]
        good_next = next_pts[status.ravel() == 1]

        flow = (good_next - good_prev).reshape(-1, 2)  # displacement vectors

        # EMA temporal smoothing
        if self._smooth_flow is None or self._smooth_flow.shape != flow.shape:
            self._smooth_flow = flow
        else:
            self._smooth_flow = self._alpha * flow + (1 - self._alpha) * self._smooth_flow

        # Update state: re-detect corners periodically
        self._prev_gray = gray
        self._prev_pts  = good_next.reshape(-1, 1, 2)

        # If too few corners remain, re-detect
        if len(self._prev_pts) < self._max_cor // 4:
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self._max_cor,
                qualityLevel=self._quality, minDistance=self._min_dist,
            )

        return self._smooth_flow

    # ── Dense (Farneback) ─────────────────────────────────────────────────────

    def update_dense(
        self,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute dense optical flow using Gunnar Farneback's algorithm.

        Returns:
            (H, W, 2) flow field [flow_x, flow_y] per pixel, or None on first frame.
        """
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )

        # EMA temporal smoothing (entire flow field)
        if self._smooth_flow is None or self._smooth_flow.shape != flow.shape:
            self._smooth_flow = flow
        else:
            self._smooth_flow = self._alpha * flow + (1 - self._alpha) * self._smooth_flow

        self._prev_gray = gray
        return self._smooth_flow

    # ── Unified entry point ───────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Route to the configured flow backend."""
        if self._mode == "sparse":
            return self.update_sparse(frame)
        return self.update_dense(frame)

    def mean_magnitude(self) -> float:
        """
        Mean optical flow magnitude across all tracked points/pixels.

        Returns:
            Mean displacement in pixels/frame.  0.0 if no flow available.
        """
        if self._smooth_flow is None:
            return 0.0
        if self._mode == "sparse":
            return float(np.linalg.norm(self._smooth_flow, axis=1).mean())
        # Dense: compute magnitude map, take mean
        mag = np.linalg.norm(self._smooth_flow, axis=2)
        return float(mag.mean())

    def visualise(self, frame: np.ndarray) -> np.ndarray:
        """
        Render the optical flow as a colour-coded HSV image.

        Hue = direction, Saturation = 255, Value = magnitude (normalised).
        """
        if self._smooth_flow is None or self._mode == "sparse":
            return frame

        flow = self._smooth_flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2   # Hue = angle
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def reset(self) -> None:
        """Reset flow state (call when scene changes significantly)."""
        self._prev_gray   = None
        self._prev_pts    = None
        self._smooth_flow = None
