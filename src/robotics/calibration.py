"""
calibration.py — Hand-Eye Calibration Helper
=============================================
Computes the transformation from camera frame to robot frame
using a set of known correspondences (pixel → robot xyz).

Typical workflow:
  1. Place calibration target at N known robot positions.
  2. Record pixel coordinates of target centroid.
  3. Run solve() to get the affine transform matrix T.
  4. Save T for use in CoordinateMapper.

Reference: Tsai-Lenz hand-eye calibration (simplified homography approach).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("robotics.calibration")

_CALIB_DIR = Path(__file__).resolve().parents[3] / "data" / "calibration"


class HandEyeCalibrator:
    """
    Accumulates pixel ↔ robot-xyz correspondences and solves
    for a linear mapping using least-squares regression.

    Usage::

        cal = HandEyeCalibrator()
        cal.add_point(pixel=(320, 240), robot_xyz=(0.15, 0.20, 0.0))
        ...
        T = cal.solve()
        cal.save()
    """

    def __init__(self) -> None:
        self._pixel_pts: List[Tuple[int, int]]             = []
        self._robot_pts: List[Tuple[float, float, float]]  = []

    def add_point(
        self,
        pixel: Tuple[int, int],
        robot_xyz: Tuple[float, float, float],
    ) -> None:
        """Add a calibration correspondence."""
        self._pixel_pts.append(pixel)
        self._robot_pts.append(robot_xyz)
        log.info(f"Added point #{len(self._pixel_pts)}: pixel={pixel} → xyz={robot_xyz}")

    def solve(self) -> np.ndarray:
        """
        Solve for 3×3 affine transform T such that:
            [X, Y, Z]^T ≈ T · [u, v, 1]^T

        Returns:
            T (3×3 float64)

        Raises:
            ValueError if fewer than 4 correspondences.
        """
        n = len(self._pixel_pts)
        if n < 4:
            raise ValueError(f"Need at least 4 correspondences, got {n}.")

        # Build linear system A·x = b
        A = np.array([[u, v, 1] for u, v in self._pixel_pts], dtype=np.float64)
        B = np.array(self._robot_pts, dtype=np.float64)

        # Least-squares: T^T = pinv(A) @ B
        T_T, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        T = T_T.T      # shape (3, 3) — each row maps to X, Y, Z

        # Compute reprojection error
        B_pred = (A @ T_T)
        errors = np.linalg.norm(B - B_pred, axis=1)
        log.info(
            f"Calibration solved with {n} points | "
            f"mean error={errors.mean()*1000:.2f} mm | "
            f"max error={errors.max()*1000:.2f} mm"
        )
        return T

    def save(self, out_path: Path = _CALIB_DIR / "hand_eye.json") -> None:
        """Solve and persist calibration to JSON."""
        T = self.solve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "transform_3x3": T.tolist(),
            "n_points":       len(self._pixel_pts),
            "pixel_pts":      self._pixel_pts,
            "robot_pts":      self._robot_pts,
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        log.success(f"Hand-eye calibration saved to [cyan]{out_path}[/cyan] ✓")

    @staticmethod
    def load(path: Path = _CALIB_DIR / "hand_eye.json") -> np.ndarray:
        """Load a previously saved calibration matrix."""
        if not path.exists():
            raise FileNotFoundError(f"Calibration not found: {path}")

        with open(path) as f:
            data = json.load(f)

        T = np.array(data["transform_3x3"], dtype=np.float64)
        log.success(f"Hand-eye matrix loaded from [cyan]{path}[/cyan] ✓")
        return T

    @staticmethod
    def pixel_to_robot(
        pixel: Tuple[int, int],
        T: np.ndarray,
    ) -> np.ndarray:
        """
        Apply loaded transform to a new pixel coordinate.

        Args:
            pixel: (u, v) in image space.
            T:     3×3 calibration matrix from load().

        Returns:
            [X, Y, Z] in robot frame (metres).
        """
        uvh = np.array([pixel[0], pixel[1], 1.0])
        return T @ uvh


def save_camera_intrinsics(
    camera_matrix: np.ndarray,
    dist_coeffs:   np.ndarray,
    image_size:    Tuple[int, int],
    out_path:      Path = _CALIB_DIR / "camera_params.json",
) -> None:
    """
    Persist OpenCV camera calibration output.
    Called by scripts/calibrate_camera.py after cv2.calibrateCamera().
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs":   dist_coeffs.tolist(),
        "image_size":    list(image_size),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    log.success(f"Camera intrinsics saved to [cyan]{out_path}[/cyan] ✓")
