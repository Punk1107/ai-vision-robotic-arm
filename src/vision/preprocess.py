"""
preprocess.py — Image Preprocessing Pipeline
==============================================
Camera calibration correction and colour-space helpers
used upstream of YOLO inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("vision.preprocess")

# Default calibration directory
_CALIB_DIR = Path(__file__).resolve().parents[3] / "data" / "calibration"


class CameraPreprocessor:
    """
    Applies lens undistortion, resizing, and optional histogram
    equalisation to raw camera frames.

    If a calibration file exists the lens distortion is corrected;
    otherwise frames are passed through unchanged with a warning.
    """

    def __init__(
        self,
        target_width: int  = 640,
        target_height: int = 480,
        equalize_hist: bool = False,
        calib_path: Optional[Path] = None,
    ) -> None:
        self.target_size   = (target_width, target_height)
        self.equalize_hist = equalize_hist

        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs:   Optional[np.ndarray] = None
        self._map1:          Optional[np.ndarray] = None
        self._map2:          Optional[np.ndarray] = None

        self._load_calibration(calib_path or (_CALIB_DIR / "camera_params.json"))

    # ── Calibration ───────────────────────────────────────────────────────────
    def _load_calibration(self, path: Path) -> None:
        if not path.exists():
            log.warning(
                f"Calibration file not found at [yellow]{path}[/yellow]. "
                "Running without lens correction — run scripts/calibrate_camera.py first."
            )
            return

        with open(path) as f:
            data = json.load(f)

        self._camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        self._dist_coeffs   = np.array(data["dist_coeffs"],   dtype=np.float64)

        # Pre-compute undistortion maps for speed
        w, h = self.target_size
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._camera_matrix, self._dist_coeffs, None,
            self._camera_matrix, (w, h), cv2.CV_16SC2,
        )
        log.success(f"Camera calibration loaded from [cyan]{path}[/cyan] ✓")

    @property
    def camera_matrix(self) -> Optional[np.ndarray]:
        return self._camera_matrix

    # ── Main pipeline ─────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize → undistort → (optional) CLAHE.

        Args:
            frame: Raw BGR frame from cv2.VideoCapture.

        Returns:
            Processed BGR frame.
        """
        # 1. Resize
        out = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 2. Undistort
        if self._map1 is not None:
            out = cv2.remap(out, self._map1, self._map2, cv2.INTER_LINEAR)

        # 3. Adaptive histogram equalisation (CLAHE) on luminance
        if self.equalize_hist:
            lab   = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return out

    # ── Pixel → normalised coords ─────────────────────────────────────────────
    def pixel_to_normalised(self, px: int, py: int) -> tuple[float, float]:
        """
        Convert pixel coordinates to normalised camera coordinates
        using the loaded intrinsic matrix.

        Returns (u_norm, v_norm) or raises RuntimeError if not calibrated.
        """
        if self._camera_matrix is None:
            raise RuntimeError("Camera not calibrated. Load calibration first.")

        fx = self._camera_matrix[0, 0]
        fy = self._camera_matrix[1, 1]
        cx = self._camera_matrix[0, 2]
        cy = self._camera_matrix[1, 2]

        return (px - cx) / fx, (py - cy) / fy
