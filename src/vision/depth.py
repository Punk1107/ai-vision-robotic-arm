"""
depth.py — Depth Estimation & 3-D Coordinate Mapping
======================================================
Provides two backends:
  1. MonocularDepthEstimator  — MiDaS (no extra hardware)
  2. CoordinateMapper         — maps pixel → real-world XYZ
                                using either depth map or known plane

The CoordinateMapper is the critical bridge between
pixel detections and robot workspace coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Literal

import cv2
import numpy as np
import torch

from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("vision.depth")

_CALIB_DIR = Path(__file__).resolve().parents[3] / "data" / "calibration"


# ─────────────────────────────────────────────────────────────────────────────
# Monocular Depth Estimator (MiDaS)
# ─────────────────────────────────────────────────────────────────────────────

class MonocularDepthEstimator:
    """
    Estimates relative depth using Intel's MiDaS model via torch.hub.

    Output is a disparity map (larger value = closer to camera).
    Use CoordinateMapper.disparity_to_depth() to get metric depth.
    """

    def __init__(self, model_name: str = "MiDaS_small") -> None:
        log.info(f"Loading MiDaS model: [cyan]{model_name}[/cyan] ...")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
        self._model.to(self._device).eval()

        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self._transform = (
            transforms.small_transform
            if model_name == "MiDaS_small"
            else transforms.dpt_transform
        )
        log.success(f"MiDaS ready on [{self._device}] ✓")

    @torch.inference_mode()
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a disparity map (float32, same H×W as input).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb).to(self._device)

        prediction = self._model(inp)
        disp = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return disp.astype(np.float32)

    def colourize(self, disp: np.ndarray) -> np.ndarray:
        """Return a colour-mapped disparity image for visualisation."""
        norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate Mapper  (pixel → real-world XYZ)
# ─────────────────────────────────────────────────────────────────────────────

class CoordinateMapper:
    """
    Maps image-plane pixel coordinates to robot-frame XYZ (metres).

    Two strategies:
        "plane"  — Object sits on a flat table; z is known/fixed.
                   Uses the camera intrinsic matrix + known z-height.
        "depth"  — Uses a depth/disparity map (MiDaS or RealSense).

    The output coordinate frame:
        X   = right  (+)
        Y   = forward (+, away from camera)
        Z   = up     (+)

    Run scripts/calibrate_camera.py first to generate camera_params.json.
    """

    def __init__(
        self,
        strategy: Literal["plane", "depth"] = "plane",
        table_z:  float = 0.0,
    ) -> None:
        self.strategy = strategy
        self.table_z  = table_z

        self._K: Optional[np.ndarray] = None   # camera matrix 3×3
        self._alpha = config.depth.alpha
        self._beta  = config.depth.beta

        self._load_calibration()

    # ── Calibration ───────────────────────────────────────────────────────────
    def _load_calibration(self) -> None:
        calib_file = _CALIB_DIR / "camera_params.json"
        if calib_file.exists():
            with open(calib_file) as f:
                data = json.load(f)
            self._K = np.array(data["camera_matrix"], dtype=np.float64)
            log.success(f"CoordinateMapper: camera matrix loaded ✓")
        else:
            log.warning(
                "camera_params.json not found. "
                "Using estimated focal length — accuracy will be poor."
            )
            # Rough fallback for 1280×720 webcam
            fx = fy = 800.0
            self._K = np.array(
                [[fx, 0, 640], [0, fy, 360], [0, 0, 1]], dtype=np.float64
            )

    # ── Pixel → XYZ (plane strategy) ─────────────────────────────────────────
    def pixel_to_world_plane(
        self,
        px: int,
        py: int,
        z_world: Optional[float] = None,
    ) -> np.ndarray:
        """
        Back-project a pixel onto a known horizontal plane.

        Args:
            px, py:  Pixel coords (image frame).
            z_world: Height of the object plane in robot frame (metres).
                     Uses self.table_z if None.

        Returns:
            [X, Y, Z] in robot-frame metres.
        """
        z = z_world if z_world is not None else self.table_z

        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]

        # Camera-frame ray direction (normalised)
        x_cam = (px - cx) / fx
        y_cam = (py - cy) / fy

        # Camera is mounted looking downward at angle (simplified: straight down)
        # Assumes camera center is at height `camera_height` above table
        camera_height = config.robotics.workspace_z[1] + 0.15   # ~15 cm above max arm z
        scale = camera_height / 1.0   # ray scale to reach z_world plane

        X = x_cam * scale
        Y = camera_height        # Approximate forward distance
        Z = z

        return np.array([X, Y, Z], dtype=np.float64)

    # ── Pixel → XYZ (depth strategy) ──────────────────────────────────────────
    def pixel_to_world_depth(
        self,
        px: int,
        py: int,
        depth_map: np.ndarray,
    ) -> np.ndarray:
        """
        Back-project using metric depth (from RealSense) or
        calibrated disparity (from MiDaS).

        Args:
            px, py:    Pixel coords.
            depth_map: Float32 array same size as camera frame.
                       RealSense → metres directly.
                       MiDaS     → disparity; converted via alpha/beta.

        Returns:
            [X, Y, Z] in robot-frame metres.
        """
        disparity = float(depth_map[py, px])
        # Convert disparity → metric depth
        z_cam = self._alpha / (disparity + 1e-6) + self._beta

        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]

        X_cam = (px - cx) * z_cam / fx
        Y_cam = (py - cy) * z_cam / fy
        Z_cam = z_cam

        # Rotate from camera frame to robot frame
        # (camera mounted facing downward, rotated 90° around X-axis)
        X_robot =  X_cam
        Y_robot =  Z_cam
        Z_robot = -Y_cam

        return np.array([X_robot, Y_robot, Z_robot], dtype=np.float64)

    # ── Unified interface ─────────────────────────────────────────────────────
    def map(
        self,
        px: int,
        py: int,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        High-level convenience: chooses strategy automatically.
        """
        if self.strategy == "depth" and depth_map is not None:
            return self.pixel_to_world_depth(px, py, depth_map)
        return self.pixel_to_world_plane(px, py)
