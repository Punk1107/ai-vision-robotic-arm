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
    Suitable for setups without a hardware depth camera.
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
        
        # Temporal smoothing
        self._last_depth: Optional[np.ndarray] = None
        self._alpha_smooth = 0.6  # Smoothing factor (0.0 - 1.0)

    @torch.inference_mode()
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a disparity map (float32, same H×W as input).
        Larger value = closer to camera (disparity, not metric depth).
        Pass this to CoordinateMapper.map() with strategy="depth".
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
        
        disp = disp.astype(np.float32)

        # Apply temporal smoothing
        if self._last_depth is not None:
            disp = self._alpha_smooth * disp + (1 - self._alpha_smooth) * self._last_depth
        self._last_depth = disp.copy()

        return disp

    def colourize(self, disp: np.ndarray) -> np.ndarray:
        """Return a colour-mapped disparity image for visualisation."""
        norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)


# ─────────────────────────────────────────────────────────────────────────────
# RealSense Depth Estimator (Hardware Metric Depth)
# ─────────────────────────────────────────────────────────────────────────────

class RealSenseDepthEstimator:
    """
    Captures aligned RGB + metric depth frames from an Intel RealSense D4xx.

    Advantages over MiDaS:
      - True metric depth (metres) — no alpha/beta calibration needed.
      - Hardware-aligned colour + depth frames (no registration step).
      - ~30fps depth at 848×480 with minimal CPU overhead.

    Requirements:
      pip install pyrealsense2

    Falls back gracefully: if RealSense hardware or SDK is unavailable,
    raises RuntimeError with a clear install message so the system can
    fall back to MonocularDepthEstimator automatically.
    """

    def __init__(
        self,
        width:  int = 848,
        height: int = 480,
        fps:    int = 30,
    ) -> None:
        try:
            import pyrealsense2 as rs   # type: ignore
            self._rs = rs
        except ImportError:
            raise RuntimeError(
                "pyrealsense2 not installed. "
                "Run: pip install pyrealsense2   "
                "(or install the Intel RealSense SDK 2.0)"
            )

        self._pipeline = self._rs.pipeline()
        cfg = self._rs.config()
        cfg.enable_stream(self._rs.stream.depth, width, height,
                          self._rs.format.z16, fps)
        cfg.enable_stream(self._rs.stream.color, width, height,
                          self._rs.format.bgr8, fps)

        try:
            profile = self._pipeline.start(cfg)
        except RuntimeError as e:
            raise RuntimeError(
                f"RealSense camera not found: {e}. "
                "Check USB connection or set depth.method=monocular in config.yaml."
            ) from e

        # Align depth to colour frame
        self._align = self._rs.align(self._rs.stream.color)

        # Depth scale: converts raw uint16 → metres
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        # Retrieve colour camera intrinsics for CoordinateMapper
        intr = (profile
                .get_stream(self._rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics())
        self._intrinsics = intr
        self._width  = width
        self._height = height

        log.success(
            f"RealSense ready | {width}×{height}@{fps}fps | "
            f"depth_scale={self._depth_scale:.5f} m/unit ✓"
        )

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Capture one aligned (colour, depth) frame pair.

        Returns:
            colour_bgr:  np.ndarray (H, W, 3) uint8 — BGR image.
            depth_m:     np.ndarray (H, W)    float32 — metric depth in metres.
                         Zero values indicate invalid / out-of-range pixels.
        """
        frames        = self._pipeline.wait_for_frames()
        aligned       = self._align.process(frames)
        colour_frame  = aligned.get_color_frame()
        depth_frame   = aligned.get_depth_frame()

        colour_bgr = np.asanyarray(colour_frame.get_data())
        depth_raw  = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_m    = depth_raw * self._depth_scale   # uint16 → metres

        return colour_bgr, depth_m

    def colourize(self, depth_m: np.ndarray) -> np.ndarray:
        """Return a false-colour depth image (TURBO colourmap, clipped to 2m)."""
        clipped  = np.clip(depth_m, 0, 2.0)
        norm_u8  = (clipped / 2.0 * 255).astype(np.uint8)
        return cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)

    def camera_matrix(self) -> np.ndarray:
        """
        Return a 3×3 intrinsic matrix from the live RealSense stream.
        Can be passed directly to CoordinateMapper for accurate back-projection.
        """
        i = self._intrinsics
        return np.array([
            [i.fx,  0,    i.ppx],
            [0,     i.fy, i.ppy],
            [0,     0,    1.0  ],
        ], dtype=np.float64)

    def stop(self) -> None:
        self._pipeline.stop()
        log.info("RealSense pipeline stopped.")

    def __enter__(self) -> "RealSenseDepthEstimator":
        return self

    def __exit__(self, *_) -> None:
        self.stop()


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


# ─────────────────────────────────────────────────────────────────────────────
# Factory: create the correct depth backend from config
# ─────────────────────────────────────────────────────────────────────────────

def create_depth_backend(
    method:   Optional[str] = None,
) -> "MonocularDepthEstimator | RealSenseDepthEstimator | None":
    """
    Factory that reads `depth.method` from config and returns the appropriate
    estimator, or None if depth is disabled.

    method override (str):
        "monocular"  → MonocularDepthEstimator (MiDaS, software-only)
        "realsense"  → RealSenseDepthEstimator (hardware, pyrealsense2)
        "stereo"     → Not yet implemented — falls back to monocular

    Falls back to monocular if RealSense is requested but hardware/SDK
    is unavailable.
    """
    if not config.depth.enabled:
        log.info("Depth backend disabled (config depth.enabled=false)")
        return None

    m = (method or config.depth.method).lower()

    if m == "realsense":
        try:
            estimator = RealSenseDepthEstimator()
            log.success("Depth backend: [bold]RealSense[/bold] (metric) ✓")
            return estimator
        except RuntimeError as e:
            log.warning(
                f"RealSense unavailable ({e}) — "
                "falling back to MiDaS monocular depth."
            )
            m = "monocular"   # fall through

    if m in ("monocular", "stereo"):
        if m == "stereo":
            log.warning("Stereo backend not yet implemented — using MiDaS monocular.")
        estimator = MonocularDepthEstimator(config.depth.monocular_model)
        log.success("Depth backend: [bold]MiDaS monocular[/bold] ✓")
        return estimator

    log.error(f"Unknown depth method '{m}'. Valid: monocular | realsense")
    return None
