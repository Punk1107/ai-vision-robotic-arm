"""
pose.py — Grasp Pose Estimator
================================
Combines 2D segmentation masks with depth information to compute a
full 6-DoF-style grasp pose for a 4-DoF arm:

  [X, Y, Z]   — 3D centroid position in robot frame (metres)
  [θ_wrist]   — required wrist rotation to align gripper with object

Pipeline
--------
  1. InstanceSegmentor  → SegmentedObject (mask + orientation_deg)
  2. MonocularDepthEstimator or RealSense  → depth map
  3. GraspPoseEstimator.estimate(seg_obj, depth_map, mapper)
     → GraspPose(xyz, wrist_angle, approach_vector, confidence)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.vision.segmentation import SegmentedObject
from src.vision.depth import CoordinateMapper
from src.utils.logger import get_logger

log = get_logger("vision.pose")

_DEPTH_PATCH_HALF = 3


@dataclass
class GraspPose:
    """Full grasp specification for the robot arm."""
    xyz:             np.ndarray
    wrist_angle_deg: float
    approach_vector: np.ndarray
    depth_m:         float
    confidence:      float
    class_name:      str           = ""
    track_id:        Optional[int] = None

    def __str__(self) -> str:
        x, y, z = self.xyz
        return (
            f"GraspPose | {self.class_name} | "
            f"xyz=({x:.3f},{y:.3f},{z:.3f})m | "
            f"wrist={self.wrist_angle_deg:+.1f}° | "
            f"depth={self.depth_m:.3f}m | conf={self.confidence:.2f}"
        )


class GraspPoseEstimator:
    """
    Converts a SegmentedObject + depth map into an actionable GraspPose.

    Args:
        mapper:            CoordinateMapper instance (handles pixel→XYZ).
        min_confidence:    Minimum pose confidence to be considered valid.
        depth_std_thresh:  Max allowed std-dev in the depth patch (metres).
    """

    def __init__(
        self,
        mapper:           CoordinateMapper,
        min_confidence:   float = 0.3,
        depth_std_thresh: float = 0.08,
    ) -> None:
        self._mapper        = mapper
        self._min_conf      = min_confidence
        self._depth_std_thr = depth_std_thresh
        log.info("GraspPoseEstimator ready ✓")

    def estimate(
        self,
        seg_obj:   SegmentedObject,
        depth_map: Optional[np.ndarray] = None,
    ) -> Optional[GraspPose]:
        """
        Compute a GraspPose from a segmented object.

        Uses patch-median depth at the mask centroid for robustness.
        Returns None if depth confidence is too low.
        """
        cx, cy = seg_obj.center_px
        depth_m, depth_conf = self._sample_depth(cx, cy, depth_map, seg_obj.mask)

        if depth_map is not None and depth_conf > self._min_conf:
            xyz = self._mapper.pixel_to_world_depth(cx, cy, depth_map)
        else:
            xyz = self._mapper.pixel_to_world_plane(cx, cy)

        approach    = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        wrist_angle = seg_obj.wrist_angle_deg

        pose_conf = float(
            seg_obj.confidence * 0.5
            + depth_conf       * 0.3
            + self._mask_quality(seg_obj) * 0.2
        )

        if pose_conf < self._min_conf:
            log.debug(
                f"GraspPose for {seg_obj.class_name} rejected "
                f"(conf={pose_conf:.2f} < {self._min_conf})"
            )
            return None

        pose = GraspPose(
            xyz             = xyz,
            wrist_angle_deg = wrist_angle,
            approach_vector = approach,
            depth_m         = depth_m,
            confidence      = pose_conf,
            class_name      = seg_obj.class_name,
            track_id        = seg_obj.track_id,
        )
        log.debug(str(pose))
        return pose

    def estimate_all(
        self,
        seg_objects: List[SegmentedObject],
        depth_map:   Optional[np.ndarray] = None,
    ) -> List[GraspPose]:
        """Estimate poses for all objects in a segmentation result."""
        return [
            p for seg in seg_objects
            if (p := self.estimate(seg, depth_map)) is not None
        ]

    # ── Depth sampling ────────────────────────────────────────────────────────
    def _sample_depth(
        self,
        cx:        int,
        cy:        int,
        depth_map: Optional[np.ndarray],
        mask:      np.ndarray,
    ) -> Tuple[float, float]:
        """Patch-median depth at (cx, cy). Returns (depth_metres, confidence)."""
        if depth_map is None:
            return 0.0, 0.2

        H, W = depth_map.shape[:2]
        h    = _DEPTH_PATCH_HALF
        y0, y1 = max(0, cy - h), min(H, cy + h + 1)
        x0, x1 = max(0, cx - h), min(W, cx + h + 1)
        patch = depth_map[y0:y1, x0:x1]

        if patch.size == 0:
            return 0.0, 0.1

        patch_mask = mask[y0:y1, x0:x1]
        valid_vals = patch[patch_mask > 0] if patch_mask.any() else patch.ravel()
        if len(valid_vals) == 0:
            valid_vals = patch.ravel()
        valid_vals = valid_vals[np.isfinite(valid_vals) & (valid_vals > 0)]
        if len(valid_vals) == 0:
            return 0.0, 0.1

        depth_median = float(np.median(valid_vals))
        depth_std    = float(np.std(valid_vals))

        # Heuristic: if values are small (< 20), treat as metric depth (RealSense)
        if depth_median < 20.0:
            depth_m = depth_median
        else:
            alpha   = self._mapper._alpha
            beta    = self._mapper._beta
            depth_m = float(alpha / (depth_median + 1e-6) + beta)

        conf = float(np.clip(1.0 - depth_std / self._depth_std_thr, 0.0, 1.0))
        return depth_m, conf

    def _mask_quality(self, seg: SegmentedObject) -> float:
        """Fill ratio of mask vs bounding box → proxy for segmentation quality."""
        x1, y1, x2, y2 = seg.bbox_xyxy
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        fill_ratio = float(seg.area_px) / bbox_area
        return float(np.clip((fill_ratio - 0.1) / 0.8, 0.0, 1.0))

    @staticmethod
    def annotate_pose(
        frame:  np.ndarray,
        pose:   GraspPose,
        colour: Tuple[int, int, int] = (0, 255, 255),
    ) -> np.ndarray:
        vis  = frame.copy()
        x, y, z = pose.xyz
        text = (
            f"GRASP: ({x:.2f},{y:.2f},{z:.2f})m  "
            f"wrist={pose.wrist_angle_deg:+.1f}°  "
            f"conf={pose.confidence:.0%}"
        )
        cv2.putText(
            vis, text, (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2, cv2.LINE_AA,
        )
        return vis
