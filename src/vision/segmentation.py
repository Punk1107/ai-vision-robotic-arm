"""
segmentation.py — Instance Segmentation + Orientation Estimation
=================================================================
Upgrades from bounding-box detection to pixel-level instance masks
using YOLOv8-Seg.  The key addition is **grasp orientation**:

  1. Extract the binary mask for each detected object.
  2. Run PCA on the mask pixels to find the primary axis.
  3. Return the orientation angle (degrees) so the IK solver can
     rotate the wrist to align the gripper with the object.

Why PCA instead of bounding-box aspect ratio?
  - Works for irregular shapes (L-shaped objects, tools, etc.)
  - Not sensitive to near-square bounding boxes that give ambiguous orientation
  - Consistent even when object is partially occluded

Usage::

    seg = InstanceSegmentor()
    result = seg.segment(frame)
    for seg_det in result.segmentations:
        print(seg_det.orientation_deg)   # angle to rotate wrist
        vis = seg.annotate_frame(frame, result)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("vision.segmentation")

_PALETTE = [
    (0, 114, 189), (217, 83, 25),  (237, 177, 32),
    (126, 47, 142), (119, 172, 48), (77, 190, 238),
    (162, 20, 47),  (76, 153, 0),   (255, 128, 0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentedObject:
    """A single instance-segmented object with orientation."""
    class_id:       int
    class_name:     str
    confidence:     float
    bbox_xyxy:      np.ndarray          # [x1, y1, x2, y2] in original frame
    center_px:      Tuple[int, int]     # centroid of the mask
    area_px:        int                 # mask pixel count
    mask:           np.ndarray          # binary mask (H, W) bool
    orientation_deg: float              # primary axis angle, 0–180°
    world_xyz:      Optional[np.ndarray] = field(default=None, repr=False)
    track_id:       Optional[int]        = field(default=None)

    def colour(self) -> Tuple[int, int, int]:
        return _PALETTE[self.class_id % len(_PALETTE)]

    @property
    def wrist_angle_deg(self) -> float:
        """
        Angle to command the wrist servo so the gripper aligns with the object.
        Remaps orientation_deg (0–180°) to wrist joint range (-90°–+90°).
        """
        # Orientation is measured from horizontal (0° = pointing right).
        # Wrist angle: 0° = neutral, positive = rotate clockwise.
        return self.orientation_deg - 90.0


@dataclass
class SegmentationResult:
    frame_id:       int
    timestamp:      float
    segmentations:  List[SegmentedObject]
    inference_ms:   float

    @property
    def count(self) -> int:
        return len(self.segmentations)

    def best(self) -> Optional[SegmentedObject]:
        return max(self.segmentations, key=lambda s: s.confidence, default=None)

    def filter_by_class(self, *names: str) -> List[SegmentedObject]:
        return [s for s in self.segmentations if s.class_name in names]


# ─────────────────────────────────────────────────────────────────────────────
# PCA Orientation Helper
# ─────────────────────────────────────────────────────────────────────────────

def _pca_orientation(mask: np.ndarray) -> Tuple[Tuple[int, int], float]:
    """
    Compute the centroid and primary axis orientation of a binary mask.

    Args:
        mask: Boolean or uint8 2D array (H, W).

    Returns:
        (centroid_xy, orientation_degrees)
        orientation_degrees: Angle of the major axis from the positive X-axis,
                             in [0, 180°).
    """
    pts = np.column_stack(np.where(mask > 0))   # (N, 2) → row, col
    if len(pts) < 5:
        # Degenerate mask — can't compute orientation
        ys, xs = np.where(mask > 0)
        cy = int(np.mean(ys)) if len(ys) > 0 else mask.shape[0] // 2
        cx = int(np.mean(xs)) if len(xs) > 0 else mask.shape[1] // 2
        return (cx, cy), 0.0

    # pts columns: [row=y, col=x] — flip for (x, y) convention
    xy = pts[:, ::-1].astype(np.float32)   # (N, 2) in (x, y) order
    mean_xy = xy.mean(axis=0)

    # Covariance matrix
    centred = xy - mean_xy
    cov = (centred.T @ centred) / max(len(centred) - 1, 1)

    # Eigen-decomposition → largest eigenvector = primary axis
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]  # (2,)

    angle_rad = np.arctan2(major_axis[1], major_axis[0])
    angle_deg = float(np.degrees(angle_rad)) % 180.0      # keep in [0, 180)

    cx, cy = int(mean_xy[0]), int(mean_xy[1])
    return (cx, cy), angle_deg


# ─────────────────────────────────────────────────────────────────────────────
# InstanceSegmentor
# ─────────────────────────────────────────────────────────────────────────────

class InstanceSegmentor:
    """
    YOLOv8-Seg based instance segmentor with automatic orientation detection.

    Tries to load a custom segmentation model (best.pt) first; if not found,
    falls back to the official yolov8n-seg.pt (auto-downloaded).

    Args:
        frame_skip: Run segmentation inference every N frames (same as detect.py).
    """

    # Seg model file takes priority; if absent, use the standard seg variant
    _SEG_MODEL_NAME = "yolov8n-seg.pt"

    def __init__(self, frame_skip: int = 3) -> None:
        seg_model_path = Path(config.yolo.model_path).with_name(
            Path(config.yolo.model_path).stem + "-seg.pt"
        )

        if seg_model_path.exists():
            log.info(f"Loading custom seg model: [cyan]{seg_model_path}[/cyan]")
            self._model = YOLO(str(seg_model_path))
        else:
            log.warning(
                f"Custom seg model not found → using [green]{self._SEG_MODEL_NAME}[/green]"
            )
            self._model = YOLO(self._SEG_MODEL_NAME)

        self._half = torch.cuda.is_available()
        if self._half:
            self._model.model.half()
            log.info("FP16 segmentation inference enabled (CUDA detected)")

        self._conf       = config.yolo.confidence_threshold
        self._iou        = config.yolo.iou_threshold
        self._imgsz      = config.yolo.input_size
        self._targets    = set(config.yolo.target_classes)
        self._frame_skip = max(1, frame_skip)

        self._frame_id       = 0
        self._last_result:   Optional[SegmentationResult] = None

        self._warmup()
        log.success("InstanceSegmentor ready ✓")

    # ── Warm-up ───────────────────────────────────────────────────────────────
    def _warmup(self) -> None:
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        log.info("Warming up segmentation model ...")
        t0 = time.perf_counter()
        self._model(dummy, verbose=False)
        log.info(f"Seg warm-up done in {(time.perf_counter()-t0)*1000:.0f}ms")

    # ── Core inference ────────────────────────────────────────────────────────
    def segment(self, frame: np.ndarray) -> SegmentationResult:
        """
        Run instance segmentation on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            SegmentationResult with masks, centroids, and orientation angles.
        """
        self._frame_id += 1

        if (self._frame_id % self._frame_skip != 0
                and self._last_result is not None):
            return self._last_result

        t0 = time.perf_counter()
        results = self._model(
            frame,
            conf    = self._conf,
            iou     = self._iou,
            imgsz   = self._imgsz,
            half    = self._half,
            verbose = False,
        )[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        segmentations = self._parse(results, frame.shape[:2])

        result = SegmentationResult(
            frame_id      = self._frame_id,
            timestamp     = time.time(),
            segmentations = segmentations,
            inference_ms  = inference_ms,
        )
        self._last_result = result
        return result

    def _parse(
        self,
        results,
        frame_shape: Tuple[int, int],   # (H, W)
    ) -> List[SegmentedObject]:
        seg_objs: List[SegmentedObject] = []
        H, W = frame_shape

        if results.boxes is None:
            return seg_objs

        names  = results.names
        boxes  = results.boxes
        masks  = results.masks   # may be None if no masks predicted

        for i, box in enumerate(boxes):
            cls_id   = int(box.cls[0])
            cls_name = names[cls_id]

            if self._targets and cls_name not in self._targets:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

            # ── Extract mask ──────────────────────────────────────────────────
            if masks is not None and i < len(masks.data):
                # masks.data is (N, H_mask, W_mask); resize to frame size
                mask_raw = masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(
                    mask_raw, (W, H), interpolation=cv2.INTER_NEAREST
                )
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
            else:
                # Fallback: filled bounding box as approximate mask
                binary_mask = np.zeros((H, W), dtype=np.uint8)
                binary_mask[y1:y2, x1:x2] = 1

            # ── PCA orientation ───────────────────────────────────────────────
            (cx, cy), orient = _pca_orientation(binary_mask)
            area_px = int(binary_mask.sum())

            seg_objs.append(SegmentedObject(
                class_id        = cls_id,
                class_name      = cls_name,
                confidence      = conf,
                bbox_xyxy       = np.array([x1, y1, x2, y2]),
                center_px       = (cx, cy),
                area_px         = area_px,
                mask            = binary_mask.astype(bool),
                orientation_deg = orient,
            ))

        return seg_objs

    # ── Annotation ────────────────────────────────────────────────────────────
    def annotate_frame(
        self,
        frame:   np.ndarray,
        result:  SegmentationResult,
        alpha:   float = 0.35,   # mask overlay transparency
    ) -> np.ndarray:
        """
        Draw segmentation masks, orientation arrows, and labels on a copy of frame.
        """
        vis = frame.copy()
        overlay = frame.copy()

        for seg in result.segmentations:
            colour  = seg.colour()
            cx, cy  = seg.center_px

            # ── Coloured mask overlay ─────────────────────────────────────────
            overlay[seg.mask] = colour

            # ── Bounding box ──────────────────────────────────────────────────
            x1, y1, x2, y2 = seg.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

            # ── Orientation arrow (primary axis) ──────────────────────────────
            arrow_len = min(50, min(x2 - x1, y2 - y1) // 2)
            angle_rad = np.radians(seg.orientation_deg)
            dx = int(arrow_len * np.cos(angle_rad))
            dy = int(arrow_len * np.sin(angle_rad))
            cv2.arrowedLine(
                vis,
                (cx - dx, cy - dy), (cx + dx, cy + dy),
                colour, 2, tipLength=0.3,
            )

            # ── Label ─────────────────────────────────────────────────────────
            label = (
                f"{seg.class_name}  {seg.confidence:.0%}  "
                f"θ={seg.orientation_deg:.1f}°  "
                f"wrist={seg.wrist_angle_deg:+.1f}°"
            )
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(
                vis, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # ── Centroid marker ───────────────────────────────────────────────
            cv2.drawMarker(vis, (cx, cy), colour, cv2.MARKER_STAR, 10, 2)

        # Blend mask overlay
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

        # HUD
        fps = 1000 / result.inference_ms if result.inference_ms > 0 else 0
        cv2.putText(
            vis,
            f"SEG FPS: {fps:.1f}  |  Objects: {result.count}  [skip={self._frame_skip}]",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 200), 2, cv2.LINE_AA,
        )
        return vis
