"""
detect.py — YOLO Object Detection Engine (v2)
=============================================
Improvements over v1:
  - Letterbox resizing (preserves aspect ratio, standard YOLO pre-process)
  - Model warm-up on first call (removes latency spike on frame 1)
  - Frame-skip support (run heavy inference every N frames, interpolate)
  - Half-precision (FP16) inference when CUDA is available
  - Rich Detection / DetectionResult dataclasses (unchanged API)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("vision.detect")

_PALETTE = [
    (0, 114, 189), (217, 83, 25),  (237, 177, 32),
    (126, 47, 142), (119, 172, 48), (77, 190, 238),
    (162, 20, 47),  (76, 153, 0),   (255, 128, 0),
]


# ── Letterbox helper ──────────────────────────────────────────────────────────

def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to new_shape×new_shape with grey padding (letterbox).
    Preserves aspect ratio — standard YOLOv8 preprocessing.

    Returns:
        (resized_img, scale_ratio, (pad_w, pad_h))
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)

    new_w, new_h = int(round(w * r)), int(round(h * r))
    pad_w = (new_shape - new_w) / 2
    pad_h = (new_shape - new_h) / 2

    if w != new_w or h != new_h:
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top, bot  = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bot, left, right,
        cv2.BORDER_CONSTANT, value=color,
    )
    return img, r, (pad_w, pad_h)


def unletterbox_point(
    px: int, py: int,
    scale: float,
    pad: Tuple[float, float],
) -> Tuple[int, int]:
    """Map a point from letterboxed space back to original image space."""
    return (
        int((px - pad[0]) / scale),
        int((py - pad[1]) / scale),
    )


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    bbox_xyxy:  np.ndarray
    center_px:  Tuple[int, int]
    area_px:    int
    world_xyz:  Optional[np.ndarray] = field(default=None, repr=False)
    track_id:   Optional[int]        = field(default=None)   # filled by tracker

    @property
    def width_px(self) -> int:
        return int(self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height_px(self) -> int:
        return int(self.bbox_xyxy[3] - self.bbox_xyxy[1])

    def colour(self) -> Tuple[int, int, int]:
        return _PALETTE[self.class_id % len(_PALETTE)]


@dataclass
class DetectionResult:
    frame_id:     int
    timestamp:    float
    detections:   List[Detection]
    inference_ms: float

    @property
    def count(self) -> int:
        return len(self.detections)

    def filter_by_class(self, *names: str) -> List[Detection]:
        return [d for d in self.detections if d.class_name in names]

    def best(self) -> Optional[Detection]:
        return max(self.detections, key=lambda d: d.confidence, default=None)


# ── Detector ──────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    YOLOv8 detector with letterbox preprocessing and warm-up.

    Args:
        frame_skip: Run inference every N frames.  Between inference frames
                    the previous result is returned (good for 30fps cameras
                    where arm control only needs ~10 detections/s).
    """

    def __init__(self, frame_skip: int = 2) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for ObjectDetector. "
                "Install it with: pip install ultralytics"
            ) from exc

        model_path = Path(config.yolo.model_path)

        if model_path.exists():
            log.info(f"Loading custom model: [cyan]{model_path}[/cyan]")
            self._model = YOLO(str(model_path))
        else:
            log.warning(
                f"Custom model not found → using [green]{config.yolo.fallback_model}[/green]"
            )
            self._model = YOLO(config.yolo.fallback_model)

        # FP16 on CUDA for ~2× throughput
        self._half = torch.cuda.is_available()
        if self._half:
            self._model.model.half()
            log.info("FP16 inference enabled (CUDA detected)")

        self._conf    = config.yolo.confidence_threshold
        self._iou     = config.yolo.iou_threshold
        self._imgsz   = config.yolo.input_size
        self._targets = set(config.yolo.target_classes)

        self._frame_id       = 0
        self._frame_skip     = max(1, frame_skip)
        self._last_result:   Optional[DetectionResult] = None
        self._last_scale:    float               = 1.0
        self._last_pad:      Tuple[float, float] = (0.0, 0.0)

        self._warmup()
        log.success("ObjectDetector ready ✓")

    def _warmup(self) -> None:
        """Run one dummy forward pass so the first real frame isn't slow."""
        dummy = np.zeros(
            (self._imgsz, self._imgsz, 3), dtype=np.uint8
        )
        log.info("Warming up YOLO model ...")
        t0 = time.perf_counter()
        self._model(dummy, verbose=False)
        log.info(f"Warm-up done in {(time.perf_counter()-t0)*1000:.0f}ms")

    # ── Core inference ────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> DetectionResult:
        self._frame_id += 1

        # Return cached result for skipped frames (still increment frame_id)
        if (self._frame_id % self._frame_skip != 0
                and self._last_result is not None):
            return self._last_result

        lb_frame, scale, pad = letterbox(frame, self._imgsz)
        self._last_scale = scale
        self._last_pad   = pad

        t0 = time.perf_counter()
        results = self._model(
            lb_frame,
            conf    = self._conf,
            iou     = self._iou,
            imgsz   = self._imgsz,
            half    = self._half,
            verbose = False,
        )[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = self._parse(results, scale, pad)

        result = DetectionResult(
            frame_id     = self._frame_id,
            timestamp    = time.time(),
            detections   = detections,
            inference_ms = inference_ms,
        )
        self._last_result = result
        return result

    def _parse(
        self,
        results,
        scale: float,
        pad:   Tuple[float, float],
    ) -> List[Detection]:
        dets: List[Detection] = []
        if results.boxes is None:
            return dets

        names = results.names

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = names[cls_id]

            if self._targets and cls_name not in self._targets:
                continue

            # Unmap letterbox → original frame coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            ox1, oy1 = unletterbox_point(x1, y1, scale, pad)
            ox2, oy2 = unletterbox_point(x2, y2, scale, pad)

            cx   = (ox1 + ox2) // 2
            cy   = (oy1 + oy2) // 2
            area = (ox2 - ox1) * (oy2 - oy1)
            conf = float(box.conf[0])

            dets.append(Detection(
                class_id   = cls_id,
                class_name = cls_name,
                confidence = conf,
                bbox_xyxy  = np.array([ox1, oy1, ox2, oy2]),
                center_px  = (cx, cy),
                area_px    = max(0, area),
            ))

        return dets

    # ── Annotation ────────────────────────────────────────────────────────────
    def annotate_frame(
        self,
        frame:    np.ndarray,
        result:   DetectionResult,
        show_fps: bool = True,
        show_track_id: bool = True,
    ) -> np.ndarray:
        vis = frame.copy()

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            colour = det.colour()

            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

            label = det.class_name
            if show_track_id and det.track_id is not None:
                label += f" #{det.track_id}"
            label += f"  {det.confidence:.0%}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(
                vis, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cx, cy = det.center_px
            cv2.drawMarker(vis, (cx, cy), colour, cv2.MARKER_CROSS, 12, 2)

            if det.world_xyz is not None:
                xyz = det.world_xyz
                cv2.putText(
                    vis,
                    f"({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})m",
                    (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA,
                )

        if show_fps:
            fps = 1000 / result.inference_ms if result.inference_ms > 0 else 0
            cv2.putText(
                vis,
                f"FPS: {fps:.1f}  |  Objects: {result.count}  "
                f"[skip={self._frame_skip}]",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 128), 2, cv2.LINE_AA,
            )
        return vis
