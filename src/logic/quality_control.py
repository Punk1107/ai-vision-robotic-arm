"""
quality_control.py — Visual Defect Detection & QA Pipeline
===========================================================
Implements an automated quality control (QA) inspection pipeline that
runs BEFORE the pick decision.  An object flagged as defective is routed
to the reject bin instead of its normal sort destination.

Detection Approach
------------------
We use a two-stage pipeline:

  Stage 1 — Texture anomaly (statistical):
      Compute local contrast (Laplacian variance) and colour uniformity
      (std-dev of HSV channels) within the object mask.
      Objects with unusual texture or colour distribution are flagged.

  Stage 2 — Structural anomaly (shape-based):
      Compare the object's mask solidity (filled area / convex hull area).
      A badly dented or broken object will have lower solidity.

  Stage 3 — ML-based (optional, if defect model is trained):
      Classify the masked ROI with a custom binary classifier
      (defect / no-defect) for high-accuracy deployment.

Each stage returns a DefectScore (0.0 = perfect → 1.0 = severely defective).
The final verdict is a weighted combination.

Novelty (for your presentation)
--------------------------------
- This turns the project from "demo sorter" to "industrial QA system".
- You can demo it by introducing a dented can / crumpled paper to the camera.
- The rejection bin routing is handled by decision.py, not here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("logic.quality_control")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds — tune for your physical objects
LAPLACIAN_LOW_THRESH   = 15.0    # very smooth/blurry mask → possibly defective
COLOUR_STD_HIGH_THRESH = 45.0   # very noisy colour → possible surface damage
SOLIDITY_LOW_THRESH    = 0.82   # < 0.82 → non-convex / dented shape
ML_DEFECT_THRESH       = 0.60   # ML model output > this → defect

# Stage weights for final score
WEIGHT_TEXTURE  = 0.30
WEIGHT_COLOUR   = 0.25
WEIGHT_SHAPE    = 0.25
WEIGHT_ML       = 0.20  # only if model is available


@dataclass
class DefectReport:
    """Inspection result for a single object."""
    class_name:       str
    is_defective:     bool
    defect_score:     float          # 0.0 = OK, 1.0 = very defective
    texture_score:    float          # contribution from Laplacian analysis
    colour_score:     float          # contribution from HSV std-dev
    shape_score:      float          # contribution from solidity
    ml_score:         float          # ML model output (−1 = not available)
    inspection_ms:    float
    notes:            List[str]      = field(default_factory=list)

    def __str__(self) -> str:
        verdict = "DEFECTIVE ⚠" if self.is_defective else "OK ✓"
        return (
            f"QA [{verdict}] {self.class_name} | "
            f"score={self.defect_score:.2f} "
            f"(tex={self.texture_score:.2f} "
            f"col={self.colour_score:.2f} "
            f"shp={self.shape_score:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# QualityInspector
# ─────────────────────────────────────────────────────────────────────────────

class QualityInspector:
    """
    Multi-stage visual quality inspector.

    Args:
        defect_threshold:   Overall defect_score cutoff.  Above → reject.
        ml_model_path:      Optional path to a custom ONNX/TorchScript
                            binary classifier (defect/no-defect).
                            If None, ML stage is skipped.
        per_class_config:   Optional dict of per-class threshold overrides.
                            Example: {"can": {"solidity_thresh": 0.88}}
    """

    _REJECT_ZONE = np.array([0.00, 0.35, 0.05])   # robot frame (m)

    def __init__(
        self,
        defect_threshold:  float = 0.55,
        ml_model_path:     Optional[Path] = None,
        per_class_config:  Optional[Dict] = None,
    ) -> None:
        self._defect_thr     = defect_threshold
        self._per_class      = per_class_config or {}
        self._ml_model       = None
        self._ml_available   = False

        if ml_model_path and Path(ml_model_path).exists():
            self._load_ml_model(ml_model_path)

        log.info(
            f"QualityInspector ready | "
            f"defect_thr={defect_threshold:.2f} | "
            f"ml={'enabled' if self._ml_available else 'disabled'}"
        )

    def _load_ml_model(self, path: Path) -> None:
        """Load ONNX defect classifier. Falls back gracefully if unavailable."""
        try:
            import onnxruntime as ort   # type: ignore
            self._ml_model    = ort.InferenceSession(str(path))
            self._ml_input_name = self._ml_model.get_inputs()[0].name
            self._ml_available  = True
            log.success(f"ML defect model loaded: {path.name} ✓")
        except ImportError:
            log.warning(
                "onnxruntime not installed — ML defect stage disabled. "
                "Run: pip install onnxruntime"
            )
        except Exception as e:
            log.warning(f"ML model load failed: {e} — falling back to heuristics")

    # ── Main API ──────────────────────────────────────────────────────────────
    def inspect(
        self,
        frame:      np.ndarray,
        mask:       np.ndarray,
        class_name: str = "unknown",
    ) -> DefectReport:
        """
        Inspect a single object.

        Args:
            frame:      Full BGR camera frame.
            mask:       Boolean / uint8 binary mask (H, W) for this object.
            class_name: Object class label (used for per-class thresholds).

        Returns:
            DefectReport with verdict and breakdown scores.
        """
        t0    = time.perf_counter()
        notes: List[str] = []

        # ── Extract masked ROI ────────────────────────────────────────────────
        mask_u8 = (mask > 0).astype(np.uint8)
        roi     = cv2.bitwise_and(frame, frame, mask=mask_u8)

        # ── Stage 1: Texture (Laplacian variance) ─────────────────────────────
        tex_score = self._texture_score(roi, mask_u8, notes)

        # ── Stage 2: Colour uniformity ────────────────────────────────────────
        col_score = self._colour_score(roi, mask_u8, notes)

        # ── Stage 3: Shape / Solidity ─────────────────────────────────────────
        shp_score = self._shape_score(mask_u8, class_name, notes)

        # ── Stage 4: ML ───────────────────────────────────────────────────────
        ml_score = self._ml_score(roi, mask_u8) if self._ml_available else -1.0

        # ── Weighted combination ──────────────────────────────────────────────
        if self._ml_available and ml_score >= 0:
            defect_score = (
                WEIGHT_TEXTURE * tex_score
                + WEIGHT_COLOUR  * col_score
                + WEIGHT_SHAPE   * shp_score
                + WEIGHT_ML      * ml_score
            )
        else:
            # Redistribute ML weight evenly
            w = 1.0 / 3.0
            defect_score = (
                w * tex_score + w * col_score + w * shp_score
            )

        is_defective = defect_score >= self._defect_thr

        report = DefectReport(
            class_name    = class_name,
            is_defective  = is_defective,
            defect_score  = float(defect_score),
            texture_score = float(tex_score),
            colour_score  = float(col_score),
            shape_score   = float(shp_score),
            ml_score      = float(ml_score),
            inspection_ms = (time.perf_counter() - t0) * 1000,
            notes         = notes,
        )
        log.info(str(report))
        return report

    # ── Stage 1: Texture ──────────────────────────────────────────────────────
    def _texture_score(
        self,
        roi:    np.ndarray,
        mask:   np.ndarray,
        notes:  List[str],
    ) -> float:
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap   = cv2.Laplacian(gray, cv2.CV_64F)
        # Only measure within mask
        lap_vals = lap[mask > 0]
        if len(lap_vals) == 0:
            return 0.0

        variance = float(np.var(lap_vals))

        if variance < LAPLACIAN_LOW_THRESH:
            notes.append(f"Low texture variance ({variance:.1f}) → blurry/smooth surface")
            # Clamp: 0 variance → score 1.0, high variance → score 0.0
            score = 1.0 - np.clip(variance / LAPLACIAN_LOW_THRESH, 0.0, 1.0)
        else:
            score = 0.0

        return float(score)

    # ── Stage 2: Colour uniformity ────────────────────────────────────────────
    def _colour_score(
        self,
        roi:   np.ndarray,
        mask:  np.ndarray,
        notes: List[str],
    ) -> float:
        hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        pixels = hsv[mask > 0]  # shape (N, 3)
        if len(pixels) == 0:
            return 0.0

        # High std-dev in Saturation + Value channels → irregular surface
        std_s = float(np.std(pixels[:, 1]))
        std_v = float(np.std(pixels[:, 2]))
        avg_std = (std_s + std_v) / 2.0

        if avg_std > COLOUR_STD_HIGH_THRESH:
            notes.append(
                f"High colour variance (std_S={std_s:.1f}, std_V={std_v:.1f})"
            )
            score = np.clip(
                (avg_std - COLOUR_STD_HIGH_THRESH) / COLOUR_STD_HIGH_THRESH,
                0.0, 1.0,
            )
        else:
            score = 0.0

        return float(score)

    # ── Stage 3: Shape solidity ───────────────────────────────────────────────
    def _shape_score(
        self,
        mask:       np.ndarray,
        class_name: str,
        notes:      List[str],
    ) -> float:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0

        cnt   = max(contours, key=cv2.contourArea)
        area  = cv2.contourArea(cnt)
        hull  = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        if hull_area < 1:
            return 0.0

        solidity = float(area / hull_area)

        # Per-class threshold override
        thresh = float(
            self._per_class.get(class_name, {}).get("solidity_thresh",
                                                      SOLIDITY_LOW_THRESH)
        )

        if solidity < thresh:
            notes.append(
                f"Low solidity ({solidity:.3f} < {thresh:.3f}) → irregular/dented shape"
            )
            score = np.clip((thresh - solidity) / thresh, 0.0, 1.0)
        else:
            score = 0.0

        return float(score)

    # ── Stage 4: ML classifier ────────────────────────────────────────────────
    def _ml_score(self, roi: np.ndarray, mask: np.ndarray) -> float:
        """Run ONNX defect classifier. Returns probability of defect [0, 1]."""
        if self._ml_model is None:
            return -1.0
        try:
            # Crop tight to mask bounding box, resize to 64×64
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                return 0.0
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            x0, x1 = int(xs.min()), int(xs.max() + 1)
            crop    = roi[y0:y1, x0:x1]
            resized = cv2.resize(crop, (64, 64)).astype(np.float32) / 255.0
            inp     = resized.transpose(2, 0, 1)[None]  # (1, 3, 64, 64)

            outputs = self._ml_model.run(None, {self._ml_input_name: inp})
            # Expected output: (1, 2) logits or (1, 1) sigmoid
            out = np.array(outputs[0]).ravel()
            if len(out) == 2:
                # Softmax class 1 = defective
                exp = np.exp(out - out.max())
                prob = float(exp[1] / exp.sum())
            else:
                prob = float(out[0])

            return float(np.clip(prob, 0.0, 1.0))
        except Exception as e:
            log.debug(f"ML defect inference error: {e}")
            return -1.0

    # ── Batch inspect ─────────────────────────────────────────────────────────
    def inspect_batch(
        self,
        frame:      np.ndarray,
        objects:    list,   # List[SegmentedObject] or list of (mask, class_name)
    ) -> List[DefectReport]:
        """
        Inspect a list of objects.  Accepts either SegmentedObjects or
        raw (mask, class_name) tuples.
        """
        reports = []
        for obj in objects:
            if hasattr(obj, "mask"):
                report = self.inspect(frame, obj.mask, obj.class_name)
            else:
                mask, cls_name = obj
                report = self.inspect(frame, mask, cls_name)
            reports.append(report)
        return reports

    # ── Visualisation ─────────────────────────────────────────────────────────
    @staticmethod
    def annotate_frame(
        frame:   np.ndarray,
        reports: List[DefectReport],
        masks:   Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Overlay defect verdict on frame.  Green contour = OK, Red = defective.
        """
        vis = frame.copy()

        for i, report in enumerate(reports):
            colour = (0, 60, 255) if report.is_defective else (0, 220, 80)

            if masks and i < len(masks):
                mask_u8 = (masks[i] > 0).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, colour, 3)

            text = (
                f"{'DEFECT' if report.is_defective else 'OK'} "
                f"{report.defect_score:.2f}"
            )
            log.debug(f"QA annotation: {text}")

        return vis

    @property
    def reject_zone(self) -> np.ndarray:
        """3D position of the reject bin in robot frame."""
        return self._REJECT_ZONE.copy()
