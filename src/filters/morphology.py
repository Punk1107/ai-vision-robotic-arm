"""
morphology.py — Morphological Filters (Category 5)
===================================================
Morphological operations are CRITICAL for segmentation and colour tracking:

  5.1  Erosion   — shrink objects, remove small noise blobs
  5.2  Dilation  — grow objects, fill small gaps
  5.3  Opening   = Erosion → Dilation  (remove noise while preserving shape)
  5.4  Closing   = Dilation → Erosion  (close holes while preserving shape)

Additional operations:
  5.5  Morphological Gradient  = Dilation − Erosion  (outlines objects)
  5.6  Top-Hat  = Original − Opening  (bright spots on dark background)
  5.7  Black-Hat = Closing − Original  (dark spots on bright background)

All operations work on binary (0/255) masks or grayscale images.

Mathematical definitions
------------------------
  Let B = structuring element (kernel), A = input set/image:

  Erosion(A, B)  = {z | B_z ⊆ A}         ← B fits inside A at position z
  Dilation(A, B) = {z | B̂_z ∩ A ≠ ∅}    ← B hits A at position z
  Opening(A, B)  = Dilation(Erosion(A, B), B)
  Closing(A, B)  = Erosion(Dilation(A, B), B)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("filters.morphology")


# ── Helper: structuring element factory ──────────────────────────────────────

def _se(
    shape:  str  = "ellipse",
    ksize:  int  = 5,
    anchor: Tuple[int, int] = (-1, -1),
) -> np.ndarray:
    """
    Build a morphological structuring element.

    Args:
        shape: "rect" | "ellipse" | "cross"
        ksize: Kernel size (odd).

    Returns:
        uint8 kernel array.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    shapes = {
        "rect":    cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross":   cv2.MORPH_CROSS,
    }
    shape_id = shapes.get(shape.lower(), cv2.MORPH_ELLIPSE)
    return cv2.getStructuringElement(shape_id, (ksize, ksize), anchor)


# =============================================================================
# 5.1  Erosion
# =============================================================================

class Erosion:
    """
    Morphological Erosion.

    Shrinks bright regions (foreground) in a binary/grayscale mask.
    Each output pixel is the MINIMUM of the input within the structuring element.

    Effect:
        - Removes thin protrusions, small noise blobs
        - Separates objects that are barely touching
        - Shrinks the boundary of objects inward by kernel_radius pixels

    Use in this project:
        - Remove 1–3 px noise blobs from colour-segmented masks before tracking
        - Separate touching objects in crowded scenes

    Args:
        ksize:      Structuring element size (larger → more erosion).
        shape:      "ellipse" (default), "rect", "cross"
        iterations: Number of times to apply erosion.
    """

    def __init__(
        self,
        ksize:      int  = 5,
        shape:      str  = "ellipse",
        iterations: int  = 1,
    ) -> None:
        self._kernel = _se(shape, ksize)
        self._iters  = iterations
        log.debug(f"Erosion | ksize={ksize} | shape={shape} | iters={iterations}")

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Apply erosion to a binary or grayscale mask."""
        return cv2.erode(mask, self._kernel, iterations=self._iters)


# =============================================================================
# 5.2  Dilation
# =============================================================================

class Dilation:
    """
    Morphological Dilation.

    Grows bright regions (foreground) in a binary/grayscale mask.
    Each output pixel is the MAXIMUM of the input within the structuring element.

    Effect:
        - Fills small holes and gaps within objects
        - Bridges small breaks in edges
        - Expands the boundary of objects outward

    Use in this project:
        - After YOLO segmentation mask: fill gaps in the detected mask
        - Before contour extraction: ensure object boundary is continuous
        - Connect fragmented Canny edges

    Args:
        ksize:      Structuring element size.
        shape:      "ellipse", "rect", "cross"
        iterations: Number of times to apply dilation.
    """

    def __init__(
        self,
        ksize:      int  = 5,
        shape:      str  = "ellipse",
        iterations: int  = 1,
    ) -> None:
        self._kernel = _se(shape, ksize)
        self._iters  = iterations
        log.debug(f"Dilation | ksize={ksize} | shape={shape} | iters={iterations}")

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Apply dilation to a binary or grayscale mask."""
        return cv2.dilate(mask, self._kernel, iterations=self._iters)


# =============================================================================
# 5.3  Opening (Erosion → Dilation)
# =============================================================================

class Opening:
    """
    Morphological Opening = Erosion followed by Dilation.

    Opening removes small, isolated noise blobs while preserving the
    overall size and shape of larger objects.

    Mathematical identity:
        Opening(A, B) = Dilation(Erosion(A, B), B)

    Properties:
        - Idempotent: Opening(Opening(A)) = Opening(A)
        - Anti-extensive: Opening(A) ⊆ A  (never adds new pixels)

    Use in this project:
        - PRIMARY noise removal after colour segmentation
        - Remove 1–5 px noise pixels from detection masks
        - Clean up segmentation boundaries before measurement

    Args:
        ksize: Structuring element size.  Must be > largest noise blob.
    """

    def __init__(
        self,
        ksize: int  = 5,
        shape: str  = "ellipse",
    ) -> None:
        self._kernel = _se(shape, ksize)
        log.debug(f"Opening | ksize={ksize} | shape={shape}")

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Apply opening (erosion then dilation)."""
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)


# =============================================================================
# 5.4  Closing (Dilation → Erosion)
# =============================================================================

class Closing:
    """
    Morphological Closing = Dilation followed by Erosion.

    Closing fills small holes and gaps inside objects while preserving
    their overall contour.

    Mathematical identity:
        Closing(A, B) = Erosion(Dilation(A, B), B)

    Properties:
        - Extensive: A ⊆ Closing(A)  (never removes pixels)

    Use in this project:
        - Close small gaps in YOLO segmentation masks (partial occlusions)
        - Fill hollow bounding-box regions for better centroid estimation
        - Post-processing QC shape masks

    Args:
        ksize: Structuring element size.  Should be ≥ size of holes to fill.
    """

    def __init__(
        self,
        ksize: int  = 7,
        shape: str  = "ellipse",
    ) -> None:
        self._kernel = _se(shape, ksize)
        log.debug(f"Closing | ksize={ksize} | shape={shape}")

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Apply closing (dilation then erosion)."""
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)


# =============================================================================
# 5.5  Morphological Gradient
# =============================================================================

class MorphGradient:
    """
    Morphological Gradient = Dilation − Erosion.

    Highlights the boundary (outline) of objects in binary masks.
    Equivalent to a morphological approximation of the gradient operator.

    Use in this project:
        - Extract precise object outlines for measurement
        - Create contour visualisations without cv2.findContours overhead
    """

    def __init__(self, ksize: int = 3, shape: str = "ellipse") -> None:
        self._kernel = _se(shape, ksize)

    def apply(self, mask: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self._kernel)


# =============================================================================
# 5.6  MorphologicalProcessor — Complete Pipeline
# =============================================================================

class MorphologicalProcessor:
    """
    Production morphological post-processing pipeline for segmentation masks.

    Applies a configurable sequence of morphological operations to clean up
    raw binary masks from YOLO segmentation or colour thresholding.

    Default pipeline (suitable for most robotics pick-and-place scenarios):
        1. Opening  (remove noise pixels)
        2. Closing  (fill object gaps)
        3. Area filter (remove objects smaller than min_area_px)

    The pipeline is designed to be fast (< 2ms for 640×480 mask) and
    composable — add/remove stages via the constructor.

    Args:
        open_ksize:   Opening kernel size.  5–9 is typical.
        close_ksize:  Closing kernel size.  7–15 to fill larger holes.
        min_area_px:  Minimum connected-component area to keep.
                      Components smaller than this are removed as noise.
        morph_shape:  Structuring element shape ("ellipse", "rect", "cross").

    Example::

        proc = MorphologicalProcessor(open_ksize=7, close_ksize=11, min_area_px=500)
        clean_mask = proc.process(raw_segmentation_mask)
        contours = proc.find_contours(clean_mask)
    """

    def __init__(
        self,
        open_ksize:   int  = 7,
        close_ksize:  int  = 11,
        min_area_px:  int  = 500,
        morph_shape:  str  = "ellipse",
    ) -> None:
        self._opening  = Opening(open_ksize,  morph_shape)
        self._closing  = Closing(close_ksize, morph_shape)
        self._min_area = min_area_px
        log.info(
            f"MorphologicalProcessor | open={open_ksize} | close={close_ksize} | "
            f"min_area={min_area_px}px | shape={morph_shape}"
        )

    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply the full morphological pipeline to a binary mask.

        Args:
            mask: Binary uint8 mask (0 or 255, or 0 or 1).

        Returns:
            Cleaned binary mask (same type as input).
        """
        # Normalise to 0/255
        if mask.max() <= 1:
            m = (mask * 255).astype(np.uint8)
        else:
            m = mask.astype(np.uint8)

        # 1. Opening: remove noise
        m = self._opening.apply(m)

        # 2. Closing: fill gaps
        m = self._closing.apply(m)

        # 3. Area filter: remove tiny blobs
        if self._min_area > 0:
            m = self._area_filter(m)

        return m

    def _area_filter(self, mask: np.ndarray) -> np.ndarray:
        """Remove connected components smaller than min_area_px."""
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        out = np.zeros_like(mask)
        for lbl in range(1, n_labels):  # 0 = background
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= self._min_area:
                out[labels == lbl] = 255
        return out

    def find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Find external contours in a binary mask.

        Returns:
            List of contour arrays, sorted by area (largest first).
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return sorted(contours, key=cv2.contourArea, reverse=True)

    def largest_contour_bbox(
        self,
        mask: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Return bounding box [x, y, w, h] of the largest contour, or None.

        Useful for extracting a single dominant object after segmentation.
        """
        contours = self.find_contours(mask)
        if not contours:
            return None
        return cv2.boundingRect(contours[0])
