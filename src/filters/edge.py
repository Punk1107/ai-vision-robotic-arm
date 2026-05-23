"""
edge.py — Edge Detection / Feature Filters (Category 4)
========================================================
Classical computer vision edge detection operators:

  4.1  Sobel Filter      — gradient magnitude / direction
  4.2  Canny Edge Detector — multi-stage, most popular in robotics vision
  4.3  Laplacian Filter  — 2nd-derivative edge detection
  4.4  Prewitt Filter    — simpler gradient operator (similar to Sobel)

Mathematical background
-----------------------
  Sobel:
    Gx = [[-1, 0, +1],        Gy = [[-1, -2, -1],
          [-2, 0, +2],               [ 0,  0,  0],
          [-1, 0, +1]]               [+1, +2, +1]]
    |G| = √(Gx² + Gy²),   θ = atan2(Gy, Gx)

  Laplacian (isotropic 2nd derivative):
    ∇²f = ∂²f/∂x² + ∂²f/∂y²
    Kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]

  Canny pipeline:
    1. Gaussian smooth     (reduce noise)
    2. Sobel gradient      (find edge candidates)
    3. Non-max suppression (thin edges to 1 px)
    4. Double threshold    (strong / weak edge classes)
    5. Hysteresis linking  (connect weak edges to strong)

  Prewitt:
    Px = [[-1, 0, +1],        Py = [[-1, -1, -1],
          [-1, 0, +1],               [ 0,  0,  0],
          [-1, 0, +1]]               [+1, +1, +1]]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("filters.edge")


# =============================================================================
# 4.1  Sobel Filter
# =============================================================================

@dataclass
class SobelResult:
    """Output of the Sobel filter operation."""
    gradient_x:   np.ndarray   # Horizontal gradient Gx
    gradient_y:   np.ndarray   # Vertical gradient Gy
    magnitude:    np.ndarray   # |G| = sqrt(Gx² + Gy²)
    direction_rad: np.ndarray  # θ = atan2(Gy, Gx) in radians


class SobelFilter:
    """
    Sobel Edge Gradient Filter.

    Computes the image gradient in X and Y directions using 3×3 (or 5×5)
    convolution kernels.  The gradient magnitude highlights edges and the
    gradient direction indicates edge orientation — both are important for:

        ∇f = [∂f/∂x, ∂f/∂y]   (gradient vector at each pixel)

    Applications in this project:
        - Line detection for conveyor belt edge tracking
        - Feature extraction preprocessing (gradient input to CNN first layer)
        - Workspace boundary detection

    Args:
        ksize:         Sobel kernel size (1, 3, 5, or 7).  3 is standard.
        scale:         Optional scale factor applied to the gradient values.
        delta:         Optional offset added to the gradient values.
        output_type:   "magnitude" | "both" | "direction"
        normalise:     If True, normalise magnitude to [0, 255] uint8.
    """

    def __init__(
        self,
        ksize:       int  = 3,
        scale:       float = 1.0,
        delta:       float = 0.0,
        normalise:   bool  = True,
    ) -> None:
        if ksize not in (1, 3, 5, 7):
            raise ValueError("Sobel ksize must be 1, 3, 5, or 7.")
        self._ksize   = ksize
        self._scale   = scale
        self._delta   = delta
        self._norm    = normalise
        log.debug(f"SobelFilter | ksize={ksize} | norm={normalise}")

    def apply(self, frame: np.ndarray) -> SobelResult:
        """
        Args:
            frame: BGR or grayscale image.

        Returns:
            SobelResult with Gx, Gy, magnitude, and direction arrays.
        """
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sobel in X and Y (float64 for precision)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._ksize,
                       scale=self._scale, delta=self._delta)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._ksize,
                       scale=self._scale, delta=self._delta)

        mag = np.sqrt(gx ** 2 + gy ** 2)
        ang = np.arctan2(gy, gx)   # radians, [-π, π]

        if self._norm:
            mag_out = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        else:
            mag_out = mag

        return SobelResult(
            gradient_x    = gx,
            gradient_y    = gy,
            magnitude     = mag_out,
            direction_rad = ang,
        )

    def magnitude_only(self, frame: np.ndarray) -> np.ndarray:
        """Fast path: return only the gradient magnitude image."""
        return self.apply(frame).magnitude


# =============================================================================
# 4.2  Canny Edge Detector
# =============================================================================

class CannyEdgeDetector:
    """
    Canny Multi-Stage Edge Detector.

    The gold standard for edge detection in robotics CV:

        1. Gaussian smooth  (σ controlled by ksize)
        2. Sobel gradient   (edge strength and direction)
        3. Non-max suppression (NMS) → thin edges
        4. Double threshold → strong / weak edge pixels
        5. Hysteresis edge tracking (connect weak→strong chains)

    Produces binary edge maps (0 = no edge, 255 = edge).

    Parameter tuning guide:
        low_threshold:   Lower hysteresis bound.
                         Typical: 50–100 for 8-bit images.
        high_threshold:  Upper hysteresis bound.
                         Rule: high ≈ 2×low or 3×low.
        aperture_size:   Sobel kernel size (3, 5, or 7).
        auto_sigma:      If True, use Otsu's method to auto-set thresholds.

    Applications in this project:
        - Contour detection for object grasping polygon extraction
        - Lane / conveyor edge detection
        - Obstacle boundary detection before path planning

    Args:
        low_threshold:   Lower threshold for hysteresis (0–255).
        high_threshold:  Upper threshold for hysteresis (0–255).
        aperture_size:   Sobel kernel size (3, 5, or 7).
        L2_gradient:     Use L2 norm for gradient magnitude (more accurate).
        blur_ksize:      Gaussian pre-blur kernel size.  0 = no blur.
        auto_threshold:  If True, compute thresholds from image statistics.
    """

    def __init__(
        self,
        low_threshold:  float = 50.0,
        high_threshold: float = 150.0,
        aperture_size:  int   = 3,
        L2_gradient:    bool  = True,
        blur_ksize:     int   = 5,
        auto_threshold: bool  = False,
    ) -> None:
        self._low    = low_threshold
        self._high   = high_threshold
        self._ap     = aperture_size
        self._L2     = L2_gradient
        self._bk     = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self._auto   = auto_threshold
        log.debug(
            f"CannyEdgeDetector | thr=[{low_threshold},{high_threshold}] | "
            f"ap={aperture_size} | L2={L2_gradient} | auto={auto_threshold}"
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR or grayscale image.

        Returns:
            Binary edge map (uint8, 0 or 255).
        """
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._bk > 1:
            gray = cv2.GaussianBlur(gray, (self._bk, self._bk), 0)

        low, high = self._low, self._high
        if self._auto:
            low, high = self._auto_thresholds(gray)

        return cv2.Canny(
            gray,
            low, high,
            apertureSize = self._ap,
            L2gradient   = self._L2,
        )

    def _auto_thresholds(self, gray: np.ndarray) -> Tuple[float, float]:
        """Otsu-based automatic threshold estimation."""
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high = float(np.percentile(gray[gray > 0], 75)) if gray.any() else 150.0
        low  = high * 0.4
        return low, high

    def apply_and_overlay(
        self,
        frame:       np.ndarray,
        edge_colour: Tuple[int, int, int] = (0, 255, 80),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Canny and overlay edges on the original frame.

        Returns:
            (edges, overlay)  — both as BGR images.
        """
        edges   = self.apply(frame)
        overlay = frame.copy()
        overlay[edges > 0] = edge_colour
        return edges, overlay

    def update_thresholds(self, low: float, high: float) -> None:
        """Hot-swap thresholds at runtime (e.g., for adaptive tuning)."""
        self._low  = low
        self._high = high


# =============================================================================
# 4.3  Laplacian Filter
# =============================================================================

class LaplacianFilter:
    """
    Laplacian (2nd-order derivative) Edge Detection Filter.

    ∇²f = ∂²f/∂x² + ∂²f/∂y²

    The Laplacian is isotropic (no preferred direction) and detects all
    edges regardless of orientation.  It is often used as:
        1. Direct edge detector (LoG = Laplacian-of-Gaussian)
        2. Sharpening kernel (Unsharp Mask variant)
        3. Texture quantification (variance of Laplacian = image sharpness)

    The Laplacian-of-Gaussian (LoG) pre-blurs with Gaussian to reduce
    noise sensitivity before the second derivative is applied.

    Applications in this project:
        - Image sharpness measurement (blur detection for autofocus / QC)
        - Edge enhancement for small-object detection
        - Texture feature extraction in quality_control.py

    Args:
        ksize:   Laplacian kernel size (1, 3, or 5).
        use_log: If True, apply Gaussian before Laplacian (LoG).
        sigma:   Gaussian sigma for LoG pre-smoothing.
        normalise: Normalise output to [0, 255] uint8.
    """

    def __init__(
        self,
        ksize:     int   = 3,
        use_log:   bool  = True,
        sigma:     float = 1.0,
        normalise: bool  = True,
    ) -> None:
        self._ksize  = ksize
        self._use_log = use_log
        self._sigma  = sigma
        self._norm   = normalise
        log.debug(f"LaplacianFilter | ksize={ksize} | LoG={use_log} | σ={sigma}")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR or grayscale image.

        Returns:
            Laplacian edge image (uint8 if normalise=True, else float64).
        """
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._use_log:
            # Gaussian pre-smoothing (LoG)
            ksize_g = max(3, int(2 * round(3 * self._sigma) + 1))
            ksize_g = ksize_g if ksize_g % 2 == 1 else ksize_g + 1
            gray    = cv2.GaussianBlur(gray, (ksize_g, ksize_g), self._sigma)

        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=self._ksize)
        lap_abs = np.abs(lap)

        if self._norm:
            return cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return lap_abs

    def variance(self, frame: np.ndarray) -> float:
        """
        Compute Laplacian variance — a measure of image sharpness.

        Used in quality_control.py (texture_score) to detect blurry objects.
        A high value = sharp image; low value = blurry / defective texture.
        """
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())


# =============================================================================
# 4.4  Prewitt Filter
# =============================================================================

class PrewittFilter:
    """
    Prewitt Gradient Filter — simpler, faster alternative to Sobel.

    Prewitt kernels:
        Px = [[-1, 0, +1],   Py = [[-1, -1, -1],
              [-1, 0, +1],          [ 0,  0,  0],
              [-1, 0, +1]]          [+1, +1, +1]]

    Difference from Sobel:
        - Sobel weights central rows/cols more (isotropic)
        - Prewitt uses equal weights (simpler, separable)
        - Prewitt is slightly more sensitive to noise than Sobel

    Applications in this project:
        - Quick gradient estimate on embedded hardware (fewer multiplications)
        - Direction of conveyor belt features

    Args:
        normalise: Normalise gradient magnitude to [0, 255] uint8.
    """

    _KX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    _KY = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    def __init__(self, normalise: bool = True) -> None:
        self._norm = normalise
        log.debug("PrewittFilter | initialised")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR or grayscale image.

        Returns:
            Gradient magnitude image.
        """
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f    = gray.astype(np.float32)
        gx   = cv2.filter2D(f, -1, self._KX)
        gy   = cv2.filter2D(f, -1, self._KY)
        mag  = np.sqrt(gx ** 2 + gy ** 2)
        if self._norm:
            return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return mag
