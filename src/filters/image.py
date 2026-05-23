"""
image.py — Camera / Image Preprocessing Filters (Category 3)
=============================================================
Image-domain filters applied to camera frames before AI inference.

  3.1  Gaussian Filter         — blur / denoise / preprocessing
  3.2  Median Filter           — salt-and-pepper noise removal
  3.3  Bilateral Filter        — edge-preserving denoise
  3.4  Wiener Filter           — statistical denoising / deblurring
  3.5  High-Pass Filter        — sharpening / edge enhancement
  3.6  CLAHE Filter            — adaptive contrast enhancement (bonus)

All filters operate on NumPy BGR images (OpenCV convention) and are
designed to be composable in a preprocessing chain:

    frame = Gaussian.apply(frame)
    frame = Bilateral.apply(frame)
    frame → YOLO inference

Mathematical highlights
-----------------------
  Gaussian kernel:
    G(x,y) = 1/(2π σ²) · exp(−(x²+y²)/(2σ²))

  Bilateral:
    BF[I](p) = Σ_q G_σs(||p-q||) · G_σr(|I(p)−I(q)|) · I(q)
    where G_σs = spatial kernel, G_σr = range (colour) kernel

  Wiener (frequency domain):
    Ŵ(u,v) = H*(u,v)·P_s(u,v) / (|H(u,v)|²·P_s(u,v) + P_n(u,v))
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("filters.image")


# =============================================================================
# 3.1  Gaussian Filter
# =============================================================================

class GaussianFilter:
    """
    Gaussian Blur / Denoise filter.

    Convolves the image with a Gaussian kernel to reduce high-frequency
    noise while preserving low-frequency structure.

    G(x,y) = (1 / 2πσ²) · exp(−(x²+y²) / 2σ²)

    The kernel size and σ are related: for a ksize×ksize kernel,
    a reasonable σ ≈ 0.3·((ksize−1)·0.5−1) + 0.8  (OpenCV rule).

    Uses in this project:
        - Pre-processing before YOLO detection (reduces spurious detections)
        - Before Canny edge detection (Canny internally applies Gaussian,
          but a pre-blur with a larger σ can improve results on noisy feeds)
        - First stage of Wiener / morphological pipelines

    Args:
        ksize_or_sigma:  If int → kernel size (must be odd ≥ 3).
                         If float → σ (kernel size auto-determined).
        sigma:           Gaussian standard deviation.  If ksize is provided,
                         sigma is auto-computed from ksize.
        border_type:     OpenCV border padding type.
    """

    def __init__(
        self,
        ksize_or_sigma: int | float = 5,
        sigma:          Optional[float] = None,
        border_type:    int = cv2.BORDER_REFLECT_101,
    ) -> None:
        if isinstance(ksize_or_sigma, float):
            # σ given → auto ksize
            self._sigma = ksize_or_sigma
            # Rule: ksize = 2·ceil(3σ)+1 (covers 99.7% of Gaussian)
            k = int(2 * math.ceil(3 * self._sigma) + 1)
            self._ksize = k if k % 2 == 1 else k + 1
        else:
            self._ksize = int(ksize_or_sigma)
            if self._ksize % 2 == 0:
                self._ksize += 1  # ensure odd
            self._sigma = 0.0   # let OpenCV compute from ksize
        self._sigma_eff = sigma or self._sigma
        self._border    = border_type
        log.debug(f"GaussianFilter | ksize={self._ksize} | σ={self._sigma_eff:.2f}")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR or grayscale NumPy array (uint8 or float32).

        Returns:
            Gaussian-blurred frame (same type and shape).
        """
        return cv2.GaussianBlur(
            frame,
            (self._ksize, self._ksize),
            self._sigma_eff,
            borderType=self._border,
        )

    @staticmethod
    def kernel(ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Return the Gaussian kernel matrix (for visualisation / convolution)."""
        k = cv2.getGaussianKernel(ksize, sigma)
        return k @ k.T


import math  # noqa


# =============================================================================
# 3.2  Median Filter
# =============================================================================

class MedianFilter:
    """
    Median Filter — optimal for salt-and-pepper noise.

    Replaces each pixel with the median of its (ksize × ksize) neighbourhood.
    Median is a non-linear operation that preserves edges better than a
    Gaussian blur for impulse noise (dead pixels, camera artifacts).

    Uses in this project:
        - Depth map denoising (depth cameras produce salt-and-pepper artifacts)
        - Camera artifact removal before colour segmentation
        - Pre-processing for morphological operations

    Args:
        ksize: Neighbourhood size (must be odd).  Larger → more smoothing
               but slower.  Typical: 3 or 5.
    """

    def __init__(self, ksize: int = 5) -> None:
        if ksize % 2 == 0:
            ksize += 1
        self._k = ksize
        log.debug(f"MedianFilter | ksize={ksize}")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply median blur.  Works on both uint8 and float32 images."""
        if frame.dtype == np.float32:
            # cv2.medianBlur doesn't support float32 — normalise to uint8 first
            norm   = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            result = cv2.medianBlur(norm, self._k)
            return result.astype(np.float32) / 255.0 * (frame.max() - frame.min()) + frame.min()
        return cv2.medianBlur(frame, self._k)


# =============================================================================
# 3.3  Bilateral Filter
# =============================================================================

class BilateralFilter:
    """
    Bilateral Filter — edge-preserving denoise.

    Unlike Gaussian blur (which blurs edges), the bilateral filter
    preserves sharp edges by weighting nearby pixels by both spatial
    proximity AND colour similarity.

    BF[I](p) = (1/Wp) · Σ_q G_σd(||p−q||) · G_σr(|I(p)−I(q)|) · I(q)

    where:
        G_σd = Gaussian in spatial domain (distance)
        G_σr = Gaussian in colour (intensity) domain

    Tuning guide:
        d         — diameter of pixel neighbourhood (9 is good default)
        sigma_color — larger → more distant colours are mixed (wider colour range)
        sigma_space — larger → far pixels have more influence

    Relationship: for smooth regions, bilateral ≈ Gaussian.
                  At edges (large ΔI), range kernel → 0, edge preserved.

    Uses in this project:
        - Depth map smoothing before coordinate mapping (preserves object edges)
        - Pre-processing segmentation masks
        - Noise reduction while keeping detection bounding-box boundaries sharp

    Args:
        d:            Diameter of pixel neighbourhood (-1 = auto from sigma_space).
        sigma_color:  Filter sigma in colour space.  Typical: 75–150.
        sigma_space:  Filter sigma in spatial space.  Typical: 75–150.
    """

    def __init__(
        self,
        d:            int   = 9,
        sigma_color:  float = 75.0,
        sigma_space:  float = 75.0,
    ) -> None:
        self._d   = d
        self._sc  = sigma_color
        self._ss  = sigma_space
        log.debug(f"BilateralFilter | d={d} | σ_color={sigma_color} | σ_space={sigma_space}")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter.

        NOTE: Bilateral is significantly slower than Gaussian for large d.
        Use d=5 or d=9 for real-time performance.  d=15 for offline/QC use.
        """
        return cv2.bilateralFilter(frame, self._d, self._sc, self._ss)


# =============================================================================
# 3.4  Wiener Filter (Statistical Denoising)
# =============================================================================

class WienerFilter:
    """
    Wiener Filter — optimal linear filter for Gaussian noise / deblurring.

    Operates in the frequency (Fourier) domain:

        Ŵ(u,v) = H*(u,v) · P_s / (|H(u,v)|² · P_s + P_n)

    where:
        H(u,v) = degradation function (Gaussian blur model)
        P_s    = signal power spectral density
        P_n    = noise power spectral density (= noise_var)
        *      = complex conjugate

    For simple noise-only denoising (no blur), H ≡ 1 and the filter
    reduces to the local Wiener filter in spatial domain.

    Implementation:
        Uses scipy.ndimage.gaussian_filter and the local statistics (mean,
        variance) in a sliding window to compute the optimal filter weights.
        This is the "local Wiener" variant — more computationally feasible
        than the full frequency-domain version.

    Uses in this project:
        - High-quality image restoration for QC inspection
        - Depth map restoration (preserves fine depth details)
        - Research-grade preprocessing for publication figures

    Args:
        noise_var:  Estimated noise variance.  If None, estimated from image.
        ksize:      Neighbourhood size for local statistics computation.
    """

    def __init__(
        self,
        noise_var: Optional[float] = None,
        ksize:     int = 5,
    ) -> None:
        self._noise_var = noise_var
        self._ksize     = ksize
        log.debug(f"WienerFilter | noise_var={noise_var} | ksize={ksize}")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply local Wiener filter.

        Args:
            frame: Grayscale uint8 image.  If BGR, converted internally.

        Returns:
            Denoised image as uint8.
        """
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f    = gray.astype(np.float64)

        from scipy.ndimage import uniform_filter

        # Local mean and variance
        local_mean = uniform_filter(f, size=self._ksize)
        local_sq   = uniform_filter(f ** 2, size=self._ksize)
        local_var  = local_sq - local_mean ** 2

        # Estimate noise variance from image if not provided
        noise_var = self._noise_var
        if noise_var is None:
            noise_var = float(np.mean(local_var))

        # Wiener formula: out = μ + max(0, σ²−σ²_n)/(σ²) · (x − μ)
        # Clamp local_var to avoid division by zero
        denom  = np.maximum(local_var, 1e-6)
        weight = np.maximum(0.0, local_var - noise_var) / denom
        out    = local_mean + weight * (f - local_mean)
        out    = np.clip(out, 0, 255).astype(np.uint8)

        if len(frame.shape) == 3:
            # Apply per-channel for colour images
            channels = [
                self.apply(frame[:, :, c]) for c in range(frame.shape[2])
            ]
            return np.stack(channels, axis=2)
        return out


# =============================================================================
# 3.5  High-Pass Filter (Sharpening)
# =============================================================================

class HighPassFilter:
    """
    High-Pass Filter — image sharpening and detail enhancement.

    The high-pass is computed as the complement of the low-pass:
        HPF = Original − LPF(Original)

    This isolates high-frequency information (edges, fine texture).
    Adding it back to the original amplifies detail:
        Sharpened = Original + strength × HPF

    Uses in this project:
        - Sharpening camera feed before YOLO for small-object detection
        - Edge enhancement before edge detection pipelines
        - QC texture analysis preprocessing

    Args:
        ksize:    Blur kernel size for the LPF component.  Larger → stronger HPF.
        strength: How much to amplify the high-frequency component [0, ∞).
                  0.0 = no effect, 1.0 = standard unsharp mask, 2.0 = aggressive.
    """

    def __init__(self, ksize: int = 5, strength: float = 1.5) -> None:
        self._ksize    = ksize if ksize % 2 == 1 else ksize + 1
        self._strength = strength
        log.debug(f"HighPassFilter | ksize={self._ksize} | strength={strength}")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply high-pass sharpening (unsharp masking)."""
        blurred = cv2.GaussianBlur(frame, (self._ksize, self._ksize), 0)
        hpf     = cv2.subtract(frame, blurred)   # high frequencies
        # Unsharp mask: sharpened = original + strength × HPF
        sharpened = cv2.addWeighted(frame, 1.0, hpf, self._strength, 0)
        return sharpened

    def extract_hpf(self, frame: np.ndarray) -> np.ndarray:
        """Return only the high-frequency component (edges, details)."""
        blurred = cv2.GaussianBlur(frame, (self._ksize, self._ksize), 0)
        return cv2.subtract(frame, blurred)


# =============================================================================
# 3.6  CLAHE Filter (Adaptive Contrast Enhancement)
# =============================================================================

class CLAHEFilter:
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization.

    Enhances local contrast without blowing out bright regions (unlike
    global histogram equalization).  Divides the image into small tiles
    and applies HE in each, with contrast limiting to suppress noise
    amplification.

    This is already used in preprocess.py for camera frames, but is
    exposed here as a standalone composable filter for:
        - Depth map contrast enhancement
        - Nighttime / low-light camera feeds
        - QC inspection preprocessing (reveal surface defects)

    Args:
        clip_limit:   Contrast amplification limit per tile.  2.0 is standard.
        tile_size:    Grid size for local HE tiles (width, height).
        channel:      "L" = apply only to luminance (Lab colour space, default).
                      "gray" = apply to single-channel image.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_size:  Tuple[int, int] = (8, 8),
        channel:    str = "L",
    ) -> None:
        self._clahe   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        self._channel = channel
        log.debug(f"CLAHEFilter | clip={clip_limit} | tile={tile_size}")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR frame or grayscale image.

        Returns:
            Contrast-enhanced frame (same shape/type as input).
        """
        if len(frame.shape) == 2:
            # Grayscale
            return self._clahe.apply(frame)

        # Colour: work in LAB colour space on L channel only
        lab        = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        lab_out    = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
