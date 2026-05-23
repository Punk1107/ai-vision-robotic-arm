"""
pipeline.py — Production Filter Pipeline (Category 7 integration)
=================================================================
The FilterPipeline assembles all filter stages into one coherent
production-style system that maps to the final system architecture:

    Sensors / Camera
    ↓
    Signal Conditioning      → filters.signal (LPF, Butterworth, EMA, Anti-aliasing)
    ↓
    Image Preprocessing      → filters.image  (Gaussian, Bilateral, Median, CLAHE)
    ↓
    Feature Extraction        → filters.edge + filters.morphology
    ↓
    AI / Object Detection     → YOLO / CNN (external, passed through)
    ↓
    Tracking & Sensor Fusion → filters.motion + tracker.py (Kalman, Complementary)
    ↓
    Control System           → filters.control (DerivativeLPF, PID)
    ↓
    Pulse Shaping / PWM      → robotics.shaping (ZV, ZVD, EI, Adaptive)
    ↓
    Motor Driver → Robot Motion

This file provides:
    1. FilterPipeline  — unified interface for all image preprocessing stages
    2. SensorFusionPipeline — IMU + encoder signal conditioning
    3. ControlFilterBundle  — all PID-related filters bundled for visual_servo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from src.filters.signal import (
    LowPassFilter, EMAFilter, ButterworthLowPassFilter, AntiAliasingFilter
)
from src.filters.control import (
    DerivativeLPF, ComplementaryFilter, KalmanFilter1D
)
from src.filters.image import (
    GaussianFilter, BilateralFilter, MedianFilter, CLAHEFilter, HighPassFilter
)
from src.filters.edge import CannyEdgeDetector, SobelFilter
from src.filters.morphology import MorphologicalProcessor
from src.filters.motion import OpticalFlowSmoother, KalmanTracker2D
from src.utils.logger import get_logger

log = get_logger("filters.pipeline")


# =============================================================================
# Image Preprocessing Pipeline
# =============================================================================

@dataclass
class ImagePreprocessConfig:
    """Configuration for the image preprocessing pipeline stages."""
    # Stage 1: Noise reduction
    use_gaussian:     bool  = True
    gaussian_ksize:   int   = 5

    # Stage 2: Edge-preserving denoise (mutually exclusive with median)
    use_bilateral:    bool  = False   # heavier, use for QC
    bilateral_d:      int   = 9
    bilateral_sc:     float = 75.0
    bilateral_ss:     float = 75.0

    use_median:       bool  = False   # for salt-and-pepper noise
    median_ksize:     int   = 5

    # Stage 3: Contrast enhancement
    use_clahe:        bool  = False
    clahe_clip:       float = 2.0

    # Stage 4: Sharpening (for small-object detection improvement)
    use_sharpen:      bool  = False
    sharpen_strength: float = 1.5
    sharpen_ksize:    int   = 5

    # Edge detection (optional — for visualisation or contour preprocessing)
    use_canny:        bool  = False
    canny_low:        float = 50.0
    canny_high:       float = 150.0


class FilterPipeline:
    """
    Production image preprocessing pipeline.

    Composable filter stages applied in sequence before YOLO/CNN inference.

    The default configuration is tuned for this project's pick-and-place
    scenario (640×480 or 1280×720 USB camera, 30fps, indoor lighting).

    Usage::

        pipeline = FilterPipeline()
        # Or customise:
        cfg = ImagePreprocessConfig(use_bilateral=True, use_clahe=True)
        pipeline = FilterPipeline(cfg)

        for raw_frame in camera:
            processed = pipeline.process(raw_frame)
            detections = yolo.detect(processed)
    """

    def __init__(
        self,
        config: Optional[ImagePreprocessConfig] = None,
    ) -> None:
        self._cfg = config or ImagePreprocessConfig()
        c = self._cfg

        # Instantiate only enabled stages to avoid overhead
        self._gaussian   = GaussianFilter(c.gaussian_ksize) if c.use_gaussian else None
        self._bilateral  = BilateralFilter(c.bilateral_d, c.bilateral_sc, c.bilateral_ss) if c.use_bilateral else None
        self._median     = MedianFilter(c.median_ksize) if c.use_median else None
        self._clahe      = CLAHEFilter(c.clahe_clip) if c.use_clahe else None
        self._sharpen    = HighPassFilter(c.sharpen_ksize, c.sharpen_strength) if c.use_sharpen else None
        self._canny      = CannyEdgeDetector(c.canny_low, c.canny_high) if c.use_canny else None

        log.info(
            f"FilterPipeline initialised | "
            f"stages: [{', '.join(self._active_stages())}]"
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all enabled preprocessing stages.

        Args:
            frame: Raw BGR camera frame (uint8).

        Returns:
            Preprocessed BGR frame ready for YOLO inference.
        """
        out = frame

        if self._gaussian:
            out = self._gaussian.apply(out)

        if self._bilateral:
            out = self._bilateral.apply(out)
        elif self._median:
            out = self._median.apply(out)

        if self._clahe:
            out = self._clahe.apply(out)

        if self._sharpen:
            out = self._sharpen.apply(out)

        return out

    def process_and_detect_edges(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess frame and optionally compute Canny edges.

        Returns:
            (preprocessed_frame, edge_map_or_None)
        """
        processed = self.process(frame)
        edges     = self._canny.apply(processed) if self._canny else None
        return processed, edges

    def _active_stages(self):
        stages = []
        if self._gaussian:   stages.append("gaussian")
        if self._bilateral:  stages.append("bilateral")
        if self._median:     stages.append("median")
        if self._clahe:      stages.append("clahe")
        if self._sharpen:    stages.append("sharpen")
        if self._canny:      stages.append("canny")
        return stages or ["passthrough"]


# =============================================================================
# Sensor Fusion Pipeline (IMU + encoders → clean state estimates)
# =============================================================================

class SensorFusionPipeline:
    """
    Signal conditioning and fusion pipeline for IMU and encoder data.

    Pipeline per sensor:
        IMU Acc/Gyro: Butterworth LPF → EMA → Complementary Filter
        Encoder:      LPF → Kalman 1D

    This provides the clean (ωₙ, ζ) estimates fed to ai_estimator.py and
    the complementary filter angle fed to the control system.

    Args:
        imu_sample_rate_hz:  IMU sample rate (Hz).
        imu_cutoff_hz:       IMU LPF cutoff (Hz).  Typical: 20–50 Hz.
        encoder_rate_hz:     Encoder reading rate (Hz).
        encoder_cutoff_hz:   Encoder velocity LPF cutoff (Hz).

    Example::

        fusion = SensorFusionPipeline(imu_sample_rate_hz=200.0)
        # In serial-read callback:
        fusion.update_imu(ax, ay, az, gx, gy, gz)
        roll_deg  = fusion.roll
        pitch_deg = fusion.pitch
    """

    def __init__(
        self,
        imu_sample_rate_hz:   float = 200.0,
        imu_cutoff_hz:        float = 30.0,
        encoder_rate_hz:      float = 200.0,
        encoder_cutoff_hz:    float = 20.0,
        complementary_alpha:  float = 0.98,
    ) -> None:
        # IMU accelerometer LPF (per axis)
        self._acc_lpf_x = ButterworthLowPassFilter(imu_cutoff_hz, imu_sample_rate_hz)
        self._acc_lpf_y = ButterworthLowPassFilter(imu_cutoff_hz, imu_sample_rate_hz)
        self._acc_lpf_z = ButterworthLowPassFilter(imu_cutoff_hz, imu_sample_rate_hz)

        # IMU gyroscope LPF (per axis)
        self._gyro_lpf_x = ButterworthLowPassFilter(imu_cutoff_hz, imu_sample_rate_hz)
        self._gyro_lpf_y = ButterworthLowPassFilter(imu_cutoff_hz, imu_sample_rate_hz)
        self._gyro_lpf_z = ButterworthLowPassFilter(imu_cutoff_hz, imu_sample_rate_hz)

        # EMA for IMU magnitude
        self._acc_mag_ema = EMAFilter(alpha=0.3)

        # Complementary filter for orientation (roll + pitch)
        self._cf_roll  = ComplementaryFilter(
            alpha=complementary_alpha,
            sample_rate_hz=imu_sample_rate_hz,
            axis="roll",
        )
        self._cf_pitch = ComplementaryFilter(
            alpha=complementary_alpha,
            sample_rate_hz=imu_sample_rate_hz,
            axis="pitch",
        )

        # Encoder velocity Kalman (per joint — placeholder for 5 joints)
        self._enc_kf = [
            KalmanFilter1D(process_noise=0.01, measurement_noise=0.5)
            for _ in range(5)
        ]

        # Anti-aliasing filter for 50Hz output from 200Hz IMU
        self._imu_aa = AntiAliasingFilter(output_rate_hz=50.0, input_rate_hz=imu_sample_rate_hz)

        self._roll  = 0.0
        self._pitch = 0.0
        log.info(
            f"SensorFusionPipeline | fs_imu={imu_sample_rate_hz:.0f}Hz | "
            f"fc={imu_cutoff_hz:.1f}Hz | CF_α={complementary_alpha:.3f}"
        )

    def update_imu(
        self,
        ax: float, ay: float, az: float,
        gx: float, gy: float, gz: float,
        dt: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Process one IMU sample through the full conditioning pipeline.

        Args:
            ax, ay, az: Accelerometer readings (m/s²).
            gx, gy, gz: Gyroscope readings (rad/s).
            dt:         Elapsed time since last sample (s).

        Returns:
            (roll_deg, pitch_deg) — filtered orientation angles.
        """
        import math

        # Stage 1: Butterworth LPF on raw IMU
        ax_f = self._acc_lpf_x.update(ax)
        ay_f = self._acc_lpf_y.update(ay)
        az_f = self._acc_lpf_z.update(az)
        gx_f = self._gyro_lpf_x.update(gx)
        gy_f = self._gyro_lpf_y.update(gy)

        # Stage 2: Compute accelerometer-derived angles
        acc_roll  =  math.degrees(math.atan2(ay_f, az_f))
        acc_pitch =  math.degrees(math.atan2(-ax_f, math.sqrt(ay_f**2 + az_f**2)))

        # Stage 3: Complementary filter (gyro + acc fusion)
        self._roll  = self._cf_roll.update(
            acc_angle_deg=acc_roll,
            gyro_rate_dps=math.degrees(gx_f),
            dt=dt,
        )
        self._pitch = self._cf_pitch.update(
            acc_angle_deg=acc_pitch,
            gyro_rate_dps=math.degrees(gy_f),
            dt=dt,
        )

        # Stage 4: EMA on acc magnitude (for ResonanceEstimator input conditioning)
        acc_mag   = math.sqrt(ax_f**2 + ay_f**2 + az_f**2)
        self._acc_mag_ema.update(acc_mag)

        return self._roll, self._pitch

    def update_encoder_velocity(
        self,
        joint_idx: int,
        raw_velocity: float,
    ) -> float:
        """
        Filter encoder velocity for one joint using Kalman.

        Args:
            joint_idx:    Joint index (0=base, 1=shoulder, ..., 4=gripper).
            raw_velocity: Raw encoder velocity (deg/s).

        Returns:
            Filtered velocity estimate.
        """
        return self._enc_kf[joint_idx].step(raw_velocity)

    @property
    def roll(self) -> float:
        """Filtered roll angle (degrees)."""
        return self._roll

    @property
    def pitch(self) -> float:
        """Filtered pitch angle (degrees)."""
        return self._pitch

    @property
    def acc_magnitude_smooth(self) -> float:
        """EMA-smoothed accelerometer magnitude (m/s²)."""
        return self._acc_mag_ema.output


# =============================================================================
# Control Filter Bundle (for visual_servo.py PID)
# =============================================================================

@dataclass
class ControlFilterBundle:
    """
    Bundle of PID control filters for the visual servoing controller.

    Wraps DerivativeLPF for each servo axis (X, Y, Z/depth) and provides
    a clean interface to be used inside visual_servo.VisualServoController.

    These replace the raw finite-difference derivative in the existing _PID
    class and add proper bandwidth limiting.

    Usage::

        bundle = ControlFilterBundle.from_config(kp=0.0004, kd=0.0001)
        d_x = bundle.d_x.update(error_x, dt)
        d_y = bundle.d_y.update(error_y, dt)
        d_z = bundle.d_z.update(error_area, dt)
    """
    d_x: DerivativeLPF
    d_y: DerivativeLPF
    d_z: DerivativeLPF

    # Position EMA smoothers for bounding-box centre
    ema_cx: EMAFilter = field(default_factory=lambda: EMAFilter(0.5))
    ema_cy: EMAFilter = field(default_factory=lambda: EMAFilter(0.5))
    ema_area: EMAFilter = field(default_factory=lambda: EMAFilter(0.4))

    @classmethod
    def from_config(
        cls,
        kd:             float = 0.0001,
        cutoff_hz:      float = 10.0,
        sample_rate_hz: float = 30.0,
        ema_alpha:      float = 0.6,
    ) -> "ControlFilterBundle":
        """Construct with identical parameters for all three axes."""
        return cls(
            d_x     = DerivativeLPF(kd, cutoff_hz, sample_rate_hz),
            d_y     = DerivativeLPF(kd, cutoff_hz, sample_rate_hz),
            d_z     = DerivativeLPF(kd * 0.5, cutoff_hz, sample_rate_hz),
            ema_cx  = EMAFilter(ema_alpha),
            ema_cy  = EMAFilter(ema_alpha),
            ema_area = EMAFilter(ema_alpha * 0.7),  # area smoother → more smoothing
        )

    def reset(self) -> None:
        """Reset all integrators (call at the start of each pick attempt)."""
        self.d_x.reset()
        self.d_y.reset()
        self.d_z.reset()
        self.ema_cx.reset()
        self.ema_cy.reset()
        self.ema_area.reset()

    def smooth_observation(
        self,
        cx: float,
        cy: float,
        area: float,
    ) -> Tuple[float, float, float]:
        """
        Apply EMA smoothing to bounding-box observations before PID.

        Returns:
            (smooth_cx, smooth_cy, smooth_area)
        """
        return (
            self.ema_cx.update(cx),
            self.ema_cy.update(cy),
            self.ema_area.update(area),
        )
