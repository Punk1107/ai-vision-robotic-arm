"""
signal.py — Sensor Signal Filters (Category 1)
===============================================
Production-grade signal conditioning filters for:
  - IMU (accelerometer / gyroscope)
  - Motor encoders
  - Ultrasonic distance sensors
  - Current sensors
  - Motor feedback signals

All filters are stateful, discrete-time, and designed for real-time
embedded/PC pipelines at typical robotics sample rates (50–1000 Hz).

Mathematical conventions
------------------------
  x[n]  = current raw input sample
  y[n]  = current filtered output sample
  α      = smoothing coefficient (0 < α ≤ 1)
  fc     = cutoff frequency (Hz)
  fs     = sample rate (Hz)
  ωc     = 2π·fc  (rad/s)
  τ      = 1/ωc   (time constant, seconds)
  dt     = 1/fs   (sample period, seconds)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger("filters.signal")


# =============================================================================
# 1.1  Low-Pass Filter (RC / first-order IIR)
# =============================================================================

class LowPassFilter:
    """
    First-order discrete-time Low-Pass Filter (IIR).

    Transfer function (bilinear transform):
        y[n] = α·x[n] + (1−α)·y[n−1]

    where:
        α = dt / (τ + dt) = dt·ωc / (1 + dt·ωc)
        τ = 1 / (2π·fc)   (time constant)

    Applications in this project:
        - IMU accelerometer / gyroscope noise reduction
        - Encoder velocity smoothing
        - Ultrasonic distance smoothing
        - Anti-noise on PID derivative term (see DerivativeLPF)

    Args:
        cutoff_hz: Cutoff frequency (Hz).
        sample_rate_hz: Input sample rate (Hz).
        initial_value: Optional initial state (avoids step at t=0).

    Example::

        lpf = LowPassFilter(cutoff_hz=10.0, sample_rate_hz=200.0)
        for raw in imu_stream:
            filtered = lpf.update(raw)
    """

    def __init__(
        self,
        cutoff_hz:      float,
        sample_rate_hz: float,
        initial_value:  float = 0.0,
    ) -> None:
        if cutoff_hz <= 0 or sample_rate_hz <= 0:
            raise ValueError("cutoff_hz and sample_rate_hz must be positive.")
        if cutoff_hz >= sample_rate_hz / 2:
            raise ValueError(
                f"cutoff_hz ({cutoff_hz}) must be < Nyquist = {sample_rate_hz/2:.1f} Hz"
            )

        self._dt = 1.0 / sample_rate_hz
        tau      = 1.0 / (2 * math.pi * cutoff_hz)
        self._alpha = self._dt / (tau + self._dt)

        self._y = initial_value
        self._cutoff_hz = cutoff_hz
        log.debug(
            f"LowPassFilter | fc={cutoff_hz:.2f}Hz | fs={sample_rate_hz:.1f}Hz | "
            f"α={self._alpha:.4f}"
        )

    def update(self, x: float) -> float:
        """Process one sample and return the filtered output."""
        self._y = self._alpha * x + (1.0 - self._alpha) * self._y
        return self._y

    def update_vector(self, x: np.ndarray) -> np.ndarray:
        """Process a multi-channel sample (e.g. [ax, ay, az])."""
        if not hasattr(self, "_y_vec"):
            self._y_vec = x.copy()
        self._y_vec = self._alpha * x + (1.0 - self._alpha) * self._y_vec
        return self._y_vec.copy()

    def reset(self, value: float = 0.0) -> None:
        """Reset filter state."""
        self._y = value

    @property
    def output(self) -> float:
        return self._y

    @property
    def cutoff_hz(self) -> float:
        return self._cutoff_hz


# =============================================================================
# 1.2  Exponential Moving Average (EMA)
# =============================================================================

class EMAFilter:
    """
    Exponential Moving Average (EMA) filter.

    Equation:
        y[n] = α·x[n] + (1−α)·y[n−1]

    The EMA is mathematically identical to a first-order IIR LPF but
    parameterised directly by the smoothing factor α instead of a cutoff
    frequency — making it the preferred choice for embedded code where
    the sample rate may be irregular or unknown.

    Computational cost:
        1 multiply + 1 add + 1 subtract per sample — extremely lightweight.

    Applications in this project:
        - Smooth bounding-box position between YOLO frames
        - Smooth detection confidence scores
        - Smooth encoder velocity estimates
        - Any lightweight continuous smoothing need

    Args:
        alpha: Smoothing factor (0 < α ≤ 1).
               Larger → faster tracking, less smoothing.
               Smaller → more smoothing, more lag.
               Rule of thumb: α = 2/(N+1) for an N-sample window equivalent.
        initial_value: Initial state (default 0.0).
    """

    def __init__(self, alpha: float, initial_value: float = 0.0) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._y     = initial_value

    def update(self, x: float) -> float:
        """y[n] = α·x[n] + (1−α)·y[n−1]"""
        self._y = self._alpha * x + (1.0 - self._alpha) * self._y
        return self._y

    def update_vector(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_y_vec"):
            self._y_vec = x.astype(float).copy()
        self._y_vec = self._alpha * x + (1.0 - self._alpha) * self._y_vec
        return self._y_vec.copy()

    def reset(self, value: float = 0.0) -> None:
        self._y = value

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {value}")
        self._alpha = value

    @property
    def output(self) -> float:
        return self._y

    @classmethod
    def from_window(cls, window: int, initial_value: float = 0.0) -> "EMAFilter":
        """Construct from an equivalent simple-moving-average window length."""
        return cls(alpha=2.0 / (window + 1), initial_value=initial_value)

    @classmethod
    def from_cutoff(
        cls,
        cutoff_hz: float,
        sample_rate_hz: float,
        initial_value: float = 0.0,
    ) -> "EMAFilter":
        """Construct from cutoff frequency (identical to LowPassFilter)."""
        dt  = 1.0 / sample_rate_hz
        tau = 1.0 / (2 * math.pi * cutoff_hz)
        return cls(alpha=dt / (tau + dt), initial_value=initial_value)


# =============================================================================
# 1.3  Butterworth Low-Pass Filter (2nd order IIR)
# =============================================================================

class ButterworthLowPassFilter:
    """
    2nd-order Butterworth Low-Pass Filter (maximally flat passband).

    The Butterworth filter has the flattest possible frequency response in
    the passband — preferred over the simple RC filter when frequency
    accuracy matters (motor control, precision sensing).

    Implementation:
        Direct Form II Transposed (numerically stable for embedded use).

    Coefficients are derived analytically via the bilinear transform:
        ωc_pre = (2/T)·tan(ωc·T/2)   (pre-warping)
        b0, b1, b2, a1, a2  computed from Butterworth pole positions

    Applications in this project:
        - Motor current sensor smoothing
        - High-quality IMU conditioning before Kalman
        - Anti-aliasing before ADC decimation

    Args:
        cutoff_hz:      Filter cutoff frequency (Hz).
        sample_rate_hz: Input sample rate (Hz).

    Example::

        bwf = ButterworthLowPassFilter(cutoff_hz=20.0, sample_rate_hz=1000.0)
        for sample in motor_current_stream:
            smoothed = bwf.update(sample)
    """

    def __init__(self, cutoff_hz: float, sample_rate_hz: float) -> None:
        if cutoff_hz >= sample_rate_hz / 2:
            raise ValueError(
                f"cutoff_hz ({cutoff_hz}) must be < Nyquist = {sample_rate_hz/2:.1f} Hz"
            )

        # Bilinear transform with pre-warping
        fs   = sample_rate_hz
        fc   = cutoff_hz
        T    = 1.0 / fs
        wc   = 2 * math.pi * fc

        # Pre-warped analogue cutoff
        wc_a = (2.0 / T) * math.tan(wc * T / 2.0)

        # Butterworth 2nd-order poles: s = wc·exp(j·π(2k+n+1)/(2n)), k=0,1
        # Both poles: s = wc_a · exp(±j·3π/4) → Q = 1/√2 (Butterworth)
        # Normalised by wc_a → standard form denominator
        k  = wc_a * T / 2.0
        k2 = k * k

        # Bilinear transform denominator/numerator coefficients
        denom = 1.0 + math.sqrt(2.0) * k + k2
        self._b0 = k2 / denom
        self._b1 = 2.0 * self._b0
        self._b2 = self._b0
        self._a1 = (2.0 * (k2 - 1.0)) / denom
        self._a2 = (1.0 - math.sqrt(2.0) * k + k2) / denom

        # Direct Form II state
        self._w1 = 0.0
        self._w2 = 0.0

        log.debug(
            f"ButterworthLPF(2) | fc={fc:.2f}Hz | fs={fs:.1f}Hz | "
            f"b=[{self._b0:.5f},{self._b1:.5f},{self._b2:.5f}] | "
            f"a=[{self._a1:.5f},{self._a2:.5f}]"
        )

    def update(self, x: float) -> float:
        """Direct Form II transposed biquad step."""
        w0 = x - self._a1 * self._w1 - self._a2 * self._w2
        y  = self._b0 * w0 + self._b1 * self._w1 + self._b2 * self._w2
        self._w2 = self._w1
        self._w1 = w0
        return y

    def reset(self) -> None:
        self._w1 = 0.0
        self._w2 = 0.0


# =============================================================================
# 1.4  Notch Filter (IIR Band-Reject)
# =============================================================================

class NotchFilter:
    """
    2nd-order IIR Notch (Band-Reject) Filter.

    Attenuates a specific frequency while passing all others.
    Uses a slightly modified biquad notch design (constant 0 dB gain at DC).

    Transfer function (z-domain):
        H(z) = (z² − 2cos(ω₀)·z + 1) / (z² − 2r·cos(ω₀)·z + r²)

    where:
        ω₀ = 2π·f_notch / fs   (normalised notch frequency)
        r  = 1 − bandwidth_factor   (pole radius, 0 < r < 1)
              A narrower notch means r closer to 1.

    Applications in this project:
        - Remove 50/60 Hz power-line interference from current sensors
        - Reject PWM switching noise (typically 1–20 kHz)
        - Cancel structural resonance that falls in a narrow band

    Args:
        notch_hz:         Centre frequency to reject (Hz).
        sample_rate_hz:   Input sample rate (Hz).
        bandwidth_hz:     Notch bandwidth (Hz).  Wider → deeper but broader cut.
    """

    def __init__(
        self,
        notch_hz:       float,
        sample_rate_hz: float,
        bandwidth_hz:   float = 2.0,
    ) -> None:
        w0  = 2 * math.pi * notch_hz / sample_rate_hz
        bw  = 2 * math.pi * bandwidth_hz / sample_rate_hz
        r   = 1.0 - bw / 2.0
        r   = max(0.0, min(0.9999, r))

        cos_w0   = math.cos(w0)
        self._b0 =  1.0
        self._b1 = -2.0 * cos_w0
        self._b2 =  1.0
        self._a1 = -2.0 * r * cos_w0
        self._a2 =  r * r

        # Normalise so DC gain = 1
        gain_dc  = (self._b0 + self._b1 + self._b2) / (1.0 + self._a1 + self._a2)
        self._b0 /= gain_dc
        self._b1 /= gain_dc
        self._b2 /= gain_dc

        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0

        log.debug(
            f"NotchFilter | f_notch={notch_hz:.1f}Hz | bw={bandwidth_hz:.1f}Hz | "
            f"fs={sample_rate_hz:.1f}Hz | r={r:.4f}"
        )

    def update(self, x: float) -> float:
        """Direct Form I biquad update."""
        y = (self._b0 * x + self._b1 * self._x1 + self._b2 * self._x2
             - self._a1 * self._y1 - self._a2 * self._y2)
        self._x2, self._x1 = self._x1, x
        self._y2, self._y1 = self._y1, y
        return y

    def reset(self) -> None:
        self._x1 = self._x2 = self._y1 = self._y2 = 0.0


# =============================================================================
# 1.5  Band-Pass Filter (2nd order IIR)
# =============================================================================

class BandPassFilter:
    """
    2nd-order IIR Band-Pass Filter.

    Passes frequencies in the range [f_low, f_high] and attenuates others.
    Implemented as a cascade of high-pass then low-pass biquad stages.

    Applications in this project:
        - Vibration analysis (arm resonance band isolation)
        - Communication signal demodulation
        - Isolating a specific motor harmonic

    Args:
        low_hz:         Lower cutoff frequency (Hz).
        high_hz:        Upper cutoff frequency (Hz).
        sample_rate_hz: Input sample rate (Hz).
    """

    def __init__(
        self,
        low_hz:         float,
        high_hz:        float,
        sample_rate_hz: float,
    ) -> None:
        if low_hz >= high_hz:
            raise ValueError("low_hz must be < high_hz")

        # Cascade: High-pass (fc=low_hz) → Low-pass (fc=high_hz)
        self._lpf = ButterworthLowPassFilter(high_hz, sample_rate_hz)

        # 2nd-order Butterworth High-Pass via spectral inversion:
        # HPF coefficients from HPF bilinear transform
        fs  = sample_rate_hz
        T   = 1.0 / fs
        wc  = 2 * math.pi * low_hz
        wc_a = (2.0 / T) * math.tan(wc * T / 2.0)
        k   = wc_a * T / 2.0
        k2  = k * k
        denom   = 1.0 + math.sqrt(2.0) * k + k2
        self._hb0 = 1.0 / denom
        self._hb1 = -2.0 * self._hb0
        self._hb2 = self._hb0
        self._ha1 = (2.0 * (k2 - 1.0)) / denom
        self._ha2 = (1.0 - math.sqrt(2.0) * k + k2) / denom
        self._hw1 = 0.0
        self._hw2 = 0.0

        log.debug(
            f"BandPassFilter | [{low_hz:.1f}–{high_hz:.1f}]Hz | fs={fs:.1f}Hz"
        )

    def update(self, x: float) -> float:
        # High-pass stage (Direct Form II)
        w0_hp = x - self._ha1 * self._hw1 - self._ha2 * self._hw2
        hp    = self._hb0 * w0_hp + self._hb1 * self._hw1 + self._hb2 * self._hw2
        self._hw2 = self._hw1
        self._hw1 = w0_hp
        # Low-pass stage
        return self._lpf.update(hp)

    def reset(self) -> None:
        self._hw1 = 0.0
        self._hw2 = 0.0
        self._lpf.reset()


# =============================================================================
# 1.6  Anti-Aliasing Filter
# =============================================================================

class AntiAliasingFilter:
    """
    Anti-Aliasing Low-Pass Filter — must be applied BEFORE downsampling / ADC.

    Prevents high-frequency aliasing by ensuring signal content above the
    Nyquist frequency (fs_out / 2) is attenuated before the sample rate
    is reduced.

    Pipeline:
        Raw sensor → AntiAliasingFilter → ADC / downsampler → DSP

    This is a 4th-order Butterworth implemented as two cascaded 2nd-order
    biquad sections for improved stopband attenuation and numerical stability.

    Args:
        output_rate_hz: Desired output (decimated) sample rate (Hz).
                        The cutoff is automatically set to 40 % of output Nyquist
                        to provide a safe guard band.
        input_rate_hz:  Input sample rate before decimation (Hz).
        cutoff_ratio:   Cutoff as fraction of output Nyquist (default 0.4).
    """

    def __init__(
        self,
        output_rate_hz: float,
        input_rate_hz:  float,
        cutoff_ratio:   float = 0.40,
    ) -> None:
        nyquist_out = output_rate_hz / 2.0
        fc          = nyquist_out * cutoff_ratio
        # Two cascaded 2nd-order Butterworth sections = 4th-order overall
        self._stage1 = ButterworthLowPassFilter(fc, input_rate_hz)
        self._stage2 = ButterworthLowPassFilter(fc, input_rate_hz)
        self._ratio  = int(max(1, round(input_rate_hz / output_rate_hz)))
        self._n      = 0
        log.info(
            f"AntiAliasingFilter (4th-order) | fc={fc:.2f}Hz | "
            f"in={input_rate_hz:.0f}Hz → out={output_rate_hz:.0f}Hz | "
            f"ratio={self._ratio}"
        )

    def update(self, x: float) -> Optional[float]:
        """
        Filter one input sample.

        Returns a decimated output sample every `ratio` input samples,
        otherwise returns None (use as: ``if (y := aaf.update(x)) is not None``).
        """
        y = self._stage2.update(self._stage1.update(x))
        self._n += 1
        if self._n >= self._ratio:
            self._n = 0
            return y
        return None

    def reset(self) -> None:
        self._stage1.reset()
        self._stage2.reset()
        self._n = 0
