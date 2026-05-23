"""
control.py — PID / Control Filters (Category 2)
================================================
Filters specifically designed for closed-loop control systems:

  2.1  Derivative Low-Pass Filter (Dirty Derivative)   — critical for real PID
  2.2  Complementary Filter                            — IMU orientation fusion
  2.3  Kalman Filter 1D                                — optimal state estimation
  2.4  Adaptive Filter (LMS)                           — noise that changes over time

These filters are used directly inside the control pipeline:
  visual_servo.py (PID), ai_estimator.py (Kalman/EKF), and shaping.py.

Mathematical basis
------------------
  Derivative LPF:  D(s) = Kd·s / (τ·s + 1)
  Complementary:   angle = α·(angle + gyro·dt) + (1−α)·acc_angle
  Kalman:          Standard discrete linear KF (see full derivation below)
  LMS:             w[n+1] = w[n] + μ·e[n]·x[n]
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger("filters.control")


# =============================================================================
# 2.1  Derivative Low-Pass Filter (Dirty Derivative)
# =============================================================================

class DerivativeLPF:
    """
    Derivative term with built-in Low-Pass Filter — the "dirty derivative".

    A pure discrete derivative amplifies high-frequency noise aggressively:
        D[n] = Kd · (e[n] − e[n−1]) / dt   ← noisy, unusable in practice

    The dirty derivative filters the derivative before applying gain:
        D(s) = Kd · s / (τ·s + 1)

    Discretised via bilinear transform (Tustin):
        τ_d = Kd / (N · Kp)   where N is typically 5–20 (derivative filter coefficient)
        Or equivalently: set cutoff_hz to limit derivative bandwidth.

    This is MANDATORY for any real PID on a robotics system.

    Applications in this project:
        - PID D-term in visual_servo.py
        - Joint angle velocity estimation from encoder counts

    Args:
        kd:           Derivative gain.
        cutoff_hz:    Derivative term bandwidth limit (Hz).
                      Typical: 5–50 Hz.  Lower = smoother but slower response.
        sample_rate_hz: Loop rate (Hz).

    Example::

        pid_d = DerivativeLPF(kd=0.001, cutoff_hz=20.0, sample_rate_hz=30.0)
        # In control loop:
        d_out = pid_d.update(error, dt)
    """

    def __init__(
        self,
        kd:             float,
        cutoff_hz:      float,
        sample_rate_hz: float,
    ) -> None:
        self._kd  = kd
        self._dt  = 1.0 / sample_rate_hz
        tau       = 1.0 / (2 * math.pi * cutoff_hz)
        self._tau = tau

        # Bilinear-discretised coefficients for H(z) = Kd·s/(τs+1)
        # After bilinear transform with T=dt:
        #   b0 = 2Kd/(2τ+T),  b1 = -2Kd/(2τ+T)
        #   a0 = 1,            a1 = (T-2τ)/(T+2τ)
        T = self._dt
        denom   = 2.0 * tau + T
        self._b0 =  2.0 * kd / denom
        self._b1 = -2.0 * kd / denom
        self._a1 = (T - 2.0 * tau) / (T + 2.0 * tau)

        self._prev_e = 0.0
        self._prev_y = 0.0

        log.debug(
            f"DerivativeLPF | Kd={kd:.5f} | fc={cutoff_hz:.1f}Hz | "
            f"fs={sample_rate_hz:.1f}Hz | τ={tau:.4f}s"
        )

    def update(self, error: float, dt: Optional[float] = None) -> float:
        """
        Compute the filtered derivative output for one step.

        Args:
            error: Current PID error signal.
            dt:    Optional real elapsed time (overrides constructor dt).

        Returns:
            Filtered D-term output.
        """
        if dt is not None and dt > 1e-6:
            # Recalculate bilinear coefficients for variable dt
            T     = dt
            denom = 2.0 * self._tau + T
            b0 =  2.0 * self._kd / denom
            b1 = -2.0 * self._kd / denom
            a1 = (T - 2.0 * self._tau) / (T + 2.0 * self._tau)
        else:
            b0, b1, a1 = self._b0, self._b1, self._a1

        y = b0 * error + b1 * self._prev_e - a1 * self._prev_y
        self._prev_e = error
        self._prev_y = y
        return y

    def reset(self) -> None:
        self._prev_e = 0.0
        self._prev_y = 0.0


# =============================================================================
# 2.2  Complementary Filter (IMU Sensor Fusion)
# =============================================================================

class ComplementaryFilter:
    """
    Complementary Filter for IMU orientation estimation.

    Fuses accelerometer (accurate long-term, noisy short-term) and
    gyroscope (accurate short-term, drifts long-term) data:

        angle[n] = α · (angle[n−1] + gyro_rate · dt) + (1−α) · acc_angle

    Where:
        α ≈ τ/(τ + dt)   with τ >> dt for most of the weight on gyro
        Typical α = 0.95 – 0.99 for low-noise MEMS gyros

    This is the standard baseline for orientation estimation on embedded
    systems before full Kalman/Mahony filters are needed.

    Applications in this project:
        - Arm base tilt estimation (for gravity compensation)
        - IMU data preprocessing before sending to ai_estimator.py
        - Balancing robot or drone orientation (future extension)

    Args:
        alpha:          High-pass weight for gyroscope (0 < α < 1).
                        Typical: 0.96 – 0.98.
        sample_rate_hz: IMU sample rate (Hz).
        axis:           Which axis to filter ("roll", "pitch", or "yaw").
                        Yaw note: acc cannot measure yaw → magnetometer needed.

    Example::

        cf = ComplementaryFilter(alpha=0.98, sample_rate_hz=200.0)
        for sample in imu_stream:
            # acc_angle_deg = atan2(ay, az) in degrees
            # gyro_rate_dps = gx in degrees/second
            angle = cf.update(acc_angle_deg, gyro_rate_dps, dt=0.005)
    """

    def __init__(
        self,
        alpha:          float = 0.98,
        sample_rate_hz: float = 200.0,
        axis:           str   = "roll",
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        self._dt    = 1.0 / sample_rate_hz
        self._axis  = axis
        self._angle = 0.0
        log.info(
            f"ComplementaryFilter | axis={axis} | α={alpha:.3f} | "
            f"fs={sample_rate_hz:.1f}Hz"
        )

    def update(
        self,
        acc_angle_deg:  float,
        gyro_rate_dps:  float,
        dt:             Optional[float] = None,
    ) -> float:
        """
        Args:
            acc_angle_deg:  Angle from accelerometer (degrees).
            gyro_rate_dps:  Angular rate from gyroscope (degrees/second).
            dt:             Elapsed time (s).  Uses 1/fs if not provided.

        Returns:
            Fused angle estimate (degrees).
        """
        elapsed = dt if dt is not None else self._dt
        self._angle = (
            self._alpha * (self._angle + gyro_rate_dps * elapsed)
            + (1.0 - self._alpha) * acc_angle_deg
        )
        return self._angle

    def reset(self, angle_deg: float = 0.0) -> None:
        self._angle = angle_deg

    @property
    def angle(self) -> float:
        return self._angle


# =============================================================================
# 2.3  Kalman Filter 1D (Linear, Scalar State)
# =============================================================================

class KalmanFilter1D:
    """
    Scalar (1D) Linear Kalman Filter.

    State: x (scalar, e.g. position or angle)
    Measurement: z = x + noise

    Equations:
        Predict:
            x̂⁻[n] = A·x̂[n−1] + B·u[n]   (state transition)
            P⁻[n]  = A²·P[n−1] + Q         (covariance prediction)
        Update:
            K[n]   = P⁻[n] / (P⁻[n] + R)  (Kalman gain)
            x̂[n]  = x̂⁻[n] + K·(z − x̂⁻[n])
            P[n]   = (1 − K) · P⁻[n]

    Parameters:
        Q: Process noise variance   — how much does the true state change?
           Low Q → trust model,  High Q → trust measurements.
        R: Measurement noise variance — how noisy is the sensor?
           High R → trust model more.

    Applications in this project:
        - Single-joint angle filtering
        - Distance sensor noise reduction
        - Confidence score smoothing

    Note:
        For multi-dimensional state (position + velocity, 2D centroid),
        use the full CentroidTracker in tracker.py which implements the
        vector Kalman filter with state [x, y, vx, vy].
    """

    def __init__(
        self,
        process_noise:      float = 0.01,   # Q
        measurement_noise:  float = 1.0,    # R
        initial_estimate:   float = 0.0,
        initial_covariance: float = 1.0,
    ) -> None:
        self._Q  = process_noise
        self._R  = measurement_noise
        self._x  = initial_estimate
        self._P  = initial_covariance

    def predict(self, control_input: float = 0.0, A: float = 1.0, B: float = 0.0) -> float:
        """
        Prediction step.

        Args:
            control_input: Known control input u (optional).
            A: State transition coefficient (default 1.0 = constant model).
            B: Control-to-state coefficient (default 0.0 = no control input).

        Returns:
            Prior state estimate x̂⁻.
        """
        self._x = A * self._x + B * control_input
        self._P = A * A * self._P + self._Q
        return self._x

    def update(self, measurement: float) -> float:
        """
        Measurement update step.

        Args:
            measurement: New sensor reading z.

        Returns:
            Posterior state estimate x̂.
        """
        K       = self._P / (self._P + self._R)
        self._x = self._x + K * (measurement - self._x)
        self._P = (1.0 - K) * self._P
        return self._x

    def step(self, measurement: float) -> float:
        """Convenience: predict + update in one call (no control input)."""
        self.predict()
        return self.update(measurement)

    @property
    def estimate(self) -> float:
        return self._x

    @property
    def gain(self) -> float:
        """Current Kalman gain K (for diagnostics)."""
        return self._P / (self._P + self._R)

    def reset(self, value: float = 0.0) -> None:
        self._x = value
        self._P = 1.0

    def set_noise(self, Q: Optional[float] = None, R: Optional[float] = None) -> None:
        """Hot-swap noise parameters at runtime."""
        if Q is not None:
            self._Q = Q
        if R is not None:
            self._R = R


# =============================================================================
# 2.4  Adaptive Filter — LMS (Least Mean Squares)
# =============================================================================

class AdaptiveFilterLMS:
    """
    Adaptive FIR filter using the Least Mean Squares (LMS) algorithm.

    Unlike fixed-parameter filters, the LMS filter continuously updates its
    coefficients to minimise the mean-squared error between its output and
    a desired reference signal.  This allows it to track noise that changes
    over time (non-stationary environments).

    Algorithm:
        y[n]        = wᵀ[n] · x[n]      (filter output)
        e[n]        = d[n] − y[n]        (error vs desired signal)
        w[n+1]      = w[n] + μ·e[n]·x[n] (weight update)

    where:
        w[n]: filter tap weights (length = order)
        x[n]: input delay-line vector [x[n], x[n-1], ..., x[n-order+1]]
        d[n]: desired signal (reference / clean signal)
        μ:    step size (learning rate) — critical tuning parameter

    Stability condition: 0 < μ < 2 / (order · P_x)
    where P_x is the input signal power. Rule of thumb: start with μ ≈ 0.01.

    Applications in this project:
        - Active noise cancellation on motor current feedback
        - Dynamic environment noise tracking
        - Adaptive notch filter (when the noise frequency is unknown / drifting)

    Args:
        order:    Filter order (number of taps).  More taps → better at low freq.
        mu:       LMS step size (learning rate).  Typical: 0.001 – 0.1.
        initial_weights: Optional numpy array of length `order`.

    Example::

        lms = AdaptiveFilterLMS(order=32, mu=0.01)
        for raw, reference in zip(noisy_signal, reference_clean):
            filtered = lms.update(raw, desired=reference)
        # After training, use in inference mode (no desired needed):
        filtered = lms.update(noisy_sample, desired=None)
    """

    def __init__(
        self,
        order:           int   = 32,
        mu:              float = 0.01,
        initial_weights: Optional[np.ndarray] = None,
    ) -> None:
        if mu <= 0:
            raise ValueError("LMS step size μ must be positive.")
        self._order  = order
        self._mu     = mu
        self._w      = initial_weights.copy() if initial_weights is not None else np.zeros(order)
        self._buffer: Deque[float] = deque([0.0] * order, maxlen=order)
        log.info(f"AdaptiveFilterLMS | order={order} | μ={mu:.4f}")

    def update(self, x: float, desired: Optional[float] = None) -> float:
        """
        Process one sample.

        Args:
            x:        Current input sample.
            desired:  Reference / clean signal (d[n]).  If None, no weight update.

        Returns:
            Filter output y[n].
        """
        self._buffer.appendleft(x)
        x_vec = np.array(self._buffer)
        y     = float(self._w @ x_vec)

        if desired is not None:
            e       = desired - y
            self._w = self._w + self._mu * e * x_vec
        return y

    def reset(self) -> None:
        self._w[:] = 0.0
        for _ in range(self._order):
            self._buffer.append(0.0)

    @property
    def weights(self) -> np.ndarray:
        return self._w.copy()

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        if value <= 0:
            raise ValueError("μ must be positive.")
        self._mu = value


