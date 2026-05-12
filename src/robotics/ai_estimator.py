"""
ai_estimator.py — Stage 3: AI-Driven Resonance Estimator
==========================================================
Estimates the natural frequency (ωₙ) and damping ratio (ζ) of the
robotic arm in real-time using IMU accelerometer data fed back from
the Arduino firmware.

Why does ωₙ change during operation?
  The arm is *not* a fixed-parameter system.  Its resonant frequency
  depends on:
    - Current joint configuration (moment of inertia changes with extension)
    - Payload mass (holding a 200g bottle ≠ empty gripper)
    - Servo speed & mechanical wear over time

Three estimators are provided, ordered by capability:
  1. FrequencyEstimator  — FFT-based peak detection.  Fast, no tuning.
                           Best for post-move analysis or calibration runs.
  2. RLSEstimator        — Recursive Least Squares on the 2nd-order ODE.
                           Runs in real-time at every IMU sample.
  3. KalmanEstimator     — Extended Kalman Filter treating (ωₙ, ζ) as
                           states.  Most noise-robust; best for production.

All three implement the same interface:
    update(imu_sample: IMUSample) -> None    ← call from serial-read callback
    get_estimate()  -> Optional[Tuple[float, float]]  ← (ωₙ rad/s, ζ)

The ``AdaptiveShaper`` in shaping.py calls ``get_estimate()`` periodically
and hot-swaps the ZV/ZVD impulse parameters.

Serial protocol assumed (firmware → Python):
    {
      "status": "ok",
      "angles": [b, sh, el, wr, g],
      "imu":    {"ax": float, "ay": float, "az": float,
                 "gx": float, "gy": float, "gz": float}
    }
"""

from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger("robotics.ai_estimator")


# =============================================================================
# IMU data container
# =============================================================================

@dataclass
class IMUSample:
    """Single IMU reading from the Arduino telemetry stream."""
    timestamp: float    # monotonic clock (seconds)
    ax: float = 0.0     # linear acceleration X (m/s²)
    ay: float = 0.0     # linear acceleration Y (m/s²)
    az: float = 0.0     # linear acceleration Z (m/s²)
    gx: float = 0.0     # angular velocity X (rad/s)
    gy: float = 0.0     # angular velocity Y (rad/s)
    gz: float = 0.0     # angular velocity Z (rad/s)

    @classmethod
    def from_dict(cls, d: dict) -> "IMUSample":
        """Parse from the ``imu`` sub-dict in the Arduino JSON response."""
        ts = d.get("ts", time.monotonic())
        return cls(
            timestamp=ts,
            ax=d.get("ax", 0.0), ay=d.get("ay", 0.0), az=d.get("az", 0.0),
            gx=d.get("gx", 0.0), gy=d.get("gy", 0.0), gz=d.get("gz", 0.0),
        )

    @property
    def acc_magnitude(self) -> float:
        return math.sqrt(self.ax**2 + self.ay**2 + self.az**2)


# =============================================================================
# 1. FFT-based Frequency Estimator
# =============================================================================

class FrequencyEstimator:
    """
    Estimates ωₙ using Fast Fourier Transform on a rolling buffer of
    IMU acceleration data.

    Best used for:
      - Offline / post-move calibration
      - Initial ωₙ seed for the RLS or Kalman estimators
      - Sanity-checking the adaptive estimator

    Args:
        buffer_size:     Number of IMU samples to accumulate before FFT.
                         At 200 Hz IMU, 256 samples = 1.28s window.
        sample_rate_hz:  Expected IMU sample rate.
        omega_n_min:     Minimum plausible resonance (rad/s). Default 5.
        omega_n_max:     Maximum plausible resonance (rad/s). Default 80.
        zeta_default:    Returned damping ratio (FFT can't estimate ζ easily;
                         use RLS/Kalman for that).
    """

    def __init__(
        self,
        buffer_size:    int   = 256,
        sample_rate_hz: float = 200.0,
        omega_n_min:    float = 5.0,
        omega_n_max:    float = 80.0,
        zeta_default:   float = 0.10,
    ) -> None:
        self._buf: Deque[float] = collections.deque(maxlen=buffer_size)
        self._fs   = sample_rate_hz
        self._wmin = omega_n_min
        self._wmax = omega_n_max
        self._zeta = zeta_default
        self._lock = threading.Lock()
        self._estimate: Optional[Tuple[float, float]] = None

    def update(self, sample: IMUSample) -> None:
        """Feed one IMU sample into the rolling buffer."""
        with self._lock:
            # Use total acceleration magnitude minus gravity baseline
            self._buf.append(sample.acc_magnitude)
            if len(self._buf) == self._buf.maxlen:
                self._recompute()

    def get_estimate(self) -> Optional[Tuple[float, float]]:
        with self._lock:
            return self._estimate

    def _recompute(self) -> None:
        """Run FFT and find dominant frequency in the plausible band."""
        data = np.array(self._buf, dtype=float)
        # Remove DC component
        data -= data.mean()

        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(data))
        windowed = data * window

        fft_mag = np.abs(np.fft.rfft(windowed))
        freqs_hz = np.fft.rfftfreq(len(windowed), d=1.0 / self._fs)

        # Filter to plausible ωₙ band [rad/s → Hz]
        f_min = self._wmin / (2 * math.pi)
        f_max = self._wmax / (2 * math.pi)
        mask  = (freqs_hz >= f_min) & (freqs_hz <= f_max)

        if not mask.any():
            return

        peak_idx    = np.argmax(fft_mag[mask])
        # Map back to global index
        valid_freqs = freqs_hz[mask]
        peak_freq   = valid_freqs[peak_idx]
        omega_n     = 2 * math.pi * peak_freq

        if omega_n > 1.0:
            self._estimate = (omega_n, self._zeta)
            log.debug(f"FFT estimator: ωₙ={omega_n:.2f} rad/s ({peak_freq:.2f} Hz)")


# =============================================================================
# 2. Recursive Least Squares Estimator
# =============================================================================

class RLSEstimator:
    """
    Estimates ωₙ and ζ using Recursive Least Squares (RLS) on the
    discretised 2nd-order ODE:

        ẍ[k] = −ωₙ² x[k] − 2ζωₙ ẋ[k]

    Rewritten as linear regression:
        y[k] = θᵀ φ[k]

    where:
        y[k]   = ẍ[k]         (measured acceleration)
        φ[k]   = [x[k], ẋ[k]] (position & velocity, integrated from acc)
        θ      = [−ωₙ², −2ζωₙ]

    Args:
        forgetting_factor (λ): 0 < λ ≤ 1.  Closer to 1 = slower adaptation.
                               0.98 is a good default for arm motion (moves
                               at ~0.5-2 Hz, IMU at ~200 Hz).
        sample_rate_hz:        IMU sample rate for numerical integration.
        omega_n_min/max:       Validity bounds — reject wildly wrong estimates.
    """

    def __init__(
        self,
        forgetting_factor: float = 0.98,
        sample_rate_hz:    float = 200.0,
        omega_n_min:       float = 3.0,
        omega_n_max:       float = 100.0,
    ) -> None:
        self._lam  = forgetting_factor
        self._fs   = sample_rate_hz
        self._dt   = 1.0 / sample_rate_hz
        self._wmin = omega_n_min
        self._wmax = omega_n_max

        # RLS state
        self._theta = np.array([-20.0**2, -2 * 0.10 * 20.0])  # [−ωₙ², −2ζωₙ]
        self._P     = np.eye(2) * 1000.0          # covariance
        self._lock  = threading.Lock()

        # Signal integrators (trapezoidal)
        self._vel: float = 0.0
        self._pos: float = 0.0
        self._prev_acc: float = 0.0

        self._estimate: Optional[Tuple[float, float]] = None

    def update(self, sample: IMUSample) -> None:
        """Feed one IMU sample, perform one RLS step."""
        acc = sample.acc_magnitude - 9.81   # remove gravity (approx)

        with self._lock:
            # Integrate to get velocity and position (trapezoidal rule)
            self._vel += 0.5 * (acc + self._prev_acc) * self._dt
            self._pos += self._vel * self._dt
            self._prev_acc = acc

            phi = np.array([self._pos, self._vel])

            # RLS update
            K     = self._P @ phi / (self._lam + phi @ self._P @ phi)
            err   = acc - float(phi @ self._theta)
            self._theta += K * err
            self._P     = (self._P - np.outer(K, phi @ self._P)) / self._lam

            # Extract ωₙ and ζ from θ = [−ωₙ², −2ζωₙ]
            wn2    = -self._theta[0]
            two_zw = -self._theta[1]

            if wn2 > 0:
                omega_n = math.sqrt(wn2)
                zeta    = two_zw / (2.0 * omega_n + 1e-9)
                zeta    = max(0.001, min(0.99, zeta))

                if self._wmin <= omega_n <= self._wmax:
                    self._estimate = (omega_n, zeta)
                    log.debug(
                        f"RLS: ωₙ={omega_n:.2f} rad/s  ζ={zeta:.4f}  "
                        f"|err|={abs(err):.4f}"
                    )

    def get_estimate(self) -> Optional[Tuple[float, float]]:
        with self._lock:
            return self._estimate

    def reset(self) -> None:
        """Reset integrators (call between moves to avoid drift)."""
        with self._lock:
            self._vel = 0.0
            self._pos = 0.0
            self._prev_acc = 0.0
            self._P = np.eye(2) * 1000.0


# =============================================================================
# 3. Extended Kalman Filter Estimator
# =============================================================================

class KalmanEstimator:
    """
    Extended Kalman Filter (EKF) that treats (ωₙ, ζ) as slowly-varying
    states alongside the mechanical state (position, velocity).

    State vector: x = [pos, vel, ωₙ², 2ζωₙ]ᵀ   (4 states)

    Measurement: acceleration = −ωₙ² pos − 2ζωₙ vel + noise

    The EKF linearises the (nonlinear) state-space around the current
    estimate at each step.  This is the most noise-robust of the three
    estimators and the preferred choice for the AdaptiveShaper in production.

    Args:
        sample_rate_hz:  IMU sample rate.
        q_mech:          Process noise on mechanical states (pos, vel).
        q_param:         Process noise on parameter states (ωₙ², 2ζωₙ).
                         Small value = parameters assumed slowly varying.
        r_meas:          Measurement noise variance (IMU acc noise).
        omega_n_init:    Initial guess for ωₙ (rad/s).  Default 18.0.
        zeta_init:       Initial guess for ζ.  Default 0.10.
    """

    def __init__(
        self,
        sample_rate_hz: float = 200.0,
        q_mech:         float = 1e-2,
        q_param:        float = 1e-4,
        r_meas:         float = 0.5,
        omega_n_init:   float = 18.0,
        zeta_init:      float = 0.10,
        omega_n_min:    float = 3.0,
        omega_n_max:    float = 100.0,
    ) -> None:
        self._fs   = sample_rate_hz
        self._dt   = 1.0 / sample_rate_hz
        self._wmin = omega_n_min
        self._wmax = omega_n_max
        self._lock = threading.Lock()

        # Initial state: [pos=0, vel=0, wn2=(ωₙ_init)², 2ζωₙ=2*ζ*ωₙ]
        wn2  = omega_n_init**2
        two_zw = 2.0 * zeta_init * omega_n_init
        self._x = np.array([0.0, 0.0, wn2, two_zw])

        # Initial covariance
        self._P = np.diag([1.0, 1.0, (omega_n_init * 0.5)**2, (two_zw * 0.5)**2])

        # Process noise (diagonal Q)
        self._Q = np.diag([q_mech, q_mech, q_param, q_param])

        # Measurement noise
        self._R = np.array([[r_meas]])

        self._estimate: Optional[Tuple[float, float]] = None

    def update(self, sample: IMUSample) -> None:
        """EKF predict + update step from one IMU acceleration sample."""
        acc = sample.acc_magnitude - 9.81   # remove gravity

        with self._lock:
            dt  = self._dt
            x   = self._x
            P   = self._P

            # ── Predict step ─────────────────────────────────────────────────
            # State transition: simple Euler integration for pos/vel
            # Parameters (wn2, 2zw) treated as constant between steps
            pos, vel, wn2, two_zw = x
            pos_new = pos + vel * dt
            vel_new = vel + (-wn2 * pos - two_zw * vel) * dt
            x_pred  = np.array([pos_new, vel_new, wn2, two_zw])

            # Jacobian of f w.r.t. x (linearisation)
            F = np.array([
                [1.0,         dt,          0.0,         0.0],
                [-wn2 * dt,   1.0 - two_zw * dt, -pos * dt,  -vel * dt],
                [0.0,         0.0,          1.0,         0.0],
                [0.0,         0.0,          0.0,         1.0],
            ])
            P_pred = F @ P @ F.T + self._Q

            # ── Update step ───────────────────────────────────────────────────
            # Measurement model: z = −wn2*pos − 2zw*vel
            z_pred = -wn2 * pos_new - two_zw * vel_new
            innov  = acc - z_pred

            # Measurement Jacobian H
            H = np.array([[-wn2, -two_zw, -pos_new, -vel_new]])
            S = H @ P_pred @ H.T + self._R
            K = P_pred @ H.T @ np.linalg.inv(S)   # Kalman gain (1×4)

            self._x = x_pred + (K @ np.array([[innov]])).flatten()
            self._P = (np.eye(4) - K @ H) @ P_pred

            # ── Extract ωₙ, ζ ─────────────────────────────────────────────────
            wn2_est   = max(0.1, self._x[2])
            two_zw_est = self._x[3]
            omega_n    = math.sqrt(wn2_est)
            zeta       = two_zw_est / (2.0 * omega_n + 1e-9)
            zeta       = max(0.001, min(0.99, zeta))

            if self._wmin <= omega_n <= self._wmax:
                self._estimate = (omega_n, zeta)
                log.debug(
                    f"EKF: ωₙ={omega_n:.2f} rad/s  ζ={zeta:.4f}  "
                    f"innov={innov:.4f}"
                )

    def get_estimate(self) -> Optional[Tuple[float, float]]:
        with self._lock:
            return self._estimate

    def reset(self) -> None:
        """Reset mechanical states between moves (keep parameter estimates)."""
        with self._lock:
            self._x[0] = 0.0
            self._x[1] = 0.0
            self._P[0, :] = 0.0
            self._P[:, 0] = 0.0
            self._P[1, :] = 0.0
            self._P[:, 1] = 0.0
            self._P[0, 0] = 1.0
            self._P[1, 1] = 1.0


# =============================================================================
# Resonance Estimator — unified façade used by AdaptiveShaper
# =============================================================================

class ResonanceEstimator:
    """
    High-level façade that routes IMU samples through the KalmanEstimator
    (primary) and uses FrequencyEstimator as a seed / sanity-check.

    This is the object that should be passed to ``AdaptiveShaper``.

    Usage::

        estimator = ResonanceEstimator(sample_rate_hz=200.0)

        # In serial-read callback (control.py → _parse_telemetry):
        imu = IMUSample.from_dict(resp["imu"])
        estimator.update(imu)

        # AdaptiveShaper calls internally:
        omega_n, zeta = estimator.get_estimate()
    """

    def __init__(
        self,
        sample_rate_hz: float = 200.0,
        omega_n_init:   float = 18.0,
        zeta_init:      float = 0.10,
        use_fft_seed:   bool  = True,
    ) -> None:
        self._kalman = KalmanEstimator(
            sample_rate_hz=sample_rate_hz,
            omega_n_init=omega_n_init,
            zeta_init=zeta_init,
        )
        self._fft = FrequencyEstimator(sample_rate_hz=sample_rate_hz) if use_fft_seed else None
        self._seeded = not use_fft_seed
        self._lock   = threading.Lock()
        self._sample_count = 0

        log.info(
            f"ResonanceEstimator ready | fs={sample_rate_hz:.0f}Hz | "
            f"ωₙ_init={omega_n_init:.1f} rad/s | ζ_init={zeta_init:.3f}"
        )

    def update(self, sample: IMUSample) -> None:
        """Feed a new IMU sample into both estimators."""
        self._kalman.update(sample)
        if self._fft:
            self._fft.update(sample)
        with self._lock:
            self._sample_count += 1

        # After the FFT has enough data, use its estimate to seed the Kalman
        if not self._seeded and self._fft:
            fft_est = self._fft.get_estimate()
            if fft_est is not None:
                log.info(
                    f"ResonanceEstimator: FFT seed → "
                    f"ωₙ={fft_est[0]:.2f} rad/s  ζ={fft_est[1]:.3f}"
                )
                # The Kalman will naturally adapt; just log the seed
                self._seeded = True

    def get_estimate(self) -> Optional[Tuple[float, float]]:
        """Return the best current (ωₙ, ζ) estimate."""
        return self._kalman.get_estimate()

    def reset_between_moves(self) -> None:
        """
        Call this at the start of each new move to reset the mechanical
        states of the estimators (prevents position drift accumulation).
        """
        self._kalman.reset()
        log.debug("ResonanceEstimator: mechanical states reset for new move.")

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._sample_count
