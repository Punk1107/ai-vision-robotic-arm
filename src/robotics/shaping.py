"""
shaping.py — Input Shaping Filters (Stage 2 + Stage 3 Integration)
====================================================================
Implements classical and modern input-shaping filters that post-process
a pre-planned trajectory to suppress residual vibration without adding
extra actuator effort.

Background
----------
A robotic arm joint (with load) behaves like a *second-order oscillator*:

    ẍ + 2ζωₙẋ + ωₙ²x = u(t)

where:
  ωₙ = natural frequency (rad/s)  ← depends on arm extension & payload
  ζ  = damping ratio               ← typically 0.05–0.30 for hobby arms

If u(t) changes too abruptly (e.g., a trapezoidal velocity step), the
system rings at ωₙ after the move.  *Input shaping* convolves u(t) with
a sequence of timed impulses chosen to cancel the ring.

Two classical filters:
  ZV  (Zero Vibration)          — 2 impulses, eliminates residual vibration
                                   exactly at the modelled ωₙ.
  ZVD (Zero Vibration Derivative) — 3 impulses, also zeroes the sensitivity
                                   derivative → more robust to ωₙ uncertainty.

Stage-3 hook
------------
``AdaptiveShaper`` wraps any base shaper and accepts live (ωₙ, ζ) estimates
from ``ai_estimator.py``.  Drop-in replacement for ZVShaper / ZVDShaper.

API
---
    from src.robotics.shaping import ZVShaper, ZVDShaper, AdaptiveShaper
    from src.robotics.trajectory import SCurvePlanner, TrajectoryPoint

    planner = SCurvePlanner()
    traj    = planner.plan(start, end)

    shaper  = ZVDShaper(omega_n=18.0, zeta=0.10)
    shaped  = shaper.apply(traj)
    ctrl.play_trajectory(shaped)

References
----------
  Neil C. Singer & Warren P. Seering, "Preshaping Command Inputs to Reduce
  System Vibration", Journal of Dynamic Systems, Measurement, and Control,
  1990.

  William Singhose, "Command shaping for flexible systems", Automatica, 2009.
"""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.robotics.kinematics import JointAngles
from src.robotics.trajectory import TrajectoryPoint, TRAJ_HZ
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("robotics.shaping")


# =============================================================================
# Impulse sequence
# =============================================================================

@dataclass
class Impulse:
    """A single impulse in the shaping sequence."""
    amplitude: float    # Relative amplitude (impulses sum to 1.0)
    delay_s:   float    # Time offset from the start of the move (seconds)


# =============================================================================
# Base class
# =============================================================================

class BaseShaper(ABC):
    """
    Abstract base for all input shapers.

    Subclasses implement ``_build_impulses()`` to define the impulse
    sequence.  ``apply()`` performs the time-domain convolution using
    that sequence.
    """

    def __init__(self, omega_n_rad_s: float, zeta: float, hz: int = TRAJ_HZ) -> None:
        """
        Args:
            omega_n_rad_s: Natural frequency of the arm joint (rad/s).
                           Estimate from free-oscillation tests or IMU data.
                           Typical hobby-arm range: 10–30 rad/s.
            zeta:          Damping ratio (unitless, 0 < ζ < 1).
                           Typical: 0.05–0.25.
            hz:            Trajectory sample rate (Hz) — must match planner.
        """
        self._omega_n = omega_n_rad_s
        self._zeta    = zeta
        self._hz      = hz
        self._dt      = 1.0 / hz
        self._impulses: List[Impulse] = []
        self._rebuild()

    # ── Derived quantities ────────────────────────────────────────────────────
    @property
    def omega_n(self) -> float:
        return self._omega_n

    @property
    def omega_d(self) -> float:
        """Damped natural frequency (rad/s)."""
        return self._omega_n * math.sqrt(max(0.0, 1.0 - self._zeta**2))

    @property
    def T_d(self) -> float:
        """Period of damped oscillation (seconds)."""
        return (2 * math.pi / self.omega_d) if self.omega_d > 1e-6 else float("inf")

    @property
    def impulses(self) -> List[Impulse]:
        return list(self._impulses)

    # ── Public interface ──────────────────────────────────────────────────────
    def update_params(self, omega_n_rad_s: float, zeta: float) -> None:
        """
        Hot-swap resonance parameters without creating a new shaper.
        Thread-safe — used by AdaptiveShaper when the AI estimator updates.
        """
        self._omega_n = omega_n_rad_s
        self._zeta    = zeta
        self._rebuild()
        log.debug(
            f"{self.__class__.__name__} updated: "
            f"ωₙ={omega_n_rad_s:.2f} rad/s  ζ={zeta:.4f}"
        )

    def apply(self, trajectory: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """
        Convolve the trajectory with the impulse sequence.

        Each joint angle channel is convolved independently.  The resulting
        trajectory is extended by the longest impulse delay to allow the
        system to settle.

        Args:
            trajectory: Output of any trajectory planner (CubicSplinePlanner,
                        SCurvePlanner, JerkLimitedPlanner …).

        Returns:
            New List[TrajectoryPoint] with vibration-suppressing shaping
            applied.  Length ≥ len(trajectory).
        """
        if not trajectory:
            return trajectory

        # ── Build full time axis (original + settling tail) ──────────────────
        max_delay   = max(imp.delay_s for imp in self._impulses)
        n_tail      = int(math.ceil(max_delay / self._dt))
        n_orig      = len(trajectory)
        n_total     = n_orig + n_tail

        dt = self._dt
        t0 = trajectory[0].t

        # ── Extract per-joint angle arrays ─────────────────────────────────────
        n_joints    = 5
        orig_angles = np.zeros((n_total, n_joints))
        orig_vel    = np.zeros((n_total, n_joints))
        orig_acc    = np.zeros((n_total, n_joints))

        for i, pt in enumerate(trajectory):
            orig_angles[i] = pt.angles.as_list()
            orig_vel[i]    = pt.vel
            orig_acc[i]    = pt.acc

        # Hold last position in the settling tail (velocity and acc stay zero)
        orig_angles[n_orig:] = orig_angles[n_orig - 1]

        # ── Convolve each joint with impulse sequence (vectorised) ─────────────
        # All joints are processed simultaneously per impulse using NumPy slicing.
        # No temporary per-joint arrays are allocated inside the loop.
        shaped_angles = np.zeros((n_total, n_joints))
        shaped_vel    = np.zeros((n_total, n_joints))
        shaped_acc    = np.zeros((n_total, n_joints))

        for imp in self._impulses:
            shift = int(round(imp.delay_s / dt))
            if shift == 0:
                # No shift: add weighted original directly
                shaped_angles += imp.amplitude * orig_angles
                shaped_vel    += imp.amplitude * orig_vel
                shaped_acc    += imp.amplitude * orig_acc
            else:
                # Shift all joints at once: src rows [0:n-shift] -> dst rows [shift:n]
                end = n_total - shift
                if end > 0:
                    shaped_angles[shift:, :] += imp.amplitude * orig_angles[:end, :]
                    shaped_vel[shift:, :]    += imp.amplitude * orig_vel[:end, :]
                    shaped_acc[shift:, :]    += imp.amplitude * orig_acc[:end, :]
                # Rows [0:shift] stay zero (no wrap-around — causal filter)

        # ── Clamp to joint limits & build output list ─────────────────────────
        limits     = config.robotics.joint_limits
        joint_keys = ["base", "shoulder", "elbow", "wrist", "gripper"]

        # Build clamp bounds as arrays for vectorised clip
        lo_arr = np.array([limits[k][0] for k in joint_keys], dtype=float)
        hi_arr = np.array([limits[k][1] for k in joint_keys], dtype=float)
        shaped_angles = np.clip(shaped_angles, lo_arr, hi_arr)

        out: List[TrajectoryPoint] = []
        for i in range(n_total):
            t_i = t0 + i * dt
            ja  = JointAngles(*shaped_angles[i].tolist())
            out.append(TrajectoryPoint(
                t=t_i,
                angles=ja,
                vel=shaped_vel[i].tolist(),
                acc=shaped_acc[i].tolist(),
            ))

        log.debug(
            f"{self.__class__.__name__}.apply(): "
            f"{n_orig} → {n_total} pts "
            f"(+{n_tail} settling pts @ ωₙ={self._omega_n:.1f} rad/s)"
        )
        return out


    # ── Subclass contract ─────────────────────────────────────────────────────
    @abstractmethod
    def _build_impulses(self) -> List[Impulse]:
        """Return the impulse sequence that defines this shaper."""
        ...

    def _rebuild(self) -> None:
        self._impulses = self._build_impulses()
        log.debug(
            f"{self.__class__.__name__}: "
            f"ωₙ={self._omega_n:.2f} rad/s  ζ={self._zeta:.4f}  "
            f"Td={self.T_d:.4f}s  "
            f"impulses={[(round(im.amplitude,4), round(im.delay_s,4)) for im in self._impulses]}"
        )


# =============================================================================
# Stage 2-A: ZV Shaper
# =============================================================================

class ZVShaper(BaseShaper):
    """
    Zero-Vibration (ZV) input shaper.

    Uses 2 impulses to cancel residual vibration exactly at the
    modelled (ωₙ, ζ).  Sensitive to parameter uncertainty.

    Impulse parameters (Singer & Seering, 1990):

        K   = exp(−πζ / √(1−ζ²))
        A₁  = 1 / (1 + K)          at t=0
        A₂  = K / (1 + K)          at t=Td/2
    """

    def _build_impulses(self) -> List[Impulse]:
        zeta  = self._zeta
        K     = math.exp(-math.pi * zeta / math.sqrt(max(1e-9, 1.0 - zeta**2)))
        denom = 1.0 + K
        t_half = self.T_d / 2.0

        return [
            Impulse(amplitude=1.0 / denom,  delay_s=0.0),
            Impulse(amplitude=K   / denom,  delay_s=t_half),
        ]


# =============================================================================
# Stage 2-B: ZVD Shaper
# =============================================================================

class ZVDShaper(BaseShaper):
    """
    Zero-Vibration-Derivative (ZVD) input shaper.

    Uses 3 impulses.  Zeroes both the vibration *and* its first derivative
    with respect to ωₙ → more robust against natural-frequency modelling error.
    Recommended over ZV for real-world arms with uncertain dynamics.

    Impulse parameters:

        K   = exp(−πζ / √(1−ζ²))
        A₁  = 1 / (1 + 2K + K²)   at t=0
        A₂  = 2K / (…)             at t=Td/2
        A₃  = K² / (…)             at t=Td
    """

    def _build_impulses(self) -> List[Impulse]:
        zeta   = self._zeta
        K      = math.exp(-math.pi * zeta / math.sqrt(max(1e-9, 1.0 - zeta**2)))
        denom  = (1.0 + K)**2
        t_half = self.T_d / 2.0

        return [
            Impulse(amplitude=1.0      / denom, delay_s=0.0),
            Impulse(amplitude=2.0 * K  / denom, delay_s=t_half),
            Impulse(amplitude=K**2     / denom, delay_s=2 * t_half),
        ]


# =============================================================================
# Stage 2-C: EI Shaper (Extra-Insensitive — bonus)
# =============================================================================

class EIShaper(BaseShaper):
    """
    Extra-Insensitive (EI) shaper — 3 impulses with wider ωₙ tolerance band
    than ZVD.  Useful when the arm's natural frequency varies significantly
    (heavy payloads, long extension).

    After Singhose et al., "Extra-Insensitive Input Shapers for Controlling
    Flexible Spacecraft", Journal of Guidance, 1996.

    Tolerance parameter ``V_tol``:
        Fraction of unshaped vibration allowed at ωₙ ± 10 % (default 0.05 = 5 %).
    """

    def __init__(
        self,
        omega_n_rad_s: float,
        zeta: float,
        hz: int = TRAJ_HZ,
        v_tol: float = 0.05,
    ) -> None:
        self._v_tol = v_tol
        super().__init__(omega_n_rad_s, zeta, hz)

    def _build_impulses(self) -> List[Impulse]:
        zeta   = self._zeta
        v      = self._v_tol
        K      = math.exp(-zeta * math.pi / math.sqrt(max(1e-9, 1.0 - zeta**2)))
        t_half = self.T_d / 2.0

        # EI amplitudes (Singhose 1996, simplified closed-form)
        a1 = (1.0 + v) / 4.0
        a2 = (1.0 - v) / 2.0
        a3 = (1.0 + v) / 4.0

        return [
            Impulse(amplitude=a1, delay_s=0.0),
            Impulse(amplitude=a2, delay_s=t_half),
            Impulse(amplitude=a3, delay_s=2 * t_half),
        ]


# =============================================================================
# Stage 3: Adaptive Shaper (AI-driven hot-swap of ωₙ & ζ)
# =============================================================================

class AdaptiveShaper:
    """
    Wraps any BaseShaper subclass and lets an external AI estimator update
    the resonance parameters at runtime without rebuilding the shaper object.

    Thread-safe: the estimator callback can run in a separate thread (e.g.,
    the ai_estimator.py update loop) while trajectories are being planned
    in the control thread.

    Usage::

        from src.robotics.shaping    import AdaptiveShaper, ZVDShaper
        from src.robotics.ai_estimator import ResonanceEstimator

        base    = ZVDShaper(omega_n_rad_s=18.0, zeta=0.10)
        estimator = ResonanceEstimator()
        shaper  = AdaptiveShaper(base_shaper=base, estimator=estimator)

        # In control loop:
        shaped_traj = shaper.apply(traj)

        # In IMU callback (separate thread):
        estimator.update(imu_acc_xyz)     # auto-updates shaper params

    Args:
        base_shaper:  An already-constructed ZVShaper / ZVDShaper / EIShaper.
        estimator:    A ``ResonanceEstimator`` instance (from ai_estimator.py).
                      If None, the shaper acts as a fixed-parameter shaper.
        update_hz:    How often to query the estimator for new params (Hz).
                      Default: every trajectory application.
    """

    def __init__(
        self,
        base_shaper: BaseShaper,
        estimator=None,      # Optional[ResonanceEstimator]
        update_hz: float = 2.0,
    ) -> None:
        self._shaper    = base_shaper
        self._estimator = estimator
        self._lock      = threading.Lock()
        self._update_period = 1.0 / update_hz if update_hz > 0 else float("inf")
        self._last_update   = 0.0

    @property
    def impulses(self) -> List[Impulse]:
        return self._shaper.impulses

    @property
    def current_omega_n(self) -> float:
        return self._shaper.omega_n

    def apply(self, trajectory: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Apply shaping, optionally refreshing params from the estimator first."""
        self._maybe_update_params()
        with self._lock:
            return self._shaper.apply(trajectory)

    def force_update(self, omega_n: float, zeta: float) -> None:
        """Manually override the shaper parameters (e.g., from a calibration run)."""
        with self._lock:
            self._shaper.update_params(omega_n, zeta)

    def _maybe_update_params(self) -> None:
        import time
        now = time.monotonic()
        if self._estimator is None:
            return
        if (now - self._last_update) < self._update_period:
            return

        try:
            est = self._estimator.get_estimate()
            if est is not None:
                omega_n_new, zeta_new = est
                with self._lock:
                    old_wn = self._shaper.omega_n
                    self._shaper.update_params(omega_n_new, zeta_new)
                    if abs(omega_n_new - old_wn) > 0.5:
                        log.info(
                            f"AdaptiveShaper: ωₙ updated "
                            f"{old_wn:.2f} → {omega_n_new:.2f} rad/s  "
                            f"ζ={zeta_new:.4f}"
                        )
        except Exception as exc:
            log.warning(f"AdaptiveShaper: estimator error — {exc}")
        finally:
            self._last_update = now


# =============================================================================
# Factory helper
# =============================================================================

def build_shaper(
    kind: str = "zvd",
    omega_n_rad_s: Optional[float] = None,
    zeta: Optional[float]          = None,
    hz:   int = TRAJ_HZ,
    estimator = None,
) -> BaseShaper | AdaptiveShaper:
    """
    Convenience factory to build a shaper from config or explicit params.

    Args:
        kind:          ``"zv"`` | ``"zvd"`` | ``"ei"`` | ``"adaptive_zv"``
                       | ``"adaptive_zvd"``  (default ``"zvd"``).
        omega_n_rad_s: Natural frequency (rad/s).  Falls back to
                       ``config.input_shaping.omega_n`` if not given.
        zeta:          Damping ratio.  Falls back to
                       ``config.input_shaping.zeta`` if not given.
        hz:            Sample rate matching the trajectory planner.
        estimator:     ``ResonanceEstimator`` for adaptive variants.

    Returns:
        A configured shaper ready to call ``.apply(trajectory)``.
    """
    # Resolve config defaults (with graceful fallback if section missing)
    cfg_section = getattr(config, "input_shaping", None)
    _omega_n = omega_n_rad_s or (getattr(cfg_section, "omega_n", 18.0) if cfg_section else 18.0)
    _zeta    = zeta           or (getattr(cfg_section, "zeta",    0.10) if cfg_section else 0.10)

    kind = kind.lower()
    if kind == "zv":
        shaper = ZVShaper(omega_n_rad_s=_omega_n, zeta=_zeta, hz=hz)
    elif kind == "zvd":
        shaper = ZVDShaper(omega_n_rad_s=_omega_n, zeta=_zeta, hz=hz)
    elif kind == "ei":
        shaper = EIShaper(omega_n_rad_s=_omega_n, zeta=_zeta, hz=hz)
    elif kind in ("adaptive_zv",):
        base   = ZVShaper(omega_n_rad_s=_omega_n, zeta=_zeta, hz=hz)
        return AdaptiveShaper(base, estimator=estimator)
    elif kind in ("adaptive_zvd", "adaptive"):
        base   = ZVDShaper(omega_n_rad_s=_omega_n, zeta=_zeta, hz=hz)
        return AdaptiveShaper(base, estimator=estimator)
    else:
        log.warning(f"Unknown shaper kind '{kind}' — defaulting to ZVD.")
        shaper = ZVDShaper(omega_n_rad_s=_omega_n, zeta=_zeta, hz=hz)

    log.info(
        f"Input shaper ready: {shaper.__class__.__name__} | "
        f"ωₙ={_omega_n:.2f} rad/s | ζ={_zeta:.4f} | "
        f"Td={shaper.T_d:.4f}s"
    )
    return shaper
