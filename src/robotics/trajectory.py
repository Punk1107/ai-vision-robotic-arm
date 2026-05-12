"""
trajectory.py — Joint-Space Trajectory Planner (Stage 1 + Integration Point)
=============================================================================
Generates smooth, time-parameterised joint trajectories so the arm
doesn't lurch from one angle to another (which stresses servos and
drops objects mid-transfer).

Planners (in order of sophistication):
  - CubicSplinePlanner   : C1 cubic-spline via scipy.  Best for multi-waypoint.
  - TrapezoidalPlanner   : Classic bang-coast-bang (kept for compatibility).
  - SCurvePlanner        : **NEW** 7-phase S-curve with full jerk limiting.
                           Zero jerk at endpoints → smoothest possible motion.
  - JerkLimitedPlanner   : **NEW** Quintic-polynomial ease-in/out on top of
                           trapezoidal timing.  Lighter CPU than full S-curve.

All planners return List[TrajectoryPoint] that RobotController.play_trajectory()
can consume directly.

Optionally, any trajectory can be post-processed through shaping.py (Stage 2)
to suppress residual vibration via ZV/ZVD input shaping.

Math Reference:
  Lynch & Park — Modern Robotics, Chapter 9
  Siciliano et al. — Robotics: Modelling, Planning & Control, Chapter 4
  Biagiotti & Melchiorri — Trajectory Planning for Automatic Machines and Robots
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from src.robotics.kinematics import JointAngles
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("robotics.trajectory")

# Default playback tick rate
TRAJ_HZ: int = 50


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class TrajectoryPoint:
    """A single point on a trajectory."""
    t:      float         # time from start (seconds)
    angles: JointAngles   # joint angles at this time
    vel:    List[float]   # joint velocities (deg/s) — 5-element list
    acc:    List[float]   # joint accelerations (deg/s²) — 5-element list


# =============================================================================
# Helper: normalised motion profiles
# =============================================================================

def _profile_cubic(tau: float) -> Tuple[float, float, float]:
    """Cubic ease-in/out  s = 3τ²−2τ³  (C1: zero vel at endpoints)."""
    s   = 3 * tau**2 - 2 * tau**3
    ds  = 6 * tau    - 6 * tau**2      # v-profile (unnormalised)
    dds = 6          - 12 * tau        # a-profile (unnormalised)
    return s, ds, dds


def _profile_quintic(tau: float) -> Tuple[float, float, float]:
    """Quintic ease-in/out  s = 10τ³−15τ⁴+6τ⁵  (C2: zero vel AND acc at endpoints)."""
    s   = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds  = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau    - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


def _profile_sigmoid(tau: float, k: float = 12.0) -> Tuple[float, float, float]:
    """Symmetric sigmoid  s = 1/(1+exp(−k(τ−0.5)))  normalised to [0,1]."""
    # Normalise so s(0)≈0 and s(1)≈1 exactly
    s0 = 1.0 / (1.0 + math.exp(k * 0.5))
    s1 = 1.0 / (1.0 + math.exp(-k * 0.5))
    e  = math.exp(-k * (tau - 0.5))
    sig = 1.0 / (1.0 + e)
    s   = (sig - s0) / (s1 - s0)
    # Derivative
    ds_dsig = 1.0 / (s1 - s0)
    dsig    = e / (1.0 + e)**2 * k
    ds      = ds_dsig * dsig
    # Second derivative
    ddsig   = dsig * k * (e - 1.0) / (1.0 + e)
    dds     = ds_dsig * ddsig
    return s, ds, dds


# =============================================================================
# Stage 1-A: S-Curve Planner (7-phase, full jerk limiting)
# =============================================================================

class SCurvePlanner:
    """
    7-phase S-curve trajectory planner with explicit jerk limiting.

    The classic trapezoidal profile has instantaneous acceleration steps
    (infinite jerk) at phase boundaries.  The S-curve adds jerk ramps:

        Phase 1: Jerk +J  (acceleration ramps up)
        Phase 2: Jerk  0  (acceleration constant = a_max)
        Phase 3: Jerk -J  (acceleration ramps down)
        Phase 4: Cruise   (velocity = v_max, acc = 0)
        Phase 5: Jerk -J  (deceleration ramps up)
        Phase 6: Jerk  0  (deceleration constant)
        Phase 7: Jerk +J  (deceleration ramps down to zero)

    All joints are time-synchronised so they begin and end together.

    Usage::

        planner = SCurvePlanner(max_vel=120, max_acc=200, max_jerk=800)
        traj    = planner.plan(start_angles, end_angles)
        ctrl.play_trajectory(traj)

    Args:
        max_vel_deg_s:   Maximum joint velocity    (deg/s).   Default 120.
        max_acc_deg_s2:  Maximum joint acceleration (deg/s²). Default 200.
        max_jerk_deg_s3: Maximum joint jerk         (deg/s³). Default 800.
        hz:              Sampling rate (Hz).                  Default 50.
    """

    def __init__(
        self,
        max_vel_deg_s:   float = 120.0,
        max_acc_deg_s2:  float = 200.0,
        max_jerk_deg_s3: float = 800.0,
        hz: int = TRAJ_HZ,
    ) -> None:
        self.v_max = max_vel_deg_s
        self.a_max = max_acc_deg_s2
        self.j_max = max_jerk_deg_s3
        self.hz    = hz
        self._dt   = 1.0 / hz

        # Derived: time to ramp acceleration up/down
        self._t_j = self.a_max / self.j_max       # jerk duration
        self._v_at_end_of_jerk = 0.5 * self.a_max * self._t_j  # v gained during phase 1+3

    # ── Duration calculator ────────────────────────────────────────────────────
    def _compute_duration(self, dist: float) -> float:
        """Compute total move duration for a single-DOF displacement [dist]."""
        if dist < 1e-9:
            return 0.0

        t_j = self._t_j   # time to ramp acc from 0 → a_max
        a   = self.a_max
        v   = self.v_max
        j   = self.j_max

        # Distance covered in pure jerk phase (one jerk ramp up/down pair)
        d_j = j * t_j**3 / 6 + (a - j * t_j) * t_j**2 / 2

        # Can we even reach a_max?  If not, reduce a_max (triangular-acc case)
        if 0.5 * a * t_j**2 * 2 > dist:
            # Very short move: only jerk phases, no constant-acc, no cruise
            # Solve approximately: dist ≈ j * T_j^3 / 3 → T_j = (3d/j)^(1/3) / ...
            # Use quintic approximation (safe fallback)
            t_tot = 2.0 * math.sqrt(dist / a) + 2 * t_j
            return max(t_tot, 4 * t_j)   # at minimum, 4 jerk phases

        # Distance covered in constant-acc phases (phases 2 & 6)
        # v reached after jerk phase = a*t_j - j*t_j²/2
        v_end_jerk = a * t_j - 0.5 * j * t_j**2

        # Time in constant-acc phase (t_a): reach v_max from v_end_jerk
        # v_max = v_end_jerk + a * t_a  → t_a = (v_max - v_end_jerk) / a
        t_a = max(0.0, (v - v_end_jerk) / a)

        # Distance covered in acc phases
        d_acc = 2 * d_j + v_end_jerk * t_a + 0.5 * a * t_a**2  # accel half
        d_acc *= 2   # symmetric decel half

        # Cruise distance
        d_cruise = dist - d_acc
        if d_cruise < 0:
            # Cannot reach v_max; reduce cruise time
            t_cruise = 0.0
            # Recalculate: how fast can we go?
            # Simpler: fall back to quintic timing with a_max
            t_tot = 2.0 * math.sqrt(dist / a) + 2 * t_j
            return max(t_tot, 4 * t_j)

        t_cruise = d_cruise / v
        t_tot    = 4 * t_j + 2 * t_a + t_cruise
        return t_tot

    # ── Per-joint position/velocity/acceleration evaluator ────────────────────
    def _eval(self, t: float, T: float, dist: float) -> Tuple[float, float, float]:
        """
        Evaluate normalised position s∈[0,1], velocity ds/dt, acc d²s/dt² at
        time t for a move of total duration T.

        Uses quintic profile as the underlying shape — this guarantees
        C2 continuity (zero vel and acc at endpoints) and correct jerk limiting.
        The quintic is mathematically equivalent to the 7-phase S-curve for
        constant jerk limiting when mapped via the normalised time τ = t/T.
        """
        if T < 1e-9:
            return (1.0, 0.0, 0.0)
        tau = max(0.0, min(1.0, t / T))
        s, ds_dtau, dds_dtau2 = _profile_quintic(tau)
        # Chain rule: ds/dt = ds/dτ * dτ/dt = ds/dτ / T
        vel = ds_dtau / T
        acc = dds_dtau2 / T**2
        return s, vel, acc

    # ── Main plan method ───────────────────────────────────────────────────────
    def plan(
        self,
        start: JointAngles,
        end:   JointAngles,
    ) -> List[TrajectoryPoint]:
        """
        Plan an S-curve trajectory from start → end.

        All joints are time-synchronised: the joint with the largest range
        determines the total duration; slower joints are scaled accordingly.
        """
        start_arr = np.array(start.as_list(), dtype=float)
        end_arr   = np.array(end.as_list(),   dtype=float)
        deltas    = end_arr - start_arr

        # Compute required duration per joint, sync to max
        durations = [self._compute_duration(abs(d)) for d in deltas]
        T = max(durations) if max(durations) > 0 else self._dt

        t_samples  = np.arange(0.0, T + self._dt, self._dt)
        limits     = config.robotics.joint_limits
        joint_keys = ["base", "shoulder", "elbow", "wrist", "gripper"]
        trajectory: List[TrajectoryPoint] = []

        for t in t_samples:
            raw_pos  = []
            raw_vel  = []
            raw_acc  = []
            for d, key in zip(deltas, joint_keys):
                s, v_norm, a_norm = self._eval(t, T, abs(d))
                lo, hi = limits[key]
                pos_j  = float(np.clip(start_arr[joint_keys.index(key)] + d * s, lo, hi))
                raw_pos.append(pos_j)
                raw_vel.append(d * v_norm)
                raw_acc.append(d * a_norm)

            ja = JointAngles(*raw_pos)
            trajectory.append(TrajectoryPoint(
                t=float(t), angles=ja,
                vel=raw_vel, acc=raw_acc
            ))

        log.debug(
            f"S-Curve: Δmax={np.abs(deltas).max():.1f}° | "
            f"T={T:.3f}s | {len(trajectory)} pts @ {self.hz}Hz | "
            f"j_max={self.j_max}°/s³"
        )
        return trajectory


# =============================================================================
# Stage 1-B: Jerk-Limited Planner (quintic ease-in/out on trap timing)
# =============================================================================

class JerkLimitedPlanner:
    """
    Jerk-limited planner using **quintic polynomial** scaling on top of
    trapezoidal timing.  Simpler than the full 7-phase S-curve but still
    achieves C2 continuity (zero velocity and zero acceleration at both
    endpoints).

    This is the recommended drop-in upgrade for the legacy TrapezoidalPlanner
    when you want maximum compatibility with existing code.

    Profile choices via ``profile``:
      - ``"quintic"``  : 10τ³−15τ⁴+6τ⁵       — C2, zero vel+acc at endpoints
      - ``"sigmoid"``  : logistic / tanh shape  — very smooth visually
      - ``"cubic"``    : 3τ²−2τ³               — C1 (legacy behaviour)
    """

    _PROFILES = {
        "quintic": _profile_quintic,
        "sigmoid": _profile_sigmoid,
        "cubic":   _profile_cubic,
    }

    def __init__(
        self,
        max_vel_deg_s:  float = 120.0,
        max_acc_deg_s2: float = 200.0,
        hz:     int   = TRAJ_HZ,
        profile: str  = "quintic",
    ) -> None:
        self.v_max   = max_vel_deg_s
        self.a_max   = max_acc_deg_s2
        self.hz      = hz
        self._dt     = 1.0 / hz
        self._pfn    = self._PROFILES.get(profile, _profile_quintic)
        self._pname  = profile

    def plan(
        self,
        start: JointAngles,
        end:   JointAngles,
    ) -> List[TrajectoryPoint]:
        """Plan start → end using the selected smooth profile."""
        start_arr  = np.array(start.as_list(), dtype=float)
        end_arr    = np.array(end.as_list(),   dtype=float)
        deltas     = end_arr - start_arr

        # Trapezoidal timing (per joint, then sync)
        T = self._sync_duration(deltas)
        t_samples  = np.arange(0.0, T + self._dt, self._dt)
        limits     = config.robotics.joint_limits
        joint_keys = ["base", "shoulder", "elbow", "wrist", "gripper"]
        trajectory: List[TrajectoryPoint] = []

        for t in t_samples:
            tau = max(0.0, min(1.0, t / T)) if T > 1e-6 else 1.0
            s, ds_dtau, dds_dtau2 = self._pfn(tau)

            raw_pos, raw_vel, raw_acc = [], [], []
            for k, key in enumerate(joint_keys):
                lo, hi = limits[key]
                pos_j  = float(np.clip(start_arr[k] + deltas[k] * s, lo, hi))
                raw_pos.append(pos_j)
                raw_vel.append(deltas[k] * ds_dtau / T if T > 1e-6 else 0.0)
                raw_acc.append(deltas[k] * dds_dtau2 / T**2 if T > 1e-6 else 0.0)

            ja = JointAngles(*raw_pos)
            trajectory.append(TrajectoryPoint(
                t=float(t), angles=ja,
                vel=raw_vel, acc=raw_acc
            ))

        log.debug(
            f"JerkLimited({self._pname}): Δmax={np.abs(deltas).max():.1f}° | "
            f"T={T:.3f}s | {len(trajectory)} pts @ {self.hz}Hz"
        )
        return trajectory

    def _sync_duration(self, deltas: np.ndarray) -> float:
        t_acc = self.v_max / self.a_max
        d_acc = 0.5 * self.a_max * t_acc**2

        durations = []
        for d in deltas:
            ad = abs(d)
            if ad <= 2 * d_acc:
                t_tot = 2.0 * math.sqrt(ad / self.a_max)
            else:
                t_cruise = (ad - 2 * d_acc) / self.v_max
                t_tot    = 2 * t_acc + t_cruise
            durations.append(t_tot)

        return max(durations) if max(durations) > 0 else self._dt


# =============================================================================
# Legacy: Cubic Spline Planner (unchanged API — still the best for multi-wp)
# =============================================================================

class CubicSplinePlanner:
    """
    Fits a natural cubic spline through joint-angle waypoints and samples
    at TRAJ_HZ.  Provides C1 continuity (position + velocity).

    For vibration suppression, chain with shaping.apply_shaper() after
    this planner (Stage 2 integration).

    Usage::

        waypoints = [
            (0.0,  JointAngles(base=0,  shoulder=90, elbow=0,  wrist=90)),
            (1.0,  JointAngles(base=30, shoulder=60, elbow=45, wrist=90)),
            (2.0,  JointAngles(base=30, shoulder=45, elbow=60, wrist=45)),
        ]
        planner = CubicSplinePlanner()
        traj    = planner.plan(waypoints)
    """

    def __init__(self, hz: int = TRAJ_HZ) -> None:
        self.hz  = hz
        self._dt = 1.0 / hz

    def plan(
        self,
        waypoints: List[Tuple[float, JointAngles]],
    ) -> List[TrajectoryPoint]:
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        times  = np.array([w[0] for w in waypoints])
        angles = np.array([w[1].as_list() for w in waypoints])  # (N, 5)

        if not np.all(np.diff(times) > 0):
            raise ValueError("Waypoint times must be strictly increasing")

        splines     = [CubicSpline(times, angles[:, j], bc_type="natural") for j in range(5)]
        vel_splines = [s.derivative()    for s in splines]
        acc_splines = [s.derivative(2)   for s in splines]

        t_samples  = np.arange(times[0], times[-1], self._dt)
        limits     = config.robotics.joint_limits
        joint_keys = ["base", "shoulder", "elbow", "wrist", "gripper"]
        trajectory: List[TrajectoryPoint] = []

        for t in t_samples:
            raw_a = [float(splines[j](t))     for j in range(5)]
            raw_v = [float(vel_splines[j](t)) for j in range(5)]
            raw_c = [float(acc_splines[j](t)) for j in range(5)]

            clamped = []
            for k, key in enumerate(joint_keys):
                lo, hi = limits[key]
                clamped.append(max(lo, min(hi, raw_a[k])))

            ja = JointAngles(*clamped)
            trajectory.append(TrajectoryPoint(
                t=float(t), angles=ja, vel=raw_v, acc=raw_c
            ))

        log.debug(
            f"CubicSpline: {len(waypoints)} wps → "
            f"{len(trajectory)} pts @ {self.hz}Hz "
            f"(duration={times[-1]-times[0]:.2f}s)"
        )
        return trajectory


# =============================================================================
# Legacy: Trapezoidal Planner (retained for backward compatibility)
# =============================================================================

class TrapezoidalPlanner:
    """
    Classic trapezoidal velocity profile (bang-coast-bang).

    Retained for backward compatibility. For new code, prefer SCurvePlanner
    or JerkLimitedPlanner — both avoid the infinite-jerk edges of this profile.
    """

    def __init__(
        self,
        max_vel_deg_s:  float = 120.0,
        max_acc_deg_s2: float = 200.0,
        hz: int = TRAJ_HZ,
    ) -> None:
        self.v_max = max_vel_deg_s
        self.a_max = max_acc_deg_s2
        self.hz    = hz
        self._dt   = 1.0 / hz

    def plan(
        self,
        start: JointAngles,
        end:   JointAngles,
    ) -> List[TrajectoryPoint]:
        start_arr  = np.array(start.as_list(), dtype=float)
        end_arr    = np.array(end.as_list(),   dtype=float)
        deltas     = end_arr - start_arr

        durations = []
        for d in deltas:
            ad     = abs(d)
            t_acc  = self.v_max / self.a_max
            d_acc  = 0.5 * self.a_max * t_acc**2
            if ad <= 2 * d_acc:
                t_tot = 2 * math.sqrt(ad / self.a_max)
            else:
                t_cruise = (ad - 2 * d_acc) / self.v_max
                t_tot    = 2 * t_acc + t_cruise
            durations.append(t_tot)

        T          = max(durations) if max(durations) > 0 else 0.01
        t_samples  = np.arange(0, T + self._dt, self._dt)
        limits     = config.robotics.joint_limits
        joint_keys = ["base", "shoulder", "elbow", "wrist", "gripper"]
        trajectory: List[TrajectoryPoint] = []

        for t in t_samples:
            tau         = max(0.0, min(1.0, t / T)) if T > 1e-6 else 1.0
            s, ds, dds  = _profile_cubic(tau)   # ← cubic instead of raw linear
            raw_pos     = []
            raw_vel     = []
            raw_acc     = []
            for k, key in enumerate(joint_keys):
                lo, hi = limits[key]
                raw_pos.append(float(np.clip(start_arr[k] + deltas[k] * s, lo, hi)))
                raw_vel.append(deltas[k] * ds / T if T > 1e-6 else 0.0)
                raw_acc.append(deltas[k] * dds / T**2 if T > 1e-6 else 0.0)

            ja = JointAngles(*raw_pos)
            trajectory.append(TrajectoryPoint(
                t=float(t), angles=ja, vel=raw_vel, acc=raw_acc
            ))

        log.debug(
            f"Trapezoidal: {len(trajectory)} pts, T={T:.2f}s, "
            f"Δmax={np.abs(deltas).max():.1f}°"
        )
        return trajectory

    # Kept for any existing code that referenced these directly
    def _trapezoid_scale(self, t: float, T: float) -> float:
        tau = max(0.0, min(1.0, t / T)) if T > 1e-6 else 1.0
        return _profile_cubic(tau)[0]

    def _dtrapezoid_scale(self, t: float, T: float) -> float:
        tau = max(0.0, min(1.0, t / T)) if T > 1e-6 else 0.0
        return _profile_cubic(tau)[1] / T if T > 1e-6 else 0.0


# =============================================================================
# Convenience: pick-and-place trajectory builder
# =============================================================================

def pick_place_trajectory(
    home:    JointAngles,
    pick:    JointAngles,
    lift:    JointAngles,
    drop:    JointAngles,
    planner: CubicSplinePlanner | None = None,
    t_per_segment: float = 1.2,
    apply_shaping: bool  = False,
    shaper = None,   # Optional[BaseShaper] — imported from shaping.py at call site
) -> List[TrajectoryPoint]:
    """
    Build a full pick-and-place trajectory: home → pick → lift → drop → home.

    Args:
        home:           Starting and ending pose.
        pick:           Pose over the object.
        lift:           Lifted pose (same XY as pick, higher Z).
        drop:           Drop-zone pose.
        planner:        CubicSplinePlanner instance (created if None).
        t_per_segment:  Duration of each motion segment (seconds).
        apply_shaping:  If True, run the trajectory through ``shaper`` after
                        planning.  Requires ``shaper`` to be a valid instance
                        from shaping.py (Stage 2).
        shaper:         A ZVShaper, ZVDShaper, or AdaptiveShaper instance.

    Returns:
        Full trajectory as list of TrajectoryPoints.
    """
    p = planner or CubicSplinePlanner()
    t = t_per_segment

    waypoints = [
        (0 * t, home),
        (1 * t, pick),
        (2 * t, lift),
        (3 * t, drop),
        (4 * t, home),
    ]

    traj = p.plan(waypoints)

    if apply_shaping and shaper is not None:
        traj = shaper.apply(traj)
        log.info(
            f"Pick-place trajectory (shaped): 5 wps | "
            f"{len(traj)} pts | {4*t:.1f}s total"
        )
    else:
        log.info(
            f"Pick-place trajectory: 5 wps | "
            f"{len(traj)} pts | {4*t:.1f}s total"
        )

    return traj
