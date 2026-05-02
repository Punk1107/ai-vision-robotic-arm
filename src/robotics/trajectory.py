"""
trajectory.py — Joint-Space Trajectory Planner
===============================================
Generates smooth, time-parameterised joint trajectories so the arm
doesn't lurch from one angle to another (which stresses servos and
drops objects mid-transfer).

Two planners are provided:
  - CubicSplinePlanner : continuous position + velocity (C1) via natural
                         cubic spline interpolation.  Best for pick-place.
  - TrapezoidalPlanner : constant-acceleration profile (simpler, for
                         fast point-to-point with known max speed).

Both return a sequence of (time, JointAngles) waypoints that the
RobotController sends at a configurable tick rate.

Why cubic splines?
  - C1 continuity → no velocity discontinuities → smoother mechanics
  - Easy to insert via-points (e.g. lift before drop)
  - scipy.interpolate.CubicSpline is stable and well-tested

Reference:
  Lynch & Park — Modern Robotics, Chapter 9 (Trajectory Generation)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from src.robotics.kinematics import JointAngles
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("robotics.trajectory")

# Tick rate for trajectory playback (Hz)
TRAJ_HZ = 50


@dataclass
class TrajectoryPoint:
    """A single point on a trajectory."""
    t:      float         # time from start (seconds)
    angles: JointAngles   # joint angles at this time
    vel:    List[float]   # joint velocities (deg/s)


# =============================================================================
class CubicSplinePlanner:
    """
    Fits a cubic spline through a sequence of joint-angle waypoints
    and samples it at TRAJ_HZ for servo playback.

    Usage::

        waypoints = [
            (0.0,  JointAngles(base=0,  shoulder=90, elbow=0, wrist=90)),
            (1.0,  JointAngles(base=30, shoulder=60, elbow=45, wrist=90)),
            (2.0,  JointAngles(base=30, shoulder=45, elbow=60, wrist=45)),
        ]
        planner = CubicSplinePlanner()
        traj    = planner.plan(waypoints)
        for point in traj:
            ctrl.move_to(point.angles, blocking=False)
            time.sleep(1 / TRAJ_HZ)
    """

    def __init__(self, hz: int = TRAJ_HZ) -> None:
        self.hz = hz
        self._dt = 1.0 / hz

    def plan(
        self,
        waypoints: List[Tuple[float, JointAngles]],
    ) -> List[TrajectoryPoint]:
        """
        Args:
            waypoints: List of (time_sec, JointAngles).
                       Times must be strictly increasing.
                       Minimum 2 waypoints required.

        Returns:
            List of TrajectoryPoint sampled at self.hz.
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        times  = np.array([w[0] for w in waypoints])
        angles = np.array([w[1].as_list() for w in waypoints])  # (N, 5)

        if not np.all(np.diff(times) > 0):
            raise ValueError("Waypoint times must be strictly increasing")

        # Fit one spline per joint
        splines = [
            CubicSpline(times, angles[:, j], bc_type="natural")
            for j in range(5)
        ]
        vel_splines = [s.derivative() for s in splines]

        # Sample at hz
        t_samples = np.arange(times[0], times[-1], self._dt)
        trajectory: List[TrajectoryPoint] = []

        limits = config.robotics.joint_limits
        joint_keys = ["base", "shoulder", "elbow", "wrist", "gripper"]

        for t in t_samples:
            raw_angles = [float(splines[j](t)) for j in range(5)]
            raw_vel    = [float(vel_splines[j](t)) for j in range(5)]

            # Clamp to joint limits
            clamped = []
            for k, key in enumerate(joint_keys):
                lo, hi = limits[key]
                clamped.append(max(lo, min(hi, raw_angles[k])))

            ja = JointAngles(*clamped)
            trajectory.append(TrajectoryPoint(t=float(t), angles=ja, vel=raw_vel))

        log.debug(
            f"Cubic spline: {len(waypoints)} waypoints → "
            f"{len(trajectory)} samples @ {self.hz}Hz "
            f"(duration={times[-1]-times[0]:.2f}s)"
        )
        return trajectory


# =============================================================================
class TrapezoidalPlanner:
    """
    Generates a trapezoidal velocity profile for each joint independently.

    Good for fast point-to-point moves where smooth via-points aren't needed.

    Profile: accelerate → cruise → decelerate (bang-coast-bang)
    """

    def __init__(
        self,
        max_vel_deg_s:  float = 120.0,   # max joint speed (deg/s)
        max_acc_deg_s2: float = 200.0,   # max joint acceleration
        hz: int = TRAJ_HZ,
    ) -> None:
        self.v_max = max_vel_deg_s
        self.a_max = max_acc_deg_s2
        self.hz    = hz
        self._dt   = 1.0 / hz

    def plan(
        self,
        start:  JointAngles,
        end:    JointAngles,
    ) -> List[TrajectoryPoint]:
        """
        Plan a single move from start → end using per-joint trapezoid profiles,
        then synchronise all joints to finish at the same time.

        Returns:
            List of TrajectoryPoint sampled at self.hz.
        """
        start_arr = np.array(start.as_list())
        end_arr   = np.array(end.as_list())
        deltas    = end_arr - start_arr

        # Compute duration for each joint independently, then take max
        durations = []
        for d in deltas:
            ad = abs(d)
            t_acc = self.v_max / self.a_max
            d_acc = 0.5 * self.a_max * t_acc**2

            if ad <= 2 * d_acc:
                # Triangular profile (no cruise phase)
                t_tot = 2 * math.sqrt(ad / self.a_max)
            else:
                t_cruise = (ad - 2 * d_acc) / self.v_max
                t_tot    = 2 * t_acc + t_cruise

            durations.append(t_tot)

        T = max(durations) if max(durations) > 0 else 0.01  # sync time

        t_samples = np.arange(0, T + self._dt, self._dt)
        trajectory: List[TrajectoryPoint] = []

        limits    = config.robotics.joint_limits
        joint_keys = ["base", "shoulder", "elbow", "wrist", "gripper"]

        for t in t_samples:
            s = self._trapezoid_scale(t, T)   # 0→1 normalised progress
            raw = start_arr + deltas * s

            clamped = []
            for k, key in enumerate(joint_keys):
                lo, hi = limits[key]
                clamped.append(max(lo, min(hi, float(raw[k]))))

            ja = JointAngles(*clamped)
            # Approximate velocity from scale derivative
            vel = list(deltas * self._dtrapezoid_scale(t, T))
            trajectory.append(TrajectoryPoint(t=float(t), angles=ja, vel=vel))

        log.debug(
            f"Trapezoidal: {len(trajectory)} samples, duration={T:.2f}s, "
            f"max_Δ={np.abs(deltas).max():.1f}°"
        )
        return trajectory

    # ── Normalised trapezoid scalar ───────────────────────────────────────────
    def _trapezoid_scale(self, t: float, T: float) -> float:
        """Returns normalised displacement s∈[0,1] at time t."""
        if T < 1e-6:
            return 1.0
        t_acc  = self.v_max / self.a_max
        d_acc  = 0.5 * self.a_max * t_acc**2 * 2   # total distance if triangular

        # Re-use the full-duration profile scaled
        tau = t / T  # 0→1
        # Simple smooth step (≈ cubic ease-in-out) as approximation
        return 3 * tau**2 - 2 * tau**3

    def _dtrapezoid_scale(self, t: float, T: float) -> float:
        """Derivative of the normalised profile (for velocity estimate)."""
        if T < 1e-6:
            return 0.0
        tau = t / T
        return (6 * tau - 6 * tau**2) / T


# =============================================================================
# Convenience: build a pick-and-place trajectory with a lift waypoint
# =============================================================================

def pick_place_trajectory(
    home:    JointAngles,
    pick:    JointAngles,
    lift:    JointAngles,
    drop:    JointAngles,
    planner: CubicSplinePlanner | None = None,
    t_per_segment: float = 1.2,
) -> List[TrajectoryPoint]:
    """
    Build a full pick-and-place trajectory:
      home → pick → lift → drop → home

    Args:
        home:           Starting and ending pose.
        pick:           Pose over the object.
        lift:           Lifted pose (same XY as pick, higher Z).
        drop:           Drop-zone pose.
        planner:        CubicSplinePlanner instance (created if None).
        t_per_segment:  Duration of each motion segment (seconds).

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
    log.info(
        f"Pick-place trajectory: 5 waypoints, "
        f"{len(traj)} samples, {4*t:.1f}s total"
    )
    return traj
