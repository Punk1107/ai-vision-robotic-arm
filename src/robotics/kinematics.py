"""
kinematics.py — Inverse Kinematics Solver (v2)
===============================================
Improvements over v1:
  - Singularity detection (warns before sending to near-singular configs)
  - Joint velocity cap (compute max Δθ per dt, reject unsafe commands)
  - Condition number check on Jacobian (numerical quality indicator)
  - Elbow-up vs elbow-down selection with configurable preference
  - Improved numerical IK: SLSQP with inequality joint-limit constraints
    (more reliable than L-BFGS-B for constrained problems)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from src.utils.config import config, RoboticsConfig
from src.utils.logger import get_logger

log = get_logger("robotics.kinematics")

# Joints must change < this per step to be considered safe
MAX_DELTA_DEG_PER_STEP = 45.0
# Warn when arm is within this distance of workspace boundary
BOUNDARY_MARGIN_M = 0.02


@dataclass
class JointAngles:
    base:     float
    shoulder: float
    elbow:    float
    wrist:    float
    gripper:  float = 0.0

    def clamp(self, limits: dict) -> "JointAngles":
        def _c(v, k):
            lo, hi = limits[k]
            return max(lo, min(hi, v))
        return JointAngles(
            base     = _c(self.base,     "base"),
            shoulder = _c(self.shoulder, "shoulder"),
            elbow    = _c(self.elbow,    "elbow"),
            wrist    = _c(self.wrist,    "wrist"),
            gripper  = _c(self.gripper,  "gripper"),
        )

    def as_list(self) -> list[float]:
        return [self.base, self.shoulder, self.elbow, self.wrist, self.gripper]

    def delta_to(self, other: "JointAngles") -> list[float]:
        a, b = self.as_list(), other.as_list()
        return [abs(b[i] - a[i]) for i in range(5)]

    def __str__(self) -> str:
        return (
            f"Base={self.base:+.1f}°  Shoulder={self.shoulder:.1f}°  "
            f"Elbow={self.elbow:.1f}°  Wrist={self.wrist:.1f}°  "
            f"Gripper={self.gripper:.1f}°"
        )


class IKSolver:
    """
    4-DOF IK solver with singularity detection and velocity validation.

    Args:
        elbow_up: Prefer elbow-up configuration when True (default).
                  Elbow-down is more stable for picking from floors.
    """

    def __init__(
        self,
        cfg:      Optional[RoboticsConfig] = None,
        elbow_up: bool = True,
    ) -> None:
        self._cfg     = cfg or config.robotics
        self.L1       = self._cfg.l1
        self.L2       = self._cfg.l2
        self.L3       = self._cfg.l3
        self.L4       = self._cfg.l4
        self._limits  = self._cfg.joint_limits
        self._elbow_up = elbow_up

        self._max_reach = self.L2 + self.L3 + self.L4
        self._min_reach = abs(self.L2 - self.L3)

        log.info(
            f"IK Solver v2 | L=[{self.L1},{self.L2},{self.L3},{self.L4}]m "
            f"| reach=[{self._min_reach:.3f}, {self._max_reach:.3f}]m "
            f"| elbow_up={elbow_up}"
        )

    # ── Forward kinematics ────────────────────────────────────────────────────
    def forward(self, angles: JointAngles) -> np.ndarray:
        b = math.radians(angles.base)
        s = math.radians(angles.shoulder)
        e = math.radians(angles.elbow)
        w = math.radians(angles.wrist)

        r = (self.L2 * math.cos(s)
             + self.L3 * math.cos(s + e)
             + self.L4 * math.cos(s + e + w))
        z = (self.L1
             + self.L2 * math.sin(s)
             + self.L3 * math.sin(s + e)
             + self.L4 * math.sin(s + e + w))

        return np.array([r * math.cos(b), r * math.sin(b), z])

    # ── Jacobian (analytical, 3×4) ────────────────────────────────────────────
    def jacobian(self, angles: JointAngles) -> np.ndarray:
        """
        Compute the geometric Jacobian ∂EE/∂θ for singularity analysis.
        Shape: (3, 4) — rows=[X,Y,Z], cols=[θ1,θ2,θ3,θ4]
        """
        b = math.radians(angles.base)
        s = math.radians(angles.shoulder)
        e = math.radians(angles.elbow)
        w = math.radians(angles.wrist)

        # Partial derivatives via chain rule (symbolic pre-computed)
        r  = (self.L2 * math.cos(s)
              + self.L3 * math.cos(s + e)
              + self.L4 * math.cos(s + e + w))

        dr_ds = (-self.L2 * math.sin(s)
                 - self.L3 * math.sin(s + e)
                 - self.L4 * math.sin(s + e + w))
        dr_de = (-self.L3 * math.sin(s + e)
                 - self.L4 * math.sin(s + e + w))
        dr_dw = -self.L4 * math.sin(s + e + w)

        cb, sb = math.cos(b), math.sin(b)

        J = np.array([
            [-r * sb,  dr_ds * cb,  dr_de * cb,  dr_dw * cb],  # ∂X
            [ r * cb,  dr_ds * sb,  dr_de * sb,  dr_dw * sb],  # ∂Y
            [ 0,       -dr_ds,      -dr_de,       -dr_dw    ],  # ∂Z (sign flip)
        ], dtype=np.float64)

        return J

    def condition_number(self, angles: JointAngles) -> float:
        """
        Returns σ_max / σ_min of the Jacobian.
        High condition number (>100) → near singularity.
        """
        J  = self.jacobian(angles)
        sv = np.linalg.svd(J, compute_uv=False)
        if sv[-1] < 1e-10:
            return float("inf")
        return float(sv[0] / sv[-1])

    def is_singular(self, angles: JointAngles, threshold: float = 80.0) -> bool:
        cond = self.condition_number(angles)
        if cond > threshold:
            log.warning(
                f"Near singularity detected: cond={cond:.1f} > {threshold}. "
                "Consider adjusting target or using elbow-down config."
            )
            return True
        return False

    # ── Velocity safety check ─────────────────────────────────────────────────
    def is_safe_delta(
        self,
        current: JointAngles,
        target:  JointAngles,
        max_deg: float = MAX_DELTA_DEG_PER_STEP,
    ) -> bool:
        """
        Reject a move if any joint must change more than max_deg in one step.
        Call this before every move_to() to prevent dangerous lurches.
        """
        deltas = current.delta_to(target)
        worst  = max(deltas)
        if worst > max_deg:
            log.warning(
                f"Unsafe joint delta: max={worst:.1f}° > {max_deg}°. "
                "Insert intermediate waypoints."
            )
            return False
        return True

    # ── Analytical IK ─────────────────────────────────────────────────────────
    def solve_analytical(
        self,
        target_xyz:      np.ndarray,
        wrist_pitch_deg: float = 0.0,
    ) -> Optional[JointAngles]:
        x_t, y_t, z_t = target_xyz.astype(float)

        theta1_deg = math.degrees(math.atan2(y_t, x_t))
        r_total    = math.sqrt(x_t**2 + y_t**2)

        wrist_rad = math.radians(wrist_pitch_deg)
        r = r_total - self.L4 * math.cos(wrist_rad)
        z = (z_t - self.L1) - self.L4 * math.sin(wrist_rad)

        D = (r**2 + z**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)

        if abs(D) > 1.0:
            return None

        # Choose elbow configuration
        sign = -1.0 if self._elbow_up else 1.0
        theta3_rad = math.atan2(sign * math.sqrt(max(0, 1 - D**2)), D)
        theta2_rad = (math.atan2(z, r)
                      - math.atan2(
                          self.L3 * math.sin(theta3_rad),
                          self.L2 + self.L3 * math.cos(theta3_rad),
                      ))

        theta2_deg = math.degrees(theta2_rad)
        theta3_deg = math.degrees(theta3_rad)
        theta4_deg = wrist_pitch_deg - theta2_deg - theta3_deg

        angles = JointAngles(
            base     = theta1_deg,
            shoulder = theta2_deg,
            elbow    = theta3_deg,
            wrist    = theta4_deg,
        ).clamp(self._limits)

        log.debug(f"Analytical IK → {angles}")
        return angles

    # ── Numerical IK (SLSQP) ─────────────────────────────────────────────────
    def solve_numerical(
        self,
        target_xyz:      np.ndarray,
        wrist_pitch_deg: float = 0.0,
        initial_guess:   Optional[JointAngles] = None,
    ) -> Optional[JointAngles]:
        """
        SLSQP minimises position error subject to joint-limit inequality
        constraints.  More reliable than L-BFGS-B for the constrained case.
        """
        lims   = self._limits
        bounds = [
            (math.radians(lims["base"][0]),     math.radians(lims["base"][1])),
            (math.radians(lims["shoulder"][0]), math.radians(lims["shoulder"][1])),
            (math.radians(lims["elbow"][0]),    math.radians(lims["elbow"][1])),
            (math.radians(lims["wrist"][0]),    math.radians(lims["wrist"][1])),
        ]

        x0 = (
            [math.radians(a) for a in initial_guess.as_list()[:4]]
            if initial_guess
            else [0.0, math.pi / 4, math.pi / 4, 0.0]
        )

        def _cost(q):
            a  = JointAngles(*[math.degrees(qi) for qi in q])
            ee = self.forward(a)
            return float(np.linalg.norm(ee - target_xyz) ** 2
                         + 0.01 * (math.degrees(q[3]) - wrist_pitch_deg) ** 2)

        def _grad(q):
            eps = 1e-5
            g   = np.zeros_like(q)
            f0  = _cost(q)
            for i in range(len(q)):
                qe    = q.copy(); qe[i] += eps
                g[i]  = (_cost(qe) - f0) / eps
            return g

        result = minimize(
            _cost, x0, jac=_grad,
            method  = "SLSQP",
            bounds  = bounds,
            options = {"maxiter": 300, "ftol": 1e-8},
        )

        if not result.success and result.fun > 1e-4:
            log.error(f"Numerical IK failed: {result.message} (fun={result.fun:.6f})")
            return None

        angles = JointAngles(
            *[math.degrees(qi) for qi in result.x]
        ).clamp(self._limits)

        log.debug(f"Numerical IK → {angles} (fun={result.fun:.6f})")
        return angles

    def _clamp_to_workspace(self, xyz: np.ndarray) -> np.ndarray:
        """Ensure XYZ is within the physical workspace defined in config."""
        ws = config.robotics
        x = np.clip(xyz[0], *ws.workspace_x)
        y = np.clip(xyz[1], *ws.workspace_y)
        z = np.clip(xyz[2], *ws.workspace_z)
        clamped = np.array([x, y, z])
        
        if not np.allclose(xyz, clamped, atol=1e-5):
            log.warning(f"Target {xyz} out of bounds → clamped to {clamped}")
        return clamped

    # ── Unified solve ─────────────────────────────────────────────────────────
    def solve(
        self,
        target_xyz:      np.ndarray,
        wrist_pitch_deg: float = 0.0,
        current:         Optional[JointAngles] = None,
    ) -> Optional[JointAngles]:
        """
        Solve IK.  Checks singularity and (optionally) velocity safety.

        Args:
            target_xyz:      Target position in robot frame (m).
            wrist_pitch_deg: Desired wrist angle (deg).
            current:         Current joint state for velocity check.

        Returns:
            JointAngles or None.
        """
        # 1. Safety first: Clamp target to reachable workspace
        target_xyz = self._clamp_to_workspace(target_xyz)

        # 2. Attempt Analytical IK (Fast)
        angles = self.solve_analytical(target_xyz, wrist_pitch_deg)
        if angles is None:
            log.info("Analytical IK out of range — trying numerical ...")
            angles = self.solve_numerical(target_xyz, wrist_pitch_deg)

        if angles is None:
            return None

        # Post-solve checks
        if self.is_singular(angles):
            log.warning("Solution is near-singular — proceed with caution.")

        if current is not None and not self.is_safe_delta(current, angles):
            log.warning("Large joint delta flagged — trajectory planner recommended.")

        return angles

    def is_reachable(self, target_xyz: np.ndarray) -> bool:
        x, y, z = target_xyz
        r    = math.sqrt(x**2 + y**2)
        dist = math.sqrt(r**2 + (z - self.L1)**2)
        return self._min_reach <= dist <= self._max_reach


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Interception — predict and intercept moving objects
# ─────────────────────────────────────────────────────────────────────────────

class DynamicInterceptor:
    """
    Predicts the future position of a moving object (e.g. on a conveyor belt)
    and computes an interception waypoint for the arm.

    Algorithm:
        1. Maintain a sliding window of recent (time, xyz) observations.
        2. Fit a linear velocity model via least-squares regression.
        3. Project the object's position forward by `arm_delay_s` seconds
           (the time it takes the arm to reach the interception point).
        4. Return the predicted XYZ for the IK solver.

    Usage::

        interceptor = DynamicInterceptor(solver)
        # Call every frame:
        intercept_xyz = interceptor.update(obj_xyz, time.monotonic())
        if intercept_xyz is not None:
            angles = solver.solve(intercept_xyz)
    """

    def __init__(
        self,
        solver:        IKSolver,
        window_s:      float = 0.5,    # history window for velocity estimation (s)
        arm_delay_s:   float = 0.8,    # estimated arm travel time to intercept (s)
        min_speed_mps: float = 0.005,  # below this speed → treat as stationary
    ) -> None:
        self._solver       = solver
        self._window_s     = window_s
        self._arm_delay    = arm_delay_s
        self._min_speed    = min_speed_mps

        # Ring buffer: list of (timestamp, xyz)
        self._history: list = []

        log.info(
            f"DynamicInterceptor | window={window_s:.2f}s | "
            f"arm_delay={arm_delay_s:.2f}s"
        )

    def update(
        self,
        xyz:  np.ndarray,
        t:    float,
    ) -> Optional[np.ndarray]:
        """
        Feed a new observation and return the predicted interception point.

        Args:
            xyz: Current object position in robot frame (m).
            t:   Current timestamp (seconds, e.g. time.monotonic()).

        Returns:
            Predicted XYZ (np.ndarray) if object is moving and reachable,
            or None if stationary / insufficient history.
        """
        self._history.append((t, xyz.copy()))

        # Prune old samples outside the window
        cutoff = t - self._window_s
        self._history = [(ts, p) for ts, p in self._history if ts >= cutoff]

        if len(self._history) < 3:
            return None  # Need at least 3 points for reliable regression

        times  = np.array([h[0] for h in self._history])
        points = np.array([h[1] for h in self._history])   # (N, 3)

        # Normalise time to avoid numerical issues
        t0     = times[0]
        t_norm = times - t0

        # Least-squares linear fit per axis: xyz = a + b * t
        velocity = np.zeros(3, dtype=np.float64)
        for axis in range(3):
            A = np.column_stack([np.ones_like(t_norm), t_norm])
            result = np.linalg.lstsq(A, points[:, axis], rcond=None)
            velocity[axis] = result[0][1]   # slope = velocity (m/s)

        speed = float(np.linalg.norm(velocity))

        if speed < self._min_speed:
            log.debug(f"DynamicInterceptor: object stationary (speed={speed:.4f} m/s)")
            return None

        # Predict position at (now + arm_delay)
        dt_predict   = (t - t0) + self._arm_delay
        latest_xyz   = points[-1]
        intercept    = latest_xyz + velocity * self._arm_delay

        if not self._solver.is_reachable(intercept):
            log.warning(
                f"Interception point {intercept} is outside workspace — "
                f"skipping."
            )
            return None

        log.debug(
            f"DynamicInterceptor: speed={speed:.3f}m/s | "
            f"intercept={intercept} (in {self._arm_delay:.2f}s)"
        )
        return intercept

    def reset(self) -> None:
        """Clear history (call when tracking a new object)."""
        self._history.clear()
