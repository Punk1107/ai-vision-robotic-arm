"""
visual_servo.py — Closed-Loop Visual Servoing Controller
=========================================================
Implements Image-Based Visual Servoing (IBVS) using a PID controller
to continuously correct the arm's position based on live camera feedback.

Instead of "look once → plan → move blind":
  1. Lock onto target's centroid in image plane.
  2. Each frame: compute error between desired pixel position and current.
  3. PID controller generates a correction ΔX, ΔY, ΔZ in Cartesian space.
  4. IK solver maps correction to joint angles → sent to arm.
  5. Loop until error < threshold OR max iterations reached.

Benefits over open-loop pick:
  - Compensates for IK calibration errors and servo backlash.
  - Handles objects that shift slightly after the initial detection.
  - Gives a measurable "approach quality" metric for the project report.

Reference:
  Chaumette & Hutchinson, "Visual Servo Control — Part I: Basic
  Approaches," IEEE Robotics & Automation Magazine, 2006.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Tuple

import numpy as np

from src.robotics.kinematics import IKSolver, JointAngles
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("robotics.visual_servo")


# ─────────────────────────────────────────────────────────────────────────────
# PID Controller (single axis)
# ─────────────────────────────────────────────────────────────────────────────

class _PID:
    """Simple discrete PID with anti-windup clamp."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        out_min: float = -0.05,
        out_max: float =  0.05,
    ) -> None:
        self.kp = kp; self.ki = ki; self.kd = kd
        self._out_min = out_min; self._out_max = out_max
        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = 0.0

    def step(self, error: float, dt: float) -> float:
        self._integral += error * dt
        # Anti-windup clamp
        self._integral = np.clip(
            self._integral,
            self._out_min / (self.ki + 1e-9),
            self._out_max / (self.ki + 1e-9),
        )
        derivative      = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(out, self._out_min, self._out_max))


# ─────────────────────────────────────────────────────────────────────────────
# Servo state
# ─────────────────────────────────────────────────────────────────────────────

class ServoState(Enum):
    IDLE      = auto()
    SERVOING  = auto()
    CONVERGED = auto()
    TIMEOUT   = auto()
    LOST      = auto()


@dataclass
class ServoResult:
    state:         ServoState
    iterations:    int
    final_error_px: float
    elapsed_s:     float
    final_angles:  Optional[JointAngles] = None


# ─────────────────────────────────────────────────────────────────────────────
# VisualServoController
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ServoConfig:
    """Tunable parameters for the visual servo loop."""
    # PID gains (tune per robot)
    kp_x: float = 0.0004    # pixel-error → metres correction
    ki_x: float = 0.00005
    kd_x: float = 0.0001

    kp_y: float = 0.0004
    ki_y: float = 0.00005
    kd_y: float = 0.0001

    kp_z: float = 0.0002    # area-error → Z correction (depth)
    ki_z: float = 0.00002
    kd_z: float = 0.00005

    # Convergence criteria
    pixel_tol:   float = 8.0    # pixels — lateral convergence
    area_tol:    float = 500.0  # px²  — depth convergence
    max_iter:    int   = 120    # frames before giving up
    max_dt_s:    float = 0.2    # skip large time gaps (e.g. debugger pauses)
    
    # New: Convergence window (must be within tolerance for N frames)
    convergence_window: int = 5

    # Desired pixel location of target (image centre by default)
    desired_cx: Optional[int] = None
    desired_cy: Optional[int] = None
    desired_area_px: float = 15000.0  # target object area when "at pick height"


class VisualServoController:
    """
    Closed-loop image-based visual servoing for the robotic arm.

    The controller expects a callable `get_observation()` that returns the
    current (centroid_x, centroid_y, area_px) of the tracked object in the
    live frame, or None if the object is lost.

    Usage::

        def get_obs():
            det = detector.detect(cam.read())
            best = det.best()
            if best is None:
                return None
            cx, cy = best.center_px
            return cx, cy, best.area_px

        servo = VisualServoController(ik_solver, servo_cfg)
        result = servo.run(
            get_observation=get_obs,
            move_callback=controller.move_to,
            current_angles=ctrl.current_angles,
        )
    """

    def __init__(
        self,
        ik_solver:  IKSolver,
        cfg:        Optional[ServoConfig] = None,
    ) -> None:
        self._ik  = ik_solver
        self._cfg = cfg or ServoConfig()

        # Frame size for default desired_cx, desired_cy
        cam = config.camera
        self._frame_cx = (self._cfg.desired_cx
                          if self._cfg.desired_cx is not None
                          else cam.width // 2)
        self._frame_cy = (self._cfg.desired_cy
                          if self._cfg.desired_cy is not None
                          else cam.height // 2)

        # Three independent PIDs: lateral X, lateral Y, axial Z (depth)
        c = self._cfg
        self._pid_x = _PID(c.kp_x, c.ki_x, c.kd_x)
        self._pid_y = _PID(c.kp_y, c.ki_y, c.kd_y)
        self._pid_z = _PID(c.kp_z, c.ki_z, c.kd_z)

        log.info(
            f"VisualServoController ready | "
            f"desired_center=({self._frame_cx},{self._frame_cy}) | "
            f"max_iter={c.max_iter}"
        )

    def reset(self) -> None:
        """Reset PID integrators (call before each new pick attempt)."""
        self._pid_x.reset()
        self._pid_y.reset()
        self._pid_z.reset()

    def run(
        self,
        get_observation: Callable[[], Optional[Tuple[int, int, float]]],
        move_callback:   Callable[[JointAngles], None],
        current_angles:  Optional[JointAngles] = None,
        current_xyz:     Optional[np.ndarray]  = None,
    ) -> ServoResult:
        """
        Run the visual servo loop.

        Args:
            get_observation: Callable returning (cx, cy, area_px) or None.
            move_callback:   Callable to send JointAngles to the arm.
            current_angles:  Last known joint angles (for IK initial guess).
            current_xyz:     Last known EE position (if available).

        Returns:
            ServoResult describing termination state.
        """
        self.reset()

        cfg       = self._cfg
        t_start   = time.monotonic()
        t_last    = t_start
        iteration = 0
        
        # Track convergence over time
        self._conv_count = 0

        # Estimate current EE XYZ if not provided
        if current_xyz is None and current_angles is not None:
            current_xyz = self._ik.forward(current_angles)
        if current_xyz is None:
            # Default: hover above workspace center
            ws = config.robotics
            current_xyz = np.array([
                0.0,
                (ws.workspace_y[0] + ws.workspace_y[1]) / 2,
                ws.workspace_z[1] * 0.7,
            ], dtype=np.float64)

        xyz = current_xyz.copy()

        log.info(f"Visual servo START | start_xyz={xyz}")

        while iteration < cfg.max_iter:
            obs = get_observation()
            if obs is None:
                log.warning(f"Servo: target lost at iteration {iteration}")
                return ServoResult(
                    state          = ServoState.LOST,
                    iterations     = iteration,
                    final_error_px = float("inf"),
                    elapsed_s      = time.monotonic() - t_start,
                    final_angles   = current_angles,
                )

            cx_obs, cy_obs, area_obs = obs
            t_now = time.monotonic()
            dt    = min(t_now - t_last, cfg.max_dt_s)
            t_last = t_now

            # ── Compute errors ────────────────────────────────────────────────
            err_x     = float(cx_obs - self._frame_cx)   # positive → obj is right
            err_y     = float(cy_obs - self._frame_cy)   # positive → obj is below
            err_area  = float(area_obs - cfg.desired_area_px)  # positive → too close

            lateral_err = math.hypot(err_x, err_y)

            # ── Convergence check ─────────────────────────────────────────────
            if lateral_err < cfg.pixel_tol and abs(err_area) < cfg.area_tol:
                self._conv_count += 1
            else:
                self._conv_count = 0

            if self._conv_count >= cfg.convergence_window:
                elapsed = time.monotonic() - t_start
                log.success(
                    f"Visual servo CONVERGED | "
                    f"iter={iteration} | err_px={lateral_err:.1f} | "
                    f"elapsed={elapsed:.2f}s"
                )
                return ServoResult(
                    state          = ServoState.CONVERGED,
                    iterations     = iteration,
                    final_error_px = lateral_err,
                    elapsed_s      = elapsed,
                    final_angles   = current_angles,
                )

            # ── PID outputs → Cartesian corrections ──────────────────────────
            # Image-plane: +X_img → +X_robot, +Y_img → −Z_robot (camera looking down)
            delta_x = self._pid_x.step(err_x, dt)    # +err means obj is right -> move arm right (+)
            delta_z = -self._pid_y.step(err_y, dt)   # +err means obj is down  -> move arm down (-)
            delta_y = -self._pid_z.step(err_area, dt) # +err means too close -> move arm back (-)

            xyz = xyz + np.array([delta_x, delta_y, delta_z])

            # ── IK ────────────────────────────────────────────────────────────
            new_angles = self._ik.solve(xyz, current=current_angles)
            if new_angles is None:
                log.warning(f"Servo: IK failed at xyz={xyz}. Skipping step.")
                iteration += 1
                continue

            move_callback(new_angles)
            current_angles = new_angles

            iteration += 1

        elapsed = time.monotonic() - t_start
        log.warning(
            f"Visual servo TIMEOUT after {iteration} iterations "
            f"({elapsed:.2f}s)"
        )
        return ServoResult(
            state          = ServoState.TIMEOUT,
            iterations     = iteration,
            final_error_px = float("inf"),
            elapsed_s      = elapsed,
            final_angles   = current_angles,
        )


import math  # noqa: E402 — needed for math.hypot in run()
