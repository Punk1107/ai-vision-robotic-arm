#!/usr/bin/env python3
"""
scripts/measure_resonance.py — Empirical Resonance Measurement Utility
=======================================================================
Runs a step input on each joint, reads back IMU acceleration from the
firmware, and uses the FFT estimator to measure the arm's natural frequency.

Output is printed and optionally written to config.yaml automatically.

Usage::

    python -m scripts.measure_resonance [--joint base] [--update-config]

Requires:
  - Firmware that outputs {"imu": {...}} in its serial telemetry.
  - An IMU (e.g. MPU-6050) mounted near the arm end-effector.

Run with --dry (no hardware) for a simulation using synthetic data.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# ── ensure project root on sys.path ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import config, CONFIG_FILE
from src.utils.logger import setup_logger, get_logger
from src.robotics.control import RobotController
from src.robotics.kinematics import JointAngles
from src.robotics.ai_estimator import FrequencyEstimator, ResonanceEstimator, IMUSample

log = get_logger("scripts.measure_resonance")

_JOINTS = ["base", "shoulder", "elbow", "wrist"]
_HOME   = JointAngles(base=0, shoulder=90, elbow=0, wrist=0, gripper=0)


def run_measurement(
    joint:      str   = "base",
    amplitude:  float = 20.0,   # degrees — small step to excite oscillation
    wait_s:     float = 2.0,    # record for this long after the step
    dry:        bool  = False,
) -> tuple[float, float]:
    """
    Execute a step move on ``joint``, capture IMU response, return (ωₙ, ζ).
    """
    if joint not in _JOINTS:
        raise ValueError(f"joint must be one of {_JOINTS}")

    setup_logger()
    log.info(f"Measuring resonance on joint: {joint}  amplitude: {amplitude}°")

    estimator = ResonanceEstimator(
        sample_rate_hz=config.input_shaping.imu_sample_rate,
        use_fft_seed=True,
    )
    fft_est = FrequencyEstimator(
        buffer_size=512,
        sample_rate_hz=config.input_shaping.imu_sample_rate,
    )

    if dry:
        log.warning("[DRY] Generating synthetic damped-oscillation data.")
        omega_true = 18.0
        zeta_true  = 0.10
        fs = config.input_shaping.imu_sample_rate
        dt = 1.0 / fs
        n  = int(wait_s * fs)
        t  = np.arange(n) * dt
        acc_data = np.exp(-zeta_true * omega_true * t) * np.cos(omega_true * t) * 3.0

        t0 = time.monotonic()
        for i, a in enumerate(acc_data):
            s = IMUSample(timestamp=t0 + i * dt, ax=a, ay=0.0, az=9.81)
            estimator.update(s)
            fft_est.update(s)
            time.sleep(dt * 0.01)   # fast-forward

        est = estimator.get_estimate()
        fft = fft_est.get_estimate()
        log.info(f"[DRY] Kalman estimate:  ωₙ={est[0]:.2f} rad/s  ζ={est[1]:.4f}" if est else "[DRY] No Kalman estimate yet.")
        log.info(f"[DRY] FFT    estimate:  ωₙ={fft[0]:.2f} rad/s  ζ=N/A" if fft else "[DRY] No FFT estimate yet.")

        omega_n = est[0] if est else omega_true
        zeta    = est[1] if est else zeta_true
        return omega_n, zeta

    # ── Real hardware path ────────────────────────────────────────────────────
    imu_buffer: list[IMUSample] = []

    def _imu_hook(sample: IMUSample) -> None:
        estimator.update(sample)
        fft_est.update(sample)
        imu_buffer.append(sample)

    ctrl = RobotController()
    if not ctrl.connect():
        log.error("Cannot connect to arm. Use --dry for simulation.")
        return 18.0, 0.10

    # Patch the controller to intercept IMU data
    ctrl._estimator = estimator

    # Move to home, then step
    ctrl.home(blocking=True)
    time.sleep(0.5)

    step_angles = dict(zip(_JOINTS, _HOME.as_list()[:4]))
    step_angles[joint] += amplitude
    target = JointAngles(**step_angles, gripper=0)

    log.info(f"Applying step: {joint} +{amplitude}°")
    ctrl.move_to(target, blocking=True)

    log.info(f"Recording IMU for {wait_s:.1f}s ...")
    time.sleep(wait_s)

    ctrl.home(blocking=True)
    ctrl.disconnect()

    est = estimator.get_estimate()
    fft = fft_est.get_estimate()
    log.info(f"Kalman estimate:  ωₙ={est[0]:.2f} rad/s  ζ={est[1]:.4f}" if est else "No Kalman estimate.")
    log.info(f"FFT    estimate:  ωₙ={fft[0]:.2f} rad/s" if fft else "No FFT estimate.")

    omega_n = est[0] if est else 18.0
    zeta    = est[1] if est else 0.10
    return omega_n, zeta


def update_config_yaml(omega_n: float, zeta: float) -> None:
    """Overwrite omega_n and zeta in config.yaml."""
    import re
    text = CONFIG_FILE.read_text(encoding="utf-8")
    text = re.sub(r"(omega_n\s*:\s*)[\d.]+", rf"\g<1>{omega_n:.2f}", text)
    text = re.sub(r"(zeta\s*:\s*)[\d.]+",    rf"\g<1>{zeta:.4f}",    text)
    CONFIG_FILE.write_text(text, encoding="utf-8")
    log.success(f"config.yaml updated: omega_n={omega_n:.2f}  zeta={zeta:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure arm resonance frequency.")
    parser.add_argument("--joint",         default="base",  choices=_JOINTS)
    parser.add_argument("--amplitude",     default=20.0,    type=float)
    parser.add_argument("--wait",          default=2.0,     type=float)
    parser.add_argument("--update-config", action="store_true")
    parser.add_argument("--dry",           action="store_true")
    args = parser.parse_args()

    omega_n, zeta = run_measurement(
        joint=args.joint,
        amplitude=args.amplitude,
        wait_s=args.wait,
        dry=args.dry,
    )

    print(f"\n{'='*50}")
    print(f"  Measured: ωₙ = {omega_n:.2f} rad/s  ({omega_n/(2*math.pi):.2f} Hz)")
    print(f"            ζ  = {zeta:.4f}")
    print(f"{'='*50}\n")

    if args.update_config:
        update_config_yaml(omega_n, zeta)
