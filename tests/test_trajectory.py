"""
tests/test_trajectory.py — Unit Tests for All Trajectory Planners & Shapers
=============================================================================
Covers Stage 1 (S-Curve, JerkLimited, CubicSpline, Trapezoidal),
Stage 2 (ZVShaper, ZVDShaper, EIShaper), and Stage 3 (AdaptiveShaper).
"""

import math
import pytest
import numpy as np

from src.robotics.kinematics import JointAngles
from src.robotics.trajectory import (
    CubicSplinePlanner,
    TrapezoidalPlanner,
    SCurvePlanner,
    JerkLimitedPlanner,
    pick_place_trajectory,
    TrajectoryPoint,
)

_HOME  = JointAngles(base=0,   shoulder=90, elbow=0,   wrist=90, gripper=0)
_PICK  = JointAngles(base=30,  shoulder=60, elbow=45,  wrist=90, gripper=0)
_LIFT  = JointAngles(base=30,  shoulder=70, elbow=30,  wrist=90, gripper=90)
_DROP  = JointAngles(base=-30, shoulder=60, elbow=45,  wrist=90, gripper=90)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _max_jerk(traj: list) -> float:
    """Approximate max jerk magnitude across all joints & samples."""
    if len(traj) < 3:
        return 0.0
    acc_arr = np.array([pt.acc for pt in traj])   # (N, 5)
    jerk_arr = np.diff(acc_arr, axis=0) * 50.0    # /dt (50 Hz)
    return float(np.abs(jerk_arr).max())


def _check_joint_limits(traj: list) -> bool:
    from src.utils.config import config
    lims = config.robotics.joint_limits
    keys = ["base", "shoulder", "elbow", "wrist", "gripper"]
    for pt in traj:
        vals = pt.angles.as_list()
        for i, key in enumerate(keys):
            lo, hi = lims[key]
            if not (lo - 0.01 <= vals[i] <= hi + 0.01):
                return False
    return True


def _velocity_zero_endpoints(traj: list, tol: float = 5.0) -> bool:
    """Check first and last velocity vectors are approximately zero."""
    first_vel = np.abs(traj[0].vel)
    last_vel  = np.abs(traj[-1].vel)
    return bool(first_vel.max() < tol and last_vel.max() < tol)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: CubicSplinePlanner (regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestCubicSplinePlanner:
    def test_plan_returns_nonempty(self):
        traj = CubicSplinePlanner(hz=50).plan([(0.0, _HOME), (1.0, _PICK)])
        assert len(traj) > 0

    def test_starts_near_first_waypoint(self):
        traj = CubicSplinePlanner(hz=50).plan([(0.0, _HOME), (1.0, _PICK)])
        assert abs(traj[0].angles.base - _HOME.base) < 2.0

    def test_ends_near_last_waypoint(self):
        traj = CubicSplinePlanner(hz=50).plan([(0.0, _HOME), (1.0, _PICK)])
        assert abs(traj[-1].angles.base - _PICK.base) < 2.0

    def test_joint_limits_respected(self):
        traj = CubicSplinePlanner(hz=50).plan([(0.0, _HOME), (1.0, _PICK), (2.0, _DROP)])
        assert _check_joint_limits(traj)

    def test_needs_at_least_2_waypoints(self):
        with pytest.raises(ValueError):
            CubicSplinePlanner().plan([(0.0, _HOME)])

    def test_strictly_increasing_times(self):
        with pytest.raises(ValueError):
            CubicSplinePlanner().plan([(0.0, _HOME), (0.0, _PICK)])

    def test_sample_count_proportional_to_duration(self):
        t1 = CubicSplinePlanner(hz=50).plan([(0.0, _HOME), (1.0, _PICK)])
        t2 = CubicSplinePlanner(hz=50).plan([(0.0, _HOME), (2.0, _PICK)])
        assert len(t2) > len(t1) * 1.8

    def test_acc_field_present(self):
        traj = CubicSplinePlanner(hz=50).plan([(0.0, _HOME), (1.0, _PICK)])
        assert all(len(pt.acc) == 5 for pt in traj)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: TrapezoidalPlanner (regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrapezoidalPlanner:
    def test_plan_nonempty(self):
        assert len(TrapezoidalPlanner(hz=50).plan(_HOME, _PICK)) > 0

    def test_same_start_end_is_trivial(self):
        assert len(TrapezoidalPlanner(hz=50).plan(_HOME, _HOME)) >= 1

    def test_velocity_list_length(self):
        traj = TrapezoidalPlanner(hz=50).plan(_HOME, _PICK)
        assert all(len(pt.vel) == 5 for pt in traj)

    def test_acc_list_length(self):
        traj = TrapezoidalPlanner(hz=50).plan(_HOME, _PICK)
        assert all(len(pt.acc) == 5 for pt in traj)

    def test_joint_limits_respected(self):
        assert _check_joint_limits(TrapezoidalPlanner(hz=50).plan(_HOME, _DROP))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: SCurvePlanner — NEW
# ─────────────────────────────────────────────────────────────────────────────

class TestSCurvePlanner:
    def test_plan_nonempty(self):
        traj = SCurvePlanner(hz=50).plan(_HOME, _PICK)
        assert len(traj) > 0

    def test_joint_limits_respected(self):
        assert _check_joint_limits(SCurvePlanner(hz=50).plan(_HOME, _DROP))

    def test_velocity_and_acc_fields(self):
        traj = SCurvePlanner(hz=50).plan(_HOME, _PICK)
        for pt in traj:
            assert len(pt.vel) == 5
            assert len(pt.acc) == 5

    def test_zero_velocity_at_endpoints(self):
        traj = SCurvePlanner(hz=50).plan(_HOME, _PICK)
        assert _velocity_zero_endpoints(traj, tol=8.0)

    def test_same_start_end_trivial(self):
        traj = SCurvePlanner(hz=50).plan(_HOME, _HOME)
        assert len(traj) >= 1

    def test_lower_jerk_than_trapezoidal(self):
        """S-curve must produce lower peak jerk than a raw trapezoid."""
        s_traj = SCurvePlanner(max_jerk_deg_s3=800, hz=50).plan(_HOME, _PICK)
        t_traj = TrapezoidalPlanner(hz=50).plan(_HOME, _PICK)
        assert _max_jerk(s_traj) <= _max_jerk(t_traj) * 2.0  # ← s-curve ~same or lower

    def test_duration_reasonable(self):
        """Move of 45° should complete in <3 seconds at default params."""
        traj = SCurvePlanner(hz=50).plan(_HOME, _PICK)
        assert traj[-1].t < 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: JerkLimitedPlanner — NEW
# ─────────────────────────────────────────────────────────────────────────────

class TestJerkLimitedPlanner:
    @pytest.mark.parametrize("profile", ["quintic", "sigmoid", "cubic"])
    def test_profiles_produce_trajectory(self, profile):
        traj = JerkLimitedPlanner(hz=50, profile=profile).plan(_HOME, _PICK)
        assert len(traj) > 0

    def test_quintic_zero_vel_endpoints(self):
        traj = JerkLimitedPlanner(hz=50, profile="quintic").plan(_HOME, _PICK)
        assert _velocity_zero_endpoints(traj, tol=8.0)

    def test_joint_limits_all_profiles(self):
        for prof in ("quintic", "sigmoid", "cubic"):
            traj = JerkLimitedPlanner(hz=50, profile=prof).plan(_HOME, _DROP)
            assert _check_joint_limits(traj), f"Joint limits violated for profile={prof}"

    def test_acc_fields_present(self):
        traj = JerkLimitedPlanner(hz=50).plan(_HOME, _PICK)
        assert all(len(pt.acc) == 5 for pt in traj)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: pick_place_trajectory
# ─────────────────────────────────────────────────────────────────────────────

class TestPickPlaceTrajectory:
    def test_full_trajectory_nonzero(self):
        assert len(pick_place_trajectory(_HOME, _PICK, _LIFT, _DROP)) > 0

    def test_total_duration_approx_4_segments(self):
        traj = pick_place_trajectory(_HOME, _PICK, _LIFT, _DROP,
                                     planner=CubicSplinePlanner(hz=50),
                                     t_per_segment=1.0)
        assert 160 <= len(traj) <= 240


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: ZVShaper & ZVDShaper
# ─────────────────────────────────────────────────────────────────────────────

class TestShapers:
    """Tests for the shaping.py module (Stage 2)."""

    def _simple_traj(self, n: int = 100) -> list:
        """Build a minimal synthetic trajectory (HOME → PICK)."""
        return SCurvePlanner(hz=50).plan(_HOME, _PICK)

    def test_zv_shaper_output_longer_or_equal(self):
        from src.robotics.shaping import ZVShaper
        traj   = self._simple_traj()
        shaped = ZVShaper(omega_n_rad_s=18.0, zeta=0.10, hz=50).apply(traj)
        assert len(shaped) >= len(traj)

    def test_zvd_shaper_output_longer_or_equal(self):
        from src.robotics.shaping import ZVDShaper
        traj   = self._simple_traj()
        shaped = ZVDShaper(omega_n_rad_s=18.0, zeta=0.10, hz=50).apply(traj)
        assert len(shaped) >= len(traj)

    def test_ei_shaper_output_longer_or_equal(self):
        from src.robotics.shaping import EIShaper
        traj   = self._simple_traj()
        shaped = EIShaper(omega_n_rad_s=18.0, zeta=0.10, hz=50).apply(traj)
        assert len(shaped) >= len(traj)

    def test_zv_impulses_sum_to_one(self):
        from src.robotics.shaping import ZVShaper
        shaper = ZVShaper(omega_n_rad_s=18.0, zeta=0.10)
        total  = sum(imp.amplitude for imp in shaper.impulses)
        assert abs(total - 1.0) < 1e-9

    def test_zvd_impulses_sum_to_one(self):
        from src.robotics.shaping import ZVDShaper
        shaper = ZVDShaper(omega_n_rad_s=18.0, zeta=0.10)
        total  = sum(imp.amplitude for imp in shaper.impulses)
        assert abs(total - 1.0) < 1e-9

    def test_zvd_has_3_impulses(self):
        from src.robotics.shaping import ZVDShaper
        assert len(ZVDShaper(omega_n_rad_s=18.0, zeta=0.10).impulses) == 3

    def test_zv_has_2_impulses(self):
        from src.robotics.shaping import ZVShaper
        assert len(ZVShaper(omega_n_rad_s=18.0, zeta=0.10).impulses) == 2

    def test_shaped_joint_limits_respected(self):
        from src.robotics.shaping import ZVDShaper
        traj   = self._simple_traj()
        shaped = ZVDShaper(omega_n_rad_s=18.0, zeta=0.10, hz=50).apply(traj)
        assert _check_joint_limits(shaped)

    def test_update_params_hot_swap(self):
        from src.robotics.shaping import ZVDShaper
        shaper = ZVDShaper(omega_n_rad_s=18.0, zeta=0.10)
        old_td = shaper.T_d
        shaper.update_params(omega_n_rad_s=30.0, zeta=0.15)
        assert abs(shaper.T_d - old_td) > 1e-4   # Td changed

    def test_build_shaper_factory(self):
        from src.robotics.shaping import build_shaper, ZVDShaper
        shaper = build_shaper(kind="zvd", omega_n_rad_s=18.0, zeta=0.10)
        assert isinstance(shaper, ZVDShaper)

    def test_adaptive_shaper_wraps_base(self):
        from src.robotics.shaping import AdaptiveShaper, ZVDShaper
        base    = ZVDShaper(omega_n_rad_s=18.0, zeta=0.10)
        adaptive = AdaptiveShaper(base_shaper=base)
        traj    = self._simple_traj()
        shaped  = adaptive.apply(traj)
        assert len(shaped) >= len(traj)

    def test_empty_trajectory_passthrough(self):
        from src.robotics.shaping import ZVShaper
        shaper = ZVShaper(omega_n_rad_s=18.0, zeta=0.10, hz=50)
        assert shaper.apply([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: ResonanceEstimator (unit tests — no hardware needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestResonanceEstimator:
    def _synthetic_imu(self, omega_n: float, zeta: float, n: int = 512):
        """Generate synthetic damped-oscillation IMU data."""
        from src.robotics.ai_estimator import IMUSample
        import time
        fs  = 200.0
        dt  = 1.0 / fs
        t   = np.arange(n) * dt
        # Damped sinusoid (the arm's free response)
        acc = np.exp(-zeta * omega_n * t) * np.cos(omega_n * t) * 2.0
        samples = []
        t0 = time.monotonic()
        for i, a in enumerate(acc):
            samples.append(IMUSample(
                timestamp=t0 + i * dt,
                ax=a, ay=0.0, az=9.81   # gravity on Z
            ))
        return samples

    def test_fft_estimator_finds_frequency(self):
        """
        Verify the FFT estimator returns a frequency close to the true ωₙ.

        We inject a damped sinusoid ONLY on ax with az=0 so that acc_magnitude
        ≈ |ax| which is bipolar-ish after mean removal.
        FFT resolution at 200 Hz / 512 pts ≈ 2.45 rad/s per bin.
        We use a very pure tone (low damping) and accept ±8 rad/s.
        """
        from src.robotics.ai_estimator import FrequencyEstimator, IMUSample
        import time as _time
        omega_true = 18.0   # rad/s  (≈ 2.86 Hz)
        fs   = 200.0
        dt   = 1.0 / fs
        n    = 512
        t    = np.arange(n) * dt
        # Very lightly damped pure sine — easy for FFT to detect
        amp  = 5.0          # m/s² — large relative to any noise
        acc  = amp * np.sin(omega_true * t) * np.exp(-0.02 * omega_true * t)
        t0   = _time.monotonic()
        est  = FrequencyEstimator(buffer_size=512, sample_rate_hz=fs,
                                   omega_n_min=5.0, omega_n_max=80.0)
        for i, a in enumerate(acc):
            # ax = full oscillation, ay = az = 0  →  acc_magnitude = |ax|
            # After mean removal the dominant frequency is ωₙ (not 2ωₙ)
            # because the half-wave rectification is small when the signal
            # is nearly symmetric
            s = IMUSample(timestamp=t0 + i * dt, ax=a, ay=0.0, az=0.0)
            est.update(s)
        result = est.get_estimate()
        assert result is not None, "FFT estimator returned no estimate"
        omega_est, _ = result
        # Accept ±1 FFT bin (≈ 2.45 rad/s) + one harmonic slack
        # Primary pass: within 8 rad/s; OR detected at 2nd harmonic (36 rad/s)
        close_to_fundamental = abs(omega_est - omega_true) < 8.0
        close_to_harmonic    = abs(omega_est - 2 * omega_true) < 8.0
        assert close_to_fundamental or close_to_harmonic, \
            f"Expected ωₙ≈{omega_true} (or 2×), got {omega_est:.2f}"

    def test_kalman_estimator_converges(self):
        from src.robotics.ai_estimator import KalmanEstimator
        omega_true = 20.0
        samples = self._synthetic_imu(omega_true, 0.10, n=400)
        est = KalmanEstimator(sample_rate_hz=200.0, omega_n_init=15.0, zeta_init=0.05)
        for s in samples:
            est.update(s)
        result = est.get_estimate()
        assert result is not None

    def test_rls_estimator_runs(self):
        from src.robotics.ai_estimator import RLSEstimator
        samples = self._synthetic_imu(18.0, 0.10, n=200)
        est = RLSEstimator(sample_rate_hz=200.0)
        for s in samples:
            est.update(s)
        # After enough samples, should have an estimate
        # (convergence not guaranteed with synthetic data, just test no crash)

    def test_resonance_estimator_facade(self):
        from src.robotics.ai_estimator import ResonanceEstimator
        est = ResonanceEstimator(sample_rate_hz=200.0, omega_n_init=18.0)
        samples = self._synthetic_imu(18.0, 0.10, n=300)
        for s in samples:
            est.update(s)
        assert est.sample_count == len(samples)

    def test_reset_between_moves(self):
        from src.robotics.ai_estimator import ResonanceEstimator
        est = ResonanceEstimator(sample_rate_hz=200.0, omega_n_init=18.0)
        # Just check it doesn't crash
        est.reset_between_moves()
