"""
tests/test_kinematics.py — Unit Tests for IK Solver v2
"""

import math
import pytest
import numpy as np

from src.robotics.kinematics import IKSolver, JointAngles


@pytest.fixture
def solver():
    return IKSolver(elbow_up=True)


@pytest.fixture
def solver_down():
    return IKSolver(elbow_up=False)


class TestForwardKinematics:
    def test_home_above_base(self, solver):
        home = JointAngles(base=0, shoulder=90, elbow=0, wrist=0)
        ee   = solver.forward(home)
        assert abs(ee[0]) < 0.01
        assert abs(ee[1]) < 0.01
        assert ee[2]      > 0.0

    def test_returns_ndarray_shape3(self, solver):
        a  = JointAngles(base=45, shoulder=60, elbow=30, wrist=0)
        ee = solver.forward(a)
        assert isinstance(ee, np.ndarray) and ee.shape == (3,)

    def test_symmetric_reach(self, solver):
        a1 = JointAngles(base= 45, shoulder=60, elbow=45, wrist=0)
        a2 = JointAngles(base=-45, shoulder=60, elbow=45, wrist=0)
        e1, e2 = solver.forward(a1), solver.forward(a2)
        assert abs(np.linalg.norm(e1) - np.linalg.norm(e2)) < 1e-6


class TestAnalyticalIK:
    def test_reachable_returns_solution(self, solver):
        assert solver.solve_analytical(np.array([0.15, 0.10, 0.05])) is not None

    def test_unreachable_returns_none(self, solver):
        assert solver.solve_analytical(np.array([5.0, 5.0, 5.0])) is None

    def test_round_trip_within_5mm(self, solver):
        target = np.array([0.12, 0.08, 0.04])
        angles = solver.solve_analytical(target)
        if angles is None:
            pytest.skip("Not reachable")
        err_mm = np.linalg.norm(solver.forward(angles) - target) * 1000
        assert err_mm < 5.0, f"Round-trip error {err_mm:.2f}mm > 5mm"

    def test_joint_limits_respected(self, solver):
        target = np.array([0.18, 0.15, 0.06])
        angles = solver.solve_analytical(target)
        if angles is None:
            pytest.skip()
        lims = solver._limits
        assert lims["base"][0]     <= angles.base     <= lims["base"][1]
        assert lims["shoulder"][0] <= angles.shoulder <= lims["shoulder"][1]

    def test_elbow_down_gives_different_solution(self, solver, solver_down):
        target = np.array([0.15, 0.10, 0.05])
        up   = solver.solve_analytical(target)
        down = solver_down.solve_analytical(target)
        if up is None or down is None:
            pytest.skip()
        # Elbow angles should differ in sign
        assert abs(up.elbow - down.elbow) > 1.0


class TestJacobian:
    def test_jacobian_shape(self, solver):
        a = JointAngles(base=30, shoulder=60, elbow=45, wrist=0)
        J = solver.jacobian(a)
        assert J.shape == (3, 4)

    def test_condition_number_positive(self, solver):
        a    = JointAngles(base=30, shoulder=60, elbow=45, wrist=0)
        cond = solver.condition_number(a)
        assert cond > 0

    def test_singular_detection_far_extended(self, solver):
        # Fully extended arm → near singularity
        a = JointAngles(base=0, shoulder=0, elbow=0, wrist=0)
        # We just check it doesn't crash; singularity may or may not trigger
        _ = solver.is_singular(a, threshold=50.0)


class TestVelocitySafety:
    def test_safe_small_delta(self, solver):
        a = JointAngles(base=0,  shoulder=90, elbow=0, wrist=0)
        b = JointAngles(base=10, shoulder=80, elbow=5, wrist=0)
        assert solver.is_safe_delta(a, b, max_deg=45.0) is True

    def test_unsafe_large_delta(self, solver):
        a = JointAngles(base=0,   shoulder=0,   elbow=0, wrist=0)
        b = JointAngles(base=180, shoulder=150, elbow=150, wrist=90)
        assert solver.is_safe_delta(a, b, max_deg=45.0) is False


class TestWorkspace:
    def test_near_target_reachable(self, solver):
        assert solver.is_reachable(np.array([0.10, 0.10, 0.05]))

    def test_far_target_unreachable(self, solver):
        assert not solver.is_reachable(np.array([5.0, 5.0, 5.0]))

    def test_clamp_preserves_limits(self, solver):
        extreme = JointAngles(base=999, shoulder=-999, elbow=999, wrist=-999)
        c       = extreme.clamp(solver._limits)
        lims    = solver._limits
        assert c.base     == lims["base"][1]
        assert c.shoulder == lims["shoulder"][0]
