"""
tests/test_trajectory.py — Unit Tests for Trajectory Planners
"""

import pytest
import numpy as np

from src.robotics.kinematics  import JointAngles
from src.robotics.trajectory  import (
    CubicSplinePlanner,
    TrapezoidalPlanner,
    pick_place_trajectory,
)

_HOME  = JointAngles(base=0,  shoulder=90, elbow=0,  wrist=90, gripper=0)
_PICK  = JointAngles(base=30, shoulder=60, elbow=45, wrist=90, gripper=0)
_LIFT  = JointAngles(base=30, shoulder=70, elbow=30, wrist=90, gripper=90)
_DROP  = JointAngles(base=-30, shoulder=60, elbow=45, wrist=90, gripper=90)


class TestCubicSplinePlanner:
    def test_plan_returns_nonempty(self):
        p   = CubicSplinePlanner(hz=50)
        wp  = [(0.0, _HOME), (1.0, _PICK)]
        traj = p.plan(wp)
        assert len(traj) > 0

    def test_starts_near_first_waypoint(self):
        p    = CubicSplinePlanner(hz=50)
        wp   = [(0.0, _HOME), (1.0, _PICK)]
        traj = p.plan(wp)
        first = traj[0].angles
        assert abs(first.base     - _HOME.base)     < 2.0
        assert abs(first.shoulder - _HOME.shoulder) < 2.0

    def test_ends_near_last_waypoint(self):
        p    = CubicSplinePlanner(hz=50)
        wp   = [(0.0, _HOME), (1.0, _PICK)]
        traj = p.plan(wp)
        last = traj[-1].angles
        assert abs(last.base     - _PICK.base)     < 2.0
        assert abs(last.shoulder - _PICK.shoulder) < 2.0

    def test_joint_limits_respected(self):
        from src.utils.config import config
        p    = CubicSplinePlanner(hz=50)
        wp   = [(0.0, _HOME), (1.0, _PICK), (2.0, _DROP)]
        traj = p.plan(wp)
        lims = config.robotics.joint_limits
        for pt in traj:
            a = pt.angles
            assert lims["base"][0]     <= a.base     <= lims["base"][1]
            assert lims["shoulder"][0] <= a.shoulder <= lims["shoulder"][1]

    def test_needs_at_least_2_waypoints(self):
        p = CubicSplinePlanner()
        with pytest.raises(ValueError):
            p.plan([(0.0, _HOME)])

    def test_strictly_increasing_times(self):
        p = CubicSplinePlanner()
        with pytest.raises(ValueError):
            p.plan([(0.0, _HOME), (0.0, _PICK)])

    def test_sample_count_proportional_to_duration(self):
        p    = CubicSplinePlanner(hz=50)
        t1   = p.plan([(0.0, _HOME), (1.0, _PICK)])
        t2   = p.plan([(0.0, _HOME), (2.0, _PICK)])
        # Double duration → approximately double samples
        assert len(t2) > len(t1) * 1.8


class TestTrapezoidalPlanner:
    def test_plan_nonempty(self):
        p    = TrapezoidalPlanner(hz=50)
        traj = p.plan(_HOME, _PICK)
        assert len(traj) > 0

    def test_same_start_end_is_trivial(self):
        p    = TrapezoidalPlanner(hz=50)
        traj = p.plan(_HOME, _HOME)
        # Should still return at least 1 point
        assert len(traj) >= 1

    def test_velocity_list_length(self):
        p    = TrapezoidalPlanner(hz=50)
        traj = p.plan(_HOME, _PICK)
        for pt in traj:
            assert len(pt.vel) == 5


class TestPickPlaceTrajectory:
    def test_full_trajectory_nonzero(self):
        traj = pick_place_trajectory(_HOME, _PICK, _LIFT, _DROP)
        assert len(traj) > 0

    def test_total_duration_approx_4_segments(self):
        p    = CubicSplinePlanner(hz=50)
        traj = pick_place_trajectory(_HOME, _PICK, _LIFT, _DROP,
                                     planner=p, t_per_segment=1.0)
        # 4 segments × 1.0s × 50Hz ≈ 200 samples (±20%)
        assert 160 <= len(traj) <= 240
