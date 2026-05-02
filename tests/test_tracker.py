"""
tests/test_tracker.py — Unit Tests for CentroidTracker
"""

import time
import pytest
import numpy as np

from src.vision.detect import Detection
from src.vision.tracker import CentroidTracker, Track


def _det(cx: int, cy: int, cls: str = "bottle", conf: float = 0.85) -> Detection:
    return Detection(
        class_id   = 0,
        class_name = cls,
        confidence = conf,
        bbox_xyxy  = np.array([cx-30, cy-30, cx+30, cy+30]),
        center_px  = (cx, cy),
        area_px    = 3600,
        world_xyz  = np.array([0.1, 0.2, 0.0]),
    )


@pytest.fixture
def tracker():
    return CentroidTracker(max_disappeared=3, max_distance_px=80.0)


class TestCentroidTracker:
    def test_registers_new_track(self, tracker):
        dets   = [_det(320, 240)]
        tracks = tracker.update(dets)
        assert len(tracks) == 1

    def test_assigns_consistent_id(self, tracker):
        dets = [_det(320, 240)]
        t1   = tracker.update(dets)
        t2   = tracker.update(dets)
        assert list(t1.keys())[0] == list(t2.keys())[0]

    def test_disappears_after_max_gone(self, tracker):
        tracker.update([_det(320, 240)])
        # Stop sending → track should disappear after 3 frames
        for _ in range(4):
            tracks = tracker.update([])
        assert len(tracks) == 0

    def test_separate_objects_get_different_ids(self, tracker):
        t1 = tracker.update([_det(100, 100)])
        t2 = tracker.update([_det(100, 100), _det(400, 400)])
        assert len(t2) == 2
        ids = set(t2.keys())
        assert len(ids) == 2

    def test_track_age_increments(self, tracker):
        dets = [_det(320, 240)]
        for _ in range(5):
            tracks = tracker.update(dets)
        track = list(tracks.values())[0]
        assert track.age >= 4

    def test_confirmed_after_3_frames(self, tracker):
        dets = [_det(320, 240)]
        for _ in range(3):
            tracks = tracker.update(dets)
        track = list(tracks.values())[0]
        assert track.is_confirmed

    def test_not_confirmed_before_3_frames(self, tracker):
        dets   = [_det(320, 240)]
        tracks = tracker.update(dets)
        track  = list(tracks.values())[0]
        assert not track.is_confirmed   # age=0 after 1 frame

    def test_kalman_smooths_jitter(self, tracker):
        """Kalman position should be smoother than raw jittered centroids."""
        # Feed noisy positions around (320, 240)
        rng = np.random.default_rng(42)
        positions = []
        for _ in range(15):
            noise = rng.integers(-10, 10, size=2)
            cx, cy = 320 + int(noise[0]), 240 + int(noise[1])
            tracks = tracker.update([_det(cx, cy)])
            if tracks:
                t = list(tracks.values())[0]
                positions.append(t.centroid.copy())

        positions = np.array(positions)
        # Kalman centroid should be close to (320, 240)
        mean_pos = positions.mean(axis=0)
        assert abs(mean_pos[0] - 320) < 20
        assert abs(mean_pos[1] - 240) < 20

    def test_smoothed_confidence_weighted(self, tracker):
        dets = [_det(320, 240, conf=0.9)]
        for _ in range(5):
            tracks = tracker.update(dets)
        track = list(tracks.values())[0]
        assert 0.85 <= track.smoothed_confidence <= 0.95
