"""
tests/test_decision.py — Unit Tests for Decision Engine
"""

import time
import pytest
import numpy as np

from src.vision.detect import Detection, DetectionResult
from src.logic.decision import DecisionEngine, TaskType, MIN_AREA_PX


def _make_detection(class_name: str, conf: float = 0.8, area: int = 5000) -> Detection:
    cx, cy = 320, 240
    det = Detection(
        class_id   = 0,
        class_name = class_name,
        confidence = conf,
        bbox_xyxy  = np.array([cx - 50, cy - 50, cx + 50, cy + 50]),
        center_px  = (cx, cy),
        area_px    = area,
        world_xyz  = np.array([0.12, 0.20, 0.0]),
    )
    return det


def _make_result(detections):
    return DetectionResult(
        frame_id    = 1,
        timestamp   = time.time(),
        detections  = detections,
        inference_ms = 20.0,
    )


@pytest.fixture
def engine():
    return DecisionEngine(mode="sort")


class TestDecisionEngine:
    def test_idle_on_no_detections(self, engine):
        result = _make_result([])
        task   = engine.decide(result)
        assert task.task_type == TaskType.IDLE

    def test_sort_task_on_valid_detection(self, engine):
        det    = _make_detection("bottle")
        result = _make_result([det])
        task   = engine.decide(result)
        assert task.task_type == TaskType.SORT
        assert task.class_name == "bottle"
        assert task.bin_label  == "recycle"

    def test_hazardous_bin_classification(self, engine):
        det    = _make_detection("cell phone")
        result = _make_result([det])
        task   = engine.decide(result)
        assert task.bin_label == "hazardous"

    def test_unknown_class_goes_to_unknown_bin(self, engine):
        det    = _make_detection("unknown_object")
        result = _make_result([det])
        task   = engine.decide(result)
        assert task.bin_label == "unknown"

    def test_too_small_detection_is_idle(self, engine):
        det    = _make_detection("bottle", area=100)   # below MIN_AREA_PX
        result = _make_result([det])
        task   = engine.decide(result)
        assert task.task_type == TaskType.IDLE

    def test_no_world_xyz_is_idle(self, engine):
        det = _make_detection("bottle")
        det.world_xyz = None   # strip coordinates
        result = _make_result([det])
        task   = engine.decide(result)
        assert task.task_type == TaskType.IDLE

    def test_highest_confidence_selected(self, engine):
        low  = _make_detection("apple",  conf=0.5)
        high = _make_detection("bottle", conf=0.9)
        result = _make_result([low, high])
        task   = engine.decide(result)
        assert task.class_name == "bottle"

    def test_stats_accumulate(self, engine):
        for _ in range(3):
            result = _make_result([])
            engine.decide(result)
        assert engine.stats["total_frames"] == 3
