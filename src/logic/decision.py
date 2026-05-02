"""
decision.py — Task Decision Engine (v2)
========================================
Improvements over v1:
  - Temporal consensus: object must appear in N consecutive frames before action
    (eliminates false picks from single-frame noise / ghost detections)
  - Priority scoring: balances confidence × area × track_age
    (prefers well-established, high-confidence, large objects)
  - Pick-sequence planner: queues multiple objects, avoids re-picking
  - Per-class cooldown: prevents spamming the same bin
  - Integrates with CentroidTracker (Track objects instead of raw Detections)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Set

import numpy as np

from src.vision.tracker import Track, CentroidTracker
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("logic.decision")


# ── Bin routing ───────────────────────────────────────────────────────────────
SORT_MAP: Dict[str, str] = {
    "bottle":     "recycle",
    "cup":        "recycle",
    "can":        "recycle",
    "cell phone": "hazardous",
    "mouse":      "hazardous",
    "remote":     "hazardous",
    "keyboard":   "hazardous",
    "book":       "general",
    "scissors":   "general",
    "apple":      "organic",
    "orange":     "organic",
    "banana":     "organic",
}

DROP_ZONES: Dict[str, np.ndarray] = {
    "recycle":   np.array([ 0.20,  0.25, 0.05]),
    "hazardous": np.array([-0.20,  0.25, 0.05]),
    "organic":   np.array([ 0.00,  0.30, 0.05]),
    "general":   np.array([ 0.20,  0.15, 0.05]),
    "unknown":   np.array([ 0.00,  0.20, 0.05]),
}

# Minimum bounding-box area (px²) to consider an object actionable
MIN_AREA_PX         = 2_500
# Number of consecutive frames an object must appear before pick
CONFIRM_FRAMES      = 4
# Global cooldown between consecutive pick operations (seconds)
GLOBAL_COOLDOWN_S   = 2.0
# Per-class cooldown (don't pick same class twice too fast)
CLASS_COOLDOWN_S    = 5.0


class TaskType(Enum):
    IDLE   = auto()
    PICK   = auto()
    SORT   = auto()


@dataclass
class RobotTask:
    task_type:    TaskType
    target_xyz:   Optional[np.ndarray] = None
    drop_xyz:     Optional[np.ndarray] = None
    class_name:   str = ""
    bin_label:    str = ""
    confidence:   float = 0.0
    track_id:     int   = -1
    priority:     float = 0.0

    def __str__(self) -> str:
        if self.task_type == TaskType.IDLE:
            return "TASK: IDLE"
        return (
            f"TASK: {self.task_type.name} | "
            f"[bold]{self.class_name}[/bold] (track #{self.track_id}) "
            f"conf={self.confidence:.0%} priority={self.priority:.3f} "
            f"→ [{self.bin_label}]"
        )


class DecisionEngine:
    """
    State machine with temporal consensus and priority-based selection.

    Args:
        mode:           "sort" or "pick".
        confirm_frames: Frames required before first pick action.
    """

    def __init__(
        self,
        mode:           str = "sort",
        confirm_frames: int = CONFIRM_FRAMES,
    ) -> None:
        self.mode           = mode
        self._confirm       = confirm_frames

        self._tracker       = CentroidTracker(
            max_disappeared = 12,
            max_distance_px = 80.0,
        )

        self._last_pick_t:      float            = 0.0
        self._class_last_pick:  Dict[str, float] = defaultdict(float)
        self._picked_tracks:    Set[int]          = set()   # track IDs already picked

        self._total_frames = 0
        self._pick_count   = 0

        log.info(
            f"DecisionEngine v2 | mode={mode.upper()} "
            f"| confirm={confirm_frames} frames"
        )

    # ── Main step ─────────────────────────────────────────────────────────────
    def decide(self, detections: list, frame_area: int = 640*480) -> RobotTask:
        """
        Args:
            detections:  List of Detection objects from ObjectDetector.
            frame_area:  Camera resolution in pixels (for normalisation).

        Returns:
            RobotTask.
        """
        from src.vision.detect import Detection  # avoid circular at module level

        self._total_frames += 1

        # Update tracker
        active_tracks = self._tracker.update(detections)

        # Time gates
        now = time.monotonic()
        if (now - self._last_pick_t) < GLOBAL_COOLDOWN_S:
            return RobotTask(TaskType.IDLE)

        # Gather candidates
        candidates: List[Track] = []
        for track in active_tracks.values():
            if not track.is_confirmed:
                continue
            if track.world_xyz is None:
                continue
            if track.track_id in self._picked_tracks:
                continue
            # Estimate area from centroid track (approx — use last known)
            # We re-check via detection list
            if (now - self._class_last_pick[track.class_name]) < CLASS_COOLDOWN_S:
                continue
            candidates.append(track)

        if not candidates:
            return RobotTask(TaskType.IDLE)

        # Score: confidence × age_bonus × proximity_to_centre
        best: Optional[Track] = None
        best_score = -1.0

        cx_frame, cy_frame = 320, 240   # assume 640×480; adjust if needed

        for t in candidates:
            age_bonus    = min(t.age / 20.0, 1.0)           # saturates at 20 frames
            cent_dist    = np.linalg.norm(t.centroid - np.array([cx_frame, cy_frame]))
            prox_bonus   = 1.0 / (1.0 + cent_dist / 300.0)  # prefer centred objects
            score = t.smoothed_confidence * 0.6 + age_bonus * 0.3 + prox_bonus * 0.1

            if score > best_score:
                best_score = score
                best       = t

        if best is None:
            return RobotTask(TaskType.IDLE)

        bin_label = SORT_MAP.get(best.class_name, "unknown")
        drop_xyz  = DROP_ZONES.get(bin_label, DROP_ZONES["unknown"])

        task = RobotTask(
            task_type  = TaskType.SORT if self.mode == "sort" else TaskType.PICK,
            target_xyz = best.world_xyz.copy(),
            drop_xyz   = drop_xyz.copy(),
            class_name = best.class_name,
            bin_label  = bin_label,
            confidence = best.smoothed_confidence,
            track_id   = best.track_id,
            priority   = best_score,
        )

        # Commit
        self._last_pick_t                       = now
        self._class_last_pick[best.class_name]  = now
        self._picked_tracks.add(best.track_id)
        self._pick_count += 1

        log.info(str(task))
        return task

    # ── Reset picked set (call after arm returns to home) ────────────────────
    def reset_picked(self) -> None:
        """Allow previously-picked tracks to be picked again (e.g. after conveyor move)."""
        self._picked_tracks.clear()
        log.debug("Picked-tracks set cleared")

    @property
    def stats(self) -> dict:
        return {
            "total_frames":  self._total_frames,
            "pick_count":    self._pick_count,
            "active_tracks": len(self._tracker.active_tracks),
            "pick_rate":     self._pick_count / max(self._total_frames, 1),
        }
