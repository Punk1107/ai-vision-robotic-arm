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

# Semantic Priority Map (higher is more important)
PRIORITY_MAP: Dict[str, int] = {
    "hazardous": 100,
    "recycle":   80,
    "organic":   60,
    "general":   40,
    "reject":    20,
    "unknown":   0,
}


class TaskType(Enum):
    IDLE       = auto()
    PICK       = auto()
    SORT       = auto()
    ABORT      = auto()   # In-progress pick must be aborted (target moved too far)
    RECOVERING = auto()   # Recalculating pick after abort


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
    is_defective: bool  = False   # True → route to reject bin
    defect_score: float = 0.0

    def __str__(self) -> str:
        if self.task_type == TaskType.IDLE:
            return "TASK: IDLE"
        if self.task_type == TaskType.ABORT:
            return "TASK: ABORT (target moved — recalculating)"
        if self.task_type == TaskType.RECOVERING:
            return f"TASK: RECOVERING → [{self.bin_label}]"
        defect_tag = " [DEFECTIVE⚠]" if self.is_defective else ""
        return (
            f"TASK: {self.task_type.name}{defect_tag} | "
            f"[bold]{self.class_name}[/bold] (track #{self.track_id}) "
            f"conf={self.confidence:.0%} priority={self.priority:.3f} "
            f"→ [{self.bin_label}]"
        )


class DecisionEngine:
    """
    State machine with temporal consensus, priority-based selection,
    adaptive recovery, and quality-control integration.

    Args:
        mode:           "sort" or "pick".
        confirm_frames: Frames required before first pick action.
        enable_qc:      Enable quality control inspection before pick.
        abort_distance_m: If a tracked target moves more than this during
                          approach, abort and recalculate (adaptive recovery).
    """

    def __init__(
        self,
        mode:             str   = "sort",
        confirm_frames:   int   = 1,
        enable_qc:        bool  = True,
        abort_distance_m: float = 0.04,
    ) -> None:
        self.mode             = mode
        self._confirm         = confirm_frames
        self._enable_qc       = enable_qc
        self._abort_dist      = abort_distance_m

        self._tracker = CentroidTracker(
            max_disappeared = 12,
            max_distance_px = 80.0,
        )

        self._last_pick_t:      float            = 0.0
        self._class_last_pick:  Dict[str, float] = defaultdict(float)
        self._picked_tracks:    Set[int]          = set()

        # Adaptive recovery: track the in-progress pick target
        self._active_track_id:   Optional[int]         = None
        self._active_target_xyz: Optional[np.ndarray]  = None
        self._in_approach:       bool                   = False

        # QA inspector (lazy init to avoid import cycle)
        self._qc = None
        if enable_qc:
            try:
                from src.logic.quality_control import QualityInspector
                self._qc = QualityInspector()
                log.info("QualityInspector integrated into DecisionEngine")
            except Exception as e:
                log.warning(f"QC init failed: {e} — running without QC")

        self._total_frames = 0
        self._pick_count   = 0
        self._defect_count = 0
        self._abort_count  = 0
        
        # Plan Queue for multi-object sequences
        self.plan_queue: Deque[int] = deque()
        self._last_scene_hash = ""
        
        # Persistent memory for picked tracks (track_id -> expiration_time)
        self._picked_history: Dict[int, float] = {}

        log.info(
            f"Intelligent Task Planner v3 | mode={mode.upper()} "
            f"| confirm={confirm_frames} frames "
            f"| qc={'on' if enable_qc else 'off'} "
            f"| abort_dist={abort_distance_m:.3f}m"
        )

    # ── Main step ─────────────────────────────────────────────────────────────
    def decide(
        self,
        detections: list,
        frame_area: int = 640 * 480,
        frame: Optional[np.ndarray] = None,
    ) -> RobotTask:
        """
        Args:
            detections:  List of Detection objects from ObjectDetector.
            frame_area:  Camera resolution in pixels (for normalisation).
            frame:       Optional current BGR frame (for QC inspection).

        Returns:
            RobotTask.
        """
        self._total_frames += 1
        if hasattr(detections, "detections"):
            detections = detections.detections

        detections = [
            d for d in detections
            if getattr(d, "area_px", 0) >= MIN_AREA_PX
        ]
        active_tracks = self._tracker.update(detections)

        # ── Adaptive recovery check ───────────────────────────────────────────
        if self._in_approach and self._active_track_id is not None:
            active_track = active_tracks.get(self._active_track_id)
            if active_track is None or active_track.world_xyz is None:
                log.warning(
                    f"Adaptive Recovery: track #{self._active_track_id} "
                    "lost during approach — aborting"
                )
                self._abort_approach()
                return RobotTask(TaskType.ABORT)

            drift = float(
                np.linalg.norm(active_track.world_xyz - self._active_target_xyz)
            )
            if drift > self._abort_dist:
                log.warning(
                    f"Adaptive Recovery: target drifted {drift:.3f}m "
                    f"(> {self._abort_dist:.3f}m) — recalculating"
                )
                self._abort_count += 1
                self._abort_approach()

                new_xyz   = active_track.world_xyz.copy()
                bl        = SORT_MAP.get(active_track.class_name, "unknown")
                drop_xyz  = DROP_ZONES.get(bl, DROP_ZONES["unknown"])
                return RobotTask(
                    task_type  = TaskType.RECOVERING,
                    target_xyz = new_xyz,
                    drop_xyz   = drop_xyz.copy(),
                    class_name = active_track.class_name,
                    bin_label  = bl,
                    confidence = active_track.smoothed_confidence,
                    track_id   = self._active_track_id,
                    priority   = 1.0,
                )

        # ── Time gates ────────────────────────────────────────────────────────
        now = time.monotonic()
        if (now - self._last_pick_t) < GLOBAL_COOLDOWN_S:
            return RobotTask(TaskType.IDLE)

        # Gather candidates
        candidates: List[Track] = []
        scene_summary = defaultdict(int)
        
        for track in active_tracks.values():
            if (track.age + 1) < self._confirm:
                continue
            if track.world_xyz is None:
                continue
            
            # Persistent memory check
            if track.track_id in self._picked_history:
                if (now - self._picked_history[track.track_id]) < 30.0: # 30s cooldown
                    continue
                else:
                    del self._picked_history[track.track_id]
            
            if track.track_id in self._picked_tracks:
                continue
            
            # Scene graph count
            bin_category = SORT_MAP.get(track.class_name, "unknown")
            scene_summary[bin_category] += 1
                
            if (now - self._class_last_pick[track.class_name]) < CLASS_COOLDOWN_S:
                continue
            candidates.append(track)

        # ── Environment Analysis (Scene Graph) ────────────────────────────────
        # Print scene analysis if it changed significantly (using simple hash)
        scene_hash = str(dict(scene_summary))
        if scene_hash != self._last_scene_hash and sum(scene_summary.values()) > 0:
            scene_desc = ", ".join(f"{v} {k}" for k, v in scene_summary.items())
            log.info(f"[AI Planner] 👁️ Scene Analyzed: {sum(scene_summary.values())} actionable objects ({scene_desc})")
            self._last_scene_hash = scene_hash

        if not candidates:
            return RobotTask(TaskType.IDLE)

        # ── Semantic Prioritization & Plan Generation ─────────────────────────
        # Score: Semantic Priority (dominates) + confidence + age + proximity
        cx_frame, cy_frame = 320, 240
        scored_candidates = []
        
        for t in candidates:
            bl = SORT_MAP.get(t.class_name, "unknown")
            sem_priority = PRIORITY_MAP.get(bl, 0)
            
            age_bonus  = min(t.age / 20.0, 1.0)
            cent_dist  = np.linalg.norm(t.centroid - np.array([cx_frame, cy_frame]))
            prox_bonus = 1.0 / (1.0 + cent_dist / 300.0)
            
            # Final score weighting
            score = (sem_priority * 1000) + (t.smoothed_confidence * 100) + (age_bonus * 30) + (prox_bonus * 10)
            scored_candidates.append((score, t))
            
        # Sort highest score first
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Build plan queue if empty
        if not self.plan_queue:
            self.plan_queue.extend([t.track_id for _, t in scored_candidates])
            plan_desc = " -> ".join([f"Track #{t.track_id} ({SORT_MAP.get(t.class_name,'unknown')})" for _, t in scored_candidates])
            log.info(f"[AI Planner] 📝 Generated Plan: [{plan_desc}]")
            
        # Extract best valid track from plan queue
        best: Optional[Track] = None
        best_score = 0.0
        
        while self.plan_queue:
            target_id = self.plan_queue[0]
            # Check if still valid candidate
            valid = [item for item in scored_candidates if item[1].track_id == target_id]
            if valid:
                best_score, best = valid[0]
                break
            else:
                # Track lost or became invalid (e.g. cooldown), pop from plan
                self.plan_queue.popleft()

        if best is None:
            return RobotTask(TaskType.IDLE)

        # ── Preemption check ──────────────────────────────────────────────────
        if self._in_approach and self._active_track_id is not None:
            active_prio = PRIORITY_MAP.get(SORT_MAP.get(active_tracks[self._active_track_id].class_name, ""), 0)
            best_prio   = PRIORITY_MAP.get(SORT_MAP.get(best.class_name, ""), 0)
            
            if best_prio > active_prio + 20: # Significant priority jump
                log.warning(f"[AI Planner] ⚡ Preemption: High priority {best.class_name} detected. Switching from track #{self._active_track_id}")
                self._abort_approach()
                # Continue below to issue new task
            else:
                return RobotTask(TaskType.IDLE) # Keep current approach

        # ── Quality Control Inspection ────────────────────────────────────────
        is_defective = False
        defect_score = 0.0
        bin_label    = SORT_MAP.get(best.class_name, "unknown")

        if self._qc is not None and frame is not None:
            try:
                qc_report    = self._qc.inspect(
                    frame      = frame,
                    mask       = self._bbox_to_mask(frame, best.centroid),
                    class_name = best.class_name,
                )
                is_defective = qc_report.is_defective
                defect_score = qc_report.defect_score
                if is_defective:
                    bin_label = "reject"
                    self._defect_count += 1
                    log.warning(
                        f"QC: {best.class_name} DEFECTIVE "
                        f"(score={defect_score:.2f}) → reject bin"
                    )
            except Exception as e:
                log.debug(f"QC inspection failed: {e}")

        if is_defective and self._qc is not None:
            drop_xyz = self._qc.reject_zone
        else:
            drop_xyz = DROP_ZONES.get(bin_label, DROP_ZONES["unknown"])

        task = RobotTask(
            task_type    = TaskType.SORT if self.mode == "sort" else TaskType.PICK,
            target_xyz   = best.world_xyz.copy(),
            drop_xyz     = drop_xyz.copy(),
            class_name   = best.class_name,
            bin_label    = bin_label,
            confidence   = best.smoothed_confidence,
            track_id     = best.track_id,
            priority     = best_score,
            is_defective = is_defective,
            defect_score = defect_score,
        )

        # Commit + start adaptive approach tracking
        self._last_pick_t                      = now
        self._class_last_pick[best.class_name] = now
        self._picked_tracks.add(best.track_id)
        self._picked_history[best.track_id]    = now
        self._pick_count += 1

        self._active_track_id   = best.track_id
        self._active_target_xyz = best.world_xyz.copy()
        self._in_approach       = True

        # Pop from plan since we are committing to it
        if self.plan_queue and self.plan_queue[0] == best.track_id:
            self.plan_queue.popleft()

        log.info(f"[AI Planner] 🤖 Executing Task: Prioritizing Track #{best.track_id} ({best.class_name}) due to rule: {bin_label.upper()} priority.")
        return task

    # ── Notify pick complete (stops adaptive tracking) ────────────────────────
    def notify_pick_complete(self) -> None:
        """Call this when the arm has physically completed the pick motion."""
        self._abort_approach()
        log.debug("Pick complete — approach tracking stopped")

    def _abort_approach(self) -> None:
        self._in_approach       = False
        self._active_track_id   = None
        self._active_target_xyz = None

    # ── Mask helper ───────────────────────────────────────────────────────────
    @staticmethod
    def _bbox_to_mask(
        frame:    np.ndarray,
        centroid: np.ndarray,
        size_px:  int = 60,
    ) -> np.ndarray:
        """Fallback: create a square mask around the centroid for QC."""
        H, W  = frame.shape[:2]
        h     = size_px // 2
        cx, cy = int(centroid[0]), int(centroid[1])
        mask   = np.zeros((H, W), dtype=np.uint8)
        y0, y1 = max(0, cy - h), min(H, cy + h)
        x0, x1 = max(0, cx - h), min(W, cx + h)
        mask[y0:y1, x0:x1] = 1
        return mask

    # ── Reset picked set (call after arm returns to home) ─────────────────────
    def reset_picked(self) -> None:
        """Allow previously-picked tracks to be picked again."""
        self._picked_tracks.clear()
        log.debug("Picked-tracks set cleared")

    @property
    def stats(self) -> dict:
        return {
            "total_frames":  self._total_frames,
            "pick_count":    self._pick_count,
            "defect_count":  self._defect_count,
            "abort_count":   self._abort_count,
            "active_tracks": len(self._tracker.active_tracks),
            "pick_rate":     self._pick_count / max(self._total_frames, 1),
        }
