"""
tracker.py — Multi-Object Centroid Tracker with Kalman Smoothing
=================================================================
Assigns consistent IDs to detections across frames so the decision
engine doesn't re-pick the same object twice and can require an
object to be *confirmed* over N consecutive frames before acting.

Algorithm:
  1. Compute pairwise centroid distances between existing tracks and
     new detections (Hungarian assignment via scipy).
  2. Update matched tracks with Kalman-filtered position.
  3. Deregister tracks unseen for > max_disappeared frames.

This is a classic approach — similar to SORT (Simple Online and
Realtime Tracking) but without the IoU matching overhead, which
is overkill for a slow-moving arm pick-and-place scenario.

Reference:
  Bewley et al. "Simple Online and Realtime Tracking" ICIP 2016.
  (We deliberately use a lighter version for embedded performance.)
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.vision.detect import Detection
from src.utils.logger import get_logger

log = get_logger("vision.tracker")

# ── Kalman Filter parameters ──────────────────────────────────────────────────
# State vector: [x, y, vx, vy]  (centroid position + velocity)
# Measurement:  [x, y]

_F = np.array([           # State transition
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)

_H = np.array([           # Measurement model
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=np.float32)

_Q = np.eye(4, dtype=np.float32) * 0.03   # process noise
_R = np.eye(2, dtype=np.float32) * 2.0    # measurement noise


@dataclass
class Track:
    """A single tracked object."""
    track_id:        int
    class_name:      str
    confidence:      float
    centroid:        np.ndarray          # [x, y] smoothed
    world_xyz:       Optional[np.ndarray]
    disappeared:     int = 0
    age:             int = 0             # frames since creation

    # Kalman state [x, y, vx, vy] and covariance
    kf_x:            np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    kf_P:            np.ndarray = field(default_factory=lambda: np.eye(4, np.float32) * 10)

    # Detection history for temporal consensus
    conf_history:    deque = field(default_factory=lambda: deque(maxlen=10))

    def predict(self) -> None:
        """Kalman predict step (call each frame even if not matched)."""
        self.kf_x = _F @ self.kf_x
        self.kf_P = _F @ self.kf_P @ _F.T + _Q
        self.centroid = self.kf_x[:2].copy()

    def update(self, measurement: np.ndarray, conf: float) -> None:
        """Kalman update step with new centroid measurement."""
        z = measurement.astype(np.float32)
        S = _H @ self.kf_P @ _H.T + _R
        K = self.kf_P @ _H.T @ np.linalg.inv(S)
        self.kf_x += K @ (z - _H @ self.kf_x)
        self.kf_P = (np.eye(4) - K @ _H) @ self.kf_P
        self.centroid       = self.kf_x[:2].copy()
        self.disappeared    = 0
        self.age           += 1
        self.confidence     = conf
        self.conf_history.append(conf)

    @property
    def smoothed_confidence(self) -> float:
        """Exponentially weighted average confidence."""
        if not self.conf_history:
            return self.confidence
        weights = np.exp(np.linspace(-1, 0, len(self.conf_history)))
        weights /= weights.sum()
        return float(np.dot(weights, list(self.conf_history)))

    @property
    def is_confirmed(self) -> bool:
        """Track must survive at least 3 frames to be considered real."""
        return self.age >= 3


class CentroidTracker:
    """
    Assigns persistent track IDs to detections across frames.

    Args:
        max_disappeared: Frames a track can be unmatched before deletion.
        max_distance_px: Max centroid distance (px) for a valid match.
    """

    def __init__(
        self,
        max_disappeared:  int   = 10,
        max_distance_px:  float = 80.0,
    ) -> None:
        self._next_id         = 0
        self._tracks:  Dict[int, Track] = OrderedDict()
        self._max_gone        = max_disappeared
        self._max_dist        = max_distance_px

    # ── Update ────────────────────────────────────────────────────────────────
    def update(self, detections: List[Detection]) -> Dict[int, Track]:
        """
        Match detections to existing tracks, update Kalman filters,
        create/delete tracks as needed.

        Args:
            detections: Raw detections from ObjectDetector for one frame.

        Returns:
            Dict of active track_id → Track.
        """
        # Kalman predict all existing tracks
        for track in self._tracks.values():
            track.predict()

        if not detections:
            for track in self._tracks.values():
                track.disappeared += 1
            self._prune()
            return dict(self._tracks)

        det_centroids = np.array(
            [d.center_px for d in detections], dtype=np.float32
        )

        if not self._tracks:
            for det in detections:
                self._register(det)
            return dict(self._tracks)

        # Build cost matrix: Euclidean distance
        track_ids  = list(self._tracks.keys())
        trk_cents  = np.array(
            [self._tracks[tid].centroid for tid in track_ids], dtype=np.float32
        )

        cost = np.linalg.norm(
            trk_cents[:, None, :] - det_centroids[None, :, :], axis=2
        )  # shape (n_tracks, n_dets)

        # Hungarian assignment
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_trk  = set()
        matched_det  = set()

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] > self._max_dist:
                continue  # too far — treat as new object

            tid = track_ids[r]
            det = detections[c]
            meas = np.array(det.center_px, dtype=np.float32)

            self._tracks[tid].update(meas, det.confidence)
            self._tracks[tid].class_name = det.class_name
            self._tracks[tid].world_xyz  = det.world_xyz

            matched_trk.add(tid)
            matched_det.add(c)

        # Unmatched tracks → increment disappeared
        for i, tid in enumerate(track_ids):
            if tid not in matched_trk:
                self._tracks[tid].disappeared += 1

        # Unmatched detections → new tracks
        for i, det in enumerate(detections):
            if i not in matched_det:
                self._register(det)

        self._prune()
        return dict(self._tracks)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _register(self, det: Detection) -> None:
        track = Track(
            track_id   = self._next_id,
            class_name = det.class_name,
            confidence = det.confidence,
            centroid   = np.array(det.center_px, dtype=np.float32),
            world_xyz  = det.world_xyz,
        )
        track.kf_x[:2] = track.centroid
        track.conf_history.append(det.confidence)
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _prune(self) -> None:
        dead = [tid for tid, t in self._tracks.items()
                if t.disappeared > self._max_gone]
        for tid in dead:
            log.debug(f"Track {tid} ({self._tracks[tid].class_name}) deregistered")
            del self._tracks[tid]

    @property
    def active_tracks(self) -> List[Track]:
        return list(self._tracks.values())

    @property
    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.is_confirmed]
