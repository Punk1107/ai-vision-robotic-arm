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

# ── Filter stack integration ──────────────────────────────────────────
try:
    from src.filters.signal import EMAFilter
    _EMA_AVAILABLE = True
except ImportError:
    _EMA_AVAILABLE = False

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

# Pre-built identity matrices reused every frame (avoid repeated allocation)
_I4 = np.eye(4, dtype=np.float32)
_I3 = np.eye(3, dtype=np.float32)

# 3D World Filter (X, Y, Z position only)
_F_3D = np.eye(3, dtype=np.float32)
_H_3D = np.eye(3, dtype=np.float32)
_Q_3D = np.eye(3, dtype=np.float32) * 0.001 # lower noise for world coords
_R_3D = np.eye(3, dtype=np.float32) * 0.01

# Precomputed 3D Kalman denominator (P_xyz + R_3D)^-1 is trivially P/(P+R)
# since both _H_3D and _F_3D are identity. Cached to avoid per-frame rebuild.


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
    kf_P:            np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32) * 10)

    # 3D Kalman state [x, y, z] and covariance
    kf_xyz:          np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    kf_P_xyz:        np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32) * 1.0)

    # Detection history for temporal consensus
    conf_history:    deque = field(default_factory=lambda: deque(maxlen=10))

    # EMAFilter for smoothed confidence (replaces manual exponential weighting)
    _conf_ema:       object = field(default=None, init=False, repr=False)

    def predict(self) -> None:
        """Kalman predict step (call each frame even if not matched)."""
        # 2D Centroid predict
        self.kf_x = _F @ self.kf_x
        self.kf_P = _F @ self.kf_P @ _F.T + _Q
        self.centroid = self.kf_x[:2].copy()

        # 3D World predict
        self.kf_P_xyz = _F_3D @ self.kf_P_xyz @ _F_3D.T + _Q_3D

    def update(self, measurement: np.ndarray, conf: float, world_xyz: Optional[np.ndarray] = None) -> None:
        """Kalman update step with new centroid measurement."""
        # 2D Update — S is 2×2; use closed-form inverse to avoid LAPACK dispatch
        z = measurement.astype(np.float32)
        S = _H @ self.kf_P @ _H.T + _R   # 2×2
        # Closed-form 2×2 inverse: inv([[a,b],[c,d]]) = 1/(ad-bc) * [[d,-b],[-c,a]]
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if abs(det) < 1e-9:
            det = 1e-9   # numerical guard
        S_inv = np.array([
            [ S[1, 1], -S[0, 1]],
            [-S[1, 0],  S[0, 0]],
        ], dtype=np.float32) / det
        K = self.kf_P @ _H.T @ S_inv
        self.kf_x += K @ (z - _H @ self.kf_x)
        self.kf_P = (_I4 - K @ _H) @ self.kf_P
        self.centroid = self.kf_x[:2].copy()

        # 3D Update — since _H_3D = _F_3D = I, the KF reduces to a simple
        # per-element scalar update. S3 = P_xyz + R_3D (both diagonal).
        if world_xyz is not None:
            z3 = world_xyz.astype(np.float32)
            # S3 diagonal: Pii + R_3D_ii; K3 = P / (P + R) (element-wise)
            s3_diag = np.diag(self.kf_P_xyz) + np.diag(_R_3D)  # (3,)
            k3_diag = np.diag(self.kf_P_xyz) / np.maximum(s3_diag, 1e-9)  # (3,)
            innov   = z3 - self.kf_xyz            # (3,)
            self.kf_xyz   += k3_diag * innov
            # Joseph form covariance update (numerically stable)
            self.kf_P_xyz  = ((1 - k3_diag)[:, None] * self.kf_P_xyz
                              + np.diag((k3_diag ** 2) * np.diag(_R_3D)))
            self.world_xyz = self.kf_xyz.copy()

        self.disappeared    = 0
        self.age           += 1
        self.confidence     = conf
        self.conf_history.append(conf)

    @property
    def smoothed_confidence(self) -> float:
        """EMA-smoothed confidence (uses EMAFilter from filters.signal if available)."""
        if not self.conf_history:
            return self.confidence

        if _EMA_AVAILABLE:
            # Use EMAFilter with α=0.3 (heavier smoothing for confidence scores)
            if self._conf_ema is None:
                # Lazy init: can't use mutable default in dataclass
                object.__setattr__(self, '_conf_ema', EMAFilter(alpha=0.3, initial_value=self.confidence))
            # Re-run EMA over full history to get current smoothed value
            ema = EMAFilter(alpha=0.3, initial_value=list(self.conf_history)[0])
            result = self.confidence
            for c in list(self.conf_history):
                result = ema.update(c)
            return result
        else:
            # Legacy fallback: manual exponential weighting
            weights = np.exp(np.linspace(-1, 0, len(self.conf_history)))
            weights /= weights.sum()
            return float(np.dot(weights, list(self.conf_history)))

    @property
    def is_confirmed(self) -> bool:
        """Track must survive at least 3 frames to be considered real."""
        return self.age >= 2


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

            self._tracks[tid].update(meas, det.confidence, world_xyz=det.world_xyz)
            self._tracks[tid].class_name = det.class_name

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
        if det.world_xyz is not None:
            track.kf_xyz = det.world_xyz.astype(np.float32)
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
