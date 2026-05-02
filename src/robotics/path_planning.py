"""
path_planning.py — RRT* Collision-Free Path Planner
====================================================
Implements RRT* (Optimal Rapidly-exploring Random Trees) in the robot's
3D Cartesian workspace.  The planner:

  1. Maintains a 3D occupancy grid of known obstacles.
  2. Grows a tree from the start configuration toward the goal.
  3. Uses the RRT* rewiring step to asymptotically approach the optimal path.
  4. Returns a list of 3D waypoints that are then fed to the IK solver
     and trajectory planner.

Why Cartesian-space RRT* instead of joint-space?
  - Obstacle avoidance is naturally expressed in 3D space.
  - The arm's workspace is relatively small, so Cartesian planning
    is computationally cheap enough to run in real-time (< 50ms).
  - The resulting waypoints plug directly into the IK solver.

Reference:
  Karaman & Frazzoli, "Sampling-based Algorithms for Optimal Motion
  Planning," IJRR 2011.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("robotics.path_planning")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_ITER       = 2000       # RRT* iterations
STEP_SIZE_M    = 0.025      # metres per extension step
GOAL_BIAS      = 0.15       # probability of sampling goal directly
REWIRE_RADIUS  = 0.08       # RRT* neighbour search radius (m)
GOAL_THRESHOLD = 0.015      # metres — "close enough" to goal


@dataclass
class Obstacle:
    """Axis-aligned bounding box obstacle in robot frame."""
    center: np.ndarray    # (3,) [X, Y, Z] metres
    half:   np.ndarray    # (3,) half-extents [dx, dy, dz] metres

    def contains(self, pt: np.ndarray, margin: float = 0.01) -> bool:
        """True if pt is inside (with optional safety margin)."""
        diff = np.abs(pt - self.center)
        return bool(np.all(diff <= self.half + margin))


@dataclass
class _Node:
    pos:    np.ndarray
    parent: Optional["_Node"]      = None
    cost:   float                  = 0.0
    children: List["_Node"]        = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# RRTStarPlanner
# ─────────────────────────────────────────────────────────────────────────────

class RRTStarPlanner:
    """
    Optimal RRT* path planner in 3D Cartesian robot space.

    Usage::

        planner = RRTStarPlanner()
        planner.add_obstacle(Obstacle(center=np.array([0.15, 0.2, 0.1]),
                                      half=np.array([0.05, 0.05, 0.1])))
        waypoints = planner.plan(
            start=np.array([0.0, 0.15, 0.05]),
            goal=np.array([0.18, 0.28, 0.05]),
        )
        # waypoints: List[np.ndarray], each shape (3,)
    """

    def __init__(
        self,
        max_iter:      int   = MAX_ITER,
        step_size:     float = STEP_SIZE_M,
        goal_bias:     float = GOAL_BIAS,
        rewire_radius: float = REWIRE_RADIUS,
        goal_thresh:   float = GOAL_THRESHOLD,
    ) -> None:
        self._max_iter      = max_iter
        self._step          = step_size
        self._goal_bias     = goal_bias
        self._rewire_r      = rewire_radius
        self._goal_thresh   = goal_thresh
        self._obstacles:    List[Obstacle] = []

        # Workspace bounds from config
        ws = config.robotics
        self._x_bounds = ws.workspace_x
        self._y_bounds = ws.workspace_y
        self._z_bounds = ws.workspace_z

        log.info(
            f"RRT* Planner | max_iter={max_iter} | step={step_size:.3f}m | "
            f"rewire_r={rewire_radius:.3f}m"
        )

    # ── Obstacle management ───────────────────────────────────────────────────
    def add_obstacle(self, obs: Obstacle) -> None:
        self._obstacles.append(obs)
        log.debug(f"Obstacle added: center={obs.center} half={obs.half}")

    def clear_obstacles(self) -> None:
        self._obstacles.clear()

    def update_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Replace current obstacle list (call each frame if obstacles move)."""
        self._obstacles = obstacles

    # ── Main plan method ──────────────────────────────────────────────────────
    def plan(
        self,
        start: np.ndarray,
        goal:  np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """
        Compute a collision-free path from start to goal.

        Args:
            start: Start position (3,) in robot frame (m).
            goal:  Goal  position (3,) in robot frame (m).

        Returns:
            List of 3D waypoints (including start and goal),
            or None if no path was found within max_iter.
        """
        t0 = time.perf_counter()

        # Fast check: if start→goal is collision-free, return direct path
        if self._collision_free(start, goal):
            log.info("RRT*: Direct path is collision-free — skipping tree growth.")
            return [start.copy(), goal.copy()]

        root = _Node(pos=start.copy())
        tree: List[_Node] = [root]
        goal_node: Optional[_Node] = None

        for _ in range(self._max_iter):
            # Sample
            q_rand = (goal if random.random() < self._goal_bias
                      else self._random_sample())

            # Nearest
            q_near = self._nearest(tree, q_rand)

            # Steer
            q_new_pos = self._steer(q_near.pos, q_rand)

            if not self._collision_free(q_near.pos, q_new_pos):
                continue

            # Find neighbours in rewire radius
            neighbours = self._near(tree, q_new_pos, self._rewire_r)

            # Choose best parent
            best_parent = q_near
            best_cost   = q_near.cost + np.linalg.norm(q_new_pos - q_near.pos)

            for nb in neighbours:
                if not self._collision_free(nb.pos, q_new_pos):
                    continue
                c = nb.cost + np.linalg.norm(q_new_pos - nb.pos)
                if c < best_cost:
                    best_cost   = c
                    best_parent = nb

            q_new = _Node(pos=q_new_pos, parent=best_parent, cost=best_cost)
            best_parent.children.append(q_new)
            tree.append(q_new)

            # Rewire neighbours through q_new
            for nb in neighbours:
                if nb is best_parent:
                    continue
                c = q_new.cost + np.linalg.norm(nb.pos - q_new_pos)
                if c < nb.cost and self._collision_free(q_new_pos, nb.pos):
                    # Detach from old parent
                    if nb.parent and nb in nb.parent.children:
                        nb.parent.children.remove(nb)
                    nb.parent = q_new
                    nb.cost   = c
                    q_new.children.append(nb)

            # Check if goal reached
            if np.linalg.norm(q_new_pos - goal) <= self._goal_thresh:
                if goal_node is None or q_new.cost < goal_node.cost:
                    goal_node = q_new

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if goal_node is None:
            log.warning(
                f"RRT*: No path found in {self._max_iter} iterations "
                f"({elapsed_ms:.1f}ms)"
            )
            return None

        path = self._extract_path(goal_node)
        # Append exact goal
        path.append(goal.copy())

        log.info(
            f"RRT*: Path found | {len(path)} waypoints | "
            f"cost={goal_node.cost:.3f}m | {elapsed_ms:.1f}ms"
        )
        return path

    # ── Tree operations ───────────────────────────────────────────────────────
    def _random_sample(self) -> np.ndarray:
        return np.array([
            random.uniform(*self._x_bounds),
            random.uniform(*self._y_bounds),
            random.uniform(*self._z_bounds),
        ], dtype=np.float64)

    def _nearest(self, tree: List[_Node], pt: np.ndarray) -> _Node:
        dists = [np.linalg.norm(n.pos - pt) for n in tree]
        return tree[int(np.argmin(dists))]

    def _near(self, tree: List[_Node], pt: np.ndarray, r: float) -> List[_Node]:
        return [n for n in tree if np.linalg.norm(n.pos - pt) <= r]

    def _steer(self, from_pt: np.ndarray, to_pt: np.ndarray) -> np.ndarray:
        diff = to_pt - from_pt
        dist = np.linalg.norm(diff)
        if dist <= self._step:
            return to_pt.copy()
        return from_pt + diff / dist * self._step

    def _collision_free(self, a: np.ndarray, b: np.ndarray, n_checks: int = 8) -> bool:
        """Discretise segment a→b and check each point against obstacles."""
        if not self._obstacles:
            return True
        for t in np.linspace(0, 1, n_checks):
            pt = a + t * (b - a)
            for obs in self._obstacles:
                if obs.contains(pt):
                    return False
        return True

    def _extract_path(self, node: _Node) -> List[np.ndarray]:
        path: List[np.ndarray] = []
        cur: Optional[_Node] = node
        while cur is not None:
            path.append(cur.pos.copy())
            cur = cur.parent
        path.reverse()
        return path

    # ── Path smoothing ────────────────────────────────────────────────────────
    def smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """
        Greedy path shortcutting: iteratively remove unnecessary waypoints
        when the direct segment is collision-free.
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._collision_free(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j

        log.debug(
            f"Path smoothed: {len(path)} → {len(smoothed)} waypoints"
        )
        return smoothed
