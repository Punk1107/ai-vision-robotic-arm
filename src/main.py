#!/usr/bin/env python3
"""
main.py — AI Vision + Robotic Arm — Dual-Thread Pipeline (v2)
=============================================================
Architecture:
  Thread 1 (vision)  : Camera → Preprocess → YOLO → Depth → Track → Decide
  Thread 2 (control) : Consume RobotTask → IK → Trajectory → Arm

The two threads communicate through a thread-safe task queue so the
30fps vision loop is never blocked by the ~1-2s arm motion.

Performance improvements:
  - Vision runs independently of arm motion
  - Frame-skip YOLO (inference every N frames, tracker interpolates)
  - Kalman-filtered object positions (no jitter pick targets)
  - Cubic spline trajectory (smooth multi-waypoint execution)
  - Live performance dashboard (FPS, latency, pick count)
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from typing import Optional

import click
import cv2
import numpy as np

from src.utils.config import config, load_config
from src.utils.logger import setup_logger, get_logger

from src.vision.preprocess  import CameraPreprocessor
from src.vision.detect      import ObjectDetector
from src.vision.depth       import CoordinateMapper, MonocularDepthEstimator

from src.robotics.kinematics  import IKSolver, JointAngles
from src.robotics.control     import RobotController
from src.robotics.trajectory  import CubicSplinePlanner, pick_place_trajectory

from src.logic.decision import DecisionEngine, TaskType

log = get_logger("main")

_HOME = JointAngles(base=0, shoulder=90, elbow=0, wrist=0, gripper=0)


# ─────────────────────────────────────────────────────────────────────────────
# Performance monitor (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

class PerfMonitor:
    def __init__(self, window: int = 60) -> None:
        self._lock       = threading.Lock()
        self._fps_times: list = []
        self._ik_ms:     list = []
        self._window     = window
        self.pick_count  = 0
        self.drop_count  = 0

    def record_frame(self) -> None:
        with self._lock:
            now = time.perf_counter()
            self._fps_times.append(now)
            if len(self._fps_times) > self._window:
                self._fps_times.pop(0)

    def record_ik(self, ms: float) -> None:
        with self._lock:
            self._ik_ms.append(ms)
            if len(self._ik_ms) > self._window:
                self._ik_ms.pop(0)

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._fps_times) < 2:
                return 0.0
            return (len(self._fps_times) - 1) / (
                self._fps_times[-1] - self._fps_times[0] + 1e-9
            )

    @property
    def avg_ik_ms(self) -> float:
        with self._lock:
            return float(np.mean(self._ik_ms)) if self._ik_ms else 0.0

    def summary(self) -> str:
        return (
            f"FPS={self.fps:.1f} | IK={self.avg_ik_ms:.1f}ms | "
            f"picks={self.pick_count} | drops={self.drop_count}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dual-thread pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ArmPipeline:
    """
    Two-thread pick-and-place pipeline.

    Vision thread  → task_queue → Control thread
    """

    def __init__(self, mode: str = "sort", frame_skip: int = 2) -> None:
        self._mode       = mode
        self._frame_skip = frame_skip
        self._running    = False

        # Shared task queue (maxsize=1 prevents stale tasks piling up)
        self._task_queue: queue.Queue = queue.Queue(maxsize=1)

        # Performance monitor
        self._perf = PerfMonitor()

        # Components (init in setup())
        self._cap:        Optional[cv2.VideoCapture]        = None
        self._pre:        Optional[CameraPreprocessor]      = None
        self._detector:   Optional[ObjectDetector]          = None
        self._mapper:     Optional[CoordinateMapper]        = None
        self._depth_est:  Optional[MonocularDepthEstimator] = None
        self._ik:         Optional[IKSolver]                = None
        self._ctrl:       Optional[RobotController]         = None
        self._decision:   Optional[DecisionEngine]          = None
        self._planner:    Optional[CubicSplinePlanner]      = None
        self._current_ja: JointAngles                       = _HOME

    # ── Setup ─────────────────────────────────────────────────────────────────
    def setup(self) -> None:
        log.info("=" * 64)
        log.info("  AI Vision + Robotic Arm — Dual-Thread Pipeline v2")
        log.info("=" * 64)

        cam = config.camera
        self._cap = cv2.VideoCapture(cam.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
        self._cap.set(cv2.CAP_PROP_FPS,          cam.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimise latency

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam.device_id}")

        log.success(f"Camera: dev={cam.device_id} {cam.width}×{cam.height}@{cam.fps}fps ✓")

        self._pre      = CameraPreprocessor(equalize_hist=False)
        self._detector = ObjectDetector(frame_skip=self._frame_skip)
        self._mapper   = CoordinateMapper(
            strategy = "depth" if config.depth.enabled else "plane"
        )

        if config.depth.enabled and config.depth.method == "monocular":
            self._depth_est = MonocularDepthEstimator(config.depth.monocular_model)

        self._ik      = IKSolver(elbow_up=True)
        self._planner = CubicSplinePlanner(hz=50)
        self._ctrl    = RobotController()

        if not self._ctrl.connect():
            log.warning("Arm not connected — continuing in DRY RUN mode.")

        self._ctrl.home(blocking=True)
        self._decision = DecisionEngine(mode=self._mode, confirm_frames=4)

        log.success("Pipeline ready. Press Ctrl+C or 'q' in video window to stop.\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self) -> None:
        self._running = True

        vision_thread  = threading.Thread(
            target=self._vision_loop, name="vision", daemon=True
        )
        control_thread = threading.Thread(
            target=self._control_loop, name="control", daemon=True
        )

        vision_thread.start()
        control_thread.start()

        log.info("Both threads started. Waiting ...")
        vision_thread.join()
        self._running = False

        # Signal control thread to exit
        try:
            self._task_queue.put_nowait(None)
        except queue.Full:
            pass
        control_thread.join(timeout=5.0)

        self._cleanup()

    # ── Vision thread ─────────────────────────────────────────────────────────
    def _vision_loop(self) -> None:
        log.info("[vision] thread started")

        while self._running:
            ret, raw = self._cap.read()
            if not ret:
                log.error("[vision] Camera read failed"); break

            self._perf.record_frame()

            # 1. Preprocess
            frame = self._pre.process(raw)

            # 2. Depth (optional)
            depth_map = (
                self._depth_est.estimate(frame)
                if self._depth_est else None
            )

            # 3. Detect
            result = self._detector.detect(frame)

            # 4. Coordinate map
            for det in result.detections:
                cx, cy      = det.center_px
                det.world_xyz = self._mapper.map(cx, cy, depth_map)

            # 5. Decide (tracker runs inside decision engine)
            task = self._decision.decide(result.detections, frame.shape[0]*frame.shape[1])

            # 6. Push task (non-blocking: drop if control still busy)
            if task.task_type != TaskType.IDLE:
                try:
                    self._task_queue.put_nowait(task)
                except queue.Full:
                    pass   # control thread busy, skip frame

            # 7. Visualise
            if config.debug_video:
                vis = self._detector.annotate_frame(frame, result)
                self._draw_hud(vis, task)
                cv2.imshow("AI Robotic Arm — Vision", vis)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("[vision] 'q' pressed — stopping.")
                    break

        self._running = False
        log.info("[vision] thread stopped")

    # ── Control thread ────────────────────────────────────────────────────────
    def _control_loop(self) -> None:
        log.info("[control] thread started")

        while self._running:
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:   # sentinel
                break

            self._execute_task(task)
            self._task_queue.task_done()

        log.info("[control] thread stopped")

    # ── Task execution ────────────────────────────────────────────────────────
    def _execute_task(self, task) -> None:
        ik = self._ik

        t_pick = task.target_xyz
        t_drop = task.drop_xyz

        if t_pick is None or t_drop is None:
            return

        # Approach: straight down (wrist -90°)
        t0 = time.perf_counter()
        pick_angles = ik.solve(t_pick, wrist_pitch_deg=-90,
                               current=self._current_ja)
        self._perf.record_ik((time.perf_counter() - t0) * 1000)

        if pick_angles is None:
            log.warning(f"IK failed for {t_pick}. Skipping task.")
            return

        # Lift: same XY, +8cm Z
        lift_xyz    = t_pick.copy(); lift_xyz[2] += 0.08
        lift_angles = ik.solve(lift_xyz, wrist_pitch_deg=-90)

        drop_angles = ik.solve(t_drop, wrist_pitch_deg=0,
                               current=lift_angles or pick_angles)

        if lift_angles is None or drop_angles is None:
            log.warning("IK failed for lift/drop — executing simpler pick.")
            self._ctrl.move_to(pick_angles, blocking=False)
            time.sleep(0.4)
            self._ctrl.grip(close=True)
            self._ctrl.home()
            self._ctrl.grip(close=False)
            self._perf.pick_count += 1
            self._current_ja = _HOME
            return

        # Build smooth trajectory: home → pick → lift → drop → home
        try:
            traj = pick_place_trajectory(
                home    = _HOME,
                pick    = pick_angles,
                lift    = lift_angles,
                drop    = drop_angles,
                planner = self._planner,
                t_per_segment = 1.0,
            )
        except Exception as e:
            log.warning(f"Trajectory planning failed: {e} — falling back to direct move.")
            traj = None

        if traj:
            # Inject gripper open/close at correct waypoints
            pick_frame = len(traj) // 4         # ~end of segment 1
            drop_frame = 3 * len(traj) // 4     # ~end of segment 3

            for i, pt in enumerate(traj):
                self._ctrl.move_to(pt.angles)
                if i == pick_frame:
                    self._ctrl.grip(close=True)
                if i == drop_frame:
                    self._ctrl.grip(close=False)
                    self._perf.drop_count += 1
                time.sleep(1 / 50)   # 50Hz playback
        else:
            # Fallback: direct moves
            for angles, delay in [
                (pick_angles,  0.6),
                (lift_angles,  0.4),
                (drop_angles,  0.6),
                (_HOME,        0.4),
            ]:
                self._ctrl.move_to(angles)
                time.sleep(delay)
            self._ctrl.grip(close=True)
            time.sleep(0.3)
            self._ctrl.grip(close=False)

        self._current_ja = _HOME
        self._perf.pick_count += 1
        self._decision.reset_picked()

        log.success(
            f"✓ Picked [{task.class_name}] → [{task.bin_label}] | "
            f"{self._perf.summary()}"
        )

    # ── HUD overlay ───────────────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray, task) -> None:
        h, w = frame.shape[:2]
        mode_str   = f"Mode: {self._mode.upper()}  FPS: {self._perf.fps:.1f}"
        task_str   = f"Task: {task.task_type.name}"
        if task.task_type != TaskType.IDLE:
            task_str += f"  [{task.class_name}] → [{task.bin_label}]"
        perf_str = f"Picks: {self._perf.pick_count}  IK: {self._perf.avg_ik_ms:.1f}ms"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 48), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"{mode_str}  |  {task_str}", (8, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 1, cv2.LINE_AA)
        cv2.putText(frame, perf_str, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def _cleanup(self) -> None:
        log.info("Shutting down ...")
        log.info(f"Final stats: {self._perf.summary()}")
        log.info(f"Decision: {self._decision.stats}")

        if self._ctrl:
            self._ctrl.home(blocking=True)
            self._ctrl.disconnect()

        if self._cap:
            self._cap.release()

        cv2.destroyAllWindows()

    def stop(self) -> None:
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--mode",       default="sort",
              type=click.Choice(["sort", "pick"]),
              show_default=True)
@click.option("--frame-skip", default=2,   type=int,   show_default=True,
              help="Run YOLO every N frames (tracker interpolates).")
@click.option("--dry",        is_flag=True, default=False)
@click.option("--port",       default=None)
@click.option("--conf",       default=None, type=float)
@click.option("--debug/--no-debug", default=True, show_default=True)
@click.option("--camera",     default=0,   type=int,   show_default=True)
def main(
    mode: str, frame_skip: int, dry: bool,
    port: str, conf: float, debug: bool, camera: int,
) -> None:
    """AI Vision + Robotic Arm — Dual-Thread Pipeline."""
    if dry:   config.dry_run = True
    if port:  config.robotics.serial_port = port
    if conf:  config.yolo.confidence_threshold = conf
    if not debug: config.debug_video = False
    config.camera.device_id = camera

    setup_logger(level=config.log_level, log_dir=config.log_dir)

    pipeline = ArmPipeline(mode=mode, frame_skip=frame_skip)

    def _sig(sig, _):
        log.warning(f"Signal {sig} — stopping ...")
        pipeline.stop()

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        pipeline.setup()
        pipeline.run()
    except Exception as e:
        log.exception(f"Fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
