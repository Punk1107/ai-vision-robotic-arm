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

from src.vision.preprocess   import CameraPreprocessor
from src.vision.segmentation import InstanceSegmentor
from src.vision.pose         import GraspPoseEstimator
from src.vision.depth        import CoordinateMapper, create_depth_backend

from src.robotics.kinematics   import IKSolver, JointAngles, DynamicInterceptor
from src.robotics.control      import RobotController
from src.robotics.path_planning import RRTStarPlanner
from src.robotics.visual_servo  import VisualServoController

from src.logic.decision import DecisionEngine, TaskType

log = get_logger("main")

_HOME = JointAngles(base=90, shoulder=90, elbow=0, wrist=90, gripper=0)


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
        self._cap:         Optional[cv2.VideoCapture]    = None
        self._pre:         Optional[CameraPreprocessor]  = None
        self._segmentor:   Optional[InstanceSegmentor]   = None
        self._pose_est:    Optional[GraspPoseEstimator]  = None
        self._mapper:      Optional[CoordinateMapper]    = None
        self._depth_est                                  = None
        self._ik:          Optional[IKSolver]            = None
        self._ctrl:        Optional[RobotController]     = None
        self._decision:    Optional[DecisionEngine]      = None
        self._rrt:         Optional[RRTStarPlanner]      = None
        self._servo:       Optional[VisualServoController] = None
        self._interceptor: Optional[DynamicInterceptor]  = None
        self._current_ja:  JointAngles                   = _HOME

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

        self._pre       = CameraPreprocessor(equalize_hist=False)
        self._segmentor = InstanceSegmentor(frame_skip=self._frame_skip)
        self._depth_est = create_depth_backend()
        self._mapper    = CoordinateMapper(
            strategy = "depth" if config.depth.enabled else "plane"
        )
        self._pose_est  = GraspPoseEstimator(mapper=self._mapper)

        self._ik          = IKSolver(elbow_up=True)
        self._ctrl        = RobotController()
        self._rrt         = RRTStarPlanner()
        self._servo       = VisualServoController(ik_solver=self._ik)
        self._interceptor = DynamicInterceptor(solver=self._ik)

        if not self._ctrl.connect():
            log.warning("Arm not connected — continuing in DRY RUN mode.")

        self._ctrl.home(blocking=True)
        
        self._decision = DecisionEngine(
            mode=self._mode, 
            confirm_frames=4,
            enable_qc=config.quality_control.enabled
        )

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
            # 1. Read Frame & Depth
            if hasattr(self._depth_est, "get_frames"):
                # RealSense backend
                raw, depth_map = self._depth_est.get_frames()
                frame = self._pre.process(raw)
            else:
                # Monocular / No Depth
                ret, raw = self._cap.read()
                if not ret:
                    log.error("[vision] Camera read failed")
                    break
                frame = self._pre.process(raw)
                depth_map = self._depth_est.estimate(frame) if self._depth_est else None

            self._perf.record_frame()

            # 2. Instance Segmentation
            result = self._segmentor.segment(frame)

            # 3. Pose Estimation & Coordinate Mapping
            for seg in result.segmentations:
                pose = self._pose_est.estimate(seg, depth_map)
                if pose is not None:
                    seg.world_xyz = pose.xyz
                    seg.wrist_angle_deg = pose.wrist_angle_deg
                else:
                    seg.world_xyz = self._mapper.map(*seg.center_px, depth_map)

            # 4. Decide (Tracker runs inside decision engine)
            task = self._decision.decide(result.segmentations, frame.shape[0]*frame.shape[1], frame=frame)

            # 5. Push task (non-blocking: drop if control still busy)
            if task.task_type in (TaskType.PICK, TaskType.SORT, TaskType.RECOVERING):
                try:
                    self._task_queue.put_nowait(task)
                except queue.Full:
                    pass   # control thread busy, skip frame

            # 6. Visualise
            if config.debug_video:
                vis = self._segmentor.annotate_frame(frame, result)
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

        log.info(f"[control] Executing {task.task_type.name} to {t_pick}")

        # 1. Compute Approach Waypoint
        lift_xyz = t_pick.copy()
        lift_xyz[2] += 0.08  # Hover 8cm above pick point
        
        start_xyz = ik.forward(self._current_ja)
        if start_xyz is None:
            start_xyz = np.array([0.0, 0.15, 0.15]) # Safe fallback

        # 2. Plan path with RRT*
        if config.path_planning.enabled:
            path = self._rrt.plan(start=start_xyz, goal=lift_xyz)
            if not path:
                log.warning("RRT* failed to find approach path.")
                return
            if config.path_planning.smooth_path:
                path = self._rrt.smooth_path(path)
        else:
            path = [lift_xyz]

        # 3. Execute Approach
        for wp in path:
            angles = ik.solve(wp, wrist_pitch_deg=-90)
            if angles:
                self._ctrl.move_to(angles)
                self._current_ja = angles
                time.sleep(0.08)

        # 4. Closed-Loop Visual Servoing (Optional)
        if config.visual_servo.enabled and self._servo:
            def get_obs():
                # Fetch fresh frame
                if hasattr(self._depth_est, "get_frames"):
                    raw, _ = self._depth_est.get_frames()
                else:
                    ret, raw = self._cap.read()
                    if not ret: return None
                fr = self._pre.process(raw)
                res = self._segmentor.segment(fr)
                
                # Find matching target
                objs = [s for s in res.segmentations if s.class_name == task.class_name]
                if not objs: return None
                best = max(objs, key=lambda o: o.confidence)
                return (best.center_px[0], best.center_px[1], best.area_px)

            log.info("Visual Servo Approach Started...")
            self._servo.run(
                get_observation=get_obs,
                move_callback=lambda a: self._ctrl.move_to(a),
                current_angles=self._current_ja,
                current_xyz=lift_xyz
            )

        # 5. Final Pick Execution
        pick_angles = ik.solve(t_pick, wrist_pitch_deg=-90)
        if pick_angles:
            self._ctrl.move_to(pick_angles)
            time.sleep(0.5)
            self._ctrl.grip(close=True)
            self._current_ja = pick_angles
        
        # 6. Drop Execution
        drop_angles = ik.solve(t_drop, wrist_pitch_deg=0)
        if drop_angles:
            self._ctrl.move_to(drop_angles)
            time.sleep(0.8)
            self._ctrl.grip(close=False)
            self._current_ja = drop_angles
        
        # 7. Finalise Task
        self._decision.notify_pick_complete()
        self._ctrl.home()
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
        
        # Add AI Planner info
        queue_len = len(self._decision.plan_queue)
        plan_str = f"Plan Queue: {queue_len} pending tasks"
        
        task_str   = f"Task: {task.task_type.name}"
        if task.task_type != TaskType.IDLE:
            task_str += f"  [{task.class_name}] → [{task.bin_label}]"
        perf_str = f"Picks: {self._perf.pick_count}  IK: {self._perf.avg_ik_ms:.1f}ms"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 68), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"[AI PLANNER] {plan_str}", (8, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 150, 50), 1, cv2.LINE_AA)
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
