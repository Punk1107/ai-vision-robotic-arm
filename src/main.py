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
from enum import Enum, auto
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
from src.robotics.shaping       import build_shaper, ZVDShaper
from src.robotics.ai_estimator  import ResonanceEstimator
from src.robotics.trajectory    import (
    SCurvePlanner, JerkLimitedPlanner, CubicSplinePlanner, TrapezoidalPlanner
)

from src.logic.decision import DecisionEngine, RobotTask, TaskType

log = get_logger("main")

_HOME = JointAngles(base=90, shoulder=90, elbow=0, wrist=90, gripper=0)

class TaskState(Enum):
    IDLE       = auto()
    PLANNING   = auto()
    APPROACH   = auto()
    SERVOING   = auto()
    PICKING    = auto()
    PLACING    = auto()
    HOMING     = auto()
    ERROR      = auto()
    ABORTING   = auto()


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
        self._task_queue: queue.Queue[Optional[RobotTask]] = queue.Queue(maxsize=1)

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
        self._shaper                                     = None
        self._estimator                                  = None
        self._traj_planner                               = None
        
        # State machine members
        self._current_task: Optional[RobotTask]          = None
        self._task_lock:    threading.Lock                = threading.Lock()
        self._abort_flag:   bool                          = False
        self._task_state:  TaskState                     = TaskState.IDLE
        self._active_task: Optional[RobotTask]           = None
        self._interrupt:   bool                          = False

        # ── Thread-safety: frame cache ──────────────────────────────────────────
        # cv2.VideoCapture is NOT thread-safe. The vision thread is the ONLY
        # thread that ever calls _cap.read(). The control thread (visual
        # servoing) reads from these cached copies under _frame_lock.
        self._frame_lock:        threading.Lock                = threading.Lock()
        self._latest_frame:      Optional[np.ndarray]          = None
        self._latest_detections: list                          = []

        # ── Thread-safety: target coordinate lock ───────────────────────────
        # target_xyz in _current_task is written by the vision loop and read by
        # the control loop. Any numpy array assignment is NOT atomic — guard it.
        self._xyz_lock: threading.Lock = threading.Lock()

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

        self._pre       = CameraPreprocessor(
            equalize_hist = False,
            use_gaussian  = True,    # Gaussian denoise always on (reduces YOLO FP)
            use_clahe     = getattr(config.camera, 'use_clahe', False),
        )
        self._segmentor = InstanceSegmentor(frame_skip=self._frame_skip)
        self._depth_est = create_depth_backend()
        self._mapper    = CoordinateMapper(
            strategy = "depth" if config.depth.enabled else "plane"
        )
        self._pose_est  = GraspPoseEstimator(mapper=self._mapper)

        self._ik          = IKSolver(elbow_up=True)

        # ── Build input shaping stack from config ──────────────────────────────────
        is_cfg = config.input_shaping

        # Stage 1: select trajectory planner
        planner_name = is_cfg.planner.lower()
        if planner_name == "s_curve":
            self._traj_planner = SCurvePlanner(
                max_vel_deg_s=is_cfg.max_vel,
                max_acc_deg_s2=is_cfg.max_acc,
                max_jerk_deg_s3=is_cfg.max_jerk,
            )
        elif planner_name == "jerk_limited":
            self._traj_planner = JerkLimitedPlanner(
                max_vel_deg_s=is_cfg.max_vel,
                max_acc_deg_s2=is_cfg.max_acc,
                profile=is_cfg.profile,
            )
        elif planner_name == "cubic":
            self._traj_planner = CubicSplinePlanner()
        else:
            self._traj_planner = TrapezoidalPlanner(
                max_vel_deg_s=is_cfg.max_vel,
                max_acc_deg_s2=is_cfg.max_acc,
            )
        log.info(f"Trajectory planner: {self._traj_planner.__class__.__name__}")

        # Stage 3: build resonance estimator (if adaptive mode on)
        self._estimator = None
        if is_cfg.adaptive:
            self._estimator = ResonanceEstimator(
                sample_rate_hz=is_cfg.imu_sample_rate,
                omega_n_init=is_cfg.omega_n,
                zeta_init=is_cfg.zeta,
            )
            log.info("Stage 3 — AI Resonance Estimator initialised.")

        # Stage 2: build the shaper (optionally wrapping in AdaptiveShaper)
        self._shaper = None
        if is_cfg.shaper_kind.lower() != "none":
            adaptive_kind = (
                f"adaptive_{is_cfg.shaper_kind}" if is_cfg.adaptive else is_cfg.shaper_kind
            )
            self._shaper = build_shaper(
                kind=adaptive_kind,
                omega_n_rad_s=is_cfg.omega_n,
                zeta=is_cfg.zeta,
                estimator=self._estimator,
            )
            log.info(
                f"Stage 2 — Shaper: {self._shaper.__class__.__name__} "
                f"(ωₙ={is_cfg.omega_n:.1f} rad/s  ζ={is_cfg.zeta:.3f})"
            )

        # ── Build controller with shaping stack ────────────────────────────────────
        self._ctrl        = RobotController(shaper=self._shaper, estimator=self._estimator)
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

            # Publish enriched frame + detections (world_xyz populated) for the
            # control thread. Done AFTER pose estimation so servoing gets 3-D data.
            with self._frame_lock:
                self._latest_frame      = frame
                self._latest_detections = list(result.segmentations)

            # 4. Decide (Tracker runs inside decision engine)
            task = self._decision.decide(result.segmentations, frame.shape[0]*frame.shape[1], frame=frame)

            # 5. Push/Update task
            if task.task_type in (TaskType.PICK, TaskType.SORT, TaskType.RECOVERING, TaskType.ABORT):
                # If we get an ABORT task, trigger flag immediately
                if task.task_type == TaskType.ABORT:
                    self._abort_flag = True
                else:
                    try:
                        self._task_queue.put_nowait(task)
                    except queue.Full:
                        # Update current task XYZ atomically so the control thread
                        # never reads a half-written numpy array.
                        with self._task_lock:
                            if self._current_task and self._current_task.track_id == task.track_id:
                                with self._xyz_lock:
                                    self._current_task.target_xyz = task.target_xyz.copy()

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
            # 1. Check for controller error
            status = self._ctrl.get_status()
            if status["state"] == "ERROR":
                log.error("[control] Hardware Error detected! Attempting reset...")
                self._task_state = TaskState.ERROR
                self._ctrl.clear_error()
                time.sleep(1.0)
                continue

            # 2. Fetch new task if IDLE — block on the queue instead of busy-spinning
            if self._task_state == TaskState.IDLE:
                try:
                    task = self._task_queue.get(timeout=0.2)
                    if task is None: break
                    with self._task_lock:
                        self._current_task = task
                        self._active_task = task
                    self._task_state = TaskState.PLANNING
                    self._abort_flag = False
                except queue.Empty:
                    continue  # stay IDLE, no sleep wasted
            if self._interrupt:
                log.warning("[control] Interrupt received — aborting current state.")
                self._task_state = TaskState.ABORTING
                self._interrupt  = False

            # 3. State Machine
            try:
                if self._task_state == TaskState.PLANNING:
                    self._state_planning()
                elif self._task_state == TaskState.APPROACH:
                    self._state_approach()
                elif self._task_state == TaskState.SERVOING:
                    self._state_servoing()
                elif self._task_state == TaskState.PICKING:
                    self._state_picking()
                elif self._task_state == TaskState.PLACING:
                    self._state_placing()
                elif self._task_state == TaskState.HOMING:
                    self._state_homing()
                elif self._task_state == TaskState.ABORTING:
                    self._state_aborting()
            except Exception as e:
                log.error(f"[control] Error in state {self._task_state}: {e}")
                self._task_state = TaskState.ABORTING

            # Yield CPU briefly only during active motion states, not when IDLE
            # (IDLE blocking is already handled by queue.get(timeout=0.2) above)
            if self._task_state != TaskState.IDLE:
                time.sleep(0.01)

        log.info("[control] thread stopped")

    # ── State Handlers ────────────────────────────────────────────────────────

    def _state_planning(self) -> None:
        task = self._active_task
        ik = self._ik
        
        with self._xyz_lock:
            target_xyz_copy = task.target_xyz.copy() if task.target_xyz is not None else None

        if target_xyz_copy is None:
            log.warning("[planning] target_xyz is None — aborting.")
            self._task_state = TaskState.ABORTING
            return

        start_xyz = ik.forward(self._current_ja)
        if start_xyz is None: start_xyz = np.array([0.0, 0.15, 0.15])

        lift_xyz = target_xyz_copy.copy()
        lift_xyz[2] += 0.08

        if config.path_planning.enabled:
            path = self._rrt.plan(start=start_xyz, goal=lift_xyz)
            if not path:
                log.warning("RRT* failed. Aborting.")
                self._task_state = TaskState.ABORTING
                return
            self._approach_path = self._rrt.smooth_path(path) if config.path_planning.smooth_path else path
        else:
            self._approach_path = [lift_xyz]
            
        self._approach_idx = 0
        self._task_state = TaskState.APPROACH

    def _state_approach(self) -> None:
        if self._approach_idx >= len(self._approach_path):
            self._task_state = TaskState.SERVOING if config.visual_servo.enabled else TaskState.PICKING
            return

        wp = self._approach_path[self._approach_idx]
        angles = self._ik.solve(wp, wrist_pitch_deg=-90, current=self._current_ja)
        if angles:
            self._ctrl.move_to(angles)
            self._current_ja = angles
            self._approach_idx += 1
            time.sleep(0.1)
        else:
            log.error(f"IK failed for waypoint {self._approach_idx}. Aborting.")
            self._task_state = TaskState.ABORTING

    def _state_servoing(self) -> None:
        log.info("Visual Servo Started...")
        task = self._active_task
        
        def get_obs():
            """Read the latest detection cache — never touches the camera directly."""
            with self._frame_lock:
                detections = list(self._latest_detections)
            objs = [s for s in detections if s.class_name == task.class_name]
            if not objs:
                return None
            best = max(objs, key=lambda o: o.confidence)
            return (best.center_px[0], best.center_px[1], best.area_px)

        res = self._servo.run(
            get_observation=get_obs,
            move_callback=lambda a: self._ctrl.move_to(a),
            current_angles=self._current_ja
        )
        self._current_ja = res.final_angles or self._current_ja
        self._task_state = TaskState.PICKING

    def _state_picking(self) -> None:
        task = self._active_task
        with self._xyz_lock:
            target_xyz = task.target_xyz.copy() if task.target_xyz is not None else None
        if target_xyz is None:
            self._task_state = TaskState.ABORTING
            return
        angles = self._ik.solve(target_xyz, wrist_pitch_deg=-90, current=self._current_ja)
        if angles:
            self._ctrl.move_to(angles, blocking=True)
            self._ctrl.grip(close=True)
            time.sleep(0.5)
            self._current_ja = angles
            self._task_state = TaskState.PLACING
        else:
            self._task_state = TaskState.ABORTING

    def _state_placing(self) -> None:
        task = self._active_task
        angles = self._ik.solve(task.drop_xyz, wrist_pitch_deg=0, current=self._current_ja)
        if angles:
            self._ctrl.move_to(angles, blocking=True)
            time.sleep(0.5)
            self._ctrl.grip(close=False)
            time.sleep(0.5)
            self._current_ja = angles
            self._task_state = TaskState.HOMING
        else:
            self._task_state = TaskState.ABORTING

    def _state_homing(self) -> None:
        self._ctrl.home(blocking=True)
        self._current_ja = _HOME
        self._decision.notify_pick_complete()
        self._perf.pick_count += 1
        self._decision.reset_picked()
        self._task_state = TaskState.IDLE
        log.success(f"✓ Task Complete: {self._active_task.class_name}")

    def _state_aborting(self) -> None:
        log.warning("Aborting task and returning home.")
        self._ctrl.grip(close=False)
        self._ctrl.home(blocking=True)
        self._current_ja = _HOME
        self._task_state = TaskState.IDLE
        self._active_task = None

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
