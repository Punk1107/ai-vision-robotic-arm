"""
control.py — Robot Arm Serial Controller (v2)
=============================================
Improvements over v1:
  - Async command queue: vision pipeline never blocks waiting for serial ACK.
    A dedicated worker thread drains the queue and sends one command at a time.
  - Retry logic with exponential back-off (up to 3 attempts per command).
  - Position feedback cache: firmware can optionally echo current angles back.
  - Trajectory playback: accepts a list of TrajectoryPoints and plays them
    at the configured tick rate without blocking the caller.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import serial
import serial.tools.list_ports

from src.robotics.kinematics import JointAngles
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger("robotics.control")

MAX_RETRIES   = 3
RETRY_DELAY_S = 0.05


class ArmState(Enum):
    DISCONNECTED = auto()
    IDLE         = auto()
    MOVING       = auto()
    ERROR        = auto()


@dataclass
class _Command:
    payload: dict
    retries: int = 0


class RobotController:
    """
    Robot arm serial controller with non-blocking command queue.

    The caller pushes commands to the queue; an internal worker thread
    sends them over serial in order.  This decouples the 30fps vision
    loop from the ~0.3s serial round-trips.

    Usage::

        with RobotController() as ctrl:
            ctrl.home()
            ctrl.move_to(angles)
            ctrl.grip(close=True)
    """

    _HOME = JointAngles(base=0, shoulder=90, elbow=0, wrist=0, gripper=0)

    def __init__(self) -> None:
        self._port    = config.robotics.serial_port
        self._baud    = config.robotics.baud_rate
        self._timeout = config.robotics.timeout
        self._dry_run = config.dry_run

        self._serial: Optional[serial.Serial] = None
        self._state   = ArmState.DISCONNECTED
        self._lock    = threading.Lock()

        # Async queue
        self._q:      queue.Queue[Optional[_Command]] = queue.Queue(maxsize=16)
        self._worker: Optional[threading.Thread]      = None
        self._worker_running = False

        # Last known joint angles (from firmware echo or our own commands)
        self._current_angles: Optional[JointAngles] = None

        if self._dry_run:
            log.warning("[DRY RUN] All serial commands will be simulated.")

    # ── Connection ────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        if self._dry_run:
            self._state = ArmState.IDLE
            self._start_worker()
            log.info("[DRY RUN] Connected (simulated)")
            return True

        try:
            self._serial = serial.Serial(
                self._port, self._baud, timeout=self._timeout
            )
            time.sleep(2.0)
            self._state = ArmState.IDLE
            self._start_worker()
            log.success(f"Connected on [cyan]{self._port}[/cyan] @ {self._baud} ✓")
            return True
        except serial.SerialException:
            log.warning(f"{self._port} unavailable. Auto-scanning ...")
            return self._auto_detect()

    def _auto_detect(self) -> bool:
        candidates = [
            p.device for p in serial.tools.list_ports.comports()
            if any(kw in (p.description or "")
                   for kw in ("Arduino", "CH340", "USB Serial", "CP210"))
        ]
        for port in candidates:
            try:
                self._serial = serial.Serial(port, self._baud, timeout=self._timeout)
                time.sleep(2.0)
                self._port  = port
                self._state = ArmState.IDLE
                self._start_worker()
                log.success(f"Auto-detected on [cyan]{port}[/cyan] ✓")
                return True
            except serial.SerialException:
                continue

        log.error("No arm found. Check USB and config.yaml serial_port.")
        self._state = ArmState.ERROR
        return False

    def disconnect(self) -> None:
        self._stop_worker()
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._state = ArmState.DISCONNECTED
        log.info("Disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._state not in (ArmState.DISCONNECTED, ArmState.ERROR)

    @property
    def current_angles(self) -> Optional[JointAngles]:
        return self._current_angles

    def clear_error(self) -> None:
        """Reset the controller state to IDLE if it was in ERROR."""
        with self._lock:
            if self._state == ArmState.ERROR:
                log.info("Clearing controller error state...")
                self._state = ArmState.IDLE
            
    def get_status(self) -> dict:
        """Return current health and queue status."""
        return {
            "state": self._state.name,
            "queue_size": self._q.qsize(),
            "is_connected": self.is_connected,
            "dry_run": self._dry_run
        }

    # ── Worker thread ─────────────────────────────────────────────────────────
    def _start_worker(self) -> None:
        self._worker_running = True
        self._worker = threading.Thread(
            target=self._run_worker, name="arm-ctrl", daemon=True
        )
        self._worker.start()

    def _stop_worker(self) -> None:
        self._worker_running = False
        self._q.put(None)   # sentinel to unblock the worker
        if self._worker:
            self._worker.join(timeout=3.0)

    def _run_worker(self) -> None:
        """Worker: dequeue → send → retry on failure."""
        while self._worker_running:
            try:
                cmd = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            if cmd is None:   # sentinel
                break

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                success = self._send_raw(cmd.payload)
                if success:
                    break
                delay = RETRY_DELAY_S * (2 ** (attempt - 1))
                log.warning(f"Retry {attempt}/{MAX_RETRIES} in {delay:.2f}s ...")
                time.sleep(delay)

            if not success:
                log.error(f"Command failed after {MAX_RETRIES} retries: {cmd.payload}")
                with self._lock:
                    self._state = ArmState.ERROR
            
            # Heartbeat check: if queue is empty, occasionally ping firmware
            if self._q.empty() and not self._dry_run:
                self._do_heartbeat()

            self._q.task_done()

    def _do_heartbeat(self) -> None:
        """Send a no-op ping to check connection health."""
        now = time.monotonic()
        if not hasattr(self, "_last_hb_t"): self._last_hb_t = now
        
        if (now - self._last_hb_t) > 5.0:  # every 5 seconds
            try:
                self._send_raw({"cmd": "ping"})
                self._last_hb_t = now
            except Exception:
                log.error("Heartbeat failed — connection lost?")
                with self._lock:
                    self._state = ArmState.ERROR

    def _send_raw(self, payload: dict) -> bool:
        if self._dry_run:
            log.debug(f"[DRY RUN] → {payload}")
            time.sleep(0.01)   # simulate minimal latency
            return True

        if not self._serial or not self._serial.is_open:
            return False

        try:
            with self._lock:
                self._serial.write((json.dumps(payload) + "\n").encode())
                self._serial.flush()
                raw = self._serial.readline().decode().strip()

            if not raw:
                return False

            resp = json.loads(raw)

            # Optional: parse echoed angles from firmware
            if "angles" in resp:
                a = resp["angles"]
                self._current_angles = JointAngles(*a[:5]) if len(a) >= 5 else None

            return resp.get("status") == "ok"

        except (serial.SerialException, json.JSONDecodeError, OSError) as e:
            log.error(f"Serial error: {e}")
            return False

    # ── Public API ────────────────────────────────────────────────────────────
    def move_to(
        self,
        angles:   JointAngles,
        speed:    Optional[int] = None,
        blocking: bool = False,
    ) -> None:
        """
        Queue a joint-angle move.

        Args:
            angles:   Target joint angles.
            speed:    0-100 (firmware interprets as motor speed).
            blocking: If True, wait until queue is empty after pushing.
        """
        if not self.is_connected:
            log.error("Not connected."); return

        payload = {
            "cmd":    "move",
            "angles": [round(a, 1) for a in angles.as_list()],
            "speed":  speed or config.robotics.move_speed,
        }
        self._q.put(_Command(payload))
        self._current_angles = angles   # optimistic update

        if blocking:
            self._q.join()

    def home(self, speed: Optional[int] = None, blocking: bool = True) -> None:
        log.info("Homing ...")
        self.move_to(self._HOME, speed=speed or config.robotics.home_speed,
                     blocking=blocking)

    def grip(self, close: bool = True, value: Optional[int] = None) -> None:
        angle = value if value is not None else (90 if close else 0)
        self._q.put(_Command({"cmd": "grip", "value": angle}))
        log.info(f"GRIP {'CLOSE' if close else 'OPEN'} ({angle}°) queued")

    def emergency_stop(self) -> None:
        """Drain queue and send estop immediately (bypasses queue)."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                break
        self._send_raw({"cmd": "estop"})
        self._state = ArmState.IDLE
        log.warning("⚠ EMERGENCY STOP")

    def play_trajectory(
        self,
        trajectory: list,   # List[TrajectoryPoint]
        hz: int = 50,
    ) -> None:
        """
        Play a pre-planned trajectory by queuing each waypoint.

        Args:
            trajectory: Output of CubicSplinePlanner or TrapezoidalPlanner.
            hz:         Playback rate (Hz).  Controls sleep between frames.
        """
        dt = 1.0 / hz
        for point in trajectory:
            self.move_to(point.angles)
            time.sleep(dt)
        self._q.join()   # wait for all to be sent

    # ── Context manager ───────────────────────────────────────────────────────
    def __enter__(self) -> "RobotController":
        self.connect(); return self

    def __exit__(self, *_) -> None:
        self.home()
        self._q.join()
        self.disconnect()
