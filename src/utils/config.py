"""
config.py — Centralized Configuration Manager
==============================================
Loads settings from config.yaml and environment variables.
All modules import from here — single source of truth.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# ── Project root (two levels up from this file) ──────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "config.yaml"


@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    focal_length_mm: float = 3.67      # default for typical webcam
    sensor_width_mm: float = 3.68


@dataclass
class YOLOConfig:
    model_path: str = str(ROOT / "models" / "yolo" / "best.pt")
    fallback_model: str = "yolov8n.pt"  # auto-download from ultralytics hub
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    target_classes: list = field(default_factory=lambda: [
        "cup", "bottle", "book", "cell phone", "scissors",
        "remote", "keyboard", "mouse", "apple", "orange"
    ])
    input_size: int = 640


@dataclass
class RoboticsConfig:
    # Serial port to Arduino / servo controller
    serial_port: str = os.getenv("ROBOT_SERIAL_PORT", "COM3")   # Windows default
    baud_rate: int = 115200
    timeout: float = 1.0

    # 4-DOF arm link lengths (meters) — adjust to your kit
    l1: float = 0.105   # Base to shoulder (vertical)
    l2: float = 0.105   # Upper arm
    l3: float = 0.090   # Forearm
    l4: float = 0.060   # Wrist to gripper tip

    # Joint angle limits (degrees) [min, max]
    joint_limits: dict = field(default_factory=lambda: {
        "base":     (-90,  90),
        "shoulder": (  0, 150),
        "elbow":    (  0, 150),
        "wrist":    (-90,  90),
        "gripper":  (  0,  90),   # 0 = open, 90 = closed
    })

    # Workspace bounds (meters from base center)
    workspace_x: Tuple[float, float] = (-0.25, 0.25)
    workspace_y: Tuple[float, float] = ( 0.10, 0.40)
    workspace_z: Tuple[float, float] = ( 0.00, 0.30)

    # Speed profile
    move_speed: int = 50    # 0-100 (sent to firmware)
    home_speed: int = 30


@dataclass
class DepthConfig:
    enabled: bool = False
    method: str = "monocular"   # "monocular" | "stereo" | "realsense"
    monocular_model: str = "MiDaS_small"   # torch.hub
    # Calibration: real_z = alpha / estimated_disparity + beta
    alpha: float = 1200.0
    beta: float = -0.05


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    robotics: RoboticsConfig = field(default_factory=RoboticsConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)

    log_level: str = "INFO"
    log_dir: str = str(ROOT / "logs")
    debug_video: bool = True    # show annotated feed while running
    dry_run: bool = False       # run vision only, skip serial commands


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load config.yaml (if exists) and override with environment variables.
    Falls back to dataclass defaults if no file is found.
    """
    cfg = AppConfig()

    path = config_path or CONFIG_FILE
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Camera
        if cam := data.get("camera"):
            for k, v in cam.items():
                if hasattr(cfg.camera, k):
                    setattr(cfg.camera, k, v)

        # YOLO
        if yolo := data.get("yolo"):
            for k, v in yolo.items():
                if hasattr(cfg.yolo, k):
                    setattr(cfg.yolo, k, v)

        # Robotics
        if robot := data.get("robotics"):
            for k, v in robot.items():
                if hasattr(cfg.robotics, k):
                    setattr(cfg.robotics, k, v)

        # Depth
        if depth := data.get("depth"):
            for k, v in depth.items():
                if hasattr(cfg.depth, k):
                    setattr(cfg.depth, k, v)

        # App-level
        for k in ("log_level", "debug_video", "dry_run"):
            if k in data:
                setattr(cfg, k, data[k])

    # Environment overrides (CI-friendly)
    if port := os.getenv("ROBOT_SERIAL_PORT"):
        cfg.robotics.serial_port = port
    if conf := os.getenv("YOLO_CONFIDENCE"):
        cfg.yolo.confidence_threshold = float(conf)
    if dry := os.getenv("DRY_RUN"):
        cfg.dry_run = dry.lower() in ("1", "true", "yes")

    return cfg


# Singleton — import this everywhere
config = load_config()
