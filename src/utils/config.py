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
        "elbow":    (-150, 150),
        "wrist":    (-180, 180),
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
class SegmentationConfig:
    enabled: bool = True
    model_path: str = str(ROOT / "models" / "yolo" / "best-seg.pt")
    fallback_model: str = "yolov8n-seg.pt"
    frame_skip: int = 3
    min_mask_area_px: int = 1500


@dataclass
class GraspPoseConfig:
    min_confidence: float = 0.30
    depth_std_thresh: float = 0.08


@dataclass
class PathPlanningConfig:
    enabled: bool = True
    max_iterations: int = 2000
    step_size_m: float = 0.025
    goal_bias: float = 0.15
    rewire_radius_m: float = 0.08
    smooth_path: bool = True


@dataclass
class VisualServoConfig:
    enabled: bool = False
    pixel_tolerance: float = 8.0
    area_tolerance: float = 500.0
    max_iterations: int = 120
    kp_x: float = 0.0004
    ki_x: float = 0.00005
    kd_x: float = 0.0001
    kp_y: float = 0.0004
    ki_y: float = 0.00005
    kd_y: float = 0.0001
    kp_z: float = 0.0002
    ki_z: float = 0.00002
    kd_z: float = 0.00005
    desired_area_px: float = 15000.0


@dataclass
class QualityControlConfig:
    enabled: bool = True
    defect_threshold: float = 0.55
    ml_model_path: str = str(ROOT / "models" / "qc" / "defect_classifier.onnx")
    laplacian_low_thresh: float = 15.0
    colour_std_thresh: float = 45.0
    solidity_low_thresh: float = 0.82
    reject_zone_xyz: Tuple[float, float, float] = (0.00, 0.35, 0.05)


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    robotics: RoboticsConfig = field(default_factory=RoboticsConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    grasp_pose: GraspPoseConfig = field(default_factory=GraspPoseConfig)
    path_planning: PathPlanningConfig = field(default_factory=PathPlanningConfig)
    visual_servo: VisualServoConfig = field(default_factory=VisualServoConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)

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

        # Optional feature sections
        for section_name in (
            "segmentation",
            "grasp_pose",
            "path_planning",
            "visual_servo",
            "quality_control",
        ):
            if section := data.get(section_name):
                target = getattr(cfg, section_name)
                for k, v in section.items():
                    if hasattr(target, k):
                        setattr(target, k, v)

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
