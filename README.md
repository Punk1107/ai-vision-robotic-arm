# 🤖 AI Vision + Robotic Arm — Autonomous Pick-and-Place System

> **Senior Capstone Project** · Computer Science + Robotics  
> *Real-time object detection → coordinate mapping → inverse kinematics → arm control*

---

## 📌 Abstract

This project implements a **fully autonomous pick-and-place robotic arm** driven by
computer vision and AI decision-making.  A camera feed is processed in real-time by
a YOLOv8 object detector; detected objects are localised in 3-D space via camera
calibration and (optional) depth estimation, classified by a rule-based decision engine,
and finally picked up and sorted into bins by a 4-DOF robot arm — controlled through
analytically solved inverse kinematics.

The system is designed to demonstrate the **full CS + Robotics stack**:

| Layer | Technology |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Coordinate mapping | OpenCV camera calibration, monocular depth (MiDaS) |
| Decision logic | State-machine + class-based bin routing |
| Arm kinematics | Analytical 4-DOF IK + SciPy numerical fallback |
| Hardware control | Arduino serial (JSON protocol, 115 200 baud) |
| Testing | pytest + pytest-cov |

---

## 🗂 Project Structure

```
ai_robotic_arm/
├── config.yaml              ← Master configuration (edit without touching code)
├── requirements.txt
│
├── data/
│   ├── images/              ← Training / validation images
│   ├── labels/              ← YOLO annotation (.txt)
│   └── calibration/         ← camera_params.json, hand_eye.json (auto-generated)
│
├── models/
│   └── yolo/
│       └── best.pt          ← Custom-trained model (place here after training)
│
├── src/
│   ├── vision/
│   │   ├── detect.py        ← YOLOv8 wrapper, Detection / DetectionResult
│   │   ├── depth.py         ← MiDaS + CoordinateMapper (pixel → XYZ)
│   │   └── preprocess.py    ← Undistortion, resize, CLAHE
│   │
│   ├── robotics/
│   │   ├── kinematics.py    ← Analytical + numerical IK solver
│   │   ├── control.py       ← Serial controller (JSON protocol)
│   │   └── calibration.py   ← Hand-eye calibration, camera intrinsics
│   │
│   ├── logic/
│   │   └── decision.py      ← Task state machine, SORT_MAP, DROP_ZONES
│   │
│   ├── utils/
│   │   ├── config.py        ← Dataclass config + YAML + env-var loading
│   │   └── logger.py        ← Loguru + Rich structured logging
│   │
│   └── main.py              ← CLI entry point (Click), pipeline orchestrator
│
├── scripts/
│   └── calibrate_camera.py  ← Interactive chessboard calibration
│
├── tests/
│   ├── test_kinematics.py   ← IK unit tests (FK round-trip, clamp, reach)
│   └── test_decision.py     ← Decision engine unit tests
│
└── notebooks/               ← Jupyter for EDA / model evaluation
```

---

## ⚙️ Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` → `.env` and adjust:

```dotenv
ROBOT_SERIAL_PORT=COM3     # Windows: COMx  |  Linux: /dev/ttyUSB0
YOLO_CONFIDENCE=0.45
DRY_RUN=false              # true = skip serial (safe for first run)
```

Or edit `config.yaml` directly for persistent settings.

---

## 🚀 Running the Pipeline

```bash
# Dry run (no hardware needed — great for first test)
python src/main.py --dry --mode sort

# Full pipeline with hardware
python src/main.py --mode sort --port COM3

# Pick mode (single drop zone)
python src/main.py --mode pick
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `sort` | `sort` = classify + bin, `pick` = single drop |
| `--dry` | off | Skip all serial commands |
| `--port` | config | Override serial port |
| `--conf` | 0.45 | YOLO confidence threshold |
| `--camera` | 0 | Camera device index |
| `--debug / --no-debug` | on | Show annotated video window |

---

## 🔄 System Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  USB Camera │───▶│  Preprocess  │───▶│  YOLO Detector  │
│  (OpenCV)   │    │  undistort   │    │  (YOLOv8)       │
└─────────────┘    │  resize      │    └────────┬────────┘
                   └──────────────┘             │ DetectionResult
                                                ▼
                                    ┌──────────────────────┐
                                    │  Coordinate Mapper   │
                                    │  pixel → real XYZ    │
                                    │  (calibration + IK)  │
                                    └──────────┬───────────┘
                                               │ world_xyz
                                               ▼
                                    ┌──────────────────────┐
                                    │   Decision Engine    │
                                    │   class → bin label  │
                                    │   → RobotTask        │
                                    └──────────┬───────────┘
                                               │ target_xyz, drop_xyz
                                               ▼
                           ┌───────────────────────────────────┐
                           │         IK Solver                 │
                           │  Analytical (closed-form, fast)   │
                           │  ↓ fallback: SciPy L-BFGS-B       │
                           └──────────────┬────────────────────┘
                                          │ JointAngles (deg)
                                          ▼
                           ┌─────────────────────────────┐
                           │     Robot Controller        │
                           │  JSON over serial (115200)  │
                           │  → Arduino servo firmware   │
                           └─────────────────────────────┘
```

---

## 🧠 AI Components

### Object Detection (YOLOv8)

- Uses **Ultralytics YOLOv8** with a fine-tunable backbone
- Falls back to pre-trained `yolov8n.pt` if no custom model exists
- Target classes configurable in `config.yaml`

### Depth Estimation (Optional)

Enable with `depth.enabled: true` in `config.yaml`:

| Method | Hardware | Accuracy |
|---|---|---|
| `monocular` | Any webcam | ~15–30 cm (relative) |
| `stereo` | Two cams | ~2–5 mm |
| `realsense` | Intel D435i | ~1–3 mm |

Monocular uses **Intel MiDaS** (`torch.hub`) with a calibrated
disparity-to-depth mapping (`alpha / disparity + beta`).

### Inverse Kinematics

- **Analytical (closed-form)** for maximum speed — sub-millisecond
- **Numerical fallback** (SciPy L-BFGS-B) when target approaches workspace boundary
- All solutions clamped to hardware joint limits before transmission

---

## 🎯 Bin Sorting Logic

| Object | Category |
|---|---|
| bottle, cup, can | ♻️ Recycle |
| phone, mouse, remote, keyboard | ⚠️ Hazardous |
| book, scissors | 🗑 General |
| apple, orange, banana | 🍎 Organic |

Bin positions are configured as `DROP_ZONES` in `src/logic/decision.py`.

---

## 📷 Camera Calibration

Run this **once** before the first deployment:

```bash
python scripts/calibrate_camera.py --board 9x6 --size 0.025
```

Hold a **9×6 chessboard** (25mm squares) in 20+ orientations, press **SPACE** each time.
Results saved to `data/calibration/camera_params.json`.

Reprojection error < 0.5 px is excellent.

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Current coverage target: **≥ 80%** on `kinematics.py` and `decision.py`.

---

## 🛠 Hardware

| Component | Recommendation |
|---|---|
| Robot arm | MeArm Pi / Yahboom 4-DOF Arm Kit |
| Microcontroller | Arduino Uno / Nano |
| Camera | Logitech C920 (1080p) or better |
| Optional depth | Intel RealSense D435i |

### Arduino Firmware Protocol

The firmware (`scripts/arduino/arm_firmware.ino`) expects newline-delimited JSON:

```json
{"cmd": "move", "angles": [30.0, 60.0, 45.0, 10.0, 0.0], "speed": 50}
{"cmd": "grip", "value": 90}
{"cmd": "home"}
{"cmd": "estop"}
```

Response:
```json
{"status": "ok"}
```

---

## 📈 Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Done | Project scaffold, IK solver, decision engine |
| 2 | 🔄 Next | Camera calibration + serial integration test |
| 3 | 📋 Planned | Custom YOLO training on target objects |
| 4 | 📋 Planned | Hand-eye calibration (pixel → robot frame) |
| 5 | 💡 Stretch | Reinforcement learning gripper control |
| 6 | 💡 Stretch | ROS 2 integration |

---

## 📚 References

1. Redmon, J. et al. — *You Only Look Once* (YOLO family)
2. Ranftl, R. et al. — *Towards Robust Monocular Depth Estimation* (MiDaS)
3. Craig, J. — *Introduction to Robotics: Mechanics and Control*
4. OpenCV documentation — Camera Calibration
5. Tsai, R. — *A Versatile Camera Calibration Technique*

---

## 👨‍💻 Author

**[Your Name]**  
B.Sc. Computer Science — Capstone Project  
*Advisor: [Advisor Name] · Department of Computer Science*

---

*Built with Python 3.11 · YOLOv8 · OpenCV · PyTorch · SciPy · Arduino*
