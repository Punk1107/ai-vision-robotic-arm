## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         AI ROBOTIC ARM SYSTEM                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│   │ USB Camera  │───▶│ Preprocess   │───▶│   YOLOv8 Detector        │   │
│   │ (OpenCV)    │    │ • undistort  │    │   • confidence filter    │   │
│   └─────────────┘    │ • resize     │    │   • class filter         │   │
│                      │ • CLAHE      │    │   • Detection dataclass  │   │
│                      └──────────────┘    └────────────┬─────────────┘   │
│                                                       │                  │
│   ┌─────────────────────────────────────┐            │ DetectionResult   │
│   │ Depth Estimation (optional)         │            ▼                  │
│   │ MiDaS / RealSense / Stereo         │◀──▶ CoordinateMapper          │
│   │ disparity → metric depth           │     pixel (u,v) → XYZ [m]    │
│   └─────────────────────────────────────┘            │                  │
│                                                       ▼                  │
│                                            ┌──────────────────────┐     │
│                                            │  Decision Engine     │     │
│                                            │  • SORT_MAP          │     │
│                                            │  • cooldown gate     │     │
│                                            │  • RobotTask output  │     │
│                                            └──────────┬───────────┘     │
│                                                       │                  │
│                                                       ▼                  │
│                                            ┌──────────────────────┐     │
│                                            │     IK Solver        │     │
│                                            │  Analytical (fast)   │     │
│                                            │  ↓ Scipy fallback    │     │
│                                            │  JointAngles (deg)   │     │
│                                            └──────────┬───────────┘     │
│                                                       │                  │
│                                                       ▼                  │
│                                            ┌──────────────────────┐     │
│                                            │  Robot Controller    │     │
│                                            │  JSON over Serial    │     │
│                                            │  115200 baud         │     │
│                                            └──────────┬───────────┘     │
│                                                       │ USB              │
└───────────────────────────────────────────────────────┼──────────────────┘
                                                        ▼
                                           ┌────────────────────────┐
                                           │  Arduino Firmware      │
                                           │  Servo PWM control     │
                                           │  5x Servo motors       │
                                           └────────────────────────┘
```

## Data Flow Summary

| Step | Input | Output | Module |
|------|-------|--------|--------|
| 1 | Raw frame | Undistorted 640×640 | `preprocess.py` |
| 2 | BGR frame | Depth map (float32) | `depth.py` |
| 3 | BGR frame | DetectionResult | `detect.py` |
| 4 | pixel (u,v) | world XYZ [m] | `depth.py` |
| 5 | DetectionResult | RobotTask | `decision.py` |
| 6 | target XYZ | JointAngles [°] | `kinematics.py` |
| 7 | JointAngles | serial JSON | `control.py` |
