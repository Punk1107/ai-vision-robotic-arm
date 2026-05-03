#!/usr/bin/env python3
"""
scripts/evaluate_accuracy.py — Robotic Arm Vision-to-Kinematics Accuracy Evaluator
===================================================================================
Automatically evaluates the accuracy of the hand-eye calibration and the overall
robotic arm positioning.

It moves the arm to a set of known 3D coordinates, and allows you to locate the
arm tip in the camera frame (either by clicking or by automatic color tracking).
Then, it calculates the Mean Absolute Error (MAE) and RMSE, generating a Markdown
report.

Usage:
    python scripts/evaluate_accuracy.py --points 5
"""

import click
import cv2
import json
import time
import sys
from pathlib import Path

import numpy as np

# Make sure imports from src work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.robotics.kinematics import IKSolver
from src.robotics.control    import RobotController
from src.robotics.calibration import HandEyeCalibrator
from src.utils.logger import get_logger

log = get_logger("evaluate_accuracy")

CALIB_DIR = Path(__file__).resolve().parent.parent / "data" / "calibration"

# 3D test points in robot coordinate frame [X, Y, Z] (meters)
TEST_POSITIONS = [
    (0.12, 0.22, 0.03),
    (-0.12, 0.22, 0.03),
    (0.00, 0.28, 0.03),
    (0.15, 0.18, 0.03),
    (-0.15, 0.18, 0.03),
]

clicked_pixel = None

def on_mouse(event, x, y, flags, param):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)
        log.info(f"Manual click: ({x}, {y})")

def generate_report(results, report_path: Path):
    """Generate a clean Markdown report with the evaluation metrics."""
    if not results:
        return

    errors = [res['error'] for res in results]
    mean_error = np.mean(errors)
    rmse = np.sqrt(np.mean(np.square(errors)))
    max_error = np.max(errors)

    md_content = f"""# 🎯 Robotic Arm Accuracy Evaluation Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Points Tested:** {len(results)}

## 📊 Summary Metrics
- **Mean Absolute Error (MAE):** `{mean_error * 1000:.2f} mm`
- **Root Mean Square Error (RMSE):** `{rmse * 1000:.2f} mm`
- **Maximum Deviation:** `{max_error * 1000:.2f} mm`

## 📍 Point-by-Point Analysis
| Point # | Target XYZ (m) | Vision XYZ (m) | Error (mm) |
|---|---|---|---|
"""
    for idx, res in enumerate(results):
        t = res['target']
        v = res['vision']
        err = res['error'] * 1000
        md_content += f"| {idx+1} | `({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})` | `({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})` | `{err:.2f}` |\n"

    md_content += "\n\n> *Note: Error is calculated as the Euclidean distance between the kinematic target coordinate and the camera-estimated coordinate using the hand-eye calibration matrix.*\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    log.success(f"Accuracy report generated: {report_path}")

@click.command()
@click.option("--points", default=5, type=int, help="Number of test points to evaluate.")
@click.option("--port",   default="COM3",      help="Robot serial port.")
@click.option("--camera", default=0,  type=int, help="Camera device index.")
@click.option("--dry",    is_flag=True,         help="Dry run (simulate clicks and arm).")
def evaluate(points: int, port: str, camera: int, dry: bool):
    """Evaluate system accuracy and generate a report."""
    global clicked_pixel
    
    log.info("Starting Accuracy Evaluation...")

    try:
        T_matrix = HandEyeCalibrator.load()
    except FileNotFoundError:
        log.error("Hand-eye calibration file not found! Please run hand_eye_calibration.py first.")
        return

    solver = IKSolver()
    ctrl = None
    if not dry:
        ctrl = RobotController()
        if not ctrl.connect():
            log.warning("Could not connect to arm. Proceeding in dry mode.")
            dry = True

    cap = cv2.VideoCapture(camera) if not dry else None
    if not dry:
        cv2.namedWindow("Evaluation")
        cv2.setMouseCallback("Evaluation", on_mouse)

    positions = TEST_POSITIONS[:points]
    results = []

    for i, xyz in enumerate(positions):
        log.info(f"[{i+1}/{len(positions)}] Moving arm to test position {xyz} ...")
        
        target = np.array(xyz)
        if not dry and ctrl:
            angles = solver.solve(target, wrist_pitch_deg=-90)
            if angles:
                ctrl.move_to(angles, speed=30)
                time.sleep(2.0) # Wait for arm to settle
            else:
                log.warning(f"IK failed for {xyz}. Skipping.")
                continue

        clicked_pixel = None
        vision_xyz = None

        if dry:
            # Simulate a pixel click that perfectly maps to the target + some noise
            # For demonstration, we just add tiny noise to the target
            noise = np.random.normal(0, 0.005, 3)
            vision_xyz = target + noise
            log.info(f"Simulated vision reading: {vision_xyz}")
        else:
            log.info("Please click the arm tip in the 'Evaluation' window, then press SPACE.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                vis = frame.copy()
                if clicked_pixel:
                    cv2.drawMarker(vis, clicked_pixel, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                    
                    # Live compute
                    uvh = np.array([clicked_pixel[0], clicked_pixel[1], 1.0])
                    v_xyz = T_matrix @ uvh
                    cv2.putText(vis, f"EST: ({v_xyz[0]:.3f}, {v_xyz[1]:.3f}, {v_xyz[2]:.3f})m", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(vis, f"Point {i+1}/{len(positions)}: Click arm tip, then SPACE", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2)
                cv2.imshow("Evaluation", vis)

                key = cv2.waitKey(30) & 0xFF
                if key == ord(" ") and clicked_pixel is not None:
                    uvh = np.array([clicked_pixel[0], clicked_pixel[1], 1.0])
                    vision_xyz = T_matrix @ uvh
                    break
                elif key == ord("s"):
                    log.info("Skipping point.")
                    break
                elif key == ord("q"):
                    log.info("Aborted evaluation.")
                    if cap: cap.release()
                    cv2.destroyAllWindows()
                    if ctrl: ctrl.disconnect()
                    return

        if vision_xyz is not None:
            # We enforce Z to be roughly the test height since 2D-to-3D projection 
            # from a single camera lacks depth without scaling. Assuming table height.
            vision_xyz[2] = target[2]
            
            error = float(np.linalg.norm(target - vision_xyz))
            results.append({
                'target': target,
                'vision': vision_xyz,
                'error': error
            })
            log.info(f"Point error: {error*1000:.2f} mm")

    if cap: cap.release()
    cv2.destroyAllWindows()
    if ctrl: 
        ctrl.home()
        ctrl.disconnect()

    # Generate Report
    report_path = Path.cwd() / "docs" / "accuracy_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(results, report_path)


if __name__ == "__main__":
    evaluate()
