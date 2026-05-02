#!/usr/bin/env python3
"""
scripts/hand_eye_calibration.py — Interactive Hand-Eye Calibration
====================================================================
Moves the arm to N known positions, asks you to click the
corresponding pixel in the live camera feed, then solves for
the pixel → robot coordinate transform.

Usage:
    python scripts/hand_eye_calibration.py --points 8 --port COM3

Result saved to: data/calibration/hand_eye.json
"""

import click
import cv2
import json
from pathlib import Path

import numpy as np

CALIB_DIR = Path(__file__).resolve().parent.parent / "data" / "calibration"

# Pre-defined robot positions to visit (metres from base)
CALIBRATION_POSITIONS = [
    (0.15, 0.20, 0.02),
    (0.00, 0.25, 0.02),
    (-0.15, 0.20, 0.02),
    (0.15, 0.15, 0.02),
    (-0.15, 0.15, 0.02),
    (0.10, 0.30, 0.02),
    (-0.10, 0.30, 0.02),
    (0.00, 0.18, 0.02),
]

clicked_pixel = None


def on_mouse(event, x, y, flags, param):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)
        print(f"  → Clicked pixel: ({x}, {y})")


@click.command()
@click.option("--points", default=8, type=int, help="Number of calibration points.")
@click.option("--port",   default="COM3",      help="Robot serial port.")
@click.option("--camera", default=0,  type=int, help="Camera device index.")
@click.option("--dry",    is_flag=True,         help="Skip arm movement (manual positioning).")
def calibrate(points: int, port: str, camera: int, dry: bool) -> None:
    """Interactive hand-eye calibration."""
    global clicked_pixel

    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.robotics.kinematics import IKSolver
    from src.robotics.control    import RobotController
    from src.robotics.calibration import HandEyeCalibrator

    solver     = IKSolver()
    calibrator = HandEyeCalibrator()

    ctrl = None
    if not dry:
        ctrl = RobotController()
        ctrl.connect()

    cap = cv2.VideoCapture(camera)
    cv2.namedWindow("Hand-Eye Calibration")
    cv2.setMouseCallback("Hand-Eye Calibration", on_mouse)

    positions = CALIBRATION_POSITIONS[:points]

    print(f"\n{'='*55}")
    print(f"  Hand-Eye Calibration  ({points} points)")
    print(f"  Click the RED marker in the image, then press SPACE")
    print(f"{'='*55}\n")

    for i, xyz in enumerate(positions):
        print(f"\n[{i+1}/{points}] Moving arm to {xyz} ...")

        if not dry and ctrl:
            target = np.array(xyz)
            angles = solver.solve(target, wrist_pitch_deg=-90)
            if angles:
                ctrl.move_to(angles, speed=30)
            else:
                print("  WARNING: IK failed — skip or reposition manually")
        else:
            print("  [DRY] Place a marker at this position manually.")

        clicked_pixel = None
        print(f"  Click the arm tip in the window, then press SPACE ...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            vis = frame.copy()
            if clicked_pixel:
                cv2.drawMarker(vis, clicked_pixel, (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 2)

            cv2.putText(vis,
                f"Point {i+1}/{points} — Click arm tip, then SPACE",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2)
            cv2.imshow("Hand-Eye Calibration", vis)

            key = cv2.waitKey(30) & 0xFF
            if key == ord(" ") and clicked_pixel is not None:
                calibrator.add_point(pixel=clicked_pixel, robot_xyz=xyz)
                break
            elif key == ord("s"):
                print("  Skipping this point.")
                break
            elif key == ord("q"):
                print("Aborted.")
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    if ctrl:
        ctrl.home()
        ctrl.disconnect()

    print("\nSolving calibration ...")
    calibrator.save()
    print(f"✓ Saved to {CALIB_DIR / 'hand_eye.json'}")


if __name__ == "__main__":
    calibrate()
