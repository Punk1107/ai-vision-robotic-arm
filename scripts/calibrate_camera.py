#!/usr/bin/env python3
"""
scripts/calibrate_camera.py — OpenCV Chessboard Camera Calibration
===================================================================
Usage:
    python scripts/calibrate_camera.py --board 9x6 --size 0.025

Steps:
  1. Print a chessboard pattern (9×6 inner corners, 25mm squares).
  2. Hold it in front of the camera in 20+ different orientations.
  3. Press SPACE to capture each frame, 'q' when done.
  4. camera_params.json will be saved to data/calibration/.
"""

import json
from pathlib import Path

import click
import cv2
import numpy as np

CALIB_DIR = Path(__file__).resolve().parent.parent / "data" / "calibration"


@click.command()
@click.option("--board",  default="9x6",   help="Inner corners WxH (e.g. 9x6).")
@click.option("--size",   default=0.025,   type=float,
              help="Physical square size in metres (e.g. 0.025 = 25mm).")
@click.option("--camera", default=0,       type=int, help="Camera device index.")
@click.option("--frames", default=20,      type=int, help="Minimum capture frames.")
def calibrate(board: str, size: float, camera: int, frames: int) -> None:
    cols, rows = map(int, board.lower().split("x"))
    board_shape = (cols, rows)

    # Object points for one chessboard view
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * size

    objpoints, imgpoints = [], []

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {camera}")

    print(f"\n{'='*50}")
    print(f"  Chessboard Calibration ({cols}×{rows}, {size*1000:.0f}mm squares)")
    print(f"  SPACE = capture   |   q = done & solve")
    print(f"{'='*50}\n")

    captured = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_shape, None)

        vis = frame.copy()
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, board_shape, corners2, found)
            cv2.putText(vis, "FOUND — SPACE to capture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No pattern detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(vis, f"Captured: {captured}/{frames}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow("Camera Calibration", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            objpoints.append(objp)
            imgpoints.append(corners2 if found else corners)
            captured += 1
            print(f"  Captured frame {captured}/{frames}")
        elif key == ord("q") or (captured >= frames and key != 255):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured < 6:
        raise SystemExit(f"Not enough frames ({captured} < 6). Calibration aborted.")

    print(f"\nSolving with {captured} frames ...")
    h, w = gray.shape

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    print(f"  RMS reprojection error: {rms:.4f} px")
    print(f"  fx={camera_matrix[0,0]:.2f}  fy={camera_matrix[1,1]:.2f}")
    print(f"  cx={camera_matrix[0,2]:.2f}  cy={camera_matrix[1,2]:.2f}")

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CALIB_DIR / "camera_params.json"
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs":   dist_coeffs.tolist(),
        "image_size":    [w, h],
        "rms_error":     rms,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n  ✓ Saved to {out_path}")


if __name__ == "__main__":
    calibrate()
