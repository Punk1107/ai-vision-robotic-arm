#!/usr/bin/env python3
"""
test_vision.py — Minimum Runnable Vision Test
=============================================
This script satisfies STEP 1: Open camera -> Detect Object -> Print Position.
It does NOT connect to the robotic arm. It only tests the AI Vision core.

Usage:
  python scripts/test_vision.py
"""

import sys
import os
import cv2
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import config
from src.vision.detect import ObjectDetector
from src.vision.preprocess import CameraPreprocessor

def main():
    print("=" * 50)
    print(" 🚀 STEP 1: Vision Minimum Runnable Test")
    print("=" * 50)
    
    # Initialize Camera
    cam_id = config.camera.device_id
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}.")
        return

    print(f"[INFO] Camera {cam_id} opened successfully.")
    
    # Initialize Preprocessor & YOLO Detector
    preprocessor = CameraPreprocessor()
    detector = ObjectDetector(frame_skip=1)
    
    print("[INFO] YOLOv8 Model loaded. Starting detection loop...")
    print("[INFO] Press 'q' in the video window or Ctrl+C to exit.\n")
    
    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break
                
            # Preprocess & Detect
            frame = preprocessor.process(raw_frame)
            result = detector.detect(frame)
            
            # Print positions of detected objects
            for det in result.detections:
                cx, cy = det.center_px
                print(f"🎯 Detected: {det.class_name.upper():<10} | Confidence: {det.confidence:.2f} | Position (px): X={cx}, Y={cy}")
            
            # Show video feed
            annotated_frame = detector.annotate_frame(frame, result)
            cv2.imshow("Vision Test (Press 'q' to quit)", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.03) # Prevent console spam
            
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")

if __name__ == "__main__":
    main()
