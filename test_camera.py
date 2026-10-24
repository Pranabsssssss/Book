#!/usr/bin/env python3
"""
Diagnostic utility: test_camera.py
Verifies camera connection and captures a single test snapshot to 'test_frame.png'.
Run on Raspberry Pi to confirm camera hardware and picamera2 / OpenCV functionality.
"""

import time
import cv2
from camera import Camera

def main():
    print("=" * 50)
    print("        Smart Notebook - Camera Diagnostic")
    print("=" * 50)

    try:
        cam = Camera()
        print("[Diagnostic] Starting camera...")
        cam.start()

        print("[Diagnostic] Sensor warm-up (1 second)...")
        time.sleep(1.0)

        print("[Diagnostic] Capturing frame...")
        frame = cam.capture_frame()
        print(f"[Diagnostic] Capture success! Frame dimensions: {frame.shape[1]}x{frame.shape[0]}, channels: {frame.shape[2]}")

        # Save image (converting RGB to BGR for OpenCV imwrite)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite("test_frame.png", bgr)
        print("[Diagnostic] Saved snapshot as 'test_frame.png'.")

        cam.stop()
        print("[Diagnostic] Camera test passed successfully.")
    except Exception as e:
        print(f"\n[Diagnostic] FAILED: {e}")

if __name__ == "__main__":
    main()
