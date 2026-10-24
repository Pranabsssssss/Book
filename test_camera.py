#!/usr/bin/env python3
"""
Diagnostic utility: test_camera.py
Verifies camera connection and captures a single test snapshot to 'test_frame.png'.
Supports selecting ribbon cable (picamera2), USB / Windows webcam (OpenCV), or auto-detect.
"""

import time
import argparse
import cv2
from config import CAMERA_BACKEND, CAMERA_DEVICE_INDEX
from camera import Camera, normalize_backend


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Notebook - Camera Diagnostic")
    parser.add_argument(
        "-c", "--camera",
        type=str,
        default=None,
        choices=["auto", "ribbon", "usb", "csi", "opencv", "webcam"],
        help="Camera type: 'ribbon' (RPi CSI ribbon cable), 'usb' (Windows/Linux webcam), or 'auto'"
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=CAMERA_DEVICE_INDEX,
        help=f"Device index for USB/OpenCV webcam (default: {CAMERA_DEVICE_INDEX})"
    )
    return parser.parse_args()


def select_camera_backend(cli_camera_arg: str) -> str:
    """Prompt user if CLI flag not provided."""
    if cli_camera_arg is not None:
        return normalize_backend(cli_camera_arg)

    print("\nCamera Source Options:")
    print("  [1] Auto-detect (RPi Ribbon Cable -> USB/Windows)")
    print("  [2] Raspberry Pi Ribbon Cable (CSI / picamera2)")
    print("  [3] USB Webcam / Windows / Regular Linux (OpenCV)")

    try:
        choice = input(f"Select camera [1-3] or press [Enter] for default ({CAMERA_BACKEND}): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice == "2":
        return "ribbon"
    elif choice == "3":
        return "usb"
    elif choice == "1":
        return "auto"
    else:
        return normalize_backend(CAMERA_BACKEND)


def main():
    args = parse_args()
    print("=" * 50)
    print("        Smart Notebook - Camera Diagnostic")
    print("=" * 50)

    chosen_backend = select_camera_backend(args.camera)
    print(f"[Diagnostic] Selected camera backend: '{chosen_backend}' (device index: {args.device})")

    try:
        cam = Camera(backend=chosen_backend, device_index=args.device)
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
