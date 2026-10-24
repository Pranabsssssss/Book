#!/usr/bin/env python3
"""
Smart Notebook - Main Application Entry Point
Hardware: Raspberry Pi 3B+ (or Windows/Linux PC) with OV5647 IR Camera / Webcam & HDMI Monitor
Supports Production (clean whiteboard) and Testing/Development (live OpenCV camera & HUD) modes.
"""

import sys
import argparse
import traceback

from config import IR_THRESHOLD, CAMERA_BACKEND, CAMERA_DEVICE_INDEX, DEFAULT_MODE
from camera import Camera, normalize_backend
from detector import IRDetector
from transformer import CoordinateTransformer
from calibration import calibration_exists, load_calibration, run_calibration
from whiteboard import Whiteboard


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Notebook IR Whiteboard")
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
        help=f"Camera device index for USB/OpenCV webcam (default: {CAMERA_DEVICE_INDEX})"
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default=None,
        choices=["production", "testing", "prod", "test", "dev", "development"],
        help="Operating mode: 'production' (clean whiteboard) or 'testing' (live OpenCV camera & HUD)"
    )
    parser.add_argument(
        "-r", "--recalibrate",
        action="store_true",
        help="Force recalibration regardless of saved calibration data"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=IR_THRESHOLD,
        help=f"IR brightness threshold (0-255, default: {IR_THRESHOLD})"
    )
    return parser.parse_args()


def select_camera_backend(cli_camera_arg: str) -> str:
    """Determine camera backend from CLI argument or interactive prompt."""
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


def select_operating_mode(cli_mode_arg: str) -> str:
    """Prompt user for Production vs Testing / Development mode if not specified on CLI."""
    if cli_mode_arg is not None:
        m = cli_mode_arg.strip().lower()
        if m in ("testing", "test", "dev", "development", "2"):
            return "testing"
        return "production"

    print("\n" + "=" * 60)
    print("Select Operating Mode:")
    print("  [1] Production Mode (Clean fullscreen whiteboard canvas)")
    print("  [2] Testing / Development Mode (Live camera, OpenCV threshold & tracking HUD)")
    print("=" * 60)

    try:
        choice = input(f"Enter mode [1-2] or press [Enter] for default ({DEFAULT_MODE}): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice in ("2", "test", "testing", "dev", "development"):
        return "testing"
    else:
        return "production"


def main():
    args = parse_args()
    print("=" * 60)
    print("             SMART NOTEBOOK — IR WHITEBOARD")
    print("=" * 60)

    # 1. Resolve Camera Backend & Initialize
    chosen_backend = select_camera_backend(args.camera)
    print(f"[Main] Initializing camera using backend: '{chosen_backend}' (device index: {args.device})...")

    try:
        camera = Camera(backend=chosen_backend, device_index=args.device)
        camera.start()
    except Exception as e:
        print(f"\n[Main] CRITICAL ERROR: Could not start camera.\n{e}")
        print("Tip: If using ribbon cable on Raspberry Pi, verify camera is enabled.")
        print("     If using USB webcam or on Windows/PC, run with: python main.py --camera usb")
        sys.exit(1)

    homography_matrix = None

    # 2. Calibration Check & Routine
    try:
        if args.recalibrate or not calibration_exists():
            if args.recalibrate:
                print("[Main] Recalibration requested via --recalibrate flag.")
            else:
                print("[Main] No existing calibration found on disk.")
            homography_matrix = run_calibration(camera)
        else:
            print("\nExisting calibration found.")
            try:
                user_choice = input("Press 'r' to recalibrate, or [Enter] to use existing data: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                user_choice = ""

            if user_choice == "r":
                homography_matrix = run_calibration(camera)
            else:
                try:
                    homography_matrix = load_calibration()
                    print("[Main] Successfully loaded existing calibration data.")
                except Exception as e:
                    print(f"[Main] Failed to load calibration file ({e}). Starting fresh calibration...")
                    homography_matrix = run_calibration(camera)

    except KeyboardInterrupt:
        print("\n[Main] Calibration aborted by user. Exiting.")
        camera.stop()
        sys.exit(0)
    except Exception as e:
        print(f"\n[Main] Error during calibration: {e}")
        traceback.print_exc()
        camera.stop()
        sys.exit(1)

    # 3. Select Operating Mode (Production vs Testing / Dev)
    chosen_mode = select_operating_mode(args.mode)
    is_dev = (chosen_mode == "testing")
    print(f"[Main] Active mode: {'TESTING / DEVELOPMENT' if is_dev else 'PRODUCTION'}")

    # 4. Setup Coordinate Transformer, IR Detector, and Whiteboard Canvas
    print("[Main] Initializing coordinate transformer & detector...")
    transformer = CoordinateTransformer(homography_matrix)
    display_quad = transformer.get_display_quad()
    detector = IRDetector(threshold=args.threshold)

    print("[Main] Launching Whiteboard canvas...")
    whiteboard = Whiteboard(dev_mode=is_dev)
    whiteboard.set_current_threshold(detector.threshold)

    # 5. Main Event & Tracking Loop
    print("\n[Main] Smart Notebook is active!")
    print(" - Press the IR pen button / remote control to draw on screen.")
    print(" - Release the button to lift the pen.")
    print(" - Tap or blink light on ERASER or SAVE buttons to trigger.")
    print(" - Press [TAB] or [D] to toggle between Production and Testing modes.")
    print(" - Press [ESC] on keyboard to exit.\n")

    try:
        while not whiteboard.should_quit():
            # Capture camera frame
            frame = camera.capture_frame()

            if whiteboard.dev_mode:
                # Testing / Development Mode: full OpenCV diagnostics and DocScanner quad
                cam_point, debug_frame, thresh_rgb, metadata = detector.detect_debug(
                    frame, display_quad=display_quad
                )
                screen_point = transformer.transform(cam_point) if cam_point is not None else None
                whiteboard.update_dev_overlay(
                    annotated_frame=debug_frame,
                    thresh_mask=thresh_rgb,
                    info=metadata,
                    screen_point=screen_point
                )
            else:
                # Production Mode: streamlined high-speed circular dot detection in display quad
                cam_point = detector.detect(frame, display_quad=display_quad)
                screen_point = transformer.transform(cam_point) if cam_point is not None else None

            if screen_point is not None:
                whiteboard.handle_point(screen_point)
            else:
                whiteboard.handle_no_point()

            # Dynamic threshold tuning via dev mode hotkeys '[' and ']'
            if whiteboard.threshold_delta != 0:
                detector.threshold = max(50, min(255, detector.threshold + whiteboard.threshold_delta))
                whiteboard.threshold_delta = 0
                whiteboard.set_current_threshold(detector.threshold)
                print(f"[Main] Adjusted IR threshold to: {detector.threshold}")

            # Render canvas, button overlays, dev HUD, and process window events
            whiteboard.render()

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    except Exception as e:
        print(f"\n[Main] Runtime error in tracking loop: {e}")
        traceback.print_exc()
    finally:
        print("[Main] Cleaning up resources...")
        whiteboard.cleanup()
        camera.stop()
        print("[Main] Application shutdown complete.")


if __name__ == "__main__":
    main()
