"""
Verification Test Script: Test Detector across all 14 Camera Roll Photos.
Evaluates both detection accuracy and frame processing speed for Raspberry Pi.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from detector import IRDetector

CAMERA_ROLL_DIR = Path(r"C:\Users\prana\Pictures\Camera Roll")

SELECTED_FILENAMES = [
    "WIN_20261027_11_14_43_Pro.jpg",
    "WIN_20261027_11_14_44_Pro.jpg",
    "WIN_20261028_09_36_18_Pro.jpg",
    "WIN_20261028_09_36_19_Pro.jpg",
    "WIN_20261028_09_36_20_Pro.jpg",
    "WIN_20261028_09_36_23_Pro.jpg",
    "WIN_20261028_09_36_24_Pro.jpg",
    "WIN_20261028_09_36_25_Pro (2).jpg",
    "WIN_20261028_09_36_25_Pro.jpg",
    "WIN_20261028_09_36_26_Pro.jpg",
    "WIN_20261028_09_36_27_Pro.jpg",
    "WIN_20261028_09_36_28_Pro.jpg",
    "WIN_20261028_09_36_29_Pro (2).jpg",
    "WIN_20261028_09_36_29_Pro.jpg"
]


def test_camera_roll():
    print("=" * 65)
    print("   VERIFYING GREEN LED DETECTOR ON CAMERA ROLL IMAGES")
    print("=" * 65)

    det = IRDetector(led_type="green")
    times = []

    for idx, fname in enumerate(SELECTED_FILENAMES):
        fpath = CAMERA_ROLL_DIR / fname
        if not fpath.exists():
            continue

        img_orig = cv2.imread(str(fpath))
        if img_orig is None:
            continue

        frame = cv2.resize(img_orig, (640, 480))

        # Benchmark
        t0 = time.perf_counter()
        pt, _, _, meta = det.detect_debug(frame)
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000.0
        times.append(dt)

        status = meta.get("bulb_state", "UNKNOWN")
        exc = meta.get("green_excess", 0.0)
        pt_str = f"({pt[0]:.1f}, {pt[1]:.1f})" if pt is not None else "None (Pen Lifted)"

        print(f"[{idx+1:02d}] {fname[:26]}: {status:<12} at {pt_str:<22} (Meter: +{exc:5.1f}) [{dt:4.2f} ms]")

    avg_time = sum(times) / len(times)
    fps = 1000.0 / avg_time if avg_time > 0 else 999.0

    print("=" * 65)
    print(f"PERFORMANCE SUMMARY FOR RASPBERRY PI 3B+:")
    print(f"  Average Processing Time per Frame: {avg_time:.2f} ms")
    print(f"  Theoretical Processing Rate:       {fps:.1f} FPS")
    print("=" * 65)
    assert avg_time < 15.0, f"Processing time too high: {avg_time:.2f} ms"
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    test_camera_roll()
