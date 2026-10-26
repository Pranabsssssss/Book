import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from camera import Camera, normalize_backend
from config import CAMERA_BACKEND, CAMERA_DEVICE_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

DATASET_DIR = BASE_DIR / "dataset"
CAPTURED_BACKUP_DIR = BASE_DIR / "dataset_captured"

CATEGORIES = {
    1: {
        "key": "ir_led_emitting",
        "name": "IR LED EMITTING (Bulb On / Purple Glow)",
        "dir": DATASET_DIR / "ir_led_emitting",
        "backup_dir": CAPTURED_BACKUP_DIR / "ir_led_emitting",
        "color": (255, 0, 255),  # Magenta
        "rec_count": 450,
        "instructions": "Hold marker with push-button PRESSED. Move around at different distances & angles."
    },
    2: {
        "key": "ir_bulb_off",
        "name": "UNLIT 5mm LED BULB (Bulb Off / Stylus Idle)",
        "dir": DATASET_DIR / "ir_bulb_off",
        "backup_dir": CAPTURED_BACKUP_DIR / "ir_bulb_off",
        "color": (0, 240, 255),  # Cyan
        "rec_count": 300,
        "instructions": "Hold marker UNLIT (button released). Point stylus tip at camera in various directions."
    },
    3: {
        "key": "other_lights",
        "name": "BACKGROUND / WHITE LIGHT (Screen, Lamps, Hands)",
        "dir": DATASET_DIR / "other_lights",
        "backup_dir": CAPTURED_BACKUP_DIR / "other_lights",
        "color": (0, 200, 100),  # Green
        "rec_count": 250,
        "instructions": "Aim camera at whiteboard screen, monitor white drawings, room lamps, daylight, hands."
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Capture 1000 real-world camera images for IR LED model training")
    parser.add_argument("-t", "--target", type=int, default=1000, help="Target total images to capture (default: 1000)")
    parser.add_argument("-c", "--camera", default="auto", choices=["auto", "ribbon", "usb", "csi", "opencv", "webcam"],
                        help="Camera type: 'ribbon', 'usb', or 'auto'")
    parser.add_argument("-d", "--device", type=int, default=0, help="Webcam device index (default: 0)")
    return parser.parse_args()


def measure_purple_excess(frame_bgr: np.ndarray) -> tuple:
    """Check live purple excess in the camera frame for real-time feedback."""
    b = frame_bgr[:, :, 0].astype(np.float32)
    g = frame_bgr[:, :, 1].astype(np.float32)
    r = frame_bgr[:, :, 2].astype(np.float32)
    bg = b - g
    rg = r - g
    purple_active = (bg > 1.5) & ((bg + np.maximum(0.0, rg)) >= 3.0) & (b > 60)
    if np.sum(purple_active) > 3:
        max_excess = float(np.max(bg[purple_active] + np.maximum(0.0, rg[purple_active])))
        return True, max_excess
    return False, 0.0


def draw_hud(frame_bgr: np.ndarray, cat_idx: int, counts: dict, total_captured: int, target: int,
             is_recording: bool, auto_mode: bool, purple_detected: bool, purple_excess: float) -> np.ndarray:
    """Render comprehensive interactive recording HUD onto the frame."""
    hud = frame_bgr.copy()
    h, w = hud.shape[:2]

    cat = CATEGORIES[cat_idx]
    cat_color = cat["color"]

    # --- Top Banner: Category and Recording Status ---
    banner_h = 75
    overlay = hud.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (16, 20, 26), -1)
    cv2.addWeighted(overlay, 0.85, hud, 0.15, 0, hud)

    status_str = "● RECORDING (BURST)" if is_recording else ("● AUTO-RECORDING" if auto_mode else "PAUSED (HOLD SPACE)")
    status_col = (0, 60, 255) if is_recording or auto_mode else (180, 180, 180)
    cv2.putText(hud, status_str, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)

    cat_title = f"CATEGORY [{cat_idx}]: {cat['name']}"
    cv2.putText(hud, cat_title, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, cat_color, 2)

    # Purple IR live status in top-right
    if purple_detected:
        p_badge = f"PURPLE IR: DETECTED (+{purple_excess:.0f})"
        cv2.putText(hud, p_badge, (w - 290, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)
    else:
        p_badge = "PURPLE IR: NONE (OFF)"
        cv2.putText(hud, p_badge, (w - 290, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

    pct = min(100.0, (total_captured / max(1, target)) * 100.0)
    cnt_txt = f"{total_captured} / {target} ({pct:.1f}%)"
    cv2.putText(hud, cnt_txt, (w - 290, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    # Progress bar line
    bar_y = banner_h - 4
    cv2.line(hud, (0, bar_y), (w, bar_y), (40, 45, 55), 4)
    bar_fill_w = int(w * (total_captured / max(1, target)))
    if bar_fill_w > 0:
        cv2.line(hud, (0, bar_y), (bar_fill_w, bar_y), (0, 240, 255), 4)

    # --- Bottom Banner: Category Counts & Instructions ---
    bot_h = 80
    bot_y = h - bot_h
    overlay_bot = hud.copy()
    cv2.rectangle(overlay_bot, (0, bot_y), (w, h), (16, 20, 26), -1)
    cv2.addWeighted(overlay_bot, 0.88, hud, 0.12, 0, hud)

    # Instructions
    cv2.putText(hud, f"Goal: {cat['instructions']}", (16, bot_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 245, 255), 1)

    # Category counts
    c1 = counts.get(1, 0)
    c2 = counts.get(2, 0)
    c3 = counts.get(3, 0)
    counts_str = f"[1] Emitting: {c1} | [2] Bulb Off: {c2} | [3] Background: {c3}"
    cv2.putText(hud, counts_str, (16, bot_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 240), 1)

    controls_str = "Controls: [SPACE] Hold to Burst | [C] Single Frame | [A] Auto-Burst | [1/2/3] Switch Category | [Q/ESC] Finish"
    cv2.putText(hud, controls_str, (16, bot_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 160, 180), 1)

    # Flashing border when actively saving frames
    if is_recording or auto_mode:
        cv2.rectangle(hud, (0, 0), (w - 1, h - 1), cat_color, 4)

    return hud


def main():
    args = parse_args()
    print("=" * 65)
    print("      SMART NOTEBOOK — 1000-IMAGE REAL CAMERA DATASET CAPTURER")
    print("=" * 65)
    print(f"Target Total Images : {args.target}")
    print(f"Camera Backend      : {args.camera} (device index: {args.device})")
    print("Output Folders      : dataset/ (and backup in dataset_captured/)")
    print("-" * 65)
    print("Controls:")
    print("  [1]     Switch to IR LED EMITTING (marker button pressed / purple light)")
    print("  [2]     Switch to UNLIT 5mm LED BULB (stylus idle in hand / unlit)")
    print("  [3]     Switch to BACKGROUND / WHITE LIGHT (screen white drawing, lamps)")
    print("  [SPACE] HOLD to capture continuous rapid burst (10-15 fps)")
    print("  [C]     Capture single photo")
    print("  [A]     Toggle AUTO-RECORD mode (captures automatically)")
    print("  [Q/ESC] Finish capture & train model")
    print("=" * 65)

    # Ensure output directories exist
    for cat in CATEGORIES.values():
        cat["dir"].mkdir(parents=True, exist_ok=True)
        cat["backup_dir"].mkdir(parents=True, exist_ok=True)

    # Count existing images in each category
    counts = {
        1: len(list(CATEGORIES[1]["dir"].glob("*.png"))) + len(list(CATEGORIES[1]["dir"].glob("*.jpg"))),
        2: len(list(CATEGORIES[2]["dir"].glob("*.png"))) + len(list(CATEGORIES[2]["dir"].glob("*.jpg"))),
        3: len(list(CATEGORIES[3]["dir"].glob("*.png"))) + len(list(CATEGORIES[3]["dir"].glob("*.jpg")))
    }

    initial_total = sum(counts.values())
    print(f"\n[Dataset] Existing images: Emitting: {counts[1]}, Bulb Off: {counts[2]}, Background: {counts[3]}")
    print(f"[Dataset] Starting from {initial_total} images. Target: {args.target}")

    # Start Camera
    camera_backend = normalize_backend(args.camera)
    try:
        camera = Camera(backend=camera_backend, device_index=args.device, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        camera.start()
    except Exception as e:
        print(f"\n[Error] Could not start camera: {e}")
        print("Tip: If USB webcam index is not 0, try running with: py tools/capture_dataset.py --device 1")
        sys.exit(1)

    active_cat_idx = 1
    auto_mode = False
    session_captured = 0
    last_capture_time = 0.0
    capture_interval = 0.08  # ~12 fps for burst capture

    window_name = "Smart Notebook - 1000 Frame Dataset Capture (Press SPACE to record)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Capture frame from camera (RGB)
            frame_rgb = camera.capture_frame()
            if frame_rgb is None:
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            now = time.time()

            # Live purple IR detection check
            purple_detected, purple_excess = measure_purple_excess(frame_bgr)

            # Check keypresses
            key = cv2.waitKey(1) & 0xFF
            is_space_pressed = (key == 32)

            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                print("\n[Capture] Finished requested by user.")
                break

            elif key == ord('1'):
                active_cat_idx = 1
                print(f"[Mode] Switched to Category 1: {CATEGORIES[1]['name']}")
            elif key == ord('2'):
                active_cat_idx = 2
                print(f"[Mode] Switched to Category 2: {CATEGORIES[2]['name']}")
            elif key == ord('3'):
                active_cat_idx = 3
                print(f"[Mode] Switched to Category 3: {CATEGORIES[3]['name']}")

            elif key in (ord('a'), ord('A')):
                auto_mode = not auto_mode
                print(f"[Auto-Record] {'ENABLED' if auto_mode else 'PAUSED'}")

            # Capture trigger: Single press [C], Spacebar held, or Auto Mode
            should_save = False
            if key in (ord('c'), ord('C')):
                should_save = True
            elif is_space_pressed or auto_mode:
                if now - last_capture_time >= capture_interval:
                    should_save = True

            total_now = sum(counts.values())

            # Save frame if triggered
            if should_save:
                last_capture_time = now
                cat = CATEGORIES[active_cat_idx]

                ts = int(time.time() * 1000)
                img_name = f"user_cam_{cat['key']}_{counts[active_cat_idx] + 1:04d}_{ts}.png"

                save_path_1 = cat["dir"] / img_name
                save_path_2 = cat["backup_dir"] / img_name

                # Save uncompressed/lossless PNG
                cv2.imwrite(str(save_path_1), frame_bgr)
                cv2.imwrite(str(save_path_2), frame_bgr)

                counts[active_cat_idx] += 1
                session_captured += 1
                total_now = sum(counts.values())

                if session_captured % 10 == 0 or total_now >= args.target:
                    print(f"  [Captured: {total_now}/{args.target}] Category: {cat['key']} -> {img_name}")

                if total_now >= args.target:
                    print(f"\n[Goal Reached!] Successfully captured {total_now} total images!")
                    auto_mode = False

            # Render HUD overlay
            hud_frame = draw_hud(
                frame_bgr, active_cat_idx, counts, total_now, args.target,
                is_space_pressed, auto_mode, purple_detected, purple_excess
            )

            cv2.imshow(window_name, hud_frame)

    except KeyboardInterrupt:
        print("\n[Capture] Aborted by user.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    total_final = sum(counts.values())
    print("\n" + "=" * 65)
    print("DATASET CAPTURE COMPLETE!")
    print(f"  - Total Captured This Session : {session_captured} images")
    print(f"  - Total Dataset Size          : {total_final} images")
    print("  - Category Breakdown:")
    print(f"      [1] IR LED Emitting       : {counts[1]} images")
    print(f"      [2] Unlit 5mm Stylus Bulb : {counts[2]} images")
    print(f"      [3] Other Lights / Screen : {counts[3]} images")
    print(f"  - Saved to folders            : dataset/ and dataset_captured/")
    print("=" * 65)

    if session_captured > 0:
        # Prompt to immediately train the model on this exact camera data
        print("\nWould you like to train the IR LED model on these images now?")
        try:
            choice = input("Train model now? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "y"

        if choice in ("", "y", "yes"):
            print("\n[Model] Starting model training on your newly captured camera images...")
            from tools.train_ir_detector import build_and_train
            build_and_train()
            print("\n[Success] Your model is now 100% hardcoded and trained on your exact camera data!")
        else:
            print("\nYou can train anytime by running: py tools/train_ir_detector.py")


if __name__ == "__main__":
    main()
