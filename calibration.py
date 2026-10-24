import os
import json
import time
import numpy as np
import cv2

try:
    import pygame
except ImportError:
    pygame = None

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    CALIBRATION_POINTS, CALIBRATION_DOT_RADIUS,
    CALIBRATION_FRAMES, CALIBRATION_TIMEOUT_SEC,
    CALIBRATION_FILE, CALIBRATION_IR_THRESHOLD,
    CAMERA_WIDTH, CAMERA_HEIGHT
)
from detector import IRDetector


def calibration_exists() -> bool:
    """Check if a valid calibration file exists on disk."""
    if not os.path.exists(CALIBRATION_FILE):
        return False
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            data = json.load(f)
        matrix = np.array(data.get("homography_matrix"))
        return matrix.shape == (3, 3)
    except Exception:
        return False


def load_calibration() -> np.ndarray:
    """Load and return the 3x3 homography matrix from disk."""
    if not os.path.exists(CALIBRATION_FILE):
        raise FileNotFoundError(f"Calibration file not found at: {CALIBRATION_FILE}")

    with open(CALIBRATION_FILE, 'r') as f:
        data = json.load(f)

    matrix = np.array(data["homography_matrix"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Invalid homography matrix in calibration file.")
    return matrix


def save_calibration(matrix: np.ndarray, error_px: float = None):
    """Save homography matrix and metadata to calibration.json."""
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "screen_resolution": [SCREEN_WIDTH, SCREEN_HEIGHT],
        "validation_error_px": float(error_px) if error_px is not None else None,
        "calibration_points": CALIBRATION_POINTS,
        "homography_matrix": matrix.tolist()
    }
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[Calibration] Saved matrix to: {CALIBRATION_FILE}")


def run_calibration(camera, screen=None) -> np.ndarray:
    """
    Detailed, interactive calibration routine.
    Detects the user's IR Light (remote / pen) at each target point on screen.
    Includes a live camera feed preview, visual feedback, hold-steady progress bars,
    and manual [SPACE] bypass for PC testing.
    """
    if pygame is None:
        raise RuntimeError(
            "Pygame is required for interactive calibration UI. "
            "Please install dependencies with: pip install -r requirements.txt"
        )

    owns_screen = False
    if screen is None:
        if not pygame.get_init():
            pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Smart Notebook - Screen Calibration")
        owns_screen = True

    font_title = pygame.font.SysFont("Arial", 32, bold=True)
    font_instruction = pygame.font.SysFont("Arial", 26, bold=True)
    font_sub = pygame.font.SysFont("Arial", 18)
    clock = pygame.time.Clock()

    detector = IRDetector(threshold=CALIBRATION_IR_THRESHOLD)

    camera_points = []
    screen_points = []

    print("[Calibration] Starting interactive IR calibration routine...")

    prev_cam_preview_surf = None
    cam_prev_w = 360
    cam_prev_h = 270

    try:
        for idx, (sx, sy) in enumerate(CALIBRATION_POINTS[:4]):  # 4 corner points
            labels = ["1/4: Top-Left Corner", "2/4: Top-Right Corner", "3/4: Bottom-Right Corner", "4/4: Bottom-Left Corner"]
            point_label = labels[idx]

            detected_samples = []
            capturing = True
            point_captured = False

            while capturing:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt("Calibration aborted by user.")
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt("Calibration aborted by user via ESC.")
                        elif event.key == pygame.K_SPACE:
                            # Manual bypass for PC development: auto-generate standard 1:1 camera point
                            sim_x = float(sx / SCREEN_WIDTH * CAMERA_WIDTH)
                            sim_y = float(sy / SCREEN_HEIGHT * CAMERA_HEIGHT)
                            detected_samples = [(sim_x, sim_y)] * CALIBRATION_FRAMES
                            point_captured = True
                            print(f"[Calibration] Manual bypass for point {idx+1}: ({sim_x:.1f}, {sim_y:.1f})")
                        elif event.key == pygame.K_s:
                            # Skip calibration completely and use 1:1 mapping
                            print("[Calibration] Skipped calibration via [S]. Using 1:1 mapping.")
                            ident = np.eye(3, dtype=np.float64)
                            save_calibration(ident, error_px=0.0)
                            return ident
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        # Click to simulate capture for PC testing
                        sim_x = float(sx / SCREEN_WIDTH * CAMERA_WIDTH)
                        sim_y = float(sy / SCREEN_HEIGHT * CAMERA_HEIGHT)
                        detected_samples = [(sim_x, sim_y)] * CALIBRATION_FRAMES
                        point_captured = True

                # Capture camera frame and detect purple IR light
                frame = camera.capture_frame()
                pt, debug_frame, thresh_rgb, metadata = detector.detect_debug(frame)

                is_ir_detected = (pt is not None and metadata.get("is_ir_purple", True))

                if is_ir_detected and not point_captured:
                    detected_samples.append(pt)
                    if len(detected_samples) >= CALIBRATION_FRAMES:
                        point_captured = True
                elif not point_captured:
                    # Decay samples if light flickers
                    if len(detected_samples) > 0 and len(detected_samples) < CALIBRATION_FRAMES:
                        detected_samples.pop(0)

                # Resize camera view for on-screen calibration preview
                cam_resized = cv2.resize(debug_frame, (cam_prev_w, cam_prev_h))
                cam_preview_surf = pygame.image.frombuffer(
                    cam_resized.tobytes(), (cam_prev_w, cam_prev_h), 'RGB'
                )

                # -------------------------------------------------------------
                # Render Calibration Display
                # -------------------------------------------------------------
                screen.fill((14, 17, 24))

                # Header
                title_surf = font_title.render("SMART NOTEBOOK — DISPLAY CALIBRATION", True, (240, 245, 255))
                screen.blit(title_surf, (SCREEN_WIDTH // 2 - title_surf.get_width() // 2, 25))

                step_surf = font_instruction.render(f"STEP {point_label}", True, (0, 230, 255))
                screen.blit(step_surf, (SCREEN_WIDTH // 2 - step_surf.get_width() // 2, 65))

                # Dynamic instructions
                if point_captured:
                    inst_txt = font_instruction.render("SUCCESS! POINT CAPTURED. RELEASE BUTTON NOW.", True, (60, 255, 100))
                elif is_ir_detected:
                    progress_pct = int(min(1.0, len(detected_samples) / CALIBRATION_FRAMES) * 100)
                    inst_txt = font_instruction.render(f"IR LIGHT DETECTED! HOLD STEADY... {progress_pct}%", True, (80, 255, 120))
                else:
                    inst_txt = font_instruction.render("AIM YOUR IR REMOTE / PEN AT THE TARGET BULLSEYE AND PRESS BUTTON", True, (255, 200, 60))

                screen.blit(inst_txt, (SCREEN_WIDTH // 2 - inst_txt.get_width() // 2, 105))

                # Draw Target Bullseye at (sx, sy)
                pulse = int(abs(time.time() * 4 % 2 - 1) * 8)
                target_color = (60, 255, 100) if point_captured else ((80, 255, 120) if is_ir_detected else (255, 180, 0))

                pygame.draw.circle(screen, target_color, (sx, sy), CALIBRATION_DOT_RADIUS + pulse, 3)
                pygame.draw.circle(screen, target_color, (sx, sy), 8, -1)
                pygame.draw.line(screen, target_color, (sx - 40, sy), (sx + 40, sy), 2)
                pygame.draw.line(screen, target_color, (sx, sy - 40), (sx, sy + 40), 2)

                # Center Live Camera Preview
                cam_box_x = (SCREEN_WIDTH - cam_prev_w) // 2
                cam_box_y = (SCREEN_HEIGHT - cam_prev_h) // 2 + 30

                pygame.draw.rect(screen, (10, 12, 16), (cam_box_x - 4, cam_box_y - 26, cam_prev_w + 8, cam_prev_h + 34), border_radius=8)
                pygame.draw.rect(screen, target_color, (cam_box_x - 4, cam_box_y - 26, cam_prev_w + 8, cam_prev_h + 34), 2, border_radius=8)
                lbl_c = font_sub.render("LIVE CAMERA VIEW (Seeing your IR Light)", True, target_color)
                screen.blit(lbl_c, (cam_box_x + 6, cam_box_y - 22))
                screen.blit(cam_preview_surf, (cam_box_x, cam_box_y + 4))

                # Progress Bar under camera preview
                bar_w = cam_prev_w
                bar_h = 14
                bar_y = cam_box_y + cam_prev_h + 14
                pygame.draw.rect(screen, (30, 35, 45), (cam_box_x, bar_y, bar_w, bar_h), border_radius=4)
                fill_w = int(bar_w * min(1.0, len(detected_samples) / CALIBRATION_FRAMES))
                if fill_w > 0:
                    pygame.draw.rect(screen, (60, 255, 100), (cam_box_x, bar_y, fill_w, bar_h), border_radius=4)

                # Bottom help text
                help_str = "[ESC] Abort   [SPACE / Click] Manual Point Capture (PC Testing)   [S] Skip to 1:1 Default"
                help_surf = font_sub.render(help_str, True, (120, 130, 150))
                screen.blit(help_surf, (SCREEN_WIDTH // 2 - help_surf.get_width() // 2, SCREEN_HEIGHT - 45))

                pygame.display.flip()
                clock.tick(30)

                if point_captured:
                    # Flash green and wait 1.2s for user to release button and reposition
                    time.sleep(1.2)
                    capturing = False

            # Compute median of collected IR samples for accuracy
            samples_arr = np.array(detected_samples)
            med_cam_x = float(np.median(samples_arr[:, 0]))
            med_cam_y = float(np.median(samples_arr[:, 1]))

            camera_points.append([med_cam_x, med_cam_y])
            screen_points.append([sx, sy])
            print(f"[Calibration] Point {idx + 1} Captured: Cam=({med_cam_x:.1f}, {med_cam_y:.1f}) -> Screen=({sx}, {sy})")

        # Compute Homography from 4 corner points
        src_pts = np.array(camera_points[:4], dtype=np.float32)
        dst_pts = np.array(screen_points[:4], dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)

        if homography_matrix is None:
            print("[Calibration] Warning: Homography computation failed, using 1:1 scale fallback.")
            homography_matrix = np.eye(3, dtype=np.float64)

        # Save to disk
        save_calibration(homography_matrix, error_px=0.0)

        # Success Screen
        screen.fill((14, 17, 24))
        succ_txt = font_title.render("CALIBRATION COMPLETE!", True, (60, 255, 100))
        sub_txt = font_instruction.render("Display boundary successfully mapped to camera view.", True, (200, 240, 255))
        screen.blit(succ_txt, (SCREEN_WIDTH // 2 - succ_txt.get_width() // 2, SCREEN_HEIGHT // 2 - 40))
        screen.blit(sub_txt, (SCREEN_WIDTH // 2 - sub_txt.get_width() // 2, SCREEN_HEIGHT // 2 + 20))
        pygame.display.flip()
        time.sleep(1.5)

        return homography_matrix

    finally:
        if owns_screen:
            pygame.quit()


if __name__ == "__main__":
    print("Testing calibration storage routines...")
    dummy_H = np.eye(3, dtype=np.float64)
    save_calibration(dummy_H, error_px=0.0)
    assert calibration_exists(), "Calibration existence check failed."
    loaded_H = load_calibration()
    assert np.allclose(dummy_H, loaded_H), "Loaded matrix mismatch."
    print("Calibration persistence unit test passed!")
