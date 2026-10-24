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
    MAX_VALIDATION_ERROR_PX
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
    Run interactive calibration procedure.
    Shows target dots on screen, detects their positions in camera space,
    computes perspective homography, validates accuracy, and persists the result.
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

    font_large = pygame.font.SysFont("Arial", 40)
    font_small = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()

    detector = IRDetector(threshold=CALIBRATION_IR_THRESHOLD)

    camera_points = []
    screen_points = []

    print("[Calibration] Starting calibration routine...")

    try:
        for idx, (sx, sy) in enumerate(CALIBRATION_POINTS):
            is_validation_pt = (idx == len(CALIBRATION_POINTS) - 1)
            point_label = "Center Validation" if is_validation_pt else f"Corner {idx + 1}/4"

            # Render calibration target on screen
            screen.fill((0, 0, 0))

            # Outer rings & crosshairs for visual alignment
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), CALIBRATION_DOT_RADIUS)
            pygame.draw.circle(screen, (0, 0, 0), (sx, sy), 6)
            pygame.draw.line(screen, (255, 255, 255), (sx - 50, sy), (sx + 50, sy), 2)
            pygame.draw.line(screen, (255, 255, 255), (sx, sy - 50), (sx, sy + 50), 2)

            msg = font_large.render(f"Calibrating Point {idx + 1} of {len(CALIBRATION_POINTS)} ({point_label})", True, (220, 220, 220))
            submsg = font_small.render("Point your IR pen / keep screen clear. Detecting bright point...", True, (160, 160, 160))
            screen.blit(msg, (SCREEN_WIDTH // 2 - msg.get_width() // 2, 60))
            screen.blit(submsg, (SCREEN_WIDTH // 2 - submsg.get_width() // 2, 115))
            pygame.display.flip()

            # Wait briefly for screen refresh to complete
            time.sleep(0.4)

            # Frame collection loop
            detected_samples = []
            start_time = time.time()

            while len(detected_samples) < CALIBRATION_FRAMES:
                # Pump Pygame events to keep UI responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt("Calibration aborted by user.")
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt("Calibration aborted by user via ESC.")

                if time.time() - start_time > CALIBRATION_TIMEOUT_SEC:
                    # If timeout occurs with at least half the needed frames, accept it; otherwise warn and retry
                    if len(detected_samples) >= max(3, CALIBRATION_FRAMES // 3):
                        break
                    print(f"[Calibration] Warning: Low detection confidence on point {idx + 1}. Retrying sample window...")
                    start_time = time.time()

                frame = camera.capture_frame()
                pt = detector.detect(frame)
                if pt is not None:
                    detected_samples.append(pt)

                clock.tick(30)

            # Compute median coordinates to eliminate noise outliers
            samples_arr = np.array(detected_samples)
            med_cam_x = float(np.median(samples_arr[:, 0]))
            med_cam_y = float(np.median(samples_arr[:, 1]))

            camera_points.append([med_cam_x, med_cam_y])
            screen_points.append([sx, sy])
            print(f"[Calibration] Point {idx + 1} ({point_label}) mapped: Cam=({med_cam_x:.1f}, {med_cam_y:.1f}) -> Screen=({sx}, {sy})")

            # Green flash feedback for user confirmation
            pygame.draw.circle(screen, (0, 255, 0), (sx, sy), CALIBRATION_DOT_RADIUS + 8, 4)
            pygame.display.flip()
            time.sleep(0.3)

        # Compute Homography from 4 corner points
        src_pts = np.array(camera_points[:4], dtype=np.float32)
        dst_pts = np.array(screen_points[:4], dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)

        if homography_matrix is None:
            raise RuntimeError("Failed to compute valid homography matrix. Ensure 4 distinct corner points.")

        # Center point accuracy validation
        val_error_px = None
        if len(camera_points) >= 5:
            center_cam = np.array([[[camera_points[4][0], camera_points[4][1]]]], dtype=np.float32)
            projected_center = cv2.perspectiveTransform(center_cam, homography_matrix)[0][0]
            expected_center = screen_points[4]
            val_error_px = float(np.hypot(projected_center[0] - expected_center[0], projected_center[1] - expected_center[1]))

            print(f"[Calibration] Center validation: projected=({projected_center[0]:.1f}, {projected_center[1]:.1f}), "
                  f"expected=({expected_center[0]}, {expected_center[1]}), error={val_error_px:.1f}px")

            if val_error_px > MAX_VALIDATION_ERROR_PX:
                print(f"[Calibration] WARNING: Calibration error ({val_error_px:.1f}px) exceeds threshold ({MAX_VALIDATION_ERROR_PX}px). "
                      "Consider recalibrating under more uniform lighting.")

        # Persist matrix to disk
        save_calibration(homography_matrix, error_px=val_error_px)

        # Show success screen
        screen.fill((0, 0, 0))
        success_txt = font_large.render("Calibration Completed Successfully!", True, (0, 255, 0))
        err_txt = font_small.render(f"Accuracy Error: {val_error_px:.1f}px" if val_error_px else "", True, (180, 180, 180))
        screen.blit(success_txt, (SCREEN_WIDTH // 2 - success_txt.get_width() // 2, SCREEN_HEIGHT // 2 - 40))
        screen.blit(err_txt, (SCREEN_WIDTH // 2 - err_txt.get_width() // 2, SCREEN_HEIGHT // 2 + 20))
        pygame.display.flip()
        time.sleep(1.2)

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
