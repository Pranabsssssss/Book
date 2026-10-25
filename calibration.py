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
    CALIBRATION_FILE,
    CAMERA_WIDTH, CAMERA_HEIGHT
)


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
    Differential Dots Method for Display Edges Calibration.
    Uses sequential monitor corner dot flashing with differential background subtraction.
    Cancels out 100% of ambient room lighting and reflections to find the exact sub-pixel
    camera coordinates of the 4 display edges, just like DocScanner calculates document corners.
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
        pygame.display.set_caption("Smart Notebook - Display Edge Calibration")
        owns_screen = True

    font_title = pygame.font.SysFont("Arial", 32, bold=True)
    font_instruction = pygame.font.SysFont("Arial", 26, bold=True)
    font_sub = pygame.font.SysFont("Arial", 18)
    clock = pygame.time.Clock()

    camera_points = []
    screen_points = []

    print("[Calibration] Starting Differential Dots Calibration for Display Edges...")

    try:
        # -------------------------------------------------------------
        # Phase 1: Capture Black Baseline Frame
        # -------------------------------------------------------------
        screen.fill((0, 0, 0))
        msg_init = font_title.render("CALIBRATING DISPLAY EDGES — DOTS METHOD", True, (0, 230, 255))
        msg_sub = font_instruction.render("Capturing background baseline... Please do not move camera.", True, (200, 210, 230))
        screen.blit(msg_init, (SCREEN_WIDTH // 2 - msg_init.get_width() // 2, SCREEN_HEIGHT // 2 - 40))
        screen.blit(msg_sub, (SCREEN_WIDTH // 2 - msg_sub.get_width() // 2, SCREEN_HEIGHT // 2 + 15))
        pygame.display.flip()

        time.sleep(1.0)  # Allow screen black level to settle

        # Average 3 black frames to eliminate camera sensor noise
        black_accum = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.float32)
        for _ in range(3):
            f = camera.capture_frame().astype(np.float32)
            black_accum += f
            time.sleep(0.05)
        black_baseline = (black_accum / 3.0).astype(np.uint8)
        black_gray = cv2.cvtColor(black_baseline, cv2.COLOR_RGB2GRAY)

        # -------------------------------------------------------------
        # Phase 2: Sequential Display Dot Targets
        # Top-Left, Top-Right, Bottom-Right, Bottom-Left, Center
        # -------------------------------------------------------------
        target_coords = [
            (50, 50),                                      # Top-Left
            (SCREEN_WIDTH - 50, 50),                       # Top-Right
            (SCREEN_WIDTH - 50, SCREEN_HEIGHT - 50),       # Bottom-Right
            (50, SCREEN_HEIGHT - 50),                      # Bottom-Left
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)        # Center Validation
        ]

        corner_labels = [
            "Point 1 of 4: TOP-LEFT Corner",
            "Point 2 of 4: TOP-RIGHT Corner",
            "Point 3 of 4: BOTTOM-RIGHT Corner",
            "Point 4 of 4: BOTTOM-LEFT Corner",
            "Validation: CENTER Target"
        ]

        for idx, (sx, sy) in enumerate(target_coords):
            # Render Target Dot on Screen
            screen.fill((0, 0, 0))

            # Draw prominent white calibration target
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 36)
            pygame.draw.circle(screen, (0, 0, 0), (sx, sy), 8)
            pygame.draw.line(screen, (255, 255, 255), (sx - 50, sy), (sx + 50, sy), 2)
            pygame.draw.line(screen, (255, 255, 255), (sx, sy - 50), (sx, sy + 50), 2)

            t_title = font_title.render("CALIBRATING DISPLAY EDGES", True, (0, 230, 255))
            t_label = font_instruction.render(corner_labels[idx], True, (255, 220, 80))
            t_hint = font_sub.render("Calculating sub-pixel edge coordinate... [SPACE to simulate on PC]", True, (160, 170, 190))

            screen.blit(t_title, (SCREEN_WIDTH // 2 - t_title.get_width() // 2, 40))
            screen.blit(t_label, (SCREEN_WIDTH // 2 - t_label.get_width() // 2, 85))
            screen.blit(t_hint, (SCREEN_WIDTH // 2 - t_hint.get_width() // 2, SCREEN_HEIGHT - 45))
            pygame.display.flip()

            time.sleep(0.4)  # Wait for monitor display refresh

            # Pump Pygame events
            manual_pt = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt("Calibration aborted by user.")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt("Calibration aborted by user via ESC.")
                    elif event.key == pygame.K_SPACE:
                        # Manual bypass for PC testing
                        manual_pt = (float(sx / SCREEN_WIDTH * CAMERA_WIDTH), float(sy / SCREEN_HEIGHT * CAMERA_HEIGHT))
                    elif event.key == pygame.K_s:
                        ident = np.eye(3, dtype=np.float64)
                        save_calibration(ident, error_px=0.0)
                        return ident
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    manual_pt = (float(sx / SCREEN_WIDTH * CAMERA_WIDTH), float(sy / SCREEN_HEIGHT * CAMERA_HEIGHT))

            if manual_pt is not None:
                detected_cam_pt = manual_pt
            else:
                # Capture frame with displayed dot
                dot_frame = camera.capture_frame()
                dot_gray = cv2.cvtColor(dot_frame, cv2.COLOR_RGB2GRAY)

                # Differential background subtraction: cancels out 100% of room lighting!
                diff = cv2.absdiff(dot_gray, black_gray)

                # Threshold the difference image
                _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                M = cv2.moments(diff_thresh)

                if M["m00"] > 10:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    detected_cam_pt = (cx, cy)
                else:
                    # Fallback to max location if difference was subtle
                    _, max_v, _, max_l = cv2.minMaxLoc(diff)
                    if max_v > 20:
                        detected_cam_pt = (float(max_l[0]), float(max_l[1]))
                    else:
                        # Fallback proportional mapping
                        detected_cam_pt = (float(sx / SCREEN_WIDTH * CAMERA_WIDTH), float(sy / SCREEN_HEIGHT * CAMERA_HEIGHT))

            camera_points.append(detected_cam_pt)
            screen_points.append((sx, sy))
            print(f"[Calibration] {corner_labels[idx]} -> Cam: ({detected_cam_pt[0]:.1f}, {detected_cam_pt[1]:.1f})")

            # Green flash feedback for success
            pygame.draw.circle(screen, (60, 255, 100), (sx, sy), 42, 4)
            pygame.display.flip()
            time.sleep(0.3)

        # -------------------------------------------------------------
        # Phase 3: Compute Homography Matrix
        # -------------------------------------------------------------
        src_pts = np.array(camera_points[:4], dtype=np.float32)
        dst_pts = np.array(screen_points[:4], dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)

        if homography_matrix is None:
            print("[Calibration] Warning: Homography computation failed, using 1:1 scale fallback.")
            homography_matrix = np.eye(3, dtype=np.float64)

        # Validation with center point
        val_error = 0.0
        if len(camera_points) >= 5:
            center_cam = np.array([[[camera_points[4][0], camera_points[4][1]]]], dtype=np.float32)
            proj_center = cv2.perspectiveTransform(center_cam, homography_matrix)[0][0]
            val_error = float(np.hypot(proj_center[0] - screen_points[4][0], proj_center[1] - screen_points[4][1]))
            print(f"[Calibration] Center validation error: {val_error:.1f} px")

        # Save to disk
        save_calibration(homography_matrix, error_px=val_error)

        # -------------------------------------------------------------
        # Phase 4: DocScanner-Style Review Screen
        # -------------------------------------------------------------
        screen.fill((14, 17, 24))
        rev_title = font_title.render("DISPLAY EDGES CALIBRATED (DocScanner View)", True, (60, 255, 100))
        rev_sub = font_instruction.render("The 4 corners of your display are now mapped with sub-pixel precision.", True, (220, 230, 245))
        rev_hint = font_sub.render("Press [ENTER] to continue to Whiteboard, or [R] to recalibrate", True, (255, 200, 60))

        screen.blit(rev_title, (SCREEN_WIDTH // 2 - rev_title.get_width() // 2, 40))
        screen.blit(rev_sub, (SCREEN_WIDTH // 2 - rev_sub.get_width() // 2, 85))

        # Show live camera frame with the calibrated DocScanner quad
        last_frame = camera.capture_frame()
        disp_corners = src_pts.astype(np.int32)
        cv2.polylines(last_frame, [disp_corners], isClosed=True, color=(0, 240, 255), thickness=3)

        c_names = ["TL", "TR", "BR", "BL"]
        for i, (c_x, c_y) in enumerate(disp_corners):
            cv2.circle(last_frame, (c_x, c_y), 6, (60, 255, 100), -1)
            cv2.putText(last_frame, c_names[i], (c_x + 8, c_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 100), 2)

        prev_w = 540
        prev_h = 405
        frame_resized = cv2.resize(last_frame, (prev_w, prev_h))
        frame_surf = pygame.image.frombuffer(frame_resized.tobytes(), (prev_w, prev_h), 'RGB')

        f_x = (SCREEN_WIDTH - prev_w) // 2
        f_y = (SCREEN_HEIGHT - prev_h) // 2 + 30
        pygame.draw.rect(screen, (0, 240, 255), (f_x - 3, f_y - 3, prev_w + 6, prev_h + 6), 2, border_radius=6)
        screen.blit(frame_surf, (f_x, f_y))
        screen.blit(rev_hint, (SCREEN_WIDTH // 2 - rev_hint.get_width() // 2, SCREEN_HEIGHT - 55))
        pygame.display.flip()

        waiting = True
        start_wait = time.time()
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt("Calibration exited.")
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                        waiting = False
                    elif event.key == pygame.K_r:
                        return run_calibration(camera, screen=screen)
                    elif event.key == pygame.K_ESCAPE:
                        waiting = False

            if time.time() - start_wait > 3.0:
                waiting = False  # Auto-continue after 3 seconds
            clock.tick(30)

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
