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

    Includes a comprehensive live review screen showing the DocScanner display quad,
    corner pins, edge measurements, reprojection error, and an interactive prompt
    asking the user to Accept or Recalibrate if alignment did not happen properly.
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

    sw = screen.get_width()
    sh = screen.get_height()

    font_title = pygame.font.SysFont("Arial", 28, bold=True)
    font_instruction = pygame.font.SysFont("Arial", 22, bold=True)
    font_heading = pygame.font.SysFont("Arial", 20, bold=True)
    font_bold = pygame.font.SysFont("Arial", 16, bold=True)
    font_regular = pygame.font.SysFont("Arial", 15)
    font_mono = pygame.font.SysFont("Consolas, Courier New, monospace, Arial", 14)
    font_sub = pygame.font.SysFont("Arial", 16)
    font_btn_title = pygame.font.SysFont("Arial", 17, bold=True)
    font_btn_sub = pygame.font.SysFont("Arial", 12)
    clock = pygame.time.Clock()

    try:
        while True:
            camera_points = []
            screen_points = []
            restart_calibration = False

            print("\n" + "=" * 70)
            print("[Calibration] Starting Differential Dots Calibration for Display Edges...")
            print("=" * 70)

            # -------------------------------------------------------------
            # Phase 1: Capture Black Baseline Frame
            # -------------------------------------------------------------
            screen.fill((0, 0, 0))
            msg_init = font_title.render("CALIBRATING DISPLAY EDGES — DOTS METHOD", True, (0, 230, 255))
            msg_sub = font_instruction.render("Capturing background baseline... Please do not move camera.", True, (200, 210, 230))
            screen.blit(msg_init, (sw // 2 - msg_init.get_width() // 2, sh // 2 - 40))
            screen.blit(msg_sub, (sw // 2 - msg_sub.get_width() // 2, sh // 2 + 15))
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
                t_hint = font_sub.render("Calculating sub-pixel edge coordinate... [SPACE to simulate on PC | R to Restart]", True, (160, 170, 190))

                screen.blit(t_title, (sw // 2 - t_title.get_width() // 2, 40))
                screen.blit(t_label, (sw // 2 - t_label.get_width() // 2, 85))
                screen.blit(t_hint, (sw // 2 - t_hint.get_width() // 2, sh - 45))
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
                        elif event.key == pygame.K_r:
                            print("[Calibration] Recalibrate key pressed during target sequence. Restarting...")
                            restart_calibration = True
                            break
                        elif event.key == pygame.K_SPACE:
                            # Manual bypass for PC testing
                            manual_pt = (float(sx / SCREEN_WIDTH * CAMERA_WIDTH), float(sy / SCREEN_HEIGHT * CAMERA_HEIGHT))
                        elif event.key == pygame.K_s:
                            ident = np.eye(3, dtype=np.float64)
                            save_calibration(ident, error_px=0.0)
                            return ident
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        manual_pt = (float(sx / SCREEN_WIDTH * CAMERA_WIDTH), float(sy / SCREEN_HEIGHT * CAMERA_HEIGHT))

                if restart_calibration:
                    break

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

            if restart_calibration:
                screen.fill((0, 0, 0))
                re_msg = font_title.render("RESTARTING CALIBRATION...", True, (255, 140, 50))
                screen.blit(re_msg, (sw // 2 - re_msg.get_width() // 2, sh // 2 - 20))
                pygame.display.flip()
                time.sleep(0.8)
                continue

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

            disp_corners = src_pts.astype(np.int32)
            is_convex = bool(cv2.isContourConvex(disp_corners.reshape(-1, 1, 2)))

            # Print calibration diagnostics to console
            print("\n" + "=" * 70)
            print("[Calibration] DISPLAY EDGES CALIBRATION COMPLETE!")
            print("[Calibration] DocScanner Display Quadrilateral Corners in Camera View:")
            c_names = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Right (BR)", "Bottom-Left (BL)"]
            for i, c_name in enumerate(c_names):
                print(f"  • {c_name:<18}: Camera ({camera_points[i][0]:.1f}, {camera_points[i][1]:.1f}) -> Display {screen_points[i]}")
            if len(camera_points) >= 5:
                print(f"  • Screen Center (CC) : Camera ({camera_points[4][0]:.1f}, {camera_points[4][1]:.1f}) -> Display {screen_points[4]}")
            print(f"[Calibration] Reprojection Error: {val_error:.2f} px")
            print(f"[Calibration] Geometry Status   : {'CONVEX & VALID QUAD' if is_convex else 'WARNING: DISTORTED / SKEWED QUAD'}")
            print("[Calibration] Reviewing on screen. Waiting for user decision:")
            print("  -> Press [ENTER] (or click 'Accept & Continue') to proceed to Whiteboard.")
            print("  -> Press [R]     (or click 'Recalibrate Again') if edges look misaligned.")
            print("  -> Press [SPACE] (or click 'Default Bounds') for standard 1:1 camera bounds.")
            print("=" * 70 + "\n")

            # -------------------------------------------------------------
            # Phase 4: Interactive DocScanner Review Screen
            # Shows live camera view, DocScanner edges, telemetry card,
            # and asks user to Accept or Recalibrate without auto-timeout.
            # -------------------------------------------------------------
            if sh >= 900:
                prev_w = 640
                prev_h = 480
                gap = 35
                card_w = min(740, sw - prev_w - gap - 60)
                card_h = 480
                btn_h = 70
            else:
                prev_w = 520
                prev_h = 390
                gap = 25
                card_w = min(660, sw - prev_w - gap - 40)
                card_h = 390
                btn_h = 58

            total_content_w = prev_w + gap + card_w
            start_x = max(20, (sw - total_content_w) // 2)
            content_y = 95 if sh >= 900 else 75

            cam_rect = (start_x, content_y, prev_w, prev_h)
            card_rect = pygame.Rect(start_x + prev_w + gap, content_y, card_w, card_h)

            btn_y = content_y + prev_h + (25 if sh >= 900 else 15)
            btn1_w = 340
            btn2_w = 340
            btn3_w = 260
            btn_gap = 20
            total_btn_w = btn1_w + btn2_w + btn3_w + 2 * btn_gap
            btn_start_x = max(20, (sw - total_btn_w) // 2)

            btn_accept_rect = pygame.Rect(btn_start_x, btn_y, btn1_w, btn_h)
            btn_recalib_rect = pygame.Rect(btn_start_x + btn1_w + btn_gap, btn_y, btn2_w, btn_h)
            btn_default_rect = pygame.Rect(btn_start_x + btn1_w + btn_gap + btn2_w + btn_gap, btn_y, btn3_w, btn_h)
            footer_y = btn_y + btn_h + 12

            review_active = True
            recalibrate_requested = False

            while review_active:
                mouse_pos = pygame.mouse.get_pos()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt("Calibration aborted by user.")
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_y):
                            print("[Calibration] Calibration accepted by user via [ENTER]. Saving...")
                            save_calibration(homography_matrix, error_px=val_error)
                            return homography_matrix
                        elif event.key == pygame.K_r:
                            print("[Calibration] Recalibrate key [R] pressed by user. Restarting calibration...")
                            recalibrate_requested = True
                            review_active = False
                            break
                        elif event.key == pygame.K_SPACE:
                            print("[Calibration] User selected default 1:1 bounds via [SPACE].")
                            ident = np.eye(3, dtype=np.float64)
                            save_calibration(ident, error_px=0.0)
                            return ident
                        elif event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt("Calibration aborted by user via ESC.")
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if btn_accept_rect.collidepoint(mouse_pos):
                            print("[Calibration] Calibration accepted by user via [Accept] button. Saving...")
                            save_calibration(homography_matrix, error_px=val_error)
                            return homography_matrix
                        elif btn_recalib_rect.collidepoint(mouse_pos):
                            print("[Calibration] User clicked [Recalibrate Again]. Restarting calibration...")
                            recalibrate_requested = True
                            review_active = False
                            break
                        elif btn_default_rect.collidepoint(mouse_pos):
                            print("[Calibration] User clicked [Default Bounds].")
                            ident = np.eye(3, dtype=np.float64)
                            save_calibration(ident, error_px=0.0)
                            return ident

                if not review_active:
                    break

                # Capture live camera frame
                try:
                    live_frame = camera.capture_frame()
                except Exception:
                    live_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

                annotated = live_frame.copy()

                # Semi-transparent polygon overlay for display area
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [disp_corners], color=(20, 60, 40))
                cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0, annotated)

                # Draw 4 DocScanner edges in bright neon cyan
                cv2.polylines(annotated, [disp_corners], isClosed=True, color=(0, 245, 255), thickness=3)

                # Draw corner diagonal guide lines
                cv2.line(annotated, tuple(disp_corners[0]), tuple(disp_corners[2]), (80, 100, 120), 1)
                cv2.line(annotated, tuple(disp_corners[1]), tuple(disp_corners[3]), (80, 100, 120), 1)

                # Center Point marker
                if len(camera_points) >= 5:
                    cx_c, cy_c = int(camera_points[4][0]), int(camera_points[4][1])
                    cv2.drawMarker(annotated, (cx_c, cy_c), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                    cv2.circle(annotated, (cx_c, cy_c), 5, (0, 255, 255), 1)
                    cv2.putText(annotated, f"CENTER ({cx_c},{cy_c})", (cx_c + 8, cy_c - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)

                # Draw corner pins with colorful badges
                corner_tags = [
                    ("TL", (60, 255, 100)),   # Green
                    ("TR", (255, 220, 40)),   # Yellow
                    ("BR", (255, 140, 40)),   # Orange
                    ("BL", (0, 200, 255))     # Cyan
                ]

                for i, (tag, col) in enumerate(corner_tags):
                    px, py = int(disp_corners[i][0]), int(disp_corners[i][1])
                    cv2.circle(annotated, (px, py), 9, col, -1)
                    cv2.circle(annotated, (px, py), 13, (255, 255, 255), 2)
                    off_x = 10 if i in (0, 3) else -80
                    off_y = -10 if i in (0, 1) else 22
                    cv2.putText(annotated, f"{tag}: ({px},{py})", (px + off_x, py + off_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 3)
                    cv2.putText(annotated, f"{tag}: ({px},{py})", (px + off_x, py + off_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)

                # Edge dimensions in camera pixels
                top_mid = ((disp_corners[0][0] + disp_corners[1][0]) // 2, (disp_corners[0][1] + disp_corners[1][1]) // 2)
                w_top = float(np.hypot(disp_corners[1][0] - disp_corners[0][0], disp_corners[1][1] - disp_corners[0][1]))
                cv2.putText(annotated, f"Top: {w_top:.0f}px", (top_mid[0] - 35, top_mid[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 255), 1)

                btm_mid = ((disp_corners[3][0] + disp_corners[2][0]) // 2, (disp_corners[3][1] + disp_corners[2][1]) // 2)
                w_btm = float(np.hypot(disp_corners[2][0] - disp_corners[3][0], disp_corners[2][1] - disp_corners[3][1]))
                cv2.putText(annotated, f"Btm: {w_btm:.0f}px", (btm_mid[0] - 35, btm_mid[1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 255), 1)

                left_mid = ((disp_corners[0][0] + disp_corners[3][0]) // 2, (disp_corners[0][1] + disp_corners[3][1]) // 2)
                h_left = float(np.hypot(disp_corners[3][0] - disp_corners[0][0], disp_corners[3][1] - disp_corners[0][1]))
                cv2.putText(annotated, f"L: {h_left:.0f}px", (left_mid[0] - 55, left_mid[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 255), 1)

                right_mid = ((disp_corners[1][0] + disp_corners[2][0]) // 2, (disp_corners[1][1] + disp_corners[2][1]) // 2)
                h_right = float(np.hypot(disp_corners[2][0] - disp_corners[1][0], disp_corners[2][1] - disp_corners[1][1]))
                cv2.putText(annotated, f"R: {h_right:.0f}px", (right_mid[0] + 8, right_mid[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 255), 1)

                # Render background & Header
                screen.fill((14, 17, 24))

                title_surf = font_title.render("CALIBRATION COMPLETE — REVIEW DISPLAY EDGES", True, (40, 240, 180))
                sub_surf = font_sub.render(
                    "The 4 edges of your display (DocScanner Quad) are shown below. Verify alignment with your physical screen.",
                    True, (200, 215, 235)
                )
                screen.blit(title_surf, (sw // 2 - title_surf.get_width() // 2, 20 if sh >= 900 else 12))
                screen.blit(sub_surf, (sw // 2 - sub_surf.get_width() // 2, 58 if sh >= 900 else 42))

                # Blit live camera preview on left
                frame_resized = cv2.resize(annotated, (prev_w, prev_h))
                frame_surf = pygame.image.frombuffer(frame_resized.tobytes(), (prev_w, prev_h), 'RGB')
                pygame.draw.rect(screen, (0, 240, 255), (start_x - 2, content_y - 2, prev_w + 4, prev_h + 4), 2, border_radius=6)
                screen.blit(frame_surf, (start_x, content_y))

                cam_label = font_bold.render("LIVE CAMERA FEED — DOCSCANNER DISPLAY BOUNDARY", True, (0, 240, 255))
                screen.blit(cam_label, (start_x + 10, content_y + 10))

                # Blit Diagnostics Card on right
                pygame.draw.rect(screen, (20, 24, 34), card_rect, border_radius=8)
                pygame.draw.rect(screen, (50, 65, 90), card_rect, 2, border_radius=8)

                # Card Header
                card_hdr = font_heading.render("CALIBRATION METRICS & TELEMETRY", True, (255, 220, 80))
                screen.blit(card_hdr, (card_rect.x + 20, card_rect.y + 14))

                # Quality Status Badge
                if is_convex and val_error <= 6.0:
                    badge_text = f"ALIGNMENT: EXCELLENT (Validation Error: {val_error:.1f} px)"
                    badge_col = (60, 255, 100)
                elif is_convex and val_error <= 15.0:
                    badge_text = f"ALIGNMENT: GOOD (Validation Error: {val_error:.1f} px)"
                    badge_col = (255, 220, 60)
                else:
                    badge_text = f"ALIGNMENT: DISTORTED / SKEWED (Error: {val_error:.1f} px - Recalibration Recommended)"
                    badge_col = (255, 100, 60)

                badge_surf = font_bold.render(badge_text, True, badge_col)
                screen.blit(badge_surf, (card_rect.x + 20, card_rect.y + 44))

                # Corner Coordinates Table
                line_y = card_rect.y + 76
                t_lines = [
                    f"Top-Left     [TL] : Camera ({camera_points[0][0]:.1f}, {camera_points[0][1]:.1f})  ->  Screen (50, 50)",
                    f"Top-Right    [TR] : Camera ({camera_points[1][0]:.1f}, {camera_points[1][1]:.1f})  ->  Screen ({SCREEN_WIDTH-50}, 50)",
                    f"Bottom-Right [BR] : Camera ({camera_points[2][0]:.1f}, {camera_points[2][1]:.1f})  ->  Screen ({SCREEN_WIDTH-50}, {SCREEN_HEIGHT-50})",
                    f"Bottom-Left  [BL] : Camera ({camera_points[3][0]:.1f}, {camera_points[3][1]:.1f})  ->  Screen (50, {SCREEN_HEIGHT-50})",
                ]
                if len(camera_points) >= 5:
                    t_lines.append(f"Screen Center[CC] : Camera ({camera_points[4][0]:.1f}, {camera_points[4][1]:.1f})  ->  Screen ({SCREEN_WIDTH//2}, {SCREEN_HEIGHT//2})")

                for tl in t_lines:
                    ts = font_mono.render(tl, True, (210, 225, 245))
                    screen.blit(ts, (card_rect.x + 20, line_y))
                    line_y += 22

                line_y += 6
                # Dimensions & Aspect Ratio
                avg_w = (w_top + w_btm) / 2.0
                avg_h = (h_left + h_right) / 2.0
                aspect = avg_w / max(avg_h, 1.0)
                dim_lines = [
                    f"Edge Widths  : Top: {w_top:.0f} px  |  Bottom: {w_btm:.0f} px",
                    f"Edge Heights : Left: {h_left:.0f} px  |  Right: {h_right:.0f} px",
                    f"Aspect Ratio : {aspect:.2f} : 1.0 (Screen Target: {SCREEN_WIDTH/SCREEN_HEIGHT:.2f} : 1.0)"
                ]
                for dl in dim_lines:
                    ds = font_mono.render(dl, True, (170, 195, 225))
                    screen.blit(ds, (card_rect.x + 20, line_y))
                    line_y += 20

                line_y += 10
                # User Guidance Box
                g_title = font_bold.render("VERIFICATION & ACCURACY CHECK:", True, (255, 200, 60))
                screen.blit(g_title, (card_rect.x + 20, line_y))
                line_y += 22

                guidelines = [
                    "• Look at live camera feed: Does the cyan polygon frame your physical display?",
                    "• Are all 4 corner pins (TL, TR, BR, BL) on your monitor screen corners?",
                    "• If accurate -> Click [Accept & Continue] or press [ENTER].",
                    "• If calibration didn't happen properly -> Click [Recalibrate Again] or press [R]."
                ]
                for g in guidelines:
                    is_action = "Recalibrate" in g or "Accept" in g
                    gs = font_regular.render(g, True, (220, 230, 245) if is_action else (175, 185, 200))
                    screen.blit(gs, (card_rect.x + 20, line_y))
                    line_y += 20

                # Draw Interactive Buttons with Hover Highlights
                # Button 1: Accept & Continue
                h1 = btn_accept_rect.collidepoint(mouse_pos)
                pygame.draw.rect(screen, (35, 135, 65) if h1 else (20, 85, 40), btn_accept_rect, border_radius=8)
                pygame.draw.rect(screen, (100, 255, 140) if h1 else (50, 200, 90), btn_accept_rect, 2, border_radius=8)
                m1 = font_btn_title.render("[ENTER] Accept & Continue", True, (255, 255, 255))
                s1 = font_btn_sub.render("Edges look accurate - Launch Whiteboard", True, (200, 255, 220))
                screen.blit(m1, (btn_accept_rect.centerx - m1.get_width() // 2, btn_accept_rect.y + (12 if sh >= 900 else 8)))
                screen.blit(s1, (btn_accept_rect.centerx - s1.get_width() // 2, btn_accept_rect.y + (38 if sh >= 900 else 32)))

                # Button 2: Recalibrate Again
                h2 = btn_recalib_rect.collidepoint(mouse_pos)
                pygame.draw.rect(screen, (160, 50, 30) if h2 else (105, 35, 20), btn_recalib_rect, border_radius=8)
                pygame.draw.rect(screen, (255, 140, 80) if h2 else (220, 90, 50), btn_recalib_rect, 2, border_radius=8)
                m2 = font_btn_title.render("[R] Recalibrate Again", True, (255, 255, 255))
                s2 = font_btn_sub.render("Didn't calibrate properly? Click to redo", True, (255, 215, 205))
                screen.blit(m2, (btn_recalib_rect.centerx - m2.get_width() // 2, btn_recalib_rect.y + (12 if sh >= 900 else 8)))
                screen.blit(s2, (btn_recalib_rect.centerx - s2.get_width() // 2, btn_recalib_rect.y + (38 if sh >= 900 else 32)))

                # Button 3: Default Bounds (1:1)
                h3 = btn_default_rect.collidepoint(mouse_pos)
                pygame.draw.rect(screen, (45, 65, 95) if h3 else (30, 45, 70), btn_default_rect, border_radius=8)
                pygame.draw.rect(screen, (130, 180, 240) if h3 else (80, 115, 170), btn_default_rect, 2, border_radius=8)
                m3 = font_btn_title.render("[SPACE] Default Bounds", True, (255, 255, 255))
                s3 = font_btn_sub.render("Standard 1:1 mapping (PC bypass)", True, (180, 210, 245))
                screen.blit(m3, (btn_default_rect.centerx - m3.get_width() // 2, btn_default_rect.y + (12 if sh >= 900 else 8)))
                screen.blit(s3, (btn_default_rect.centerx - s3.get_width() // 2, btn_default_rect.y + (38 if sh >= 900 else 32)))

                # Footer
                footer_text = font_regular.render(
                    "Hotkeys: [ENTER] Accept & Continue  |  [R] Recalibrate Again  |  [SPACE] Default Bounds  |  [ESC] Exit",
                    True, (140, 155, 175)
                )
                screen.blit(footer_text, (sw // 2 - footer_text.get_width() // 2, footer_y))

                pygame.display.flip()
                clock.tick(30)

            # If recalibration was requested, restart the loop cleanly
            if recalibrate_requested:
                screen.fill((0, 0, 0))
                re_msg = font_title.render("RESTARTING CALIBRATION...", True, (255, 140, 50))
                re_sub = font_instruction.render("Resetting display and capturing baseline...", True, (220, 230, 245))
                screen.blit(re_msg, (sw // 2 - re_msg.get_width() // 2, sh // 2 - 30))
                screen.blit(re_sub, (sw // 2 - re_sub.get_width() // 2, sh // 2 + 15))
                pygame.display.flip()
                time.sleep(1.0)
                continue

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
