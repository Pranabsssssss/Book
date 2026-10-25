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
    CALIBRATION_DOT_RADIUS,
    CALIBRATION_FILE,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    CALIBRATION_IR_THRESHOLD,
    get_calibration_points, get_display_resolution
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


def load_calibration_resolution() -> tuple[int, int]:
    """Load the screen resolution associated with the saved calibration data."""
    if not os.path.exists(CALIBRATION_FILE):
        return get_display_resolution(SCREEN_WIDTH, SCREEN_HEIGHT)
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            data = json.load(f)
        res = data.get("screen_resolution", [SCREEN_WIDTH, SCREEN_HEIGHT])
        return int(res[0]), int(res[1])
    except Exception:
        return get_display_resolution(SCREEN_WIDTH, SCREEN_HEIGHT)


def save_calibration(matrix: np.ndarray, error_px: float = None, screen_resolution: tuple = None, points: list = None):
    """Save homography matrix and display metadata to calibration.json."""
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    res = list(screen_resolution) if screen_resolution else [SCREEN_WIDTH, SCREEN_HEIGHT]
    pts = points if points is not None else get_calibration_points(int(res[0]), int(res[1]))
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "screen_resolution": [int(res[0]), int(res[1])],
        "validation_error_px": float(error_px) if error_px is not None else None,
        "calibration_points": pts,
        "homography_matrix": matrix.tolist()
    }
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[Calibration] Saved matrix for {res[0]}x{res[1]} display to: {CALIBRATION_FILE}")


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points of a quadrilateral:
    Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-Left has smallest sum (x + y), Bottom-Right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-Right has smallest difference (y - x), Bottom-Left has largest difference
    diff = np.diff(pts, axis=1).flatten()  # y - x
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def detect_docscanner_quad(frame: np.ndarray) -> np.ndarray:
    """
    DocScanner / Page Scanner algorithm: automatically locates the
    monitor display quadrilateral in the camera view using computer vision.
    Returns: (4, 2) array of [TL, TR, BR, BL] or None.
    """
    if frame is None or frame.size == 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Edge-based polygon detection (Canny)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        area = cv2.contourArea(c)
        if area < 10000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_quad_points(approx.reshape(4, 2))

    # 2. Threshold-based fallback (Otsu threshold on glowing monitor display)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours2, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:6]

    for c in contours2:
        area = cv2.contourArea(c)
        if area < 10000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_quad_points(approx.reshape(4, 2))
        elif len(approx) >= 4:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            if cv2.contourArea(box) > 10000:
                return order_quad_points(box)

    return None


def validate_corner_points(pts: list, cam_w: int, cam_h: int) -> tuple[bool, list]:
    """
    Ensures the 4 detected corners form a wide, non-collapsed quadrilateral.
    Prevents points from clustering in the center.
    """
    if len(pts) < 4:
        return False, pts

    d_top = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
    d_right = float(np.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]))
    d_bottom = float(np.hypot(pts[3][0] - pts[2][0], pts[3][1] - pts[2][1]))
    d_left = float(np.hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1]))

    quad_arr = np.array(pts[:4], dtype=np.float32)
    area = float(cv2.contourArea(quad_arr))

    # If area is too small or edges are under 60px, points collapsed into the center!
    if area < 8000.0 or min(d_top, d_right, d_bottom, d_left) < 60.0:
        fallback_corners = [
            (25.0, 25.0),
            (float(cam_w - 25), 25.0),
            (float(cam_w - 25), float(cam_h - 25)),
            (25.0, float(cam_h - 25))
        ]
        return False, fallback_corners

    return True, pts


def run_calibration(camera, screen=None) -> np.ndarray:
    """
    Dual Mode Calibration System:
      1. AUTO DOCSCANNER: Scans camera feed using computer vision to detect the
         monitor screen edges automatically (Page Scanner algorithm).
      2. MANUAL MOUSE: Lets the user click on the 4 corners of their display in the
         camera feed, and drag corner pins to fine-tune alignment with millimeter precision.

    Includes draggable corner handles in both modes so the user can easily tweak
    any corner pin with their mouse!
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
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Smart Notebook - Display Calibration")
        owns_screen = True

    sw = screen.get_width()
    sh = screen.get_height()
    print(f"[Calibration] Active display resolution: {sw}x{sh} (Aspect ratio: {sw/sh:.2f}:1)")

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

    target_coords = get_calibration_points(sw, sh)
    dst_pts = np.array(target_coords[:4], dtype=np.float32)

    try:
        # =============================================================
        # STEP 1: Calibration Mode Selection Screen (Auto vs Manual)
        # =============================================================
        calib_mode = None  # "auto" or "manual"
        menu_active = True

        card1_w = min(460, (sw - 80) // 2)
        card1_h = 240
        card_y = sh // 2 - 110
        gap = 40
        card1_x = sw // 2 - card1_w - gap // 2
        card2_x = sw // 2 + gap // 2

        card_auto_rect = pygame.Rect(card1_x, card_y, card1_w, card1_h)
        card_manual_rect = pygame.Rect(card2_x, card_y, card1_w, card1_h)

        print("[Calibration] Prompting user for calibration method (Auto DocScanner vs Manual Mouse)...")

        while menu_active:
            mouse_pos = pygame.mouse.get_pos()
            hover_auto = card_auto_rect.collidepoint(mouse_pos)
            hover_manual = card_manual_rect.collidepoint(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt("Calibration aborted by user.")
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_1, pygame.K_a):
                        calib_mode = "auto"
                        menu_active = False
                    elif event.key in (pygame.K_2, pygame.K_m):
                        calib_mode = "manual"
                        menu_active = False
                    elif event.key == pygame.K_SPACE:
                        # Skip directly with 1:1 fallback
                        ident = np.eye(3, dtype=np.float64)
                        save_calibration(ident, error_px=0.0, screen_resolution=(sw, sh), points=target_coords)
                        return ident
                    elif event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt("Calibration aborted by user via ESC.")
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if hover_auto:
                        calib_mode = "auto"
                        menu_active = False
                    elif hover_manual:
                        calib_mode = "manual"
                        menu_active = False

            # Render Selection Screen
            screen.fill((14, 17, 24))

            title_surf = font_title.render("SMART NOTEBOOK — DISPLAY CALIBRATION", True, (40, 240, 180))
            sub_surf = font_sub.render("Choose how you want to calibrate your monitor display boundaries:", True, (200, 215, 235))
            screen.blit(title_surf, (sw // 2 - title_surf.get_width() // 2, card_y - 95))
            screen.blit(sub_surf, (sw // 2 - sub_surf.get_width() // 2, card_y - 55))

            # Card 1: AUTO DOCSCANNER
            c1_bg = (18, 55, 75) if hover_auto else (12, 35, 48)
            c1_border = (0, 220, 255) if hover_auto else (0, 150, 180)
            pygame.draw.rect(screen, c1_bg, card_auto_rect, border_radius=12)
            pygame.draw.rect(screen, c1_border, card_auto_rect, 3 if hover_auto else 2, border_radius=12)

            t_a1 = font_heading.render("[1] AUTO DOCSCANNER MODE", True, (0, 240, 255))
            t_a2 = font_regular.render("Computer Vision Page Scanner algorithm.", True, (230, 240, 255))
            t_a3 = font_regular.render("The screen flashes an illuminated border;", True, (170, 195, 220))
            t_a4 = font_regular.render("the camera automatically detects display edges.", True, (170, 195, 220))
            t_a5 = font_bold.render("Click here or press [1] to start Auto", True, (60, 255, 100))

            screen.blit(t_a1, (card_auto_rect.x + 25, card_auto_rect.y + 25))
            screen.blit(t_a2, (card_auto_rect.x + 25, card_auto_rect.y + 68))
            screen.blit(t_a3, (card_auto_rect.x + 25, card_auto_rect.y + 100))
            screen.blit(t_a4, (card_auto_rect.x + 25, card_auto_rect.y + 125))
            screen.blit(t_a5, (card_auto_rect.x + 25, card_auto_rect.y + 180))

            # Card 2: MANUAL MOUSE CALIBRATION
            c2_bg = (65, 40, 18) if hover_manual else (42, 25, 12)
            c2_border = (255, 160, 40) if hover_manual else (190, 110, 25)
            pygame.draw.rect(screen, c2_bg, card_manual_rect, border_radius=12)
            pygame.draw.rect(screen, c2_border, card_manual_rect, 3 if hover_manual else 2, border_radius=12)

            t_m1 = font_heading.render("[2] MANUAL MOUSE CALIBRATION", True, (255, 180, 50))
            t_m2 = font_regular.render("Interactive 4-corner mouse placement.", True, (255, 235, 215))
            t_m3 = font_regular.render("Click directly on your screen corners in camera feed,", True, (230, 195, 170))
            t_m4 = font_regular.render("then drag pins to align with millimeter precision.", True, (230, 195, 170))
            t_m5 = font_bold.render("Click here or press [2] to start Manual", True, (255, 220, 80))

            screen.blit(t_m1, (card_manual_rect.x + 25, card_manual_rect.y + 25))
            screen.blit(t_m2, (card_manual_rect.x + 25, card_manual_rect.y + 68))
            screen.blit(t_m3, (card_manual_rect.x + 25, card_manual_rect.y + 100))
            screen.blit(t_m4, (card_manual_rect.x + 25, card_manual_rect.y + 125))
            screen.blit(t_m5, (card_manual_rect.x + 25, card_manual_rect.y + 180))

            hint_text = font_regular.render("[SPACE] Use 1:1 Default Bounds   |   [ESC] Exit", True, (130, 145, 165))
            screen.blit(hint_text, (sw // 2 - hint_text.get_width() // 2, sh - 45))

            pygame.display.flip()
            clock.tick(30)

        # =============================================================
        # STEP 2: Execute Initial Setup Based on Mode
        # =============================================================
        pins = [
            (25.0, 25.0),
            (float(CAMERA_WIDTH - 25), 25.0),
            (float(CAMERA_WIDTH - 25), float(CAMERA_HEIGHT - 25)),
            (25.0, float(CAMERA_HEIGHT - 25))
        ]
        manual_click_mode = False
        manual_step = 4  # 0 to 3 when placing points, 4 when all placed

        if calib_mode == "auto":
            print("[Calibration] Running Auto DocScanner Display Boundary Detection...")
            # Flash high-contrast perimeter boundary for camera detection
            screen.fill((0, 0, 0))
            # 35px white border around perimeter of screen
            pygame.draw.rect(screen, (255, 255, 255), (0, 0, sw, sh), 35)
            # Corner markers
            pygame.draw.rect(screen, (255, 255, 255), (0, 0, 120, 120))
            pygame.draw.rect(screen, (255, 255, 255), (sw - 120, 0, 120, 120))
            pygame.draw.rect(screen, (255, 255, 255), (sw - 120, sh - 120, 120, 120))
            pygame.draw.rect(screen, (255, 255, 255), (0, sh - 120, 120, 120))
            pygame.display.flip()

            time.sleep(0.4)

            # Try to detect DocScanner quad across several frames
            auto_detected = None
            for _ in range(6):
                f = camera.capture_frame()
                q = detect_docscanner_quad(f)
                if q is not None:
                    auto_detected = q
                    break
                time.sleep(0.05)

            if auto_detected is not None:
                pins = [tuple(p) for p in auto_detected]
                print(f"[Calibration] Auto DocScanner successfully detected display edges!")
            else:
                print("[Calibration] Auto DocScanner could not distinguish borders; using standard bounds for manual adjustment.")

        elif calib_mode == "manual":
            print("[Calibration] Starting Manual Mouse Corner Pinning...")
            manual_click_mode = True
            manual_step = 0
            pins = [
                (float(CAMERA_WIDTH * 0.1), float(CAMERA_HEIGHT * 0.1)),
                (float(CAMERA_WIDTH * 0.9), float(CAMERA_HEIGHT * 0.1)),
                (float(CAMERA_WIDTH * 0.9), float(CAMERA_HEIGHT * 0.9)),
                (float(CAMERA_WIDTH * 0.1), float(CAMERA_HEIGHT * 0.9))
            ]

        # =============================================================
        # STEP 3: Interactive Studio (Draggable Pins, Live Video, Fine-Tuning)
        # =============================================================
        # Responsive dimensions for camera feed and diagnostics panel
        if sh >= 900:
            cam_w = 640
            cam_h = 480
            gap = 35
            card_w = min(740, sw - cam_w - gap - 60)
            card_h = 480
            btn_h = 68
            content_y = 90
        else:
            cam_w = min(560, (sw - 80) // 2)
            cam_h = int(cam_w * 0.75)
            gap = 25
            card_w = min(660, sw - cam_w - gap - 40)
            card_h = cam_h
            btn_h = 56
            content_y = 65

        total_content_w = cam_w + gap + card_w
        cam_x = max(20, (sw - total_content_w) // 2)
        card_rect = pygame.Rect(cam_x + cam_w + gap, content_y, card_w, card_h)

        # 4 Action buttons at the bottom
        btn_y = content_y + cam_h + (22 if sh >= 900 else 14)
        btn1_w = 300  # Accept
        btn2_w = 260  # Auto Scan
        btn3_w = 260  # Manual Click
        btn4_w = 230  # Default Bounds
        btn_gap = 16
        total_btn_w = btn1_w + btn2_w + btn3_w + btn4_w + 3 * btn_gap
        btn_start_x = max(15, (sw - total_btn_w) // 2)

        btn_accept_rect = pygame.Rect(btn_start_x, btn_y, btn1_w, btn_h)
        btn_auto_rect = pygame.Rect(btn_start_x + btn1_w + btn_gap, btn_y, btn2_w, btn_h)
        btn_manual_rect = pygame.Rect(btn_start_x + btn1_w + btn_gap + btn2_w + btn_gap, btn_y, btn3_w, btn_h)
        btn_default_rect = pygame.Rect(btn_start_x + btn1_w + btn_gap + btn2_w + btn_gap + btn3_w + btn_gap, btn_y, btn4_w, btn_h)
        footer_y = btn_y + btn_h + 10

        dragging_pin_idx = None
        studio_active = True

        print("[Calibration] Entering Interactive Visual Studio (Draggable pins + Live Cam)...")

        while studio_active:
            mouse_pos = pygame.mouse.get_pos()
            mx, my = mouse_pos

            # Check if mouse is hovering over any of the 4 corner pins
            hover_pin_idx = None
            if not manual_click_mode:
                for i in range(4):
                    pin_scr_x = cam_x + int(pins[i][0] / CAMERA_WIDTH * cam_w)
                    pin_scr_y = content_y + int(pins[i][1] / CAMERA_HEIGHT * cam_h)
                    if np.hypot(mx - pin_scr_x, my - pin_scr_y) <= 20:
                        hover_pin_idx = i
                        break

            hover_btn_accept = btn_accept_rect.collidepoint(mouse_pos)
            hover_btn_auto = btn_auto_rect.collidepoint(mouse_pos)
            hover_btn_manual = btn_manual_rect.collidepoint(mouse_pos)
            hover_btn_default = btn_default_rect.collidepoint(mouse_pos)

            # ---------------------------------------------------------
            # Process Pygame Events & Mouse Dragging
            # ---------------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt("Calibration aborted by user.")
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_y):
                        # Accept and proceed
                        src_pts = np.array(pins, dtype=np.float32)
                        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
                        if homography_matrix is None:
                            homography_matrix = np.eye(3, dtype=np.float64)
                        save_calibration(homography_matrix, error_px=0.0, screen_resolution=(sw, sh), points=target_coords)
                        return homography_matrix

                    elif event.key == pygame.K_a:
                        # Re-run Auto DocScanner
                        print("[Calibration] Auto DocScanner re-scan requested...")
                        f = camera.capture_frame()
                        q = detect_docscanner_quad(f)
                        if q is not None:
                            pins = [tuple(p) for p in q]
                            print("[Calibration] DocScanner successfully re-aligned to display edges!")
                        else:
                            print("[Calibration] Could not detect display automatically. Use mouse to drag corner pins.")

                    elif event.key in (pygame.K_m, pygame.K_r):
                        # Switch to manual sequential clicking
                        manual_click_mode = True
                        manual_step = 0
                        print("[Calibration] Switched to Manual Click mode. Click the 4 corners in sequence.")

                    elif event.key == pygame.K_SPACE:
                        # Reset to 1:1 default bounds
                        pins = [
                            (25.0, 25.0),
                            (float(CAMERA_WIDTH - 25), 25.0),
                            (float(CAMERA_WIDTH - 25), float(CAMERA_HEIGHT - 25)),
                            (25.0, float(CAMERA_HEIGHT - 25))
                        ]
                        manual_click_mode = False
                        manual_step = 4

                    elif event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt("Calibration aborted by user via ESC.")

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Check button clicks
                    if hover_btn_accept:
                        src_pts = np.array(pins, dtype=np.float32)
                        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
                        if homography_matrix is None:
                            homography_matrix = np.eye(3, dtype=np.float64)
                        save_calibration(homography_matrix, error_px=0.0, screen_resolution=(sw, sh), points=target_coords)
                        return homography_matrix

                    elif hover_btn_auto:
                        f = camera.capture_frame()
                        q = detect_docscanner_quad(f)
                        if q is not None:
                            pins = [tuple(p) for p in q]
                            manual_click_mode = False
                            manual_step = 4
                            print("[Calibration] Auto DocScanner locked onto edges!")
                        else:
                            print("[Calibration] Could not detect quad. Use mouse to drag pins.")

                    elif hover_btn_manual:
                        manual_click_mode = True
                        manual_step = 0
                        print("[Calibration] Click 4 display corners in sequence.")

                    elif hover_btn_default:
                        pins = [
                            (25.0, 25.0),
                            (float(CAMERA_WIDTH - 25), 25.0),
                            (float(CAMERA_WIDTH - 25), float(CAMERA_HEIGHT - 25)),
                            (25.0, float(CAMERA_HEIGHT - 25))
                        ]
                        manual_click_mode = False
                        manual_step = 4

                    # Check clicking inside camera feed
                    elif cam_x <= mx <= cam_x + cam_w and content_y <= my <= content_y + cam_h:
                        cam_px = max(0.0, min(float(CAMERA_WIDTH - 1), (mx - cam_x) / cam_w * CAMERA_WIDTH))
                        cam_py = max(0.0, min(float(CAMERA_HEIGHT - 1), (my - content_y) / cam_h * CAMERA_HEIGHT))

                        if manual_click_mode:
                            pins[manual_step] = (cam_px, cam_py)
                            manual_step += 1
                            c_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                            print(f"[Calibration] Placed {c_names[manual_step-1]} pin at: ({cam_px:.1f}, {cam_py:.1f})")
                            if manual_step >= 4:
                                manual_click_mode = False
                                print("[Calibration] All 4 corners placed! You can now drag any pin to fine-tune.")
                        elif hover_pin_idx is not None:
                            dragging_pin_idx = hover_pin_idx

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging_pin_idx = None

                elif event.type == pygame.MOUSEMOTION:
                    if dragging_pin_idx is not None:
                        # Clamp dragging position within camera frame
                        new_cx = max(0.0, min(float(CAMERA_WIDTH - 1), (mx - cam_x) / cam_w * CAMERA_WIDTH))
                        new_cy = max(0.0, min(float(CAMERA_HEIGHT - 1), (my - content_y) / cam_h * CAMERA_HEIGHT))
                        pins[dragging_pin_idx] = (new_cx, new_cy)

            # ---------------------------------------------------------
            # Compute Live Homography & Edge Measurements
            # ---------------------------------------------------------
            src_pts = np.array(pins, dtype=np.float32)
            disp_corners = src_pts.astype(np.int32)
            is_convex = bool(cv2.isContourConvex(disp_corners.reshape(-1, 1, 2)))

            w_top = float(np.hypot(pins[1][0] - pins[0][0], pins[1][1] - pins[0][1]))
            w_btm = float(np.hypot(pins[2][0] - pins[3][0], pins[2][1] - pins[3][1]))
            h_left = float(np.hypot(pins[3][0] - pins[0][0], pins[3][1] - pins[0][1]))
            h_right = float(np.hypot(pins[2][0] - pins[1][0], pins[2][1] - pins[1][1]))
            avg_w = (w_top + w_btm) / 2.0
            avg_h = (h_left + h_right) / 2.0
            aspect = avg_w / max(avg_h, 1.0)

            # ---------------------------------------------------------
            # Capture Live Camera Frame & Render Overlay
            # ---------------------------------------------------------
            try:
                live_frame = camera.capture_frame()
            except Exception:
                live_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

            annotated = live_frame.copy()

            # Semi-transparent polygon overlay for the display zone
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [disp_corners], color=(20, 60, 40))
            cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0, annotated)

            # Draw 4 DocScanner edges
            cv2.polylines(annotated, [disp_corners], isClosed=True, color=(0, 245, 255), thickness=3)

            # Diagonals & Center
            cv2.line(annotated, tuple(disp_corners[0]), tuple(disp_corners[2]), (80, 100, 120), 1)
            cv2.line(annotated, tuple(disp_corners[1]), tuple(disp_corners[3]), (80, 100, 120), 1)
            center_x = int((disp_corners[0][0] + disp_corners[2][0]) // 2)
            center_y = int((disp_corners[0][1] + disp_corners[2][1]) // 2)
            cv2.drawMarker(annotated, (center_x, center_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

            # Draw 4 Draggable Corner Pins with badges
            pin_info = [
                ("TL", (60, 255, 100)),   # Green
                ("TR", (255, 220, 40)),   # Yellow
                ("BR", (255, 140, 40)),   # Orange
                ("BL", (0, 200, 255))     # Cyan
            ]
            for i, (tag, col) in enumerate(pin_info):
                px, py = int(disp_corners[i][0]), int(disp_corners[i][1])
                is_pin_dragged = (dragging_pin_idx == i)
                is_pin_hovered = (hover_pin_idx == i)

                rad = 14 if (is_pin_dragged or is_pin_hovered) else 10
                cv2.circle(annotated, (px, py), rad, col, -1)
                cv2.circle(annotated, (px, py), rad + 4, (255, 255, 255), 2)
                if is_pin_dragged or is_pin_hovered:
                    cv2.circle(annotated, (px, py), rad + 8, col, 2)

                off_x = 12 if i in (0, 3) else -85
                off_y = -12 if i in (0, 1) else 24
                cv2.putText(annotated, f"{tag}: ({px},{py})", (px + off_x, py + off_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 3)
                cv2.putText(annotated, f"{tag}: ({px},{py})", (px + off_x, py + off_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

            # Edge length text
            top_mid = ((disp_corners[0][0] + disp_corners[1][0]) // 2, (disp_corners[0][1] + disp_corners[1][1]) // 2)
            cv2.putText(annotated, f"Top: {w_top:.0f}px", (top_mid[0] - 35, top_mid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 255), 1)

            btm_mid = ((disp_corners[3][0] + disp_corners[2][0]) // 2, (disp_corners[3][1] + disp_corners[2][1]) // 2)
            cv2.putText(annotated, f"Btm: {w_btm:.0f}px", (btm_mid[0] - 35, btm_mid[1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 255), 1)

            left_mid = ((disp_corners[0][0] + disp_corners[3][0]) // 2, (disp_corners[0][1] + disp_corners[3][1]) // 2)
            cv2.putText(annotated, f"L: {h_left:.0f}px", (left_mid[0] - 55, left_mid[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 255), 1)

            right_mid = ((disp_corners[1][0] + disp_corners[2][0]) // 2, (disp_corners[1][1] + disp_corners[2][1]) // 2)
            cv2.putText(annotated, f"R: {h_right:.0f}px", (right_mid[0] + 8, right_mid[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 255), 1)

            # ---------------------------------------------------------
            # Render Pygame UI & Diagnostics
            # ---------------------------------------------------------
            screen.fill((14, 17, 24))

            # Header & Status
            title_surf = font_title.render("DISPLAY CALIBRATION STUDIO — DOCSCANNER & MANUAL EDGES", True, (40, 240, 180))
            if manual_click_mode:
                c_prompts = [
                    "Step 1/4: Click on the TOP-LEFT corner of your physical display in the camera feed",
                    "Step 2/4: Click on the TOP-RIGHT corner of your physical display in the camera feed",
                    "Step 3/4: Click on the BOTTOM-RIGHT corner of your physical display in the camera feed",
                    "Step 4/4: Click on the BOTTOM-LEFT corner of your physical display in the camera feed"
                ]
                sub_surf = font_sub.render(c_prompts[min(manual_step, 3)], True, (255, 220, 80))
            elif dragging_pin_idx is not None:
                tag_name = pin_info[dragging_pin_idx][0]
                sub_surf = font_sub.render(f"DRAGGING [{tag_name}] CORNER: Move mouse to align pin with monitor edge", True, (60, 255, 120))
            else:
                sub_surf = font_sub.render(
                    "Review edges below. Click and drag any corner pin (TL, TR, BR, BL) with mouse to fine-tune alignment.",
                    True, (200, 215, 235)
                )

            screen.blit(title_surf, (sw // 2 - title_surf.get_width() // 2, 18 if sh >= 900 else 10))
            screen.blit(sub_surf, (sw // 2 - sub_surf.get_width() // 2, 54 if sh >= 900 else 38))

            # Camera View (with interactive pins)
            frame_resized = cv2.resize(annotated, (cam_w, cam_h))
            frame_surf = pygame.image.frombuffer(frame_resized.tobytes(), (cam_w, cam_h), 'RGB')
            pygame.draw.rect(screen, (0, 240, 255), (cam_x - 2, content_y - 2, cam_w + 4, cam_h + 4), 2, border_radius=6)
            screen.blit(frame_surf, (cam_x, content_y))

            cam_badge = "MANUAL CLICK MODE: CLICK CORNER" if manual_click_mode else "DRAGGABLE PINS ACTIVE: DRAG WITH MOUSE"
            cam_badge_surf = font_bold.render(cam_badge, True, (255, 220, 80) if manual_click_mode else (0, 240, 255))
            screen.blit(cam_badge_surf, (cam_x + 10, content_y + 10))

            # Diagnostics Card on Right
            pygame.draw.rect(screen, (20, 24, 34), card_rect, border_radius=8)
            pygame.draw.rect(screen, (50, 65, 90), card_rect, 2, border_radius=8)

            card_hdr = font_heading.render("DISPLAY CORNER TELEMETRY", True, (255, 220, 80))
            screen.blit(card_hdr, (card_rect.x + 20, card_rect.y + 14))

            # Status Badge
            if is_convex and min(w_top, w_btm, h_left, h_right) > 80:
                badge_text = "STATUS: VALID DISPLAY QUAD (Accurate & Ready)"
                badge_col = (60, 255, 100)
            else:
                badge_text = "STATUS: DISTORTED / SKEWED (Adjust corner pins)"
                badge_col = (255, 120, 50)

            badge_surf = font_bold.render(badge_text, True, badge_col)
            screen.blit(badge_surf, (card_rect.x + 20, card_rect.y + 44))

            # Corner Table
            line_y = card_rect.y + 76
            t_lines = [
                f"Top-Left     [TL] : Camera ({pins[0][0]:.1f}, {pins[0][1]:.1f})  ->  Display ({target_coords[0][0]}, {target_coords[0][1]})",
                f"Top-Right    [TR] : Camera ({pins[1][0]:.1f}, {pins[1][1]:.1f})  ->  Display ({target_coords[1][0]}, {target_coords[1][1]})",
                f"Bottom-Right [BR] : Camera ({pins[2][0]:.1f}, {pins[2][1]:.1f})  ->  Display ({target_coords[2][0]}, {target_coords[2][1]})",
                f"Bottom-Left  [BL] : Camera ({pins[3][0]:.1f}, {pins[3][1]:.1f})  ->  Display ({target_coords[3][0]}, {target_coords[3][1]})",
            ]
            for tl in t_lines:
                ts = font_mono.render(tl, True, (210, 225, 245))
                screen.blit(ts, (card_rect.x + 20, line_y))
                line_y += 22

            line_y += 6
            dim_lines = [
                f"Edge Widths  : Top: {w_top:.0f} px  |  Bottom: {w_btm:.0f} px",
                f"Edge Heights : Left: {h_left:.0f} px  |  Right: {h_right:.0f} px",
                f"Aspect Ratio : {aspect:.2f} : 1.0 (Display Native: {sw/sh:.2f} : 1.0)"
            ]
            for dl in dim_lines:
                ds = font_mono.render(dl, True, (170, 195, 225))
                screen.blit(ds, (card_rect.x + 20, line_y))
                line_y += 20

            line_y += 10
            g_title = font_bold.render("ALIGNMENT INSTRUCTIONS:", True, (255, 200, 60))
            screen.blit(g_title, (card_rect.x + 20, line_y))
            line_y += 22

            instructions = [
                "• Click and drag any corner pin (TL, TR, BR, BL) to match your screen edges.",
                "• Click [Auto DocScanner Scan] to re-detect display edges automatically.",
                "• Click [Manual Re-Click] to re-place the 4 corners in sequence.",
                "• When edges frame your display, click [Accept & Continue] or press [ENTER]."
            ]
            for g in instructions:
                is_primary = "Accept" in g or "drag" in g
                gs = font_regular.render(g, True, (225, 235, 250) if is_primary else (175, 185, 200))
                screen.blit(gs, (card_rect.x + 20, line_y))
                line_y += 20

            # ---------------------------------------------------------
            # Render Action Buttons with Hover Effects
            # ---------------------------------------------------------
            # Button 1: Accept & Continue
            pygame.draw.rect(screen, (35, 135, 65) if hover_btn_accept else (20, 85, 40), btn_accept_rect, border_radius=8)
            pygame.draw.rect(screen, (100, 255, 140) if hover_btn_accept else (50, 200, 90), btn_accept_rect, 2, border_radius=8)
            m1 = font_btn_title.render("[ENTER] Accept & Continue", True, (255, 255, 255))
            s1 = font_btn_sub.render("Edges aligned - Launch Whiteboard", True, (200, 255, 220))
            screen.blit(m1, (btn_accept_rect.centerx - m1.get_width() // 2, btn_accept_rect.y + (12 if sh >= 900 else 8)))
            screen.blit(s1, (btn_accept_rect.centerx - s1.get_width() // 2, btn_accept_rect.y + (36 if sh >= 900 else 30)))

            # Button 2: Auto DocScanner Scan
            pygame.draw.rect(screen, (15, 80, 105) if hover_btn_auto else (10, 48, 65), btn_auto_rect, border_radius=8)
            pygame.draw.rect(screen, (0, 220, 255) if hover_btn_auto else (0, 160, 195), btn_auto_rect, 2, border_radius=8)
            m2 = font_btn_title.render("[A] Auto DocScanner Scan", True, (255, 255, 255))
            s2 = font_btn_sub.render("Scan camera for screen edges", True, (190, 240, 255))
            screen.blit(m2, (btn_auto_rect.centerx - m2.get_width() // 2, btn_auto_rect.y + (12 if sh >= 900 else 8)))
            screen.blit(s2, (btn_auto_rect.centerx - s2.get_width() // 2, btn_auto_rect.y + (36 if sh >= 900 else 30)))

            # Button 3: Manual Re-Click Corners
            pygame.draw.rect(screen, (120, 65, 20) if hover_btn_manual else (75, 40, 15), btn_manual_rect, border_radius=8)
            pygame.draw.rect(screen, (255, 170, 50) if hover_btn_manual else (205, 125, 35), btn_manual_rect, 2, border_radius=8)
            m3 = font_btn_title.render("[M] Manual Re-Click", True, (255, 255, 255))
            s3 = font_btn_sub.render("Click 4 corners in sequence", True, (255, 225, 195))
            screen.blit(m3, (btn_manual_rect.centerx - m3.get_width() // 2, btn_manual_rect.y + (12 if sh >= 900 else 8)))
            screen.blit(s3, (btn_manual_rect.centerx - s3.get_width() // 2, btn_manual_rect.y + (36 if sh >= 900 else 30)))

            # Button 4: Default 1:1 Bounds
            pygame.draw.rect(screen, (45, 65, 95) if hover_btn_default else (30, 45, 70), btn_default_rect, border_radius=8)
            pygame.draw.rect(screen, (130, 180, 240) if hover_btn_default else (80, 115, 170), btn_default_rect, 2, border_radius=8)
            m4 = font_btn_title.render("[SPACE] Default Bounds", True, (255, 255, 255))
            s4 = font_btn_sub.render("Reset to 1:1 full camera", True, (180, 210, 245))
            screen.blit(m4, (btn_default_rect.centerx - m4.get_width() // 2, btn_default_rect.y + (12 if sh >= 900 else 8)))
            screen.blit(s4, (btn_default_rect.centerx - s4.get_width() // 2, btn_default_rect.y + (36 if sh >= 900 else 30)))

            # Footer
            footer_text = font_regular.render(
                "Controls: Drag pins with mouse  |  [ENTER] Accept  |  [A] Auto Scan  |  [M] Manual Click  |  [SPACE] Default  |  [ESC] Exit",
                True, (140, 155, 175)
            )
            screen.blit(footer_text, (sw // 2 - footer_text.get_width() // 2, footer_y))

            pygame.display.flip()
            clock.tick(30)

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
