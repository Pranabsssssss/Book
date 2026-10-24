import os
import time
from datetime import datetime
import cv2

try:
    import pygame
except ImportError:
    pygame = None

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    PEN_COLOR, PEN_RADIUS, ERASER_RADIUS,
    PEN_LIFT_TIMEOUT, MAX_STROKE_JUMP,
    ERASER_BUTTON_RECT, SAVE_BUTTON_RECT,
    BUTTON_DWELL_FRAMES, SAVE_DIR,
    DEV_CAMERA_WIDTH, DEV_CAMERA_HEIGHT
)


class Whiteboard:
    """
    Smart Notebook Whiteboard System.
    Provides two distinct operating interfaces:
      1. Production Mode: Clean fullscreen interactive whiteboard with IR pen drawing.
      2. Development / Testing Mode: Dedicated OpenCV test bench showing large live
         camera feed, threshold mask, lighting telemetry, and test drawing pad.
    """
    def __init__(self, screen=None, dev_mode=False):
        if pygame is None:
            raise RuntimeError(
                "Pygame is required for the Whiteboard interface. "
                "Please install dependencies with: pip install -r requirements.txt"
            )

        if not pygame.get_init():
            pygame.init()

        if screen is None:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        else:
            self.screen = screen

        pygame.display.set_caption("Smart Notebook")
        self.clock = pygame.time.Clock()

        # Canvas surface holds the production artwork (saved on export)
        self.canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.canvas.fill((0, 0, 0))

        # Test pad surface for Development Mode (dedicated drawing area)
        self.dev_test_pad_w = DEV_CAMERA_WIDTH
        self.dev_test_pad_h = 280
        self.dev_test_pad = pygame.Surface((self.dev_test_pad_w, self.dev_test_pad_h))
        self.dev_test_pad.fill((0, 0, 0))

        # Runtime states
        self.eraser_mode = False
        self.prev_point = None
        self.dev_prev_point = None
        self.last_point_time = 0.0
        self.quit_requested = False

        # Operating Mode (Production vs Testing / Development)
        self.dev_mode = dev_mode
        self.threshold_delta = 0
        self.current_threshold = 240
        self.current_screen_point = None

        # Dev Overlay Surfaces
        self.dev_frame_surf = None
        self.dev_thresh_surf = None
        self.dev_info = {}

        # Dwell & Tap / Blink tracking
        self.eraser_dwell = 0
        self.save_dwell = 0
        self.last_button_trigger_time = 0.0

        # Notifications
        mode_label = "TESTING / DEV BENCH" if self.dev_mode else "PRODUCTION WHITEBOARD"
        self.notification = (mode_label, time.time() + 2.5)

        # Fonts
        self.font_title = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_btn = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_status = pygame.font.SysFont("Arial", 18)
        self.font_hud_title = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_hud_val = pygame.font.SysFont("Consolas", 15)

    def set_current_threshold(self, thresh: int):
        self.current_threshold = thresh

    def update_dev_overlay(self, annotated_frame=None, thresh_mask=None, info=None, screen_point=None):
        """
        Receive live OpenCV diagnostic imagery and metadata for development rendering.
        """
        if not self.dev_mode:
            return

        self.current_screen_point = screen_point
        self.dev_info = info or {}

        if annotated_frame is not None:
            resized_cam = cv2.resize(annotated_frame, (DEV_CAMERA_WIDTH, DEV_CAMERA_HEIGHT))
            self.dev_frame_surf = pygame.image.frombuffer(
                resized_cam.tobytes(), (DEV_CAMERA_WIDTH, DEV_CAMERA_HEIGHT), 'RGB'
            )

        if thresh_mask is not None:
            resized_thresh = cv2.resize(thresh_mask, (DEV_CAMERA_WIDTH, DEV_CAMERA_HEIGHT))
            self.dev_thresh_surf = pygame.image.frombuffer(
                resized_thresh.tobytes(), (DEV_CAMERA_WIDTH, DEV_CAMERA_HEIGHT), 'RGB'
            )

    def handle_point(self, screen_point: tuple):
        """
        Process a detected pen location in screen coordinates.
        Determines whether point interacts with UI controls or paints on canvas.
        Includes anti-spike distance checking and button blink/tap detection.
        """
        now = time.time()
        px = max(0, min(SCREEN_WIDTH - 1, int(screen_point[0])))
        py = max(0, min(SCREEN_HEIGHT - 1, int(screen_point[1])))
        pt = (px, py)
        self.current_screen_point = pt

        # -------------------------------------------------------------
        # 1. Check Button Zones (ERASER & SAVE)
        # When light hits or blinks on a button: DO NOT draw on canvas!
        # -------------------------------------------------------------
        if self._point_in_rect(pt, ERASER_BUTTON_RECT):
            self.eraser_dwell += 1
            self.save_dwell = 0
            self.prev_point = None  # Never draw lines through buttons!

            if self.eraser_dwell >= BUTTON_DWELL_FRAMES:
                if now - self.last_button_trigger_time > 0.6:
                    self.eraser_mode = not self.eraser_mode
                    status_str = "Eraser Mode: ON" if self.eraser_mode else "Pen Mode: ON"
                    self.notification = (status_str, now + 1.8)
                    self.last_button_trigger_time = now
                    print(f"[Whiteboard] {status_str}")
                self.eraser_dwell = 0
            return

        if self._point_in_rect(pt, SAVE_BUTTON_RECT):
            self.save_dwell += 1
            self.eraser_dwell = 0
            self.prev_point = None

            if self.save_dwell >= BUTTON_DWELL_FRAMES:
                if now - self.last_button_trigger_time > 0.8:
                    self.save_canvas()
                    self.last_button_trigger_time = now
                self.save_dwell = 0
            return

        # Reset button dwells when light points away from buttons
        self.eraser_dwell = max(0, self.eraser_dwell - 1)
        self.save_dwell = max(0, self.save_dwell - 1)

        # -------------------------------------------------------------
        # 2. Canvas Drawing / Erasing
        # -------------------------------------------------------------
        # Anti-spike / Anti-jump filter: if the coordinate jumped wildly (> MAX_STROKE_JUMP px)
        # in a single frame, start a new stroke rather than drawing a huge diagonal line!
        if self.prev_point is not None:
            dx = pt[0] - self.prev_point[0]
            dy = pt[1] - self.prev_point[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > MAX_STROKE_JUMP:
                self.prev_point = None

        if self.eraser_mode:
            # Erasing with solid black
            if self.prev_point is not None:
                self._draw_interpolated_stroke(self.canvas, (0, 0, 0), self.prev_point, pt, ERASER_RADIUS)
            else:
                pygame.draw.circle(self.canvas, (0, 0, 0), pt, ERASER_RADIUS)
        else:
            # Drawing with white pen
            if self.prev_point is not None:
                self._draw_interpolated_stroke(self.canvas, PEN_COLOR, self.prev_point, pt, PEN_RADIUS)
            else:
                pygame.draw.circle(self.canvas, PEN_COLOR, pt, PEN_RADIUS)

        self.prev_point = pt
        self.last_point_time = now

        # Also update Dev Mode test pad if in dev mode
        if self.dev_mode:
            self._handle_dev_test_pad_point(pt)

    def _handle_dev_test_pad_point(self, pt):
        """Draw on the mini test pad inside Development Mode."""
        # Map screen coordinate into test pad relative coordinates
        t_x = int(pt[0] * (self.dev_test_pad_w / SCREEN_WIDTH))
        t_y = int(pt[1] * (self.dev_test_pad_h / SCREEN_HEIGHT))
        pad_pt = (t_x, t_y)

        if self.dev_prev_point is not None:
            dx = pad_pt[0] - self.dev_prev_point[0]
            dy = pad_pt[1] - self.dev_prev_point[1]
            if (dx * dx + dy * dy) ** 0.5 > 60:
                self.dev_prev_point = None

        color = (0, 0, 0) if self.eraser_mode else PEN_COLOR
        radius = max(2, PEN_RADIUS // 2)

        if self.dev_prev_point is not None:
            self._draw_interpolated_stroke(self.dev_test_pad, color, self.dev_prev_point, pad_pt, radius)
        else:
            pygame.draw.circle(self.dev_test_pad, color, pad_pt, radius)

        self.dev_prev_point = pad_pt

    def handle_no_point(self):
        """
        Invoked when no IR light is detected in the current frame (pen button released).
        Immediately stops continuous drawing stroke.
        """
        self.current_screen_point = None
        # Pen lift: immediately break stroke continuity
        if time.time() - self.last_point_time > PEN_LIFT_TIMEOUT:
            self.prev_point = None
            self.dev_prev_point = None

        # Gentle decay on dwell
        self.eraser_dwell = max(0, self.eraser_dwell - 1)
        self.save_dwell = max(0, self.save_dwell - 1)

    def render(self):
        """
        Render canvas to display and overlay UI controls and feedback notifications.
        Renders either Production Whiteboard or Development Test Bench.
        """
        # Event processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
                elif event.key == pygame.K_e:
                    self.eraser_mode = not self.eraser_mode
                    msg = "Eraser Mode: ON" if self.eraser_mode else "Pen Mode: ON"
                    self.notification = (msg, time.time() + 1.5)
                elif event.key == pygame.K_s:
                    self.save_canvas()
                elif event.key == pygame.K_c:
                    self.clear_canvas()
                elif event.key in (pygame.K_TAB, pygame.K_d):
                    # Toggle between Production and Development Mode
                    self.dev_mode = not self.dev_mode
                    label = "Mode: TESTING / DEV BENCH" if self.dev_mode else "Mode: PRODUCTION (Clean Whiteboard)"
                    self.notification = (label, time.time() + 2.0)
                    self.prev_point = None
                    self.dev_prev_point = None
                    print(f"[Whiteboard] Switched to {label}")
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    self.threshold_delta -= 2
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS):
                    self.threshold_delta += 2
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Mouse click support for PC testing
                self.handle_point(event.pos)
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                self.handle_point(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.handle_no_point()

        if self.dev_mode:
            # ---------------------------------------------------------
            # DEVELOPMENT / TESTING BENCH LAYOUT
            # Dedicated testing screen: Large camera, threshold mask, telemetry, test pad.
            # (Does NOT draw all over the fullscreen background!)
            # ---------------------------------------------------------
            self._render_dev_bench()
        else:
            # ---------------------------------------------------------
            # PRODUCTION MODE LAYOUT
            # Pure fullscreen black canvas with corner buttons
            # ---------------------------------------------------------
            self.screen.blit(self.canvas, (0, 0))
            self._render_buttons()
            self._render_status()

        pygame.display.flip()
        self.clock.tick(30)

    def _render_dev_bench(self):
        """Dedicated Development Mode Screen with large OpenCV feeds and telemetry."""
        # Solid dark studio background
        self.screen.fill((18, 22, 28))

        # Top Header Bar
        pygame.draw.rect(self.screen, (26, 32, 42), (0, 0, SCREEN_WIDTH, 56))
        pygame.draw.line(self.screen, (40, 50, 68), (0, 56), (SCREEN_WIDTH, 56), 2)

        title = self.font_title.render("SMART NOTEBOOK — OPENCV TESTING & LIGHTING BENCH", True, (240, 245, 255))
        self.screen.blit(title, (24, 14))

        badge_txt = self.font_btn.render("TESTING / DEV MODE", True, (60, 255, 100))
        badge_rect = pygame.Rect(SCREEN_WIDTH - 260, 12, 230, 32)
        pygame.draw.rect(self.screen, (20, 50, 30), badge_rect, border_radius=6)
        pygame.draw.rect(self.screen, (60, 220, 100), badge_rect, 2, border_radius=6)
        self.screen.blit(badge_txt, (SCREEN_WIDTH - 245, 16))

        # Layout coordinates
        margin_x = 40
        top_y = 80
        col_gap = 40
        col2_x = margin_x + DEV_CAMERA_WIDTH + col_gap
        row2_y = top_y + DEV_CAMERA_HEIGHT + 36

        # --- 1. Top Left: Live Camera Tracking Feed ---
        pygame.draw.rect(self.screen, (10, 12, 16), (margin_x - 4, top_y - 28, DEV_CAMERA_WIDTH + 8, DEV_CAMERA_HEIGHT + 36), border_radius=8)
        pygame.draw.rect(self.screen, (0, 180, 255), (margin_x - 4, top_y - 28, DEV_CAMERA_WIDTH + 8, DEV_CAMERA_HEIGHT + 36), 2, border_radius=8)
        lbl_cam = self.font_hud_title.render("LIVE CAMERA FEED (Tracking & Peak LED)", True, (0, 220, 255))
        self.screen.blit(lbl_cam, (margin_x + 8, top_y - 24))
        if self.dev_frame_surf:
            self.screen.blit(self.dev_frame_surf, (margin_x, top_y + 4))

        # --- 2. Top Right: OpenCV Threshold Mask ---
        pygame.draw.rect(self.screen, (10, 12, 16), (col2_x - 4, top_y - 28, DEV_CAMERA_WIDTH + 8, DEV_CAMERA_HEIGHT + 36), border_radius=8)
        pygame.draw.rect(self.screen, (255, 180, 0), (col2_x - 4, top_y - 28, DEV_CAMERA_WIDTH + 8, DEV_CAMERA_HEIGHT + 36), 2, border_radius=8)
        lbl_thresh = self.font_hud_title.render(f"OPENCV THRESHOLD MASK (Threshold: {self.current_threshold})", True, (255, 190, 20))
        self.screen.blit(lbl_thresh, (col2_x + 8, top_y - 24))
        if self.dev_thresh_surf:
            self.screen.blit(self.dev_thresh_surf, (col2_x, top_y + 4))

        # --- 3. Bottom Left: Telemetry & Controls Card ---
        tele_w = DEV_CAMERA_WIDTH
        tele_h = 280
        pygame.draw.rect(self.screen, (12, 15, 20), (margin_x - 4, row2_y - 28, tele_w + 8, tele_h + 36), border_radius=8)
        pygame.draw.rect(self.screen, (80, 220, 100), (margin_x - 4, row2_y - 28, tele_w + 8, tele_h + 36), 2, border_radius=8)
        lbl_tele = self.font_hud_title.render("DIAGNOSTICS & TELEMETRY", True, (100, 255, 120))
        self.screen.blit(lbl_tele, (margin_x + 8, row2_y - 24))

        fps = self.clock.get_fps()
        cam_pt = self.dev_info.get("point")
        cam_str = f"({cam_pt[0]:.1f}, {cam_pt[1]:.1f})" if cam_pt else "NONE (No light)"
        scr_str = f"({self.current_screen_point[0]}, {self.current_screen_point[1]})" if self.current_screen_point else "NONE (Off-screen)"
        area_val = self.dev_info.get("area", 0.0)
        peak_val = self.dev_info.get("peak_brightness", 0)
        status = self.dev_info.get("status", "SEARCHING")

        action_str = "PEN UP (LIFTED / OFF)"
        if self.current_screen_point:
            if self._point_in_rect(self.current_screen_point, ERASER_BUTTON_RECT):
                action_str = f"OVER ERASER BUTTON [Hits: {self.eraser_dwell}]"
            elif self._point_in_rect(self.current_screen_point, SAVE_BUTTON_RECT):
                action_str = f"OVER SAVE BUTTON [Hits: {self.save_dwell}]"
            elif self.eraser_mode:
                action_str = "ERASING"
            else:
                action_str = "DRAWING (PEN DOWN / ON)"

        tele_lines = [
            f"FPS              : {fps:.1f}",
            f"Threshold        : {self.current_threshold} (Press [ to lower, ] to raise)",
            f"Camera Point     : {cam_str}",
            f"Screen Point     : {scr_str}",
            f"Peak Brightness  : {peak_val} / 255 (Remote/IR target: 240-255)",
            f"Contour Area     : {area_val:.1f} px (Target LED: 3 - 150 px)",
            f"Detection Status : {status}",
            f"Pen Action State : {action_str}",
            f"Shortcuts        : [TAB] Switch to Whiteboard  [C] Clear Test Pad"
        ]

        text_y = row2_y + 8
        for line in tele_lines:
            color = (120, 255, 150) if "Pen Action" in line and "DRAWING" in line else (210, 215, 225)
            line_surf = self.font_hud_val.render(line, True, color)
            self.screen.blit(line_surf, (margin_x + 14, text_y))
            text_y += 26

        # --- 4. Bottom Right: Interactive Test Drawing Pad ---
        pygame.draw.rect(self.screen, (10, 12, 16), (col2_x - 4, row2_y - 28, tele_w + 8, tele_h + 36), border_radius=8)
        pygame.draw.rect(self.screen, (180, 100, 255), (col2_x - 4, row2_y - 28, tele_w + 8, tele_h + 36), 2, border_radius=8)
        lbl_pad = self.font_hud_title.render("INTERACTIVE TEST DRAWING PAD (Preview)", True, (200, 130, 255))
        self.screen.blit(lbl_pad, (col2_x + 8, row2_y - 24))
        self.screen.blit(self.dev_test_pad, (col2_x, row2_y + 4))

        # Bottom help text
        help_text = "[ESC] Quit   [TAB] Switch to Fullscreen Whiteboard   [C] Clear Test Pad   [[ / ]] Tune IR Threshold"
        help_surf = self.font_status.render(help_text, True, (130, 140, 160))
        self.screen.blit(help_surf, (40, SCREEN_HEIGHT - 35))

    def save_canvas(self):
        """Save the current drawing to a timestamped PNG file."""
        os.makedirs(SAVE_DIR, exist_ok=True)
        filename = f"whiteboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(SAVE_DIR, filename)

        try:
            pygame.image.save(self.canvas, filepath)
            self.notification = (f"Saved: {filename}", time.time() + 2.5)
            print(f"[Whiteboard] Saved drawing to: {filepath}")
            # Visual green flash on screen border
            pygame.draw.rect(self.screen, (60, 255, 60), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 8)
            pygame.display.flip()
            pygame.time.wait(150)
        except Exception as e:
            self.notification = (f"Save Failed: {e}", time.time() + 3.0)
            print(f"[Whiteboard] Error saving file: {e}")

    def clear_canvas(self):
        """Clear the drawing canvas and test pad."""
        self.canvas.fill((0, 0, 0))
        self.dev_test_pad.fill((0, 0, 0))
        self.notification = ("Canvas Cleared", time.time() + 1.5)
        print("[Whiteboard] Canvas & Test Pad cleared.")

    def should_quit(self) -> bool:
        return self.quit_requested

    def cleanup(self):
        pygame.quit()

    def _render_buttons(self):
        """Render on-screen button controls with active states and dwell progress bars."""
        # --- ERASER BUTTON ---
        ex, ey, ew, eh = ERASER_BUTTON_RECT
        if self.eraser_mode:
            bg_color = (180, 40, 40)
            border_color = (255, 80, 80)
            text_color = (255, 255, 255)
            label = "ERASER: ON"
        else:
            bg_color = (35, 35, 35)
            border_color = (100, 100, 100)
            text_color = (180, 180, 180)
            label = "ERASER"

        pygame.draw.rect(self.screen, bg_color, ERASER_BUTTON_RECT, border_radius=6)
        pygame.draw.rect(self.screen, border_color, ERASER_BUTTON_RECT, 2, border_radius=6)
        lbl_surf = self.font_btn.render(label, True, text_color)
        self.screen.blit(lbl_surf, (ex + (ew - lbl_surf.get_width()) // 2, ey + (eh - lbl_surf.get_height()) // 2))

        # Dwell progress bar
        if self.eraser_dwell > 0:
            fill_w = int(ew * min(1.0, self.eraser_dwell / BUTTON_DWELL_FRAMES))
            pygame.draw.rect(self.screen, (255, 120, 120), (ex, ey + eh - 6, fill_w, 6), border_radius=3)

        # --- SAVE BUTTON ---
        sx, sy, sw, sh = SAVE_BUTTON_RECT
        pygame.draw.rect(self.screen, (35, 35, 35), SAVE_BUTTON_RECT, border_radius=6)
        pygame.draw.rect(self.screen, (80, 200, 80), SAVE_BUTTON_RECT, 2, border_radius=6)
        save_surf = self.font_btn.render("SAVE", True, (100, 240, 100))
        self.screen.blit(save_surf, (sx + (sw - save_surf.get_width()) // 2, sy + (sh - save_surf.get_height()) // 2))

        # Dwell progress bar
        if self.save_dwell > 0:
            fill_w = int(sw * min(1.0, self.save_dwell / BUTTON_DWELL_FRAMES))
            pygame.draw.rect(self.screen, (100, 255, 100), (sx, sy + sh - 6, fill_w, 6), border_radius=3)

    def _render_status(self):
        """Render temporary notifications and keyboard hints."""
        now = time.time()
        msg, expiry = self.notification
        if now < expiry:
            txt_surf = self.font_status.render(msg, True, (240, 240, 240))
            bg_rect = pygame.Rect(20, SCREEN_HEIGHT - 75, txt_surf.get_width() + 20, txt_surf.get_height() + 10)
            pygame.draw.rect(self.screen, (40, 40, 40), bg_rect, border_radius=4)
            pygame.draw.rect(self.screen, (90, 90, 90), bg_rect, 1, border_radius=4)
            self.screen.blit(txt_surf, (30, SCREEN_HEIGHT - 70))

        # Bottom help text in production
        help_text = "[ESC] Quit   [TAB] Testing Bench   [E] Eraser   [S] Save   [C] Clear   [[ / ]] Tune Thresh"
        help_surf = self.font_status.render(help_text, True, (100, 100, 100))
        self.screen.blit(help_surf, (20, SCREEN_HEIGHT - 35))

    @staticmethod
    def _draw_interpolated_stroke(surface, color, p1, p2, radius):
        """
        Draw a solid continuous stroke between two points by interpolating circles.
        Prevents gaps when pen moves rapidly between camera frames.
        """
        x1, y1 = p1
        x2, y2 = p2
        dist = int(max(abs(x2 - x1), abs(y2 - y1), 1))
        for step in range(dist + 1):
            cx = int(x1 + (x2 - x1) * step / dist)
            cy = int(y1 + (y2 - y1) * step / dist)
            pygame.draw.circle(surface, color, (cx, cy), radius)

    @staticmethod
    def _point_in_rect(point, rect):
        px, py = point
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh


if __name__ == "__main__":
    print("Testing Whiteboard module standalone (windowed mode for test)...")
    wb = Whiteboard()
    print("Whiteboard initialized successfully.")
    wb.cleanup()
