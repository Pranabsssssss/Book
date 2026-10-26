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
    BUTTON_DWELL_FRAMES, SAVE_DIR,
    DEV_CAMERA_WIDTH, DEV_CAMERA_HEIGHT,
    get_button_rects, get_display_resolution
)


class Whiteboard:
    """
    Smart Notebook Whiteboard System.
    Draws circular dots and strokes wherever light turns on inside the calibrated display area.
    Supports continuous drawing, pulsating/blinking remote lights, eraser toggle,
    button blink/tap detection, and a dedicated DocScanner-style testing bench.
    Dynamically adapts to any display resolution (1366x768, 1280x720, 1080p, 2K/4K).
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
            # Auto-detect native resolution of current connected monitor
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = screen

        self.width = self.screen.get_width()
        self.height = self.screen.get_height()
        self.aspect_ratio = self.width / max(1, self.height)

        pygame.display.set_caption(f"Smart Notebook ({self.width}x{self.height})")
        self.clock = pygame.time.Clock()

        # Canvas surface holds the production artwork at maximum native display resolution!
        self.canvas = pygame.Surface((self.width, self.height))
        self.canvas.fill((0, 0, 0))

        # Dynamic button zones for ERASER and SAVE anchored to bottom right of this screen
        self.eraser_rect, self.save_rect = get_button_rects(self.width, self.height)

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
        Process a detected light location in screen coordinates.
        Draws a dot wherever the light turns on (regardless of blinking or pulsing).
        Also checks button hits (Save/Eraser).
        """
        now = time.time()
        px = max(0, min(self.width - 1, int(screen_point[0])))
        py = max(0, min(self.height - 1, int(screen_point[1])))
        pt = (px, py)
        self.current_screen_point = pt

        # -------------------------------------------------------------
        # 1. Check Button Zones (ERASER & SAVE)
        # When light hits or blinks on a button: trigger action and DO NOT draw!
        # -------------------------------------------------------------
        if self._point_in_rect(pt, self.eraser_rect):
            self.eraser_dwell += 1
            self.save_dwell = 0
            self.prev_point = None

            if self.eraser_dwell >= BUTTON_DWELL_FRAMES:
                if now - self.last_button_trigger_time > 0.6:
                    self.eraser_mode = not self.eraser_mode
                    status_str = "Eraser Mode: ON" if self.eraser_mode else "Pen Mode: ON"
                    self.notification = (status_str, now + 1.8)
                    self.last_button_trigger_time = now
                    print(f"[Whiteboard] {status_str}")
                self.eraser_dwell = 0
            return

        if self._point_in_rect(pt, self.save_rect):
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
        # 2. Draw or Erase at the Light Location
        # Whenever light turns on (even if blinking), stamp a circular dot!
        # -------------------------------------------------------------
        color = (0, 0, 0) if self.eraser_mode else PEN_COLOR
        radius = ERASER_RADIUS if self.eraser_mode else PEN_RADIUS

        # ALWAYS draw a solid circle dot at the exact location where light turned on / blinks
        pygame.draw.circle(self.canvas, color, pt, radius)

        # Smooth connection: only connect if hand is smoothly drawing (< 35px between frames)
        # If pen blinks or jumps across the board, do not draw a connecting line!
        if self.prev_point is not None:
            dx = pt[0] - self.prev_point[0]
            dy = pt[1] - self.prev_point[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= 35:
                self._draw_interpolated_stroke(self.canvas, color, self.prev_point, pt, radius)
            else:
                self.prev_point = None

        self.prev_point = pt
        self.last_point_time = now

        # Also update Dev Mode test pad if in dev mode
        if self.dev_mode:
            self._handle_dev_test_pad_point(pt, color, radius)

    def _handle_dev_test_pad_point(self, pt, color, radius):
        """Draw on the mini test pad inside Development Mode."""
        t_x = max(0, min(self.dev_test_pad_w - 1, int(pt[0] * (self.dev_test_pad_w / self.width))))
        t_y = max(0, min(self.dev_test_pad_h - 1, int(pt[1] * (self.dev_test_pad_h / self.height))))
        pad_pt = (t_x, t_y)
        pad_radius = max(4, radius)

        # Stamp visible dot on test pad
        pygame.draw.circle(self.dev_test_pad, color, pad_pt, pad_radius)

        if self.dev_prev_point is not None:
            dx = pad_pt[0] - self.dev_prev_point[0]
            dy = pad_pt[1] - self.dev_prev_point[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= 20:
                self._draw_interpolated_stroke(self.dev_test_pad, color, self.dev_prev_point, pad_pt, pad_radius)
            else:
                self.dev_prev_point = None

        self.dev_prev_point = pad_pt

    def handle_no_point(self):
        """
        Invoked when no IR light is detected in the current frame (light off / lifted).
        """
        self.current_screen_point = None
        if time.time() - self.last_point_time > PEN_LIFT_TIMEOUT:
            self.prev_point = None
            self.dev_prev_point = None

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
                self.handle_point(event.pos)
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                self.handle_point(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.handle_no_point()

        if self.dev_mode:
            # Development Mode: dedicated OpenCV testing studio
            self._render_dev_bench()
        else:
            # Production Mode: clean fullscreen black whiteboard
            self.screen.blit(self.canvas, (0, 0))
            self._render_buttons()
            self._render_status()

        pygame.display.flip()
        self.clock.tick(30)

    def _render_dev_bench(self):
        """Dedicated Development Mode Screen with large OpenCV feeds and DocScanner bounds."""
        self.screen.fill((18, 22, 28))

        # Top Header Bar across full native display width
        pygame.draw.rect(self.screen, (26, 32, 42), (0, 0, self.width, 56))
        pygame.draw.line(self.screen, (40, 50, 68), (0, 56), (self.width, 56), 2)

        title = self.font_title.render("SMART NOTEBOOK — OPENCV TESTING & LIGHTING BENCH", True, (240, 245, 255))
        self.screen.blit(title, (24, 14))

        badge_txt = self.font_btn.render("TESTING / DEV MODE", True, (60, 255, 100))
        badge_rect = pygame.Rect(self.width - 260, 12, 230, 32)
        pygame.draw.rect(self.screen, (20, 50, 30), badge_rect, border_radius=6)
        pygame.draw.rect(self.screen, (60, 220, 100), badge_rect, 2, border_radius=6)
        self.screen.blit(badge_txt, (self.width - 245, 16))

        # Dynamically scale feed dimensions for the current screen height & width
        if self.height >= 900:
            cam_w = DEV_CAMERA_WIDTH   # 540
            cam_h = DEV_CAMERA_HEIGHT  # 405
            top_y = 78
            tele_h = 280
            col_gap = 35
        else:
            cam_w = min(DEV_CAMERA_WIDTH, max(400, (self.width - 100) // 2))
            cam_h = int(cam_w * 0.75)
            top_y = 66
            tele_h = max(170, min(240, self.height - cam_h - top_y - 70))
            col_gap = 25

        margin_x = max(15, (self.width - (cam_w * 2 + col_gap)) // 2)
        col2_x = margin_x + cam_w + col_gap
        row2_y = top_y + cam_h + 30

        # --- 1. Top Left: Live Camera Tracking Feed with DocScanner Quad ---
        pygame.draw.rect(self.screen, (10, 12, 16), (margin_x - 4, top_y - 28, cam_w + 8, cam_h + 36), border_radius=8)
        pygame.draw.rect(self.screen, (0, 180, 255), (margin_x - 4, top_y - 28, cam_w + 8, cam_h + 36), 2, border_radius=8)
        lbl_cam = self.font_hud_title.render("LIVE CAMERA FEED (DocScanner Bounds & LED)", True, (0, 220, 255))
        self.screen.blit(lbl_cam, (margin_x + 8, top_y - 24))
        if self.dev_frame_surf:
            if self.dev_frame_surf.get_width() != cam_w or self.dev_frame_surf.get_height() != cam_h:
                scaled_cam = pygame.transform.smoothscale(self.dev_frame_surf, (cam_w, cam_h))
                self.screen.blit(scaled_cam, (margin_x, top_y + 4))
            else:
                self.screen.blit(self.dev_frame_surf, (margin_x, top_y + 4))

        # --- 2. Top Right: OpenCV Threshold Mask ---
        pygame.draw.rect(self.screen, (10, 12, 16), (col2_x - 4, top_y - 28, cam_w + 8, cam_h + 36), border_radius=8)
        pygame.draw.rect(self.screen, (255, 180, 0), (col2_x - 4, top_y - 28, cam_w + 8, cam_h + 36), 2, border_radius=8)
        lbl_thresh = self.font_hud_title.render(f"OPENCV THRESHOLD MASK (Threshold: {self.current_threshold})", True, (255, 190, 20))
        self.screen.blit(lbl_thresh, (col2_x + 8, top_y - 24))
        if self.dev_thresh_surf:
            if self.dev_thresh_surf.get_width() != cam_w or self.dev_thresh_surf.get_height() != cam_h:
                scaled_thresh = pygame.transform.smoothscale(self.dev_thresh_surf, (cam_w, cam_h))
                self.screen.blit(scaled_thresh, (col2_x, top_y + 4))
            else:
                self.screen.blit(self.dev_thresh_surf, (col2_x, top_y + 4))

        # --- 3. Bottom Left: Telemetry & Controls Card ---
        pygame.draw.rect(self.screen, (12, 15, 20), (margin_x - 4, row2_y - 28, cam_w + 8, tele_h + 36), border_radius=8)
        pygame.draw.rect(self.screen, (80, 220, 100), (margin_x - 4, row2_y - 28, cam_w + 8, tele_h + 36), 2, border_radius=8)
        lbl_tele = self.font_hud_title.render("DIAGNOSTICS & TELEMETRY", True, (100, 255, 120))
        self.screen.blit(lbl_tele, (margin_x + 8, row2_y - 24))

        fps = self.clock.get_fps()
        cam_pt = self.dev_info.get("point")
        cam_str = f"({cam_pt[0]:.1f}, {cam_pt[1]:.1f})" if cam_pt else "NONE (Light Off)"
        scr_str = f"({self.current_screen_point[0]}, {self.current_screen_point[1]})" if self.current_screen_point else "NONE"
        circ_val = self.dev_info.get("circularity", 0.0)
        aspect_val = self.dev_info.get("aspect_ratio", 0.0)
        in_display = self.dev_info.get("in_display_area", False)
        excess_val = self.dev_info.get("purple_excess", 0.0)
        is_ir = self.dev_info.get("is_ir_purple", False)
        is_emitting = self.dev_info.get("is_emitting", False)
        bulb_state = self.dev_info.get("bulb_state", "SEARCHING")
        bulb_pos = self.dev_info.get("bulb_pos")
        bulb_pos_str = f"({bulb_pos[0]:.1f}, {bulb_pos[1]:.1f})" if bulb_pos else "SEARCHING"

        action_str = "LIGHT OFF (PEN LIFTED)"
        if self.current_screen_point:
            if self._point_in_rect(self.current_screen_point, self.eraser_rect):
                action_str = f"OVER ERASER BUTTON [Hits: {self.eraser_dwell}]"
            elif self._point_in_rect(self.current_screen_point, self.save_rect):
                action_str = f"OVER SAVE BUTTON [Hits: {self.save_dwell}]"
            elif self.eraser_mode:
                action_str = "ERASING DOT"
            else:
                action_str = "LIGHT ON (MARKING ON BOARD)"

        if is_emitting:
            bulb_status_str = f"EMITTING PURPLE IR (+{excess_val:.0f} Excess) [DRAWING]"
        elif bulb_state == "LOCATED_IDLE":
            bulb_status_str = f"5mm BULB LOCATED AT {bulb_pos_str} [IDLE - PRESS BUTTON TO DRAW]"
        else:
            bulb_status_str = "SEARCHING FOR 5mm LED BULB..."

        tele_lines = [
            f"FPS              : {fps:.1f}",
            f"Display Size     : {self.width}x{self.height} (Aspect: {self.aspect_ratio:.2f}:1)",
            f"Threshold        : {self.current_threshold} (Press [ to lower, ] to raise)",
            f"Physical 5mm Bulb: {bulb_status_str}",
            f"Bulb Tip Position: {bulb_pos_str}",
            f"IR Excess Meter  : +{excess_val:.1f} (Threshold: > 8.0) -> {'[IR ON: YES]' if is_emitting else '[IR ON: NO]'}",
            f"Purple IR Light  : {'ACTIVE (EMITTING PURPLE)' if is_emitting else 'OFF / IDLE (PEN LIFTED)'}",
            f"White Board Disc : ACTIVE (Whiteboard drawings 100% rejected)",
            f"Display Bounds   : DocScanner Quad (Active)",
            f"In Display Area  : {'YES (Inside Display Quad)' if in_display else 'NO (Outside Screen Quad)'}",
            f"Dot Geometry     : Circ: {circ_val:.2f} | Aspect: {aspect_val:.2f} (Circles Only)",
            f"Camera Point     : {cam_str}",
            f"Screen Point     : {scr_str}",
            f"Marker Action    : {action_str}",
        ]

        text_y = row2_y + 4
        line_step = 22 if self.height < 900 else 24
        for line in tele_lines:
            color = (120, 255, 150) if "LIGHT ON" in line or "YES" in line else (210, 215, 225)
            line_surf = self.font_hud_val.render(line, True, color)
            self.screen.blit(line_surf, (margin_x + 14, text_y))
            text_y += line_step

        # --- 4. Bottom Right: Interactive Test Drawing Pad ---
        pygame.draw.rect(self.screen, (10, 12, 16), (col2_x - 4, row2_y - 28, cam_w + 8, tele_h + 36), border_radius=8)
        pygame.draw.rect(self.screen, (180, 100, 255), (col2_x - 4, row2_y - 28, cam_w + 8, tele_h + 36), 2, border_radius=8)
        lbl_pad = self.font_hud_title.render("TEST DRAWING PAD (Preview: Dots Appear Here)", True, (200, 130, 255))
        self.screen.blit(lbl_pad, (col2_x + 8, row2_y - 24))
        if self.dev_test_pad.get_width() != cam_w or self.dev_test_pad.get_height() != tele_h:
            scaled_pad = pygame.transform.scale(self.dev_test_pad, (cam_w, tele_h))
            self.screen.blit(scaled_pad, (col2_x, row2_y + 4))
        else:
            self.screen.blit(self.dev_test_pad, (col2_x, row2_y + 4))

        # Bottom help text
        help_text = "[ESC] Quit   [TAB] Switch to Fullscreen Whiteboard   [C] Clear Test Pad   [[ / ]] Tune IR Threshold"
        help_surf = self.font_status.render(help_text, True, (130, 140, 160))
        self.screen.blit(help_surf, (margin_x, self.height - 35))

    def save_canvas(self):
        """Save the current drawing to a timestamped PNG file."""
        os.makedirs(SAVE_DIR, exist_ok=True)
        filename = f"whiteboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(SAVE_DIR, filename)

        try:
            pygame.image.save(self.canvas, filepath)
            self.notification = (f"Saved: {filename}", time.time() + 2.5)
            print(f"[Whiteboard] Saved drawing to: {filepath}")
            pygame.draw.rect(self.screen, (60, 255, 60), (0, 0, self.width, self.height), 8)
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
        ex, ey, ew, eh = self.eraser_rect
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

        pygame.draw.rect(self.screen, bg_color, self.eraser_rect, border_radius=6)
        pygame.draw.rect(self.screen, border_color, self.eraser_rect, 2, border_radius=6)
        lbl_surf = self.font_btn.render(label, True, text_color)
        self.screen.blit(lbl_surf, (ex + (ew - lbl_surf.get_width()) // 2, ey + (eh - lbl_surf.get_height()) // 2))

        # Dwell progress bar
        if self.eraser_dwell > 0:
            fill_w = int(ew * min(1.0, self.eraser_dwell / BUTTON_DWELL_FRAMES))
            pygame.draw.rect(self.screen, (255, 120, 120), (ex, ey + eh - 6, fill_w, 6), border_radius=3)

        # --- SAVE BUTTON ---
        sx, sy, sw, sh = self.save_rect
        pygame.draw.rect(self.screen, (35, 35, 35), self.save_rect, border_radius=6)
        pygame.draw.rect(self.screen, (80, 200, 80), self.save_rect, 2, border_radius=6)
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
            bg_rect = pygame.Rect(20, self.height - 75, txt_surf.get_width() + 20, txt_surf.get_height() + 10)
            pygame.draw.rect(self.screen, (40, 40, 40), bg_rect, border_radius=4)
            pygame.draw.rect(self.screen, (90, 90, 90), bg_rect, 1, border_radius=4)
            self.screen.blit(txt_surf, (30, self.height - 70))

        # Bottom help text in production
        help_text = "[ESC] Quit   [TAB] Testing Bench   [E] Eraser   [S] Save   [C] Clear   [[ / ]] Tune Thresh"
        help_surf = self.font_status.render(help_text, True, (100, 100, 100))
        self.screen.blit(help_surf, (20, self.height - 35))

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
    print("Testing Whiteboard module standalone...")
    wb = Whiteboard()
    print("Whiteboard initialized successfully.")
    wb.cleanup()
