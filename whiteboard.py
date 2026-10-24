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
    PEN_LIFT_TIMEOUT,
    ERASER_BUTTON_RECT, SAVE_BUTTON_RECT,
    BUTTON_DWELL_FRAMES, SAVE_DIR,
    DEV_OVERLAY_WIDTH, DEV_OVERLAY_HEIGHT
)


class Whiteboard:
    """
    Interactive full-screen Pygame whiteboard application.
    Supports continuous drawing, eraser mode, dwell-based IR button triggers,
    keyboard shortcuts, PNG export, and dual Production / Testing modes.
    """
    def __init__(self, screen=None, dev_mode=False):
        if pygame is None:
            raise RuntimeError(
                "Pygame is required for the Whiteboard canvas. "
                "Please install dependencies with: pip install -r requirements.txt"
            )

        if not pygame.get_init():
            pygame.init()

        if screen is None:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        else:
            self.screen = screen

        pygame.display.set_caption("Smart Notebook - Whiteboard")
        self.clock = pygame.time.Clock()

        # Canvas surface holds the actual artwork (saved on export, excludes UI overlays)
        self.canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.canvas.fill((0, 0, 0))

        # Runtime states
        self.eraser_mode = False
        self.prev_point = None
        self.last_point_time = 0.0
        self.quit_requested = False

        # Operating Mode (Production vs Testing/Development)
        self.dev_mode = dev_mode
        self.threshold_delta = 0
        self.current_threshold = 245
        self.current_screen_point = None

        # Dev Overlay cache
        self.dev_frame_surf = None
        self.dev_thresh_surf = None
        self.dev_info = {}

        # Dwell progress counters
        self.eraser_dwell = 0
        self.save_dwell = 0

        # Visual feedback notification banner (text, expiry timestamp)
        mode_label = "TESTING / DEV MODE" if self.dev_mode else "PRODUCTION MODE"
        self.notification = (mode_label, time.time() + 2.5)

        # Fonts
        self.font_btn = pygame.font.SysFont("Arial", 22, bold=True)
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
            resized_cam = cv2.resize(annotated_frame, (DEV_OVERLAY_WIDTH, DEV_OVERLAY_HEIGHT))
            self.dev_frame_surf = pygame.image.frombuffer(
                resized_cam.tobytes(), (DEV_OVERLAY_WIDTH, DEV_OVERLAY_HEIGHT), 'RGB'
            )

        if thresh_mask is not None:
            resized_thresh = cv2.resize(thresh_mask, (DEV_OVERLAY_WIDTH, DEV_OVERLAY_HEIGHT))
            self.dev_thresh_surf = pygame.image.frombuffer(
                resized_thresh.tobytes(), (DEV_OVERLAY_WIDTH, DEV_OVERLAY_HEIGHT), 'RGB'
            )

    def handle_point(self, screen_point: tuple):
        """
        Process a detected pen location in screen coordinates.
        Determines whether point interacts with UI controls or paints on canvas.
        """
        now = time.time()
        px = max(0, min(SCREEN_WIDTH - 1, int(screen_point[0])))
        py = max(0, min(SCREEN_HEIGHT - 1, int(screen_point[1])))
        pt = (px, py)
        self.current_screen_point = pt

        # 1. Check Eraser Button Zone
        if self._point_in_rect(pt, ERASER_BUTTON_RECT):
            self.eraser_dwell += 1
            self.save_dwell = 0
            self.prev_point = None  # Inhibit drawing while interacting with controls

            if self.eraser_dwell >= BUTTON_DWELL_FRAMES:
                self.eraser_mode = not self.eraser_mode
                self.eraser_dwell = 0
                status_str = "Eraser Mode: ON" if self.eraser_mode else "Pen Mode: ON"
                self.notification = (status_str, now + 1.5)
                print(f"[Whiteboard] {status_str}")
            return

        # 2. Check Save Button Zone
        if self._point_in_rect(pt, SAVE_BUTTON_RECT):
            self.save_dwell += 1
            self.eraser_dwell = 0
            self.prev_point = None

            if self.save_dwell >= BUTTON_DWELL_FRAMES:
                self.save_canvas()
                self.save_dwell = 0
            return

        # Not over buttons: reset dwell counters
        self.eraser_dwell = 0
        self.save_dwell = 0

        # 3. Canvas Drawing / Erasing
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

    def handle_no_point(self):
        """
        Invoked when no IR light is detected in the current frame (pen button released).
        """
        self.current_screen_point = None
        # If pen has been absent longer than grace period, break stroke continuity
        if time.time() - self.last_point_time > PEN_LIFT_TIMEOUT:
            self.prev_point = None

        self.eraser_dwell = 0
        self.save_dwell = 0

    def render(self):
        """
        Render canvas to display and overlay UI controls and feedback notifications.
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
                    # Toggle between Production and Development / Testing Mode
                    self.dev_mode = not self.dev_mode
                    label = "Mode: TESTING / DEV" if self.dev_mode else "Mode: PRODUCTION (Clean)"
                    self.notification = (label, time.time() + 2.0)
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

        # Blit canvas to screen
        self.screen.blit(self.canvas, (0, 0))

        # In Dev Mode: render OpenCV camera feeds, threshold mask, and telemetry HUD
        if self.dev_mode:
            self._render_dev_hud()

        # Overlay UI buttons and status widgets
        self._render_buttons()
        self._render_status()

        pygame.display.flip()
        self.clock.tick(30)

    def _render_dev_hud(self):
        """Render Live Camera, Threshold Mask, and Telemetry cards for Development / Testing."""
        top_y = 20
        box_w = DEV_OVERLAY_WIDTH
        box_h = DEV_OVERLAY_HEIGHT

        # 1. Camera Tracking Card
        x_cam = 20
        pygame.draw.rect(self.screen, (20, 20, 20), (x_cam - 4, top_y - 4, box_w + 8, box_h + 32), border_radius=6)
        pygame.draw.rect(self.screen, (0, 200, 255), (x_cam - 4, top_y - 4, box_w + 8, box_h + 32), 2, border_radius=6)
        title_cam = self.font_hud_title.render("LIVE CAMERA (Tracking)", True, (0, 220, 255))
        self.screen.blit(title_cam, (x_cam + 4, top_y))
        if self.dev_frame_surf:
            self.screen.blit(self.dev_frame_surf, (x_cam, top_y + 24))

        # 2. Threshold Mask Card
        x_thresh = x_cam + box_w + 16
        pygame.draw.rect(self.screen, (20, 20, 20), (x_thresh - 4, top_y - 4, box_w + 8, box_h + 32), border_radius=6)
        pygame.draw.rect(self.screen, (255, 180, 0), (x_thresh - 4, top_y - 4, box_w + 8, box_h + 32), 2, border_radius=6)
        title_thresh = self.font_hud_title.render("OPENCV THRESHOLD MASK", True, (255, 190, 20))
        self.screen.blit(title_thresh, (x_thresh + 4, top_y))
        if self.dev_thresh_surf:
            self.screen.blit(self.dev_thresh_surf, (x_thresh, top_y + 24))

        # 3. Telemetry & Stats Card
        x_info = x_thresh + box_w + 16
        info_w = 360
        pygame.draw.rect(self.screen, (15, 15, 20), (x_info, top_y - 4, info_w, box_h + 32), border_radius=6)
        pygame.draw.rect(self.screen, (80, 220, 100), (x_info, top_y - 4, info_w, box_h + 32), 2, border_radius=6)

        title_info = self.font_hud_title.render("OPENCV TELEMETRY & CONTROLS", True, (100, 255, 120))
        self.screen.blit(title_info, (x_info + 10, top_y))

        fps = self.clock.get_fps()
        cam_pt = self.dev_info.get("point")
        cam_str = f"({cam_pt[0]:.1f}, {cam_pt[1]:.1f})" if cam_pt else "None"
        scr_str = f"({self.current_screen_point[0]}, {self.current_screen_point[1]})" if self.current_screen_point else "None"
        area_val = self.dev_info.get("area", 0.0)
        status = self.dev_info.get("status", "SEARCHING")

        action_str = "PEN UP (LIFTED)"
        if self.current_screen_point:
            if self._point_in_rect(self.current_screen_point, ERASER_BUTTON_RECT):
                action_str = f"DWELLING (ERASER) [{self.eraser_dwell}/{BUTTON_DWELL_FRAMES}]"
            elif self._point_in_rect(self.current_screen_point, SAVE_BUTTON_RECT):
                action_str = f"DWELLING (SAVE) [{self.save_dwell}/{BUTTON_DWELL_FRAMES}]"
            elif self.eraser_mode:
                action_str = "ERASING"
            else:
                action_str = "DRAWING"

        lines = [
            f"Mode      : TESTING / DEV ([TAB] toggles)",
            f"FPS       : {fps:.1f}",
            f"Threshold : {self.current_threshold} (Press [ / ] to tune)",
            f"Status    : {status}",
            f"Action    : {action_str}",
            f"Camera Pt : {cam_str}",
            f"Screen Pt : {scr_str}",
            f"Blob Area : {area_val:.1f} px",
        ]

        text_y = top_y + 26
        for line in lines:
            rendered_line = self.font_hud_val.render(line, True, (210, 210, 210))
            self.screen.blit(rendered_line, (x_info + 12, text_y))
            text_y += 20

        # Draw visual screen reticle at current tracking position
        if self.current_screen_point:
            sx, sy = self.current_screen_point
            pygame.draw.circle(self.screen, (0, 255, 255), (sx, sy), 10, 2)
            pygame.draw.circle(self.screen, (0, 255, 255), (sx, sy), 2)

    def save_canvas(self):
        """Save the current drawing to a timestamped PNG file."""
        os.makedirs(SAVE_DIR, exist_ok=True)
        filename = f"whiteboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(SAVE_DIR, filename)

        try:
            pygame.image.save(self.canvas, filepath)
            self.notification = (f"Saved: {filename}", time.time() + 2.5)
            print(f"[Whiteboard] Saved drawing to: {filepath}")
        except Exception as e:
            self.notification = (f"Save Failed: {e}", time.time() + 3.0)
            print(f"[Whiteboard] Error saving file: {e}")

    def clear_canvas(self):
        """Clear the drawing canvas."""
        self.canvas.fill((0, 0, 0))
        self.notification = ("Canvas Cleared", time.time() + 1.5)
        print("[Whiteboard] Canvas cleared.")

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
            fill_w = int(ew * (self.eraser_dwell / BUTTON_DWELL_FRAMES))
            pygame.draw.rect(self.screen, (255, 120, 120), (ex, ey + eh - 6, fill_w, 6), border_radius=3)

        # --- SAVE BUTTON ---
        sx, sy, sw, sh = SAVE_BUTTON_RECT
        pygame.draw.rect(self.screen, (35, 35, 35), SAVE_BUTTON_RECT, border_radius=6)
        pygame.draw.rect(self.screen, (80, 200, 80), SAVE_BUTTON_RECT, 2, border_radius=6)
        save_surf = self.font_btn.render("SAVE", True, (100, 240, 100))
        self.screen.blit(save_surf, (sx + (sw - save_surf.get_width()) // 2, sy + (sh - save_surf.get_height()) // 2))

        # Dwell progress bar
        if self.save_dwell > 0:
            fill_w = int(sw * (self.save_dwell / BUTTON_DWELL_FRAMES))
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

        # Bottom help text
        help_text = "[ESC] Quit   [TAB/D] Mode Toggle   [E] Eraser   [S] Save   [C] Clear   [[ / ]] Tune Thresh"
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
