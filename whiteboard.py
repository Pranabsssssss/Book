import os
import time
from datetime import datetime
import pygame

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    PEN_COLOR, PEN_RADIUS, ERASER_RADIUS,
    PEN_LIFT_TIMEOUT,
    ERASER_BUTTON_RECT, SAVE_BUTTON_RECT,
    BUTTON_DWELL_FRAMES, SAVE_DIR
)


class Whiteboard:
    """
    Interactive full-screen Pygame whiteboard application.
    Supports continuous drawing, eraser mode, dwell-based IR button triggers,
    keyboard shortcuts, and PNG export.
    """
    def __init__(self, screen=None):
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

        # Dwell progress counters
        self.eraser_dwell = 0
        self.save_dwell = 0

        # Visual feedback notification banner (text, expiry timestamp)
        self.notification = ("Ready", time.time() + 2.0)

        # Fonts
        self.font_btn = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_status = pygame.font.SysFont("Arial", 18)

    def handle_point(self, screen_point: tuple):
        """
        Process a detected pen location in screen coordinates.
        Determines whether point interacts with UI controls or paints on canvas.
        """
        now = time.time()
        px = max(0, min(SCREEN_WIDTH - 1, int(screen_point[0])))
        py = max(0, min(SCREEN_HEIGHT - 1, int(screen_point[1])))
        pt = (px, py)

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
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Mouse click support for PC testing
                self.handle_point(event.pos)
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                self.handle_point(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.handle_no_point()

        # Blit canvas to screen
        self.screen.blit(self.canvas, (0, 0))

        # Overlay UI buttons and status widgets
        self._render_buttons()
        self._render_status()

        pygame.display.flip()
        self.clock.tick(30)

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
            bg_rect = pygame.Rect(20, 20, txt_surf.get_width() + 20, txt_surf.get_height() + 10)
            pygame.draw.rect(self.screen, (40, 40, 40), bg_rect, border_radius=4)
            pygame.draw.rect(self.screen, (90, 90, 90), bg_rect, 1, border_radius=4)
            self.screen.blit(txt_surf, (30, 25))

        # Bottom help text
        help_surf = self.font_status.render("[ESC] Quit   [E] Toggle Eraser   [S] Save   [C] Clear", True, (90, 90, 90))
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
    # Run in small windowed mode for quick sanity check
    wb = Whiteboard()
    print("Whiteboard initialized successfully.")
    wb.cleanup()
