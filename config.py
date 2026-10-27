import os

# === Application Operating Mode ===
# "production" - Clean fullscreen whiteboard canvas
# "testing"    - Dedicated OpenCV testing bench (large camera feed, threshold mask, telemetry, and test pad)
DEFAULT_MODE = "production"

# Dimensions for OpenCV debug windows in testing mode
DEV_CAMERA_WIDTH = 540
DEV_CAMERA_HEIGHT = 405

# === Display Configuration & Auto-Detection ===
def get_display_resolution(fallback_w: int = 1920, fallback_h: int = 1080) -> tuple[int, int]:
    """
    Auto-detect the native resolution of the current connected display monitor.
    Works seamlessly across Windows, Raspberry Pi OS (X11 / Wayland / KMS-DRM), macOS, and Linux.
    Does not assume a fixed 1920x1080 display.
    """
    try:
        import pygame
        if not pygame.get_init():
            pygame.init()
        info = pygame.display.Info()
        if info.current_w > 0 and info.current_h > 0:
            return int(info.current_w), int(info.current_h)
    except Exception:
        pass
    return fallback_w, fallback_h


def get_calibration_points(width: int, height: int, margin: int = 50) -> list[tuple[int, int]]:
    """
    Calculate dynamic calibration target coordinates for any screen resolution.
    Returns: [Top-Left, Top-Right, Bottom-Right, Bottom-Left, Center].
    """
    m_x = min(margin, max(25, int(width * 0.035)))
    m_y = min(margin, max(25, int(height * 0.035)))
    return [
        (m_x, m_y),                             # Top-Left
        (width - m_x, m_y),                     # Top-Right
        (width - m_x, height - m_y),            # Bottom-Right
        (m_x, height - m_y),                    # Bottom-Left
        (width // 2, height // 2)               # Center Validation
    ]


def get_button_rects(width: int, height: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """
    Calculate dynamic ERASER and SAVE button zones anchored to the bottom-right corner
    of any screen resolution. Returns ((ex, ey, ew, eh), (sx, sy, sw, sh)).
    """
    btn_w = min(140, max(100, int(width * 0.08)))
    btn_h = min(60, max(42, int(height * 0.055)))
    margin_right = min(30, max(15, int(width * 0.02)))
    margin_bottom = min(30, max(15, int(height * 0.03)))
    gap = 16

    save_x = width - margin_right - btn_w
    save_y = height - margin_bottom - btn_h
    eraser_x = save_x - gap - btn_w
    eraser_y = save_y

    return (eraser_x, eraser_y, btn_w, btn_h), (save_x, save_y, btn_w, btn_h)


# Native display resolution auto-detected at startup
DETECTED_SCREEN_WIDTH, DETECTED_SCREEN_HEIGHT = get_display_resolution(1920, 1080)
SCREEN_WIDTH = DETECTED_SCREEN_WIDTH
SCREEN_HEIGHT = DETECTED_SCREEN_HEIGHT

# === Camera Configuration ===
# Backend selection:
#   "auto"   - Automatically detect (tries ribbon cable picamera2, then USB/OpenCV)
#   "ribbon" - Force Raspberry Pi CSI camera via onboard ribbon cable (picamera2)
#   "usb"    - Force USB / standard webcam via OpenCV (for Windows, macOS, or regular Linux)
CAMERA_BACKEND = "auto"
CAMERA_DEVICE_INDEX = 0       # Device index for USB/OpenCV webcam (0 is default webcam)

# Captured at 640x480 for real-time 30fps processing on Raspberry Pi 3B+
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Ignore bright spots located on the very outer border of the camera frame
# (Ceiling lights, lamps, and monitor bezel glares are typically at the top/sides of the FOV)
CAMERA_EDGE_MARGIN_TOP = 35
CAMERA_EDGE_MARGIN_BOTTOM = 15
CAMERA_EDGE_MARGIN_SIDE = 15

# === LED Optical Configuration & Color Selection ===
# "green" - High-brightness 5mm Green LED stylus (Green channel dominant: G > R and G > B)
# "ir"    - 850-940nm Infrared LED stylus (Purple/violet camera tint: B > G and R > G)
# "auto"  - Dynamically detect both Green LED and IR LED
LED_TYPE = "green"

# --- Green LED Spectral Thresholds ---
# Green LEDs emit at ~520-530nm. When active, the camera sensor records high Green (G)
# with suppressed Red (R) and Blue (B).
# Green Excess = (G - R) + (G - B). White light has G ≈ R ≈ B (excess ≈ 0), room lamps have R > G (excess < 0).
GREEN_EXCESS_THRESHOLD = 24.0      # Empirical threshold (> 15.0 = Green LED emission guaranteed)
GREEN_SPECTRAL_RATIO_MIN = 1.15    # 2*G / (max(1.0, R + B))
GREEN_THRESHOLD = 110              # Brightness threshold for green light

# --- IR / Near-Infrared Spectral Thresholds ---
# Digital camera sensors see near-IR (850-940nm) with a distinct light-purple/violet tint
# (Red & Blue high, Green significantly lower).
IR_THRESHOLD = 240
CALIBRATION_IR_THRESHOLD = 210
IR_PURPLE_EXCESS_THRESHOLD = 8.0  # Canonical empirical threshold (> 8.0 = IR light guaranteed)
IR_SPECTRAL_RATIO_MIN = 1.14      # (R + B) / (2 * max(G, 1)) must exceed this to qualify as IR
IR_PURPLE_EXCESS_MIN = 8.0        # (B - G) + (R - G) must exceed this (calibrated to >= 8.0)

# Minimum and maximum contour areas in pixels.
# An IR LED / remote light tip is compact (typically 4 - 500 px).
# Blobs larger than 500px are ambient ceiling fixtures or window glare, not an IR pen tip.
IR_MIN_AREA = 4
IR_MAX_AREA = 500

# === Pen & Eraser Configuration ===
PEN_COLOR = (255, 255, 255)    # Pure white
PEN_RADIUS = 4                 # ~8px diameter (medium stroke)
ERASER_RADIUS = 25             # ~50px diameter eraser

# Pen lift timeout in seconds (grace period to prevent stroke breakage on fast movement or brief dropped frames)
PEN_LIFT_TIMEOUT = 0.16

# Anti-spike / Anti-jump threshold in screen pixels.
# Increased to 280px to allow fast hand sweeps and writing to connect seamlessly as smooth solid lines.
MAX_STROKE_JUMP = 280

# === Calibration Configuration (Dynamically measured for current display) ===
CALIBRATION_POINTS = get_calibration_points(SCREEN_WIDTH, SCREEN_HEIGHT)
CALIBRATION_DOT_RADIUS = 30
CALIBRATION_FRAMES = 10
CALIBRATION_TIMEOUT_SEC = 4
MAX_VALIDATION_ERROR_PX = 50

# === UI Zones & Controls (Dynamically anchored to current display) ===
ERASER_BUTTON_RECT, SAVE_BUTTON_RECT = get_button_rects(SCREEN_WIDTH, SCREEN_HEIGHT)

# Frames of light presence required to trigger button.
# Set to 2 frames so a quick tap or remote blink reliably triggers without needing long continuous hold.
BUTTON_DWELL_FRAMES = 2

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration_data")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")
SAVE_DIR = os.path.join(BASE_DIR, "saves")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IR_MODEL_FILE = os.path.join(MODELS_DIR, "ir_led_model.json")
GREEN_MODEL_FILE = os.path.join(MODELS_DIR, "green_led_model.json")
PEN_MODEL_FILE = os.path.join(MODELS_DIR, "pen_model.json")
