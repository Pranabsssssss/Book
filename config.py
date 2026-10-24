import os

# === Application Operating Mode ===
# "production" - Clean fullscreen whiteboard canvas
# "testing"    - Dedicated OpenCV testing bench (large camera feed, threshold mask, telemetry, and test pad)
DEFAULT_MODE = "production"

# Dimensions for OpenCV debug windows in testing mode
DEV_CAMERA_WIDTH = 540
DEV_CAMERA_HEIGHT = 405

# === Display Configuration ===
# Default target is full HD 1080p HDMI monitor
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

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
CAMERA_EDGE_MARGIN_TOP = 20
CAMERA_EDGE_MARGIN_BOTTOM = 10
CAMERA_EDGE_MARGIN_SIDE = 10

# === IR / White Light Detection ===
# Brightness threshold (0-255). Values above this are considered bright IR/white light.
# Remote LEDs and IR pens have peak brightness ~245-255.
IR_THRESHOLD = 240
CALIBRATION_IR_THRESHOLD = 200

# Minimum and maximum contour areas in pixels.
# An IR LED / remote light tip is compact (typically 3 - 150 px).
# Blobs larger than 180px are ambient ceiling fixtures or window glare, not an IR pen tip.
IR_MIN_AREA = 3
IR_MAX_AREA = 180

# === Pen & Eraser Configuration ===
PEN_COLOR = (255, 255, 255)    # Pure white
PEN_RADIUS = 4                 # ~8px diameter (medium stroke)
ERASER_RADIUS = 25             # ~50px diameter eraser

# Pen lift timeout in seconds (grace period to prevent stroke breakage on brief dropped frames)
PEN_LIFT_TIMEOUT = 0.10

# Anti-spike / Anti-jump threshold in screen pixels.
# If distance between consecutive points > MAX_STROKE_JUMP in 1/30s, do not connect with a line.
# Prevents spiderweb lines when pen moves while off, or on brief lighting shifts.
MAX_STROKE_JUMP = 150

# === Calibration Configuration ===
CALIBRATION_POINTS = [
    (100, 100),              # Top-Left
    (1820, 100),             # Top-Right
    (1820, 980),             # Bottom-Right
    (100, 980),              # Bottom-Left
    (960, 540),              # Center (Validation)
]
CALIBRATION_DOT_RADIUS = 30
CALIBRATION_FRAMES = 10
CALIBRATION_TIMEOUT_SEC = 4
MAX_VALIDATION_ERROR_PX = 50

# === UI Zones & Controls ===
# Format: (x, y, width, height) in bottom-right corner
ERASER_BUTTON_RECT = (1600, 990, 140, 60)
SAVE_BUTTON_RECT = (1760, 990, 140, 60)

# Frames of light presence required to trigger button.
# Set to 2 frames so a quick tap or remote blink reliably triggers without needing long continuous hold.
BUTTON_DWELL_FRAMES = 2

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration_data")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")
SAVE_DIR = os.path.join(BASE_DIR, "saves")
