import os

# === Display Configuration ===
# Default target is full HD 1080p HDMI monitor
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# === Camera Configuration ===
# Captured at 640x480 for real-time 30fps processing on Raspberry Pi 3B+
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# === IR / White Light Detection ===
# Brightness threshold (0-255). Values above this are considered bright IR/white light.
IR_THRESHOLD = 245
# Lower threshold during calibration in case monitor white dot is less intense than direct IR LED
CALIBRATION_IR_THRESHOLD = 200

# Minimum and maximum contour areas in pixels to filter noise and reflections
IR_MIN_AREA = 4
IR_MAX_AREA = 5000

# === Pen & Eraser Configuration ===
PEN_COLOR = (255, 255, 255)    # Pure white
PEN_RADIUS = 4                 # ~8px diameter (medium stroke)
ERASER_RADIUS = 25             # ~50px diameter eraser

# Pen lift timeout in seconds (grace period to prevent stroke breakage on brief dropped frames)
PEN_LIFT_TIMEOUT = 0.12

# === Calibration Configuration ===
# 5 reference points: 4 corners (with 100px inset margin) + center validation point
CALIBRATION_POINTS = [
    (100, 100),              # Top-Left
    (1820, 100),             # Top-Right
    (1820, 980),             # Bottom-Right
    (100, 980),              # Bottom-Left
    (960, 540),              # Center (Validation)
]
CALIBRATION_DOT_RADIUS = 30   # Radius of visual calibration target on screen
CALIBRATION_FRAMES = 10       # Number of stable frames to collect per point to take median
CALIBRATION_TIMEOUT_SEC = 4   # Max seconds to wait for detection per point
MAX_VALIDATION_ERROR_PX = 50  # Maximum acceptable pixel deviation on center test point

# === UI Zones & Controls ===
# Format: (x, y, width, height) in bottom-right corner
ERASER_BUTTON_RECT = (1600, 990, 140, 60)
SAVE_BUTTON_RECT = (1760, 990, 140, 60)

# Dwell frames required to trigger a button (~0.5s at 30fps)
BUTTON_DWELL_FRAMES = 15

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration_data")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")
SAVE_DIR = os.path.join(BASE_DIR, "saves")
