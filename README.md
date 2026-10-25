# 📒 Smart Notebook (IR Pen Whiteboard)

An interactive digital whiteboard system designed for **Raspberry Pi 3B+ (64-bit OS)** paired with an **OV5647 1080p IR Camera** and a **1080p HDMI Monitor/TV**.

The system tracks an IR pen (an IR LED with a momentary push-button) pointed at the screen, calibrates for camera perspective distortion via Homography, and draws lines in real-time on a fullscreen black canvas.

---

## 🛠️ Hardware Requirements

1. **Raspberry Pi 3B+** running **Raspberry Pi OS 64-bit** (Bookworm or Bullseye).
2. **OV5647 IR Camera** (3.6mm lens) connected via the onboard CSI ribbon cable.
3. **HDMI Display / TV** (1080p recommended).
4. **IR Pen / Light Source**:
   - Built with an IR LED (850nm or 940nm), 3V battery, and momentary push button.
   - Or a standard TV remote control (transmits IR bursts for testing).

---

## 🚀 Setup & Installation on Raspberry Pi

### 1. Clone or Pull the Repository
```bash
git clone <YOUR_REPO_URL> Book
cd Book
```
*(If already cloned, simply run `git pull` to fetch updates).*

### 2. Install Dependencies
On Raspberry Pi OS Bookworm (Debian 12), use a virtual environment with system site-packages (required for `picamera2` and `libcamera`):

```bash
# Create venv with access to pre-installed picamera2 and libcamera
python3 -m venv venv --system-site-packages

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

*(Alternatively, without venv: `pip install -r requirements.txt --break-system-packages`)*

### 3. Verify Camera Connection (Diagnostic)
```bash
python3 test_camera.py
```
This will capture a frame and save `test_frame.png` to confirm your camera is working.

---

## 🎮 Running the Application

Launch the main application:
```bash
python3 main.py
```

### Camera Selection Options
When running `main.py` (or `test_camera.py`), you can select which camera to use:
- **Interactive selection**: If run without flags, you will see a prompt:
  ```text
  Camera Source Options:
    [1] Auto-detect (RPi Ribbon Cable -> USB/Windows)
    [2] Raspberry Pi Ribbon Cable (CSI / picamera2)
    [3] USB Webcam / Windows / Regular Linux (OpenCV)
  Select camera [1-3] or press [Enter] for default (auto):
  ```
- **CLI Flags**:
  - Force Raspberry Pi ribbon cable (CSI):
    ```bash
    python3 main.py --camera ribbon
    ```
  - Force USB / Windows / Regular Linux webcam:
    ```bash
    python3 main.py --camera usb
    # Or specify device index (e.g. index 1 for external USB webcam):
    python3 main.py --camera usb --device 1
    ```
- **Config setting**: Change `CAMERA_BACKEND = "ribbon"` or `"usb"` in `config.py`.

### Auto-Detect Native Display Resolution & Dynamic Sizing
The system does **NOT** assume a fixed 1920×1080 display. It automatically measures and adapts to whatever display is connected:
- **Universal Display Compatibility**: Supports 1366×768 (standard laptops), 1280×720 (720p monitors/TVs), 1920×1080 (1080p Full HD), 2560×1440 (2K QHD), 4K UHD, and Raspberry Pi touchscreens.
- **Maximum Screen Canvas & Native Aspect Ratio**: The blackboard canvas and coordinate space dynamically expand to the full maximum resolution and exact aspect ratio of the connected display.
- **Dynamic Calibration Targets**: Calibration points are automatically placed at the true edges and corners of the detected display resolution rather than off-screen fixed coordinates:
  - Top-Left: `(margin_x, margin_y)`
  - Top-Right: `(width - margin_x, margin_y)`
  - Bottom-Right: `(width - margin_x, height - margin_y)`
  - Bottom-Left: `(margin_x, height - margin_y)`
  - Center: `(width // 2, height // 2)`
- **Adaptive UI Controls**: On-screen ERASER and SAVE buttons are dynamically anchored to the bottom-right corner of the active display with proportional margins, guaranteeing they are always visible and accessible.

### Dual Calibration System: Auto DocScanner & Manual Mouse Modes
The calibration system maps your physical monitor's display boundaries to the camera view with two dedicated modes:
1. **Auto DocScanner Mode (Page Scanner Algorithm)**:
   - Uses real computer vision page scanner algorithms (`cv2.Canny`, `cv2.dilate`, `cv2.findContours`, `cv2.approxPolyDP`, and Otsu threshold fallback) to automatically locate the 4-sided polygon of your glowing monitor in the camera feed.
   - Automatically orders the quad vertices clockwise into `[Top-Left, Top-Right, Bottom-Right, Bottom-Left]`.
2. **Manual Mouse Mode (Click 4 Corners in Sequence)**:
   - Click directly on your monitor's 4 physical corners in the live camera feed:
     1. Top-Left `[TL]`
     2. Top-Right `[TR]`
     3. Bottom-Right `[BR]`
     4. Bottom-Left `[BL]`

### Interactive Calibration Studio with Mouse-Draggable Pins
Both Auto and Manual modes open into the **Calibration Studio**:
- **Draggable Corner Pins**: Hover over and drag any corner pin (`TL`, `TR`, `BR`, `BL`) with your mouse in real time to match your monitor's bezel with millimeter precision!
- **Live Camera Feed**: 30 FPS camera preview with the neon cyan quadrilateral, semi-transparent screen overlay, center crosshair, and edge dimension badges (`Top`, `Bottom`, `Left`, `Right` in pixels).
- **Telemetry & Convexity Checks**: Live display aspect ratio comparison, edge measurements, sub-pixel camera-to-display mapping, and status validation.
- **Controls & Action Buttons**:
  - **`[ENTER]` or Button [Accept & Continue]**: Locks in the calibration matrix, saves to `calibration_data/calibration.json`, and launches the Whiteboard.
  - **`[A]` or Button [Auto DocScanner Scan]**: Re-scans the camera feed using the DocScanner algorithm.
  - **`[M]` or Button [Manual Re-Click]**: Clears corners and prompts you to click all 4 corners again in sequence.
  - **`[SPACE]` or Button [Default Bounds]**: Resets to standard 1:1 camera boundary.
  - **`[ESC]`**: Aborts calibration.

### Strict IR Discrimination vs Normal White Light & Blackboard Drawings
Camera sensors without IR-cut filters perceive near-infrared light (850nm / 940nm) with a distinct **light-purple / violet / magenta signature** (Red & Blue channels are high, Green is substantially lower):
- **Authentic IR Light (Marker / Remote)**: $B \ge 240, R \ge 240, G \le 235$, Purple Excess $(B - G) + (R - G) \ge 18.0$, Spectral Ratio $\frac{R+B}{2G} \ge 1.08 \rightarrow$ **ACCEPTED & TRACKED**.
- **White Drawings on the Blackboard**: Drawn by equal RGB subpixels on your monitor ($R \approx G \approx B \approx 255$, Purple Excess $\le 0.0$, Spectral Ratio $\approx 1.00$) $\rightarrow$ **100% HARD REJECTED**. The system never detects its own white drawings as IR LEDs, preventing ghost runaway drawing loops.
- **Normal White Light**: Ceiling lamps, window daylight, flashlights, or white clothing are identified as neutral ($G \ge B - 5$) $\rightarrow$ **REJECTED**.

### Operating Modes (Production vs Testing/Development)
After calibration is complete, the application prompts you to choose an operating mode:
```text
Select Operating Mode:
  [1] Production Mode (Clean fullscreen whiteboard canvas)
  [2] Testing / Development Mode (Dedicated OpenCV camera feed, threshold mask & test pad)
Enter mode [1-2] or press [Enter] for default (production):
```

- **Production Mode**: Clean fullscreen blackboard canvas with no UI clutter. Direct IR drawing experience with automatic pen lift and anti-spike line filtering.
- **Testing / Development Mode**: Dedicated OpenCV test bench (does not scribble all over your fullscreen background):
  1. **Large Live Camera Feed (540x405)**: Real-time camera view showing the **DocScanner Display Boundary Quadrilateral** (the 4 edges and corners of the monitor outlined in neon cyan/green) and precision reticle tracking circular IR dots inside the display area.
  2. **Large OpenCV Threshold Mask (540x405)**: Binary mask showing exactly what pixels pass the active brightness threshold and chromaticity filter.
  3. **Diagnostics & Telemetry HUD**: Real-time FPS, threshold value, IR spectral type & purple excess, whiteboard drawing rejection status, DocScanner quadrilateral status, dot circularity & aspect ratio, camera coordinates `(cx, cy)`, screen coordinates `(sx, sy)`, and pen action.
  4. **Interactive Test Drawing Pad**: Dedicated preview pad where you can test strokes, pulsating/blinking remote lights, and button tapping in real time.
- **DocScanner Display Area & Dot Drawing**:
  - The camera calculates the 4 corners of the display just like a document scanner app (CamScanner / DocScanner).
  - Any light detected outside the display boundary is automatically ignored.
  - Whenever circular IR light turns on inside the display area (even if pulsating or blinking), it immediately stamps a dot at that exact position. Smooth strokes connect naturally as you move.
- **Command-Line Selection**:
  ```bash
  # Start directly in Testing / Dev Bench:
  python3 main.py --mode dev

  # Start directly in Production Mode:
  python3 main.py --mode prod
  ```
- **Runtime Mode Toggle**: Press **`TAB`** or **`D`** on your keyboard anytime while the app is running to switch between Production Whiteboard and Testing Bench instantly!
- **Dynamic Threshold Tuning**: Press **`[`** to decrease or **`]`** to increase the IR detection threshold live in response to ambient room lighting.

---

## ✏️ Controls & Features

| Action | How to Trigger |
|---|---|
| **Draw White Stroke** | Press and hold button on the IR pen / remote control. |
| **Stop Drawing (Pen Lift)** | Release button on the IR pen / remote (light turns off $\rightarrow$ immediate pen lift). |
| **Toggle Eraser Mode** | Blink/tap IR light on the **ERASER** button (bottom-right), or press **`E`** on keyboard. |
| **Save Drawing (PNG)** | Blink/tap IR light on the **SAVE** button (bottom-right), or press **`S`** on keyboard. Files are saved in `saves/`. |
| **Clear Canvas / Test Pad** | Press **`C`** on keyboard. |
| **Toggle Dev / Prod Mode** | Press **`TAB`** or **`D`** on keyboard. |
| **Adjust IR Threshold Live** | Press **`[`** (lower threshold) or **`]`** (raise threshold). |
| **Exit Application** | Press **`ESC`** on keyboard. |

---

## 📂 Project Structure

```
Book/
├── main.py              # Application entry point & main event loop
├── camera.py            # Hardware camera interface (picamera2 + OpenCV fallback)
├── calibration.py       # 5-point calibration & homography persistence
├── detector.py          # High-speed IR / bright white light detection
├── transformer.py       # Camera-to-screen perspective coordinate projection
├── whiteboard.py        # Pygame fullscreen drawing canvas & UI overlay
├── config.py            # Central configuration constants & tuning params
├── test_camera.py       # Camera hardware diagnostic script
├── requirements.txt     # Python package requirements
├── .gitignore           # Ignores runtime data (saves/, calibration_data/)
└── README.md            # Documentation
```

---

## ⚙️ Configuration & Tuning (`config.py`)

- **`IR_THRESHOLD`** (default: `240`): Brightness threshold for the IR pen. Increase if ambient light creates false positives; decrease if the IR LED is dim.
- **`IR_PURPLE_EXCESS_MIN`** (`18.0`): Minimum purple excess `(B - G) + (R - G)` required to identify authentic IR emissions. Rejects 100% of white drawings on the blackboard.
- **`IR_SPECTRAL_RATIO_MIN`** (`1.08`): Minimum spectral ratio `(R + B) / (2 * G)` required to validate IR emissions.
- **`CAMERA_WIDTH`, `CAMERA_HEIGHT`** (`640x480`): Frame capture resolution optimized for 30+ FPS real-time tracking on RPi 3B+.
- **`MAX_STROKE_JUMP`** (`150`): Anti-teleport filter threshold in screen pixels to reject spurious fast jumps.
- **`PEN_RADIUS`** (`4`): Pen stroke thickness (~8px diameter).
- **`ERASER_RADIUS`** (`25`): Eraser head size (~50px diameter).
- **`BUTTON_DWELL_FRAMES`** (`2`): Number of consecutive frames (~60ms) required to activate on-screen buttons (supports quick blinking remote taps).
