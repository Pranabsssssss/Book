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

### Differential Dots Display Calibration & Live Edge Review
The calibration maps your monitor's display boundaries to the camera view using automated **Differential Dots Calibration** (DocScanner Quad calculation):
- **Automated Target Dots on Display Edges**: The monitor displays a full black screen, captures a baseline camera frame, and then flashes white calibration targets one by one at each corner:
  1. Top-Left corner: `(50, 50)`
  2. Top-Right corner: `(W-50, 50)`
  3. Bottom-Right corner: `(W-50, H-50)`
  4. Bottom-Left corner: `(50, H-50)`
  5. Center verification point: `(W/2, H/2)`
- **Differential Background Subtraction**: Uses `cv2.absdiff(dot_frame, black_baseline)` to cancel out 100% of room lighting, ceiling lamps, and ambient reflections.
- **Sub-Pixel Centroid Precision**: Calculates exact sub-pixel centroids `(cx, cy)` from the differential moments, eliminating human hand jitter and parallax error.
- **Comprehensive Live Edge Review Screen (DocScanner View)**:
  - **Live Camera Feed**: Real-time 30 FPS camera feed with the detected display quadrilateral drawn in neon cyan `(0, 245, 255)`, semi-transparent polygon fill, diagonal guidelines, and edge dimension measurements (`Top`, `Bottom`, `Left`, `Right` in camera pixels).
  - **Color-Coded Corner Pins**: Corner nodes labeled with badges: `TL` (Green), `TR` (Yellow), `BR` (Orange), `BL` (Cyan), and `CENTER` crosshair.
  - **Diagnostics & Metrics Panel**: Shows sub-pixel camera coordinates for all corners, mapped screen coordinates, edge widths/heights, display aspect ratio, reprojection error, and geometry convexity check (`EXCELLENT` vs `WARNING: SKEWED`).
- **Interactive Decision Prompts (Accept or Recalibrate)**:
  - No rushed auto-timeouts! The review stays open so you can inspect the edges and alignment carefully.
  - **Button 1 / `[ENTER]`**: **Accept & Continue** — Confirms calibration, saves data to disk, and proceeds to Whiteboard mode.
  - **Button 2 / `[R]`**: **Recalibrate Again** — If calibration didn't happen properly, lighting caused glare, or edges look misaligned, click or press **`[R]`** to immediately restart the calibration routine.
  - **Button 3 / `[SPACE]`**: **Default Bounds (1:1)** — Standard 1:1 camera boundary fallback (convenient for PC development).
  - **`[ESC]`**: Abort calibration and exit.

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
