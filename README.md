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

### Calibration on Startup
- **First Run**: The app will automatically enter calibration mode. It displays 5 targets (4 corners + 1 center validation point) sequentially. The camera detects the points to compute a 3×3 perspective homography matrix.
- **Subsequent Runs**: When started, it will ask:
  ```text
  Existing calibration found.
  Press 'r' to recalibrate, or [Enter] to use existing data:
  ```
  Press **Enter** to instantly jump into drawing mode using saved calibration data (`calibration_data/calibration.json`), or type **`r`** to recalibrate.
- **Force Recalibrate Flag**:
  ```bash
  python3 main.py --recalibrate
  ```

---

## ✏️ Controls & Features

| Action | How to Trigger |
|---|---|
| **Draw White Stroke** | Press and hold button on the IR pen. |
| **Stop Drawing (Pen Lift)** | Release button on the IR pen. |
| **Toggle Eraser Mode** | Hover IR pen over the **ERASER** button (bottom-right) for **0.5s** (dwell), or press **`E`** on keyboard. |
| **Save Drawing (PNG)** | Hover IR pen over the **SAVE** button (bottom-right) for **0.5s** (dwell), or press **`S`** on keyboard. Files are saved in `saves/`. |
| **Clear Canvas** | Press **`C`** on keyboard. |
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

- **`IR_THRESHOLD`** (default: `245`): Brightness threshold for the IR pen. Increase if ambient light creates false positives; decrease if the IR LED is dim.
- **`CAMERA_WIDTH`, `CAMERA_HEIGHT`** (`640x480`): Frame capture resolution optimized for 30+ FPS real-time tracking on RPi 3B+.
- **`PEN_RADIUS`** (`4`): Pen stroke thickness (~8px diameter).
- **`ERASER_RADIUS`** (`25`): Eraser head size (~50px diameter).
- **`BUTTON_DWELL_FRAMES`** (`15`): Number of consecutive frames (~0.5s at 30fps) required to activate on-screen buttons.
