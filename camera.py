import sys
import time
import argparse
import numpy as np
from config import CAMERA_BACKEND, CAMERA_DEVICE_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

# Attempt to import Picamera2 (standard on Raspberry Pi OS 64-bit for CSI ribbon cable)
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    pass

# OpenCV fallback for Windows, standard Linux, or USB webcams
try:
    import cv2
except ImportError:
    cv2 = None


def normalize_backend(backend: str) -> str:
    """Normalize user input string to canonical backend: 'ribbon', 'usb', or 'auto'."""
    if backend is None:
        return "auto"
    b = str(backend).strip().lower()
    if b in ("ribbon", "csi", "picam", "picam2", "picamera", "picamera2", "rpi", "2"):
        return "ribbon"
    if b in ("usb", "opencv", "webcam", "windows", "linux", "v4l2", "3"):
        return "usb"
    return "auto"


class Camera:
    """
    Flexible Camera interface supporting:
      1. 'ribbon' - Raspberry Pi CSI camera via ribbon cable (picamera2)
      2. 'usb'    - Windows, macOS, or regular Linux USB webcam (OpenCV VideoCapture)
      3. 'auto'   - Detects ribbon cable first; falls back to USB/OpenCV webcam
    """
    def __init__(
        self,
        backend=CAMERA_BACKEND,
        device_index=CAMERA_DEVICE_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS
    ):
        self.requested_backend = normalize_backend(backend)
        self.device_index = int(device_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.backend = None
        self.picam2 = None
        self.cap = None
        self.running = False

        self._configure_backend()

    def _configure_backend(self):
        """Resolve requested backend to either 'picamera2' or 'opencv'."""
        if self.requested_backend == "ribbon":
            if not PICAMERA2_AVAILABLE:
                raise RuntimeError(
                    "[Camera] Ribbon cable camera requested, but 'picamera2' is not installed.\n"
                    "On Raspberry Pi OS 64-bit, picamera2 is pre-installed. If using a venv, create it with:\n"
                    "  python3 -m venv venv --system-site-packages"
                )
            self._init_picam2()

        elif self.requested_backend == "usb":
            if cv2 is None:
                raise RuntimeError(
                    "[Camera] USB/OpenCV webcam requested, but 'cv2' is not installed.\n"
                    "Run: pip install -r requirements.txt"
                )
            self.backend = "opencv"

        else:  # 'auto'
            if PICAMERA2_AVAILABLE:
                try:
                    self._init_picam2()
                except Exception as e:
                    print(f"[Camera] Auto-detect: Ribbon camera init failed ({e}). Falling back to USB/OpenCV...")
                    self.picam2 = None

            if self.backend is None:
                if cv2 is None:
                    raise RuntimeError(
                        "Neither picamera2 nor cv2 is available. "
                        "Please install dependencies from requirements.txt."
                    )
                self.backend = "opencv"

    def _init_picam2(self):
        """Initialize Picamera2 configuration for CSI ribbon cable camera."""
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.backend = "picamera2"
        except Exception as e:
            self.picam2 = None
            raise RuntimeError(
                f"[Camera] Failed to configure Raspberry Pi ribbon cable camera: {e}\n"
                "Ensure ribbon cable is seated correctly and camera is enabled in raspi-config."
            )

    def start(self):
        """Start the camera video stream."""
        if self.backend == "picamera2":
            try:
                self.picam2.start()
                self.running = True
                time.sleep(1.0)  # Allow sensor exposure and white-balance to settle
                print(f"[Camera] Started Ribbon Cable Camera (picamera2: {self.width}x{self.height} @ {self.fps}fps)")
            except Exception as e:
                raise RuntimeError(
                    f"[Camera] Failed to start Ribbon Cable Camera: {e}\n"
                    "Check ribbon cable pin orientation and connection."
                )

        elif self.backend == "opencv":
            # On Windows, try DirectShow (CAP_DSHOW) first to prevent startup delays
            if sys.platform == "win32":
                self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.device_index)
            else:
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"[Camera] Could not open USB/webcam device index {self.device_index}.\n"
                    "Verify webcam is connected, or try a different index with --device 1."
                )

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.running = True
            time.sleep(0.5)
            print(f"[Camera] Started USB/Webcam (OpenCV device index {self.device_index}: {self.width}x{self.height})")

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame.
        Returns:
            RGB numpy array of shape (H, W, 3) and dtype uint8.
        """
        if not self.running:
            raise RuntimeError("[Camera] Cannot capture: camera not started.")

        if self.backend == "picamera2":
            return self.picam2.capture_array()
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError("[Camera] Failed to read frame from capture device.")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame_rgb.shape[1], frame_rgb.shape[0]) != (self.width, self.height):
                frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))
            return frame_rgb

    def stop(self):
        """Stop and release camera resources."""
        if not self.running:
            return

        if self.backend == "picamera2" and self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                print(f"[Camera] Warning on closing picamera2: {e}")
        elif self.backend == "opencv" and self.cap is not None:
            self.cap.release()

        self.running = False
        print("[Camera] Stopped.")

    def open(self) -> bool:
        """Alias for start() returning boolean success."""
        try:
            self.start()
            return True
        except Exception as e:
            print(f"[Camera] Error opening camera: {e}")
            return False

    def close(self):
        """Alias for stop()."""
        self.stop()

    def read(self):
        """Standard OpenCV-like read method returning (ret, frame_bgr)."""
        if not self.running:
            return False, None
        try:
            frame_rgb = self.capture_frame()
            if frame_rgb is None:
                return False, None
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        except Exception:
            return False, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Camera Interface")
    parser.add_argument(
        "-c", "--camera",
        type=str,
        default=CAMERA_BACKEND,
        choices=["auto", "ribbon", "usb", "csi", "opencv", "webcam"],
        help="Camera type: 'ribbon' (RPi CSI), 'usb' (Windows/Linux webcam), or 'auto'"
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=CAMERA_DEVICE_INDEX,
        help="Device index for USB/OpenCV webcam (default: 0)"
    )
    args = parser.parse_args()

    print(f"Testing Camera with backend='{args.camera}', device={args.device}...")
    cam = Camera(backend=args.camera, device_index=args.device)
    try:
        cam.start()
        test_frame = cam.capture_frame()
        print(f"Captured frame successfully: shape={test_frame.shape}, dtype={test_frame.dtype}")
        if cv2 is not None:
            cv2.imwrite("test_frame.png", cv2.cvtColor(test_frame, cv2.COLOR_RGB2BGR))
            print("Saved 'test_frame.png' for verification.")
    finally:
        cam.stop()
