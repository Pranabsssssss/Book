import sys
import time
import numpy as np
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

# Attempt to import Picamera2 (standard on Raspberry Pi OS 64-bit)
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    pass

# OpenCV fallback for development / USB webcams
try:
    import cv2
except ImportError:
    cv2 = None


class Camera:
    """
    Camera interface supporting Raspberry Pi CSI camera via picamera2,
    with automatic fallback to cv2.VideoCapture for development on PC/USB webcams.
    """
    def __init__(self, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = None
        self.picam2 = None
        self.cap = None
        self.running = False

        if PICAMERA2_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.backend = "picamera2"
            except Exception as e:
                print(f"[Camera] Picamera2 init failed ({e}), falling back to cv2.VideoCapture.")
                self.picam2 = None

        if self.backend is None:
            if cv2 is None:
                raise RuntimeError(
                    "Neither picamera2 nor cv2 is available. "
                    "Please install dependencies from requirements.txt."
                )
            self.backend = "opencv"

    def start(self):
        """Start the camera stream."""
        if self.backend == "picamera2":
            try:
                self.picam2.start()
                self.running = True
                # Allow camera sensor auto-exposure and gain to settle
                time.sleep(1.0)
                print(f"[Camera] Started Picamera2 ({self.width}x{self.height} @ {self.fps}fps)")
            except Exception as e:
                raise RuntimeError(
                    f"[Camera] Failed to start Raspberry Pi Camera: {e}\n"
                    "Ensure ribbon cable is securely connected and camera is enabled in raspi-config."
                )
        elif self.backend == "opencv":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError(
                    "[Camera] OpenCV could not open video device index 0. "
                    "Check webcam/camera connection."
                )
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.running = True
            time.sleep(0.5)
            print(f"[Camera] Started OpenCV VideoCapture ({self.width}x{self.height})")

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame.
        Returns:
            RGB numpy array with shape (H, W, 3) and dtype uint8.
        """
        if not self.running:
            raise RuntimeError("[Camera] Cannot capture frame: camera not started.")

        if self.backend == "picamera2":
            # Picamera2 returns RGB888 array directly
            frame = self.picam2.capture_array()
            return frame
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError("[Camera] Failed to read frame from OpenCV capture device.")
            # OpenCV captures BGR by default, convert to RGB for consistency
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


if __name__ == "__main__":
    print("Testing Camera module standalone...")
    cam = Camera()
    try:
        cam.start()
        test_frame = cam.capture_frame()
        print(f"Captured frame successfully: shape={test_frame.shape}, dtype={test_frame.dtype}")
        if cv2 is not None:
            # Save test image in BGR format
            cv2.imwrite("test_frame.png", cv2.cvtColor(test_frame, cv2.COLOR_RGB2BGR))
            print("Saved 'test_frame.png' for verification.")
    finally:
        cam.stop()
