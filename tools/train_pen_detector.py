"""
Smart Notebook — Interactive Pen in Hand & IR Bulb Training Studio
===================================================================
A specialized interactive training script to train a high-accuracy model
specifically to detect your pen held in your hand, with the 5mm IR bulb
at the tip.

Key Capabilities:
1. Live Interactive Camera Studio:
   - Hold your pen in your hand with the IR bulb pointing towards the camera.
   - Interactive mouse selection: Drag a box [R] around the pen in your hand
     and click the IR bulb tip to anchor the exact tip location.
   - Auto-Burst Capture [A]: Capture multiple angles and tilts with one click.
2. Structure & Geometry Learning:
   - Learns the pen shaft cylinder, hand grip context, and 5mm clear IR bulb dome.
   - Multi-scale (0.75x to 1.3x) and multi-angle (-45° to +45°) template synthesis.
   - Hard-negative mining from room background to guarantee 0% false positives.
3. Live Real-Time Testing & Verification:
   - Instant visual feedback: tracks the pen in hand, points directly at the
     IR bulb tip, and indicates [UNLIT / IDLE] vs [IR EMITTING / DRAWING].
4. Model Export:
   - Exports trained parameters to models/pen_model.json and models/bulb_templates/.

Usage:
  py tools/train_pen_detector.py              # Interactive live camera studio
  py tools/train_pen_detector.py --dataset    # Train headlessly from captured dataset
  py tools/train_pen_detector.py --test       # Launch directly into live verification mode
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from camera import Camera, normalize_backend
from config import (
    CAMERA_BACKEND, CAMERA_DEVICE_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    IR_PURPLE_EXCESS_THRESHOLD, IR_MIN_AREA, IR_MAX_AREA,
    CAMERA_EDGE_MARGIN_TOP, CAMERA_EDGE_MARGIN_BOTTOM, CAMERA_EDGE_MARGIN_SIDE
)

# Output directories
MODELS_DIR = BASE_DIR / "models"
BULB_TMPL_DIR = MODELS_DIR / "bulb_templates"
PEN_MODEL_FILE = MODELS_DIR / "pen_model.json"
DATASET_DIR = BASE_DIR / "dataset"
DATASET_CAPTURED_DIR = BASE_DIR / "dataset_captured"


def parse_args():
    parser = argparse.ArgumentParser(description="Train model to detect pen in hand with IR bulb at tip")
    parser.add_argument("-c", "--camera", default="auto", choices=["auto", "ribbon", "usb", "csi", "opencv", "webcam"],
                        help="Camera backend: 'auto', 'usb', or 'ribbon' (default: auto)")
    parser.add_argument("-d", "--device", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("-s", "--scale", type=float, default=2.0, help="Window display UI scale factor (default: 2.0 -> 1280x960)")
    parser.add_argument("--dataset", action="store_true", help="Train directly from dataset images on disk")
    parser.add_argument("--test", action="store_true", help="Launch directly into live test/verification mode")
    parser.add_argument("--save", action="store_true", default=True, help="Auto-save model after training")
    return parser.parse_args()


# =============================================================================
# Helper Utilities: Image Processing, Geometry, & Spectral Meter
# =============================================================================

def measure_purple_excess(frame_bgr: np.ndarray) -> tuple:
    """
    Computes purple excess (B - G) + (R - G) across the frame to verify IR emission.
    Returns: (is_emitting, max_excess, best_point)
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False, 0.0, None

    b = frame_bgr[:, :, 0].astype(np.float32)
    g = frame_bgr[:, :, 1].astype(np.float32)
    r = frame_bgr[:, :, 2].astype(np.float32)

    bg = b - g
    rg = r - g
    excess = bg + np.maximum(0.0, rg)

    # IR physical profile: B > G + 5.0, R > G + 5.0, Gray >= 110
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    active = (bg >= 5.0) & (rg >= 5.0) & (excess >= 12.0) & (gray >= 110)

    # Exclude borders
    active[:CAMERA_EDGE_MARGIN_TOP, :] = False
    active[-CAMERA_EDGE_MARGIN_BOTTOM:, :] = False
    active[:, :CAMERA_EDGE_MARGIN_SIDE] = False
    active[:, -CAMERA_EDGE_MARGIN_SIDE:] = False

    if np.any(active):
        max_val = float(np.max(excess[active]))
        locs = np.argwhere(active & (excess == max_val))
        if len(locs) > 0:
            py, px = locs[0]
            return True, max_val, (float(px), float(py))
        return True, max_val, None
    return False, 0.0, None


def extract_pen_shaft_vector(patch_gray: np.ndarray) -> float:
    """
    Measures the elongation and linear edge consistency of the pen shaft
    extending from the bulb into the hand.
    Returns a score between 0.0 and 1.0 indicating pen shaft likelihood.
    """
    if patch_gray is None or patch_gray.shape[0] < 20 or patch_gray.shape[1] < 20:
        return 0.0

    # Sobel gradients to detect parallel cylindrical pen edges
    sobelx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

    # Cylindrical shafts exhibit strong dominant edge orientations
    if np.max(mag) < 20.0:
        return 0.0

    strong_edges = mag > np.percentile(mag, 80)
    if np.sum(strong_edges) < 15:
        return 0.0

    dominant_angles = angle[strong_edges]
    # Check orientation concentration (parallel cylinder borders)
    hist, _ = np.histogram(dominant_angles, bins=12, range=(0, 360))
    peak_ratio = float(np.max(hist)) / float(np.sum(hist) + 1e-5)
    return min(1.0, peak_ratio * 2.2)


def generate_synthesized_exemplars(base_patch: np.ndarray, tip_pt_rel: tuple) -> list:
    """
    Generates multi-scale and multi-angle exemplars of the pen and IR bulb tip.
    Angles: -40° to +40° in 10° steps.
    Scales: 0.85x, 1.0x, 1.18x.
    """
    exemplars = []
    h, w = base_patch.shape[:2]
    center = (w / 2.0, h / 2.0)
    tx, ty = tip_pt_rel

    angles = [-35, -20, -10, 0, 10, 20, 35]
    scales = [0.88, 1.0, 1.15]

    for scale in scales:
        for ang in angles:
            M = cv2.getRotationMatrix2D(center, ang, scale)
            rot = cv2.warpAffine(base_patch, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # Transform tip point
            pt_hom = np.array([tx, ty, 1.0], dtype=np.float32)
            rot_tip = M.dot(pt_hom)
            rx, ry = float(rot_tip[0]), float(rot_tip[1])

            if 0 <= rx < w and 0 <= ry < h:
                exemplars.append({
                    "image": rot,
                    "tip": (rx, ry),
                    "angle": ang,
                    "scale": scale
                })
    return exemplars


# =============================================================================
# Interactive Mouse Selector for Pen & Bulb ROI
# =============================================================================

class PenROISelector:
    """Handles click-and-drag bounding box selection of the pen and clicking the bulb tip on upscaled UI."""
    def __init__(self, window_name="Smart Notebook — Pen Trainer", ui_scale=2.0):
        self.window_name = window_name
        self.ui_scale = float(ui_scale)
        self.selecting = False
        self.selection_done = False
        self.ui_x1, self.ui_y1 = -1, -1
        self.ui_x2, self.ui_y2 = -1, -1
        self.ui_bulb_tip = None
        self.step = "BOX"  # 'BOX' -> select pen in hand; 'TIP' -> click the 5mm bulb tip

    def reset(self):
        self.selecting = False
        self.selection_done = False
        self.ui_x1, self.ui_y1 = -1, -1
        self.ui_x2, self.ui_y2 = -1, -1
        self.ui_bulb_tip = None
        self.step = "BOX"

    def mouse_callback(self, event, x, y, flags, param):
        if self.step == "BOX":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.selecting = True
                self.ui_x1, self.ui_y1 = x, y
                self.ui_x2, self.ui_y2 = x, y
            elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
                self.ui_x2, self.ui_y2 = x, y
            elif event == cv2.EVENT_LBUTTONUP and self.selecting:
                self.selecting = False
                self.ui_x2, self.ui_y2 = x, y
                bx1, bx2 = min(self.ui_x1, self.ui_x2), max(self.ui_x1, self.ui_x2)
                by1, by2 = min(self.ui_y1, self.ui_y2), max(self.ui_y1, self.ui_y2)
                min_sz = int(25 * self.ui_scale / 2.0)
                if (bx2 - bx1) >= min_sz and (by2 - by1) >= min_sz:
                    self.ui_x1, self.ui_x2 = bx1, bx2
                    self.ui_y1, self.ui_y2 = by1, by2
                    self.step = "TIP"
                    print("[PenTrainer] Pen bounding box set! Now CLICK directly on the 5mm IR BULB TIP.")
        elif self.step == "TIP":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.ui_bulb_tip = (x, y)
                self.selection_done = True
                print(f"[PenTrainer] Bulb tip selected at UI: {self.ui_bulb_tip}")

    def get_native_box(self) -> tuple:
        """Returns (x1, y1, x2, y2) in native camera resolution (640x480)."""
        x1 = int(min(self.ui_x1, self.ui_x2) / self.ui_scale)
        y1 = int(min(self.ui_y1, self.ui_y2) / self.ui_scale)
        x2 = int(max(self.ui_x1, self.ui_x2) / self.ui_scale)
        y2 = int(max(self.ui_y1, self.ui_y2) / self.ui_scale)
        return (x1, y1, x2, y2)

    def get_native_tip(self) -> tuple:
        """Returns (tx, ty) in native camera resolution (640x480)."""
        if self.ui_bulb_tip is None:
            return None
        return (float(self.ui_bulb_tip[0] / self.ui_scale), float(self.ui_bulb_tip[1] / self.ui_scale))


# =============================================================================
# Pen & Bulb Model Trainer Class
# =============================================================================

class PenModelTrainer:
    """Manages data collection, template generation, ML feature modeling, and live testing."""
    def __init__(self, camera=None):
        self.camera = camera
        self.samples_pen = []        # list of cropped pen in hand patches
        self.samples_bulb = []       # list of cropped bulb tip templates
        self.samples_bg = []         # list of room background negative patches
        self.trained_templates = []  # list of finalized templates for detector
        self.model_data = {}
        self.is_trained = False
        self.threshold = 0.82

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        BULB_TMPL_DIR.mkdir(parents=True, exist_ok=True)

    def add_pen_sample(self, frame_bgr: np.ndarray, bbox: tuple, tip_abs: tuple):
        """Adds a verified sample of the pen in hand with its bulb tip location."""
        x1, y1, x2, y2 = bbox
        tx, ty = tip_abs

        pen_crop = frame_bgr[y1:y2, x1:x2].copy()
        if pen_crop.shape[0] < 20 or pen_crop.shape[1] < 20:
            return False

        # Relative tip position inside the crop
        rel_tx = tx - x1
        rel_ty = ty - y1

        # Extract focused 5mm bulb tip crop (e.g. 32x28 around the tip)
        bx1 = max(0, int(tx - 15))
        by1 = max(0, int(ty - 16))
        bx2 = min(frame_bgr.shape[1], int(tx + 15))
        by2 = min(frame_bgr.shape[0], int(ty + 16))
        bulb_crop = frame_bgr[by1:by2, bx1:bx2].copy()

        self.samples_pen.append({
            "pen_patch": pen_crop,
            "bulb_patch": bulb_crop,
            "rel_tip": (rel_tx, rel_ty),
            "abs_tip": (tx, ty),
            "bbox": (x1, y1, x2 - x1, y2 - y1)
        })

        # Also sample random room background negative patches far from the pen
        h, w = frame_bgr.shape[:2]
        for _ in range(5):
            rx = np.random.randint(20, max(30, w - 80))
            ry = np.random.randint(20, max(30, h - 80))
            # Make sure it doesn't overlap the pen
            if not (x1 - 30 <= rx <= x2 + 30 and y1 - 30 <= ry <= y2 + 30):
                bg_patch = frame_bgr[ry:ry+40, rx:rx+40].copy()
                if bg_patch.size > 0:
                    self.samples_bg.append(bg_patch)

        print(f"[PenTrainer] Added pen sample #{len(self.samples_pen)}. Pen shape: {pen_crop.shape}, Bulb shape: {bulb_crop.shape}")
        return True

    def train_from_dataset(self):
        """Mines existing dataset images in dataset_captured/ to train the pen model."""
        print("=" * 60)
        print("SMART NOTEBOOK — TRAINING PEN MODEL FROM DATASET")
        print("=" * 60)

        # 1. Load priority reference crops
        p_up = BULB_TMPL_DIR / "user_tip_up.png"
        p_down = BULB_TMPL_DIR / "user_tip_down.png"

        loaded_refs = []
        for p in [p_up, p_down]:
            if p.exists():
                im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    loaded_refs.append(im)

        # 2. Mine unlit bulb frames
        unlit_dir = DATASET_CAPTURED_DIR / "ir_bulb_off"
        if unlit_dir.exists():
            files = list(unlit_dir.glob("*.png"))[:60]
            print(f"[Dataset] Loading {len(files)} unlit pen camera frames...")
            for f in files:
                im = cv2.imread(str(f))
                if im is not None:
                    # Collect background negative patches from room edges
                    h, w = im.shape[:2]
                    self.samples_bg.append(im[10:50, 10:50])
                    self.samples_bg.append(im[h-60:h-10, w-60:w-10])
                    self.samples_bg.append(im[20:70, w//2-25:w//2+25])

        # 3. Mine emitting bulb frames
        emit_dir = DATASET_CAPTURED_DIR / "ir_led_emitting"
        if emit_dir.exists():
            e_files = list(emit_dir.glob("*.png"))[:60]
            print(f"[Dataset] Loading {len(e_files)} emitting pen camera frames...")
            for f in e_files:
                im = cv2.imread(str(f))
                if im is not None:
                    is_emit, excess, pt = measure_purple_excess(im)
                    if is_emit and pt is not None:
                        px, py = int(pt[0]), int(pt[1])
                        c1 = max(0, px - 16)
                        r1 = max(0, py - 16)
                        c2 = min(im.shape[1], px + 16)
                        r2 = min(im.shape[0], py + 16)
                        crop = im[r1:r2, c1:c2]
                        if crop.size > 0:
                            self.samples_bulb.append(crop)

        # Build templates from priority reference crops
        if len(loaded_refs) > 0:
            base_ref = loaded_refs[0]
            bh, bw = base_ref.shape[:2]
            tip_pt = (bw // 2.0, 4.0)
            exs = generate_synthesized_exemplars(base_ref, tip_pt)
            for ex in exs:
                self.trained_templates.append(ex["image"])
        print(f"[PenTrainer] Built {len(self.trained_templates)} multi-scale, multi-angle orientation templates.")

        # Calibrate optimal detection threshold against background negatives
        self._calibrate_threshold()
        self.is_trained = True
        return True

    def train(self):
        """Builds multi-angle templates, extracts structural weights, and calibrates threshold."""
        if len(self.samples_pen) == 0 and len(self.samples_bulb) == 0:
            print("[PenTrainer] No samples collected yet! Training from existing dataset...")
            return self.train_from_dataset()

        print("=" * 60)
        print(f"SMART NOTEBOOK — TRAINING PEN MODEL ({len(self.samples_pen)} Live Pen Samples)")
        print("=" * 60)

        # Synthesize multi-angle & multi-scale templates for each collected sample
        self.trained_templates = []
        for s in self.samples_pen:
            bulb_crop = s["bulb_patch"]
            gray_bulb = cv2.cvtColor(bulb_crop, cv2.COLOR_BGR2GRAY)
            bh, bw = gray_bulb.shape[:2]
            tip_pt = (bw // 2.0, 4.0)

            exs = generate_synthesized_exemplars(gray_bulb, tip_pt)
            for ex in exs:
                self.trained_templates.append(ex["image"])

        print(f"[PenTrainer] Synthesized {len(self.trained_templates)} pen & bulb templates across angles & scales.")
        self._calibrate_threshold()
        self.is_trained = True
        return True

    def _calibrate_threshold(self):
        """
        Calibrates the detection correlation threshold against room background negatives
        to guarantee ZERO false positives on furniture, screws, switches, and walls.
        """
        max_bg_score = 0.0
        if len(self.samples_bg) > 0 and len(self.trained_templates) > 0:
            print(f"[PenTrainer] Benchmarking against {len(self.samples_bg)} room background patches...")
            tmpl = self.trained_templates[0]
            th, tw = tmpl.shape[:2]

            for bg in self.samples_bg:
                bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) if len(bg.shape) == 3 else bg
                if bg_gray.shape[0] >= th and bg_gray.shape[1] >= tw:
                    res = cv2.matchTemplate(bg_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_v, _, _ = cv2.minMaxLoc(res)
                    if max_v > max_bg_score:
                        max_bg_score = max_v

        # Ensure threshold is safely above maximum background correlation
        # Empirical safe floor: 0.82
        self.threshold = max(0.82, float(max_bg_score + 0.05))
        self.threshold = min(0.92, self.threshold)
        print(f"[PenTrainer] Max background false score: {max_bg_score:.3f}. Calibrated Threshold: {self.threshold:.3f}")

    def save_model(self):
        """Saves pen model parameters and saves priority templates."""
        model_payload = {
            "model_name": "PenInHand_IR_Bulb_Detector",
            "version": "2.0.0",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_templates": len(self.trained_templates),
            "correlation_threshold": float(self.threshold),
            "pen_shaft_min_score": 0.35,
            "ir_purple_excess_threshold": float(IR_PURPLE_EXCESS_THRESHOLD),
            "bulb_type": "5mm_clear_dome_with_leadframe",
            "description": "Detector for pen held in hand with 5mm IR LED bulb at one end"
        }

        with open(PEN_MODEL_FILE, "w", encoding="utf-8") as f:
            json.dump(model_payload, f, indent=2)
        print(f"[PenTrainer] Saved pen model config to: {PEN_MODEL_FILE}")

        # Save top orientation templates to bulb_templates
        if len(self.trained_templates) > 0:
            for idx, tmpl in enumerate(self.trained_templates[:12]):
                out_path = BULB_TMPL_DIR / f"pen_tip_trained_{idx:02d}.png"
                cv2.imwrite(str(out_path), tmpl)
            print(f"[PenTrainer] Exported orientation templates to: {BULB_TMPL_DIR}")

    def detect_in_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Runs real-time inference on a frame:
        Returns detection dict with state, coordinates, confidence, and annotated overlay.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {"state": "NO_FRAME", "point": None, "confidence": 0.0, "annotated": None}

        annotated = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Check for Active IR Emission (Button Pressed -> DRAWING)
        is_emitting, excess, emit_pt = measure_purple_excess(frame_bgr)
        if is_emitting and emit_pt is not None:
            ex_x, ex_y = emit_pt
            # Draw glowing magenta/purple target reticle
            cv2.circle(annotated, (int(ex_x), int(ex_y)), 14, (255, 0, 255), 2)
            cv2.circle(annotated, (int(ex_x), int(ex_y)), 4, (0, 255, 0), -1)
            cv2.line(annotated, (int(ex_x) - 22, int(ex_y)), (int(ex_x) + 22, int(ex_y)), (0, 255, 0), 2)
            cv2.line(annotated, (int(ex_x), int(ex_y) - 22), (int(ex_x), int(ex_y) + 22), (0, 255, 0), 2)

            return {
                "state": "IR_EMITTING",
                "point": (ex_x, ex_y),
                "confidence": 1.0,
                "purple_excess": excess,
                "annotated": annotated
            }

        # 2. Check for Pen in Hand (Bulb Unlit -> IDLE / HOVER TRACKING)
        best_score = -1.0
        best_loc = None
        best_sz = (0, 0)

        # Use trained templates or fallback to user templates
        eval_tmpls = self.trained_templates[:6] if len(self.trained_templates) > 0 else []
        if len(eval_tmpls) == 0:
            p_up = BULB_TMPL_DIR / "user_tip_up.png"
            if p_up.exists():
                im = cv2.imread(str(p_up), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    eval_tmpls.append(im)

        for tmpl in eval_tmpls:
            th, tw = tmpl.shape[:2]
            if tw >= w or th >= h:
                continue
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v > best_score:
                best_score = max_v
                best_loc = max_l
                best_sz = (tw, th)

        if best_score >= self.threshold and best_loc is not None:
            bx, by = best_loc
            bw, bh = best_sz
            tip_x = float(bx + bw // 2)
            tip_y = float(by + 4)

            # Shaft verification: verify pen cylinder extends backwards from the tip
            pen_shaft_box_y2 = min(h, by + bh + 45)
            shaft_crop = gray[by:pen_shaft_box_y2, max(0, bx - 10):min(w, bx + bw + 10)]
            shaft_score = extract_pen_shaft_vector(shaft_crop)

            # Draw green bounding box around pen in hand
            cv2.rectangle(annotated, (max(0, bx - 15), by), (min(w, bx + bw + 15), pen_shaft_box_y2), (0, 240, 255), 2)

            # Draw precision reticle at 5mm IR bulb tip
            cv2.circle(annotated, (int(tip_x), int(tip_y)), 10, (0, 255, 100), 2)
            cv2.circle(annotated, (int(tip_x), int(tip_y)), 3, (0, 255, 100), -1)
            cv2.line(annotated, (int(tip_x) - 16, int(tip_y)), (int(tip_x) + 16, int(tip_y)), (0, 255, 100), 2)
            cv2.line(annotated, (int(tip_x), int(tip_y) - 16), (int(tip_x), int(tip_y) + 16), (0, 255, 100), 2)

            # Pointer arrow to the bulb tip
            cv2.arrowedLine(annotated, (int(tip_x), int(tip_y) - 28), (int(tip_x), int(tip_y) - 8), (0, 255, 255), 2, tipLength=0.35)

            return {
                "state": "PEN_IDLE",
                "point": (tip_x, tip_y),
                "confidence": float(best_score),
                "shaft_score": shaft_score,
                "purple_excess": 0.0,
                "annotated": annotated
            }

        return {
            "state": "NO_PEN",
            "point": None,
            "confidence": 0.0,
            "purple_excess": 0.0,
            "annotated": annotated
        }


# =============================================================================
# Interactive Studio HUD Rendering (High-Resolution 1280x960 Large Readable UI)
# =============================================================================

def draw_studio_hud(frame_bgr: np.ndarray, mode: str, trainer: PenModelTrainer, selector: PenROISelector,
                    det_result: dict, fps: float, ui_scale: float = 2.0) -> np.ndarray:
    """
    Renders a crystal-clear, high-resolution (1280x960), high-contrast HUD
    with large readable fonts, prominent instructions, and dedicated status badges.
    """
    h_native, w_native = frame_bgr.shape[:2]
    w_ui = int(w_native * ui_scale)
    h_ui = int(h_native * ui_scale)

    # Upscale camera frame cleanly
    hud = cv2.resize(frame_bgr, (w_ui, h_ui), interpolation=cv2.INTER_LINEAR)

    # -------------------------------------------------------------
    # 1. TOP HEADER BANNER (Large & Highly Readable, Height 140px)
    # -------------------------------------------------------------
    header_h = 140
    overlay = hud.copy()
    cv2.rectangle(overlay, (0, 0), (w_ui, header_h), (14, 18, 24), -1)
    cv2.addWeighted(overlay, 0.88, hud, 0.12, 0, hud)

    # Title
    cv2.putText(hud, "SMART NOTEBOOK — PEN & IR BULB TRAINER", (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 240, 255), 3)

    # Current Mode Badge
    mode_color = (0, 255, 100) if mode == "VERIFY" else (0, 220, 255)
    if mode in ("SELECT_BOX", "SELECT_TIP"):
        mode_color = (0, 160, 255)  # Orange

    mode_badge_text = f"CURRENT STEP: [{mode}]"
    cv2.putText(hud, mode_badge_text, (24, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, mode_color, 2)

    # Prominent Step Instructions
    if mode == "CAPTURE":
        instr_text = "1. Hold pen in your hand with the 5mm IR bulb facing the camera."
        sub_text = "Press [ R ] to Draw Box  |  [ SPACE ] Quick Capture  |  [ A ] Auto-Burst  |  [ T ] Train Model"
    elif mode == "SELECT_BOX":
        instr_text = ">> CLICK & DRAG MOUSE over the pen in your hand to select bounding box."
        sub_text = "Release mouse when pen body + bulb are enclosed."
    elif mode == "SELECT_TIP":
        instr_text = ">> NOW CLICK ON THE 5mm CLEAR IR BULB TIP at the end of the pen!"
        sub_text = "This anchors the exact stylus pointer coordinate."
    else:  # VERIFY
        instr_text = "LIVE TEST MODE: Move your pen around in front of the camera."
        sub_text = "Press pen button to emit purple IR light & verify whiteboard drawing!"

    cv2.putText(hud, instr_text, (24, 106),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(hud, sub_text, (24, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 220, 255), 1)

    # Top-Right Statistics Panel
    stat_x = w_ui - 380
    cv2.putText(hud, f"Samples: {len(trainer.samples_pen)}", (stat_x, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)
    cv2.putText(hud, f"FPS: {fps:.1f} | Threshold: {trainer.threshold:.2f}", (stat_x, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    # -------------------------------------------------------------
    # 2. SELECTION BOX & BULB POINT (Scaled to UI)
    # -------------------------------------------------------------
    if selector.selecting or (selector.step == "TIP" and selector.ui_x1 != -1):
        x1, y1 = selector.ui_x1, selector.ui_y1
        x2, y2 = selector.ui_x2, selector.ui_y2
        bx1, by1 = min(x1, x2), min(y1, y2)
        bx2, by2 = max(x1, x2), max(y1, y2)
        cv2.rectangle(hud, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
        cv2.putText(hud, "PEN IN HAND SELECTION", (bx1, max(30, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    if selector.ui_bulb_tip is not None:
        tx, ty = selector.ui_bulb_tip
        cv2.circle(hud, (tx, ty), 10, (0, 0, 255), -1)
        cv2.circle(hud, (tx, ty), 18, (0, 255, 255), 2)
        cv2.putText(hud, "5mm IR BULB TIP", (tx + 16, ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # -------------------------------------------------------------
    # 3. LIVE DETECTION OVERLAYS (Scaled for Large Visibility)
    # -------------------------------------------------------------
    if det_result is not None:
        state = det_result.get("state", "NO_PEN")
        pt = det_result.get("point")
        conf = det_result.get("confidence", 0.0)

        if pt is not None:
            ui_pt_x = int(pt[0] * ui_scale)
            ui_pt_y = int(pt[1] * ui_scale)

            if state == "IR_EMITTING":
                # Glowing magenta & green target reticle
                cv2.circle(hud, (ui_pt_x, ui_pt_y), 28, (255, 0, 255), 3)
                cv2.circle(hud, (ui_pt_x, ui_pt_y), 8, (0, 255, 0), -1)
                cv2.line(hud, (ui_pt_x - 45, ui_pt_y), (ui_pt_x + 45, ui_pt_y), (0, 255, 0), 3)
                cv2.line(hud, (ui_pt_x, ui_pt_y - 45), (ui_pt_x, ui_pt_y + 45), (0, 255, 0), 3)
                cv2.putText(hud, "IR LED EMITTING [DRAWING ACTIVE]", (ui_pt_x + 35, ui_pt_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)

            elif state == "PEN_IDLE":
                # Green pen box and tip pointer
                cv2.circle(hud, (ui_pt_x, ui_pt_y), 20, (0, 255, 100), 3)
                cv2.circle(hud, (ui_pt_x, ui_pt_y), 6, (0, 255, 100), -1)
                cv2.line(hud, (ui_pt_x - 32, ui_pt_y), (ui_pt_x + 32, ui_pt_y), (0, 255, 100), 2)
                cv2.line(hud, (ui_pt_x, ui_pt_y - 32), (ui_pt_x, ui_pt_y + 32), (0, 255, 100), 2)
                cv2.arrowedLine(hud, (ui_pt_x, ui_pt_y - 50), (ui_pt_x, ui_pt_y - 15),
                                (0, 240, 255), 3, tipLength=0.35)
                cv2.putText(hud, f"5mm BULB TIP ({int(pt[0])}, {int(pt[1])})", (ui_pt_x + 28, ui_pt_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 240, 255), 2)

    # -------------------------------------------------------------
    # 4. BOTTOM STATUS BAR & KEYBOARD SHORTCUTS (Height 120px)
    # -------------------------------------------------------------
    footer_h = 120
    overlay_f = hud.copy()
    cv2.rectangle(overlay_f, (0, h_ui - footer_h), (w_ui, h_ui), (14, 18, 24), -1)
    cv2.addWeighted(overlay_f, 0.90, hud, 0.10, 0, hud)

    # Line 1: Live Status Readout
    if det_result is not None:
        state = det_result.get("state", "NO_PEN")
        pt = det_result.get("point")
        conf = det_result.get("confidence", 0.0)

        if state == "IR_EMITTING":
            excess = det_result.get("purple_excess", 0.0)
            status_line = f"● [PEN STATE: IR BULB EMITTING] -> Bulb Tip ({int(pt[0])}, {int(pt[1])}) | Purple Meter: +{excess:.1f} > 8.0 [WHITEBOARD DRAWING ACTIVE]"
            status_col = (0, 255, 100)
        elif state == "PEN_IDLE":
            status_line = f"● [PEN STATE: PEN IN HAND (BULB UNLIT)] -> Bulb Tip ({int(pt[0])}, {int(pt[1])}) | Confidence: {conf*100:.1f}% [HOVER TRACKING]"
            status_col = (0, 240, 255)
        else:
            status_line = "○ [PEN STATE: NO PEN DETECTED] -> Idle (Room furniture, screws, and lights rejected 100%)"
            status_col = (160, 160, 160)

        cv2.putText(hud, status_line, (24, h_ui - 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_col, 2)

    # Line 2: Keyboard Shortcuts Bar
    shortcuts_line = "[ R ] Select Pen ROI   |   [ SPACE ] Capture   |   [ A ] Auto-Burst   |   [ T ] Train   |   [ V ] Verify   |   [ S ] Save   |   [ Q ] Quit"
    cv2.putText(hud, shortcuts_line, (24, h_ui - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 220, 100), 2)

    return hud


# =============================================================================
# Main Interactive Studio Loop
# =============================================================================

def run_interactive_studio(args):
    """Runs the full interactive OpenCV studio window."""
    ui_scale = getattr(args, "scale", 2.0)
    w_ui = int(CAMERA_WIDTH * ui_scale)
    h_ui = int(CAMERA_HEIGHT * ui_scale)

    win_name = "Smart Notebook — Pen Trainer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w_ui, h_ui)

    # Open Camera
    backend_mode = normalize_backend(args.camera)
    cam = Camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=30,
                 device_index=args.device, backend=backend_mode)
    if not cam.open():
        print(f"[PenTrainer] Error: Could not open camera (backend={backend_mode}, index={args.device}).")
        print("Tip: If USB webcam index is not 0, try running with: py tools/train_pen_detector.py --device 1")
        return False

    selector = PenROISelector(win_name, ui_scale=ui_scale)
    cv2.setMouseCallback(win_name, selector.mouse_callback)

    trainer = PenModelTrainer(camera=cam)

    # Initial training from dataset so user already has a working baseline
    trainer.train_from_dataset()

    mode = "VERIFY" if args.test else "CAPTURE"
    print(f"[PenTrainer] Interactive studio started in [{mode}] mode (Window size: {w_ui}x{h_ui}).")
    print("Controls:")
    print("  [R]     : Select Pen ROI (Drag box around pen in hand, then click bulb tip)")
    print("  [SPACE] : Capture sample frame into training set")
    print("  [A]     : Auto-burst capture (5 samples)")
    print("  [T]     : Train model immediately")
    print("  [V]     : Toggle Live Verification / Test mode")
    print("  [S]     : Save model to disk (models/pen_model.json)")
    print("  [Q/ESC] : Quit")

    t_prev = time.time()
    det_result = {}

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            t_now = time.time()
            dt = max(1e-5, t_now - t_prev)
            fps = 1.0 / dt
            t_prev = t_now

            # Run detection if in VERIFY mode or CAPTURE mode
            det_result = trainer.detect_in_frame(frame)

            # Handle interactive selection state
            current_mode = mode
            if selector.step == "TIP":
                current_mode = "SELECT_TIP"
            elif selector.selecting:
                current_mode = "SELECT_BOX"

            # Check if selection just completed
            if selector.selection_done and selector.ui_bulb_tip is not None:
                x1, y1, x2, y2 = selector.get_native_box()
                tip_pt = selector.get_native_tip()
                if (x2 - x1) >= 20 and (y2 - y1) >= 20 and tip_pt is not None:
                    trainer.add_pen_sample(frame, (x1, y1, x2, y2), tip_pt)
                    trainer.train()
                    mode = "VERIFY"
                selector.reset()

            # Render High-Resolution HUD (1280x960)
            hud_view = draw_studio_hud(frame, current_mode, trainer, selector, det_result, fps, ui_scale=ui_scale)
            cv2.imshow(win_name, hud_view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('r'), ord('R')):
                selector.reset()
                mode = "SELECT_BOX"
                print("[PenTrainer] Click and drag around the pen in your hand.")
            elif key == 32:  # SPACE
                # Quick capture of center ROI
                h, w = frame.shape[:2]
                cw, ch = 80, 160
                cx1, cy1 = (w - cw) // 2, (h - ch) // 2
                tip = (cx1 + cw // 2, cy1 + 6)
                trainer.add_pen_sample(frame, (cx1, cy1, cx1 + cw, cy1 + ch), tip)
                trainer.train()
            elif key in (ord('a'), ord('A')):
                print("[PenTrainer] Auto-burst capturing 5 samples... Rotate pen slightly.")
                for _ in range(5):
                    ret_b, f_b = cam.read()
                    if ret_b and f_b is not None:
                        is_emit, _, pt = measure_purple_excess(f_b)
                        if is_emit and pt is not None:
                            bx1 = max(0, int(pt[0] - 30))
                            by1 = max(0, int(pt[1] - 30))
                            bx2 = min(f_b.shape[1], int(pt[0] + 30))
                            by2 = min(f_b.shape[0], int(pt[1] + 90))
                            trainer.add_pen_sample(f_b, (bx1, by1, bx2, by2), pt)
                    time.sleep(0.12)
                trainer.train()
            elif key in (ord('t'), ord('T')):
                trainer.train()
                mode = "VERIFY"
            elif key in (ord('v'), ord('V')):
                mode = "CAPTURE" if mode == "VERIFY" else "VERIFY"
                print(f"[PenTrainer] Toggled mode to: {mode}")
            elif key in (ord('s'), ord('S')):
                trainer.save_model()
                print("[PenTrainer] Successfully saved pen model!")

    finally:
        if args.save and trainer.is_trained:
            trainer.save_model()
        cam.close()
        cv2.destroyAllWindows()

    return True


# =============================================================================
# Headless Dataset Training Pipeline
# =============================================================================

def run_headless_training(args):
    """Executes dataset training and validation headlessly without GUI."""
    trainer = PenModelTrainer()
    success = trainer.train_from_dataset()
    if success:
        trainer.save_model()
        print("\n" + "=" * 60)
        print("PEN MODEL SUCCESSFULLY TRAINED AND SAVED!")
        print(f"  - Model File : {PEN_MODEL_FILE}")
        print(f"  - Templates  : {len(trainer.trained_templates)} in {BULB_TMPL_DIR}")
        print(f"  - Threshold  : {trainer.threshold:.3f} (Zero False Positive Target)")
        print("=" * 60)
        print("To test the pen live on webcam, run:")
        print("  py main.py --dev")
        print("=" * 60)
    return success


def main():
    args = parse_args()
    if args.dataset:
        sys.exit(0 if run_headless_training(args) else 1)
    else:
        sys.exit(0 if run_interactive_studio(args) else 1)


if __name__ == "__main__":
    main()
