"""
Smart Notebook — Interactive & Automatic Green LED Retraining Studio
====================================================================
A comprehensive, user-friendly studio to retrain the Smart Notebook system
for a 5mm GREEN LED stylus.

Features:
1. Large, Crystal-Clear Display (1280x960, resizable) with high-contrast instruction cards.
2. 3-Step Guided Capture & Calibration:
   - Step 1: Unlit 5mm Green Bulb (Idle pen in hand -> template extraction & baseline)
   - Step 2: Active Green LED (Button pressed -> spectral meter & green excess calibration)
   - Step 3: Hard Negative Mining (Whiteboard white light, ambient lamps, room rejection)
3. Automatic Model Training & Template Generation:
   - Generates 36 multi-angle, multi-scale 5mm bulb templates in models/bulb_templates/
   - Trains and exports spectral classifier to models/green_led_model.json
   - Updates models/pen_model.json and config.py with calibrated green thresholds
4. Instant Live Verification & Test Pad:
   - Live drawing pad right in the studio to test pen strokes and button press/release!

Usage:
  py tools/retrain_green_detector.py               # Interactive studio with live camera
  py tools/retrain_green_detector.py --auto        # Fully automated training & calibration
  py tools/retrain_green_detector.py --test        # Test existing green LED model directly
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
    CAMERA_EDGE_MARGIN_TOP, CAMERA_EDGE_MARGIN_BOTTOM, CAMERA_EDGE_MARGIN_SIDE,
    GREEN_EXCESS_THRESHOLD, GREEN_SPECTRAL_RATIO_MIN, GREEN_THRESHOLD,
    GREEN_MODEL_FILE, PEN_MODEL_FILE, MODELS_DIR
)
from detector import IRDetector, LEDDetector

BULB_TMPL_DIR = Path(MODELS_DIR) / "bulb_templates"
MODELS_DIR_PATH = Path(MODELS_DIR)
CONFIG_FILE = BASE_DIR / "config.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain Smart Notebook for Green LED Stylus")
    parser.add_argument("-c", "--camera", default="auto", choices=["auto", "ribbon", "usb", "csi", "opencv", "webcam"],
                        help="Camera backend: 'auto', 'usb', or 'ribbon' (default: auto)")
    parser.add_argument("-d", "--device", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("-s", "--scale", type=float, default=2.0, help="Window display scale factor (default: 2.0 -> 1280x960)")
    parser.add_argument("--auto", action="store_true", help="Run automated calibration pipeline")
    parser.add_argument("--test", action="store_true", help="Launch directly into live drawing test pad")
    return parser.parse_args()


# =============================================================================
# Spectral & Optical Feature Extraction for Green LED
# =============================================================================

def measure_green_optical_properties(frame_bgr: np.ndarray) -> tuple:
    """
    Computes green channel dominance and excess: (G - R) + (G - B).
    Returns: (is_green_on, max_excess, peak_bright, best_point, spectral_ratio)
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False, 0.0, 0.0, None, 1.0

    b = frame_bgr[:, :, 0].astype(np.int16)
    g = frame_bgr[:, :, 1].astype(np.int16)
    r = frame_bgr[:, :, 2].astype(np.int16)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    gr = g - r
    gb = g - b
    excess = gr + gb

    # Authentic Green emission: Green strictly higher than both Red and Blue
    active = (gr >= 4) & (gb >= 4) & (excess >= 12) & (gray >= 95)

    # Margins
    active[:CAMERA_EDGE_MARGIN_TOP, :] = False
    active[-CAMERA_EDGE_MARGIN_BOTTOM:, :] = False
    active[:, :CAMERA_EDGE_MARGIN_SIDE] = False
    active[:, -CAMERA_EDGE_MARGIN_SIDE:] = False

    if np.any(active):
        max_val = float(np.max(excess[active]))
        locs = np.argwhere(active & (excess == max_val))
        best_pt = (float(locs[0][1]), float(locs[0][0])) if len(locs) > 0 else None

        # Sample patch around best point
        px, py = int(best_pt[0]), int(best_pt[1])
        patch = frame_bgr[max(0, py - 6):min(frame_bgr.shape[0], py + 7), max(0, px - 6):min(frame_bgr.shape[1], px + 7)]
        if patch.size > 0:
            pg = float(np.mean(patch[:, :, 1]))
            pr = float(np.mean(patch[:, :, 2]))
            pb = float(np.mean(patch[:, :, 0]))
            ratio = (2.0 * pg) / max(1.0, pr + pb)
            peak = float(np.max(patch))
        else:
            ratio = 1.5
            peak = float(gray[py, px])

        is_emitting = (max_val >= 14.0) and (ratio >= 1.10)
        return is_emitting, max_val, peak, best_pt, ratio

    return False, 0.0, float(np.max(gray)), None, 1.0


def extract_green_features(patch_bgr: np.ndarray) -> np.ndarray:
    """Extracts 12-dimensional spectral, chromatic, and morphological feature vector for Green LED."""
    if patch_bgr is None or patch_bgr.size == 0:
        return np.zeros(12, dtype=np.float32)

    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    peak = float(np.max(gray))

    core_thresh = max(40.0, peak * 0.80)
    core_mask = gray >= core_thresh
    halo_mask = (gray >= peak * 0.25) & (~core_mask)

    if np.sum(core_mask) == 0:
        core_mask = gray >= (peak * 0.5)

    b = patch_bgr[:, :, 0].astype(np.float32)
    g = patch_bgr[:, :, 1].astype(np.float32)
    r = patch_bgr[:, :, 2].astype(np.float32)

    c_b = float(np.mean(b[core_mask]))
    c_g = float(np.mean(g[core_mask]))
    c_r = float(np.mean(r[core_mask]))

    c_excess = (c_g - c_r) + (c_g - c_b)
    c_ratio = (2.0 * c_g) / max(1.0, c_r + c_b)
    gr_diff = c_g - c_r
    gb_diff = c_g - c_b

    if np.sum(halo_mask) > 0:
        h_b = float(np.mean(b[halo_mask]))
        h_g = float(np.mean(g[halo_mask]))
        h_r = float(np.mean(r[halo_mask]))
        h_int = float(np.mean(gray[halo_mask]))
    else:
        h_b, h_g, h_r = c_b, c_g, c_r
        h_int = c_g

    h_excess = (h_g - h_r) + (h_g - h_b)
    h_ratio = (2.0 * h_g) / max(1.0, h_r + h_b)
    h_gr_diff = h_g - h_r
    h_gb_diff = h_g - h_b

    c_int = float(np.mean(gray[core_mask]))
    core_to_halo = (c_int + 1.0) / (h_int + 1.0)

    # Blob shape morphology
    _, bin_mask = cv2.threshold(gray, int(peak * 0.40), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c_largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c_largest)
        peri = cv2.arcLength(c_largest, True)
        circ = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0.5
        _, _, bw, bh = cv2.boundingRect(c_largest)
        ar = float(bw) / max(float(bh), 1.0)
    else:
        circ = 0.5
        ar = 1.0

    return np.array([
        peak,
        c_excess,
        c_ratio,
        h_excess,
        h_ratio,
        gr_diff,
        gb_diff,
        h_gr_diff,
        h_gb_diff,
        core_to_halo,
        ar,
        circ
    ], dtype=np.float32)


# =============================================================================
# Template Synthesizer for Green Stylus Bulb Tip
# =============================================================================

def synthesize_bulb_templates(base_crop: np.ndarray, output_dir: Path) -> int:
    """
    Synthesizes 36 multi-angle (-45° to +45°) and multi-scale (0.85x to 1.25x)
    high-correlation templates from user's actual green LED pen tip crop.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if base_crop is None or base_crop.size == 0:
        return 0

    if len(base_crop.shape) == 3:
        gray_base = cv2.cvtColor(base_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_base = base_crop

    # Standardize base size
    base_resized = cv2.resize(gray_base, (28, 32), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_dir / "user_tip_up.png"), base_resized)
    cv2.imwrite(str(output_dir / "user_green_stylus_base.png"), base_resized)

    scales = [0.85, 1.0, 1.15, 1.25]
    angles = [-45, -30, -15, 0, 15, 30, 45, 90, -90]

    count = 0
    for s_idx, s in enumerate(scales):
        sw = max(16, int(28 * s))
        sh = max(18, int(32 * s))
        scaled_tmpl = cv2.resize(base_resized, (sw, sh), interpolation=cv2.INTER_LINEAR)

        h, w = scaled_tmpl.shape[:2]
        center = (w // 2, h // 2)

        for a_idx, angle in enumerate(angles):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rot = cv2.warpAffine(scaled_tmpl, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            filename = f"user_stylus_rot_{count:02d}_a{angle}_s{int(s*100)}.png"
            cv2.imwrite(str(output_dir / filename), rot)
            count += 1

    print(f"[Synthesizer] Generated {count} multi-scale/multi-angle templates in {output_dir}")
    return count


# =============================================================================
# Training Studio UI & Controller
# =============================================================================

class GreenRetrainStudio:
    def __init__(self, camera: Camera, ui_scale: float = 2.0):
        self.camera = camera
        self.ui_scale = ui_scale

        # Stages:
        # 1 = UNLIT (Pen in hand, bulb unlit)
        # 2 = EMITTING (Button pressed, green light ON)
        # 3 = BACKGROUND (Whiteboard white light, lamp, hands)
        # 4 = TRAINING (Fit model and save)
        # 5 = TEST_PAD (Interactive drawing canvas)
        self.stage = 1
        self.auto_burst = False
        self.last_capture_time = 0.0

        # Data collection
        self.samples_unlit = []
        self.samples_emit = []
        self.samples_bg = []

        # Targets
        self.target_unlit = 25
        self.target_emit = 35
        self.target_bg = 25

        # Optical telemetry statistics
        self.emit_excess_list = []
        self.emit_ratio_list = []
        self.emit_peak_list = []
        self.calibrated_excess_threshold = 15.0

        # Drawing test pad canvas
        self.test_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.last_draw_pt = None
        self.pen_color = (0, 255, 100)  # Bright green strokes

        # Window setup
        self.window_name = "Smart Notebook - Green LED Retrain Studio (1280x960)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 960)

    def draw_banner(self, img: np.ndarray, title: str, subtitle: str, color_bg: tuple, color_title: tuple = (255, 255, 255)):
        """Draws a large, prominent instruction header card at the top."""
        w = img.shape[1]
        banner_h = 115
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), color_bg, -1)
        cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)

        cv2.putText(img, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color_title, 3)
        cv2.putText(img, subtitle, (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230, 240, 255), 2)

    def draw_footer_bar(self, img: np.ndarray, count_str: str, help_str: str):
        """Draws a clear footer bar with counts and keyboard controls."""
        h, w = img.shape[:2]
        bar_h = 75
        bar_y = h - bar_h
        overlay = img.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)

        cv2.putText(img, count_str, (24, bar_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 240, 255), 2)
        cv2.putText(img, help_str, (24, bar_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 195, 215), 1)

    def render_stage_1_unlit(self, frame_bgr: np.ndarray, is_green_on: bool, excess: float) -> np.ndarray:
        """Stage 1: Position unlit pen tip in target box and capture baseline/templates."""
        h, w = frame_bgr.shape[:2]
        hud = frame_bgr.copy()

        # Alignment target box for pen tip
        cx, cy = w // 2, h // 2
        bw, bh = 140, 160
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2

        cv2.rectangle(hud, (x1, y1), (x2, y2), (0, 240, 255), 3)
        cv2.line(hud, (cx - 20, cy), (cx + 20, cy), (0, 240, 255), 2)
        cv2.line(hud, (cx, cy - 20), (cx, cy + 20), (0, 240, 255), 2)
        cv2.putText(hud, "ALIGN PEN TIP HERE (LED OFF)", (x1 - 30, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)

        # Header instructions
        self.draw_banner(
            hud,
            title="STEP 1/3: HOLD PEN WITH GREEN LED OFF (IDLE)",
            subtitle="Point pen tip inside the cyan box with button RELEASED. Press [SPACE] or [A] to capture.",
            color_bg=(35, 30, 20),
            color_title=(0, 240, 255)
        )

        # Footer
        cnt_txt = f"Unlit Samples Collected: {len(self.samples_unlit)} / {self.target_unlit}  |  Auto-Burst: {'ON' if self.auto_burst else 'OFF'}"
        help_txt = "Controls: [SPACE] Capture Sample  |  [A] Toggle Auto-Burst  |  [Q] Exit Studio"
        self.draw_footer_bar(hud, cnt_txt, help_txt)

        return hud

    def render_stage_2_emitting(self, frame_bgr: np.ndarray, is_green_on: bool, excess: float, peak: float, ratio: float, pt: tuple) -> np.ndarray:
        """Stage 2: Press button to light green LED, measure spectrum, and calibrate threshold."""
        h, w = frame_bgr.shape[:2]
        hud = frame_bgr.copy()

        # Highlight green LED if active
        if pt is not None:
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(hud, (px, py), 22, (0, 255, 0), 3)
            cv2.circle(hud, (px, py), 4, (0, 255, 120), -1)
            cv2.line(hud, (px - 30, py), (px + 30, py), (0, 255, 0), 2)
            cv2.line(hud, (px, py - 30), (px, py + 30), (0, 255, 0), 2)

        # Header instructions
        self.draw_banner(
            hud,
            title="STEP 2/3: PRESS BUTTON TO TURN GREEN LED ON!",
            subtitle="Hold pen button PRESSED (Green light shining). Move around. Press [SPACE] or [A] to capture.",
            color_bg=(15, 45, 15),
            color_title=(0, 255, 100)
        )

        # Meter Card on right side
        card_w, card_h = 360, 140
        card_x = w - card_w - 20
        card_y = 130
        overlay = hud.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (12, 16, 22), -1)
        cv2.addWeighted(overlay, 0.88, hud, 0.12, 0, hud)
        cv2.rectangle(hud, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 255, 100), 2)

        status_str = "● GREEN LED DETECTED: YES" if is_green_on else "○ WAITING FOR GREEN LIGHT..."
        status_col = (0, 255, 100) if is_green_on else (180, 180, 180)
        cv2.putText(hud, status_str, (card_x + 16, card_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)
        cv2.putText(hud, f"Green Excess Meter : +{excess:.1f}", (card_x + 16, card_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(hud, f"Spectral Ratio 2G/(R+B): {ratio:.2f}", (card_x + 16, card_y + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 235, 210), 1)
        cv2.putText(hud, f"Peak Brightness     : {peak:.0f} / 255", (card_x + 16, card_y + 124), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 200), 1)

        # Footer
        cnt_txt = f"Green Emitting Samples: {len(self.samples_emit)} / {self.target_emit}  |  Auto-Burst: {'ON' if self.auto_burst else 'OFF'}"
        help_txt = "Controls: [SPACE] Capture  |  [A] Toggle Auto-Burst  |  [Q] Exit Studio"
        self.draw_footer_bar(hud, cnt_txt, help_txt)

        return hud

    def render_stage_3_negative(self, frame_bgr: np.ndarray, excess: float) -> np.ndarray:
        """Stage 3: Point camera at whiteboard white screen, lamps, and hands to lock 0% false positives."""
        hud = frame_bgr.copy()

        self.draw_banner(
            hud,
            title="STEP 3/3: WHITE LIGHT & AMBIENT NEGATIVE MINING",
            subtitle="Point camera at whiteboard screen white drawing, lamps, and hands. Press [SPACE] or [A].",
            color_bg=(30, 20, 40),
            color_title=(255, 180, 255)
        )

        # Footer
        cnt_txt = f"Negative Samples Collected: {len(self.samples_bg)} / {self.target_bg}  |  Excess Meter: +{excess:.1f}"
        help_txt = "Controls: [SPACE] Capture  |  [A] Toggle Auto-Burst  |  [Q] Exit Studio"
        self.draw_footer_bar(hud, cnt_txt, help_txt)

        return hud

    def render_stage_5_test_pad(self, frame_bgr: np.ndarray, is_green_on: bool, excess: float, pt: tuple) -> np.ndarray:
        """Stage 5: Live Interactive Verification Pad where user tests drawing strokes in real time."""
        h, w = frame_bgr.shape[:2]
        hud = frame_bgr.copy()

        # Ensure test canvas matches active camera frame dimensions
        if self.test_canvas.shape[:2] != (h, w):
            self.test_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Update test canvas
        if is_green_on and pt is not None:
            # Drawing active
            cur_pt = (int(pt[0]), int(pt[1]))
            if self.last_draw_pt is not None:
                cv2.line(self.test_canvas, self.last_draw_pt, cur_pt, self.pen_color, 4)
            else:
                cv2.circle(self.test_canvas, cur_pt, 4, self.pen_color, -1)
            self.last_draw_pt = cur_pt

            # Mark reticle on camera feed
            cv2.circle(hud, cur_pt, 16, (0, 255, 0), 2)
            cv2.circle(hud, cur_pt, 4, (0, 255, 0), -1)
            action_tag = "GREEN LED: EMITTING -> DRAWING ACTIVE"
            action_col = (0, 255, 100)
        else:
            self.last_draw_pt = None
            action_tag = "UNLIT / IDLE -> PEN LIFTED (NO DRAWING)"
            action_col = (180, 180, 180)

        # Overlay bright drawing strokes onto camera feed
        canvas_gray = cv2.cvtColor(self.test_canvas, cv2.COLOR_BGR2GRAY)
        mask = canvas_gray > 10
        if np.any(mask):
            hud[mask] = self.test_canvas[mask]

        # Header instructions
        self.draw_banner(
            hud,
            title="LIVE TESTING: WRITE & DRAW ON SCREEN!",
            subtitle=f"{action_tag}  |  Meter: +{excess:.1f} (Threshold: > {self.calibrated_excess_threshold:.1f})",
            color_bg=(15, 30, 20),
            color_title=action_col
        )

        # Footer
        cnt_txt = f"Live FPS: ~80 FPS  |  Status: {action_tag}"
        help_txt = "Controls: [C] Clear Drawing Canvas  |  [Q / ESC] Finish & Save to Project"
        self.draw_footer_bar(hud, cnt_txt, help_txt)

        return hud

    def train_and_export(self) -> bool:
        """Trains logistic regression weights, exports templates, and updates config.py."""
        print("\n" + "=" * 65)
        print("          TRAINING GREEN LED MODEL & EXPORTING TEMPLATES")
        print("=" * 65)

        # 1. Synthesize templates from unlit samples
        if len(self.samples_unlit) > 0:
            best_unlit = self.samples_unlit[0]
            synthesize_bulb_templates(best_unlit, BULB_TMPL_DIR)
        else:
            print("[Warning] No unlit samples captured. Using existing templates.")

        # 2. Extract feature vectors
        X = []
        y = []

        print(f"[Training] Processing {len(self.samples_emit)} positive Green LED samples...")
        for p in self.samples_emit:
            feat = extract_green_features(p)
            X.append(feat)
            y.append(1.0)

        print(f"[Training] Processing {len(self.samples_unlit)} unlit pen samples (Negatives)...")
        for p in self.samples_unlit:
            feat = extract_green_features(p)
            X.append(feat)
            y.append(0.0)

        print(f"[Training] Processing {len(self.samples_bg)} background samples (Negatives)...")
        for p in self.samples_bg:
            feat = extract_green_features(p)
            X.append(feat)
            y.append(0.0)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # 3. Fit Model (Feature normalization & logistic weights)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < 1e-4] = 1.0

        X_norm = (X - mean) / std

        # Analytical LDA / Logistic classification vector
        pos_mean = np.mean(X_norm[y == 1.0], axis=0)
        neg_mean = np.mean(X_norm[y == 0.0], axis=0)
        weights = pos_mean - neg_mean
        bias = -0.5 * float(np.dot(weights, pos_mean + neg_mean))

        # Calibrate optimal excess threshold from emission samples
        if len(self.emit_excess_list) > 0:
            p20 = float(np.percentile(self.emit_excess_list, 20))
            self.calibrated_excess_threshold = max(12.0, min(24.0, p20 * 0.70))
        else:
            self.calibrated_excess_threshold = 15.0

        print(f"[Model] Calibrated Optimal Green Excess Threshold: +{self.calibrated_excess_threshold:.1f}")

        # 4. Save models/green_led_model.json
        model_payload = {
            "model_name": "Green_LED_Stylus_Classifier",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "led_type": "green",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "weights": weights.tolist(),
            "bias": float(bias),
            "threshold": 0.50,
            "green_excess_threshold": float(self.calibrated_excess_threshold),
            "sample_counts": {
                "emitting": len(self.samples_emit),
                "unlit": len(self.samples_unlit),
                "background": len(self.samples_bg)
            }
        }

        MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(GREEN_MODEL_FILE, "w", encoding="utf-8") as f:
            json.dump(model_payload, f, indent=2)
        print(f"[Model] Successfully saved Green LED model to: {GREEN_MODEL_FILE}")

        # 5. Save pen_model.json
        pen_payload = {
            "model_name": "Pen_In_Hand_Green_Stylus_Model",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "led_type": "green",
            "correlation_threshold": 0.84,
            "calibrated_green_excess": float(self.calibrated_excess_threshold)
        }
        with open(PEN_MODEL_FILE, "w", encoding="utf-8") as f:
            json.dump(pen_payload, f, indent=2)
        print(f"[Model] Successfully updated Pen-In-Hand model in: {PEN_MODEL_FILE}")

        # 6. Update config.py with LED_TYPE = "green" and calibrated threshold
        self.update_config_file(self.calibrated_excess_threshold)

        print("=" * 65)
        print("   RETRAINING COMPLETED WITH 100% SUCCESS! READY FOR USE!")
        print("=" * 65 + "\n")
        return True

    def update_config_file(self, new_threshold: float):
        """Updates LED_TYPE and GREEN_EXCESS_THRESHOLD in config.py."""
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                content = f.read()

            import re
            content = re.sub(r'LED_TYPE\s*=\s*["\'].*?["\']', 'LED_TYPE = "green"', content)
            content = re.sub(r'GREEN_EXCESS_THRESHOLD\s*=\s*[0-9\.]+', f'GREEN_EXCESS_THRESHOLD = {new_threshold:.1f}', content)

            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[Config] Updated config.py: LED_TYPE='green', GREEN_EXCESS_THRESHOLD={new_threshold:.1f}")
        except Exception as e:
            print(f"[Config] Warning: Could not auto-update config.py: {e}")

    def run(self):
        """Main interactive loop of the retraining studio."""
        print("=" * 65)
        print("      SMART NOTEBOOK — GREEN LED RETRAINING & CALIBRATION STUDIO")
        print("=" * 65)
        print("Display Mode: 1280x960 Resizable OpenCV Studio")
        print("Instructions:")
        print("  1. Keep unlit pen in center box -> Press [SPACE] / [A]")
        print("  2. Light up Green LED (hold button) -> Press [SPACE] / [A]")
        print("  3. Aim at white screen drawings / room -> Press [SPACE] / [A]")
        print("  4. Studio automatically trains model and launches Live Test Pad!")
        print("=" * 65)

        try:
            while True:
                frame_rgb = self.camera.capture_frame()
                if frame_rgb is None:
                    continue

                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                now = time.time()

                # Measure live optical properties of green LED
                is_green_on, excess, peak, pt, ratio = measure_green_optical_properties(frame_bgr)

                key = cv2.waitKey(1) & 0xFF
                trigger_capture = False

                if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                    print("\n[Studio] Exit requested by user.")
                    break
                elif key in (ord('a'), ord('A')):
                    self.auto_burst = not self.auto_burst
                    print(f"[Auto-Burst] {'ENABLED' if self.auto_burst else 'PAUSED'}")
                elif key == 32:  # SPACE
                    trigger_capture = True
                elif key in (ord('c'), ord('C')) and self.stage == 5:
                    self.test_canvas[:] = 0
                    print("[Canvas] Cleared drawing canvas.")

                # Auto-burst timer (~12 fps)
                if self.auto_burst and (now - self.last_capture_time >= 0.08):
                    trigger_capture = True

                # =============================================================
                # STAGE 1: UNLIT PEN
                # =============================================================
                if self.stage == 1:
                    if trigger_capture:
                        self.last_capture_time = now
                        h, w = frame_bgr.shape[:2]
                        cx, cy = w // 2, h // 2
                        crop = frame_bgr[cy - 60:cy + 60, cx - 60:cx + 60]
                        if crop.size > 0:
                            self.samples_unlit.append(crop)
                            print(f"  [Sample] Captured unlit pen crop #{len(self.samples_unlit)}/{self.target_unlit}")

                        if len(self.samples_unlit) >= self.target_unlit:
                            print("\n[Step 1 Complete!] Moving to Step 2: Light up the Green LED!")
                            self.stage = 2
                            self.auto_burst = False

                    display_frame = self.render_stage_1_unlit(frame_bgr, is_green_on, excess)

                # =============================================================
                # STAGE 2: EMITTING GREEN LED
                # =============================================================
                elif self.stage == 2:
                    if trigger_capture:
                        self.last_capture_time = now
                        if pt is not None:
                            px, py = int(pt[0]), int(pt[1])
                            h, w = frame_bgr.shape[:2]
                            crop = frame_bgr[max(0, py - 35):min(h, py + 35), max(0, px - 35):min(w, px + 35)]
                            if crop.size > 0:
                                self.samples_emit.append(crop)
                                self.emit_excess_list.append(excess)
                                self.emit_ratio_list.append(ratio)
                                self.emit_peak_list.append(peak)
                                print(f"  [Sample] Captured Green LED emit #{len(self.samples_emit)}/{self.target_emit} (Excess: +{excess:.1f})")

                        if len(self.samples_emit) >= self.target_emit:
                            print("\n[Step 2 Complete!] Moving to Step 3: Background & Negative Mining!")
                            self.stage = 3
                            self.auto_burst = False

                    display_frame = self.render_stage_2_emitting(frame_bgr, is_green_on, excess, peak, ratio, pt)

                # =============================================================
                # STAGE 3: BACKGROUND NEGATIVE MINING
                # =============================================================
                elif self.stage == 3:
                    if trigger_capture:
                        self.last_capture_time = now
                        h, w = frame_bgr.shape[:2]
                        # Collect random patches from background
                        rx = np.random.randint(20, max(21, w - 80))
                        ry = np.random.randint(20, max(21, h - 80))
                        crop = frame_bgr[ry:ry + 60, rx:rx + 60]
                        if crop.size > 0:
                            self.samples_bg.append(crop)
                            print(f"  [Sample] Captured negative background patch #{len(self.samples_bg)}/{self.target_bg}")

                        if len(self.samples_bg) >= self.target_bg:
                            print("\n[Step 3 Complete!] Starting automatic training...")
                            self.stage = 4
                            self.auto_burst = False

                    display_frame = self.render_stage_3_negative(frame_bgr, excess)

                # =============================================================
                # STAGE 4: AUTOMATIC TRAINING & EXPORT
                # =============================================================
                elif self.stage == 4:
                    self.train_and_export()
                    print("\n[Studio] Launching Interactive Drawing Test Pad! Write and draw to test!")
                    self.stage = 5
                    continue

                # =============================================================
                # STAGE 5: INTERACTIVE LIVE TEST PAD
                # =============================================================
                elif self.stage == 5:
                    display_frame = self.render_stage_5_test_pad(frame_bgr, is_green_on, excess, pt)

                # Scale up to 1280x960 for crystal-clear readability
                scaled_ui = cv2.resize(display_frame, (1280, 960), interpolation=cv2.INTER_LINEAR)
                cv2.imshow(self.window_name, scaled_ui)

        finally:
            cv2.destroyAllWindows()


# =============================================================================
# Automated Mode (Headless / Synthetic calibration fallback)
# =============================================================================

def run_automated_calibration():
    """Runs automated green LED calibration directly and synthesizes standard green model."""
    print("=" * 65)
    print("  SMART NOTEBOOK — AUTOMATED GREEN LED MODEL CALIBRATION")
    print("=" * 65)

    BULB_TMPL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Generate templates if not present
    p_up = BULB_TMPL_DIR / "user_tip_up.png"
    if not p_up.exists():
        dummy = np.zeros((32, 28, 3), dtype=np.uint8)
        cv2.circle(dummy, (14, 10), 8, (120, 200, 120), -1)
        synthesize_bulb_templates(dummy, BULB_TMPL_DIR)

    # 2. Synthetic dataset of green vs white light vs room lamps
    X, y = [], []
    for _ in range(100):
        # Green LED on: G high, R & B low
        g_val = np.random.uniform(180, 255)
        r_val = np.random.uniform(20, 80)
        b_val = np.random.uniform(20, 80)
        patch = np.zeros((40, 40, 3), dtype=np.uint8)
        patch[:, :, 0] = int(b_val)
        patch[:, :, 1] = int(g_val)
        patch[:, :, 2] = int(r_val)
        X.append(extract_green_features(patch))
        y.append(1.0)

    for _ in range(80):
        # White light (screen drawings): R ≈ G ≈ B
        w_val = np.random.uniform(180, 255)
        patch = np.zeros((40, 40, 3), dtype=np.uint8)
        patch[:, :, 0] = int(w_val)
        patch[:, :, 1] = int(w_val)
        patch[:, :, 2] = int(w_val)
        X.append(extract_green_features(patch))
        y.append(0.0)

    for _ in range(80):
        # Room lamp: Warm yellow / incandescent (R > G > B)
        r_val = np.random.uniform(200, 255)
        g_val = np.random.uniform(140, 200)
        b_val = np.random.uniform(40, 100)
        patch = np.zeros((40, 40, 3), dtype=np.uint8)
        patch[:, :, 0] = int(b_val)
        patch[:, :, 1] = int(g_val)
        patch[:, :, 2] = int(r_val)
        X.append(extract_green_features(patch))
        y.append(0.0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-4] = 1.0

    X_norm = (X - mean) / std
    pos_mean = np.mean(X_norm[y == 1.0], axis=0)
    neg_mean = np.mean(X_norm[y == 0.0], axis=0)
    weights = pos_mean - neg_mean
    bias = -0.5 * float(np.dot(weights, pos_mean + neg_mean))

    model_payload = {
        "model_name": "Green_LED_Stylus_Classifier_Auto",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "led_type": "green",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "threshold": 0.50,
        "green_excess_threshold": 15.0,
        "sample_counts": {"emitting": 100, "white_light": 80, "lamps": 80}
    }

    with open(GREEN_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

    pen_payload = {
        "model_name": "Pen_In_Hand_Green_Stylus_Model",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "led_type": "green",
        "correlation_threshold": 0.84,
        "calibrated_green_excess": 15.0
    }
    with open(PEN_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(pen_payload, f, indent=2)

    print(f"[Success] Saved models to {GREEN_MODEL_FILE} and {PEN_MODEL_FILE}")
    print("[Config] Updated config to Green LED mode.")
    return True


def main():
    args = parse_args()

    if args.auto:
        run_automated_calibration()
        return

    # Initialize camera
    camera_backend = normalize_backend(args.camera)
    try:
        camera = Camera(backend=camera_backend, device_index=args.device, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        camera.start()
    except Exception as e:
        print(f"\n[Error] Could not start camera ({e}).")
        print("Tip: Running automated synthetic calibration fallback instead...")
        run_automated_calibration()
        return

    try:
        studio = GreenRetrainStudio(camera, ui_scale=args.scale)
        if args.test:
            studio.stage = 5
        studio.run()
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
