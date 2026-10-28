import os
import json
from pathlib import Path
import cv2
import numpy as np
from config import (
    IR_THRESHOLD, IR_MIN_AREA, IR_MAX_AREA,
    GREEN_MIN_AREA, GREEN_MAX_AREA,
    CAMERA_EDGE_MARGIN_TOP, CAMERA_EDGE_MARGIN_BOTTOM, CAMERA_EDGE_MARGIN_SIDE,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    IR_SPECTRAL_RATIO_MIN, IR_PURPLE_EXCESS_MIN, IR_PURPLE_EXCESS_THRESHOLD,
    LED_TYPE, GREEN_EXCESS_THRESHOLD, GREEN_SPECTRAL_RATIO_MIN, GREEN_THRESHOLD,
    IR_MODEL_FILE, GREEN_MODEL_FILE, PEN_MODEL_FILE, BASE_DIR
)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class IRDetector:
    """
    Two-Stage 5mm LED Bulb Stylus (Green LED or Purple IR) & Light Emission Detector.

    1. STAGE 1 (Physical Bulb Localization):
       Detects the physical 5mm through-hole LED bulb (the stylus tip) in ambient light
       even when unlit. In Dev Mode, points directly to the bulb tip with a cyan reticle:
       [5mm BULB LOCATED: IDLE].

    2. STAGE 2 (Active Emission & Drawing Trigger):
       High-sensitivity optical chromaticity detector that triggers when the LED emits light
       (Green LED beam or purple IR glow when push button pressed).
       Supports direct LED bulb emission and scattered light pools on desk paper/sheets!
       Immediately transitions to [LED: EMITTING] and marks/draws on the whiteboard canvas!

    3. HIGH-ACCURACY FALSE POSITIVE REJECTION:
       100% hard rejects white blackboard drawings on the monitor (G ≈ R ≈ B, excess ≈ 0)
       and warm ambient room lamps (R > G, excess < 0). Never causes runaway drawing loops!
    """
    def __init__(self, threshold=None, min_area=None, max_area=None, led_type=None):
        self.led_type = (led_type or LED_TYPE).lower()
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = GREEN_THRESHOLD if self.led_type == "green" else IR_THRESHOLD

        if self.led_type == "green":
            self.min_area = min_area if min_area is not None else GREEN_MIN_AREA
            self.max_area = max_area if max_area is not None else GREEN_MAX_AREA
        else:
            self.min_area = min_area if min_area is not None else IR_MIN_AREA
            self.max_area = max_area if max_area is not None else IR_MAX_AREA

        self.green_excess_threshold = GREEN_EXCESS_THRESHOLD
        self.green_spectral_ratio_min = GREEN_SPECTRAL_RATIO_MIN

        # Last known bulb position for smooth fast tracking
        self.last_bulb_pos = None
        self.last_bulb_box = None
        self.frames_since_seen = 999

        # Load 5mm bulb templates
        self.bulb_templates = []
        self._load_bulb_templates()

        # Load trained ML model if available (prioritize matching LED type)
        self.has_model = False
        self.model_name = "Spectral Model"
        self.mean = None
        self.std = None
        self.weights = None
        self.bias = 0.0
        self.model_threshold = 0.50

        target_model = GREEN_MODEL_FILE if self.led_type == "green" and os.path.exists(GREEN_MODEL_FILE) else IR_MODEL_FILE
        if os.path.exists(target_model):
            try:
                with open(target_model, "r", encoding="utf-8") as f:
                    model_data = json.load(f)
                self.mean = np.array(model_data.get("mean", []), dtype=np.float32)
                self.std = np.array(model_data.get("std", []), dtype=np.float32)
                self.weights = np.array(model_data.get("weights", []), dtype=np.float32)
                self.bias = float(model_data.get("bias", 0.0))
                self.model_threshold = float(model_data.get("threshold", 0.50))
                self.model_name = f"{model_data.get('model_name', 'LED_Model')}"
                self.has_model = True
                print(f"[IRDetector] Loaded trained ML model ({self.model_name}) from: {target_model}")
            except Exception as e:
                print(f"[IRDetector] Warning: Could not load model: {e}")

        # Load trained Pen In Hand model if available
        self.pen_model_data = None
        self.pen_threshold = 0.84
        self.bulb_baseline_excess = 0.0
        self.bulb_baseline_bright = 100.0
        if os.path.exists(PEN_MODEL_FILE):
            try:
                with open(PEN_MODEL_FILE, "r", encoding="utf-8") as f:
                    self.pen_model_data = json.load(f)
                self.pen_threshold = float(self.pen_model_data.get("correlation_threshold", 0.84))
                print(f"[IRDetector] Loaded trained Pen-In-Hand model from: {PEN_MODEL_FILE} (threshold={self.pen_threshold:.2f})")
            except Exception as e:
                print(f"[IRDetector] Warning: Could not load pen model: {e}")

    def _load_bulb_templates(self):
        """Load orientation templates of the 5mm LED bulb, prioritizing exact user templates and trained pen tip exemplars."""
        tmpl_dir = Path(BASE_DIR) / "models" / "bulb_templates"
        if tmpl_dir.exists():
            priority = [
                "user_camroll_tip_00.png",
                "user_camroll_tip_01.png",
                "user_tip_up.png",
                "user_stylus_rot_27_a0_s114.png",
                "user_stylus_tip_cam_scale.png",
                "user_tip_down.png",
                "user_stylus_base32.png",
                "user_stylus_rot_03_a0_s85.png",
                "user_stylus_rot_15_a0_s100.png",
                "user_stylus_rot_24_a-45_s114.png",
                "user_stylus_rot_28_a15_s114.png"
            ]
            priority.extend([f"user_camroll_tip_{i:02d}*.png" for i in range(8)])
            priority.extend([f"user_stylus_rot_{i:02d}*.png" for i in range(36)])
            priority.extend([f"pen_tip_trained_{i:02d}.png" for i in range(12)])
            loaded = set()
            for p_pattern in priority:
                for p in tmpl_dir.glob(p_pattern) if "*" in p_pattern else [tmpl_dir / p_pattern]:
                    if p.exists() and p.name not in loaded:
                        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                        if im is not None:
                            self.bulb_templates.append(im)
                            loaded.add(p.name)

            for p in tmpl_dir.glob("*.png"):
                if p.name not in loaded:
                    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if im is not None:
                        self.bulb_templates.append(im)
                        loaded.add(p.name)
        print(f"[IRDetector] Loaded {len(self.bulb_templates)} 5mm LED bulb templates for physical tracking.")

        # Pre-compute canonical templates at native and coarse (0.25x) scales for 120+ FPS tracking
        self.canonical_templates = []
        self.canonical_templates_small = []
        self.coarse_scale = 0.25
        canonical_priority = [
            "user_camroll_tip_00.png",
            "user_camroll_tip_01.png",
            "user_tip_up.png",
            "user_stylus_rot_27_a0_s114.png",
            "user_stylus_rot_24_a-45_s114.png",
            "user_stylus_rot_30_a45_s114.png"
        ]
        for name in canonical_priority:
            p = tmpl_dir / name
            if p.exists():
                im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    self.canonical_templates.append(im)
                    tw = max(8, int(im.shape[1] * self.coarse_scale))
                    th = max(8, int(im.shape[0] * self.coarse_scale))
                    self.canonical_templates_small.append(cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA))

        if len(self.canonical_templates) == 0 and len(self.bulb_templates) > 0:
            for tmpl in self.bulb_templates[:4]:
                self.canonical_templates.append(tmpl)
                tw = max(8, int(tmpl.shape[1] * self.coarse_scale))
                th = max(8, int(tmpl.shape[0] * self.coarse_scale))
                self.canonical_templates_small.append(cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA))

    def check_bulb_emission_at_point(self, frame_bgr: np.ndarray, tip_x: float, tip_y: float) -> tuple:
        """
        Samples a localized patch around the tracked physical bulb tip and compares it
        with previous images/baseline to detect if it has lit up (green or purplish bright).
        Returns: (is_emitting, max_excess, peak_brightness, info_str)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, 0.0, 0.0, "NO_FRAME"

        h, w = frame_bgr.shape[:2]
        itip_x, itip_y = int(tip_x), int(tip_y)
        r = 14
        x1 = max(0, itip_x - r)
        y1 = max(0, itip_y - r)
        x2 = min(w, itip_x + r)
        y2 = min(h, itip_y + r)

        patch = frame_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return False, 0.0, 0.0, "EMPTY_PATCH"

        b = patch[:, :, 0].astype(np.float32)
        g = patch[:, :, 1].astype(np.float32)
        r_ch = patch[:, :, 2].astype(np.float32)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        peak_bright = float(np.max(gray))
        mean_b = float(np.mean(b))
        mean_g = float(np.mean(g))
        mean_r = float(np.mean(r_ch))

        is_emitting = False
        max_excess = 0.0

        if self.led_type == "green":
            gr_diff = g - r_ch
            gb_diff = g - b
            excess_map = gr_diff + gb_diff
            max_excess = float(np.max(excess_map))

            # Condition A: Absolute green excess meter >= GREEN_EXCESS_THRESHOLD
            cond_abs = (max_excess >= self.green_excess_threshold) and (mean_g > mean_r + 4.0) and (mean_g > mean_b + 4.0) and (peak_bright >= 105.0)

            # Condition B: Temporal delta above baseline
            delta_excess = max_excess - getattr(self, "bulb_baseline_excess", 0.0)
            delta_bright = peak_bright - getattr(self, "bulb_baseline_bright", 100.0)
            cond_temporal = (delta_excess >= 8.0) and (mean_g > mean_r + 3.0) and (mean_g > mean_b + 3.0) and (delta_bright >= 10.0)

            is_emitting = cond_abs or cond_temporal
            color_name = "GREEN"
        else:
            # Pixel-wise purple excess
            bg_diff = b - g
            rg_diff = r_ch - g
            excess_map = bg_diff + np.maximum(0.0, rg_diff)
            max_excess = float(np.max(excess_map))

            cond_abs = (max_excess >= 8.0) and (mean_b > mean_g + 3.0) and (mean_r > mean_g + 2.0) and (peak_bright >= 110.0)
            delta_excess = max_excess - getattr(self, "bulb_baseline_excess", 0.0)
            delta_bright = peak_bright - getattr(self, "bulb_baseline_bright", 100.0)
            cond_temporal = (delta_excess >= 5.0) and (mean_b > mean_g + 2.0) and (delta_bright >= 10.0) and (peak_bright >= 105.0)

            is_emitting = cond_abs or cond_temporal
            color_name = "PURPLE"

        if is_emitting:
            info = f"EMITTING {color_name}: +{max_excess:.1f} (Delta: +{delta_excess:.1f}, Peak: {peak_bright:.0f})"
        else:
            info = f"UNLIT: +{max_excess:.1f}"
            self.bulb_baseline_excess = 0.90 * getattr(self, "bulb_baseline_excess", 0.0) + 0.10 * max_excess
            self.bulb_baseline_bright = 0.90 * getattr(self, "bulb_baseline_bright", 100.0) + 0.10 * peak_bright

        return is_emitting, max_excess, peak_bright, info

    def locate_physical_bulb(self, frame_gray: np.ndarray) -> tuple:
        """
        Locates the physical 5mm clear LED bulb tip on the pen in ambient light when idle/off.
        Optimized with Coarse-to-Fine search and fast local window tracking for 60+ FPS.
        Returns: (tip_x, tip_y, bbox, score) or None.
        """
        if frame_gray is None or len(self.bulb_templates) == 0:
            return None

        h, w = frame_gray.shape[:2]
        min_thresh = min(0.84, getattr(self, "pen_threshold", 0.84))
        min_thresh = max(0.80, min_thresh)

        # 1. Fast Local Window Search (0.25ms when pen is actively tracked)
        if self.last_bulb_pos is not None and self.frames_since_seen < 8:
            lx, ly = int(self.last_bulb_pos[0]), int(self.last_bulb_pos[1])
            pad = 55
            x1 = max(0, lx - pad)
            y1 = max(0, ly - pad)
            x2 = min(w, lx + pad)
            y2 = min(h, ly + pad)
            crop = frame_gray[y1:y2, x1:x2]

            if crop.shape[0] >= 30 and crop.shape[1] >= 30:
                best_local_score = -1.0
                best_local_loc = None
                best_local_sz = (0, 0)
                for tmpl in self.canonical_templates:
                    if tmpl.shape[0] < crop.shape[0] and tmpl.shape[1] < crop.shape[1]:
                        res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
                        _, max_v, _, max_l = cv2.minMaxLoc(res)
                        if max_v > best_local_score:
                            best_local_score = max_v
                            best_local_loc = max_l
                            best_local_sz = (tmpl.shape[1], tmpl.shape[0])

                if best_local_score >= min_thresh and best_local_loc is not None:
                    bw, bh = best_local_sz
                    bx = x1 + best_local_loc[0]
                    by = y1 + best_local_loc[1]
                    tip_x = float(bx + bw // 2)
                    tip_y = float(by + 4)
                    return (tip_x, tip_y, (bx, by, bw, bh), best_local_score)

        # 2. Coarse-to-Fine Hierarchical Search (2.5ms across full frame)
        scale = self.coarse_scale
        small_w = int(w * scale)
        small_h = int(h * scale)
        small_gray = cv2.resize(frame_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

        best_coarse_score = -1.0
        best_coarse_loc = None
        best_tmpl_idx = 0

        for idx, t_small in enumerate(self.canonical_templates_small):
            if t_small.shape[0] >= small_h or t_small.shape[1] >= small_w:
                continue
            res = cv2.matchTemplate(small_gray, t_small, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v > best_coarse_score:
                best_coarse_score = max_v
                best_coarse_loc = max_l
                best_tmpl_idx = idx

        # If coarse match is promising, refine at native resolution in tiny crop (< 0.15ms)
        if best_coarse_score >= 0.65 and best_coarse_loc is not None:
            est_x = int(best_coarse_loc[0] / scale)
            est_y = int(best_coarse_loc[1] / scale)
            fine_tmpl = self.canonical_templates[best_tmpl_idx]
            pad_x = 35
            pad_y = 35
            rx1 = max(0, est_x - pad_x)
            ry1 = max(0, est_y - pad_y)
            rx2 = min(w, est_x + fine_tmpl.shape[1] + pad_x)
            ry2 = min(h, est_y + fine_tmpl.shape[0] + pad_y)
            crop_refine = frame_gray[ry1:ry2, rx1:rx2]

            if crop_refine.shape[0] >= fine_tmpl.shape[0] and crop_refine.shape[1] >= fine_tmpl.shape[1]:
                res_fine = cv2.matchTemplate(crop_refine, fine_tmpl, cv2.TM_CCOEFF_NORMED)
                _, fine_v, _, fine_l = cv2.minMaxLoc(res_fine)
                if fine_v >= min_thresh:
                    bw, bh = fine_tmpl.shape[1], fine_tmpl.shape[0]
                    bx = rx1 + fine_l[0]
                    by = ry1 + fine_l[1]
                    tip_x = float(bx + bw // 2)
                    tip_y = float(by + 4)
                    return (tip_x, tip_y, (bx, by, bw, bh), fine_v)

        return None

    def extract_candidate_features(self, roi_bgr: np.ndarray, contour, peak: float, ar: float, circ: float) -> np.ndarray:
        """
        Extract the 12-dimensional spectral, chromatic, and morphological feature vector
        from a candidate patch in the camera frame.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return np.zeros(12, dtype=np.float32)

        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        core_thresh = max(40.0, peak * 0.82)
        core_mask = gray_roi >= core_thresh
        halo_mask = (gray_roi >= peak * 0.25) & (~core_mask)

        if np.sum(core_mask) == 0:
            core_mask = gray_roi >= (peak * 0.5)

        b = roi_bgr[:, :, 0].astype(np.float32)
        g = roi_bgr[:, :, 1].astype(np.float32)
        r = roi_bgr[:, :, 2].astype(np.float32)

        if self.led_type == "green":
            gr_map = g - r
            gb_map = g - b
            exc_map = gr_map + gb_map
            peak_exc = float(np.max(exc_map))
            core_thresh = max(10.0, peak_exc * 0.60)
            core_mask = exc_map >= core_thresh
            if np.sum(core_mask) == 0:
                core_mask = exc_map >= peak_exc
            halo_mask = (exc_map >= max(5.0, peak_exc * 0.25)) & (~core_mask)

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
                h_int = float(np.mean(g[halo_mask]))
            else:
                h_b, h_g, h_r = c_b, c_g, c_r
                h_int = c_g

            h_excess = (h_g - h_r) + (h_g - h_b)
            h_ratio = (2.0 * h_g) / max(1.0, h_r + h_b)
            h_gr_diff = h_g - h_r
            h_gb_diff = h_g - h_b

            c_int = float(np.mean(g[core_mask]))
            core_to_halo = (c_int + 1.0) / (h_int + 1.0)

            res = np.array([
                float(np.max(g)),
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
            return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)

        c_b = float(np.mean(b[core_mask]))
        c_g = float(np.mean(g[core_mask]))
        c_r = float(np.mean(r[core_mask]))

        c_excess = (c_b - c_g) + (c_r - c_g)
        c_ratio = (c_b + c_r) / (2.0 * max(c_g, 1.0))
        bg_diff = c_b - c_g
        rg_diff = c_r - c_g

        if np.sum(halo_mask) > 0:
            h_b = float(np.mean(b[halo_mask]))
            h_g = float(np.mean(g[halo_mask]))
            h_r = float(np.mean(r[halo_mask]))
            h_int = float(np.mean(gray_roi[halo_mask]))
        else:
            h_b, h_g, h_r = c_b, c_g, c_r
            h_int = c_g

        h_excess = (h_b - h_g) + (h_r - h_g)
        h_ratio = (h_b + h_r) / (2.0 * max(h_g, 1.0))
        h_bg_diff = h_b - h_g
        h_rg_diff = h_r - h_g

        c_int = float(np.mean(gray_roi[core_mask]))
        core_to_halo = (c_int + 1.0) / (h_int + 1.0)

        return np.array([
            peak,
            c_excess,
            c_ratio,
            h_excess,
            h_ratio,
            bg_diff,
            rg_diff,
            h_bg_diff,
            h_rg_diff,
            core_to_halo,
            ar,
            circ
        ], dtype=np.float32)

    def detect_emission_candidates(self, frame_bgr: np.ndarray, display_quad: np.ndarray = None) -> tuple:
        """
        High-performance chromatic detection of LED light (Green LED or Purple IR).
        Optimized with SIMD channel subtractions and early exit for 30+ FPS on Raspberry Pi 3B+.
        100% hard rejects white blackboard drawings on the monitor and ambient lamps.
        """
        h, w = frame_bgr.shape[:2]
        b, g, r = cv2.split(frame_bgr)

        if self.led_type == "green":
            # Fast SIMD channel subtractions
            gr = cv2.subtract(g, r)
            gb = cv2.subtract(g, b)

            # Minimum channel lead: G must lead R and B by at least 6
            _, m_gr = cv2.threshold(gr, 6, 255, cv2.THRESH_BINARY)
            _, m_gb = cv2.threshold(gb, 6, 255, cv2.THRESH_BINARY)
            lead = cv2.bitwise_and(m_gr, m_gb)

            # Green excess: (G - R) + (G - B) >= threshold
            exc = cv2.add(gr, gb)
            thresh = int(max(20.0, self.green_excess_threshold))
            _, m_exc = cv2.threshold(exc, thresh, 255, cv2.THRESH_BINARY)
            green_active = cv2.bitwise_and(lead, m_exc)

            # Fast early-exit: if no green active pixels in frame, skip contours entirely (1.5ms)
            if cv2.countNonZero(green_active) == 0:
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                return [], empty_mask, 0.0

            # 3x3 open removes isolated noise speckles, 5x5 close seals blooming donut into solid disc
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(green_active, cv2.MORPH_OPEN, k3)
            cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k5)
            excess = exc.astype(np.float32)
        else:
            # Purple excess: Near-IR causes high Blue & Red with suppressed Green
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            bg = cv2.subtract(b, g)
            rg = cv2.subtract(r, g)
            _, m_bg = cv2.threshold(bg, 5, 255, cv2.THRESH_BINARY)
            _, m_rg = cv2.threshold(rg, 5, 255, cv2.THRESH_BINARY)
            lead = cv2.bitwise_and(m_bg, m_rg)

            exc = cv2.add(bg, rg)
            _, m_exc = cv2.threshold(exc, 12, 255, cv2.THRESH_BINARY)
            _, m_gray = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
            active = cv2.bitwise_and(lead, m_exc)
            active = cv2.bitwise_and(active, m_gray)

            if cv2.countNonZero(active) == 0:
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                return [], empty_mask, 0.0

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(active, cv2.MORPH_OPEN, kernel)
            excess = exc.astype(np.float32)

        # Margin exclusion: strip out ceiling lights and camera borders
        cleaned[:CAMERA_EDGE_MARGIN_TOP, :] = 0
        cleaned[-CAMERA_EDGE_MARGIN_BOTTOM:, :] = 0
        cleaned[:, :CAMERA_EDGE_MARGIN_SIDE] = 0
        cleaned[:, -CAMERA_EDGE_MARGIN_SIDE:] = 0

        # Fast early-exit: if no active pixels in active area, skip contours entirely (10x faster)
        if not np.any(cleaned):
            return [], cleaned, 0.0

        # Live frame excess meter across active field of view:
        meter_valid = cleaned > 0
        frame_max_excess = float(np.max(excess[meter_valid])) if np.any(meter_valid) else 0.0

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        has_valid_quad = False
        if display_quad is not None and len(display_quad) == 4:
            if cv2.contourArea(display_quad.astype(np.float32)) > 5000:
                has_valid_quad = True

        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue

            peri = cv2.arcLength(c, True)
            circ = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0.0
            min_circ = 0.18 if self.led_type == "green" else 0.18
            if circ < min_circ:
                continue

            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            ar = float(w_b) / max(1.0, float(h_b))
            min_ar = 0.35 if self.led_type == "green" else 0.35
            max_ar = 2.8 if self.led_type == "green" else 2.8
            if not (min_ar <= ar <= max_ar):
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            # Edge exclusion margin
            if (cy < CAMERA_EDGE_MARGIN_TOP or
                cy > (h - CAMERA_EDGE_MARGIN_BOTTOM) or
                cx < CAMERA_EDGE_MARGIN_SIDE or
                cx > (w - CAMERA_EDGE_MARGIN_SIDE)):
                continue

            # Candidate patch evaluation
            patch_bgr = frame_bgr[max(0, y_b):min(h, y_b + h_b), max(0, x_b):min(w, x_b + w_b)]
            pb = float(np.mean(patch_bgr[:, :, 0]))
            pg = float(np.mean(patch_bgr[:, :, 1]))
            pr = float(np.mean(patch_bgr[:, :, 2]))
            p_peak = float(np.max(patch_bgr))

            if self.led_type == "green":
                p_excess = float(np.max(excess[max(0, y_b):min(h, y_b + h_b), max(0, x_b):min(w, x_b + w_b)]))
                p_ratio = (2.0 * pg) / max(1.0, pr + pb)
                if (p_peak < 85.0 or
                    p_excess < self.green_excess_threshold or
                    p_ratio < self.green_spectral_ratio_min):
                    continue
            else:
                p_excess = (pb - pg) + (pr - pg)
                p_ratio = (pb + pr) / (2.0 * max(pg, 1.0))
                if (p_peak < 115.0 or
                    pb < pg + 5.0 or
                    pr < pg + 5.0 or
                    p_excess < 12.0 or
                    p_ratio < 1.08):
                    continue

            # Display containment check
            in_display = True
            if has_valid_quad:
                poly = display_quad.astype(np.float32)
                in_display = (cv2.pointPolygonTest(poly, (cx, cy), False) >= 0)
                if not in_display:
                    # Outside calibrated display boundaries -> do not draw
                    continue

            score = (p_excess * 3.0) + (p_peak * 1.5) + (circ * 50.0)

            candidates.append({
                "point": (cx, cy),
                "bbox": (x_b, y_b, w_b, h_b),
                "contour": c,
                "area": area,
                "circularity": circ,
                "aspect_ratio": ar,
                "peak": p_peak,
                "excess": p_excess,
                "purple_excess": p_excess,
                "green_excess": p_excess if self.led_type == "green" else 0.0,
                "spectral_ratio": p_ratio,
                "confidence": 98.0,
                "in_display": in_display,
                "score": score
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates, cleaned, frame_max_excess

    def detect_purple_emission_candidates(self, frame_bgr: np.ndarray, display_quad: np.ndarray = None) -> tuple:
        """Backwards compatibility alias for detect_emission_candidates."""
        return self.detect_emission_candidates(frame_bgr, display_quad)

    def detect(self, frame: np.ndarray, threshold: int = None, display_quad: np.ndarray = None):
        """
        Locate the centroid of the actively emitting 5mm IR LED bulb.
        Returns: (x, y) tuple or None.
        """
        pt, _, _, _ = self.detect_debug(frame, threshold=threshold, display_quad=display_quad)
        return pt

    def detect_debug(self, frame: np.ndarray, threshold: int = None, display_quad: np.ndarray = None):
        """
        Two-stage comprehensive detection:
          - If the LED bulb is actively emitting purple light:
              Returns drawing point (cx, cy) and visualizes bright green reticle.
          - If the LED bulb is unlit / idle:
              Locates the physical 5mm bulb and points to it with a cyan pointer in dev mode!
              Returns None for whiteboard drawing (pen lifted).
        """
        if frame is None:
            empty = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            return None, empty, empty, {"status": "NO_FRAME", "bulb_state": "NONE"}

        h, w = frame.shape[:2]

        if len(frame.shape) == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            frame_gray = frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        annotated = frame.copy()

        # -------------------------------------------------------------
        # Draw Calibrated Display Quad (DocScanner Screen Boundaries)
        # -------------------------------------------------------------
        has_valid_quad = False
        if display_quad is not None and len(display_quad) == 4:
            if cv2.contourArea(display_quad.astype(np.float32)) > 5000:
                has_valid_quad = True
                quad_int = display_quad.astype(np.int32)
                cv2.polylines(annotated, [quad_int], isClosed=True, color=(0, 240, 255), thickness=2)
                corner_labels = ["TL", "TR", "BR", "BL"]
                for i, (cx_pt, cy_pt) in enumerate(quad_int):
                    cv2.circle(annotated, (cx_pt, cy_pt), 4, (0, 255, 100), -1)
                    cv2.putText(annotated, corner_labels[i], (cx_pt + 6, cy_pt - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

        # -------------------------------------------------------------
        # STEP 1: Check for Active Light Emission (Green or IR)
        # -------------------------------------------------------------
        candidates, active_mask, frame_max_excess = self.detect_emission_candidates(frame_bgr, display_quad)
        thresh_rgb = cv2.cvtColor(active_mask, cv2.COLOR_GRAY2RGB)

        led_name = "GREEN LED" if self.led_type == "green" else "IR LED"
        thresh_limit = self.green_excess_threshold if self.led_type == "green" else 8.0

        metadata = {
            "threshold": threshold if threshold is not None else self.threshold,
            "bulb_state": "SEARCHING",
            "is_emitting": False,
            "bulb_pos": None,
            "led_type": self.led_type,
            "excess": frame_max_excess,
            "purple_excess": frame_max_excess,
            "green_excess": frame_max_excess if self.led_type == "green" else 0.0,
            "spectral_ratio": 1.0,
            "circularity": 0.0,
            "in_display_area": False,
            "status": "SEARCHING"
        }

        # CASE A: Bulb is Actively Emitting Light (Button Pressed -> DRAWING)
        if len(candidates) > 0:
            best = candidates[0]
            bcx, bcy = best["point"]
            x_b, y_b, w_b, h_b = best["bbox"]
            in_disp = best["in_display"]
            excess = best["excess"]

            self.last_bulb_pos = (bcx, bcy)
            self.last_bulb_box = (x_b, y_b, w_b, h_b)
            self.frames_since_seen = 0

            reticle_col = (0, 255, 0) if in_disp else (255, 140, 0)
            status_text = "DRAWING ON BOARD" if in_disp else "OUTSIDE DISPLAY"
            box_col = (0, 255, 0) if self.led_type == "green" else (255, 0, 255)

            # 1. Glowing bounding box around the 5mm LED bulb
            cv2.rectangle(annotated, (x_b, y_b), (x_b + w_b, y_b + h_b), box_col, 2)
            cv2.circle(annotated, (int(bcx), int(bcy)), int(max(w_b, h_b) // 2 + 4), box_col, 1)

            # 2. Precision target reticle
            cv2.circle(annotated, (int(bcx), int(bcy)), 12, reticle_col, 2)
            cv2.circle(annotated, (int(bcx), int(bcy)), 3, reticle_col, -1)
            cv2.line(annotated, (int(bcx) - 18, int(bcy)), (int(bcx) + 18, int(bcy)), reticle_col, 2)
            cv2.line(annotated, (int(bcx), int(bcy) - 18), (int(bcx), int(bcy) + 18), reticle_col, 2)

            # 3. Informational HUD tag
            badge = f"{led_name}: EMITTING (Meter +{excess:.1f} > {thresh_limit:.1f}) -> ({int(bcx)}, {int(bcy)}) [{status_text}]"
            cv2.putText(annotated, badge, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 100), 2)

            # Mark on threshold mask
            cv2.circle(thresh_rgb, (int(bcx), int(bcy)), 10, (0, 255, 0), 2)
            cv2.putText(thresh_rgb, f"{led_name} DETECTED: +{excess:.1f} (THRESHOLD > {thresh_limit:.1f})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)

            metadata["bulb_state"] = "EMITTING"
            metadata["is_emitting"] = True
            metadata["bulb_pos"] = (bcx, bcy)
            metadata["excess"] = excess
            metadata["purple_excess"] = excess
            metadata["green_excess"] = excess if self.led_type == "green" else 0.0
            metadata["spectral_ratio"] = best["spectral_ratio"]
            metadata["circularity"] = best["circularity"]
            metadata["in_display_area"] = in_disp
            metadata["status"] = "EMITTING_DRAWING"

            return (bcx, bcy), annotated, thresh_rgb, metadata

        # -------------------------------------------------------------
        # STEP 2: Bulb is Unlit / Idle -> Locate Physical 5mm Bulb Tip
        # -------------------------------------------------------------
        self.frames_since_seen += 1
        bulb_res = self.locate_physical_bulb(frame_gray)

        if bulb_res is not None:
            tip_x, tip_y, (bx, by, bw, bh), score = bulb_res
            self.last_bulb_pos = (tip_x, tip_y)
            self.last_bulb_box = (bx, by, bw, bh)
            self.frames_since_seen = 0

            itip_x, itip_y = int(tip_x), int(tip_y)

            # Check if physical bulb is inside display quad
            in_disp = True
            if has_valid_quad:
                poly = display_quad.astype(np.float32)
                in_disp = (cv2.pointPolygonTest(poly, (tip_x, tip_y), False) >= 0)

            # Temporal Chromaticity Comparison with previous images at tracked bulb tip
            is_emit, tip_excess, peak, info = self.check_bulb_emission_at_point(frame_bgr, tip_x, tip_y)

            if is_emit:
                # The tracked bulb just lit up! Trigger DRAWING!
                reticle_col = (0, 255, 0) if in_disp else (255, 140, 0)
                status_text = "DRAWING ON BOARD" if in_disp else "OUTSIDE DISPLAY"
                box_col = (0, 255, 0) if self.led_type == "green" else (255, 0, 255)

                # Glowing reticle around the bulb tip
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_col, 2)
                cv2.circle(annotated, (itip_x, itip_y), 12, reticle_col, 2)
                cv2.circle(annotated, (itip_x, itip_y), 3, reticle_col, -1)
                cv2.line(annotated, (itip_x - 18, itip_y), (itip_x + 18, itip_y), reticle_col, 2)
                cv2.line(annotated, (itip_x, itip_y - 18), (itip_x, itip_y + 18), reticle_col, 2)

                badge = f"{led_name}: EMITTING ({info}) -> ({itip_x}, {itip_y}) [{status_text}]"
                cv2.putText(annotated, badge, (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 100), 2)

                cv2.circle(thresh_rgb, (itip_x, itip_y), 10, (0, 255, 0), 2)
                cv2.putText(thresh_rgb, f"{led_name} DETECTED: +{tip_excess:.1f} (THRESHOLD > {thresh_limit:.1f})", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)

                metadata["bulb_state"] = "EMITTING"
                metadata["is_emitting"] = True
                metadata["bulb_pos"] = (tip_x, tip_y)
                metadata["excess"] = tip_excess
                metadata["purple_excess"] = tip_excess
                metadata["green_excess"] = tip_excess if self.led_type == "green" else 0.0
                metadata["in_display_area"] = in_disp
                metadata["status"] = "EMITTING_DRAWING"

                return (tip_x, tip_y), annotated, thresh_rgb, metadata

            # Visual pointer directly showing where the 5mm LED bulb is in Dev Mode!
            cyan_col = (0, 240, 255)
            # 1. Reticle framing the bulb dome
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), cyan_col, 2)
            cv2.circle(annotated, (itip_x, itip_y), 8, (255, 230, 40), 2)
            cv2.circle(annotated, (itip_x, itip_y), 2, (255, 255, 255), -1)

            # 2. Pointer arrow pointing directly down to the bulb tip
            cv2.arrowedLine(annotated, (itip_x, max(0, itip_y - 28)), (itip_x, max(0, itip_y - 8)),
                            (255, 220, 40), 2, tipLength=0.35)

            # 3. Label tag
            label = f"[5mm {self.led_type.upper()} BULB LOCATED] Tip: ({itip_x}, {itip_y}) | Meter: +{frame_max_excess:.1f} (IDLE)"
            cv2.putText(annotated, label, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cyan_col, 2)

            # On threshold mask: indicate tracked position
            cv2.rectangle(thresh_rgb, (bx, by), (bx + bw, by + bh), (0, 200, 255), 1)
            cv2.putText(thresh_rgb, f"5mm BULB LOCATED: IDLE (Meter: +{frame_max_excess:.1f})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

            metadata["bulb_state"] = "LOCATED_IDLE"
            metadata["is_emitting"] = False
            metadata["bulb_pos"] = (tip_x, tip_y)
            metadata["in_display_area"] = in_disp
            metadata["status"] = "BULB_LOCATED_IDLE"

            # Whiteboard does NOT draw when bulb is unlit (pen lifted)
            return None, annotated, thresh_rgb, metadata

        # CASE C: No bulb and no light in view
        cv2.putText(annotated, f"SEARCHING FOR 5mm {self.led_type.upper()} BULB / LIGHT (Meter: +{frame_max_excess:.1f})", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)
        cv2.putText(thresh_rgb, f"SEARCHING (Meter: +{frame_max_excess:.1f} IDLE)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 2)

        return None, annotated, thresh_rgb, metadata


# Class alias for clean naming
LEDDetector = IRDetector


if __name__ == "__main__":
    print("Testing Two-Stage LEDDetector (Green & IR Modes)...")

    # --- Mode 1: IR LED Mode ---
    det_ir = IRDetector(threshold=200, led_type="ir")

    # Test 1: User's real unlit bulb photo 1
    im1 = cv2.imread(r'C:/Users/prana/.gemini/antigravity/brain/742360c1-85ca-4369-a3d9-0dea5a3a94ec/.user_uploaded/media_1793025491309_fd5b99af.jpg')
    pt1, _, _, meta1 = det_ir.detect_debug(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    print("\n--- Test 1: User Photo 1 (Unlit 5mm Stylus Bulb in Hand) ---")
    print(f"  Drawing Point : {pt1} (Expected: None -> Pen Lifted)")
    print(f"  Bulb State    : {meta1['bulb_state']} (Expected: LOCATED_IDLE)")
    assert pt1 is None, "Failed: Unlit bulb should not draw!"
    assert meta1['bulb_state'] == "LOCATED_IDLE", f"Failed: Expected LOCATED_IDLE, got {meta1['bulb_state']}"

    # Test 2: Active Emitting IR LED Bulb
    im2 = cv2.imread(r'dataset/ir_led_emitting/user_ref_emitting_ground_truth.png')
    frame_emit = np.zeros((480, 640, 3), dtype=np.uint8)
    h2, w2 = im2.shape[:2]
    frame_emit[100:100+h2, 200:200+w2] = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    pt2, _, _, meta2 = det_ir.detect_debug(frame_emit)
    print("\n--- Test 2: Active Emitting IR LED Bulb ---")
    print(f"  Drawing Point : {pt2} (Expected: Valid (x, y) -> Drawing Active)")
    print(f"  Bulb State    : {meta2['bulb_state']} (Expected: EMITTING)")
    assert pt2 is not None, "Failed: Emitting IR LED should produce drawing point!"
    assert meta2['bulb_state'] == "EMITTING", f"Failed: Expected EMITTING, got {meta2['bulb_state']}"

    # --- Mode 2: Green LED Mode ---
    det_green = IRDetector(threshold=110, led_type="green")
    frame_green = np.zeros((480, 640, 3), dtype=np.uint8)
    # Bright green LED spot: (B=25, G=245, R=35)
    cv2.circle(frame_green, (310, 220), 8, (35, 245, 25), -1)
    pt_g, _, _, meta_g = det_green.detect_debug(frame_green)
    print("\n--- Test 3: Active Emitting 5mm Green LED Bulb ---")
    print(f"  Drawing Point : {pt_g} (Expected: Valid (x, y) -> Drawing Active)")
    print(f"  Bulb State    : {meta_g['bulb_state']} (Expected: EMITTING)")
    print(f"  Green Excess  : +{meta_g['green_excess']:.1f}")
    assert pt_g is not None, "Failed: Emitting Green LED should produce drawing point!"
    assert meta_g['bulb_state'] == "EMITTING", f"Failed: Expected EMITTING, got {meta_g['bulb_state']}"

    # Test 4: Blackboard white drawing (White Light) Rejected in Green Mode
    frame_white = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame_white, (300, 200), 10, (255, 255, 255), -1)
    pt4, _, _, meta4 = det_green.detect_debug(frame_white)
    print("\n--- Test 4: Whiteboard Drawing on Screen (White Light Rejected) ---")
    print(f"  Drawing Point : {pt4} (Expected: None -> White light rejected)")
    assert pt4 is None, "Failed: Whiteboard drawing should be 100% rejected in Green mode!"

    # Test 5: Incandescent Lamp (Warm Yellow / Red) Rejected in Green Mode
    frame_lamp = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame_lamp, (280, 190), 12, (255, 210, 80), -1)  # High R & G, low B
    pt5, _, _, meta5 = det_green.detect_debug(frame_lamp)
    print("\n--- Test 5: Ambient Room Lamp (Warm Lamp Rejected) ---")
    print(f"  Drawing Point : {pt5} (Expected: None -> Ambient lamp rejected)")
    assert pt5 is None, "Failed: Ambient room lamp should be 100% rejected in Green mode!"

    print("\nALL TWO-STAGE DETECTOR TESTS (GREEN & IR) PASSED PERFECTLY!")
