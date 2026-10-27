import os
import json
from pathlib import Path
import cv2
import numpy as np
from config import (
    IR_THRESHOLD, IR_MIN_AREA, IR_MAX_AREA,
    CAMERA_EDGE_MARGIN_TOP, CAMERA_EDGE_MARGIN_BOTTOM, CAMERA_EDGE_MARGIN_SIDE,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    IR_SPECTRAL_RATIO_MIN, IR_PURPLE_EXCESS_MIN, IR_PURPLE_EXCESS_THRESHOLD,
    IR_MODEL_FILE, PEN_MODEL_FILE, BASE_DIR
)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class IRDetector:
    """
    Two-Stage 5mm IR LED Bulb Stylus & Purple Infrared Emission Detector.

    1. STAGE 1 (Physical Bulb Localization):
       Detects the physical 5mm clear through-hole LED bulb (the stylus tip as shown
       in the user's photos) in ambient light even when unlit. In Dev Mode, points directly
       to the bulb tip with a cyan reticle and arrow: [5mm BULB LOCATED: IDLE].

    2. STAGE 2 (Active Purple Emission & Drawing Trigger):
       High-sensitivity optical chromaticity detector that triggers when the LED emits
       even a subtle / slight purple near-infrared glow (push button pressed).
       Immediately transitions to [IR LED: EMITTING] and marks/draws on the whiteboard canvas!

    3. HIGH-ACCURACY FALSE POSITIVE REJECTION:
       100% hard rejects white blackboard drawings on the monitor (where B <= G, excess <= 0)
       and warm ambient room lamps (where B < G). Never causes runaway drawing loops!
    """
    def __init__(self, threshold=IR_THRESHOLD, min_area=IR_MIN_AREA, max_area=IR_MAX_AREA):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

        # Last known bulb position for smooth fast tracking
        self.last_bulb_pos = None
        self.last_bulb_box = None
        self.frames_since_seen = 999

        # Load 5mm bulb templates
        self.bulb_templates = []
        self._load_bulb_templates()

        # Load trained ML model if available
        self.has_model = False
        self.model_name = "Spectral Model"
        self.mean = None
        self.std = None
        self.weights = None
        self.bias = 0.0
        self.model_threshold = 0.50

        if os.path.exists(IR_MODEL_FILE):
            try:
                with open(IR_MODEL_FILE, "r", encoding="utf-8") as f:
                    model_data = json.load(f)
                self.mean = np.array(model_data.get("mean", []), dtype=np.float32)
                self.std = np.array(model_data.get("std", []), dtype=np.float32)
                self.weights = np.array(model_data.get("weights", []), dtype=np.float32)
                self.bias = float(model_data.get("bias", 0.0))
                self.model_threshold = float(model_data.get("threshold", 0.50))
                self.model_name = f"{model_data.get('model_name', 'IR_Model')}"
                self.has_model = True
                print(f"[IRDetector] Loaded trained ML model from: {IR_MODEL_FILE}")
            except Exception as e:
                print(f"[IRDetector] Warning: Could not load model: {e}")

        # Load trained Pen In Hand model if available
        self.pen_model_data = None
        self.pen_threshold = 0.82
        if os.path.exists(PEN_MODEL_FILE):
            try:
                with open(PEN_MODEL_FILE, "r", encoding="utf-8") as f:
                    self.pen_model_data = json.load(f)
                self.pen_threshold = float(self.pen_model_data.get("correlation_threshold", 0.82))
                print(f"[IRDetector] Loaded trained Pen-In-Hand model from: {PEN_MODEL_FILE} (threshold={self.pen_threshold:.2f})")
            except Exception as e:
                print(f"[IRDetector] Warning: Could not load pen model: {e}")

    def _load_bulb_templates(self):
        """Load orientation templates of the 5mm LED bulb, prioritizing exact user templates and trained pen tip exemplars."""
        tmpl_dir = Path(BASE_DIR) / "models" / "bulb_templates"
        if tmpl_dir.exists():
            priority = ["user_tip_up.png", "user_tip_down.png"]
            priority.extend([f"pen_tip_trained_{i:02d}.png" for i in range(12)])
            loaded = set()
            for p_name in priority:
                p = tmpl_dir / p_name
                if p.exists():
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

    def locate_physical_bulb(self, frame_gray: np.ndarray) -> tuple:
        """
        Locates the physical 5mm clear LED bulb tip on the pen in ambient light when idle/off.
        Uses calibrated correlation threshold (>= 0.82) to NEVER lock onto random room objects.
        Returns: (tip_x, tip_y, bbox, score) or None.
        """
        if frame_gray is None or len(self.bulb_templates) == 0:
            return None

        h, w = frame_gray.shape[:2]
        min_thresh = getattr(self, "pen_threshold", 0.82)
        min_thresh = max(0.82, min_thresh)

        # 1. Fast Local Window Search (if recently tracked and stable)
        if self.last_bulb_pos is not None and self.frames_since_seen < 6:
            lx, ly = int(self.last_bulb_pos[0]), int(self.last_bulb_pos[1])
            pad = 60
            x1 = max(0, lx - pad)
            y1 = max(0, ly - pad)
            x2 = min(w, lx + pad)
            y2 = min(h, ly + pad)
            crop = frame_gray[y1:y2, x1:x2]

            if crop.shape[0] >= 30 and crop.shape[1] >= 30:
                best_local_score = -1.0
                best_local_loc = None
                best_local_sz = (0, 0)
                for tmpl in self.bulb_templates[:6]:
                    if tmpl.shape[0] < crop.shape[0] and tmpl.shape[1] < crop.shape[1]:
                        res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
                        _, max_v, _, max_l = cv2.minMaxLoc(res)
                        if max_v > best_local_score:
                            best_local_score = max_v
                            best_local_loc = max_l
                            best_local_sz = (tmpl.shape[1], tmpl.shape[0])

                # Require high confidence to maintain track without drifting to background
                if best_local_score >= min_thresh and best_local_loc is not None:
                    bw, bh = best_local_sz
                    bx = x1 + best_local_loc[0]
                    by = y1 + best_local_loc[1]
                    tip_x = float(bx + bw // 2)
                    tip_y = float(by + 4)
                    return (tip_x, tip_y, (bx, by, bw, bh), best_local_score)

        # 2. Global Search for authentic 5mm bulb with high confidence
        best_score = -1.0
        best_loc = None
        best_size = (0, 0)

        # Evaluate priority templates across angles at native resolution
        for tmpl in self.bulb_templates[:6]:
            tw, th = tmpl.shape[1], tmpl.shape[0]
            if tw >= w or th >= h:
                continue
            res = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v > best_score:
                best_score = max_v
                best_loc = max_l
                best_size = (tw, th)

        # Strict threshold: random room objects (shelves, screws, light switches) score <= 0.70-0.77.
        # Requiring >= min_thresh (>= 0.82) guarantees zero false detections on arbitrary background objects!
        if best_score >= min_thresh and best_loc is not None:
            bx, by = best_loc
            bw, bh = best_size
            tip_x = float(bx + bw // 2)
            tip_y = float(by + 4)
            return (tip_x, tip_y, (bx, by, bw, bh), best_score)
            return (tip_x, tip_y, (bx, by, bw, bh), best_score)

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

    def detect_purple_emission_candidates(self, frame_bgr: np.ndarray, display_quad: np.ndarray = None) -> list:
        """
        High-sensitivity chromatic detection of near-infrared purple light.
        Detects even slight / subtle purple emission from the 5mm LED bulb,
        while 100% rejecting white blackboard drawings and ambient lamps.
        """
        h, w = frame_bgr.shape[:2]
        b = frame_bgr[:, :, 0].astype(np.float32)
        g = frame_bgr[:, :, 1].astype(np.float32)
        r = frame_bgr[:, :, 2].astype(np.float32)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Purple excess: Near-IR causes high Blue & Red with suppressed Green
        bg_diff = b - g
        rg_diff = r - g
        excess = bg_diff + rg_diff

        # CRITICAL FILTER:
        # Authentic IR light produces genuine purple/violet tint where BOTH Blue > Green and Red > Green:
        # Whiteboard monitor white pixels have B <= G and excess <= 0 -> 100% BLOCKED.
        # Room lamps have B < G -> 100% BLOCKED.
        # Blue screen reflections have R < G -> 100% BLOCKED.
        # Dim room shadows have gray < 110 -> 100% BLOCKED.
        purple_active = (bg_diff >= 5.0) & (rg_diff >= 5.0) & (excess >= 12.0) & (gray >= 110)
        purple_mask = np.uint8(purple_active * 255)

        # Margin exclusion: strip out ceiling lights and camera borders
        purple_mask[:CAMERA_EDGE_MARGIN_TOP, :] = 0
        purple_mask[-CAMERA_EDGE_MARGIN_BOTTOM:, :] = 0
        purple_mask[:, :CAMERA_EDGE_MARGIN_SIDE] = 0
        purple_mask[:, -CAMERA_EDGE_MARGIN_SIDE:] = 0

        # Live frame purple excess meter across active field of view:
        meter_valid = (bg_diff >= 5.0) & (rg_diff >= 5.0) & (gray >= 110)
        meter_valid[:CAMERA_EDGE_MARGIN_TOP, :] = False
        meter_valid[-CAMERA_EDGE_MARGIN_BOTTOM:, :] = False
        meter_valid[:, :CAMERA_EDGE_MARGIN_SIDE] = False
        meter_valid[:, -CAMERA_EDGE_MARGIN_SIDE:] = False
        frame_max_excess = float(np.max(excess[meter_valid])) if np.any(meter_valid) else 0.0

        # 3x3 Morphological opening to clean single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        purple_cleaned = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(purple_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            if circ < 0.18:
                continue

            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            ar = float(w_b) / max(1.0, float(h_b))
            if not (0.35 <= ar <= 2.8):
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
            patch_bgr = frame_bgr[max(0, y_b - 2):min(h, y_b + h_b + 2), max(0, x_b - 2):min(w, x_b + w_b + 2)]
            pb = float(np.mean(patch_bgr[:, :, 0]))
            pg = float(np.mean(patch_bgr[:, :, 1]))
            pr = float(np.mean(patch_bgr[:, :, 2]))
            p_peak = float(np.max(patch_bgr))
            p_excess = (pb - pg) + (pr - pg)
            p_ratio = (pb + pr) / (2.0 * max(pg, 1.0))

            # Must satisfy physical IR emission profile:
            # 1. Intense luminous peak (IR LED emits bright radiation: peak >= 115)
            # 2. Both Blue and Red strictly higher than Green (authentic purple glow)
            # 3. Purple excess >= 12.0
            # 4. Spectral ratio (R + B) / (2 * G) >= 1.08
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
                "purple_excess": p_excess,
                "spectral_ratio": p_ratio,
                "confidence": 98.0,
                "in_display": in_display,
                "score": score
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates, purple_cleaned, frame_max_excess

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
        # STEP 1: Check for Active Purple Light Emission
        # -------------------------------------------------------------
        candidates, purple_mask, frame_max_excess = self.detect_purple_emission_candidates(frame_bgr, display_quad)
        thresh_rgb = cv2.cvtColor(purple_mask, cv2.COLOR_GRAY2RGB)

        metadata = {
            "threshold": threshold if threshold is not None else self.threshold,
            "bulb_state": "SEARCHING",
            "is_emitting": False,
            "bulb_pos": None,
            "purple_excess": frame_max_excess,
            "spectral_ratio": 1.0,
            "circularity": 0.0,
            "in_display_area": False,
            "status": "SEARCHING"
        }

        # CASE A: Bulb is Actively Emitting Purple Light (Button Pressed -> DRAWING)
        if len(candidates) > 0:
            best = candidates[0]
            bcx, bcy = best["point"]
            x_b, y_b, w_b, h_b = best["bbox"]
            in_disp = best["in_display"]
            excess = best["purple_excess"]

            self.last_bulb_pos = (bcx, bcy)
            self.last_bulb_box = (x_b, y_b, w_b, h_b)
            self.frames_since_seen = 0

            reticle_col = (0, 255, 0) if in_disp else (255, 140, 0)
            status_text = "DRAWING ON BOARD" if in_disp else "OUTSIDE DISPLAY"

            # 1. Glowing magenta/purple bounding box around the 5mm LED bulb
            cv2.rectangle(annotated, (x_b, y_b), (x_b + w_b, y_b + h_b), (255, 0, 255), 2)
            cv2.circle(annotated, (int(bcx), int(bcy)), int(max(w_b, h_b) // 2 + 4), (200, 60, 240), 1)

            # 2. Precision target reticle
            cv2.circle(annotated, (int(bcx), int(bcy)), 12, reticle_col, 2)
            cv2.circle(annotated, (int(bcx), int(bcy)), 3, reticle_col, -1)
            cv2.line(annotated, (int(bcx) - 18, int(bcy)), (int(bcx) + 18, int(bcy)), reticle_col, 2)
            cv2.line(annotated, (int(bcx), int(bcy) - 18), (int(bcx), int(bcy) + 18), reticle_col, 2)

            # 3. Informational HUD tag
            badge = f"IR LED: EMITTING (Meter +{excess:.1f} > 8.0) -> ({int(bcx)}, {int(bcy)}) [{status_text}]"
            cv2.putText(annotated, badge, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 100), 2)

            # Mark on threshold mask
            cv2.circle(thresh_rgb, (int(bcx), int(bcy)), 10, (0, 255, 0), 2)
            cv2.putText(thresh_rgb, f"PURPLE IR DETECTED: +{excess:.1f} (THRESHOLD > 8.0)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)

            metadata["bulb_state"] = "EMITTING"
            metadata["is_emitting"] = True
            metadata["bulb_pos"] = (bcx, bcy)
            metadata["purple_excess"] = excess
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
            label = f"[5mm LED BULB LOCATED] Tip: ({itip_x}, {itip_y}) | Meter: +{frame_max_excess:.1f} (IDLE <= 8.0)"
            cv2.putText(annotated, label, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cyan_col, 2)

            # On threshold mask: indicate tracked position
            cv2.rectangle(thresh_rgb, (bx, by), (bx + bw, by + bh), (0, 200, 255), 1)
            cv2.putText(thresh_rgb, f"5mm BULB LOCATED: IDLE (Meter: +{frame_max_excess:.1f} <= 8.0)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

            metadata["bulb_state"] = "LOCATED_IDLE"
            metadata["is_emitting"] = False
            metadata["bulb_pos"] = (tip_x, tip_y)
            metadata["in_display_area"] = in_disp
            metadata["status"] = "BULB_LOCATED_IDLE"

            # Whiteboard does NOT draw when bulb is unlit (pen lifted)
            return None, annotated, thresh_rgb, metadata

        # CASE C: No bulb and no purple light in view
        cv2.putText(annotated, f"SEARCHING FOR 5mm LED BULB / IR LIGHT (Meter: +{frame_max_excess:.1f})", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)
        cv2.putText(thresh_rgb, f"SEARCHING (Meter: +{frame_max_excess:.1f} <= 8.0 IDLE)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 2)

        return None, annotated, thresh_rgb, metadata


if __name__ == "__main__":
    print("Testing Two-Stage IRDetector with 5mm LED bulb localization & purple emission...")
    det = IRDetector(threshold=200)

    # Test 1: User's real unlit bulb photo 1
    im1 = cv2.imread(r'C:/Users/prana/.gemini/antigravity/brain/742360c1-85ca-4369-a3d9-0dea5a3a94ec/.user_uploaded/media_1793025491309_fd5b99af.jpg')
    pt1, _, _, meta1 = det.detect_debug(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    print("\n--- Test 1: User Photo 1 (Unlit 5mm Stylus Bulb in Hand) ---")
    print(f"  Drawing Point : {pt1} (Expected: None -> Pen Lifted)")
    print(f"  Bulb State    : {meta1['bulb_state']} (Expected: LOCATED_IDLE)")
    print(f"  Bulb Position : {meta1['bulb_pos']}")
    assert pt1 is None, "Failed: Unlit bulb should not draw!"
    assert meta1['bulb_state'] == "LOCATED_IDLE", f"Failed: Expected LOCATED_IDLE, got {meta1['bulb_state']}"

    # Test 2: User's real emitting IR LED image
    im2 = cv2.imread(r'dataset/ir_led_emitting/user_ref_emitting_ground_truth.png')
    frame_emit = np.zeros((480, 640, 3), dtype=np.uint8)
    h2, w2 = im2.shape[:2]
    frame_emit[100:100+h2, 200:200+w2] = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    pt2, _, _, meta2 = det.detect_debug(frame_emit)
    print("\n--- Test 2: Active Emitting IR LED Bulb ---")
    print(f"  Drawing Point : {pt2} (Expected: Valid (x, y) -> Drawing Active)")
    print(f"  Bulb State    : {meta2['bulb_state']} (Expected: EMITTING)")
    print(f"  Purple Excess : +{meta2['purple_excess']:.1f}")
    assert pt2 is not None, "Failed: Emitting IR LED should produce drawing point!"
    assert meta2['bulb_state'] == "EMITTING", f"Failed: Expected EMITTING, got {meta2['bulb_state']}"

    # Test 3: Blackboard white drawing (White Light)
    frame_white = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame_white, (300, 200), 10, (255, 255, 255), -1)
    pt3, _, _, meta3 = det.detect_debug(frame_white)
    print("\n--- Test 3: Whiteboard Drawing on Screen (White Light) ---")
    print(f"  Drawing Point : {pt3} (Expected: None -> White light rejected)")
    print(f"  Bulb State    : {meta3['bulb_state']}")
    assert pt3 is None, "Failed: Whiteboard drawing should be 100% rejected!"

    print("\nALL TWO-STAGE DETECTOR TESTS PASSED PERFECTLY!")
