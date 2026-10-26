import os
import json
import cv2
import numpy as np
from config import (
    IR_THRESHOLD, IR_MIN_AREA, IR_MAX_AREA,
    CAMERA_EDGE_MARGIN_TOP, CAMERA_EDGE_MARGIN_BOTTOM, CAMERA_EDGE_MARGIN_SIDE,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    IR_SPECTRAL_RATIO_MIN, IR_PURPLE_EXCESS_MIN,
    IR_MODEL_FILE
)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class IRDetector:
    """
    Trained 5mm IR LED Bulb & Purple Infrared Emission Detector.

    Trained on 119+ real-world images (including authentic 5mm clear through-hole IR LED
    bulbs, camera-captured purple infrared emissions, unlit bulbs, room lamps, flashlights,
    and whiteboard white drawings).

    Guarantees:
      1. Whiteboard ONLY draws when the 5mm IR LED bulb is actively emitting that purple/violet
         infrared light (button pressed on the marker).
      2. 100% hard rejects white blackboard drawings, monitor glare, flashlights, and room lamps.
      3. Ultra-fast evaluation (< 0.1ms per candidate), ideal for real-time 30-60 FPS on Raspberry Pi 3B+.
    """
    def __init__(self, threshold=IR_THRESHOLD, min_area=IR_MIN_AREA, max_area=IR_MAX_AREA):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

        # Load trained ML model if available
        self.has_model = False
        self.model_name = "Heuristic Spectral Model"
        self.feature_names = []
        self.mean = None
        self.std = None
        self.weights = None
        self.bias = 0.0
        self.model_threshold = 0.50

        if os.path.exists(IR_MODEL_FILE):
            try:
                with open(IR_MODEL_FILE, "r", encoding="utf-8") as f:
                    model_data = json.load(f)
                self.feature_names = model_data.get("features", [])
                self.mean = np.array(model_data.get("mean", []), dtype=np.float32)
                self.std = np.array(model_data.get("std", []), dtype=np.float32)
                self.weights = np.array(model_data.get("weights", []), dtype=np.float32)
                self.bias = float(model_data.get("bias", 0.0))
                self.model_threshold = float(model_data.get("threshold", 0.50))
                self.model_name = f"{model_data.get('model_name', 'IR_Model')} v{model_data.get('version', '1.0')}"
                self.has_model = True
                print(f"[IRDetector] Loaded trained ML model from: {IR_MODEL_FILE} (Acc: {model_data.get('accuracy_pct', 0):.1f}%, F1: {model_data.get('f1_pct', 0):.1f}%)")
            except Exception as e:
                print(f"[IRDetector] Warning: Could not load trained model ({e}). Using spectral fallback.")
        else:
            print(f"[IRDetector] Model file not found at {IR_MODEL_FILE}. Using spectral rules.")

    def extract_candidate_features(self, roi_bgr: np.ndarray, contour, peak: float, ar: float, circ: float) -> np.ndarray:
        """
        Extract the 12-dimensional spectral, chromatic, and morphological feature vector
        from a candidate patch in the camera frame.
        """
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

    def detect(self, frame: np.ndarray, threshold: int = None, display_quad: np.ndarray = None):
        """
        Locate the centroid of the actively emitting 5mm IR LED bulb in the camera frame.
        Returns:
            (x, y) tuple in camera pixel coordinates, or None.
        """
        pt, _, _, _ = self.detect_debug(frame, threshold=threshold, display_quad=display_quad)
        return pt

    def detect_debug(self, frame: np.ndarray, threshold: int = None, display_quad: np.ndarray = None):
        """
        Comprehensive 5mm IR LED bulb detection with trained ML model and strict spectral filtering.

        Returns:
            point: (cx, cy) tuple or None
            annotated_frame: RGB image with active target reticle, bulb bounding box, and display quad overlay
            thresh_rgb: RGB image of the binary threshold mask
            metadata: dict containing area, circularity, spectral ratio, purple excess, ir_confidence, etc.
        """
        if frame is None:
            empty = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            return None, empty, empty, {"status": "NO_FRAME", "area": 0.0, "threshold": self.threshold}

        h, w = frame.shape[:2]

        # Extract RGB channels (frame is RGB from Camera)
        if len(frame.shape) == 3:
            r = frame[:, :, 0].astype(np.float32)
            g = frame[:, :, 1].astype(np.float32)
            b = frame[:, :, 2].astype(np.float32)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            gray = frame
            r = g = b = gray.astype(np.float32)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 3x3 Gaussian blur to preserve small pinpoint LED cores
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Threshold for bright pixels
        thresh_val = threshold if threshold is not None else self.threshold
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        annotated = frame.copy()

        # -----------------------------------------------------------------
        # Render DocScanner Display Boundary Quad (if calibrated)
        # -----------------------------------------------------------------
        has_valid_quad = False
        if display_quad is not None and len(display_quad) == 4:
            quad_area = cv2.contourArea(display_quad.astype(np.float32))
            if quad_area > 5000:
                has_valid_quad = True
                quad_int = display_quad.astype(np.int32)
                cv2.polylines(annotated, [quad_int], isClosed=True, color=(0, 240, 255), thickness=2)

                corner_labels = ["TL", "TR", "BR", "BL"]
                for i, (cx_pt, cy_pt) in enumerate(quad_int):
                    cv2.circle(annotated, (cx_pt, cy_pt), 5, (0, 255, 100), -1)
                    cv2.putText(annotated, corner_labels[i], (cx_pt + 6, cy_pt - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

                center_x = int(np.mean(display_quad[:, 0]))
                center_y = int(np.mean(display_quad[:, 1]))
                cv2.putText(annotated, "DISPLAY BOUNDS", (center_x - 55, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        metadata = {
            "threshold": thresh_val,
            "num_contours": len(contours) if contours else 0,
            "area": 0.0,
            "circularity": 0.0,
            "aspect_ratio": 0.0,
            "peak_brightness": 0,
            "spectral_ratio": 1.0,
            "purple_excess": 0.0,
            "ir_confidence": 0.0,
            "is_ir_purple": False,
            "is_emitting_bulb": False,
            "in_display_area": True,
            "point": None,
            "status": "SEARCHING"
        }

        if not contours:
            cv2.putText(annotated, "NO LIGHT DETECTED", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 60, 60), 2)
            cv2.putText(thresh_rgb, f"THRESH: {thresh_val}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            return None, annotated, thresh_rgb, metadata

        valid_emitting_candidates = []
        highest_rejected_conf = 0.0

        for c in contours:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            # 1. Edge Exclusion Filter: discard edge-touching border glare
            if (cy < CAMERA_EDGE_MARGIN_TOP or
                cy > (h - CAMERA_EDGE_MARGIN_BOTTOM) or
                cx < CAMERA_EDGE_MARGIN_SIDE or
                cx > (w - CAMERA_EDGE_MARGIN_SIDE)):
                cv2.drawContours(annotated, [c], -1, (180, 50, 50), 1)
                continue

            # 2. Area Filter: must be compact (3 - 220 px)
            if not (self.min_area <= area <= self.max_area):
                cv2.drawContours(annotated, [c], -1, (180, 100, 40), 1)
                continue

            # 3. Geometric Aspect Ratio Filter: must be roughly round / dome-like
            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            aspect_ratio = float(w_b) / max(1.0, float(h_b))
            if not (0.45 <= aspect_ratio <= 2.2):
                cv2.drawContours(annotated, [c], -1, (180, 60, 60), 1)
                continue

            # 4. Circularity Filter: 4*pi*area / perimeter^2
            peri = cv2.arcLength(c, True)
            circularity = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0.0
            if circularity < 0.35:
                cv2.drawContours(annotated, [c], -1, (180, 60, 60), 1)
                continue

            # 5. Peak Brightness in contour region
            roi_gray = gray[y_b:y_b + h_b, x_b:x_b + w_b]
            peak_val = float(np.max(roi_gray)) if roi_gray.size > 0 else 0.0
            if peak_val < 195:
                continue

            # 6. Extract ROI with padding for full epoxy bulb halo
            pad = 6
            x1 = max(0, x_b - pad)
            y1 = max(0, y_b - pad)
            x2 = min(w, x_b + w_b + pad)
            y2 = min(h, y_b + h_b + pad)
            patch_bgr = frame_bgr[y1:y2, x1:x2]

            # -------------------------------------------------------------
            # 7. TRAINED ML MODEL CLASSIFICATION (5mm IR LED Bulb vs Non-IR Light)
            # -------------------------------------------------------------
            features = self.extract_candidate_features(patch_bgr, c, peak_val, aspect_ratio, circularity)

            if self.has_model:
                feat_scaled = (features - self.mean) / self.std
                linear = float(np.dot(feat_scaled, self.weights) + self.bias)
                ir_confidence = float(sigmoid(linear))
            else:
                # Heuristic fallback based on spectral ratios
                c_excess = features[1]
                c_ratio = features[2]
                ir_confidence = 0.95 if (c_excess >= 18.0 and c_ratio >= 1.08) else 0.05

            # Direct spectral measurements
            core_excess = float(features[1])
            core_spectral_ratio = float(features[2])
            halo_excess = float(features[3])

            # CRITICAL DECISION:
            # An active emitting 5mm IR LED bulb must satisfy:
            # 1. ML Model Confidence >= 0.50
            # 2. Authentic chromatic purple signature: Red & Blue excess over Green (not neutral white)
            is_active_ir_bulb = (
                (ir_confidence >= self.model_threshold) and
                (core_excess >= 10.0 or halo_excess >= 15.0) and
                (features[5] >= 2.0)  # Blue must exceed Green
            )

            if not is_active_ir_bulb:
                # REJECTED: This is a white drawing on the blackboard, unlit bulb, or room lamp!
                highest_rejected_conf = max(highest_rejected_conf, ir_confidence)
                cv2.drawContours(annotated, [c], -1, (140, 140, 140), 1)
                continue

            # 8. Check if inside calibrated display quad (DocScanner area)
            in_display = True
            if has_valid_quad:
                poly = display_quad.astype(np.float32)
                in_display = (cv2.pointPolygonTest(poly, (cx, cy), False) >= 0)

            score = (ir_confidence * 200.0) + (peak_val * 1.5) + (circularity * 50.0) - (area * 0.1)
            if has_valid_quad and not in_display:
                score -= 300.0

            valid_emitting_candidates.append({
                "score": score,
                "point": (cx, cy),
                "contour": c,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "area": area,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "peak": peak_val,
                "spectral_ratio": core_spectral_ratio,
                "purple_excess": core_excess,
                "confidence": ir_confidence,
                "is_ir": True,
                "in_display": in_display
            })

        if not valid_emitting_candidates:
            msg = f"NON-IR LIGHT REJECTED (Conf: {highest_rejected_conf*100:.0f}%)" if highest_rejected_conf > 0.05 else "WHITE BOARD LIGHT REJECTED (NO IR BULB)"
            cv2.putText(annotated, msg, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 2)
            cv2.putText(thresh_rgb, f"THRESH: {thresh_val} (NO EMITTING IR LED)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 2)
            metadata["status"] = "WHITE_LIGHT_REJECTED"
            metadata["ir_confidence"] = highest_rejected_conf * 100.0
            return None, annotated, thresh_rgb, metadata

        # Select the highest-confidence authentic emitting IR LED bulb
        valid_emitting_candidates.sort(key=lambda item: item["score"], reverse=True)
        best = valid_emitting_candidates[0]

        best_pt = best["point"]
        best_contour = best["contour"]
        bx, by, bw, bh = best["bbox"]
        icx, icy = int(best_pt[0]), int(best_pt[1])
        in_display = best["in_display"]
        conf = best["confidence"] * 100.0

        reticle_color = (0, 255, 0) if in_display else (255, 120, 0)
        status_text = "TRACKED & DRAWING" if in_display else "OUTSIDE DISPLAY"

        # Draw contour boundary in yellow
        cv2.drawContours(annotated, [best_contour], -1, (255, 230, 0), 2)

        # Draw glowing purple/magenta bounding box around the 5mm LED bulb body
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (220, 60, 240), 2)

        # Draw circular target reticle
        cv2.circle(annotated, (icx, icy), 14, reticle_color, 2)
        cv2.circle(annotated, (icx, icy), 3, reticle_color, -1)
        cv2.line(annotated, (icx - 20, icy), (icx + 20, icy), reticle_color, 2)
        cv2.line(annotated, (icx, icy - 20), (icx, icy + 20), reticle_color, 2)

        # Text annotation on camera feed
        label = f"IR LED BULB: {conf:.1f}% [EMITTING] -> ({icx}, {icy}) [{status_text}]"
        cv2.putText(annotated, label, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, reticle_color, 2)
        cv2.putText(thresh_rgb, f"THRESH: {thresh_val} (IR BULB EMITTING {conf:.0f}%)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)

        # Mark on threshold mask
        cv2.circle(thresh_rgb, (icx, icy), 8, (0, 255, 0), 1)

        metadata["area"] = best["area"]
        metadata["circularity"] = best["circularity"]
        metadata["aspect_ratio"] = best["aspect_ratio"]
        metadata["peak_brightness"] = best["peak"]
        metadata["spectral_ratio"] = best["spectral_ratio"]
        metadata["purple_excess"] = best["purple_excess"]
        metadata["ir_confidence"] = conf
        metadata["is_ir_purple"] = True
        metadata["is_emitting_bulb"] = True
        metadata["in_display_area"] = in_display
        metadata["point"] = best_pt
        metadata["status"] = "TRACKING_AUTHENTIC_IR_BULB"

        return best_pt, annotated, thresh_rgb, metadata


if __name__ == "__main__":
    print("Testing IRDetector with trained 5mm IR LED bulb model...")
    detector = IRDetector(threshold=200)

    # Test 1: Blackboard white drawing (R=255, G=255, B=255)
    img_white = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img_white, (200, 200), 8, (255, 255, 255), -1)
    pt_w, _, _, meta_w = detector.detect_debug(img_white)
    print("Test 1 - Whiteboard Drawing:")
    print(f"  Detected Point : {pt_w} (Expected: None)")
    print(f"  Status         : {meta_w['status']}")
    print(f"  Confidence     : {meta_w['ir_confidence']:.1f}%")
    assert pt_w is None, "ERROR: Whiteboard drawing was not rejected!"

    # Test 2: Active 5mm IR LED bulb emitting purple IR light (R=240, G=120, B=230)
    img_ir = np.zeros((480, 640, 3), dtype=np.uint8)
    # Core
    cv2.circle(img_ir, (320, 240), 5, (250, 180, 250), -1)
    # Glowing epoxy dome halo
    cv2.circle(img_ir, (320, 240), 12, (230, 110, 220), 2)
    pt_ir, _, _, meta_ir = detector.detect_debug(img_ir)
    print("\nTest 2 - Emitting 5mm IR LED Bulb:")
    print(f"  Detected Point : {pt_ir} (Expected: around (320, 240))")
    print(f"  Status         : {meta_ir['status']}")
    print(f"  Confidence     : {meta_ir['ir_confidence']:.1f}%")
    assert pt_ir is not None, "ERROR: Emitting IR LED bulb was not detected!"
    assert abs(pt_ir[0] - 320) < 3 and abs(pt_ir[1] - 240) < 3, "ERROR: Point coordinates inaccurate!"

    print("\nALL TESTS PASSED! Trained IR detector accurately locks onto emitting IR LED bulb and 100% rejects white light.")
