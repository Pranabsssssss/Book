import cv2
import numpy as np
from config import (
    IR_THRESHOLD, IR_MIN_AREA, IR_MAX_AREA,
    CAMERA_EDGE_MARGIN_TOP, CAMERA_EDGE_MARGIN_BOTTOM, CAMERA_EDGE_MARGIN_SIDE,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    IR_SPECTRAL_RATIO_MIN, IR_PURPLE_EXCESS_MIN
)


class IRDetector:
    """
    Spectral IR Dot Detector.
    Differentiates authentic Light-Purple / Violet IR light (from remote controls and IR pens)
    from ordinary white ambient light (ceiling lamps, flashlights, white cloth/paper, screen glare).
    Strictly accepts compact circular dots and circles.
    """
    def __init__(self, threshold=IR_THRESHOLD, min_area=IR_MIN_AREA, max_area=IR_MAX_AREA):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame: np.ndarray, threshold: int = None, display_quad: np.ndarray = None):
        """
        Locate the centroid of the circular IR dot in the camera frame.
        Returns:
            (x, y) tuple in camera pixel coordinates, or None.
        """
        pt, _, _, _ = self.detect_debug(frame, threshold=threshold, display_quad=display_quad)
        return pt

    def detect_debug(self, frame: np.ndarray, threshold: int = None, display_quad: np.ndarray = None):
        """
        Comprehensive circular dot detection with IR spectral differentiation and DocScanner quad.

        Returns:
            point: (cx, cy) tuple or None
            annotated_frame: RGB image with active target reticle and display quad overlay
            thresh_rgb: RGB image of the binary threshold mask
            metadata: dict containing area, circularity, aspect ratio, peak brightness, etc.
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
            # Standard luminance
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            r = g = b = gray.astype(np.float32)

        # Use 3x3 blur to preserve small pinpoint circular LED cores
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Threshold for intensely bright pixels
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
            "is_ir_purple": False,
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

        circular_candidates = []

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

            # 2. Area Filter: must be compact (3 - 180 px)
            if not (self.min_area <= area <= self.max_area):
                cv2.drawContours(annotated, [c], -1, (180, 100, 40), 1)
                continue

            # 3. Geometric Aspect Ratio Filter: must be roughly square / circular
            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            aspect_ratio = float(w_b) / max(1.0, float(h_b))
            if not (0.50 <= aspect_ratio <= 2.0):
                cv2.drawContours(annotated, [c], -1, (180, 60, 60), 1)
                continue

            # 4. Circularity Filter: 4*pi*area / perimeter^2
            peri = cv2.arcLength(c, True)
            circularity = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0.0
            if circularity < 0.40:
                cv2.drawContours(annotated, [c], -1, (180, 60, 60), 1)
                continue

            # 5. Peak Brightness in contour region
            roi_gray = gray[y_b:y_b + h_b, x_b:x_b + w_b]
            peak_val = int(np.max(roi_gray)) if roi_gray.size > 0 else 0

            # -------------------------------------------------------------
            # 6. IR Chromaticity & Spectral Differentiation
            # Separates Light-Purple / Violet IR Light from Normal White Light
            # -------------------------------------------------------------
            roi_r = r[y_b:y_b + h_b, x_b:x_b + w_b]
            roi_g = g[y_b:y_b + h_b, x_b:x_b + w_b]
            roi_b = b[y_b:y_b + h_b, x_b:x_b + w_b]

            g_safe = np.maximum(roi_g, 1.0)
            spectral_ratios = (roi_r + roi_b) / (2.0 * g_safe)
            max_spectral_ratio = float(np.max(spectral_ratios)) if spectral_ratios.size > 0 else 1.0

            purple_excesses = roi_r + roi_b - 2.0 * roi_g
            max_purple_excess = float(np.max(purple_excesses)) if purple_excesses.size > 0 else 0.0

            # Authentic IR Light has significant purple excess:
            # Saturated core (peak >= 230) + purple tint (ratio >= 1.14 or excess >= 10.0)
            # or distinctive purple glow (excess >= 25.0)
            is_ir = (peak_val >= 230 and (max_spectral_ratio >= IR_SPECTRAL_RATIO_MIN or max_purple_excess >= IR_PURPLE_EXCESS_MIN)) or \
                    (max_purple_excess >= 25.0 and max_spectral_ratio >= 1.20)

            # 7. Check if inside calibrated display quad (DocScanner area)
            in_display = True
            if has_valid_quad:
                poly = display_quad.astype(np.float32)
                in_display = (cv2.pointPolygonTest(poly, (cx, cy), False) >= 0)

            # Composite Score:
            # - Heavily rewards authentic IR purple spectral signature
            # - Heavily penalizes ordinary white light (lamp, paper, white reflection)
            score = (peak_val * 2.0) + (circularity * 80.0) - (area * 0.15)
            if is_ir:
                score += 500.0 + (max_spectral_ratio * 100.0)
            else:
                score -= 300.0  # Penalize ordinary white light

            if has_valid_quad and not in_display:
                score -= 200.0  # Outside display quad

            circular_candidates.append({
                "score": score,
                "point": (cx, cy),
                "contour": c,
                "area": area,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "peak": peak_val,
                "spectral_ratio": max_spectral_ratio,
                "is_ir": is_ir,
                "in_display": in_display
            })

        if not circular_candidates:
            cv2.putText(annotated, "WHITE LIGHT / GLARE REJECTED", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 140, 40), 2)
            cv2.putText(thresh_rgb, f"THRESH: {thresh_val} (NO IR DOT)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
            metadata["status"] = "REJECTED_WHITE_LIGHT"
            return None, annotated, thresh_rgb, metadata

        # Pick the highest-scoring candidate (authentic IR LED)
        circular_candidates.sort(key=lambda item: item["score"], reverse=True)
        best = circular_candidates[0]

        best_pt = best["point"]
        best_contour = best["contour"]
        icx, icy = int(best_pt[0]), int(best_pt[1])
        in_display = best["in_display"]
        is_ir = best["is_ir"]

        # Color-coded feedback on camera view:
        # Green = Authentic IR inside display
        # Blue = IR light outside display
        # Orange = Marginal/White light
        if is_ir and in_display:
            reticle_color = (0, 255, 0)
            status_text = "IR LED: TRACKING"
        elif is_ir and not in_display:
            reticle_color = (255, 120, 0)
            status_text = "IR LED: OUTSIDE DISPLAY"
        else:
            reticle_color = (0, 165, 255)
            status_text = "WHITE LIGHT FILTERED"

        # Draw contour boundary
        cv2.drawContours(annotated, [best_contour], -1, (255, 230, 0), 2)

        # Draw circular target reticle
        cv2.circle(annotated, (icx, icy), 14, reticle_color, 2)
        cv2.circle(annotated, (icx, icy), 3, reticle_color, -1)
        cv2.line(annotated, (icx - 20, icy), (icx + 20, icy), reticle_color, 2)
        cv2.line(annotated, (icx, icy - 20), (icx, icy + 20), reticle_color, 2)

        # Text annotation on camera feed
        label = f"DOT: ({icx}, {icy}) IR-Ratio:{best['spectral_ratio']:.2f} [{status_text}]"
        cv2.putText(annotated, label, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, reticle_color, 2)
        cv2.putText(thresh_rgb, f"THRESH: {thresh_val} (IR PURPLE DOT)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Mark on threshold mask
        cv2.circle(thresh_rgb, (icx, icy), 8, (0, 255, 0), 1)

        metadata["area"] = best["area"]
        metadata["circularity"] = best["circularity"]
        metadata["aspect_ratio"] = best["aspect_ratio"]
        metadata["peak_brightness"] = best["peak"]
        metadata["spectral_ratio"] = best["spectral_ratio"]
        metadata["is_ir_purple"] = is_ir
        metadata["in_display_area"] = in_display
        metadata["point"] = best_pt
        metadata["status"] = "TRACKING_IR" if is_ir else "WHITE_LIGHT_MARGINAL"

        return best_pt, annotated, thresh_rgb, metadata


if __name__ == "__main__":
    print("Testing IRDetector with authentic IR purple discriminator...")
    # 1. Create image with a purple IR LED (B=255, G=220, R=255) at (300, 200)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (300, 200), 5, (255, 220, 255), -1)

    # 2. Add an ordinary white light bulb at (150, 150) (R=255, G=255, B=255)
    cv2.circle(img, (150, 150), 6, (255, 255, 255), -1)

    det = IRDetector(threshold=230)
    pt, debug_img, mask_img, meta = det.detect_debug(img)

    print(f"Detected point: {pt}")
    print(f"Is IR Purple: {meta['is_ir_purple']}, Spectral Ratio: {meta['spectral_ratio']:.2f}")
    assert pt is not None and abs(pt[0] - 300) < 2 and abs(pt[1] - 200) < 2, f"Failed to pick IR dot! Got {pt}"
    print("Spectral IR Purple vs Normal White Light test passed successfully!")
